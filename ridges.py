"""Generate terrain ridges/valleys"""
import sys
import time
import numpy
import gdal_utils

# Neighbor directions:
#   0 1 2
#   3<4>5
#   6 7 8
VALID_NEIGHBOR_DIRS = (0, 1, 2, 3, 5, 6, 7, 8)
NEIGHBOR_SELF = 4
NEIGHBOR_LAST_VALID = 8
NEIGHBOR_PENDING = 9
NEIGHBOR_SEED = 10
NEIGHBOR_STOP = 11
NEIGHBOR_INVALID = 12
NEIGHBOR_DIR_DTYPE = numpy.int8
VALID_NEIGHBOR_DIRS = numpy.array(VALID_NEIGHBOR_DIRS, dtype=NEIGHBOR_DIR_DTYPE)

# Keep the seed away from the edges
SEED_INFLATE = 1
# Select distance caclulation method:
#   0 - Real geodetic distance (geod_distance): Use pyproj.Geod.inv()
#   1 - Transverse Mercator (tm_distance): Use TM origin at the center of raster data
#   2 - Draft (draft_distance): Use pre-calculated pixel size by tm_distance for all pixels
DISTANCE_METHOD = 0

def VECTOR_LAYER_NAME(valleys): return 'valleys' if valleys else 'ridges'
def VECTOR_FEATURE_STYLE(valleys): return 'PEN(c:#0000FF,w:2px)' if valleys else 'PEN(c:#FF0000,w:2px)'

#
# Generic tools
#
def search_sorted(array, cmp_fn, *args):
    """Generic binary search"""
    beg = 0
    end = len(array)
    res = None
    while beg < end:
        mid = (beg + end) // 2
        res = cmp_fn(array[mid], *args)
        if res == 0:
            return mid, res
        if res < 0:
            end = mid
        else:
            beg = mid + 1
    return beg, res

#
# Main processing
#
def neighbor_xy(x_y, neighbor_dir):
    """Get the coordinates of a neighbor pixel"""
    if neighbor_dir.ndim < 2:       # Performance optimization
        return (x_y.T + (neighbor_dir % 3 - 1, neighbor_dir // 3 - 1)).T
    return x_y + numpy.stack((neighbor_dir % 3 - 1, neighbor_dir // 3 - 1), -1)

def neighbor_flip(neighbor_dir):
    """Get the inverted neighbor direction"""
    return NEIGHBOR_LAST_VALID - neighbor_dir

def neighbor_xy_safe(x_y, neighbor_dir):
    """Get the coordinates of a neighbor pixel, handle invalid directions"""
    res_xy = neighbor_xy(x_y, neighbor_dir)
    # Threat NEIGHBOR_PENDING and NEIGHBOR_INVALID as NEIGHBOR_SELF,
    # to ensure the raster coordinates are valid
    mask = neighbor_dir > NEIGHBOR_LAST_VALID
    if mask.any():
        res_xy[mask] = x_y[mask]
    return res_xy

def neighbor_is_invalid(neighbor_dir):
    """Return mask of where the neighbor directions are invalid"""
    return neighbor_dir > NEIGHBOR_LAST_VALID

def process_neighbors(dem_band, distance, dir_arr, x_y, stop_mask):
    """Process the valid and pending neighbor points and return a list to be put to tentative"""
    x_y = x_y[...,numpy.newaxis,:]
    n_xy = neighbor_xy(x_y, VALID_NEIGHBOR_DIRS)
    n_dir = numpy.broadcast_to(VALID_NEIGHBOR_DIRS, n_xy.shape[:-1])
    # Filter out of bounds pixels
    mask = dem_band.in_bounds(n_xy)
    if not mask.all():
        n_xy = n_xy[mask]
        n_dir = n_dir[mask]
    # Filter already processed pixels
    mask = gdal_utils.read_arr(dir_arr['n_dir'], n_xy) == NEIGHBOR_PENDING
    if not mask.any():
        return ()
    if not mask.all():
        n_xy = n_xy[mask]
        n_dir = n_dir[mask]
    # Process selected pixels
    n_dir = neighbor_flip(n_dir)
    n_dist = distance.get_distance(numpy.broadcast_to(x_y, n_xy.shape), n_xy)
    n_dist += gdal_utils.read_arr(dir_arr['dist'], x_y)
    # Put 'stop' markers on the successors of the masked points
    # This is to split lines at the boundary pixels
    if stop_mask.any():
        n_dir[stop_mask] = NEIGHBOR_STOP
        n_dist[stop_mask] = 0.
    gdal_utils.write_arr(dir_arr['n_dir'], n_xy, n_dir)
    gdal_utils.write_arr(dir_arr['dist'], n_xy, n_dist)
    return n_xy

def reduced_distance(dir_arr, x_y):
    """Calculate trace distance upto the next NEIGHBOR_STOP"""
    start_dist = gdal_utils.read_arr(dir_arr['dist'], x_y)
    while True:
        n_dir, dist = gdal_utils.read_arr(dir_arr, x_y)
        if neighbor_is_invalid(n_dir):
            return start_dist - dist
        x_y = neighbor_xy(x_y, n_dir)

def result_lines_insert(result_lines, entry, start=0):
    """Insert entry in result_lines by keeping it sorted by distance (entry[1])"""
    idx, _ = search_sorted(result_lines[start:], lambda v: 1 if entry[1] < v[1] else -1)
    result_lines.insert(idx + start, entry)

def select_seed(elevations, valleys, mask=None):
    """Select a point to start ridge/valley tracing"""
    if mask is None:
        mask = numpy.isnan(elevations)

    # Keep the original mask if shrinking turns to disaster
    orig_mask = mask.copy()

    # Shrink/mask boundaries to select the seed away from edges
    if elevations.shape[0] > 2 * SEED_INFLATE:
        mask[:SEED_INFLATE,:] = True
        mask[-SEED_INFLATE:,:] = True
    if elevations.shape[1] > 2 * SEED_INFLATE:
        mask[:,:SEED_INFLATE] = True
        mask[:,-SEED_INFLATE:] = True

    # Revert the mask if we have masked everything
    if mask.all():
        mask = orig_mask

    # Use MaskedArray array to find min/max
    elevations = numpy.ma.array(elevations, mask=mask)
    flat_idx = elevations.argmin() if valleys else elevations.argmax()
    seed_xy = numpy.unravel_index(flat_idx, elevations.shape)
    return numpy.array(seed_xy, dtype=numpy.int32)

def trace_ridges(dem_band, valleys=False):
    """Generate terrain ridges or valleys"""
    # Start at the max/min altitude (first one, away from edges)
    elevations = dem_band.get_elevation(True)
    seed_xy = select_seed(elevations, valleys)
    print('Tracing', 'valleys' if valleys else 'ridges',
          'from seed point', seed_xy,
          ', altitude', dem_band.get_elevation(seed_xy))

    #
    # Neighbor directions and distance array
    # Initialize the points to be processed with 'pending' value.
    #
    dir_arr = numpy.empty(elevations.shape, dtype=[
            ('n_dir', NEIGHBOR_DIR_DTYPE),
            ('dist', numpy.float),
        ])
    dir_arr['n_dir'] = NEIGHBOR_PENDING
    dir_arr['dist'] = numpy.nan
    dir_arr['n_dir'][numpy.isnan(elevations)] = NEIGHBOR_INVALID
    gdal_utils.write_arr(dir_arr, seed_xy, (NEIGHBOR_SEED, 0.))
    del elevations

    #
    # Tentative point list (coord)
    # Initially contains the start point only
    #
    tentative = [seed_xy]

    #
    # End-points of generated lines (coord and distance)
    #
    result_lines = []

    distance = gdal_utils.geod_distance(dem_band) if 0 == DISTANCE_METHOD \
            else gdal_utils.tm_distance(dem_band) if 1 == DISTANCE_METHOD \
            else gdal_utils.draft_distance(dem_band)
    progress_next = 0
    while tentative:
        x_y = tentative.pop()
        #print('    Processing point %s alt %d, dist %d'%(x_y, dem_band.get_elevation(seed_xy), dist))
        # The lines can only pass-thru inner DEM pixels, the boundary ones do split
        stop_mask = numpy.logical_or((x_y < 1).any(-1), (dir_arr.shape - x_y <= 1).any(-1))
        successors = 0
        for t_xy in process_neighbors(dem_band, distance, dir_arr, x_y, stop_mask):
            successors += 1
            alt = dem_band.get_elevation(t_xy)
            assert not numpy.isnan(alt), '"NoDataValue" point %s is marked for processing'%t_xy
            # Insert the point in 'tentative' by keeping it sorted by altitude
            def cmp_fn(check_xy, *args):
                # The duplicated altitudes are placed at lowest possible index (<= or >=).
                # Thus, they are processed in order of appearance (FIFO).
                if valleys:
                    # Descending sort FIFO-duplicate (>=)
                    return -1 if alt >= dem_band.get_elevation(check_xy) else 1
                # Ascending sort FIFO-duplicate (<=)
                return -1 if alt <= dem_band.get_elevation(check_xy) else 1
            idx, _ = search_sorted(tentative, cmp_fn)
            tentative.insert(idx, t_xy)

        if successors == 0 or stop_mask.any():
            dist = gdal_utils.read_arr(dir_arr['dist'], x_y)
            #print('  Line finished at point %s total length %d'%(x_y, dist))
            # Keep this end-point in 'result_lines', but if it's at least one pixel
            if dist > 0:
                result_lines_insert(result_lines, (x_y, dist))
                #
                # Progress, each 1000-th line
                #
                if progress_next <= len(result_lines):
                    print('  Tentatives', len(tentative), 'completed', len(result_lines),
                            'max/mid/min len %d/%d/%d'%(result_lines[0][1], result_lines[len(result_lines)//2][1], result_lines[-1][1]))
                    progress_next += 1000

            # After the 'tentative' is exhausted, there still can be islands of valid elevations,
            # that were not processed, because of the surrounding invalid ones
            if not tentative:
                mask = dir_arr['n_dir'] == NEIGHBOR_PENDING
                if mask.any():
                    # Restart at the highest/lowest unprocessed point
                    seed_xy = select_seed(dem_band.get_elevation(True), valleys, numpy.logical_not(mask))
                    print('Restart tracing from seed point', seed_xy, ', altitude', dem_band.get_elevation(seed_xy))
                    gdal_utils.write_arr(dir_arr, seed_xy, (NEIGHBOR_SEED, 0.))
                    tentative.append(seed_xy)

    return result_lines, dir_arr

def combine_lines(result_lines, dir_arr, min_len=0):
    """Create polylines from previously generated ridges or valleys"""
    polylines = []
    # Process the lines longer than min_len
    prev_dist = numpy.inf    # Assert only
    while result_lines:
        x_y, dist = result_lines.pop(0)
        assert dist <= prev_dist, 'Unsorted result_lines %d->%d'%(prev_dist, dist); prev_dist = dist
        if dist < min_len:
            break
        print('Generating line starting at point %s total length %d'%(x_y, dist))
        pline = []
        # Trace route
        while x_y is not None:
            pline.append(x_y)
            n_dir = gdal_utils.read_arr(dir_arr['n_dir'], x_y)
            # Stop other lines from overlapping that one
            gdal_utils.write_arr(dir_arr['n_dir'], x_y, NEIGHBOR_STOP)
            x_y = None if neighbor_is_invalid(n_dir) else neighbor_xy(x_y, n_dir)
        polylines.append((dist, pline))

        # Update distances after some of the lines were cut
        print('  Update remaining %d lines: mid/min len %d/%d'%(len(result_lines), result_lines[len(result_lines)//2][1], result_lines[-1][1]))
        progress_next = 1
        idx = 0
        while idx < len(result_lines):
            x_y, old_dist = result_lines[idx]
            dist = reduced_distance(dir_arr, x_y)
            if dist == old_dist:
                #
                # Progress (logarithmic), total ~10 messages
                #
                if progress_next <= 1 + idx / len(result_lines):
                    print('    Updated line at %d/%d: len %d'%(idx, len(result_lines), result_lines[idx][1]))
                    progress_next *= 1.07   # 2^(1/10)
                idx += 1
            else:
                assert dist < old_dist, 'Line distance was increased %d->%d'%(old_dist, dist)
                # Move the entry to keep 'result_lines' sorted by distance
                del result_lines[idx]
                if dist >= min_len:
                    #print('    Updating line at %d/%d, distance %d->%d'%(idx, len(result_lines), old_dist, dist))
                    result_lines_insert(result_lines, (x_y, dist), idx)

    return polylines

def main(argv):
    """Main entry"""
    valleys = False
    truncate = True
    src_filename = dst_filename = None
    while argv:
        if argv[0][0] == '-':
            if argv[0] == '-h':
                return print_help()
            if argv[0] == '-valley':
                valleys = True
            else:
                return print_help('Unsupported option "%s"'%argv[0])
        else:
            if src_filename is None:
                src_filename = argv[0]
            elif dst_filename is None:
                dst_filename = argv[0]
            else:
                return print_help('Unexpected argument "%s"'%argv[0])

        argv = argv[1:]

    if src_filename is None or dst_filename is None:
        return print_help('Missing file-names')

    # Load DEM
    dem_band = gdal_utils.dem_open(src_filename)
    if dem_band is None:
        return print_help('Unable to open "%s"'%src_filename)

    dst_ds = gdal_utils.vect_create(dst_filename)
    if dst_ds is None:
        return print_help('Unable to create "%s"'%src_filename)

    dem_band.load()

    #
    # Trace ridges/valleys
    #
    start = time.time()

    result_lines, dir_arr = trace_ridges(dem_band, valleys)
    if result_lines is None:
        print('Error: Failed to trace ridges', file=sys.stderr)
        return 2

    duration = time.time() - start
    print('Traced total %d lines, max length %d, %d sec'%(len(result_lines), result_lines[0][1], duration))

    #
    # Combine traced lines, longer than quarter of the longest one
    #
    start = time.time()

    polylines = combine_lines(result_lines, dir_arr, result_lines[0][1] / 4)
    if not polylines:
        print('Error: Failed to combine lines', file=sys.stderr)
        return 2

    duration = time.time() - start
    print('Created total %d polylines, first %d, last %d points, %d sec'%(len(polylines), len(polylines[0]), len(polylines[-1]), duration))

    if dst_ds:
        start = time.time()
        # Delete existing layers
        if truncate:
            for i in reversed(range(dst_ds.get_layer_count())):
                print('  Deleting layer', gdal_utils.gdal_vect_layer(dst_ds, i).get_name())
                dst_ds.delete_layer(i)
        # Create new one
        dst_layer = gdal_utils.gdal_vect_layer.create(dst_ds, VECTOR_LAYER_NAME(valleys),
                srs=dem_band.get_spatial_ref(), geom_type=gdal_utils.wkbLineString)
        if dst_layer is None:
            print('Error: Unable to create layer', file=sys.stderr)
            return 1

        # Add fields
        dst_layer.create_field('Name', True)    # KML <name>

        for dist, pline in polylines:
            geom = dst_layer.create_feature_geometry(gdal_utils.wkbLineString)
            geom.set_field('Name', '%dm'%dist if dist < 10000 else '%dkm'%round(dist/1000))
            geom.set_style_string(VECTOR_FEATURE_STYLE(valleys))
            for x_y in reversed(pline):
                geom.add_point(*dem_band.xy2lonlatalt(x_y))
            geom.create()
        duration = time.time() - start
        print('Created total %d geometries, %d sec'%(len(polylines), duration))

    return 0

def print_help(err_msg=None):
    if err_msg:
        print('Error:', err_msg, file=sys.stderr)
    print('Usage:', sys.argv[0], '[<options>] <src_filename> <dst_filename>')
    print('\tOptions:')
    print('\t-h\t- This screen')
    print('\t-valley\t- Generate valleys, instead of ridges')
    return 0 if err_msg is None else 255

if __name__ == '__main__':
    ret = main(sys.argv[1:])
    if ret:
        exit(ret)
