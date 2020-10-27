"""Generate terrain ridges/valleys"""
import sys
import time
import numpy
import gdal_utils

# Neighbor directions:
#   0 1 2
#   3<4>5
#   6 7 8
VALID_NEIGHBOR_DIRS = numpy.array((0, 1, 2, 3, 5, 6, 7, 8))
NEIGHBOR_SELF = 4
NEIGHBOR_LAST_VALID = 8
NEIGHBOR_PENDING = 9
NEIGHBOR_SEED = 10
NEIGHBOR_STOP = 11
NEIGHBOR_INVALID = 12
NEIGHBOR_DIR_DTYPE = numpy.int8

# Keep the seed away from the edges
SEED_INFLATE = 1

def VECTOR_LAYER_NAME(valleys): return 'valleys' if valleys else 'ridges'

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

def valid_neighbors(dem_band, x_y):
    """Return list of valid neighbor points"""
    n_xy = neighbor_xy(x_y[...,numpy.newaxis,:], VALID_NEIGHBOR_DIRS)
    n_dir = numpy.broadcast_to(VALID_NEIGHBOR_DIRS, n_xy.shape[:-1])
    mask = dem_band.in_bounds(n_xy)
    if not mask.all():
        n_xy = n_xy[mask]
        n_dir = n_dir[mask]
    return zip(n_xy, n_dir)

def measure_distance(dir_dist_arr, x_y):
    """Calculate total trace distance"""
    dist = 0.
    while True:
        n_dir, n_dist = gdal_utils.read_arr(dir_dist_arr, x_y)
        if n_dir > NEIGHBOR_LAST_VALID:
            return dist
        dist += n_dist
        x_y = neighbor_xy(x_y, n_dir)

def result_lines_insert(result_lines, entry):
    """Insert entry in result_lines by keeping it sorted by distance (entry[1])"""
    idx, _ = search_sorted(result_lines, lambda v: 1 if entry[1] < v[1] else -1)
    result_lines.insert(idx, entry)

def trace_ridges(dem_band, valleys=False):
    """Generate terrain ridges or valleys"""
    # Start at the max/min altitude (first one, away from edges)
    elevations = dem_band.get_elevation(True)
    deflated_buf = elevations[SEED_INFLATE:-SEED_INFLATE,SEED_INFLATE:-SEED_INFLATE]
    flat_idx = numpy.nanargmin(deflated_buf) if valleys else numpy.nanargmax(deflated_buf)
    seed_xy = numpy.unravel_index(flat_idx, deflated_buf.shape)
    seed_xy = numpy.array(seed_xy) + [SEED_INFLATE, SEED_INFLATE]
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

    distance = gdal_utils.geod_distance(dem_band)
    old_tentative_len = 1
    while tentative:
        x_y = tentative.pop()
        #print('    Processing point %s alt %s'%(x_y, dem_band.get_elevation(seed_xy)))
        end_of_line = True
        for t_xy, n_dir in valid_neighbors(dem_band, x_y):
            if gdal_utils.read_arr(dir_arr['n_dir'], t_xy) == NEIGHBOR_PENDING:
                end_of_line = False
                # Keep the flipped neighbor direction to later track this back
                n_dist = distance.get_distance(x_y, t_xy)
                gdal_utils.write_arr(dir_arr, t_xy, (neighbor_flip(n_dir), n_dist))
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
        #
        # Progress
        #
        if abs(old_tentative_len - len(tentative)) / old_tentative_len > .5:
            print('  Tentatives', len(tentative), 'completed', len(result_lines))
            old_tentative_len = len(tentative)

        if end_of_line:
            dist = measure_distance(dir_arr, x_y)
            #print('  Line finished at point %s total length %d'%(x_y, dist))
            # Keep this end-point in 'result_lines'
            result_lines_insert(result_lines, (x_y, dist))

    return result_lines, dir_arr

def combine_lines(result_lines, dir_arr, min_len=0):
    """Create polylines from previously generated ridges or valleys"""
    polylines = []
    # Process the lines lenger than min_len
    while result_lines[0][1] > min_len:
        x_y, dist = result_lines.pop(0)
        print('Generating line starting at point %s total length %d'%(x_y, dist))
        pline = []
        # Trace route
        while x_y is not None:
            pline.append(x_y)
            n_dir = gdal_utils.read_arr(dir_arr['n_dir'], x_y)
            # Stop other lines from overlapping that one
            gdal_utils.write_arr(dir_arr['n_dir'], x_y, NEIGHBOR_STOP)
            x_y = None if n_dir > NEIGHBOR_LAST_VALID else neighbor_xy(x_y, n_dir)
        polylines.append(pline)

        # Update distances after some of the lines were cut
        print('  Updating line distances')
        idx = 0
        while idx < len(result_lines):
            x_y, old_dist = result_lines[idx]
            dist = measure_distance(dir_arr, x_y)
            if dist == old_dist:
                idx += 1
            else:
                # Move the entry to keep 'result_lines' sorted by distance
                del result_lines[idx]
                result_lines_insert(result_lines, (x_y, dist))

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

    # Trace ridges/valleys
    dem_band.load()
    start = time.time()
    result_lines, dir_arr = trace_ridges(dem_band, valleys)
    if result_lines is None:
        return 1
    duration = time.time() - start
    print('Traced total %d lines, max length %d, %d sec'%(len(result_lines), result_lines[0][1], duration))

    # Combine traced lines, longer than quarter of the longest one
    start = time.time()
    polylines = combine_lines(result_lines, dir_arr, result_lines[0][1] / 4)
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
        dst_layer = gdal_utils.gdal_vect_layer.create(dst_ds, VECTOR_LAYER_NAME(valleys), dem_band.get_spatial_ref())
        if dst_layer is None:
            print('Error: Unable to create layer', file=sys.stderr)
            return 1

        for pline in polylines:
            geom = dst_layer.create_feature_geometry(gdal_utils.wkbLineString)
            for x_y in pline:
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
