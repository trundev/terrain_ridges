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
NEIGHBOR_BOUNDARY = 13
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

KEEP_SNAPSHOT = True
RESUME_FROM_SNAPSHOT = 0    # Currently 0 to 2

#
# Internal data-types, mostly for keep/resume support
#
DIR_DIST_DTYPE = [
        ('n_dir', NEIGHBOR_DIR_DTYPE),
        ('dist', numpy.float),
]
TENTATIVE_DTYPE = [
        ('x_y', (numpy.int32, (2,))),
        ('alt', numpy.float),
]
RESULT_LINE_DTYPE = [
        ('x_y', (numpy.int32, (2,))),
        ('dist', numpy.float),
]

#
# Generic tools
#
def sorted_arr_insert(arr, entry, key, end=None):
    """Insert an entry in sorted array by keeping it sorted by key"""
    # When multiple entries are going to be inserted, ensure they are sorted
    # This is only mandatory, when multiple elements have the same insertion point
    if entry.ndim:
        argsort = numpy.argsort(entry[key])
        entry = numpy.take(entry, argsort)
    # Actual insert, do not search beyond 'end'
    idx = numpy.searchsorted(arr[:end][key], entry[key])
    return numpy.insert(arr, idx, entry)

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

#
# First stage - trace ridges
#
def select_seed(elevations, valleys, mask):
    """Select a point to start ridge/valley tracing"""
    # Keep the original mask if shrinking turns to disaster
    ma_mask = mask.copy()

    # Shrink/mask boundaries to select the seed away from edges
    if elevations.shape[0] > 2 * SEED_INFLATE:
        ma_mask[:SEED_INFLATE,:] = True
        ma_mask[-SEED_INFLATE:,:] = True
    if elevations.shape[1] > 2 * SEED_INFLATE:
        ma_mask[:,:SEED_INFLATE] = True
        ma_mask[:,-SEED_INFLATE:] = True

    # Revert the mask if we have masked everything
    if ma_mask.all():
        ma_mask = mask

    # Use MaskedArray array to find min/max
    elevations = numpy.ma.array(elevations, mask=ma_mask)
    flat_idx = elevations.argmin() if valleys else elevations.argmax()
    seed_xy = numpy.unravel_index(flat_idx, elevations.shape)
    return numpy.array(seed_xy, dtype=numpy.int32)

def process_neighbors(dem_band, distance, dir_arr, x_y):
    """Process the valid and pending neighbor points and return a list to be put to tentative"""
    x_y = x_y[...,numpy.newaxis,:]
    n_xy = neighbor_xy(x_y, VALID_NEIGHBOR_DIRS)
    n_dir = numpy.broadcast_to(VALID_NEIGHBOR_DIRS, n_xy.shape[:-1])
    # Filter out of bounds pixels
    mask = dem_band.in_bounds(n_xy)
    if not mask.all():
        n_xy = n_xy[mask]
        n_dir = n_dir[mask]
    # The lines can only pass-thru inner DEM pixels, the boundary ones do split
    stop_mask = ~mask.all(-1)
    # Filter already processed pixels
    neighs = gdal_utils.read_arr(dir_arr['n_dir'], n_xy)
    mask = neighs == NEIGHBOR_PENDING
    if not mask.any():
        return None, None
    if not mask.all():
        n_xy = n_xy[mask]
        n_dir = n_dir[mask]
        mask = neighs == NEIGHBOR_BOUNDARY
        if mask.any():
            stop_mask |= mask.any(-1)
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
    return n_xy, stop_mask

def trace_ridges(dem_band, valleys=False, boundary_val=None):
    """Generate terrain ridges or valleys"""
    # Start at the max/min altitude (first one, away from edges)
    elevations = dem_band.get_elevation(True)
    select_mask = numpy.isnan(elevations)
    if boundary_val is not None:
        select_mask |= elevations == boundary_val
    seed_xy = select_seed(elevations, valleys, select_mask)
    print('Tracing', 'valleys' if valleys else 'ridges',
          'from seed point', seed_xy,
          ', altitude', dem_band.get_elevation(seed_xy))

    #
    # Neighbor directions and distance array
    # Initialize the points to be processed with 'pending' value.
    #
    dir_arr = numpy.empty(elevations.shape, dtype=DIR_DIST_DTYPE)
    dir_arr['n_dir'] = NEIGHBOR_PENDING
    dir_arr['dist'] = numpy.nan
    # Here "select_mask" includes both boundary and "NoDataValue" points
    dir_arr['n_dir'][select_mask] = NEIGHBOR_BOUNDARY
    dir_arr['n_dir'][numpy.isnan(elevations)] = NEIGHBOR_INVALID
    gdal_utils.write_arr(dir_arr, seed_xy, (NEIGHBOR_SEED, 0.))
    del elevations

    #
    # Tentative point list (coord and altitude)
    # Initially contains the start point only
    #
    tentative = numpy.array([(seed_xy, dem_band.get_elevation(seed_xy))], dtype=TENTATIVE_DTYPE)

    #
    # End-points of generated lines (coord and distance)
    #
    result_lines = numpy.empty(0, dtype=RESULT_LINE_DTYPE)

    distance = gdal_utils.geod_distance(dem_band) if 0 == DISTANCE_METHOD \
            else gdal_utils.tm_distance(dem_band) if 1 == DISTANCE_METHOD \
            else gdal_utils.draft_distance(dem_band)
    progress_next = 0
    while tentative.size:
        x_y, _ = tentative[-1]
        tentative = tentative[:-1]
        #print('    Processing point %s alt %d, dist %d'%(x_y, _, gdal_utils.read_arr(dir_arr['dist'], x_y)))
        n_xy, stop_mask = process_neighbors(dem_band, distance, dir_arr, x_y)
        if n_xy is not None:
            alts = dem_band.get_elevation(n_xy)
            assert not numpy.isnan(alts).any(), '"NoDataValue" point(s) %s are marked for processing'%n_xy[numpy.isnan(alts)]
            # The valleys are handled by turning the elevations upside down
            if valleys:
                alts = -alts
            # Insert the points in 'tentative' by keeping it sorted by altitude.
            # The duplicated altitudes must be processed in order of appearance (FIFO),
            # i.e. numpy.searchsorted() with "side='left'".
            tentr = numpy.empty(alts.shape, dtype=tentative.dtype)
            tentr['x_y'] = n_xy
            tentr['alt'] = alts
            # The 'tentr' is flipped, only to keep the previous behavior, i.e. the FIFO rule is in
            # effect for the order of 'n_xy'. This is the same as if VALID_NEIGHBOR_DIRS is flipped.
            tentative = sorted_arr_insert(tentative, tentr[::-1], 'alt')

        if n_xy is None or stop_mask.any():
            dist = gdal_utils.read_arr(dir_arr['dist'], x_y)
            #print('  Line finished at point %s total length %d'%(x_y, dist))
            # Keep this end-point in 'result_lines', but if it's at least one pixel
            if dist > 0:
                result_lines = sorted_arr_insert(result_lines,
                        numpy.array((x_y, dist), dtype=result_lines.dtype), 'dist')
                #
                # Progress, each 1000-th line
                #
                if progress_next <= result_lines.shape[0]:
                    print('  Tentatives', tentative.shape[0], 'completed', result_lines.shape[0],
                            'max/mid/min len %d/%d/%d'%(result_lines[-1]['dist'], result_lines[result_lines.shape[0] // 2]['dist'], result_lines[0]['dist']))
                    progress_next += 1000

            # After the 'tentative' is exhausted, there still can be islands of valid elevations,
            # that were not processed, because of the surrounding invalid ones
            if not tentative.size:
                mask = dir_arr['n_dir'] == NEIGHBOR_PENDING
                if mask.any():
                    # Restart at the highest/lowest unprocessed point
                    seed_xy = select_seed(dem_band.get_elevation(True), valleys, numpy.logical_not(mask))
                    alt = dem_band.get_elevation(seed_xy)
                    print('Restart tracing from seed point', seed_xy, ', altitude', alt)
                    gdal_utils.write_arr(dir_arr, seed_xy, (NEIGHBOR_SEED, 0.))
                    tentative = numpy.array([(seed_xy, alt)], dtype=tentative.dtype)

    return result_lines, dir_arr

#
# Second stage - flip longest line(s)
#
def flip_line(new_dir_arr, dir_arr, x_y):
    """Flip all 'n_dir'-s along a line, invert the distances"""
    prev_dir, start_dist = gdal_utils.read_arr(dir_arr, x_y)
    print('Flipping line at', x_y, ', distance %d'%start_dist)

    gdal_utils.write_arr(new_dir_arr, x_y, (NEIGHBOR_SEED, 0.))
    while True:
        n_xy = neighbor_xy(x_y, prev_dir)
        n_dir, dist = gdal_utils.read_arr(dir_arr, n_xy)
        gdal_utils.write_arr(new_dir_arr, n_xy, (neighbor_flip(prev_dir), start_dist - dist))
        if neighbor_is_invalid(n_dir):
            assert n_dir == NEIGHBOR_SEED
            print('  Remove former seed at', n_xy)
            return n_xy

        x_y = n_xy
        prev_dir = n_dir

def flip_seed_lines(dir_arr, result_lines):
    """Create a new 'dir_arr', where all the first lines ending to any of "seed"-s are flipped"""
    new_dir_arr = numpy.empty_like(dir_arr)
    new_dir_arr['n_dir'] = NEIGHBOR_PENDING
    # Copy "invalid" markers
    mask = dir_arr['n_dir'] == NEIGHBOR_INVALID
    new_dir_arr['n_dir'][mask] = NEIGHBOR_INVALID
    new_dir_arr['dist'] = numpy.nan

    # Rescan and copy all lines:
    # - The ones ending in a 'seed' are flipped, thus the NEIGHBOR_SEED marker is replaced with
    #   a valid direction. This effectively changes the final part the other overlapping lines.
    #   Moreover, the ones that ended formerly in that 'seed' are extended to completely overlap
    #   the flipped one.
    # - The 'dist' members along the others are adjusted to reflect the change in that overlapping
    #   section.
    new_result_lines = numpy.empty(0, dtype=result_lines.dtype)
    progress_next = 0
    for start_xy, _ in reversed(result_lines):
        # Scan the line upto already updated point, 'seed' or 'stop' markers
        x_y = start_xy
        n_dir, start_dist = gdal_utils.read_arr(dir_arr, x_y)
        dist = start_dist
        while True:
            new_dir, new_dist = gdal_utils.read_arr(new_dir_arr, x_y)
            if new_dir != NEIGHBOR_PENDING:
                # An already updated point is reached, adjust by using the new distance
                adj_dist = new_dist - dist
                break
            if neighbor_is_invalid(n_dir):
                if n_dir == NEIGHBOR_SEED:
                    # A seed is reached, flip the line
                    adj_dist = None
                else:
                    # A stop-marker is reached, just copy the line
                    assert n_dir == NEIGHBOR_STOP
                    adj_dist = 0
                break

            x_y = neighbor_xy(x_y, n_dir)
            n_dir, dist = gdal_utils.read_arr(dir_arr, x_y)

        # Process the scanned line-segment
        if adj_dist is None:
            # This reached 'seed' - flip it, later others will merge to it
            start_xy = flip_line(new_dir_arr, dir_arr, start_xy)
            # Note:
            # The sorted_arr_insert() is called to handle the case when this is the only line
            # ending in that 'seed' (quite rare case). In other cases, at the next stage it will
            # be discarded: reduced_distance() will cut it zero, as it is completelly ovelapped.
        else:
            # This reached an already updated point - adjust all distances
            start_dist += adj_dist
            x_y = start_xy
            while True:
                new_dir, _ = gdal_utils.read_arr(new_dir_arr, x_y)
                if new_dir != NEIGHBOR_PENDING:
                    # Adjustment complete
                    break

                n_dir, dist = gdal_utils.read_arr(dir_arr, x_y)
                gdal_utils.write_arr(new_dir_arr, x_y, (n_dir, dist + adj_dist))
                if n_dir == NEIGHBOR_STOP:
                    # Copy complete
                    break
                assert not neighbor_is_invalid(n_dir)
                x_y = neighbor_xy(x_y, n_dir)

        #assert start_dist == reduced_distance(new_dir_arr, start_xy)
        new_result_lines = sorted_arr_insert(new_result_lines,
                numpy.array((start_xy, start_dist), dtype=result_lines.dtype), 'dist')

        #
        # Progress, total ~10 messages
        #
        if progress_next < new_result_lines.shape[0]:
            print('  Adjusted', new_result_lines.shape[0], 'lines, max/mid/min len %d/%d/%d'%(
                    new_result_lines[-1]['dist'], new_result_lines[new_result_lines.shape[0] // 2]['dist'], new_result_lines[0]['dist']))
            progress_next += result_lines.shape[0] // 10

    return new_dir_arr, new_result_lines

#
# Third stage - combine lines
#
def reduced_distance(dir_arr, x_y):
    """Calculate trace distance upto the next NEIGHBOR_STOP"""
    start_dist = gdal_utils.read_arr(dir_arr['dist'], x_y)
    while True:
        n_dir, dist = gdal_utils.read_arr(dir_arr, x_y)
        if neighbor_is_invalid(n_dir):
            return start_dist - dist
        x_y = neighbor_xy(x_y, n_dir)

def combine_lines(result_lines, dir_arr, min_len=0):
    """Create polylines from previously generated ridges or valleys"""
    # Remove the lines shorter than min_len
    idx = numpy.searchsorted(result_lines['dist'], min_len)
    result_lines = result_lines[idx:]

    #
    # Extract the longest lines one-by-one and cut all that overlaps it
    #
    polylines = []
    prev_dist = numpy.inf    # Assert only
    while result_lines.size:
        x_y, dist = result_lines[-1]
        result_lines = result_lines[:-1]
        assert dist <= prev_dist, 'Unsorted result_lines %d->%d'%(prev_dist, dist); prev_dist = dist
        assert dist >= min_len, 'Short line in result_lines %d/%d'%(dist, min_len)

        print('Generating line starting at point %s total length %d'%(x_y, dist))
        pline = numpy.empty((0, *x_y.shape), dtype=x_y.dtype)
        # Trace route
        start_xy = x_y
        while True:
            pline = numpy.append(pline, [x_y], axis=0)
            n_dir = gdal_utils.read_arr(dir_arr['n_dir'], x_y)
            if neighbor_is_invalid(n_dir):
                break
            # Stop other lines from overlapping that one
            gdal_utils.write_arr(dir_arr['n_dir'], x_y, NEIGHBOR_STOP)
            x_y = neighbor_xy(x_y, n_dir)
        polylines.append({'dist': dist, 'x_y': start_xy, 'line': pline})
        if not result_lines.size:
            break

        # Update distances after some of the lines were cut
        print('  Update remaining %d lines: mid/min len %d/%d'%(result_lines.shape[0], result_lines[result_lines.shape[0] // 2]['dist'], result_lines[0]['dist']))
        progress_next = 1
        idx = result_lines.shape[0] - 1
        while idx >= 0:
            rline = result_lines[idx]
            dist = reduced_distance(dir_arr, rline['x_y'])
            if dist == rline['dist']:
                #
                # Progress (logarithmic), total ~10 messages
                #
                if progress_next <= 1 + idx / result_lines.shape[0]:
                    print('    Updated line at %d/%d: len %d'%(idx, result_lines.shape[0], result_lines[idx]['dist']))
                    progress_next *= 1.07   # 2^(1/10)
            else:
                assert dist < rline['dist'], 'Line distance was increased %d->%d'%(rline['dist'], dist)
                # Move the entry to keep 'result_lines' sorted by distance
                result_lines = numpy.delete(result_lines, idx)
                if dist >= min_len:
                    #print('    Updating line at %d/%d, distance %d->%d'%(idx, result_lines.shape[0], rline['dist'], dist))
                    rline['dist'] = dist
                    result_lines = sorted_arr_insert(result_lines, rline, 'dist', idx)
                    idx += 1    # Hold the next decrement
            idx -= 1

    return polylines

#
# Keep/resume support
#
def keep_arrays(prefix, arr_slices):
    """Store snapshots of multiple arrays"""
    for arr_name in arr_slices:
        arr = arr_slices[arr_name][0]
        slices = arr_slices[arr_name][1:]
        for sl in slices:
            if sl is None:
                # Single slice - complete array
                arr_sl_name = arr_name
                arr_sl = arr
            else:
                arr_sl_name = '%s[%s]'%(arr_name, sl)
                arr_sl = arr[sl]
            fname = '%s%s.npy'%(prefix, arr_sl_name)
            print('Keeping snapshot of', arr_sl_name, arr_sl.shape, ':', fname)
            numpy.save(fname, arr_sl)

def restore_arrays(prefix, arr_slices, restore=False):
    """Load snapshots of multiple arrays"""
    res_list = []
    for arr_name in arr_slices:
        dtype = arr_slices[arr_name]
        arr = None
        for sl in dtype:
            if sl is None:
                # Single slice - complete array
                arr_sl_name = arr_name
            else:
                arr_sl_name = '%s[%s]'%(arr_name, sl[0])

            fname = '%s%s.npy'%(prefix, arr_sl_name)
            print('Restoring snapshot of', arr_sl_name, ':', fname)
            data = numpy.load(fname)
            print('  Restored: shape', data.shape, ', dtype', data.dtype)

            if sl is None:
                arr = data
            else:
                if arr is None:
                    # Create array by using the correct shape, esp. multidimensional sub-array
                    shape = numpy.empty(0, dtype=dtype)[sl[0]].shape[1:]
                    shape = data.shape[:data.ndim - len(shape)]
                    arr = numpy.empty(shape, dtype=dtype)
                arr[sl[0]] = data

        res_list.append(arr)
    return res_list

def keep_result_lines_dir_arr(prefix, result_lines, dir_arr):
    """Store snapshots of the 'result_lines' and 'dir_arr' arrays"""
    keep_arrays(prefix, {
            'result_lines': (result_lines, 'x_y', 'dist'),
            'dir_arr': (dir_arr, 'n_dir', 'dist'),
    })

def restore_result_lines_dir_arr(prefix):
    """Load snapshots of the 'result_lines' and 'dir_arr' arrays"""
    return restore_arrays(prefix, {
            'result_lines': RESULT_LINE_DTYPE,
            'dir_arr': DIR_DIST_DTYPE,
        })

#
# Main processing
#
def main(argv):
    """Main entry"""
    valleys = False
    boundary_val = None
    truncate = True
    src_filename = dst_filename = None
    while argv:
        if argv[0][0] == '-':
            if argv[0] == '-h':
                return print_help()
            if argv[0] == '-valley':
                valleys = True
            elif argv[0] == '-boundary_val':
                argv = argv[1:]
                boundary_val = float(argv[0])
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
    if RESUME_FROM_SNAPSHOT < 1:

        start = time.time()

        # Actual trace
        result_lines, dir_arr = trace_ridges(dem_band, valleys, boundary_val)
        if result_lines is None:
            print('Error: Failed to trace ridges', file=sys.stderr)
            return 2

        duration = time.time() - start
        print('Traced total %d lines, max length %d, %d sec'%(result_lines.shape[0], result_lines[-1]['dist'], duration))

        if KEEP_SNAPSHOT:
            keep_result_lines_dir_arr(src_filename + '-1-', result_lines, dir_arr)
    elif RESUME_FROM_SNAPSHOT == 1:
        result_lines, dir_arr = restore_result_lines_dir_arr(src_filename + '-1-')

    #
    # Flip the longest lines ending in each 'seed' and recalculate the 'dist' members
    # The seeds become the former last point of these lines
    #
    if RESUME_FROM_SNAPSHOT < 2:

        start = time.time()

        # Actual flip
        dir_arr, result_lines = flip_seed_lines(dir_arr, result_lines)
        if dir_arr is None:
            print('Error: Failed to flip longest lines', file=sys.stderr)
            return 2

        duration = time.time() - start
        print('Flip & merge longest lines, %d sec'%(duration))

        if KEEP_SNAPSHOT:
            keep_result_lines_dir_arr(src_filename + '-2-', result_lines, dir_arr)
    elif RESUME_FROM_SNAPSHOT == 2:
        result_lines, dir_arr = restore_result_lines_dir_arr(src_filename + '-2-')

    #
    # Combine traced lines, longer than one-tenth of the longest one
    #
    start = time.time()

    polylines = combine_lines(result_lines, dir_arr, result_lines[-1]['dist'] / 10)
    if not polylines:
        print('Error: Failed to combine lines', file=sys.stderr)
        return 2

    duration = time.time() - start
    print('Created total %d polylines, first %d, last %d points, %d sec'%(len(polylines),
            len(polylines[0]['line']), len(polylines[-1]['line']), duration))

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

        for entry in polylines:
            geom = dst_layer.create_feature_geometry(gdal_utils.wkbLineString)
            dist = entry['dist']
            geom.set_field('Name', '%dm'%dist if dist < 10000 else '%dkm'%round(dist/1000))
            geom.set_style_string(VECTOR_FEATURE_STYLE(valleys))
            # Reverse the line to match the tracing direction
            for x_y in entry['line'][::-1]:
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
    print('\t-boundary_val <ele> - Treat the neighbors next to <ele> as boundary')
    return 0 if err_msg is None else 255

if __name__ == '__main__':
    ret = main(sys.argv[1:])
    if ret:
        exit(ret)
