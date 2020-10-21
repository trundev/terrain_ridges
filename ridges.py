"""Generate terrain ridges/valleys"""
import sys
import time
import numpy
import gdal_utils

# Neighbor indices:
#   -4 -3 -2
#   -1 <0> 1
#    2  3  4
VALID_NEIGHBORS = (-4, -3, -2, -1, 1, 2, 3, 4)
NEIGHBOR_SELF = 0
NEIGHBOR_PENDING = 100
NEIGHBOR_INVALID = -100
NEIGHBOR_IDX_DTYPE = numpy.int8

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
def neighbor_xy(x_y, neighbor_idx):
    """Get the coordinates of a neighbor pixel"""
    if neighbor_idx < -4 or neighbor_idx > 4:
        return None
    neighbor_idx += 4
    return x_y + [neighbor_idx % 3 - 1, neighbor_idx // 3 - 1]

def valid_neighbors(dem_band, x_y):
    """Return list of valid neighbor points"""
    for idx in VALID_NEIGHBORS:
        nxy = neighbor_xy(x_y, idx)
        if dem_band.in_bounds(nxy):
            yield nxy, idx

def neighbor_inv(neighbor_idx):
    """Get the inverted neighbor idx"""
    if neighbor_idx in VALID_NEIGHBORS:
        return -neighbor_idx
    return None

def trace_distance(prev_arr, x_y):
    """Calculate trace distance"""
    dist = 0.
    while True:
        n_idx, n_dist = gdal_utils.read_arr(prev_arr, x_y)
        if n_idx == NEIGHBOR_SELF:
            return dist
        dist += n_dist
        x_y = neighbor_xy(x_y, n_idx)

def result_lines_insert(result_lines, entry):
    """Insert entry in result_lines by keeping it sorted by distance (entry[1])"""
    idx, _ = search_sorted(result_lines, lambda v: 1 if entry[1] < v[1] else -1)
    result_lines.insert(idx, entry)

def trace_ridges(dem_band, valleys=False):
    """Generate terrain ridges or valleys"""
    # Start at the max/min altitude (first one)
    flat_idx = dem_band.dem_buf.argmin() if valleys else dem_band.dem_buf.argmax()
    seed_xy = numpy.unravel_index(flat_idx, dem_band.dem_buf.shape)
    seed_xy = numpy.array(seed_xy)
    print('Seed point', seed_xy, ', altitude', dem_band.get_elevation(seed_xy))

    #
    # Previous index and distance array
    # Initialize with invalid value.
    #
    prev_arr = numpy.full(dem_band.dem_buf.shape, NEIGHBOR_PENDING, dtype=[
            ('n_idx', NEIGHBOR_IDX_DTYPE),
            ('dist', numpy.float),
        ])
    gdal_utils.write_arr(prev_arr, seed_xy, (NEIGHBOR_SELF, 0.))

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
        for t_xy, n_idx in valid_neighbors(dem_band, x_y):
            if gdal_utils.read_arr(prev_arr['n_idx'], t_xy) == NEIGHBOR_PENDING:
                n_dist = distance.get_distance(x_y, t_xy)
                if numpy.isnan(n_dist):
                    # Stop at this point as the altitude is unknown
                    gdal_utils.write_arr(prev_arr, t_xy, (NEIGHBOR_INVALID, 0.))
                    continue

                end_of_line = False
                # Keep the inverted neighbor index to later track this back
                gdal_utils.write_arr(prev_arr, t_xy, (neighbor_inv(n_idx), n_dist))
                alt = dem_band.get_elevation(t_xy)
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
            dist = trace_distance(prev_arr, x_y)
            #print('  Line finished at point %s total length %d'%(x_y, dist))
            # Keep this end-point in 'result_lines'
            result_lines_insert(result_lines, (x_y, dist))

    return result_lines, prev_arr

def combine_lines(result_lines, prev_arr, min_len=0):
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
            n_idx = gdal_utils.read_arr(prev_arr['n_idx'], x_y)
            # Stop other lines from overlapping that one
            gdal_utils.write_arr(prev_arr['n_idx'], x_y, NEIGHBOR_SELF)
            x_y = None if n_idx == NEIGHBOR_SELF else neighbor_xy(x_y, n_idx)
        polylines.append(pline)

        # Update distances after some of the lines were cut
        print('  Updating line distances')
        idx = 0
        while idx < len(result_lines):
            x_y, old_dist = result_lines[idx]
            dist = trace_distance(prev_arr, x_y)
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
    result_lines, prev_arr = trace_ridges(dem_band, valleys)
    if result_lines is None:
        return 1
    duration = time.time() - start
    print('Traced total %d lines, max length %d, %d sec'%(len(result_lines), result_lines[0][1], duration))

    # Combine traced lines, longer than quarter of the longest one
    start = time.time()
    polylines = combine_lines(result_lines, prev_arr, result_lines[0][1] / 4)
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
