"""Generate terrain ridges/valeys"""
import sys
import numpy
import gdal_utils

# Neighbor indices:
#   -4 -3 -2
#   -1 <0> 1
#    2  3  4
VALID_NEIGHBORS = (-4, -3, -2, -1, 1, 2, 3, 4)
NEIGHBOR_SELF = 0
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

def generate_ridges(dem_band, dst_layer=None, valleys=False):
    """Generate terrain ridges or valeys"""
    # Start at the max/min altitude (first one)
    flat_idx = dem_band.dem_buf.argmin() if valleys else dem_band.dem_buf.argmax()
    seed_xy = numpy.unravel_index(flat_idx, dem_band.dem_buf.shape)
    seed_xy = numpy.array(seed_xy)
    print('Seed point', seed_xy, ', altitude', dem_band.get_elevation(seed_xy))

    #
    # Previous index array
    # Initialize with invalid value.
    #
    prev_arr = numpy.full(dem_band.dem_buf.shape, NEIGHBOR_INVALID, dtype=NEIGHBOR_IDX_DTYPE)
    gdal_utils.write_arr(prev_arr, seed_xy, NEIGHBOR_SELF)

    #
    # Tentative point list (coord and distance)
    # Initially contains the start point only
    #
    tentative = [(seed_xy, 0)]

    #
    # End-points of generated lines (coord and distance)
    #
    result_lines = []

    distance = gdal_utils.geod_distance(dem_band)
    while tentative:
        x_y, dist = tentative.pop()
        #print('    Processing point %s dist %d alt %s'%(x_y, dist, dem_band.get_elevation(seed_xy)))
        end_of_line = True
        for t_xy, n_idx in valid_neighbors(dem_band, x_y):
            if gdal_utils.read_arr(prev_arr, t_xy) == NEIGHBOR_INVALID:
                end_of_line = False
                # Keep the inverted neighbor index to later track this back
                gdal_utils.write_arr(prev_arr, t_xy, neighbor_inv(n_idx))
                alt = dem_band.get_elevation(t_xy)
                # Insert the point in 'tentative' by keeping it sorted by altitude
                def cmp_fn(val, *args):
                    check_xy, _ = val
                    # Ascending sort  (<), insert a duplicated altitude at lower index (<=).
                    # Thus, duplicated altitudes will be processed in order of appearance (FIFO).
                    return -1 if alt <= dem_band.get_elevation(check_xy) else 1
                idx, _ = search_sorted(tentative, cmp_fn)
                tentative.insert(idx, (t_xy, dist + distance.get_distance(x_y, t_xy)))

        if end_of_line:
            print('  Line finished at point %s total length %d'%(x_y, dist))
            # Keep this end-point in 'result_lines' by keeping it sorted by distance
            idx, _ = search_sorted(result_lines, lambda v: 1 if dist < v[1] else -1)
            result_lines.insert(idx, (x_y, dist))

    # Process first 10 (longest) lines
    for x_y, dist in result_lines[:10]:
        print('Generating line starting at point %s total length %d'%(x_y, dist))
        if dst_layer is not None:
            geom = dst_layer.create_feature_geometry(gdal_utils.wkbLineString)
            # Trace route
            while x_y is not None:
                geom.add_point(*dem_band.xy2lonlatalt(x_y))
                n_idx = gdal_utils.read_arr(prev_arr, x_y)
                x_y = None if n_idx == NEIGHBOR_SELF else neighbor_xy(x_y, n_idx)
            geom.create()
            #TODO: Currently, first line only
            break

    return 0

def main(argv):
    """Main entry"""
    valleys = False
    src_filename = dst_filename = None
    while argv:
        if argv[0][0] == '-':
            if argv[0] == '-h':
                return print_help()
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

    dst_layer = gdal_utils.gdal_vect_layer.create(dst_ds, VECTOR_LAYER_NAME(valleys), dem_band.get_spatial_ref())
    if dst_layer is None:
        print('Error: Unable to create layer', file=sys.stderr)
        return 1

    # Generate ridges/valleys
    dem_band.load()
    res = generate_ridges(dem_band, dst_layer, valleys)
    return res

def print_help(err_msg=None):
    if err_msg:
        print('Error:', err_msg, file=sys.stderr)
    print('Usage:', sys.argv[0], '[<options>] <src_filename> <dst_filename>')
    print('\tOptions:')
    print('\t-h\t- This screen')
    return 0 if err_msg is None else 255

if __name__ == '__main__':
    ret = main(sys.argv[1:])
    if ret:
        exit(ret)
