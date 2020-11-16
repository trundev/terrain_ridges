"""Generate/append informational OGR geometry for a DEM"""
import sys
import time
import numpy
import gdal_utils

DEM_BOUNDARY_FEATURE_STYLE = 'PEN(c:#208020,w:1px);BRUSH(fc:#10801020)'
DEM_NODATA_FEATURE_STYLE = 'PEN(c:#802020,w:1px);BRUSH(fc:#80101080)'

def get_mgrid(org_xy, size_xy, full=False):
    """Create a 2x2 grid around a boundary, or a full grid"""
    if full:
        mgrid = numpy.mgrid[:size_xy[0], :size_xy[1]]
    else:
        mgrid = numpy.mgrid[:2,:2] * numpy.array(size_xy)[:,numpy.newaxis,numpy.newaxis]
    # The coordinates are in the last dimension
    return numpy.moveaxis(mgrid, 0, -1) + org_xy

class expand_mask:
    """Mask expansion tool"""
    def __init__(self, shape):
        """Prepare the map-array toward the 4-directions"""
        plain_map = get_mgrid((0,0), shape, True)
        # Map toward the four directions (right, up, left, down), to be used with read_arr()
        self.four_dir_map = numpy.stack((
            # right
                numpy.concatenate((plain_map[:1], plain_map[:-1]), axis=0),
            # up
                numpy.concatenate((plain_map[:,:1], plain_map[:,:-1]), axis=1),
            # left
                numpy.concatenate((plain_map[1:], plain_map[-1:]), axis=0),
            # down
                numpy.concatenate((plain_map[:,1:], plain_map[:,-1:]), axis=1)))

    def expand(self, src_mask, target_mask):
        """Expand a mask till completely fills island(s) in the target"""
        while True:
            # Expand the mask toward the 4 directions
            mask = gdal_utils.read_arr(src_mask, self.four_dir_map).any(axis=0)
            mask |= src_mask
            # ...then, cut the 'target_mask'
            mask &= target_mask
            # Check if the expansion has stopped
            if (src_mask == mask).all():
                return src_mask
            src_mask = mask

def create_ring_by_mask(dst_layer, dem_band, mask):
    """Create ring geometry around masked pixels"""
    ring = dst_layer.create_feature_geometry(gdal_utils.wkbLinearRing)
    # The 4 90-deg rotations of the unit x-vector, clockwise
    x_rots = numpy.array(((1,0), (0,1), (-1,0), (0,-1)), dtype=numpy.int8)

    # Helper finctions
    def is_masked(x_y):
        """Safe read from mask array, False if outside"""
        if (x_y < 0).any() or (x_y >= mask.shape).any():
            return False
        return gdal_utils.read_arr(mask, x_y)
    def add_point(x_y, dir):
        """Add the point at the 'correct' edge of pixel"""
        # Workaround for "Ring Self-intersection at or near point..." / "Invalid polygon":
        # Deflate the geometry by 1%
        dir = dir * .495
        ring.add_point(*dem_band.xy2lonlat(x_y - dir))
    def write_arr_safe(arr, x_y, val):
        """Write to array by ignoring out of bounds locations"""
        if (x_y >= 0).all() and (x_y < arr.shape).all():
            gdal_utils.write_arr(arr, x_y, val)

    frame_mask = numpy.ones_like(mask)
    start_xy = numpy.array(numpy.unravel_index(mask.argmax(), mask.shape))

    # Trace around this shape by using the right-hand rule
    x_y = start_xy.copy()
    while True:
        # Check the forward pixels
        if is_masked(x_y + x_rots[0]):
            # Masked: move forward
            x_y += x_rots[0]
            # Check the left pixel (counter-clockwise)
            if is_masked(x_y + x_rots[-1]):
                # Masked: add point, move to left, rotate counter-clockwise
                add_point(x_y, x_rots[0] + x_rots[1])
                x_y += x_rots[-1]
                x_rots = numpy.roll(x_rots, 1, axis=0)
            else:
                write_arr_safe(frame_mask, x_y + x_rots[-1], False)
        else:
            # Not masked: rotate clockwise, add point (no move)
            x_rots = numpy.roll(x_rots, -1, axis=0)
            add_point(x_y, x_rots[0] + x_rots[1])
            write_arr_safe(frame_mask, x_y + x_rots[-1], False)

        # Check for loop completion
        # (only allowed when moving toward as when started)
        if (x_y == start_xy).all() and x_rots[0,0] == 1:
            return ring, start_xy, frame_mask

def create_no_data_geom(dem_band, dst_layer):
    """Add polygons around "NoDataValue" points"""
    start = time.time()

    # Prepare the "expand mask" tool
    nodata_mask = numpy.isnan(dem_band.get_elevation(True))
    expand = expand_mask(nodata_mask.shape)

    num_geometries = 0
    num_islands = 0
    while nodata_mask.any():
        # Create ring geometry around some of the islands
        ring, x_y, frame_mask = create_ring_by_mask(dst_layer, dem_band, nodata_mask)
        # Expand the mask till it fills this island
        cur_mask = numpy.zeros_like(nodata_mask)
        gdal_utils.write_arr(cur_mask, x_y, True)
        cur_mask = expand.expand(cur_mask, nodata_mask)
        nodata_mask ^= cur_mask
        assert ((frame_mask & cur_mask) == cur_mask).all(), 'The "frame_mask" ovelaps "cur_mask"'

        geom = dst_layer.create_feature_geometry(gdal_utils.wkbPolygon)
        geom.set_field('Name', 'NoDataValue: %d at %d,%d'%(numpy.count_nonzero(cur_mask), *x_y))
        geom.set_style_string(DEM_NODATA_FEATURE_STYLE)
        geom.add_geometry(ring)

        #
        # Create ring geometries inside the mask (islands inside this island are not supported)
        #
        # Expand the mask to fill all its islands, then XOR it to the original
        cur_mask = expand.expand(cur_mask, frame_mask) ^ cur_mask
        while cur_mask.any():
            ring, x_y, _ = create_ring_by_mask(dst_layer, dem_band, cur_mask)
            geom.add_geometry(ring)
            # Expand the mask till it fills this island
            mask = numpy.zeros_like(cur_mask)
            gdal_utils.write_arr(mask, x_y, True)
            mask = expand.expand(mask, cur_mask)
            cur_mask ^= mask

            num_islands += 1

        geom.create()
        num_geometries += 1

    print('Created %d "NoDataValue" polygons, %d islands inside, %d sec'%(
            num_geometries, num_islands, time.time() - start))
    return True

def create_boundary_geom(dem_band, dst_layer):
    """Create polygon abound DEM boundary"""
    dst_layer.create_field('Name', True)    # KML <name>

    geom = dst_layer.create_feature_geometry(gdal_utils.wkbPolygon)
    geom.set_field('Name', 'DEM boundary')
    geom.set_style_string(DEM_BOUNDARY_FEATURE_STYLE)

    shape = dem_band.get_elevation(True).shape
    # Obtain coordinates at the boundary points
    boundary = dem_band.xy2lonlat(get_mgrid((0,0), shape), False)
    ring = dst_layer.create_feature_geometry(gdal_utils.wkbLinearRing)
    # Create geometry, second boundary row must be reversed
    for lonlat in (*boundary[0], *boundary[1][::-1]):
        ring.add_point(*lonlat)

    geom.add_geometry(ring)
    geom.create()

    print('Created boundary polygon %.6f %.6f %.6f %.6f'%(
            *boundary[0,0], *boundary[-1,-1]))
    return True

def create_info_geometries(dem_band, dst_ds):
    """Create complete OGR info layer"""
    # Create the OGR layer
    dst_layer = gdal_utils.gdal_vect_layer.create(dst_ds, 'Bounds / info',
            srs=dem_band.get_spatial_ref(), geom_type=gdal_utils.wkbPolygon)
    if dst_layer is None:
        print('Error: Unable to create layer', file=sys.stderr)
        return None

    # Visualization of DEM boundaries
    if not create_boundary_geom(dem_band, dst_layer):
        print('Error: Unable to create DEM boundary geometry', file=sys.stderr)
        return None

    # Visualization of DEM 'NoDataValue' points
    if not create_no_data_geom(dem_band, dst_layer):
        print('Warnig: Unable to create "NoDataValue" geometries', file=sys.stderr)

    return dst_layer

def main(argv):
    """Main entry"""
    truncate = True
    src_filename = dst_filename = None
    while argv:
        if argv[0][0] == '-':
            if argv[0] == '-h':
                return print_help()
            if argv[0] == '-a':
                truncate = False
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

    if truncate:
        for i in reversed(range(dst_ds.get_layer_count())):
            print('  Deleting layer', gdal_utils.gdal_vect_layer(dst_ds, i).get_name())
            dst_ds.delete_layer(i)

    # Create all info geometries
    if create_info_geometries(dem_band, dst_ds) is None:
        return 1

    return 0

def print_help(err_msg=None):
    if err_msg:
        print('Error:', err_msg, file=sys.stderr)
    print('Usage:', sys.argv[0], '[<options>] <src_filename> <dst_filename>')
    print('\tOptions:')
    print('\t-h\t- This screen')
    print('\t-a\t- Append to the existing OGR geometry')
    return 0 if err_msg is None else 255

if __name__ == '__main__':
    ret = main(sys.argv[1:])
    if ret:
        exit(ret)
