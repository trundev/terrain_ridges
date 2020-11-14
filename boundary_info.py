"""Generate/append informational OGR geometry for a DEM"""
import sys
import numpy
import gdal_utils

DEM_BOUNDARY_FEATURE_STYLE = 'PEN(c:#00FF00,w:1px);BRUSH(fc:#00FF0020)'
DEM_NODATA_FEATURE_STYLE = 'PEN(c:#FF0000,w:1px);BRUSH(fc:#FF000080)'

def get_mgrid(org_xy, size_xy, full=False):
    """Create a 2x2 grid around a boundary, or a full grid"""
    end_xy = numpy.array(org_xy) + numpy.array(size_xy)
    if full:
        size_xy = (1,1)
    else:
        end_xy += 1
    mgrid = numpy.mgrid[org_xy[0]:end_xy[0]:size_xy[0], org_xy[1]:end_xy[1]:size_xy[1]]
    # The coordinates are in the last dimension
    return numpy.moveaxis(mgrid, 0, -1)

def create_no_data_geom(dem_band, dst_layer):
    """Add polygons around "NoDataValue" points"""
    nodata_mask = numpy.isnan(dem_band.get_elevation(True))
    progress_idx = 0
    while nodata_mask.any():
        # Extract a "NoDataValue" point
        x_y = numpy.array(numpy.unravel_index(nodata_mask.argmax(), nodata_mask.shape))
        gdal_utils.write_arr(nodata_mask, x_y, False)

        # Obtain coordinates around the point
        boundary = dem_band.xy2lonlat(get_mgrid(x_y, (1,1)), False)
        ring = dst_layer.create_feature_geometry(gdal_utils.wkbLinearRing)
        # Create geometry, second boundary row must be reversed
        for lonlat in (*boundary[0], *boundary[1][::-1]):
            ring.add_point(*lonlat)

        geom = dst_layer.create_feature_geometry(gdal_utils.wkbPolygon)
        geom.set_field('Name', 'NoDataValue %d,%d'%(*x_y,))
        geom.set_style_string(DEM_NODATA_FEATURE_STYLE)
        geom.add_geometry(ring)
        geom.create()

        progress_idx += 1

    print('Created %d "NoDataValue" markers'%progress_idx)
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
