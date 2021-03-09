"""GDAL Contour generation experiments"""
import sys
import gdal_utils

TARGET_LAYER_NAME = 'contour'

#
# Main processing
#
def main(argv):
    """Main entry"""
    src_filename = argv[0]
    dst_filename = argv[1]

    # Load DEM
    dem_band = gdal_utils.dem_open(src_filename)
    if dem_band is None:
        return 1

    dst_ds = gdal_utils.vect_create(dst_filename)
    if dst_ds is None:
        return 1

    # Delete existing layer
    lyr = gdal_utils.gdal_vect_layer(dst_ds, TARGET_LAYER_NAME)
    if lyr.get_name() is not None:
        print('  Deleting layer', lyr.get_name())
        dst_ds.delete_layer(lyr.get_name())

    dst_layer = gdal_utils.gdal_vect_layer.create(dst_ds, TARGET_LAYER_NAME,
                    srs=dem_band.get_spatial_ref(),
                    geom_type=gdal_utils.wkbLineString)
    id_fld = dst_layer.create_field('ID', gdal_utils.OFTInteger)
    if 'KML' in dst_ds.get_drv_name():
        elev_fld = dst_layer.create_field('Name', gdal_utils.OFTString)   # KML <name>
    else:
        elev_fld = dst_layer.create_field('elev', gdal_utils.OFTReal)
    dst_layer.flush_cache()
    ogr_lyr = dst_layer.layer


    def cb_fn(v0,v1,v2):
        print(__name__, v0,v1,v2)

    print('Generating contours every', 100, 'm')
    dem_band.contour_generate(100, 0, ogr_lyr, id_fld, elev_fld, cb_fn, 'callback_data')
    dst_layer.flush_cache()

    dem_band.load()
    ele = dem_band.get_elevation(True).mean()
    print('Generating contours at', ele)
    dem_band.contour_generate([ele], None, ogr_lyr, id_fld, elev_fld, cb_fn, 'callback_data')
    dst_layer.flush_cache()

if __name__ == '__main__':
    ret = main(sys.argv[1:])
    if ret:
        exit(ret)
