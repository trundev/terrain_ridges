"""GDAL Contour generation experiments"""
import sys
import gdal_utils
####REMOVEME: Functionality to be moved into gdal_utils
from osgeo import gdal

TARGET_LAYER_NAME = 'contour'

#
# Main processing
#
def main(argv):
    """Main entry"""
    src_filename = argv[0]
    dst_filename = argv[1]

    # Load template-only DEM
    dem_band = gdal_utils.dem_open(src_filename)
    if dem_band is None:
        return 1

    # Made-up some data to be written
    dem_band.load(0, 0)
    ele = dem_band.get_elevation(True)
    raster = ele > ele.mean()

    src_filename = '/vsimem/mask_only'

    # Create empty mask-only dataset using the source driver
    driver = dem_band.dataset.GetDriver()
    dst_ds = driver.Create(src_filename, *dem_band.dem_buf.shape, 1, gdal.GDT_Byte)
    dst_ds.SetProjection(dem_band.get_spatial_ref().ExportToWkt())
    dst_ds.SetGeoTransform(dem_band.xform.flatten())
    dem_band = gdal_utils.gdal_dem_band(dst_ds, 1)
    del dst_ds

    # Create a diagonal line
    for i in range(1, min(raster.shape) - 1):
        raster[i,i] = True

    # Ensure there is a "frame"
    raster[0,:] = raster[-1,:] = False
    raster[:,0] = raster[:,-1] = False

    # Put the data
    dem_band.band.WriteArray(raster.T, 0, 0)

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

    ele = .2    # Give some thickness of single pixel lines
    print('Generating contours at', ele)
    dem_band.contour_generate([ele], None, ogr_lyr, id_fld, elev_fld, cb_fn, 'callback_data')
    dst_layer.flush_cache()

if __name__ == '__main__':
    ret = main(sys.argv[1:])
    if ret:
        exit(ret)
