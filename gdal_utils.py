"""GDAL/pyproj backend and utilities"""
import os
import sys
import numpy
from osgeo import gdal, ogr, osr
try:
    import pyproj
except ImportError as ex:
    print('Warning:', ex, '- geod_distance is unavailable', file=sys.stderr)


#
# Multi-index array access
#
def write_arr(arr, x_y, val):
    """Put data to multiple indices in array"""
    # Force numpy "Basic Indexing", note that '[x_y]' will trigger "Advanced Indexing"
    if numpy.isscalar(val):
        arr[tuple(x_y.T)] = val     # Performance optimization
    elif x_y.ndim < 2 or arr.ndim <= x_y.shape[-1]:
        arr[tuple(x_y.T)] = val.T   # Performance optimization
    else:
        # Double-transpose trick does not work when 'arr' has extra dimentions
        x_y = numpy.moveaxis(x_y, -1, 0)
        arr[tuple(x_y)] = val

def read_arr(arr, x_y):
    """Get multiple indices from array"""
    # Avoid numpy "Advanced Indexing"
    if x_y.ndim < 2 or arr.ndim <= x_y.shape[-1]:
        return arr[tuple(x_y.T)].T  # Performance optimization
    # Double-transpose trick does not work when 'arr' has extra dimentions
    x_y = numpy.moveaxis(x_y, -1, 0)
    return arr[tuple(x_y)]

#
# GDAL helpers
#
# Selected dataset (driver) capabilities
ODsCCreateLayer = ogr.ODsCCreateLayer
ODsCDeleteLayer = ogr.ODsCDeleteLayer
ODsCCreateGeomFieldAfterCreateLayer = ogr.ODsCCreateGeomFieldAfterCreateLayer
ODsCCurveGeometries = ogr.ODsCCurveGeometries
ODsCTransactions = ogr.ODsCTransactions
ODsCEmulatedTransactions = ogr.ODsCEmulatedTransactions
ODsCRandomLayerRead = ogr.ODsCRandomLayerRead
ODsCRandomLayerWrite = ogr.ODsCRandomLayerWrite

class gdal_dataset:
    """"GDAL dataset representation"""
    def __init__(self, dataset):
        self.dataset = dataset

    def update_xform(self, offset=None):
        """Build the transformation matrix from GetGeoTransform() with optional offset"""
        geo_xform = self.dataset.GetGeoTransform()
        self.xform = numpy.array( geo_xform ).reshape(2,3)
        # Offset the transformation matrix
        if offset is not None:
            txform = numpy.identity(3)
            txform[1:3, 0] = offset
            self.xform = numpy.matmul(self.xform, txform)

    def affine_xform(self, x_y):
        """Convert raster (x,y) to projection coordinate(s)"""
        # Add ones in front of the coordinates to handle translation
        # Note that GetGeoTransform() returns translation components at index 0
        ones = numpy.broadcast_to([1], [*x_y.shape[:-1], 1])
        x_y = numpy.concatenate((ones, x_y), axis=-1)

        # This is matmul() but x_y is always treated as a set of one-dimentional vectors
        x_y = x_y[...,numpy.newaxis,:]
        return (self.xform * x_y).sum(-1)

    def build_srs_xform(self, tgt_srs):
        """Create transformation to another SRS"""
        return osr.CoordinateTransformation(self.get_spatial_ref(), tgt_srs)

    def build_geogcs_xform(self):
        """Create transformation to GEOGCS, 'None' if not needed"""
        srs = self.get_spatial_ref()
        # Assume geographic, when SRS is missing (no .prj file)
        if srs is None or srs.IsGeographic():
            return None     # Already geographic
        return self.build_srs_xform(srs.CloneGeogCS())

    @staticmethod
    def coord_xform(srs_xform, coords):
        """Transform coordinates, see build_srs_xform()"""
        # TransformPoints() supports 2D arrays only
        orig_shape = coords.shape
        coords = coords.reshape(-1, orig_shape[-1])
        coords = srs_xform.TransformPoints(coords)
        # Strip the added 0 altitudes, if no such in the source
        coords = numpy.array(coords)[...,:orig_shape[-1]]
        # Resize back the array
        return coords.reshape(orig_shape)

    def get_spatial_ref(self):
        return self.dataset.GetSpatialRef()

    def get_drv_name(self):
        drv = self.dataset.GetDriver()
        return drv.ShortName if hasattr(drv, 'ShortName') else None

    def test_capability(self, cap):
        return self.dataset.TestCapability(cap)

    def flush_cache(self):
        return self.dataset.FlushCache()

    def get_layer_count(self):
        return self.dataset.GetLayerCount()

    def delete_layer(self, layer):
        return self.dataset.DeleteLayer(layer)

#
# GDAL raster data helpers
#
def _get_dtype(band):
    """NumPy dtype from GDAL band DataType
    See gdal_array.GDALTypeCodeToNumericTypeCode(band.DataType)"""
    if band.DataType == gdal.GDT_Byte:
        return numpy.uint8
    return gdal.array_modes[band.DataType]

class gdal_dem_band(gdal_dataset):
    """"GDAL DEM band representation"""
    dem_buf = None
    shape = None

    def __init__(self, dataset, i=None):
        super(gdal_dem_band, self).__init__(dataset)
        self.update_xform()
        if i is not None:
            self.band = dataset.GetRasterBand(i)
            # Cached parameters
            self.scale = self.band.GetScale()
            if self.scale is None:
                self.scale = 1
            self.nodata_val = self.band.GetNoDataValue()
            if self.nodata_val is not None:
                self.nodata_val = numpy.array(self.nodata_val, dtype=_get_dtype(self.band))

    def load(self, xstart=0, ystart=0):
        """Load raster DEM data"""
        xsize = self.band.XSize - xstart
        ysize = self.band.YSize - ystart

        print('Reading %d x %d pixels (total %dMP)...' % (xsize, ysize, (xsize * ysize) / 1e6))
        buf = self.band.ReadRaster(xstart, ystart, xsize, ysize)
        self.dem_buf = numpy.frombuffer(buf, dtype=_get_dtype(self.band))

        # Swap dimensions to access the data like: self.dem_buf[x,y]
        self.dem_buf = self.dem_buf.reshape(ysize, xsize).transpose()
        # Update transformation matrix with the offset
        self.update_xform(numpy.array([xstart, ystart]))
        # Create GEOGCS transformation, to be used by xy2lonlat() (non-geographics only)
        self.geogcs_xform = self.build_geogcs_xform()

        # Replace the GDAL "NoDataValue" with NaN
        if self.nodata_val is not None:
            # Convert 'dem_buf' to float (specifically xform-type) to allow NaN assignment
            if self.dem_buf.dtype.kind != 'f' or not self.dem_buf.flags.writeable:
                alt_f = numpy.array(self.dem_buf, dtype=self.xform.dtype) 
                alt_f[self.dem_buf == self.nodata_val] = numpy.nan
                self.dem_buf = alt_f
            else:
                self.dem_buf[self.dem_buf == self.nodata_val] = numpy.nan

        self.shape = self.dem_buf.shape
        return True

    def in_bounds(self, x_y):
        """Check if a coordinate is inside the DEM array"""
        return numpy.logical_and(
            (x_y >= 0).all(-1),
            (x_y < self.dem_buf.shape).all(-1))

    def get_elevation(self, x_y):
        """Retrieve elevation(s), when "x_y is True" - complete array"""
        if x_y is True:
            alt = self.dem_buf
        else:
            alt = read_arr(self.dem_buf, x_y)
        return alt

    def xy2coords(self, x_y, center=True):
        """Convert raster (x,y) to the dataset's SRS coordinate(s)"""
        if center:
            # Adjust point to the center of the raster pixel
            # (otherwise, the coordinates will be at the top-left (NW) corner)
            x_y = x_y + .5
        return self.affine_xform(x_y)

    def xy2lonlat(self, x_y, center=True):
        """Convert raster (x,y) coordinate(s) to lon/lan (east,north)"""
        coords = self.xy2coords(x_y, center)
        if self.geogcs_xform is None:
            return coords
        # The geogcs_xform is valid when coordinates are non-geographic
        return self.coord_xform(self.geogcs_xform, coords)

    def xy2lonlatalt(self, x_y, center=True):
        """Convert raster (x,y) coordinate(s) to lon/lan/alt (east,north,alt)"""
        lon_lat = self.xy2lonlat(x_y, center)
        alt = self.get_elevation(x_y)[...,numpy.newaxis]
        return numpy.concatenate((lon_lat, alt), axis=-1)

#
# Coordinate transformation
#
class tm_transform:
    """Convert to Transverse Mercator projection"""
    def __init__(self, dem_band):
        self.dem_band = dem_band
        self.tm_xform = None

    def build_xform(self, x_y, scale=1, false_easting=0, false_northing=0):
        """Create transformation to Transverse Mercator centered at a raster pixel"""
        del self.tm_xform
        tm_srs = self.dem_band.get_spatial_ref()
        lonlat = self.dem_band.xy2lonlat(x_y)
        # Swap lon/lat to lat/lon, see SetTM()
        tm_srs.SetTM(*lonlat[-1::-1], scale, false_easting, false_northing)
        self.tm_xform = self.dem_band.build_srs_xform(tm_srs)

    def xy2tm(self, x_y, center=True):
        """Convert raster (x,y) to the selected Transverse Mercator"""
        return self.dem_band.coord_xform(self.tm_xform, self.dem_band.xy2coords(x_y, center))

#
# Distance calculator
#
class geod_distance:
    """Distance calculation by using pyproj.Geod.inv()"""
    def __init__(self, gdal_dem_band):
        self.dem_band = gdal_dem_band
        srs = gdal_dem_band.get_spatial_ref()
        # Assume WGS84, when SRS is missing
        if srs is None:
            self.geod = pyproj.Geod(ellps='WGS84')
        else:
            wkt = srs.ExportToWkt()
            self.geod = pyproj.crs.CRS.from_wkt(wkt).get_geod()

    def get_distance(self, xy0, xy1, flat=False):
        """Calculate distance between two points"""
        lonlatalt = self.dem_band.xy2lonlatalt(numpy.stack((xy0, xy1), axis=-2))
        if flat:
            disp = 0
        else:
            disp = lonlatalt[...,1,2] - lonlatalt[...,0,2]
        # Precise method by calling pyproj.Geod.inv() between points
        lonlat = lonlatalt[...,:2].reshape([*lonlatalt.shape[:-2], -1])
        lonlat = lonlat.T
        _, _, dist = self.geod.inv(*lonlat)
        # The pyproj.Geod.inv() distance can be a scalar
        if not numpy.isscalar(dist):
            dist = dist.T
        # Adjust distance with the altitude displacement
        return numpy.sqrt(dist*dist + disp*disp)

class tm_distance(tm_transform):
    """Distance calculation by difference between Transverse Mercator coordinates"""
    def __init__(self, dem_band):
        super(tm_distance, self).__init__(dem_band)
        # Transformation to Transverse Mercator with origin at the center of data
        x_y = numpy.array(dem_band.get_elevation(True).shape) // 2
        self.build_xform(x_y)

    def get_distance(self, xy0, xy1, flat=False):
        """Calculate distance between two points by transforming to TM"""
        xy01 = numpy.stack((xy0, xy1))
        vect = self.xy2tm(xy01)
        vect = vect[1] - vect[0]
        if not flat:
            # Get elevation displacements
            alts = self.dem_band.get_elevation(xy01)
            disp = (alts[1] - alts[0])[...,numpy.newaxis]
            # Combine to the horizontal displacements
            vect = numpy.concatenate((vect, disp), axis=-1)
        # Return the combined vector lengths
        return numpy.sqrt((vect * vect).sum(-1))

    def get_pixel_size(self, x_y):
        """Calculate pixel size in TM"""
        x_y = numpy.broadcast_to(x_y[...,numpy.newaxis,:], [*x_y.shape, 2])
        # Get coordinates of (+1,0) and (0,+1)
        xy1 = x_y + numpy.identity(2, dtype=x_y.dtype)
        return tm_distance.get_distance(self, x_y, xy1, True)

class draft_distance(tm_distance):
    """Fast (inaccurate) distance calculation"""
    def __init__(self, dem_band):
        super(draft_distance, self).__init__(dem_band)
        # Precalculate size of the pixel at the center of data
        x_y = numpy.array(dem_band.get_elevation(True).shape) // 2
        self.pixel_size = self.get_pixel_size(x_y)
        print('Precalculated pixel size at', x_y, ':', self.pixel_size)

    def get_distance(self, xy0, xy1, flat=False):
        """Calculate distance between two points by using pre-calculated 'pixel_size'"""
        # Get elevation displacements
        alts = self.dem_band.get_elevation(numpy.stack((xy0, xy1), axis=-2))
        disp = (alts[...,1] - alts[...,0])[...,numpy.newaxis]
        # Combine to the horizontal displacements
        vect = (xy1 - xy0) * self.pixel_size
        vect = numpy.concatenate((vect, disp), axis=-1)
        # Return the combined vector lengths
        return numpy.sqrt((vect * vect).sum(-1))

def dem_open(filename, band=1):
    """Open a raster DEM file for reading"""
    dataset = gdal.OpenEx(filename, gdal.OF_RASTER | gdal.GA_ReadOnly)
    if dataset is None:
        return None

    return gdal_dem_band(dataset, band)

#
# GDAL vector data helpers
#
# Selected OGR geometry types, see OGRwkbGeometryType
wkbUnknown = ogr.wkbUnknown
wkbPoint = ogr.wkbPoint
wkbLineString = ogr.wkbLineString
wkbPolygon = ogr.wkbPolygon
wkbLinearRing = ogr.wkbLinearRing
wkbPoint25D = ogr.wkbPoint25D
wkbLineString25D = ogr.wkbLineString25D
wkbPolygon25D = ogr.wkbPolygon25D

# Selected OGR field types, see OGRFieldType
OFTInteger = ogr.OFTInteger
OFTReal = ogr.OFTReal
OFTString = ogr.OFTString

class gdal_vect_layer(gdal_dataset):
    """"GDAL vector layer representation"""
    def __init__(self, dataset, i=None):
        super(gdal_vect_layer, self).__init__(dataset)
        if i is not None:
            self.layer = self.dataset.dataset.GetLayer(i)

    @staticmethod
    def create(dataset, name, srs=None, geom_type=wkbUnknown, options=[]):
        """Create new layer"""
        layer = dataset.dataset.CreateLayer(name, srs=srs, geom_type=geom_type, options=options)
        if layer is None:
            return None
        ret = gdal_vect_layer(dataset.dataset)
        ret.layer = layer
        return ret

    def get_name(self):
        return self.layer.GetName() if self.layer is not None else None

    def create_field(self, name, ftype=OFTReal):
        fd = ogr.FieldDefn(name, ftype)
        self.layer.CreateField(fd)
        return self.layer.FindFieldIndex(name, True)

    def create_feature_geometry(self, geom_type=None):
        feat = ogr.Feature(feature_def=self.layer.GetLayerDefn())
        if feat is None:
            return None
        if geom_type is None:
            geom_type = self.layer.GetGeomType()
        geom = ogr.Geometry(geom_type)
        if geom is None:
            return None
        return gdal_feature_geometry(self.layer, feat, geom)

class gdal_feature_geometry:
    """"Common ogr.Feature ogr.Geometry representation"""
    def __init__(self, layer, feat, geom):
        self.layer = layer
        self.feat = feat
        self.geom = geom

    def get_id(self):
        return self.feat.GetFID()

    def set_field(self, name, value):
        return self.feat.SetField(name, value)

    def add_point(self, *coord):
        self.geom.AddPoint(*coord)

    def add_geometry(self, geom):
        self.geom.AddGeometry(geom.geom)

    def get_style_string(self):
        return self.feat.GetStyleString()

    def set_style_string(self, string):
        self.feat.SetStyleString(string)

    def create(self):
        # If this is a 'ring' geometry, close it
        self.geom.CloseRings()
        self.feat.SetGeometry(self.geom)
        self.layer.CreateFeature(self.feat)

def vect_create(filename, drv_name=None, xsize=0, ysize=0, bands=0, options=[]):
    """Open or create a vector file for writing"""
    # drv_name can be empty/None, list/tuple or string
    allowed_drivers, drv_name = ([], None) if not drv_name else \
            (drv_name, drv_name[0]) if isinstance(drv_name, (list, tuple)) else \
            ([drv_name], drv_name)
    # Open existing file or create new one
    dataset = gdal.OpenEx(filename, gdal.OF_VECTOR | gdal.GA_Update,
            allowed_drivers=allowed_drivers, open_options=options)
    if dataset is None:
        if drv_name is None:
            drv_name = GetOutputDriverFor(filename)
        drv = gdal.GetDriverByName(drv_name)
        dataset = drv.Create(filename, xsize, ysize, bands, options=options)

    if dataset is None:
        return None

    return gdal_dataset(dataset)


#
# From gdal/python/scripts (ogrmerge.py)
#
def DoesDriverHandleExtension(drv, ext):
    exts = drv.GetMetadataItem(gdal.DMD_EXTENSIONS)
    return exts is not None and exts.lower().find(ext.lower()) >= 0

def GetExtension(filename):
    if filename.lower().endswith('.shp.zip'):
        return 'shp.zip'
    ext = os.path.splitext(filename)[1]
    if ext.startswith('.'):
        ext = ext[1:]
    return ext

def GetOutputDriversFor(filename):
    drv_list = []
    ext = GetExtension(filename)
    if ext.lower() == 'vrt':
        return ['VRT']
    for i in range(gdal.GetDriverCount()):
        drv = gdal.GetDriver(i)
        if (drv.GetMetadataItem(gdal.DCAP_CREATE) is not None or
            drv.GetMetadataItem(gdal.DCAP_CREATECOPY) is not None) and \
           drv.GetMetadataItem(gdal.DCAP_VECTOR) is not None:
            if ext and DoesDriverHandleExtension(drv, ext):
                drv_list.append(drv.ShortName)
            else:
                prefix = drv.GetMetadataItem(gdal.DMD_CONNECTION_PREFIX)
                if prefix is not None and filename.lower().startswith(prefix.lower()):
                    drv_list.append(drv.ShortName)

    return drv_list

def GetOutputDriverFor(filename):
    drv_list = GetOutputDriversFor(filename)
    ext = GetExtension(filename)
    if not drv_list:
        if not ext:
            return 'ESRI Shapefile'
        else:
            raise Exception("Cannot guess driver for %s" % filename)
    elif len(drv_list) > 1:
        print("Several drivers matching %s extension. Using %s" % (ext if ext else '', drv_list[0]))
    return drv_list[0]
