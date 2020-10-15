"""GDAL/pyproj backend and utilities"""
import sys
import numpy
import gdal
import pyproj


def write_arr(arr, x_y, val):
    """Put data to multiple indices in array"""
    # Avoid numpy "Advanced Indexing"
    arr[tuple(numpy.moveaxis(x_y, -1, 0))] = val

def read_arr(arr, x_y):
    """Get multiple indices from array"""
    # Force numpy "Basic Indexing", note that '[x_y]' will trigger "Advanced Indexing"
    return arr[tuple(numpy.moveaxis(x_y, -1, 0))]

#
# GDAL helpers
#
def _get_dtype(band):
    """NumPy dtype from GDAL band DataType
    See gdal_array.GDALTypeCodeToNumericTypeCode(band.DataType)"""
    if band.DataType == gdal.GDT_Byte:
        return numpy.uint8
    return gdal.array_modes[band.DataType]

class gdal_dem_band:
    """"GDAL DEM band representation"""
    dem_buf = None

    def __init__(self, dataset, i=1):
        self.dataset = dataset
        self.band = dataset.GetRasterBand(i)
        # Cached parameters
        self._update_xform()
        self.scale = self.band.GetScale()
        if self.scale is None:
            self.scale = 1
        self.nodata_val = self.band.GetNoDataValue()
        if self.nodata_val is not None:
            self.nodata_val = int(self.nodata_val)

    def _update_xform(self, offset=None):
        """Build the transformation matrix from GetGeoTransform() with optional offset"""
        geo_xform = self.band.GetDataset().GetGeoTransform()
        self.xform = numpy.array( geo_xform ).reshape(2,3)
        # Offset the transformation matrix
        if offset is not None:
            txform = numpy.identity(3)
            txform[1:3, 0] = offset
            self.xform = numpy.matmul(self.xform, txform)

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
        self._update_xform(numpy.array([xstart, ystart]))
        return True

    def in_bounds(self, x_y):
        """Check if a coordinate is inside the DEM array"""
        return numpy.logical_and(
            (x_y >= 0).all(-1),
            (x_y < self.dem_buf.shape).all(-1))

    def get_elevation(self, x_y):
        """Retrieve elevation(s)"""
        alt = read_arr(self.dem_buf, x_y)
        # Replace the GDAL "NoDataValue" with NaN
        if self.nodata_val is not None:
            alt[alt == self.nodata_val] = numpy.nan
        return alt

    def xy2lonlat(self, x_y):
        """Convert raster (x,y) coordinate(s) to lon/lan (east,north)"""
        if True:
            # Adjust point to the center of the raster pixel
            # (otherwise, the coordinates will be at the top-left (NW) corner)
            x_y = x_y + .5
        # Add ones in front of the coordinates to handle translation
        # Note that GetGeoTransform() returns translation components at index 0
        ones = numpy.broadcast_to([1], [*x_y.shape[:-1], 1])
        x_y = numpy.concatenate((ones, x_y), axis=-1)

        # This is matmul() but x_y is always treated as a set of one-dimentional vectors
        x_y = x_y[...,numpy.newaxis,:]
        return (self.xform * x_y).sum(-1)

    def xy2lonlatalt(self, x_y):
        """Convert raster (x,y) coordinate(s) to lon/lan/alt (east,north,alt)"""
        lon_lat = self.xy2lonlat(x_y)
        alt = self.get_elevation(x_y)[...,numpy.newaxis]
        return numpy.concatenate((lon_lat, alt), axis=-1)

#
# Distance calculator
#
class geod_distance:
    def __init__(self, gdal_dem_band):
        self.dem_band = gdal_dem_band
        self.geod = pyproj.Geod(ellps='WGS84')

    def get_distance(self, xy0, xy1):
        """Calculate distance between two points"""
        # Precise method by calling pyproj.Geod.inv() between points
        lonlatalt = self.dem_band.xy2lonlatalt(numpy.stack((xy0, xy1)))
        _, _, dist = self.geod.inv(*lonlatalt[:,:2].flatten())
        # Adjust distance with the altitude displacement
        disp = lonlatalt[1,2] - lonlatalt[0,2]
        return numpy.sqrt(dist*dist + disp*disp)

def dem_open(filename, band=1):
    """Open a raster DEM file for reading"""
    dataset = gdal.Open(filename)
    if dataset is None:
        return None

    return gdal_dem_band(dataset, band)
