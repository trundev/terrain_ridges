"""Generate terrain ridges/valeys"""
import sys
import numpy
import gdal

# Neighbor indices:
#   -4 -3 -2
#   -1 <0> 1
#    2  3  4
VALID_NEIGHBORS = (-4, -3, -2, -1, 1, 2, 3, 4)
NEIGHBOR_SELF = 0
NEIGHBOR_INVALID = -100
NEIGHBOR_IDX_DTYPE = numpy.int8

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

def write_arr(arr, x_y, val):
    """Put data to multiple indices in array"""
    # Avoid numpy "Advanced Indexing"
    arr[tuple(x_y.T)] = val

def read_arr(arr, x_y):
    """Get multiple indices from array"""
    # Force numpy "Basic Indexing", note that '[x_y]' will trigger "Advanced Indexing"
    return arr[tuple(x_y.T)]

#
# GDAL helpers
#
def get_dtype(band):
    """NumPy dtype from GDAL band DataType
    See gdal_array.GDALTypeCodeToNumericTypeCode(band.DataType)"""
    if band.DataType == gdal.GDT_Byte:
        return numpy.uint8
    return gdal.array_modes[band.DataType]

class gdal_dem_band:
    """"GDAL DEM band representation"""
    dem_buf = None

    def __init__(self, band):
        self.band = band
        # Cached parameters
        self._update_xform()
        self.scale = band.GetScale()
        if self.scale is None:
            self.scale = 1
        self.nodata_val = band.GetNoDataValue()
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
        self.dem_buf = numpy.frombuffer(buf, dtype=get_dtype(self.band))

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
class gdal_distance:
    def __init__(self, gdal_dem_band):
        self.dem_band = gdal_dem_band

    def get_distance(self, xy0, xy1):
        """Calculate distance between two points"""
        #TODO: Use real 'pyproj' distance measurement, including the altitude displacement
        #HACK: Assume both X and Y steps are 1m
        return numpy.sqrt(((xy0 - xy1)**2).sum())

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

def generate_ridges(dem_band, valleys=False):
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
    write_arr(prev_arr, seed_xy, NEIGHBOR_SELF)

    #
    # Tentative point list (coord and distance)
    # Initially contains the start point only
    #
    tentative = [(seed_xy, 0)]

    #
    # End-points of generated lines (coord and distance)
    #
    result_lines = []

    distance = gdal_distance(dem_band)
    while tentative:
        x_y, dist = tentative.pop()
        #print('    Processing point %s dist %d alt %s'%(x_y, dist, dem_band.get_elevation(seed_xy)))
        end_of_line = True
        for t_xy, n_idx in valid_neighbors(dem_band, x_y):
            if read_arr(prev_arr, t_xy) == NEIGHBOR_INVALID:
                end_of_line = False
                # Keep the inverted neighbor index to later track this back
                write_arr(prev_arr, t_xy, neighbor_inv(n_idx))
                alt = dem_band.get_elevation(t_xy)
                # Insert the point in 'tentative' by keeping it sorted by altitude
                def cmp_fn(val, *args):
                    check_xy, dist = val
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
        #TODO:

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
    dataset = gdal.Open(src_filename)
    if dataset is None:
        return print_help('Unable to open "%s"'%src_filename)
    print(dataset.GetMetadata())

    # Generate ridges/valleys
    dem_band = gdal_dem_band(dataset.GetRasterBand(1))
    dem_band.load()
    res = generate_ridges(dem_band, valleys)
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
