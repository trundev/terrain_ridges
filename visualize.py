"""3D DEM terrain visualization

Requires: numpy matplotlib gdal pyproj
"""
import sys
import numpy
import matplotlib.pyplot as pyplot
import gdal_utils

TERRAIN_FMT = dict(cmap='terrain', color='greenyellow', edgecolor='none', alpha=.8, label='_Terrain')
NODATA_ALT = 0.

def show_plot(dem_band, title=None):
    """Display matplotlib window"""
    ax = pyplot.axes(projection='3d')
    if title is not None:
        ax.set_title(title)

    # Convert all points in the DEM buffer to 2D array of lon-lat-alt values (3D array)
    x = numpy.broadcast_to(numpy.arange(dem_band.dem_buf.shape[0])[:, numpy.newaxis], dem_band.dem_buf.shape)
    y = numpy.broadcast_to(numpy.arange(dem_band.dem_buf.shape[1])[numpy.newaxis, :], dem_band.dem_buf.shape)
    lonlatalt = dem_band.xy2lonlatalt(numpy.stack((x,y), -1))

    # When DEM contains no-data altitudes (NaN), plot_surface() may not render anything
    nodata_mask = numpy.isnan(lonlatalt[...,2])
    if nodata_mask.any():
        lonlatalt[...,2][nodata_mask] = NODATA_ALT

    ax.plot_surface(*lonlatalt.T, **TERRAIN_FMT)

    # Dummy points to reduce altitude exaggeration
    alt_range = lonlatalt[...,2].max() - lonlatalt[...,2].min()
    ax.plot(*lonlatalt[0,0,:2], lonlatalt[...,2].min() - alt_range)
    ax.plot(*lonlatalt[0,0,:2], lonlatalt[...,2].max() + alt_range)

    pyplot.show()
    return 0

def main(argv):
    """Main entry"""
    dem_filename = None
    while argv:
        if argv[0][0] == '-':
            if argv[0] == '-h':
                return print_help()
            else:
                return print_help('Unsupported option "%s"'%argv[0])
        else:
            if dem_filename is None:
                dem_filename = argv[0]
#            else:
#                return print_help('Unexpected argument "%s"'%argv[0])

        argv = argv[1:]

    if dem_filename is None:
        return print_help('Missing file-name')

    # Load DEM
    dem_band = gdal_utils.dem_open(dem_filename)
    if dem_band is None:
        return print_help('Unable to open "%s"'%dem_filename)
    dem_band.load()

    numpy.set_printoptions(suppress=True, precision=6)
    print('%s %d x %d:'%(dem_filename, *dem_band.dem_buf.shape))
    print(dem_band.xform)

    # Visualize terrain/ridges
    res = show_plot(dem_band, dem_filename)
    return res

def print_help(err_msg=None):
    if err_msg:
        print('Error:', err_msg, file=sys.stderr)
    print('Usage:', sys.argv[0], '[<options>] <dem_filename>')
    print('\tOptions:')
    print('\t-h\t- This screen')
    return 0 if err_msg is None else 255

if __name__ == '__main__':
    ret = main(sys.argv[1:])
    if ret:
        exit(ret)
