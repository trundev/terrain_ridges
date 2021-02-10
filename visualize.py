"""3D DEM terrain visualization

Requires: numpy matplotlib gdal pyproj
"""
import sys
import numpy
import matplotlib.pyplot as pyplot
import matplotlib.widgets as widgets
import gdal_utils
import ridges

NODATA_ALT = 1000.

TERRAIN_FMT = dict(cmap='terrain', color='greenyellow', edgecolor='none', alpha=.5, label='_Terrain')
NODATA_FMT = dict(color='red', marker='o', label='NoData')
PENDINGS_FMT = dict(color='yellow', marker='o', label='Pendings')
SEEDS_FMT = dict(color='orange', marker='x', label='Seeds')
STOPS_FMT = dict(color='orange', marker='+', label='Stops')
LEAFS_FMT = dict(color='lightgreen', marker='.', label='Leafs', visible=False)
DIR_ARR_FMT = dict(label='dir_arr', cmap='viridis', alpha=.5, visible=False)
DIR_ARR_CMAP_IDXS = 4
NODES_FMT = dict(color='brown', marker='o', label='Nodes', alpha=.5)
BRIDGES_FMT = dict(label='bridge_lines', cmap='viridis', alpha=.5)

QUIVER_2D_FMT = dict(angles='xy', scale_units='xy', scale=1.2)
BRIDGES_2D_FMT = QUIVER_2D_FMT.copy(); BRIDGES_2D_FMT['scale']=1.1
SHOW_MIN_RANK = 2

USE_2D = True

DIR_STYLE_LABELS = ['dir style 0', 'dir style 1']

#
# matplotlib Widget rectangles
#
AX_MARGIN = .02
AX_BTN_WIDTH = .15
AX_BTN_HEIGHT = .3

def deflate_rect(rect, hor_margin=AX_MARGIN, vert_margin=AX_MARGIN):
    """Deflate matplotlib rectangle [left, bottom, right, top]"""
    rect[0] += hor_margin
    rect[1] += vert_margin
    rect[2] -= 2 * hor_margin
    rect[3] -= 2 * vert_margin
    return rect

def set_aspect(ax, lonlatalt):
    """Try to keep the lon/lat aspect equal"""
    min_max = numpy.nanmin(lonlatalt, axis=(0,1)), numpy.nanmax(lonlatalt, axis=(0,1))
    middle = (min_max[0] + min_max[1]) / 2
    lonlan_max = (min_max[1] - min_max[0])[:2].max() / 3
    # Reduce the altitude exaggeration
    alt_max = (min_max[1] - min_max[0])[2]
    # Set the matplotlib axis limits
    ax.set_xlim(middle[0] - lonlan_max, middle[0] + lonlan_max)
    ax.set_ylim(middle[1] - lonlan_max, middle[1] + lonlan_max)
    ax.set_zlim(middle[2] - alt_max, middle[2] + alt_max)

class collections:
    """Maintain matplotlib collections"""
    colls = {}
    dir_style = 0

    def __init__(self, ax, do_redraw, *do_redraw_args):
        self.ax = ax
        self.do_redraw = do_redraw
        self.do_redraw_args = do_redraw_args

    def get_collections(self):
        """Obtain a list of collections"""
        return self.colls.values()

    def replace(self, key, new):
        """Replace collection in a dictionary by keeping its visibility"""
        old = self.colls.get(key)
        if old is not None:
            visible = old.get_visible()
            old.remove()
            new.set_visible(visible)
        self.colls[key] = new

    def redraw(self):
        return self.do_redraw(self, *self.do_redraw_args)

    def on_showhide(self, label):
        if label in DIR_STYLE_LABELS:
            idx = DIR_STYLE_LABELS.index(label)
            self.dir_style ^= 1<<idx
            self.redraw()
        else:
            # Show/hide collection
            for coll in self.get_collections():
                if coll.get_label() == label:
                    coll.set_visible(not coll.get_visible())
                    break
        self.ax.figure.canvas.draw_idle()

def do_redraw(colls, dem_band, dir_arr):
    """Generate matplotlib collections"""
    # Convert all points in the DEM buffer to 2D array of lon-lat-alt values (3D array)
    shape = dem_band.get_elevation(True).shape
    mgrid_xy = numpy.moveaxis(numpy.mgrid[:shape[0],:shape[1]], 0, -1)
    lonlatalt = dem_band.xy2lonlatalt(mgrid_xy)

    def nodata_markers(colls):
        # When DEM contains no-data altitudes (NaN), plot_surface() may not render anything
        nodata_mask = numpy.isnan(lonlatalt[...,2])
        if nodata_mask.any():
            lonlatalt[...,2][nodata_mask] = NODATA_ALT

        if not USE_2D:
            colls.replace('terrain', colls.ax.
                    plot_surface(*lonlatalt.T, **TERRAIN_FMT))

        # NoData markers at 2/3-rds of altitude
        nodata_pts = lonlatalt[nodata_mask]
        if USE_2D:
            nodata_pts = nodata_pts[...,:2]
        else:
            nodata_pts[...,2] = (2 * numpy.nanmax(lonlatalt[...,2]) + numpy.nanmin(lonlatalt[...,2])) / 3
        colls.replace('nodata', colls.ax.
                scatter(*nodata_pts.T, **NODATA_FMT))
    nodata_markers(colls)

    if dir_arr is None:
        return

    #
    # Visualize dir_arr
    #
    pts = lonlatalt
    if USE_2D:
        pts = pts[...,:2]

    def markers_by_value(colls):
        # Markers at "pending"
        mask = dir_arr == ridges.NEIGHBOR_PENDING
        print('Located %d "pending" pixels'%(numpy.count_nonzero(mask)))
        colls.replace('pendings', colls.ax.
                scatter(*pts[mask].T, **PENDINGS_FMT))

        # Markers at "seed" and "stop"
        mask = dir_arr == ridges.NEIGHBOR_SEED
        print('Located %d "seed" pixels'%(numpy.count_nonzero(mask)))
        colls.replace('seeds', colls.ax.
                scatter(*pts[mask].T, **SEEDS_FMT))
        mask = dir_arr == ridges.NEIGHBOR_STOP
        print('Located %d "stop" pixels'%(numpy.count_nonzero(mask)))
        colls.replace('stops', colls.ax.
                scatter(*pts[mask].T, **STOPS_FMT))
    markers_by_value(colls)

    #
    # Graph node analysys
    #
    # Helper array (sort of vector field)
    mgrid_n_xy = ridges.neighbor_xy_safe(mgrid_xy, dir_arr)

    # Count the number of neighbors pointing to each pixel
    n_num = numpy.zeros(dir_arr.shape, dtype=int)
    for d in ridges.VALID_NEIGHBOR_DIRS:
        n_xy = mgrid_n_xy[dir_arr == d]
        n = numpy.zeros_like(n_num)
        gdal_utils.write_arr(n, n_xy, 1)
        n_num += n
    # Put -1 at invalid nodes, except the "real" seeds (distinguish from the "leafs")
    n_num[ridges.neighbor_is_invalid(dir_arr) & (n_num == 0)] = -1
    print('Located %d "real-seed" pixels'%(numpy.count_nonzero(
            ridges.neighbor_is_invalid(dir_arr) & (n_num > 0))))

    def node_markers(colls):
        # Markers at "leafs" (pixels w/o neighbor)
        mask = n_num == 0
        print('Located %d "leaf" pixels'%(numpy.count_nonzero(mask)))
        colls.replace('leafs', colls.ax.
                scatter(*pts[mask].T, **LEAFS_FMT))

        # Add points where more than 2 neighbors pointing
        mask = n_num > 1
        print('Located %d "node" pixels (%d at "seed"), max %d forks'%(
                numpy.count_nonzero(mask), numpy.count_nonzero(mask & ridges.neighbor_is_invalid(dir_arr)),
                n_num.max()))
        colls.replace('nodes', colls.ax.
                scatter(*pts[mask].T, s=n_num[mask]**2, **NODES_FMT))
    node_markers(colls)

    # Extract graph node bridges (edges)
    def get_bridge_lines(dist_arr):
        # Extract "leafs", "nodes" and "real-seeds"
        seed_mask = (n_num > 0) & ridges.neighbor_is_invalid(dir_arr)
        mask = (n_num == 0) | (n_num > 1) | seed_mask
        print('Located %d "node-bridges"'%(numpy.count_nonzero(mask)))
        x_y = mgrid_xy[mask]
        bridge_lines = numpy.empty(x_y.shape[0], dtype=[
                ('x_y', (numpy.int32, (2,))),
                ('dist', float),
                ('next', numpy.int32),
                ('num', numpy.int32),
                ])
        bridge_lines['x_y'] = x_y
        bridge_lines['dist'] = dist_arr[mask]
        bridge_lines['next'] = -2   # Invalid index
        bridge_lines['num'] = 0
        # Map x_y to the 'next' bridge indices, reserved values:
        # -1 -- bridge, -2 -- bare-seed or invalid
        bridge_grid = numpy.full(dir_arr.shape, -2, bridge_lines['next'].dtype)
        bridge_grid[(n_num == 1) & ~seed_mask] = -1
        bridge_grid[mask] = numpy.mgrid[:numpy.count_nonzero(mask)]
        print('Creating %d bridges from %d intermediate points, %d real-seed, %d bare-seed + invalid'%(
                bridge_lines.size, numpy.count_nonzero(bridge_grid == -1),
                numpy.count_nonzero(seed_mask), numpy.count_nonzero(bridge_grid == -2)))

        # Skip the dummy seed-lines, but keep the 'next' value -1
        bridge_lines['next'][bridge_grid[seed_mask]] = -1
        res_mask = bridge_lines['next'] != -1
        x_y = x_y[res_mask]
        while x_y.size:
            bridge_lines['num'][res_mask] += 1
            x_y = gdal_utils.read_arr(mgrid_n_xy, x_y)
            bridge = gdal_utils.read_arr(bridge_grid, x_y)
            mask = bridge == -1     # intermediate bridge point

            # Get the completed lines, shrink result masks
            done_mask = res_mask.copy()
            done_mask[res_mask] = ~mask
            res_mask[res_mask] = mask
            # Update 'next' where processing in completed
            if done_mask.any():
                assert (bridge_lines['next'][done_mask] == -2).all()
                bridge_lines['next'][done_mask] = bridge[~mask]

            x_y = x_y[mask]
            dist = gdal_utils.read_arr(dist_arr, x_y)
            assert (dist > 0).all()
            bridge_lines['dist'][res_mask] += dist

        assert (bridge_lines['next'] > -2).all(), 'Unassigned bridges left'
        argsort = numpy.argsort(bridge_lines['dist'])
        bridge_lines = numpy.take(bridge_lines, argsort)
        # Update 'next' pointers by using swapped argsort
        swap_argsort = numpy.empty_like(argsort)
        swap_argsort[argsort] = numpy.mgrid[:argsort.size]
        # Keep the negative 'next' values (-1 is a dummy seed-line)
        mask = bridge_lines['next'] >= 0
        bridge_lines['next'][mask] = swap_argsort[bridge_lines['next'][mask]]
        print('Created bridges along %d points, min/max len %f/%f'%(
                bridge_lines['num'].sum(),
                # Skip the zero-sized dummy seed-lines at front
                bridge_lines['dist'][numpy.count_nonzero(seed_mask)],
                bridge_lines['dist'][-1]))
        return bridge_lines
    # Calculate distances between points (dist_arr)
    distance = gdal_utils.geod_distance(dem_band)
    dist_arr = distance.get_distance(mgrid_xy, mgrid_n_xy)
    # Extract node bridges
    bridge_lines = get_bridge_lines(dist_arr)

    # Isolate graph branches by iteratively trim the leaf graph-bridges until all pixels are processed
    def rank_branches(dir_arr):
        """Assign a branch-rank to each pixel"""
        rank_arr = numpy.zeros(dir_arr.shape, dtype=int)
        rank_arr[ridges.neighbor_is_invalid(dir_arr)] = -1
        cur_rank = 0
        unassigned_cnt = numpy.count_nonzero(rank_arr == 0)
        while unassigned_cnt > 0:
            cur_rank += 1
            # Count the number of neighbors pointing to each pixel (within this rank)
            n_num = numpy.zeros(dir_arr.shape, dtype=int)
            for d in ridges.VALID_NEIGHBOR_DIRS:
                n_xy = mgrid_n_xy[(dir_arr == d) & (rank_arr == 0)]
                n = numpy.zeros_like(n_num)
                gdal_utils.write_arr(n, n_xy, 1)
                n_num += n
            # Put -1 at invalid nodes, except the "real" seeds (to distinguish from the "leaves")
            n_num[(rank_arr != 0) & (n_num == 0)] = -1

            # Update 'rank_arr' along "leaf" branches
            x_y = numpy.array(numpy.nonzero(n_num == 0)).T
            print('Rank %d: Detected %d leaf branches...'%(cur_rank, x_y.shape[0]))
            # Mask of all bridge intermediate pixels
            bridge_mask = (n_num == 1) & ~ridges.neighbor_is_invalid(dir_arr)
            while x_y.size:
                gdal_utils.write_arr(rank_arr, x_y, cur_rank)
                x_y = gdal_utils.read_arr(mgrid_n_xy, x_y)
                mask = gdal_utils.read_arr(bridge_mask, x_y)
                x_y = x_y[mask]

            start_cnt = unassigned_cnt
            unassigned_cnt = numpy.count_nonzero(rank_arr == 0)
            print('  Assigned %d pixels'%(start_cnt - unassigned_cnt))
        return rank_arr
    rank_arr = rank_branches(dir_arr)

    #
    # Vectors along node-bridges
    #
    def bridge_markers(colls):
        """Direct vectors between nodes from the dir_arr graph"""
        valid_mask = bridge_lines['next'] >= 0
        bridges = bridge_lines[valid_mask]
        colors = numpy.zeros(bridges.shape, dtype=int)
        cidx = 0
        if colls.dir_style == 0:
            # Colors based on the bridge-rank
            colors = gdal_utils.read_arr(rank_arr, bridges['x_y'])
            # Hide low-rank bridges
            mask = colors >= SHOW_MIN_RANK
            bridges = bridges[mask]
            colors = colors[mask]
        elif colls.dir_style == 1:
            # Colors expand staring at "seed" (bridge_lines['next'] == -1)
            mask = ~valid_mask
            while mask.any():
                mask = mask[bridge_lines['next']] & valid_mask
                # Expand
                colors[mask[valid_mask]] = cidx % DIR_ARR_CMAP_IDXS
                cidx += 1
        elif colls.dir_style == 2:
            # Colors based on elevation
            if True:
                # Hide leaf bridges
                mask = numpy.zeros(bridge_lines.shape, dtype=bool)
                mask[bridges['next']] = True
                bridges = bridge_lines[mask & valid_mask]
            colors = gdal_utils.read_arr(lonlatalt[...,2], bridges['x_y'])
        else:
            # Colors based on the bridge lengths
            colors = bridges['dist']

        print('bridge_lines sorted in %d steps, %d bridges'%(cidx, colors.size))

        start_pts = gdal_utils.read_arr(pts, bridges['x_y'])
        end_pts = gdal_utils.read_arr(pts, bridge_lines[bridges['next']]['x_y'])

        if USE_2D:
            fmt = {**BRIDGES_FMT, **BRIDGES_2D_FMT}
        else:
            fmt = BRIDGES_FMT

        colls.replace('bridge_lines', colls.ax.
                quiver(*start_pts.T, *(end_pts - start_pts).T, colors, **fmt))
    bridge_markers(colls)

    #
    # Vectors along dir_arr
    #
    def n_dir_markers(colls):
        # Select how to color dir-vectors
        show_mask = ~ridges.neighbor_is_invalid(dir_arr)
        colors = numpy.zeros(dir_arr.shape, dtype=int)
        cidx = 0
        if colls.dir_style == 0:
            # Colors based on the bridge-rank
            colors = rank_arr
            # Hide low-rank pixels
            show_mask[rank_arr < SHOW_MIN_RANK] = False
        elif colls.dir_style == 1:
            # Colors expand staring at "seed" and "stop"
            show_mask[...] = False
            mask = (dir_arr == ridges.NEIGHBOR_SEED) | (dir_arr == ridges.NEIGHBOR_STOP)
            mask = gdal_utils.read_arr(mask, mgrid_n_xy) ^ mask
            while mask.any():
                colors[mask] = cidx % DIR_ARR_CMAP_IDXS
                show_mask[mask] = True
                # Expand
                mask = gdal_utils.read_arr(mask, mgrid_n_xy)
                cidx += 1
        elif colls.dir_style == 2:
            # Colors based on elevation
            colors = lonlatalt[...,2]
            if True:
                # Hide leaf bridges
                x_y = numpy.array(numpy.nonzero(n_num == 0)).T
                # Mask of all bridge intermediate pixels
                bridge_mask = (n_num == 1) & ~ridges.neighbor_is_invalid(dir_arr)
                while x_y.size:
                    gdal_utils.write_arr(show_mask, x_y, False)
                    x_y = gdal_utils.read_arr(mgrid_n_xy, x_y)
                    mask = gdal_utils.read_arr(bridge_mask, x_y)
                    x_y = x_y[mask]
        else:
            # Colors based on the bridge lengths
            x_y = bridge_lines['x_y']
            dists = bridge_lines['dist']
            gdal_utils.write_arr(colors, x_y, dists)
            # Mask of all bridge intermediate pixels
            bridge_mask = (n_num == 1) & ~ridges.neighbor_is_invalid(dir_arr)
            while x_y.size:
                gdal_utils.write_arr(colors, x_y, dists)
                x_y = gdal_utils.read_arr(mgrid_n_xy, x_y)
                mask = gdal_utils.read_arr(bridge_mask, x_y)
                x_y = x_y[mask]
                dists = dists[mask]
                cidx += 1
            assert (colors[show_mask] > 0).all(), 'Unassigned colors left'

        colors = colors[show_mask]
        print('dir_arr sorted in %d steps, %d pixels'%(cidx, colors.size))

        # Invalid and seed points, points to them-self
        vecs = dem_band.xy2lonlatalt(mgrid_n_xy) - lonlatalt
        if USE_2D:
            vecs = vecs[...,:2]
        else:
            # Suppress altitude as having problems with quiver() and different units
            vecs[...,2] = 0

        if USE_2D:
            fmt = {**DIR_ARR_FMT, **QUIVER_2D_FMT}
        else:
            fmt = DIR_ARR_FMT
            #TODO: Colors do not work in 3D-mode
            colors = None

        # Show colored n_dir-s
        colls.replace('dir_arr', colls.ax.
                quiver(*pts[show_mask].T, *vecs[show_mask].T, colors, **fmt))
    n_dir_markers(colls)

def show_plot(dem_band, dir_arr, title=None):
    """Display matplotlib window"""
    fig = pyplot.figure()
    ax = fig.add_axes(deflate_rect([0, 0, 1 - AX_BTN_WIDTH, 1], 2 * AX_MARGIN, AX_MARGIN),
            projection=None if USE_2D else '3d')
    if title is not None:
        ax.set_title(title)

    # Convert all points in the DEM buffer to 2D array of lon-lat-alt values (3D array)
    shape = dem_band.get_elevation(True).shape
    mgrid_xy = numpy.moveaxis(numpy.mgrid[:shape[0],:shape[1]], 0, -1)
    lonlatalt = dem_band.xy2lonlatalt(mgrid_xy)

    if USE_2D:
        ax.set_aspect('equal')
    else:
        set_aspect(ax, lonlatalt)

    colls = collections(ax, do_redraw, dem_band, dir_arr)
    colls.redraw()

    # Check boxes to show/hide individual elements
    rax = fig.add_axes(deflate_rect([1 - AX_BTN_WIDTH, 1 - AX_BTN_HEIGHT, AX_BTN_WIDTH, AX_BTN_HEIGHT],
            AX_MARGIN / 2, AX_MARGIN / 2))
    labels = [coll.get_label() for coll in colls.get_collections()]
    visibility = [coll.get_visible() for coll in colls.get_collections()]
    #TODO: Colors do not work in 3D-mode
    if USE_2D:
        labels += DIR_STYLE_LABELS
        visibility += [False] * len(DIR_STYLE_LABELS)
    check = widgets.CheckButtons(rax, labels, visibility)
    check.on_clicked(colls.on_showhide)

    ax.legend()
    pyplot.show()
    return 0

def main(argv):
    """Main entry"""
    dem_filename = None
    dir_arr = None
    while argv:
        if argv[0][0] == '-':
            if argv[0] == '-h':
                return print_help()
            elif argv[0] == '--dir_arr':
                argv = argv[1:]
                dir_arr = argv[0]
            else:
                return print_help('Unsupported option "%s"'%argv[0])
        else:
            if dem_filename is None:
                dem_filename = argv[0]
            else:
                return print_help('Unexpected argument "%s"'%argv[0])

        argv = argv[1:]

    if dem_filename is None:
        return print_help('Missing file-name')

    # Load DEM
    dem_band = gdal_utils.dem_open(dem_filename)
    if dem_band is None:
        return print_help('Unable to open "%s"'%dem_filename)
    dem_band.load()

    # Load 'dir_arr'
    if dir_arr is not None:
        dir_arr = numpy.load(dir_arr)
        if (dir_arr.shape != dem_band.get_elevation(True).shape):
            return print_help('DEM and "dir_arr" shape mismatch"%s"'%dem_filename)

    numpy.set_printoptions(suppress=True, precision=6)
    print('%s %d x %d:'%(dem_filename, *dem_band.dem_buf.shape))
    print(dem_band.xform)

    # Visualize terrain/ridges
    res = show_plot(dem_band, dir_arr, dem_filename)
    return res

def print_help(err_msg=None):
    if err_msg:
        print('Error:', err_msg, file=sys.stderr)
    print('Usage:', sys.argv[0], '[<options>] <dem_filename>')
    print('\tOptions:')
    print('\t-h\t- This screen')
    print('\t--dir_arr <dir_arr-npy>\t- "dir_arr" (n_dir) npy file')
    return 0 if err_msg is None else 255

if __name__ == '__main__':
    ret = main(sys.argv[1:])
    if ret:
        exit(ret)
