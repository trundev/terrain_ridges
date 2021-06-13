"""Generate terrain ridges/valleys"""
import sys
import time
import numpy
import gdal_utils

# Neighbor directions:
#   0 1 2
#   3<4>5
#   6 7 8
VALID_NEIGHBOR_DIRS = (0, 1, 2, 3, 5, 6, 7, 8)
NEIGHBOR_SELF = 4
NEIGHBOR_LAST_VALID = 8
NEIGHBOR_PENDING = 9
NEIGHBOR_SEED = 10
NEIGHBOR_STOP = 11
NEIGHBOR_INVALID = 12
NEIGHBOR_BOUNDARY = 13
NEIGHBOR_DIR_DTYPE = numpy.int8
VALID_NEIGHBOR_DIRS = numpy.array(VALID_NEIGHBOR_DIRS, dtype=NEIGHBOR_DIR_DTYPE)

# Keep the seed away from the edges
SEED_INFLATE = 1
# Select distance caclulation method:
#   0 - Real geodetic distance (geod_distance): Use pyproj.Geod.inv()
#   1 - Transverse Mercator (tm_distance): Use TM origin at the center of raster data
#   2 - Draft (draft_distance): Use pre-calculated pixel size by tm_distance for all pixels
DISTANCE_METHOD = 0

def VECTOR_LAYER_NAME(valleys): return 'valleys' if valleys else 'ridges'
def VECTOR_FEATURE_STYLE(valleys): return 'PEN(c:#0000FF,w:2px)' if valleys else 'PEN(c:#FF0000,w:2px)'
# Value for the OSM "natural" keys, to allow conversion to .osm
# by ogr2osm.py or JOSM (opendata plugin)
#   https://wiki.openstreetmap.org/wiki/Tag:natural%3Dridge
#   https://wiki.openstreetmap.org/wiki/Tag:natural%3Dvalley
def FEATURE_OSM_NATURAL(valleys): return 'valley' if valleys else 'ridge'

KEEP_SNAPSHOT = True
RESUME_FROM_SNAPSHOT = 0    # Currently 0 to 3
# GDAL layer creation options
DEF_LAYER_OPTIONS = []
BYDVR_LAYER_OPTIONS = {
    'LIBKML': ['ADD_REGION=YES', 'FOLDER=YES'],
}

# Keep each branch-line one pixes away from its parent
SEPARATED_BRANCHES = False

#
# Internal data-types, mostly for keep/resume support
#
TENTATIVE_DTYPE = [
        ('x_y', (numpy.int32, (2,))),
        ('alt', float),
]
BRANCH_LINE_DTYPE = [
        ('start_xy', (numpy.int32, (2,))),
        ('x_y', (numpy.int32, (2,))),
        ('area', float),
]

#
# Generic tools
#
def sorted_arr_insert(arr, entry, key, end=None):
    """Insert an entry in sorted array by keeping it sorted by key"""
    # When multiple entries are going to be inserted, ensure they are sorted
    # This is only mandatory, when multiple elements have the same insertion point
    if entry.ndim:
        argsort = numpy.argsort(entry[key])
        entry = numpy.take(entry, argsort)
    # Actual insert, do not search beyond 'end'
    idx = numpy.searchsorted(arr[:end][key], entry[key])
    return numpy.insert(arr, idx, entry)

def neighbor_xy(x_y, neighbor_dir):
    """Get the coordinates of a neighbor pixel"""
    if neighbor_dir.ndim < 2:       # Performance optimization
        return (x_y.T + (neighbor_dir % 3 - 1, neighbor_dir // 3 - 1)).T
    return x_y + numpy.stack((neighbor_dir % 3 - 1, neighbor_dir // 3 - 1), -1)

def neighbor_flip(neighbor_dir):
    """Get the inverted neighbor direction"""
    return NEIGHBOR_LAST_VALID - neighbor_dir

def neighbor_xy_safe(x_y, neighbor_dir):
    """Get the coordinates of a neighbor pixel, handle invalid directions"""
    res_xy = neighbor_xy(x_y, neighbor_dir)
    # Threat NEIGHBOR_PENDING and NEIGHBOR_INVALID as NEIGHBOR_SELF,
    # to ensure the raster coordinates are valid
    mask = neighbor_dir > NEIGHBOR_LAST_VALID
    if mask.any():
        res_xy[mask] = x_y[mask]
    return res_xy

def neighbor_is_invalid(neighbor_dir):
    """Return mask of where the neighbor directions are invalid"""
    return neighbor_dir > NEIGHBOR_LAST_VALID

#
# First stage - trace ridges
#
def select_seed(elevations, valleys, mask):
    """Select a point to start ridge/valley tracing"""
    # Keep the original mask if shrinking turns to disaster
    ma_mask = mask.copy()

    # Shrink/mask boundaries to select the seed away from edges
    if elevations.shape[0] > 2 * SEED_INFLATE:
        ma_mask[:SEED_INFLATE,:] = True
        ma_mask[-SEED_INFLATE:,:] = True
    if elevations.shape[1] > 2 * SEED_INFLATE:
        ma_mask[:,:SEED_INFLATE] = True
        ma_mask[:,-SEED_INFLATE:] = True

    # Revert the mask if we have masked everything
    if ma_mask.all():
        ma_mask = mask

    # Use MaskedArray array to find min/max
    elevations = numpy.ma.array(elevations, mask=ma_mask)
    flat_idx = elevations.argmin() if valleys else elevations.argmax()
    seed_xy = numpy.unravel_index(flat_idx, elevations.shape)
    return numpy.array(seed_xy, dtype=numpy.int32)

def process_neighbors(dem_band, dir_arr, x_y):
    """Process the valid and pending neighbor points and return a list to be put to tentative"""
    x_y = x_y[...,numpy.newaxis,:]
    n_xy = neighbor_xy(x_y, VALID_NEIGHBOR_DIRS)
    n_dir = numpy.broadcast_to(VALID_NEIGHBOR_DIRS, n_xy.shape[:-1])
    # Filter out of bounds pixels
    mask = dem_band.in_bounds(n_xy)
    if not mask.all():
        n_xy = n_xy[mask]
        n_dir = n_dir[mask]
    # The lines can only pass-thru inner DEM pixels, the boundary ones do split
    stop_mask = ~mask.all(-1)
    # Filter already processed pixels
    neighs = gdal_utils.read_arr(dir_arr, n_xy)
    mask = neighs == NEIGHBOR_PENDING
    if not mask.any():
        return None
    if not mask.all():
        n_xy = n_xy[mask]
        n_dir = n_dir[mask]
        mask = neighs == NEIGHBOR_BOUNDARY
        if mask.any():
            stop_mask |= mask.any(-1)
    # Process selected pixels
    n_dir = neighbor_flip(n_dir)
    # Put 'stop' markers on the successors of the masked points
    # This is to split lines at the boundary pixels
    if stop_mask.any():
        n_dir[stop_mask] = NEIGHBOR_STOP
    gdal_utils.write_arr(dir_arr, n_xy, n_dir)
    return n_xy

def trace_ridges(dem_band, valleys=False, boundary_val=None):
    """Generate terrain ridges or valleys"""
    # Start at the max/min altitude (first one, away from edges)
    elevations = dem_band.get_elevation(True)
    select_mask = numpy.isnan(elevations)
    if boundary_val is not None:
        select_mask |= elevations == boundary_val
    seed_xy = select_seed(elevations, valleys, select_mask)
    print('Tracing', 'valleys' if valleys else 'ridges',
          'from seed point', seed_xy,
          ', altitude', dem_band.get_elevation(seed_xy))

    #
    # Neighbor directions
    # Initialize the points to be processed with 'pending' value.
    #
    dir_arr = numpy.full(elevations.shape, NEIGHBOR_PENDING, dtype=NEIGHBOR_DIR_DTYPE)
    # Here "select_mask" includes both boundary and "NoDataValue" points
    dir_arr[select_mask] = NEIGHBOR_BOUNDARY
    dir_arr[numpy.isnan(elevations)] = NEIGHBOR_INVALID
    gdal_utils.write_arr(dir_arr, seed_xy, NEIGHBOR_SEED)
    del elevations, select_mask

    #
    # Tentative point list (coord and altitude)
    # Initially contains the start point only
    #
    tentative = numpy.array([(seed_xy, dem_band.get_elevation(seed_xy))], dtype=TENTATIVE_DTYPE)

    progress_idx = 0
    while tentative.size:
        x_y, _ = tentative[-1]
        tentative = tentative[:-1]
        #print('    Processing point %s alt %d, dist %d'%(x_y, _, gdal_utils.read_arr(dir_arr['dist'], x_y)))
        n_xy = process_neighbors(dem_band, dir_arr, x_y)
        if n_xy is not None:
            alts = dem_band.get_elevation(n_xy)
            assert not numpy.isnan(alts).any(), '"NoDataValue" point(s) %s are marked for processing'%n_xy[numpy.isnan(alts)]
            # The valleys are handled by turning the elevations upside down
            if valleys:
                alts = -alts
            # Insert the points in 'tentative' by keeping it sorted by altitude.
            # The duplicated altitudes must be processed in order of appearance (FIFO),
            # i.e. numpy.searchsorted() with "side='left'".
            tentr = numpy.empty(alts.shape, dtype=tentative.dtype)
            tentr['x_y'] = n_xy
            tentr['alt'] = alts
            # The 'tentr' is flipped, only to keep the previous behavior, i.e. the FIFO rule is in
            # effect for the order of 'n_xy'. This is the same as if VALID_NEIGHBOR_DIRS is flipped.
            tentative = sorted_arr_insert(tentative, tentr[::-1], 'alt')

        # After the 'tentative' is exhausted, there still can be islands of valid elevations,
        # that were not processed, because of the surrounding invalid ones
        elif not tentative.size:
            mask = dir_arr == NEIGHBOR_PENDING
            if mask.any():
                # Restart at the highest/lowest unprocessed point
                seed_xy = select_seed(dem_band.get_elevation(True), valleys, numpy.logical_not(mask))
                alt = dem_band.get_elevation(seed_xy)
                print('Restart tracing from seed point', seed_xy, ', altitude', alt)
                gdal_utils.write_arr(dir_arr, seed_xy, NEIGHBOR_SEED)
                tentative = numpy.array([(seed_xy, alt)], dtype=tentative.dtype)

        #
        # Progress, each 10000-th line
        #
        if progress_idx % 10000 == 0:
            alts = tentative['alt']
            print('  Process step %d, tentatives %d, alt max/min %d/%d, remaining %d points'%(progress_idx,
                    tentative.shape[0], alts.max(), alts.min(),
                    numpy.count_nonzero(dir_arr == NEIGHBOR_PENDING)))
        progress_idx += 1

    return dir_arr

#
# Branch identification for the second and third stages
#
def get_mgrid(shape):
    """Create a grid of self-pointing coordinates"""
    mgrid = numpy.mgrid[:shape[0], :shape[1]]
    # The coordinates must be in the last dimension
    return numpy.moveaxis(mgrid, 0, -1)

def calc_pixel_area(distance, shape):
    # Use a helper array, where each element points to it-self
    mgrid_xy = get_mgrid(shape)

    # Helper arrays, where each element points to its X or Y neighbor
    mgrid_xy_x = numpy.concatenate((mgrid_xy[1:2,...], mgrid_xy[:-1,...]), axis=0)
    mgrid_xy_y = numpy.concatenate((mgrid_xy[:,1:2,...], mgrid_xy[:,:-1,...]), axis=1)
    # Multiply distances in X adm Y directions
    area_arr = distance.get_distance(mgrid_xy, mgrid_xy_x) \
             * distance.get_distance(mgrid_xy, mgrid_xy_y)

    # Use "flat" distances for the nodata-elevation pixels
    mask = numpy.isnan(area_arr)
    if mask.any():
        area_arr[mask] = distance.get_distance(mgrid_xy[mask], mgrid_xy_x[mask], True) \
                       * distance.get_distance(mgrid_xy[mask], mgrid_xy_y[mask], True)
    return area_arr

def accumulate_by_mgrid(src_arr, mgrid_n_xy):
    """Accumulate array values into their next points in graph, esp. for graph-nodes"""
    res_arr = numpy.zeros_like(src_arr)
    # To avoid '+=' overlapping, the accumulation is performed by using unbuffered in place
    # operation, see "numpy.ufunc.at".
    indices = numpy.moveaxis(mgrid_n_xy, -1, 0)
    numpy.add.at(res_arr, tuple(indices), src_arr)

    assert abs(numpy.nansum(res_arr) - numpy.nansum(src_arr)) * 1e6 <= numpy.nanmax(src_arr), \
            f'Total sum deviation {numpy.nansum(res_arr) - numpy.nansum(src_arr)}'
    return res_arr

def get_branch_mask(x_y, mgrid_n_xy):
    """Obtains total coverage mask of single branch"""
    # Obtain mask(s) of the root point(s)
    res_mask = numpy.zeros(mgrid_n_xy.shape[:-1], dtype=bool)
    gdal_utils.write_arr(res_mask, x_y, True)
    mask = gdal_utils.read_arr(res_mask, mgrid_n_xy)
    # The AND-NOT is needed to drop the self-pointing graph-seeds
    mask &= ~res_mask
    while mask.any():
        res_mask |= mask
        mask = gdal_utils.read_arr(mask, mgrid_n_xy)
    return res_mask

def calc_branch_area(x_y, mgrid_n_xy, area_arr):
    """Accumulate total coverage area of branch"""
    mask = get_branch_mask(x_y, mgrid_n_xy)
    return area_arr[mask].sum(-1)

def arrange_lines(mgrid_n_xy, area_arr, trunks_only):
    """Arrange lines in branches by using the area of coverage"""
    area_arr = area_arr.copy()

    # Helper 'valid_mask' array where mgrid_n_xy are NOT self-pointers
    valid_mask = (mgrid_n_xy != get_mgrid(mgrid_n_xy.shape[:-1])).any(-1)

    # Count the number of neighbors pointing to each pixel
    # Only the last branch that reaches a pixel continues forward, others stop there.
    # As the branches are processed starting from the one with less coverage-area, this
    # allows the largest one to reach the graph-seed.
    n_num = numpy.ones_like(area_arr, dtype=int)
    n_num[~valid_mask] = 0
    n_num = accumulate_by_mgrid(n_num, mgrid_n_xy)
    # Put -1 at invalid nodes, except the "real" seeds (distinguish from the "leafs")
    n_num[~valid_mask & (n_num == 0)] = -1
    all_leafs = n_num == 0
    print('Detected %d "leaf" and %d "real-seed" pixels'%(
            numpy.count_nonzero(all_leafs),
            numpy.count_nonzero(~valid_mask & (n_num > 0))))

    # Start at the "leaf" pixels
    pend_lines = numpy.zeros(numpy.count_nonzero(all_leafs), dtype=BRANCH_LINE_DTYPE)
    pend_lines['start_xy'] = pend_lines['x_y'] = numpy.array(numpy.nonzero(all_leafs)).T
    pend_lines['area'] = gdal_utils.read_arr(area_arr, pend_lines['x_y'])

    #
    # Process the leaf-branches in parallel
    # The parallel processing must stop at the point before the graph-nodes
    #
    bridge_mask = gdal_utils.read_arr(n_num == 1, mgrid_n_xy)
    bridge_mask &= valid_mask
    all_leafs[...] = False
    pend_mask = numpy.ones(pend_lines.size, dtype=bool)
    x_y = pend_lines['x_y']
    while pend_mask.any():
        gdal_utils.write_arr(all_leafs, x_y, True)
        # Stop in front the graph nodes and at the "seeds"
        mask = gdal_utils.read_arr(bridge_mask, x_y)
        x_y = x_y[mask]
        # Advance the points, which are still at graph-bridges
        x_y = gdal_utils.read_arr(mgrid_n_xy, x_y)
        assert (gdal_utils.read_arr(n_num, x_y) == 1).all()
        gdal_utils.write_arr(n_num, x_y, 0)
        # Keep the intermediate results
        pend_mask[pend_mask] = mask
        pend_lines['x_y'][pend_mask] = x_y
        # Accumulate coverage area
        area = gdal_utils.read_arr(area_arr, x_y)
        pend_lines['area'][pend_mask] += area
    del bridge_mask
    del pend_mask
    assert int(pend_lines['area'].sum()) == int(area_arr[all_leafs].sum()), 'Leaf-branch coverage area mismatch %.6f / %.6f km2'%(
            pend_lines['area'].sum() / 1e6, area_arr[all_leafs].sum() / 1e6)
    # Trim leaf-trunks
    mask = gdal_utils.read_arr(valid_mask, pend_lines['x_y'])
    pend_lines = pend_lines[mask]
    print('  Detected %d pixels in "leaf" branches, area %.2f km2, trim %d leaf-trunks'%(
            numpy.count_nonzero(all_leafs), pend_lines['area'].sum() / 1e6,
            numpy.count_nonzero(~mask)))

    # Update the accumulated area, but only at the stop-points (in front the graph-nodes)
    gdal_utils.write_arr(area_arr, pend_lines['x_y'], pend_lines['area'])

    branch_lines = numpy.empty_like(pend_lines, shape=[0])
    trim_cnt = 0
    progress_idx = 0
    while pend_lines.size:
        # Process the branch with minimal coverage-area
        br_idx = pend_lines['area'].argmin()
        branch = pend_lines[br_idx]
        x_y = branch['x_y']
        area = gdal_utils.read_arr(area_arr, x_y)
        assert branch['area'] <= area, 'Branch area decreases at %s: %f -> %f m2'%(x_y, branch['area'], area)
        if gdal_utils.read_arr(valid_mask, x_y):
            # Advance to the next point
            x_y = gdal_utils.read_arr(mgrid_n_xy, x_y)
            # Accumulate the coverage-area
            area += gdal_utils.read_arr(area_arr, x_y)
            gdal_utils.write_arr(area_arr, x_y, area)

            # Handle node-bridges counter: only the last branch to proceed further
            n = gdal_utils.read_arr(n_num, x_y)
            assert n > 0
            gdal_utils.write_arr(n_num, x_y, n - 1)
            # Stop at graph-node (non-last branches)
            is_stop = n > 1
            keep_branch = not trunks_only

            # Update the end-point
            if not is_stop:
                branch['x_y'] = x_y
                branch['area'] = area

        else:
            # Stop at graph-seed (trunk branch)
            keep_branch = is_stop = True

        if is_stop:
            # Discard the "leaf" branches, with "trunks_only" -- non-trunk branches
            if keep_branch:
                if False == gdal_utils.read_arr(all_leafs, branch['x_y']):
                    branch_lines = numpy.append(branch_lines, branch)
                else:
                    trim_cnt += 1
            pend_lines = numpy.delete(pend_lines, br_idx)

        #
        # Progress, each 10000-th step
        #
        if progress_idx % 10000 == 0:
            area = branch_lines['area'].max() if branch_lines.size else 0
            if pend_lines.size:
                area = max(area, pend_lines['area'].max())
            print('  Process step %d, max. area %.2f km2, completed %d, pending %d, trimmed leaves %d'%(progress_idx,
                    area / 1e6, branch_lines.size, pend_lines.size, trim_cnt))
        progress_idx += 1

    # Confirm everything is processed
    assert (n_num <= 0).all(), 'Unprocessed pixels at %s'%numpy.array(numpy.nonzero(n_num > 0)).T
    return branch_lines

def flip_lines(mgrid_n_xy, x_y):
    """Flip all 'n_dir'-s along multiple lines"""
    n_xy = gdal_utils.read_arr(mgrid_n_xy, x_y)
    gdal_utils.write_arr(mgrid_n_xy, x_y, x_y)
    while True:
        prev_n_xy = gdal_utils.read_arr(mgrid_n_xy, n_xy)
        gdal_utils.write_arr(mgrid_n_xy, n_xy, x_y)
        mask = (prev_n_xy != n_xy).any(-1)
        if not mask.all():
            if not mask.any():
                return mgrid_n_xy
            n_xy = n_xy[mask]
            prev_n_xy = prev_n_xy[mask]

        x_y = n_xy
        n_xy = prev_n_xy

#
# Keep/resume support
#
def keep_arrays(prefix, arr_slices):
    """Store snapshots of multiple arrays"""
    for arr_name in arr_slices:
        arr = arr_slices[arr_name]
        slices = arr.dtype.fields
        for sl in [None] if slices is None else slices:
            if sl is None:
                # Single slice - complete array
                arr_sl_name = arr_name
                arr_sl = arr
            else:
                arr_sl_name = '%s[%s]'%(arr_name, sl)
                arr_sl = arr[sl]
            fname = '%s%s.npy'%(prefix, arr_sl_name)
            print('Keeping snapshot of', arr_sl_name, arr_sl.shape, ':', fname)
            numpy.save(fname, arr_sl)

def restore_arrays(prefix, arr_slices):
    """Load snapshots of multiple arrays"""
    res_list = []
    for arr_name in arr_slices:
        dtype = arr_slices[arr_name]
        arr = None
        for sl in [None] if dtype is None else dtype:
            if sl is None:
                # Single slice - complete array
                arr_sl_name = arr_name
            else:
                arr_sl_name = '%s[%s]'%(arr_name, sl[0])

            fname = '%s%s.npy'%(prefix, arr_sl_name)
            print('Restoring snapshot of', arr_sl_name, ':', fname)
            data = numpy.load(fname)
            print('  Restored: shape', data.shape, ', dtype', data.dtype)

            if sl is None:
                arr = data
            else:
                if arr is None:
                    # Create array by using the correct shape, esp. multidimensional sub-array
                    shape = numpy.empty(0, dtype=dtype)[sl[0]].shape[1:]
                    shape = data.shape[:data.ndim - len(shape)]
                    arr = numpy.empty(shape, dtype=dtype)
                arr[sl[0]] = data

        res_list.append(arr)
    return res_list

#
# Final geometry generation
#
def get_zoom_level(spatial_ref, area):
    """Select min zoom level, where an area is visible"""
    radius = spatial_ref.GetAttrValue('SPHEROID', 1)
    if radius is None:
        return None
    # Approximate total area by using sphere surface area
    radius = float(radius)
    lvl0_area = 4 * numpy.pi * radius**2
    return numpy.log2(lvl0_area / area) / 2

class dst_layer_mgr:
    """Destination layer manager"""
    layer_set = {}

    def __init__(self, dst_ds, spatial_ref, valleys, multi_layer):
        self.dst_ds = dst_ds
        self.spatial_ref = spatial_ref
        self.id_fmt = VECTOR_LAYER_NAME(valleys)
        self.multi_layer = multi_layer

    def delete_all(self):
        """Delete all existing layers"""
        for i in reversed(range(self.dst_ds.get_layer_count())):
            print('  Deleting layer', gdal_utils.gdal_vect_layer(self.dst_ds, i).get_name())
            self.dst_ds.delete_layer(i)

    def get_layer(self, branch):
        """Obtain/create layer for specific geometry"""
        # Select layer ID and chceck if it's already created
        if self.multi_layer:
            level = round(get_zoom_level(self.spatial_ref, branch['area']))
            layer_id = self.id_fmt + '_level%d'%level
            layer_options = ['NAME=' + self.id_fmt + ' - level %d'%level]
        else:
            layer_id = self.id_fmt
            layer_options = []
        if layer_id in self.layer_set:
            return self.layer_set[layer_id]

        # Add some more layer options
        layer_options += DEF_LAYER_OPTIONS
        bydrv_options = BYDVR_LAYER_OPTIONS.get(self.dst_ds.get_drv_name())
        if bydrv_options:
            layer_options += bydrv_options
        # Create the layer
        dst_layer = gdal_utils.gdal_vect_layer.create(self.dst_ds,
                layer_id,
                srs=self.spatial_ref, geom_type=gdal_utils.wkbLineString,
                options=layer_options)
        if dst_layer is None:
            print('Error: Unable to create layer', file=sys.stderr)
            return None
        self.layer_set[layer_id] = dst_layer

        # Add fields
        dst_layer.create_field('Name', gdal_utils.OFTString)    # KML <name>
        dst_layer.create_field('Description', gdal_utils.OFTString) # KML <description>
        if FEATURE_OSM_NATURAL:
            dst_layer.create_field('natural', gdal_utils.OFTString) # OSM "natural" key
        return dst_layer

#
# Main processing
#
def main(argv):
    """Main entry"""
    valleys = False
    boundary_val = None
    multi_layer = False
    maxzoom_level = None
    truncate = True
    src_filename = dst_filename = None
    while argv:
        if argv[0][0] == '-':
            if argv[0] == '-h':
                return print_help()
            if argv[0] == '-valley':
                valleys = True
            elif argv[0] == '-boundary_val':
                argv = argv[1:]
                boundary_val = float(argv[0])
            elif argv[0] == '-multi_layer':
                argv = argv[1:]
                multi_layer = True
                maxzoom_level = float(argv[0])
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

    #
    # Trace ridges/valleys
    #
    if RESUME_FROM_SNAPSHOT < 1:

        start = time.perf_counter()

        # Actual trace
        dir_arr = trace_ridges(dem_band, valleys, boundary_val)
        if dir_arr is None:
            print('Error: Failed to trace ridges', file=sys.stderr)
            return 2

        duration = time.perf_counter() - start
        print('Traced through %d/%d points, %d sec'%(numpy.count_nonzero(~neighbor_is_invalid(dir_arr)), dir_arr.size, duration))

        # Convert 'dir_arr' to 'mgrid_n_xy' array, where each element points to its neighbor
        mgrid_n_xy = neighbor_xy_safe(get_mgrid(dir_arr.shape), dir_arr)
        del dir_arr

        if KEEP_SNAPSHOT:
            keep_arrays(src_filename + '-1-', {'mgrid_n_xy': mgrid_n_xy,})
    elif RESUME_FROM_SNAPSHOT == 1:
        mgrid_n_xy, = restore_arrays(src_filename + '-1-', {'mgrid_n_xy': None,})

    #
    # The coverage-area of each pixels is needed by arrange_lines()
    # The distance object is used to calculate the branch length
    #
    distance = gdal_utils.geod_distance(dem_band) if 0 == DISTANCE_METHOD \
            else gdal_utils.tm_distance(dem_band) if 1 == DISTANCE_METHOD \
            else gdal_utils.draft_distance(dem_band)
    area_arr = calc_pixel_area(distance, dem_band.shape)
    print('Calculated total area %.2f km2, mean %.2f m2'%(area_arr.sum() / 1e6, area_arr.mean()))

    #
    # Identify and flip the "trunk" branches
    # All the real-seeds become regular graph-nodes or "leaf" pixel.
    # The former start/leaf pixel of these branches becomes a "seed".
    #
    if RESUME_FROM_SNAPSHOT < 2:

        start = time.perf_counter()

        # Arrange branches to select which one to flip (trunks_only)
        branch_lines = arrange_lines(mgrid_n_xy, area_arr, True)
        if branch_lines is None or branch_lines.size == 0:
            print('Error: Unable to identify any branch', file=sys.stderr)
            return 2

        # Actual flip
        if flip_lines(mgrid_n_xy, branch_lines['start_xy']) is None:
            print('Error: Failed to flip %d branches'%(branch_lines.size), file=sys.stderr)
            return 2

        duration = time.perf_counter() - start
        print('Flip & merge total %d trunk-branches, max/min area %.1f/%.3f km2, %d sec'%(
                branch_lines.size, branch_lines['area'].max() / 1e6, branch_lines['area'].min() / 1e6,
                duration))

        if KEEP_SNAPSHOT:
            keep_arrays(src_filename + '-2-', {
                    'mgrid_n_xy': mgrid_n_xy,
                    'branch_lines': branch_lines,
                })
    elif RESUME_FROM_SNAPSHOT == 2:
        mgrid_n_xy, branch_lines = restore_arrays(src_filename + '-2-', {
                    'mgrid_n_xy': None,
                    'branch_lines': BRANCH_LINE_DTYPE,
                })

    #
    # Identify all the branches
    #
    if RESUME_FROM_SNAPSHOT < 3:

        start = time.perf_counter()

        # Arrange branches
        branch_lines = arrange_lines(mgrid_n_xy, area_arr, False)
        if branch_lines is None or branch_lines.size == 0:
            print('Error: Unable to identify any branch', file=sys.stderr)
            return 2

        # Sort the the generated branches (descending 'area' order)
        argsort = numpy.argsort(branch_lines['area'])
        branch_lines = numpy.take(branch_lines, argsort[::-1])

        if maxzoom_level is None:
            # Trim to 5 zoom-levels (1/1024 of max area)
            min_area = area_arr.sum() / (4 ** 5)
        else:
            # Trim to the area at 'maxzoom_level'
            lvl = get_zoom_level(dem_band.get_spatial_ref(), 1)
            min_area = 4 ** (lvl - maxzoom_level - .5)  # The .5 is to match round() used by dst_layer_mgr.get_layer()

        print('  Trimming total %d branches to min area of %.3f km2 (currently %.3f km2)'%(
                branch_lines.size, min_area / 1e6, branch_lines['area'].min() / 1e6))
        branch_lines = branch_lines[branch_lines['area'] >= min_area]

        duration = time.perf_counter() - start
        print('Created total %d branches, max/min area %.1f/%.3f km2, %d sec'%(
                branch_lines.size, branch_lines['area'].max() / 1e6, branch_lines['area'].min() / 1e6,
                duration))

        if KEEP_SNAPSHOT:
            keep_arrays(src_filename + '-3-', {
                    'branch_lines': branch_lines,
                })
    elif RESUME_FROM_SNAPSHOT == 3:
        mgrid_n_xy, = restore_arrays(src_filename + '-2-', {
                    'mgrid_n_xy': None,
                })
        branch_lines, = restore_arrays(src_filename + '-3-', {
                    'branch_lines': BRANCH_LINE_DTYPE,
                })

    if dst_ds:
        start = time.perf_counter()

        layer_mgr = dst_layer_mgr(dst_ds, dem_band.get_spatial_ref(), valleys, multi_layer)
        # Delete existing layers
        if truncate:
            layer_mgr.delete_all()

        geometries = 0
        for branch in branch_lines:
            ar = calc_branch_area(branch['x_y'], mgrid_n_xy, area_arr)
            assert round(branch['area']) == round(ar), 'Accumulated branch coverage area mismatch %.6f / %.6f km2'%(
                    branch['area'] / 1e6, ar / 1e6)
            # Select the layer, where to add the geometry, create if missing
            dst_layer = layer_mgr.get_layer(branch)
            if dst_layer is None:
                return 1

            # Advance one step forward to connect to the parent branch
            if not SEPARATED_BRANCHES:
                x_y = branch['x_y']
                branch['x_y'] = gdal_utils.read_arr(mgrid_n_xy, x_y)
            # Extract the branch pixel coordinates and calculate length
            x_y = branch['start_xy']
            polyline = [x_y]
            dist = 0.
            while (x_y != branch['x_y']).any():
                # Advance to the next point
                new_xy = gdal_utils.read_arr(mgrid_n_xy, x_y)
                dist += distance.get_distance(x_y, new_xy)
                x_y = new_xy
                polyline.append(x_y)

            # Create actual geometry
            geom = dst_layer.create_feature_geometry(gdal_utils.wkbLineString)
            geom.set_field('Name', '%dm'%dist if dist < 10000 else '%dkm'%round(dist/1000))
            geom.set_field('Description', 'length: %.1f km, area: %.1f km2'%(dist / 1e3, branch['area'] / 1e6))
            if FEATURE_OSM_NATURAL:
                geom.set_field('natural', FEATURE_OSM_NATURAL(valleys))
            geom.set_style_string(VECTOR_FEATURE_STYLE(valleys))

            # Reverse the line to match the tracing direction
            for x_y in reversed(polyline):
                geom.add_point(*dem_band.xy2lonlatalt(x_y))
            geom.create()
            geometries += 1

        duration = time.perf_counter() - start
        print('Created total %d geometries, %d sec'%(geometries, duration))

    return 0

def print_help(err_msg=None):
    if err_msg:
        print('Error:', err_msg, file=sys.stderr)
    print('Usage:', sys.argv[0], '[<options>] <src_filename> <dst_filename>')
    print('\tOptions:')
    print('\t-h\t- This screen')
    print('\t-valley\t- Generate valleys, instead of ridges')
    print('\t-boundary_val <ele> - Treat the neighbors next to <ele> as boundary')
    print('\t-multi_layer <maxzoom> - Create multiple layers upto a zoom-level (check OGR driver capabilities)')
    return 0 if err_msg is None else 255

if __name__ == '__main__':
    ret = main(sys.argv[1:])
    if ret:
        exit(ret)
