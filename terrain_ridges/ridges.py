"""Generate terrain ridges/valleys"""
import sys
import time
import argparse
import numpy
import gdal_utils

# Neighbor directions:
#   0 1 2
#   3<4>5
#   6 7 8
VALID_NEIGHBOR_DIRS = numpy.array(
    (0, 1, 2, 3, 5, 6, 7, 8),
    dtype=numpy.int8)

# Keep the seed away from the edges
SEED_INFLATE = 1
# Distance calculation methods, '--distance-method' option:
DISTANCE_METHODS = {
    'draft': gdal_utils.draft_distance,     # Draft: use pre-calculated pixel size by tm_distance for all pixels
    'tm': gdal_utils.tm_distance,           # Transverse Mercator: use TM origin at the center of raster data
    'geod': gdal_utils.geod_distance,       # Real geodetic distance: use pyproj.Geod.inv()
}

def VECTOR_LAYER_NAME(valleys): return 'valleys' if valleys else 'ridges'
def VECTOR_FEATURE_STYLE(valleys): return 'PEN(c:#0000FF,w:2px)' if valleys else 'PEN(c:#FF0000,w:2px)'
# Value for the OSM "natural" keys, to allow conversion to .osm
# by ogr2osm.py or JOSM (opendata plugin)
#   https://wiki.openstreetmap.org/wiki/Tag:natural%3Dridge
#   https://wiki.openstreetmap.org/wiki/Tag:natural%3Dvalley
def FEATURE_OSM_NATURAL(valleys): return 'valley' if valleys else 'ridge'

# GDAL layer creation options
DEF_LAYER_OPTIONS = []
BYDVR_LAYER_OPTIONS = {
    'LIBKML': ['ADD_REGION=YES', 'FOLDER=YES'],
}

# Run extra (slow) internal tests
ASSERT_LEVEL = 2

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
        argsort = numpy.argsort(entry[key], kind='stable')
        entry = numpy.take(entry, argsort)
    # Actual insert, do not search beyond 'end'
    idx = numpy.searchsorted(arr[:end][key], entry[key])
    return numpy.insert(arr, idx, entry)

def neighbor_xy(x_y, neighbor_dir):
    """Get the coordinates of a neighbor pixel"""
    if neighbor_dir.ndim < 2:       # Performance optimization
        return (x_y.T + (neighbor_dir % 3 - 1, neighbor_dir // 3 - 1)).T
    return x_y + numpy.stack((neighbor_dir % 3 - 1, neighbor_dir // 3 - 1), -1)

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

def process_neighbors(dem_band, mgrid_n_xy, pending_mask, boundary_mask, x_y):
    """Process the valid and pending neighbor points and return a list to be put to tentative"""
    gdal_utils.write_arr(pending_mask, x_y, False)
    x_y = x_y[...,numpy.newaxis,:]
    n_xy = neighbor_xy(x_y, VALID_NEIGHBOR_DIRS)
    # Filter out of bounds pixels
    mask = dem_band.in_bounds(n_xy)
    if not mask.all():
        n_xy = n_xy[mask]
    # The lines can only pass-thru inner DEM pixels, the boundary ones do split
    stop_mask = ~mask.all(-1)
    # Filter already processed pixels
    mask = gdal_utils.read_arr(pending_mask, n_xy)
    if not mask.any():
        return None
    gdal_utils.write_arr(pending_mask, n_xy, False)
    if not mask.all():
        m = gdal_utils.read_arr(boundary_mask, n_xy)
        n_xy = n_xy[mask]
        if m.any():
            stop_mask |= m.any(-1)
    # Skip neighbor update for the successors of the 'stop_mask' points
    # This is to split lines at the boundary pixels
    gdal_utils.write_arr(mgrid_n_xy, n_xy[~stop_mask], x_y)
    return n_xy

def trace_ridges(dem_band, valleys=False, boundary_val=None):
    """Generate terrain ridges or valleys"""
    # Select 'pending' and 'boundary' masks
    elevations = dem_band.get_elevation(True)
    boundary_mask = numpy.zeros_like(elevations, dtype=bool)
    if boundary_val is not None:
        boundary_mask = elevations == boundary_val
    pending_mask = numpy.isfinite(elevations) & ~boundary_mask
    # Start at the max/min altitude (first one, away from edges)
    seed_xy = select_seed(elevations, valleys, ~pending_mask)
    print('Tracing', 'valleys' if valleys else 'ridges',
          'from seed point', seed_xy,
          ', altitude', dem_band.get_elevation(seed_xy))

    #
    # Neighbor mgrid pointers
    # Initially each mgrid point, points to itself
    #
    mgrid_n_xy = get_mgrid(elevations.shape)
    del elevations

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
        n_xy = process_neighbors(dem_band, mgrid_n_xy, pending_mask, boundary_mask, x_y)
        if n_xy is not None:
            alts = dem_band.get_elevation(n_xy)
            if ASSERT_LEVEL >= 1:
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
            if pending_mask.any():
                # Restart at the highest/lowest unprocessed point
                seed_xy = select_seed(dem_band.get_elevation(True), valleys, ~pending_mask)
                alt = dem_band.get_elevation(seed_xy)
                print('Restart tracing from seed point', seed_xy, ', altitude', alt)
                tentative = numpy.array([(seed_xy, alt)], dtype=tentative.dtype)

        #
        # Progress, each 10000-th line
        #
        if progress_idx % 10000 == 0:
            alts = tentative['alt']
            print('  Process step %d, tentatives %d, alt max/min %d/%d, remaining %d points'%(progress_idx,
                    tentative.shape[0], alts.max(), alts.min(),
                    numpy.count_nonzero(pending_mask)))
        progress_idx += 1

    return mgrid_n_xy

#
# Branch identification for the second and third stages
#
def get_mgrid(shape):
    """Create a grid of self-pointing coordinates"""
    mgrid = numpy.indices(shape)
    # The coordinates must be in the last dimension
    if mgrid.ndim <= 2:
        return mgrid.T  # Performance optimization
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

def accumulate_by_mgrid(src_arr, mgrid_n_xy, mask=Ellipsis):
    """Accumulate array values into their next points in graph, esp. for graph-nodes"""
    res_arr = numpy.zeros_like(src_arr)
    src_arr = src_arr[mask]
    # To avoid '+=' overlapping, the accumulation is performed by using unbuffered in place
    # operation, see "numpy.ufunc.at".
    indices = mgrid_n_xy[mask]
    indices = numpy.moveaxis(indices, -1, 0) if indices.ndim > 2 else indices.T # Performance optimization
    numpy.add.at(res_arr, tuple(indices), src_arr)

    if ASSERT_LEVEL >= 3:
        assert numpy.isclose(numpy.nansum(res_arr), numpy.nansum(src_arr)), \
                f'Total sum deviation {numpy.nansum(res_arr) - numpy.nansum(src_arr)}'
    return res_arr

def accumulate_pixel_coverage(area_arr, mgrid_n_xy):
    """Accumulate branch coverage area for each pixel"""
    area_arr = area_arr.copy()
    # Helper 'seed_mask' array where mgrid_n_xy are self-pointers
    seed_mask = (mgrid_n_xy == get_mgrid(mgrid_n_xy.shape[:-1])).all(-1)
    total_area = numpy.nansum(area_arr)
    print('Accumulating the coverage area: total %.2f km2, %d points, %d seeds'%(
            total_area / 1e6, area_arr.size, numpy.count_nonzero(seed_mask)))

    src_arr = numpy.where(seed_mask, 0, area_arr)
    progress_idx = 1
    while src_arr.any():
        src_arr = accumulate_by_mgrid(src_arr, mgrid_n_xy, src_arr != 0)
        area_arr += src_arr
        src_arr[seed_mask] = 0.

        #
        # Progress, each 1000-th step
        #
        if progress_idx % 1000 == 0:
            print('  Process step %d, max area %.2f km2, remaining %d points'%(
                    progress_idx, area_arr.max() / 1e6, numpy.count_nonzero(src_arr != 0)))
        progress_idx += 1

    print('  Accumulated area at seeds: max/mean %.2f/%.2f km2'%(
            area_arr.max() / 1e6, area_arr[seed_mask].mean() / 1e6))
    if ASSERT_LEVEL >= 2:
        assert area_arr.max() == area_arr[seed_mask].max(), 'Max area is not at a seed-point'
        assert numpy.isclose(total_area, numpy.nansum(area_arr[seed_mask])), \
                'Total area does not match the sum of seeds %.6f / %.6f km2'%(
                    total_area / 1e6, numpy.nansum(area_arr[seed_mask]) / 1e6)
    return area_arr

def arrange_lines(mgrid_n_xy, area_arr, trunks_only):
    """Arrange lines in branches by using the area of coverage"""
    # Keep caller's mgrid
    mgrid_n_xy = mgrid_n_xy.copy()
    # Helper self-pointing array
    mgrid_xy = get_mgrid(mgrid_n_xy.shape[:-1])
    # Will need the initial "seed" pixels to identify trunks
    seed_mask = (mgrid_n_xy == mgrid_xy).all(-1)
    # Will need the initial "node" pixels to identify leaf-branches
    n_num = numpy.where(seed_mask, 0, 1)
    n_num = accumulate_by_mgrid(n_num, mgrid_n_xy)
    nodes_mask = n_num > 1
    # Accumulate coverage area of each point
    area_arr = accumulate_pixel_coverage(area_arr, mgrid_n_xy)

    #
    # Extract separate branch-lines: graph where each node has single neighbor
    #
    # At each pixel, place the greatest area from all neighbors pointing it
    # (if integer, replace '-numpy.inf' with "numpy.iinfo(area_arr.dtype).min")
    max_area = numpy.full_like(area_arr, -numpy.inf)
    indices = tuple(numpy.moveaxis(mgrid_n_xy, -1, 0))
    # Ignore the areas of the self-pointing "seed" pixels
    cut_area = numpy.where(~seed_mask, area_arr, -numpy.inf)
    numpy.maximum.at(max_area, indices, cut_area)
    del cut_area
    # Take the greatest area back into neighbors (greatest value among siblings)
    # Then compare to the original one, to get the points with less coverage-area
    max_area = max_area[indices]
    mask = area_arr < max_area
    # Cut branches at selected points, by making them self-pointing
    mgrid_n_xy[mask] = mgrid_xy[mask]

    # Count the number of neighbors pointing to each pixel
    n_num = numpy.where((mgrid_n_xy == mgrid_xy).all(-1), 0, 1)
    n_num = accumulate_by_mgrid(n_num, mgrid_n_xy)

    # Create forks:
    # Cut all branches, if more than one neighbors have greatest coverage-area
    mask = n_num > 1
    if mask.any():
        mask = gdal_utils.read_arr(mask, mgrid_n_xy)
        mgrid_n_xy[mask] = mgrid_xy[mask]

    # Re-count the number of neighbors: must be one
    n_num = numpy.where((mgrid_n_xy == mgrid_xy).all(-1), 0, 1)
    n_num = accumulate_by_mgrid(n_num, mgrid_n_xy)
    if ASSERT_LEVEL >= 1:
        assert n_num.max() <= 1, 'Unprocessed pixels at %s'%numpy.array(numpy.nonzero(n_num > 1)).T

    # Helper 'valid_mask' array where mgrid_n_xy are NOT self-pointers
    valid_mask = (mgrid_n_xy != mgrid_xy).any(-1)
    # Identify "leafs", but drop the "leaf-seeds" singletons
    all_leafs = (n_num == 0) & valid_mask
    print('Detected %d "leaf" and %d "real-seed" pixels'%(
            numpy.count_nonzero(all_leafs),
            numpy.count_nonzero(seed_mask & (n_num > 0))))

    # Trace branches startinf at the "leaf" pixels
    branch_lines = numpy.zeros(numpy.count_nonzero(all_leafs), dtype=BRANCH_LINE_DTYPE)
    branch_lines['start_xy'] = numpy.argwhere(all_leafs)

    x_y = branch_lines['start_xy'].copy()
    n_nodes = numpy.zeros_like(branch_lines, dtype=int)
    pend_mask = numpy.ones(branch_lines.size, dtype=bool)
    while pend_mask.any():
        # Stop at "seed" points
        pend_mask[pend_mask] = gdal_utils.read_arr(valid_mask, x_y[pend_mask])
        # Advance the points, which are still in the middle of a branch
        x_y[pend_mask] = gdal_utils.read_arr(mgrid_n_xy, x_y[pend_mask])
        n_nodes[pend_mask] += gdal_utils.read_arr(nodes_mask, x_y[pend_mask])
    branch_lines['x_y'] = x_y
    # Drop the "leaf" (single-line) branches
    branch_lines = branch_lines[n_nodes > 0]

    if trunks_only:
        # Leave only branched ending at a "seed" from initial mgrid
        mask = gdal_utils.read_arr(seed_mask, branch_lines['x_y'])
        branch_lines = branch_lines[mask]

    # Update branch area
    branch_lines['area'] = gdal_utils.read_arr(area_arr, branch_lines['x_y'])
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
    def __init__(self, dst_ds, spatial_ref, valleys, multi_layer):
        self.dst_ds = dst_ds
        self.spatial_ref = spatial_ref
        self.id_fmt = VECTOR_LAYER_NAME(valleys)
        self.multi_layer = multi_layer
        self.layer_set = {}

    def delete_all(self):
        """Delete all existing layers"""
        for i in reversed(range(self.dst_ds.get_layer_count())):
            print('  Deleting layer', gdal_utils.gdal_vect_layer(self.dst_ds, i).get_name())
            self.dst_ds.delete_layer(i)

    def get_layer(self, branch):
        """Obtain/create layer for specific geometry"""
        # Select layer ID and check if it's already created
        layer_id = self.id_fmt
        layer_options = DEF_LAYER_OPTIONS
        if self.multi_layer:
            level = get_zoom_level(self.spatial_ref, branch['area'])
            if level is not None:
                level = round(level)
                layer_id += '_level%d'%level
                layer_options += ['NAME=' + self.id_fmt + ' - level %d'%level]
        if layer_id in self.layer_set:
            return self.layer_set[layer_id], False

        # Add some more layer options
        layer_options += BYDVR_LAYER_OPTIONS.get(self.dst_ds.get_drv_name(), [])
        # Create the layer
        dst_layer = gdal_utils.gdal_vect_layer.create(self.dst_ds,
                layer_id,
                srs=self.spatial_ref, geom_type=gdal_utils.wkbLineString,
                options=layer_options)
        if dst_layer is None:
            print('Error: Unable to create layer', file=sys.stderr)
            return None, None
        self.layer_set[layer_id] = dst_layer
        return dst_layer, True

def filter_mgrid(mgrid_n_xy, start_xy):
    """Keep only points reachable from 'start_xy', invalidate others"""
    # Start with mask at 'start_xy'
    pend_mask = numpy.zeros(shape=mgrid_n_xy.shape[:-1], dtype=bool)
    gdal_utils.write_arr(pend_mask, start_xy, True)
    mask = pend_mask.copy()
    while mask.any():
        # Contract the mask
        gdal_utils.write_arr(mask, mgrid_n_xy[mask], True)
        mask &= ~pend_mask
        pend_mask |= mask

    # Invalidate selected points, by making them self-pointing
    mgrid_xy = get_mgrid(mgrid_n_xy.shape[:-1])
    return numpy.where(pend_mask[...,numpy.newaxis], mgrid_n_xy, mgrid_xy)

def smoothen_by_mgrid(lonlatalt, mgrid_n_xy):
    """Average each point with its neighbors"""
    # Count the number of neighbors pointing to each pixel
    n_num = numpy.ones(shape=lonlatalt.shape[:-1], dtype=int)
    n_num = accumulate_by_mgrid(n_num, mgrid_n_xy)
    # Get mask of 'leaf' and 'seed' points
    keep_mask = (mgrid_n_xy == get_mgrid(mgrid_n_xy.shape[:-1])).all(-1)
    keep_mask |= n_num == 0

    # Sum of coordinates of neighbors pointing to each point (total n_num)
    lla_sum = accumulate_by_mgrid(lonlatalt, mgrid_n_xy)
    # Add the coordinates of neighbor pointed from each point (total 1)
    lla_sum += gdal_utils.read_arr(lonlatalt, mgrid_n_xy)
    n_num += 1
    # Add the up-scaled coordinates of the point itself
    # (scaling gives more weight against the neighbor points)
    n_num = n_num[...,numpy.newaxis]
    lla_sum += lonlatalt * n_num
    n_num += n_num

    # Keep 'leaf' and 'seed' points intact
    lla_sum[keep_mask] = lonlatalt[keep_mask]
    n_num[keep_mask] = 1
    return lla_sum / n_num

#
# Main processing
#
def main(args):
    """Main entry"""
    # Load DEM
    dem_band = gdal_utils.dem_open(args.src_dem_file)
    if dem_band is None:
        print(f'Error: Unable to open source DEM "{args.src_dem_file}"', file=sys.stderr)
        return 1

    dst_ds = gdal_utils.vect_create(args.dst_ogr_file, drv_name=args.dst_format)
    if dst_ds is None:
        print(f'Error: Unable to create destination OGR "{args.dst_ogr_file}"', file=sys.stderr)
        return 1

    dem_band.load()

    #
    # Trace ridges/valleys
    #
    if args.resume_from_snapshot < 1:

        start = time.perf_counter()

        # Actual trace
        mgrid_n_xy = trace_ridges(dem_band, args.valleys, args.boundary_val)
        if mgrid_n_xy is None:
            print('Error: Failed to trace ridges', file=sys.stderr)
            return 2

        duration = time.perf_counter() - start
        ch_mask = (get_mgrid(dem_band.shape) != mgrid_n_xy).any(-1)
        print('Traced through %d/%d points, %d sec'%(
                numpy.count_nonzero(ch_mask), mgrid_n_xy[...,0].size, duration))
        del ch_mask

        if args.keep_snapshots:
            keep_arrays(args.src_dem_file + '-1-', {'mgrid_n_xy': mgrid_n_xy,})
    elif args.resume_from_snapshot == 1:
        mgrid_n_xy, = restore_arrays(args.src_dem_file + '-1-', {'mgrid_n_xy': None,})
    else:
        mgrid_n_xy = None   # Workaround Static Type Checker issue

    #
    # The coverage-area of each pixels is needed by arrange_lines()
    # The distance object is used to calculate the branch length
    #
    distance = DISTANCE_METHODS[args.distance_method](dem_band)
    area_arr = calc_pixel_area(distance, dem_band.shape)
    print('Calculated total area %.2f km2, mean %.2f m2'%(area_arr.sum() / 1e6, area_arr.mean()))

    #
    # Identify and flip the "trunk" branches
    # All the real-seeds become regular graph-nodes or "leaf" pixel.
    # The former start/leaf pixel of these branches becomes a "seed".
    #
    if args.resume_from_snapshot < 2:

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

        if args.keep_snapshots:
            keep_arrays(args.src_dem_file + '-2-', {
                    'mgrid_n_xy': mgrid_n_xy,
                    'branch_lines': branch_lines,
                })
    elif args.resume_from_snapshot == 2:
        mgrid_n_xy, branch_lines = restore_arrays(args.src_dem_file + '-2-', {
                    'mgrid_n_xy': None,
                    'branch_lines': BRANCH_LINE_DTYPE,
                })

    #
    # Identify all the branches
    #
    if args.resume_from_snapshot < 3:

        start = time.perf_counter()

        # Arrange branches
        branch_lines = arrange_lines(mgrid_n_xy, area_arr, False)
        if branch_lines is None or branch_lines.size == 0:
            print('Error: Unable to identify any branch', file=sys.stderr)
            return 2

        # Sort the the generated branches (descending 'area' order)
        argsort = numpy.argsort(branch_lines['area'])
        branch_lines = numpy.take(branch_lines, argsort[::-1])

        maxzoom_level = args.multi_layer if isinstance(args.multi_layer, (int, float)) else None
        if maxzoom_level is None:
            # Trim to a zoom-level, 3 levels above the mean pixel size
            min_area = numpy.nanmean(area_arr) * (4 ** 3)
        else:
            # Trim to the area at 'maxzoom_level'
            lvl = get_zoom_level(dem_band.get_spatial_ref(), 1)     # The zoom-level of 1m^2
            if lvl is None:
                min_area = 0        # include all branches
            else:
                min_area = 4 ** (lvl - maxzoom_level - .5)  # The .5 is to match round() used by dst_layer_mgr.get_layer()

        mask = branch_lines['area'] >= min_area
        if numpy.count_nonzero(mask) > 0:
            print('  Trimming total %d branches to %d, min area of %.3f km2 (currently %.3f km2)'%(
                    branch_lines.size, numpy.count_nonzero(mask),
                    min_area / 1e6, branch_lines['area'].min() / 1e6))
            branch_lines = branch_lines[mask]

        duration = time.perf_counter() - start
        print('Created total %d branches, max/min area %.1f/%.3f km2, %d sec'%(
                branch_lines.size, branch_lines['area'].max() / 1e6, branch_lines['area'].min() / 1e6,
                duration))

        if args.keep_snapshots:
            keep_arrays(args.src_dem_file + '-3-', {
                    'branch_lines': branch_lines,
                })
    elif args.resume_from_snapshot == 3:
        mgrid_n_xy, = restore_arrays(args.src_dem_file + '-2-', {
                    'mgrid_n_xy': None,
                })
        branch_lines, = restore_arrays(args.src_dem_file + '-3-', {
                    'branch_lines': BRANCH_LINE_DTYPE,
                })
    else:
        # Workaround Type Checker issue
        raise ValueError('Unsupported --resume-from-snapshot level')

    #
    # Generate geometry
    #
    if dst_ds:
        start = time.perf_counter()

        # Branch coverage area of each pixel (branch['area'] assert only)
        acc_area_arr = accumulate_pixel_coverage(area_arr, mgrid_n_xy) if ASSERT_LEVEL >= 3 else None
        del area_arr

        layer_mgr = dst_layer_mgr(dst_ds, dem_band.get_spatial_ref(), args.valleys, args.multi_layer is not None)
        # Delete existing layers
        if not args.append:
            layer_mgr.delete_all()

        # Generate x_y to lon/lat/alt conversion grid
        mgrid_lonlatalt = dem_band.xy2lonlatalt(get_mgrid(dem_band.shape))
        if args.smoothen_geometry:
            mgrid_lonlatalt = smoothen_by_mgrid(mgrid_lonlatalt, filter_mgrid(mgrid_n_xy, branch_lines['start_xy']))

        name_field = desc_field = natural_field = None
        geometries = 0
        for branch in branch_lines:
            if ASSERT_LEVEL >= 3:
                ar = gdal_utils.read_arr(acc_area_arr, branch['x_y'])
                assert numpy.isclose(branch['area'], ar), 'Accumulated branch coverage area mismatch %.6f / %.6f km2'%(
                        branch['area'] / 1e6, ar / 1e6)
            # Select the layer, where to add the geometry, create if missing
            dst_layer, is_new = layer_mgr.get_layer(branch)
            if dst_layer is None:
                return 1

            # Add fields
            if is_new:
                name_field = dst_layer.create_field('Name', gdal_utils.OFTString)    # KML <name>
                desc_field = dst_layer.create_field('Description', gdal_utils.OFTString) # KML <description>
                if FEATURE_OSM_NATURAL:
                    natural_field = dst_layer.create_field('natural', gdal_utils.OFTString) # OSM "natural" key

            # Advance one step forward to connect to the parent branch
            if not args.separated_branches:
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
            if geom is None:
                print(f'Error: Unable to create OGR geometry', file=sys.stderr)
                return 1
            geom.set_field(name_field, '%dm'%dist if dist < 10000 else '%dkm'%round(dist/1000))
            geom.set_field(desc_field, 'length: %.1f km, area: %.1f km2'%(dist / 1e3, branch['area'] / 1e6))
            if FEATURE_OSM_NATURAL:
                geom.set_field(natural_field, FEATURE_OSM_NATURAL(args.valleys))
            geom.set_style_string(VECTOR_FEATURE_STYLE(args.valleys))

            # Reverse the line to match the tracing direction
            for x_y in reversed(polyline):
                geom.add_point(*gdal_utils.read_arr(mgrid_lonlatalt, x_y))
            geom.create()
            geometries += 1

        dst_ds.flush_cache()
        duration = time.perf_counter() - start
        print('Created total %d geometries, %d sec'%(geometries, duration))

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Terrain ridges visualization')
    parser.add_argument('src_dem_file',
            help='Input DEM file, formats supported by https://gdal.org')
    parser.add_argument('dst_ogr_file',
            help='Output vector file, formats supported by https://gdal.org')
    parser.add_argument('--dst-format', '-f',
            help='Output format name')
    parser.add_argument('--valleys', action='store_true',
            help='Generate valleys, instead of ridges')
    parser.add_argument('--boundary-val', type=float,
            help='Generate valleys, instead of ridges')
    parser.add_argument('--distance-method', choices=DISTANCE_METHODS.keys(), default=next(reversed(DISTANCE_METHODS)),
            help='Select distance calculation method')
    parser.add_argument('--multi-layer', nargs='?', type=float, const=True,
            help='Create multiple layers upto a zoom-level, auto-select if level is skipped (check OGR driver capabilities)')
    parser.add_argument('--append', action='store_true',
            help='Append to existing output geometry (do not truncate)')
    parser.add_argument('--separated-branches', action='store_true',
            help='Keep each branch-line one pixes away from its parent')
    parser.add_argument('--smoothen-geometry', action='store_true',
            help='Smoothen final geometry (avoids the jagged effect, caused by the DEM resolution)')
    parser.add_argument('--assert-level', choices=range(4), type=int, default=ASSERT_LEVEL,
            help='Select internal tests complexity (3 - slowest)')
    parser.add_argument('--keep-snapshots', action='store_true',
            help='Keep intermediate results between stages')
    parser.add_argument('--resume-from-snapshot', choices=range(4), type=int, default=0,
            help='Resume from a stage result stored by "--keep-snapshots"')

    args = parser.parse_args()
    ASSERT_LEVEL = args.assert_level
    ret = main(args)
    if ret:
        exit(ret)
