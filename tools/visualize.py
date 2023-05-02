"""Terrain ridges visualization

Requires: numpy plotly gdal pyproj
"""
import os
import argparse
import numpy as np
import plotly.graph_objects as go
import gdal_utils

DEF_MGRID_N_POSTFIX = '-1-mgrid_n_xy.npy'
#DEF_MAPBOX_STYLE = 'open-street-map'
DEF_MAPBOX_STYLE = 'mapbox://styles/mapbox/outdoors-v12'
#DEF_MAPBOX_STYLE = 'mapbox://styles/trundev/ckpn5fzfm05zk17rfj1oz6j2t'
MAPBOX_ACCESS_TOKEN = 'pk.eyJ1IjoidHJ1bmRldiIsImEiOiJja211ejdmdjMwMDVmMnZucWR0bXAydW5oIn0._cWi8O8hVesaH0m8ZEO1Cw'


# Distance methods for --gradient
DISTANCE_DRAFT = 'draft'
DISTANCE_TM = 'tm'
DISTANCE_GEOD = 'geod'

# Discrete color for seed-island coverage
import plotly.colors
SEED_ID_COLORS = plotly.colors.sequential.Plotly3
SEED_ID_COLORS_NEG = plotly.colors.sequential.Turbo

#
# From ridges.py
# (here 'mgrid_n' is with coordinates at first dimension)
#
def accumulate_by_mgrid(src_arr, mgrid_n_xy, mask=Ellipsis):
    """Accumulate array values into their next points in graph, esp. for graph-nodes"""
    res_arr = np.zeros_like(src_arr)
    src_arr = src_arr[mask]
    # To avoid '+=' overlapping, the accumulation is performed by using unbuffered in place
    # operation, see "numpy.ufunc.at".
    indices = mgrid_n_xy[mask if mask is Ellipsis else (...,mask)]
    np.add.at(res_arr, tuple(indices), src_arr)
    return res_arr

def get_n_num_seeds(mgrid_n_xy, *, leaf_seed_val: int or None=-1):
    """Count number of neighbors of each node-point"""
    # Helper self-pointing array
    mgrid_xy = np.indices(mgrid_n_xy.shape[1:])
    # Mask of self-pointing "seed" pixels
    seed_mask = (mgrid_n_xy == mgrid_xy).all(0)
    # Start with ones at each non-seed pixel
    n_num = np.asarray(seed_mask == 0, dtype=int)
    n_num = accumulate_by_mgrid(n_num, mgrid_n_xy)

    # Mark the leaf-seeds (invalid nodes, except "real" seeds)
    if leaf_seed_val is not None:
        assert leaf_seed_val < 0, f'Invalid leaf_seed_val of {leaf_seed_val}'
        n_num[(n_num == 0) & seed_mask] = leaf_seed_val
    return n_num, seed_mask

def flip_lines(mgrid_n_xy, x_y):
    """Flip all graph-edges along multiple lines"""
    n_xy = mgrid_n_xy[:, *x_y]
    mgrid_n_xy[:, *x_y] = x_y
    while True:
        prev_n_xy = mgrid_n_xy[:, *n_xy]
        mgrid_n_xy[:, *n_xy] = x_y
        mask = (prev_n_xy != n_xy).any(0)
        if not mask.all():
            if not mask.any():
                return mgrid_n_xy
            n_xy = n_xy[:, mask]
            prev_n_xy = prev_n_xy[:, mask]
        x_y = n_xy
        n_xy = prev_n_xy
    return mgrid_n_xy

#
# Figure scatter helpers
#
def add_scatter_points(fig: go.Figure, lonlat_arr: np.array, mask: np.array, text_arr: np.array=None, **scatter_kwargs) -> go.Scattermapbox:
    """Add scatter with markers at each masked point"""
    # Marker text to include x,y coordinates and extra string
    text = np.indices(mask.shape)[:,mask]
    if text.size:
        format = '%d,%d'
        if text_arr is not None:
            text = np.concatenate((text, text_arr[np.newaxis,...]))
            format += ': %s'
        text = np.apply_along_axis(
                lambda xys: np.asarray(format%tuple(xys), dtype=object),
                0, text)
    fig.add_scattermapbox(lon=lonlat_arr[mask][...,0], lat=lonlat_arr[mask][...,1],
                          text=text, **scatter_kwargs)
    return fig.data[-1]

def add_scatter_lines(fig: go.Figure, lonlat_arr_list: list[np.array], **scatter_kwargs) -> go.Scattermapbox:
    """Add scatter with array of arrows/lines between fixed number of points"""
    # Make individual lines by adding gaps: (start-point, end-point, nan)
    lines_arr = np.stack(np.broadcast_arrays(*lonlat_arr_list, np.nan))
    lines_arr = lines_arr.T.reshape(lines_arr.shape[-1], -1)
    #lines_arr = lines_arr.T.reshape(lonlat_start.shape[-1], -1)
    fig.add_scattermapbox(lon=lines_arr[0], lat=lines_arr[1], **scatter_kwargs)
    return fig.data[-1]

def add_scatter_mgrid_n(fig: go.Figure, lonlat_arr: np.array, mgrid_n_xy: np.array, len_scale: float=None,
    text_arr: np.array=None, **scatter_kwargs) -> go.Scattermapbox:
    """Add scatter with arrows (graph-edges) toward neighbors"""
    lines_arr = lonlat_arr[tuple(mgrid_n_xy)]
    # Filter-out zero lines
    mask = (mgrid_n_xy != np.indices(mgrid_n_xy.shape[1:])).any(0)
    lonlat_arr = lonlat_arr[mask]
    lines_arr = lines_arr[mask]
    if text_arr is not None:
        # Each line produces 3 points, place text at middle one
        text_arr = np.stack(np.broadcast_arrays(None, text_arr, None))[:,mask].T.flat
    # Rescale line lengths
    if len_scale is not None:
        lines_arr = lines_arr * len_scale + lonlat_arr * (1 - len_scale)
    return add_scatter_lines(fig, (lonlat_arr, lines_arr), text=text_arr, **scatter_kwargs)

def get_gradient_mgrid(altitude: np.array, *, distance: gdal_utils.tm_distance or None) -> list[np.array, np.array]:
    """Generate gradient mgrid_n"""
    mgrid = np.indices(altitude.shape)
    grad_n = mgrid.copy()
    slope_max = np.zeros_like(altitude)
    for sl_left, sl_right in [
            # along x
            [(slice(0, -1), ...), (slice(1, None), ...)],
            [(slice(1, None), ...), (slice(0, -1), ...)],
            # along y
            [(..., slice(0, -1)), (..., slice(1, None))],
            [(..., slice(1, None)), (..., slice(0, -1))],
            # along x\y
            [(slice(0, -1), slice(0, -1)), (slice(1, None), slice(1, None))],
            [(slice(1, None), slice(1, None)), (slice(0, -1), slice(0, -1))],
            # along x/y
            [(slice(0, -1), slice(1, None)), (slice(1, None), slice(0, -1))],
            [(slice(1, None), slice(0, -1)), (slice(0, -1), slice(1, None))],
        ]:
        slope = altitude[sl_right] - altitude[sl_left]
        # Use actual slope arctan(vert / hor)
        if distance is not None:
            # Distances between each point 'sl_left' and its neighbor at the 'sl_right' side
            dist = distance.get_distance(mgrid[:, *sl_right].T, mgrid[:, *sl_left].T, flat=True).T
            slope = np.arctan2(slope, dist)

        mask = slope_max[sl_left] < slope
        slope_max[sl_left][mask] = slope[mask]
        grad_n[:, *sl_left][:, mask] = mgrid[:, *sl_right][:, mask]
    return grad_n, slope_max

def get_seed_ids(mgrid_n: np.array, *, none_id: int,
        leaf_seed_id: int or None=None, boundary_id: int or None=None,
        base_id: int=0) -> list[np.array, np.array]:
    """Assign IDs to each seed island, start from `base_id`, `none_id`/`leaf_seed_id` for no-seed(loop)/leaf-seeds"""
    assert none_id < base_id, f'none_id {none_id} must be below base_id {base_id}'
    n_num, seed_mask = get_n_num_seeds(mgrid_n, leaf_seed_val=leaf_seed_id)
    pend_mask = ~seed_mask

    # Points that can never reach a "seed" (loops) will have 'none_id'
    seed_ids = np.full(n_num.shape, none_id)
    # Separate leaf-seeds from real-seeds
    if leaf_seed_id is not None:
        assert leaf_seed_id < base_id, f'leaf_seed_id {leaf_seed_id} must be below base_id {base_id}'
        # 'n_num' is 'leaf_seed_id' at self-pointing leafs
        seed_ids[n_num == leaf_seed_id] = leaf_seed_id
        seed_mask[n_num == leaf_seed_id] = False
    # Identify boundary islands
    bound_seed_mask = False
    if boundary_id is not None:
        # Assign 'boundary_id' to all boundary points
        mask = np.ones_like(seed_mask)
        mask[np.s_[1:-1,] * seed_mask.ndim] = False
        seed_ids[mask] = boundary_id
        bound_seed_mask = seed_mask.copy()
        seed_mask[mask] = False
        bound_seed_mask[~mask] = False
        # Assign unique IDs to all real-seeds at the boundaries
        assert boundary_id < none_id and (leaf_seed_id is None or boundary_id < leaf_seed_id), \
                f'boundary_id {boundary_id} must be below none_id {none_id} and leaf_seed_id {leaf_seed_id}'
        seed_ids[bound_seed_mask] = boundary_id - np.arange(np.count_nonzero(bound_seed_mask)) - 1

    seed_ids[seed_mask] = np.arange(np.count_nonzero(seed_mask)) + base_id
    # Expand the seed IDs till all non-seeds got ID
    while pend_mask.any():
        seed_ids[pend_mask] = seed_ids[tuple(mgrid_n[:, pend_mask])]
        if (seed_ids[pend_mask] == none_id).all():
            print(f'Warning: Detected {np.count_nonzero(pend_mask)} points in no-seed branches (loops)')
            break
        pend_mask[pend_mask] = pend_mask[tuple(mgrid_n[:, pend_mask])]

    np.testing.assert_equal(seed_ids[seed_mask], np.arange(np.count_nonzero(seed_mask)) + base_id,
            err_msg='Unexpected order of seed IDs')
    return seed_ids, n_num == 0, seed_mask | bound_seed_mask

def seed_ids_2_seed_xy(seed_ids: np.array, seed_mask: np.array) -> np.array:
    """Map between ID and seed coordinates, also works for negatives (boundary seeds)"""
    res_size = max(seed_ids.max() + 1, 0)
    if seed_ids.min() < 0:
        res_size -= seed_ids.min()
    # The "max()" is This is to provoke "out of bounds" for unknown IDs
    seed_xy = np.full((seed_mask.ndim, res_size), max(seed_mask.shape))
    seed_xy[:, seed_ids[seed_mask]] = np.indices(seed_mask.shape)[:, seed_mask]

    min_max = [seed_ids[seed_mask].min(), seed_ids[seed_mask].max()]
    np.testing.assert_equal(seed_ids[*seed_xy[:, min_max]], min_max, err_msg='Possible ID to xy map overlap')
    return seed_xy

# Helper index arrays (2x9 and 2x8)
NEIGHBORS = np.arange(3) - 1
NEIGHBORS_SELF = np.stack(np.broadcast_arrays(NEIGHBORS, NEIGHBORS[:,np.newaxis])).reshape(2, -1)
# Drop the "self" entry
NEIGHBORS = NEIGHBORS_SELF[:,(NEIGHBORS_SELF != 0).any(0)]

def get_gradient_mgrid_new(altitude: np.array, *, distance: gdal_utils.tm_distance or None) -> list[np.array, np.array]:
    """Generate gradient mgrid_n"""
    grad_xy = np.indices(altitude.shape)
    # Work on the "internal" points only
    int_slice = np.s_[1:-1,] * altitude.ndim
    base_xy = grad_xy[:, np.newaxis, *int_slice]
    neighbor_xy = (base_xy.T + NEIGHBORS.T).T

    slope = altitude[tuple(neighbor_xy)] - altitude[tuple(base_xy)]
    # Use actual slope arctan(vert / hor)
    if distance is not None:
        # Distances between each point and all its neighbors
        base_xy = np.broadcast_to(base_xy, shape=neighbor_xy.shape)
        dist = distance.get_distance(neighbor_xy.T, base_xy.T, flat=True).T
        slope = np.arctan2(slope, dist)

    # Gradient at max positive slope (this also excludes NaN-s)
    mask = (slope > 0).any(0)
    argmax = np.nanargmax(slope[:, mask], axis=0, keepdims=True)
    base_xy = np.take_along_axis(neighbor_xy[..., mask], argmax[np.newaxis, ...], axis=1)

    # Extract selected slope
    slope[0, mask] = np.take_along_axis(slope[:, mask], argmax, axis=0)
    slope[0, ~mask] = 0
    slope = np.pad(slope[0], 1, constant_values=0)

    # Combine result from "internal" points
    mask = np.pad(mask, 1, constant_values=False)
    grad_xy[:, mask] = base_xy[:, 0, ...]
    return grad_xy, slope

def isolate_borders(mgrid_n: np.array, seed_ids: np.array) -> np.array:
    """Obtain mask of points where any of neighbors have different seed ID"""
    data_shape = mgrid_n.shape[1:]
    # Get neighbor coordinates / IDs of all internal point (skip boundary ones)
    base_xy = np.indices(np.asarray(data_shape) - 2) + 1
    neighbor_xy = (base_xy[:, np.newaxis, ...].T + NEIGHBORS.T).T
    neighbor_ids = seed_ids[tuple(neighbor_xy)]

    # Get mask for inner points, pad boundary ones with 'True'
    res_mask = (seed_ids[(slice(1,-1),) * seed_ids.ndim] != neighbor_ids).any(0)
    res_mask = np.pad(res_mask, 1, constant_values=True)
    return res_mask

def join_seed_islands(altitude: np.array, mgrid_n: np.array, *,
        iterations: int or True=True) -> np.array:
    """Connect neighbor-grid islands along the highest adjacent leafs"""
    # Special seed IDs
    SEED_ID_NONE = -1       # Unreachable or invalid
    SEED_ID_BOUND = -2      # Boundary (outside) points
    seed_ids, leaf_mask, seed_mask = get_seed_ids(mgrid_n,
            none_id=SEED_ID_NONE, leaf_seed_id=SEED_ID_NONE)
    # Use both leaves asn seeds as join ends
    leaf_mask |= seed_mask

    #
    # Isolate leaf neighbor coordinates, altitude and seed IDs
    #
    leaf_xy = np.asarray(np.nonzero(leaf_mask))
    neighbor_xy = leaf_xy[:, np.newaxis, :] + NEIGHBORS[..., np.newaxis]
    # Isolate out-of-boundary points
    bound_mask = (neighbor_xy < 0).any(0) | (neighbor_xy.T >= altitude.shape).T.any(0)

    # Altitudes of all neighbors of each leaf, the boundary ones must be processed with priority
    neighbor_alts = np.full_like(altitude, np.inf, shape=neighbor_xy.shape[1:])
    neighbor_alts[~bound_mask] = altitude[tuple(neighbor_xy[:, ~bound_mask])]

    # Seed IDs of all neighbors of each leaf, -2 for boundary (outsiders)
    neighbor_ids = np.full_like(seed_ids, SEED_ID_BOUND, shape=neighbor_xy.shape[1:])
    neighbor_ids[~bound_mask] = seed_ids[tuple(neighbor_xy[:, ~bound_mask])]

    #
    # Prepare leafs altitude and seed IDs
    #
    leaf_alts = altitude[leaf_mask]
    leaf_ids = seed_ids[leaf_mask]
    del leaf_mask, seed_ids, bound_mask

    for _ in iter(bool, True) if iterations is True else range(iterations):
        # Drop "internal" leaves (surrounded by the same seed ID or invalid)
        mask = ((leaf_ids != neighbor_ids) & (neighbor_ids != SEED_ID_NONE)).any(0)
        if not mask.any():
            # All points are processed
            break
        leaf_ids = leaf_ids[mask]
        leaf_alts = leaf_alts[mask]
        leaf_xy = leaf_xy[:, mask]
        neighbor_ids = neighbor_ids[:, mask]
        neighbor_alts = neighbor_alts[:, mask]
        neighbor_xy = neighbor_xy[..., mask]

        # The neighbors of the same seed IDs must be ignored when processing
        neighbor_alts[leaf_ids == neighbor_ids] = -np.inf

        #
        # Process arrays based on the lowest altitude between each leaf and its highest neighbor
        # Note:
        # This is almost always the same as leaf altutude
        #
        alt_minmax = np.min((np.nanmax(neighbor_alts, 0), leaf_alts), 0)

        # Select the max-altitude leaf, neighbor pair. Merge islands
        pair_idx = np.argmax(alt_minmax)
        pair_idx = np.nanargmax(neighbor_alts[:, pair_idx]), pair_idx

        l_xy = leaf_xy[:, pair_idx[1]]
        l_id = leaf_ids[pair_idx[1]]
        n_xy = neighbor_xy[:, *pair_idx]
        n_id = neighbor_ids[pair_idx]
        print(f'  Merging {l_xy} (island {l_id}) into {n_xy} (island {n_id}), altitude {alt_minmax[pair_idx[1]]}')
        flip_lines(mgrid_n, l_xy[:, np.newaxis])
        if (n_xy >= 0).all() and (n_xy < altitude.shape).all():
            mgrid_n[:, *l_xy] = n_xy

        # Replace IDs of merged island (this makes more leaves "internal")
        if l_id == SEED_ID_BOUND:
            # The "boundary" ID must persist, to prevent merge between boundary islands
            l_id, n_id = n_id, l_id
        neighbor_ids[neighbor_ids == l_id] = n_id
        leaf_ids[leaf_ids == l_id] = n_id

        # Drop the processed pair
        leaf_ids = np.delete(leaf_ids, pair_idx[1])
        leaf_alts = np.delete(leaf_alts, pair_idx[1])
        leaf_xy = np.delete(leaf_xy, pair_idx[1], axis=-1)
        neighbor_ids = np.delete(neighbor_ids, pair_idx[1], axis=-1)
        neighbor_alts = np.delete(neighbor_alts, pair_idx[1], axis=-1)
        neighbor_xy = np.delete(neighbor_xy, pair_idx[1], axis=-1)

    return mgrid_n

def join_seed_islands_new(altitude: np.array, mgrid_n: np.array, *,
        iterations: int or True=True) -> np.array:
    """Connect neighbor-grid islands along the highest adjacent leafs"""
    # Special seed IDs
    SEED_ID_NONE = -1       # Unreachable or invalid
    SEED_ID_BOUND = -2      # Boundary (outside) points
    seed_ids, _, _ = get_seed_ids(mgrid_n,
            none_id=SEED_ID_NONE, leaf_seed_id=SEED_ID_NONE, boundary_id=SEED_ID_BOUND)

    #
    # Process first the "internal" islands only
    #
    # Start with all non-boundary points, isolate neighbors
    base_xy = np.asarray(np.nonzero(seed_ids >= 0))
    neighbor_xy = base_xy[:, np.newaxis, :] + NEIGHBORS[..., np.newaxis]
    np.testing.assert_equal(neighbor_xy >= 0, True, err_msg='Out of boundary neighbour')
    np.testing.assert_equal((neighbor_xy.T < altitude.shape).T, True, err_msg='Out of boundary neighbour')

    # Retrieve IDs/altitudes to avoid constant update of 'seed_ids'
    base_ids = seed_ids[tuple(base_xy)]
    neighbor_ids = seed_ids[tuple(neighbor_xy)]
    neighbor_alts = altitude[tuple(neighbor_xy)]

    for _ in iter(bool, True) if iterations is True else range(iterations):
        # The neighbors of the same seed IDs are ignored by lowering its altitude
        # If all are such (island internal point), the point is removed
        mask = (base_ids == neighbor_ids) | (neighbor_ids == SEED_ID_NONE)
        neighbor_alts[mask & np.isfinite(neighbor_alts)] = -np.inf      # Keep NaNs (why-not)
        mask = ~mask.all(0)

        # Drop boundary islands ("internal" just merged to a boundary one)
        mask &= base_ids >= 0
        if not mask.any():
            # All points are processed
            break
        base_xy = base_xy[:, mask]
        neighbor_xy = neighbor_xy[..., mask]
        base_ids = base_ids[mask]
        neighbor_ids = neighbor_ids[:, mask]
        neighbor_alts = neighbor_alts[:, mask]

        #
        # Process arrays based on the lowest altitude between each base and its highest neighbor
        # The neighbors of the same seed IDs must be ignored when processing
        #
        alt_minmax = np.nanmax(neighbor_alts, 0)
        alt_minmax = np.min((alt_minmax, altitude[tuple(base_xy)]), 0)

        # Select the max-altitude leaf, neighbor pair. Merge islands
        pair_idx = np.argmax(alt_minmax)
        pair_idx = np.nanargmax(neighbor_alts[:, pair_idx]), pair_idx

        b_xy = base_xy[:, pair_idx[1]]
        n_xy = neighbor_xy[:, *pair_idx]
        b_id = base_ids[pair_idx[1]]
        n_id = neighbor_ids[pair_idx]
        print(f'  Merging {b_xy} (island {b_id}/{seed_ids[*b_xy]}) into {n_xy} (island {n_id}/{seed_ids[*n_xy]}), altitude {alt_minmax[pair_idx[1]]}')
        flip_lines(mgrid_n, b_xy[:, np.newaxis])
        mgrid_n[:, *b_xy] = n_xy

        # Replace IDs of merged island (this makes more leaves "internal")
        base_ids[base_ids == b_id] = n_id
        neighbor_ids[neighbor_ids == b_id] = n_id

    return mgrid_n

def plot_figure(fig: go.Figure, dem_band: gdal_utils.gdal_dem_band, mgrid_n_list: list) -> None:
    """Create figure plot"""
    indices = np.moveaxis(np.indices(dem_band.shape), 0, -1)
    lla_arr = dem_band.xy2lonlatalt(indices)
    valid_mask = np.isfinite(lla_arr[...,2])

    # Markers at each valid grid-point
    altitude = lla_arr[valid_mask][...,2]
    data = add_scatter_points(fig, lla_arr, valid_mask,
                              # Show altitude
                              text_arr=altitude, marker_color=altitude,
                              mode='markers',
                              name='DEM')
    res = data,

    # Visualize node-graphs from 'mgrid_n_list'
    for idx, (name, mgrid_n, *info) in enumerate(mgrid_n_list):
        print(f'{idx}: {name}')
        # Arrows (graph-edges) toward neighbors (cut lines half-way)
        data = add_scatter_mgrid_n(fig, lla_arr, mgrid_n, len_scale=.5,
                                   mode='lines', text_arr=info[0] if info else None,
                                   name=f'mgrid_n', legendgroup=idx, legendgrouptitle_text=name)
        res = *res, data

        # Seed island identification
        seed_ids, leaf_mask, seed_mask = get_seed_ids(
                mgrid_n, none_id=-1, leaf_seed_id=-2, boundary_id=-3)
        border_mask = isolate_borders(mgrid_n, seed_ids)
        border_mask &= valid_mask   # Hide "NoData" altitudes
        all_mask = border_mask | leaf_mask | seed_mask
        # Map between ID and seed coordinates, also works for negatives (boundary seeds)
        seed_xy = seed_ids_2_seed_xy(seed_ids, seed_mask)
        def get_str(xy):
            """Point info, to include seed ID and its coordinates"""
            sid = seed_ids[tuple(xy)]
            text = '[Leaf] ' if leaf_mask[tuple(xy)] else ''
            if seed_mask[tuple(xy)]:
                text += f'[Seed, coverage {np.count_nonzero(sid == seed_ids)}] '
            if sid == -1:
                text += f'None'
            elif sid == -2:
                text += f'Leaf-seed'
            elif sid == -3:
                text += f'Boundary'
            else:
                text += f'Seed {sid} at {seed_xy[:, sid]}'
            return np.asarray(text, dtype=object)
        text_arr = np.apply_along_axis(get_str, 0, np.asarray(np.nonzero(all_mask)))
        del seed_xy
        # Colorize (separate color-scale for positive/negative IDs)
        color_arr = np.choose(seed_ids[border_mask], SEED_ID_COLORS, mode='wrap')
        mask = seed_ids[border_mask] < 0
        color_arr[mask] = np.choose(seed_ids[border_mask][mask], SEED_ID_COLORS_NEG, mode='wrap')
        data = add_scatter_points(fig, lla_arr, border_mask,
                                  mode='markers', text_arr=text_arr[border_mask[all_mask]],
                                  marker=dict(symbol='circle', color=color_arr),
                                  name=f'Seed-islands', legendgroup=idx)
        res = *res, data

        # Leafs
        print(f'  Leafs: {np.count_nonzero(leaf_mask)}')
        data = add_scatter_points(fig, lla_arr, leaf_mask,
                                  mode='markers', text_arr=text_arr[leaf_mask[all_mask]],
                                  marker=dict(symbol='circle'),
                                  name=f'Leafs', legendgroup=idx)
        res = *res, data
        # Real-seeds (self-pointing, but not leafs)
        print(f'  Seeds: {np.count_nonzero(seed_mask)} (non-boundary {np.count_nonzero(seed_ids[seed_mask] >= 0)}),'
              f' self-pointing: {np.count_nonzero(seed_mask | (seed_ids==-2))}')
        data = add_scatter_points(fig, lla_arr, seed_mask,
                                  mode='markers', text_arr=text_arr[seed_mask[all_mask]],
                                  marker=dict(symbol='circle'),
                                  name=f'Seeds', legendgroup=idx)
        res = *res, data

        # Nodes
        n_num, _ = get_n_num_seeds(mgrid_n)
        node_mask = n_num > 1
        n_num_masked = n_num[node_mask]
        print(f'  Nodes: {n_num_masked.size}, max: {n_num_masked.max()}')
        data = add_scatter_points(fig, lla_arr, node_mask,
                                  text_arr=n_num_masked,
                                  mode='markers', marker=dict(symbol='circle', size=4*n_num_masked),
                                  name=f'Nodes', legendgroup=idx)
        res = *res, data

        # Straight-lines between non-leaf nodes
        start_xy = np.nonzero(node_mask)
        next_xy = mgrid_n[:,*start_xy]
        # Mean value (skip start/end points)
        mean_lla = np.zeros_like(lla_arr[start_xy])
        mean_cnt = np.zeros(mean_lla.shape[:1], dtype=int)
        # "Cut" the grid at nodes, where to stop traversing
        mgrid_tmp = mgrid_n.copy()
        mgrid_tmp[:,node_mask] = np.indices(node_mask.shape)[:,node_mask]
        while True:
            prev_xy = next_xy
            next_xy = mgrid_tmp[:,*next_xy]
            # Check if there is any change
            mask = (prev_xy != next_xy).any(0)
            mean_lla[mask] += lla_arr[tuple(prev_xy[:,mask])]
            mean_cnt[mask] += 1
            if not mask.any():
                break
        # Make the middle points (mean value)
        mask = mean_cnt == 0
        mean_lla[mask] = lla_arr[start_xy][mask]
        mean_cnt[mask] = 1
        mean_lla /= mean_cnt[:,np.newaxis]
        data = add_scatter_lines(fig, (lla_arr[start_xy], mean_lla, lla_arr[tuple(next_xy)]),
                                 mode='lines',
                                 name=f'Node-edges', legendgroup=idx)
        res = *res, data

    return res

def main(args):
    """Main finction"""
    # Load input files
    print(f'Loading DEM: "{args.dem_file}"')
    dem_band = gdal_utils.dem_open(args.dem_file)
    dem_band.load()

    mgrid_n_list = []
    # Generate gradient
    if args.gradient:
        distance = gdal_utils.geod_distance(dem_band) if args.gradient == DISTANCE_GEOD \
                else gdal_utils.tm_distance(dem_band) if args.gradient == DISTANCE_TM \
                else gdal_utils.draft_distance(dem_band) if args.gradient == DISTANCE_DRAFT \
                else None
        altitude = dem_band.get_elevation(True)
        mgrid_n, slope = get_gradient_mgrid_new(altitude, distance=distance)
        # Show slope / altitude difference
        if distance is None:
            format = '%d m'
        else:
            format = '%d deg'
            slope = np.rad2deg(slope)
        slope = np.vectorize(lambda v: format%v, otypes=[object])(slope)
        mgrid_n_list.append(('Gradient', mgrid_n, slope))

        if args.merge_islands is not False:
            mgrid_n_list.append(('Gradient-joined',
                   join_seed_islands(altitude, mgrid_n.copy(), iterations=args.merge_islands)))
            mgrid_n_list.append(('Gradient-joined [new]',
                    join_seed_islands_new(altitude, mgrid_n.copy(), iterations=args.merge_islands)))

    # Load neighbor-grids
    if args.mgrid_n is not None:
        for fname in args.mgrid_n:
            print(f'Loading neighbor-grid: "{args.mgrid_n}"')
            mgrid_n = np.load(fname)
            assert dem_band.shape == mgrid_n.shape[:-1], \
                    f'DEM vs neighbor-grid shape mismatch: {dem_band.shape}, {mgrid_n.shape[:-1]}'
            # Move the coordinates into the first dimension
            # (as in numpy convention)
            mgrid_n = np.moveaxis(mgrid_n, -1, 0)
            mgrid_n_list.append((os.path.basename(fname), mgrid_n))

    # Obtain boundaries
    bounds = np.asarray([(0,0), dem_band.shape])
    bounds[1] -= 1
    bounds = dem_band.xy2lonlatalt(bounds)
    print(f'DEM boundaries: {dem_band.shape}:')
    print(f'  Upper Left : {bounds[0]}')
    print(f'  Lower Right: {bounds[1]}')

    # Greate plotly figure
    fig = go.Figure()

    plot_figure(fig, dem_band, mgrid_n_list)

    # Select zoom and center
    zoom = (bounds[1] - bounds[0])[:2].max()
    zoom = np.log2(360 / zoom)  # zoom 0 is 360 degree wide
    center = bounds.mean(0)
    fig.update_layout({
            'showlegend': True,
            'legend': dict(
                x=0, y=1,   # Top-left overlapping
                groupclick='toggleitem',
            ),
            'mapbox': dict(
                accesstoken=MAPBOX_ACCESS_TOKEN,
                style=args.mapbox_style,
                center={'lon': center[0], 'lat': center[1]},
                zoom=zoom,
            ),
            'geo': dict(
                fitbounds='geojson',
                center={'lon': center[0], 'lat': center[1]},
                resolution=50,
                showrivers=True,
                showlakes=True,
                showland=True,
            ),
        })
    # Experimental: menus
    fig.update_layout(updatemenus=[
                dict(buttons=list([
                            dict(
                                args=["type", "scattermapbox"],
                                label="Scatter Mapbox",
                                method="restyle"
                            ),
                            dict(
                                args=["type", "scattergeo"],
                                label="Scatter Geo",
                                method="restyle"
                            )
                        ]),
                        x=1,
                    ),
            ])
    fig.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Terrain ridges visualization')
    parser.add_argument('dem_file',
            help='Input DEM file, formats supported by https://gdal.org')
    parser.add_argument('--mgrid-n', action='append', nargs='?',
            help='Neighbor grid file (intermediate terrain_ridges), numpy.save() format.'
            f' If empty, append "{DEF_MGRID_N_POSTFIX}" to "dem_file"')
    parser.add_argument('--mapbox-style', default=DEF_MAPBOX_STYLE,
            help=f'Mapbox layout style, default: "{DEF_MAPBOX_STYLE}"')
    parser.add_argument('--gradient', nargs='?',
            choices=[DISTANCE_DRAFT, DISTANCE_TM, DISTANCE_GEOD], const=True,
            help=f'Generate gradient mgrid-n, specify distance calculation method to use actual slope')
    # args.merge_islands: False - no, True - all, <int> - limited iterations
    parser.add_argument('--merge-islands', nargs='?', type=int, const=True, default=False,
            help=f'Generate gradient mgrid-n, specify distance calculation method to use actual slope')
    args = parser.parse_args()

    # Apply '--mgrid-n' default
    if args.mgrid_n and (None in args.mgrid_n):
        args.mgrid_n[args.mgrid_n.index(None)] = args.dem_file + DEF_MGRID_N_POSTFIX
        if None in args.mgrid_n:
            parser.exit(255, 'Error: Empty "--mgrid-n" option was used more than once')

    res = main(args)
    if res:
        exit(res)
