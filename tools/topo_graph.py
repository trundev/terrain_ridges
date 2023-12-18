"""Topology graph implementation


"""
import sys
import pickle
import numpy as np


# Minimum node-weight to keep in a layer
LAYER0_MIN_WEIGHT = 10
EDGE_REUSE = True


#
# Generic tree-graph manipulations
#
def assert_graph_shape(tgt_nodes):
    """Validate shape and indices of a tree-graph representation"""
    assert tgt_nodes.shape[0] == tgt_nodes.ndim-1, 'Incorrect graph representation array shape'
    np.testing.assert_equal(tgt_nodes.shape[1:] > tgt_nodes.max(tuple(range(1, tgt_nodes.ndim))), True,
            f'Graph index {tgt_nodes.max(tuple(range(1, tgt_nodes.ndim)))} is out of bounds {tgt_nodes.shape[1:]}')
    np.testing.assert_equal(np.negative(tgt_nodes.shape[1:]) <= tgt_nodes.min(tuple(range(1, tgt_nodes.ndim))), True,
            f'Negative graph index {tgt_nodes.min(tuple(range(1, tgt_nodes.ndim)))} is out of bounds {np.negative(tgt_nodes.shape[1:])}')

def reshape_graph(tgt_nodes: np.array, shape: list[int], *, strict_unravel: bool=False) -> np.array:
    """Reshape tree-graph, keeping indices valid"""
    assert_graph_shape(tgt_nodes)
    result = np.asarray(np.indices(shape))
    src_shape = tgt_nodes.shape[1:]

    if not strict_unravel:
        # Regular (simple/faster) way
        result = result.reshape(result.shape[0], *src_shape)
        tgt_nodes = tgt_nodes.reshape(tgt_nodes.shape[0], *shape)
        return result[:, *tgt_nodes]

    #
    # To ensure unravel_index() / ravel_multi_index() compatibility
    #
    if (tgt_nodes < 0).any():   # Workaround unsupported negative indices
        tgt_nodes = (tgt_nodes.T % src_shape).T
    serialize = range(tgt_nodes[0].size)
    # Serialize 'tgt_nodes' from 'src_shape'
    tgt_nodes = tgt_nodes[:, *np.unravel_index(serialize, shape=src_shape)]
    # Convert indices: 'src_shape' to 'shape'
    tgt_nodes = np.unravel_index(np.ravel_multi_index(tgt_nodes, src_shape), shape=shape)
    # Deserialize into 'shape'
    result[:, *np.unravel_index(serialize, shape=shape)] = tgt_nodes
    return result

def mask_graph(tgt_nodes: np.array, mask: np.array) -> np.array:
    """Mask tree-graph by flattening to 1D, out-of-mask indices become self-pointing"""
    assert_graph_shape(tgt_nodes)
    mask = np.broadcast_to(mask, tgt_nodes.shape[1:])
    result = np.full(fill_value=-1, shape=tgt_nodes.shape[1:])
    # Flat indices template
    flat_idxs = np.arange(np.count_nonzero(mask))
    result[mask] = flat_idxs
    result = result[*tgt_nodes][mask]
    # Out-of-mask indices becomes self-pointing
    result = np.where(result < 0, flat_idxs, result)
    # Make valid tree-graph shape
    return result[np.newaxis, ...]

def merge_graphtrees(tgt_nodes: np.array, edge_list: np.array, *, keep_args: bool=True) -> (np.array, np.array):
    """Merge individual trees inside a graph by replacing a node target, then flip the chain of its targets"""
    assert_graph_shape(tgt_nodes)
    assert edge_list.shape[0] == tgt_nodes.ndim-1, 'Edge and graph dimensions mismatch'
    assert edge_list.shape[1] == 2, 'Incorrect edge representation array shape'
    if keep_args:
        tgt_nodes = tgt_nodes.copy()
        # 'edge_list' is used as temp-storage between iterations
        edge_list = edge_list.copy()

    flip_mask = np.zeros_like(tgt_nodes[0], dtype=bool)
    mask = np.ones_like(edge_list[0,0], dtype=bool)
    while edge_list.size:
        src_node = edge_list[:, 0].copy()  # Need a copy as this indexing returns a view
        edge_list[:, 0] = tgt_nodes[:, *src_node]
        # Replace target-link with requested node
        tgt_nodes[:, *src_node] = edge_list[:, 1]
        edge_list[:, 1] = src_node
        # Check for reaching a loop
        flip_mask[*src_node] = True
        mask = flip_mask[*edge_list[:, 0]]
        if mask.any():
            edge_list = edge_list[..., ~mask]

    return tgt_nodes, flip_mask

def isolate_graphtrees(tgt_nodes: np.array) -> (np.array, np.array):
    """Identify isolated graph-trees, assing their indices to each node"""
    assert_graph_shape(tgt_nodes)
    node_ids = np.arange(tgt_nodes[0].size).reshape(tgt_nodes.shape[1:])
    pend_mask = np.ones(shape=tgt_nodes[0].shape, dtype=bool)
    # Expand the IDs till stop changing
    init_node_ids = node_ids.copy()
    loop_mask = np.array(False)
    while True:
        # Expand all pendings, but smallest ID inside loops
        node_ids[loop_mask] = np.minimum(node_ids[loop_mask], node_ids[*tgt_nodes[:, loop_mask]])
        node_ids[pend_mask] = node_ids[*tgt_nodes[:, pend_mask]]
        # Detect loop nodes
        if (init_node_ids[loop_mask] == node_ids[loop_mask]).all():
            # No change in any of the loops - done?
            if not pend_mask.any():
                break
            # Reduce pendings
            pend_mask[pend_mask] = pend_mask[*tgt_nodes[:, pend_mask]]
        else:
            # 'init_node_ids' keeps smallest ID inside loops
            init_node_ids[loop_mask] = node_ids[loop_mask]
        # Update loops / pendings
        loop_mask = loop_mask | (init_node_ids == node_ids)
        pend_mask &= ~loop_mask

    # Convert the 'node_ids' to indices by using "return_inverse"
    # It generates array of same size, containing unique indices
    _, node_ids = np.unique(node_ids, return_inverse=True)
    node_ids = node_ids.reshape(pend_mask.shape)
    assert node_ids.min() == 0
    assert np.count_nonzero(loop_mask) > node_ids.max(), f'Graph-trees are more than seed-nodes'
    if np.count_nonzero(loop_mask) > node_ids.max() + 1:
        print('Warning: Detected %d extra/loop nodes in %d graph-trees'%(
                np.count_nonzero(loop_mask) - node_ids.max() - 1, node_ids.max() + 1),
                file=sys.stderr)
    # Final result: graph-tree indices and mask of seed/loop nodes
    return node_ids, loop_mask

def cut_graph_loops(tgt_nodes: np.array, tree_idx: np.array, seed_mask: np.array, *,
        sort_keys: np.array=None, return_copy: bool=True) -> np.array:
    """Cut node-loops inside tree-graph"""
    assert_graph_shape(tgt_nodes)
    # List of 'tgt_nodes' locations to be cut
    cut_idxs = np.asarray(np.nonzero(seed_mask))
    tree_idx = tree_idx[seed_mask]
    # Sort using 'sort_keys', to prioritize unique() selection
    if sort_keys is not None:
        argsort = np.argsort(sort_keys[seed_mask])
        cut_idxs = cut_idxs[:, argsort]
        tree_idx = tree_idx[argsort]
    # Take unique graph-trees only
    _, uniq_idx = np.unique(tree_idx, return_index=True)
    cut_idxs = cut_idxs[:, uniq_idx]
    # Make nodes at 'cut_idxs' self-pointing
    if return_copy:
        tgt_nodes = tgt_nodes.copy()
    tgt_nodes[:, *cut_idxs] = cut_idxs
    return tgt_nodes

#
# Generic graph-edge manipulations
#
def edge_list_to_parents(edge_list: np.array) -> np.array:
    """Obtain parent node map (like isolate_graphtrees() result) from two edge-list layers"""
    assert edge_list.ndim > 2 and edge_list.shape[1] == 2, 'The "edge_list" must be multi layered'
    # Build a 1D graph
    shape = edge_list[0].max() + 1,
    assert edge_list[0].min() >= 0, 'Node IDs are treated as indices'
    node_ids = np.full(shape, -1, dtype=edge_list.dtype)
    node_ids[edge_list[0]] = edge_list[1]
    return node_ids

def edge_list_to_graph(edge_list: np.array, edge_src_mask: np.array) -> np.array:
    """Create a tree-graph from list of edges and corresponding source-node mask"""
    assert edge_list.ndim == 2 and edge_list.shape[0] == 2, 'The "edge_list" must be from single layer'
    # Build a 1D graph
    shape = edge_list.max() + 1,
    assert edge_list.min() >= 0, 'Node IDs are treated as indices'
    tgt_nodes = np.indices(shape)
    # Link the source-node to its target
    tgt_nodes[:, edge_list[edge_src_mask]] = edge_list[::-1][edge_src_mask]
    return tgt_nodes

def find_main_edge_layer(edge_list: np.array) -> np.array:
    """Obtain layer where each edge was consumed"""
    assert edge_list.ndim > 2 and edge_list.shape[1] == 2, 'The "edge_list" must be multi layered'
    return np.argmax(edge_list[:, 0] == edge_list[:, 1], axis=0) - 1

def filter_edge_src_mask(edge_src_mask: np.array, edge_list: np.array) -> np.array:
    """Obtain source-mask for specific layer, need one or two edge-layers"""
    assert edge_src_mask.shape == edge_list[0].shape, 'Mismatch between argument shapes'
    assert edge_list.ndim > 2 and edge_list.shape[1] == 2, 'The "edge_list" must be multi layered'
    # Mask of same-node edges, only first two layers are of interest
    mask = edge_list[:2, 0] == edge_list[:2, 1]
    # Non-last layer - take edges, that collapse in the second layer
    # Last layer - just remove "ghost" edges
    mask[0] = ~mask[0]
    mask = np.logical_and.reduce(mask, axis=0)
    return edge_src_mask & mask

#
# Graph edge selection
#
def get_rel_neighbors(ndim: int, diag_lvl: int=1):
    """All relative indices to a point neighbors"""
    res = 1 - np.indices((3,) * ndim)           # Grid: 1, 0, -1
    res = res[:, res.any(0)]                    # Drop the origin
    res = res[:, :res.shape[1]//2]              # Drop opposites
    assert (res != -res.T[..., np.newaxis]).any(1).all(), 'Opposite directions remain'
    return res[:, np.count_nonzero(res, axis=0) <= diag_lvl+1]  # Drop higher diagonal levels

def build_edge_list(alt_grid: np.array, *, distance: callable or None=None) -> (np.array, np.array, np.array):
    """Get sorted list of all edges between neighbors, also return lexsort() keys for further list-merge"""
    # Obtain neighbors of each point
    src_idx = np.indices(alt_grid.shape)
    src_idx = np.expand_dims(src_idx, axis=1)
    tgt_idx = (src_idx.T + get_rel_neighbors(alt_grid.ndim).T).T
    # Drop the edges, where target-node is out-of-bounds
    mask = (tgt_idx.T < alt_grid.shape).T.all(0)
    mask &= (tgt_idx >= 0).all(0)
    src_idx = np.broadcast_to(src_idx, shape=tgt_idx.shape)[:, mask]
    tgt_idx = tgt_idx[:, mask]
    del mask

    # Get source-node altitude and slope (or just altiture difference)
    edge_list = np.stack((src_idx, tgt_idx), axis=1)
    del src_idx, tgt_idx
    node_alt = alt_grid[*edge_list]
    # Drop the edges, involving NaN
    mask = ~np.isnan(node_alt).any(0)
    if not mask.all():
        node_alt = node_alt[:, mask]
        edge_list = edge_list[..., mask]
    slope = node_alt[1] - node_alt[0]
    node_alt = node_alt.min(0)
    # Use actual slope arctan(vert / hor)
    if distance is not None:
        # Distance between source and target nodes (must be "flat")
        edge_len = distance(edge_list[:,0], edge_list[:,1])
        slope = np.arctan2(slope, edge_len)

    # Swap descending edges (make all slopes positive)
    mask = slope < 0
    edge_list[..., mask] = edge_list[:, ::-1, mask]
    np.testing.assert_equal(alt_grid[*edge_list[:, 0]] <= alt_grid[*edge_list[:, 1]], True,
            'All edges must be acsending')  #TODO: CHECKME: Potential slow operation
    slope[mask] = np.negative(slope[mask])

    # Sort by descending source-node altitude
    # For the same point, the steepest edge must come first
    # > use lexsort() to sort by multiple keys: primary key in last column
    lexsort_keys = np.stack((slope, node_alt))
    alt_lexsort = np.lexsort(lexsort_keys)[::-1]
    edge_list = edge_list[..., alt_lexsort]
    lexsort_keys = lexsort_keys[:, alt_lexsort]
    return edge_list, lexsort_keys

def unique_mask(arr: np.array, axis: int or None=None) -> np.array:
    """Returns the mask of unique elements in an array"""
    _, unique_idx = np.unique(arr, axis=axis, return_index=True)
    result = np.zeros(arr.size if axis is None else arr.shape[axis], dtype=bool)
    result[unique_idx] = True

    return result

def select_edges(edge_list: np.array, *, node_mask: np.array or True=True) -> (
        np.array, (np.array, np.array)):
    """Select the top nodes in a list of edges, to create a tree-graph"""
    assert edge_list.shape[:-1] == (2,), 'The first dimension, must contain left-right nodes'
    # Isolate the first occurrence of each node (transpose to prioritize the edge order)
    edge_src_mask = unique_mask(edge_list.T)
    edge_src_mask = edge_src_mask.reshape(edge_list.shape[::-1]).T
    assert np.count_nonzero(edge_src_mask) == np.unique(edge_list[edge_src_mask]).size, 'Elements from reshaped mask are not unique'

    # Filter-out edges, where source-node is not allowed in 'node_mask'
    if not np.all(node_mask):
        edge_src_mask[edge_src_mask] = node_mask[edge_list[edge_src_mask]]
    # Filter-out edge duplicates
    if not EDGE_REUSE:
        edge_src_mask[1, edge_src_mask.all(0)] = False
    return edge_src_mask

def build_graph_layers(alt_grid: np.array, *, distance: callable or None=None, max_layers: int=5) -> (
        tuple[np.array], list[np.array]):
    """Build edge-list and multiple graph-layters"""
    main_edge_list, _ = build_edge_list(alt_grid, distance=distance)
    # Convert to 1D node-indices
    main_edge_list = np.ravel_multi_index(main_edge_list, alt_grid.shape)
    np.testing.assert_equal(np.nonzero(main_edge_list == np.nanargmax(alt_grid))[0], 1,
            'The highest point was found at source-side of an edge')
    print(f'Build-graph starts with {main_edge_list.shape[-1]} edges')

    #
    # Control selection of nodes as edge's source-node
    # - Weight of each node (like coverage area)
    # - Boundary nodes
    #
    node_weight = np.ones(main_edge_list.max() + 1)
    boundary_mask = np.ones_like(alt_grid, dtype=bool)
    boundary_mask[(slice(1,-1), ) * boundary_mask.ndim] = False
    boundary_mask = boundary_mask.flat[:node_weight.size]
    boundary_mask[...] = False      #TODO: Remove this to block boundary nodes

    #
    # Build each graph layers from previous, start from main-graph
    # (the main-edges source-mask counterpart also tracks "used" entries)
    #
    main_edge_src_mask = np.zeros_like(main_edge_list, dtype=bool)
    main_edge_list = main_edge_list[np.newaxis, ...]
    graph_list = []
    for layer in range(max_layers):
        #
        # Select top unused/ghost edges
        #
        edge_list = main_edge_list[-1][:, ~main_edge_src_mask.any(0)]
        node_mask = node_weight < LAYER0_MIN_WEIGHT * 4**layer
        node_mask &= ~boundary_mask
        edge_src_mask = select_edges(edge_list, node_mask=node_mask)

        #
        # Move "used" edges to main-edges, create graph from them
        # In first iteration this is NOT done, in order these to be dropped (become internal)
        #
        if layer > 0:
            main_edge_src_mask[:, ~main_edge_src_mask.any(0)] = edge_src_mask
        tgt_nodes = edge_list_to_graph(edge_list, edge_src_mask)
        del edge_list, edge_src_mask
        graph_list.append(tgt_nodes)

        #
        # Isolate individual trees in this graph
        #
        tree_idx, seed_mask = isolate_graphtrees(tgt_nodes)
        main_edge_list = np.concatenate((main_edge_list, tree_idx[main_edge_list[-1:]]))
        # All nodes are in single graph-tree
        if tree_idx.max() == 0:
            break

        #
        # Accumulate graph-tree weights and boundary-flags as new nodes
        # Use unbuffered in place operation to avoid '+=' overlapping
        #
        weight = np.zeros_like(node_weight, shape=tree_idx.max() + 1)
        np.add.at(weight, tree_idx, node_weight)
        node_weight = weight
        del weight
        mask = np.zeros_like(boundary_mask, shape=tree_idx.max() + 1)
        np.logical_or.at(mask, tree_idx, boundary_mask)
        boundary_mask = mask

        #
        # Drop internal (same-node) and duplicated edges
        # Keep the top edges, as sorted by build_edge_list()
        #
        edge_list = main_edge_list[-1][:, ~main_edge_src_mask.any(0)]
        edge_mask = edge_list[0] != edge_list[1]
        # Select top edge from each duplicate
        # Move smallest node-index in-front to make <a>-<b> and <b>-<a> the same
        edge_mask[edge_mask] = unique_mask(
                np.sort(edge_list[:, edge_mask], axis=0), axis=-1)
        del edge_list
        # Drop selected edges from main-edges
        # the "used" ones remain (if they were removed from 'edge_list')
        mask = main_edge_src_mask.any(0)
        mask[~mask] = edge_mask
        main_edge_list = main_edge_list[..., mask]
        main_edge_src_mask = main_edge_src_mask[:, mask]

    # The layer 0 edge-list and graph are unravelled, other graphs remain 1D
    unravel_edge_list = np.asarray(np.unravel_index(main_edge_list[0], alt_grid.shape))
    # The 'tgt_nodes' are coming from ravel_multi_index(), so use 'strict_unravel'
    graph_list[0] = reshape_graph(graph_list[0], alt_grid.shape, strict_unravel=True)

    ghost_edges = np.count_nonzero(~main_edge_src_mask.any(0))
    print(f'Build-graph ends with {main_edge_list.shape[-1]} ({ghost_edges} ghost) edges in {len(graph_list)} layers')
    return (unravel_edge_list[:, 0], unravel_edge_list[:, 1], main_edge_list, main_edge_src_mask), graph_list

#
# Visualization related
#
def num_node_children(tgt_nodes: np.array) -> np.array:
    """Count the number of children for each graph node"""
    assert_graph_shape(tgt_nodes)
    num_child = np.zeros_like(tgt_nodes[0])
    # Avoid '+=' overlapping, by using unbuffered in place operation, see "numpy.ufunc.at"
    np.add.at(num_child, tuple(tgt_nodes), 1)
    return num_child

def average_seed_pos(node_pos: np.array, tree_idx: np.array, seed_mask: np.array) -> np.array:
    """Average positions of seeds from same graph-tree"""
    num_trees = tree_idx.max() + 1
    node_pos = node_pos[seed_mask]
    center_sum = np.zeros_like(node_pos, shape=(num_trees, node_pos.shape[-1]))
    center_num = np.zeros(shape=(num_trees,))
    indices = tree_idx[seed_mask]
    np.add.at(center_sum, indices, node_pos)
    np.add.at(center_num, indices, 1)
    return (center_sum.T / center_num).T

def get_node_center(node_pos: np.array, tree_idx_seed: iter) -> np.array:
    """Propagate averaged seed positions to higher layer"""
    for tree_idx, seed_mask in tree_idx_seed:
        node_pos = average_seed_pos(node_pos, tree_idx, seed_mask)
    return node_pos

def store_topo_graph(fname: str, graph_list: list[np.array], edge_list: np.array, edge_src_mask: np.array) -> None:
    """Pickle topo-graph representation"""
    # Take all seed-nodes from layer 0
    _, node_mask = isolate_graphtrees(graph_list[0])
    # Add all nodes from edge_list layer 0
    e_list = np.unravel_index(edge_list[0], shape=node_mask.shape)
    node_mask[*e_list] = True

    # Pickle format: key-node positions, graph-list, edge-list, edge-src-mask flags
    with open(fname, 'wb') as f:
        pickle.dump(file=f, obj=dict(
                node_pos_idx=np.nonzero(node_mask),
                node_pos=lla_grid[node_mask],
                graph_list=graph_list,
                edge_list=edge_list,
                edge_src_mask=edge_src_mask,
            ))

def load_topo_graph(fname: str, expand_node_pos: bool=True) -> dict:
    """Unpickle topo-graph representation"""
    with open(fname, 'rb') as f:
        graph_data = pickle.load(f)

    if expand_node_pos and 'node_pos_idx' in graph_data:
        # The 'node_pos' coordinates correspond to 'node_pos_idx' elements, others are unknown
        node_pos = graph_data['node_pos']
        new_pos = np.full_like(node_pos, np.nan, shape=graph_data['graph_list'][0].shape[1:] + node_pos.shape[-1:])
        new_pos[*graph_data.pop('node_pos_idx')] = node_pos
        graph_data['node_pos'] = new_pos

    return graph_data

if __name__ == '__main__':
    import gdal_utils
    USE_DISTANCE = gdal_utils.geod_distance

    dem_name = sys.argv[1]
    dem_band = gdal_utils.dem_open(dem_name)
    dem_band.load()
    indices = np.moveaxis(np.indices(dem_band.shape), 0, -1)
    lla_grid = dem_band.xy2lonlatalt(indices)

    dist = None
    if len(sys.argv) > 2:
        dem_name = sys.argv[2]
        mgrid_old = np.load(dem_name)
        mgrid = np.moveaxis(mgrid_old, -1, 0)
        graph_list = [mgrid]
        edges_info = None
    else:
        if USE_DISTANCE:
            distance = USE_DISTANCE(dem_band)
            dist = lambda src, tgt: distance.get_distance(src.T, tgt.T, flat=True).T
        edges_info, graph_list = build_graph_layers(lla_grid[...,-1], distance=dist)
        if True:
            store_topo_graph('topo_graph.pickle', graph_list, edges_info[2], edges_info[3])

        # Keep layers where each edge is used
        main_edge_layer = find_main_edge_layer(edges_info[2])
        if True:    # Test filter_edge_src_mask()
            for layer in range(edges_info[2].shape[0]):
                np.testing.assert_equal(main_edge_layer == layer,
                        filter_edge_src_mask(edges_info[3], edges_info[2][layer:]).any(0),
                        'Mismatch between find_main_edge_layer() and filter_edge_src_mask()')
        if True:    # Confirm graph-list is obsolete, need only main-graph
            for layer in range(1, min(edges_info[2].shape[0], len(graph_list))):
                edge_list = edges_info[2][layer]
                edge_src_mask = filter_edge_src_mask(edges_info[3], edges_info[2][layer:])
                tgt_nodes = edge_list_to_graph(edge_list, edge_src_mask)
                np.testing.assert_equal(tgt_nodes, graph_list[layer],
                        f'Incorrect graph recalculation for layer {layer}')

        # Combine node-addresses
        main_edge_text = np.apply_along_axis(
                lambda adr: np.asarray('.'.join(map(str, adr)), dtype=object), 0, edges_info[2][::-1])
        main_edge_text = [f'Edge {i}: Node ' for i in range(main_edge_text.shape[-1])] + main_edge_text
        main_edge_text = np.stack(np.broadcast_arrays(*main_edge_text, ''))

    import os.path
    dem_name = os.path.basename(dem_name)

    # Visualize experiments
    import visualize
    def figarg_gen(fig: object, lla_grid: np.array) -> dict:
        """Figure scatters kwargs generator"""
        fig.update_layout({'title_text': f'{dem_name}: {len(graph_list)} layer(s)'})
        # Points from the DEM-file (slow)
        yield visualize.figarg_create_demgrid(lla_grid) | dict(name='DEM', visible='legendonly')

        def create_edges(src_arr, tgt_arr, **kwargs):
            """Figure edge generation"""
            kwargs['name'] += f' ({src_arr.shape[-1]})'
            src_arr = lla_grid[*src_arr]
            tgt_arr = lla_grid[*tgt_arr]
            tgt_arr = (tgt_arr - src_arr) * .7 + src_arr
            if 'text' not in kwargs:
                kwargs['text'] = np.arange(src_arr.size)//3   # 3 points per line
            return visualize.figarg_create_lines((src_arr, tgt_arr)) | kwargs

        # All possible edges
        edge_list, _ = build_edge_list(lla_grid[...,-1], distance=dist)
        yield create_edges(edge_list[:, 0], edge_list[:, 1],
                name='All edges', mode='lines', visible='legendonly')
        # Main-graph
        yield visualize.figarg_create_graph_lines(lla_grid, graph_list[0],
                name='Main-graph') | dict(mode='lines', visible='legendonly')

        # Individual trees inside main-graph
        def create_graphtree_coverage(tree_idx, seed_mask):
            """Figure polygons generation"""
            for id in np.arange(tree_idx.max() + 1):
                mask = tree_idx == id
                num = np.count_nonzero(mask)
                if num <= 10:   #TODO: FIXME: Temporarily drop small regions
                    continue
                yield dict(name=f'Node {id} ({num})', mode='none') | \
                        visualize.figarg_create_mask_polygon(lla_grid, mask)
                mask &= seed_mask
                yield dict(name=f'Seed {id} ({np.count_nonzero(mask)})', mode='lines') | \
                        visualize.figarg_create_mask_line(lla_grid, mask)
        # Trees/seeds for each layer, selected by slider
        slider = visualize.Slider(fig, 0)   # Current layer is the first-one
        tree_idx_seed = []
        # Precache graph-trees (slow operation)
        for tgt_nodes in graph_list:
            t_idx, s_mask = isolate_graphtrees(tgt_nodes)
            print(f'  Processing layer of {tgt_nodes[0].size} nodes: {t_idx.max()+1} in next, {np.count_nonzero(s_mask)} seeds')
            tree_idx_seed.append((t_idx, s_mask))
        # Generate coverage masks
        for layer, (tree_idx, seed_mask) in enumerate(tree_idx_seed):
            print(f'Generating graph-layer {layer}')
            # Visualize links between nodes (upper layer graph)
            tgt_nodes = edge_list_to_graph(edges_info[2][layer + 1],
                    filter_edge_src_mask(edges_info[3], edges_info[2][layer + 1:]))
            lla_layer = get_node_center(lla_grid, tree_idx_seed[:layer + 1])
            yield visualize.figarg_create_graph_lines(lla_layer, tgt_nodes, len_scale=1,
                    name=f'Graph layer {layer + 1}') | dict(mode='lines+markers', marker_symbol='circle', marker_size=15)
            # Visualize coverage of layer nodes
            for t_idx, s_mask in reversed(tree_idx_seed[:layer]):
                tree_idx = tree_idx[t_idx]
                seed_mask = seed_mask[t_idx]
            for s in create_graphtree_coverage(tree_idx, seed_mask):
                yield s
            slider.add_slider_pos(f'Layer {layer + 1}')
        del tree_idx_seed

        # Separate main-edges by the layer, where are in use
        if edges_info:
            for lay in range(main_edge_layer.min(), main_edge_layer.max()+1):
                mask = main_edge_layer == lay
                yield create_edges(edges_info[0][:, mask], edges_info[1][:, mask],
                        name=f'Main-edges {lay}', mode='lines+markers', text=main_edge_text[:, mask].T.flat)

        # Create the slider
        slider.update_layout()

    visualize.figure_show(lla_grid, figarg_gen)
