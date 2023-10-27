"""Topology graph utilities


"""
import sys
from dataclasses import dataclass
import numpy as np


# Minimum node-weight to keep in a layer
LAYER0_MIN_WEIGHT = 10
EDGE_REUSE = True


@dataclass
class Node:
    """Graph node (extended version)"""
    parent_node: int        # Node from the same graph layer, linked to the highest contour point, via 'layer0_link'
                            # (index in node-list)
    layer0_link: int        # Link from the lowest layer, that connects to 'parent_node'
                            # (index in link-list)
    coordinates: tuple[float]   # Longitude and latitude of the node
    enclosing_node: int = 1 # Node from the next layer that, which this one is part of
                            # (index in node-list)

    # Classification parameters
    area_flat: float = np.nan   # Coverage area ignoring altitudes
    area_sloped: float = np.nan # Actual coverage area
    volume: float = np.nan      # Volume based on altitute 0 (mean-altitude = volume / area_flat)
    base_alt: float = np.nan    # Altitude of the 'layer0_link', usually between 'min_alt' and 'max_alt'
    min_alt: float = np.nan     # Minimum altitude of all sub-nodes
    max_alt: float = np.nan     # Maximum altitude of all sub-nodes

@dataclass
class Link:
    """Link between graph nodes (extended version)"""
    nodes: tuple[int]       # Nodes being linked (indices in node-list)
    base_alt: float         # Altitude of the lowest node
    slope: float            # Slope from nodes[1] to nodes[0]
    layer: int              # Layer where the nodes are

    def swap(self):
        """Swap nodes, adjust slope"""
        self.nodes = self.nodes[::-1]
        self.slope = -self.slope

#
# Generic tree-graph manipulations
#
def assert_graph_shape(par_nodes):
    """Validate shape and indices of a tree-graph representation"""
    assert par_nodes.shape[0] == par_nodes.ndim-1, 'Incorrect graph representation array shape'
    np.testing.assert_equal(par_nodes.shape[1:] > par_nodes.max(tuple(range(1, par_nodes.ndim))), True,
            f'Graph index {par_nodes.max(tuple(range(1, par_nodes.ndim)))} is out of bounds {par_nodes.shape[1:]}')
    np.testing.assert_equal(np.negative(par_nodes.shape[1:]) <= par_nodes.min(tuple(range(1, par_nodes.ndim))), True,
            f'Negative graph index {par_nodes.min(tuple(range(1, par_nodes.ndim)))} is out of bounds {np.negative(par_nodes.shape[1:])}')

def reshape_graph(par_nodes: np.array, shape: list[int], *, strict_unravel: bool=False) -> np.array:
    """Reshape tree-graph, keeping indices valid"""
    assert_graph_shape(par_nodes)
    result = np.asarray(np.indices(shape))
    src_shape = par_nodes.shape[1:]

    if not strict_unravel:
        # Regular (simple/faster) way
        result = result.reshape(result.shape[0], *src_shape)
        par_nodes = par_nodes.reshape(par_nodes.shape[0], *shape)
        return result[:, *par_nodes]

    #
    # To ensure unravel_index() / ravel_multi_index() compatibility
    #
    if (par_nodes < 0).any():   # Workaround unsupported negative indices
        par_nodes = (par_nodes.T % src_shape).T
    serialize = range(par_nodes[0].size)
    # Serialize 'par_nodes' from 'src_shape'
    par_nodes = par_nodes[:, *np.unravel_index(serialize, shape=src_shape)]
    # Convert indices: 'src_shape' to 'shape'
    par_nodes = np.unravel_index(np.ravel_multi_index(par_nodes, src_shape), shape=shape)
    # Deserialize into 'shape'
    result[:, *np.unravel_index(serialize, shape=shape)] = par_nodes
    return result

def mask_graph(par_nodes: np.array, mask: np.array) -> np.array:
    """Mask tree-graph by flattening to 1D, out-of-mask indices become self-pointing"""
    assert_graph_shape(par_nodes)
    mask = np.broadcast_to(mask, par_nodes.shape[1:])
    result = np.full(fill_value=-1, shape=par_nodes.shape[1:])
    # Flat indices template
    flat_idxs = np.arange(np.count_nonzero(mask))
    result[mask] = flat_idxs
    result = result[*par_nodes][mask]
    # Out-of-mask indices becomes self-pointing
    result = np.where(result < 0, flat_idxs, result)
    # Make valid tree-graph shape
    return result[np.newaxis, ...]

def isolate_graphtrees(par_nodes: np.array) -> (np.array, np.array):
    """Identify isolated graph-trees, assing their indices to each node"""
    assert_graph_shape(par_nodes)
    node_ids = np.arange(par_nodes[0].size).reshape(par_nodes.shape[1:])
    pend_mask = np.ones(shape=par_nodes[0].shape, dtype=bool)
    # Expand the IDs till stop changing
    init_node_ids = node_ids.copy()
    loop_mask = np.array(False)
    while True:
        # Expand all pendings, but smallest ID inside loops
        node_ids[loop_mask] = np.minimum(node_ids[loop_mask], node_ids[*par_nodes[:, loop_mask]])
        node_ids[pend_mask] = node_ids[*par_nodes[:, pend_mask]]
        # Detect loop nodes
        if (init_node_ids[loop_mask] == node_ids[loop_mask]).all():
            # No change in any of the loops - done?
            if not pend_mask.any():
                break
            # Reduce pendings
            pend_mask[pend_mask] = pend_mask[*par_nodes[:, pend_mask]]
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

def cut_graph_loops(par_nodes: np.array, tree_idx: np.array, seed_mask: np.array, *,
        sort_keys: np.array=None, return_copy: bool=True) -> np.array:
    """Cut node-loops inside tree-graph"""
    assert_graph_shape(par_nodes)
    # List of 'par_nodes' locations to be cut
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
        par_nodes = par_nodes.copy()
    par_nodes[:, *cut_idxs] = cut_idxs
    return par_nodes

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
    base_idx = np.indices(alt_grid.shape)
    base_idx = np.expand_dims(base_idx, axis=1)
    neighbor_idx = (base_idx.T + get_rel_neighbors(alt_grid.ndim).T).T
    # Drop the pairs, where neighbor is out-of-bounds
    mask = (neighbor_idx.T < alt_grid.shape).T.all(0)
    mask &= (neighbor_idx >= 0).all(0)
    base_idx = np.broadcast_to(base_idx, shape=neighbor_idx.shape)[:, mask]
    neighbor_idx = neighbor_idx[:, mask]
    del mask

    # Get base-altitude and slope (or just altiture difference)
    base_alt = alt_grid[*np.stack((base_idx, neighbor_idx), axis=1)]
    slope = base_alt[1] - base_alt[0]
    base_alt = base_alt.min(0)
    # Use actual slope arctan(vert / hor)
    if distance is not None:
        # Distances between each point and all its neighbors
        dist = distance(neighbor_idx, base_idx)
        slope = np.arctan2(slope, dist)

    # Swap descending edges (make all slopes positive)
    mask = slope < 0
    base_idx[:, mask], neighbor_idx[:, mask] = neighbor_idx[:, mask], base_idx[:, mask]
    slope[mask] = np.negative(slope[mask])

    # Sort by descending base-altitude
    # For the same point, the steepest edge must come first
    # > use lexsort() to sort by multiple keys: primary key in last column
    lexsort_keys = np.stack((slope, base_alt))
    alt_lexsort = np.lexsort(lexsort_keys)[::-1]
    base_idx = base_idx[:, alt_lexsort]
    neighbor_idx = neighbor_idx[:, alt_lexsort]
    lexsort_keys = lexsort_keys[:, alt_lexsort]
    return base_idx, neighbor_idx, lexsort_keys

def unique_mask(arr: np.array, axis: int or None=None) -> np.array:
    """Returns the mask of unique elements in an array"""
    _, unique_idx = np.unique(arr, axis=axis, return_index=True)
    result = np.zeros(arr.size if axis is None else arr.shape[axis], dtype=bool)
    result[unique_idx] = True
    #TODO: REMOVEME: Alternative implementation (still not working well)
    if False:
        if axis is None:
            arr = arr.reshape(1, -1)
            axis = -1
        lexsort = np.lexsort(arr, axis=axis)
        arr = arr[..., lexsort]
        mask = (arr != np.roll(arr, 1, axis=axis)).any(0)
        mask[0] = True  # First is always unique, even if matches last (all are the same)
        result2 = np.empty_like(mask)
        result2[..., lexsort] = mask
        np.testing.assert_equal(result2, result)
    return result

def edge_list_to_graph(edge_list: np.array, *, node_mask: np.array or True=True) -> (
        np.array, (np.array, np.array)):
    """Create a tree-graph from list of edges, link each node using its top edge"""
    assert edge_list.shape[:-1] == (2,), 'The first dimension, must contain left-right nodes'
    # Isolate the first occurrence of each node (transpose to prioritize the edge order)
    edge_base_mask = unique_mask(edge_list.T)
    edge_base_mask = edge_base_mask.reshape(edge_list.shape[::-1]).T
    assert np.count_nonzero(edge_base_mask) == np.unique(edge_list[edge_base_mask]).size, 'Elements from reshaped mask are not unique'

    # Filter-out edges, where base-node is not allowed in 'node_mask'
    if not np.all(node_mask):
        edge_base_mask[edge_base_mask] = node_mask[edge_list[edge_base_mask]]
    # Filter-out edge duplicates
    if not EDGE_REUSE:
        edge_base_mask[1, edge_base_mask.all(0)] = False

    # Build a 1D graph
    assert edge_list.min() >= 0, 'Node IDs are treated as indices'
    par_nodes = np.arange(edge_list.max() + 1)

    # Link the base-node to its parent (separately for both edge sides)
    par_nodes[edge_list[edge_base_mask]] = edge_list[::-1][edge_base_mask]
    return par_nodes[np.newaxis, :], edge_base_mask

def get_node_address(node: np.array, graph_list: iter, node_shape: tuple or None=None) -> np.array:
    """Convert node-indices iterating up from bottom 'graph_list' layer"""
    if node_shape is None:
        yield np.asarray(node)
    else:   # 'node' is multi-dimensional, convert to 1D
        yield np.ravel_multi_index(node, node_shape)
        node = tuple(node)
    for par_nodes in graph_list:
        tree_idx, _ = isolate_graphtrees(par_nodes)
        node = tree_idx[node]
        yield node

def find_main_edge_layer(main_edge_list: np.array, graph_list: iter, node_shape: tuple or None=None) -> np.array:
    """Obtain layer where each edge was consumed"""
    edge_layer_list = np.full(main_edge_list.shape[-1], -1)
    for addrs in get_node_address(main_edge_list, graph_list, node_shape):
        mask = addrs[0] != addrs[1]
        edge_layer_list[mask] += 1
    return edge_layer_list

def build_graph_layers(alt_grid: np.array, *, distance: callable or None=None, max_layers: int=5) -> (
        tuple[np.array], list[np.array]):
    """Build edge-list and multiple graph-layters"""
    base_idx, neighbor_idx, _ = build_edge_list(alt_grid, distance=distance)
    # Convert to 1D node-indices
    main_edge_list = np.stack((
            np.ravel_multi_index(base_idx, alt_grid.shape),
            np.ravel_multi_index(neighbor_idx, alt_grid.shape)))
    del base_idx, neighbor_idx
    np.testing.assert_equal(np.nonzero(main_edge_list == np.argmax(alt_grid))[0], 1,
            'The highest point was found at base-side of an edge')
    print(f'Build-graph starts with {main_edge_list.shape[-1]} edges')

    #
    # Take the weight of each node (like coverage area)
    #
    node_weight = np.ones(main_edge_list.max() + 1)
    LAYER0_MIN_WEIGHT = 10

    #
    # Build each graph layers from previous, start from main-graph
    # (main-edges uses "masked_array.mask" to track "used" entries)
    #
    edge_list = main_edge_list
    main_edge_list = np.ma.array(main_edge_list, mask=False)
    main_edge_layer = np.full(main_edge_list.shape[-1], -1)
    graph_list = []
    for layer in range(max_layers):
        #
        # Create graph by using these edges
        #
        node_mask = node_weight < LAYER0_MIN_WEIGHT * 4**layer
        par_nodes, edge_base_mask = edge_list_to_graph(edge_list, node_mask=node_mask)
        # Update 'main_edge_layer'
        mask = ~main_edge_list.mask.any(0)
        mask[mask] = edge_base_mask.any(0)
        main_edge_layer[mask] = layer
        if False:   #TODO: Create 'par_edge_idx' with actual indices from 'edge_main_idx'
            par_edge_idx = np.full((2, par_nodes.shape[-1]), -1)
            mask = par_nodes[0] != np.arange(par_nodes[0].size)     # Drop self-pointing nodes
            #TODO: Warning: possible huge memory comsumption
            argmax = np.argmax(edge_list[edge_base_mask] == np.arange(par_nodes[0].size)[mask, np.newaxis], axis=1)
            edge_use = np.asarray(np.nonzero(edge_base_mask))[:, argmax]
            par_edge_idx[:, mask] = edge_use
            np.testing.assert_equal(edge_list[::-1][*par_edge_idx[:, (par_edge_idx >= 0).all(0)]],
                    par_nodes[0, mask], 'Wrong "par_edge_idx"')
        # Confirm the 'main_edge_list' vs. 'graph_list' consistency
        if True:    #TODO: CHECKME: slow assert operation
            mask = ~main_edge_list.mask.any(0)
            mask[mask] = edge_base_mask.any(0)
            *_, bn_idx = get_node_address(main_edge_list.data[:, mask], graph_list)
            mask = edge_base_mask[:, edge_base_mask.any(0)]     # Cut 'edge_base_mask' to 'bn_idx' shape
            assert np.unique(bn_idx[mask]).size == bn_idx[mask].size, 'Edge base-nodes are NOT unique in current layer'
            np.testing.assert_equal(par_nodes[0, bn_idx[mask]], bn_idx[::-1][mask], 'Base-neighbor nodes do NOT match the graph')
            del bn_idx
        #
        # Move "used" edges to main-edges
        # In first iteration this is NOT done, in order these to be dropped (become internal)
        #
        if layer > 0:
            main_edge_list.mask[:, ~main_edge_list.mask.any(0)] = edge_base_mask
            edge_list = edge_list[:, ~edge_base_mask.any(0)]
        del edge_base_mask

        #
        # Isolate individual trees in this graph
        #
        tree_idx, seed_mask = isolate_graphtrees(par_nodes)
        edge_list = tree_idx[edge_list]
        # All nodes are in single graph-tree
        if tree_idx.max() == 0:
            break

        #
        # Accumulate graph-tree weights as new nodes
        # Use unbuffered in place operation to avoid '+=' overlapping
        #
        weight = np.zeros(tree_idx.max() + 1)
        np.add.at(weight, tree_idx, node_weight)
        node_weight = weight
        del weight

        #
        # Drop internal (same-node) and duplicated edges
        # Keep the top edges, as sorted by build_edge_list()
        #
        edge_mask = edge_list[0] != edge_list[1]
        # Select top edge from each duplicate
        # Move smallest node-index in-front to make <a>-<b> and <b>-<a> the same
        edge_mask[edge_mask] = unique_mask(
                np.sort(edge_list[:, edge_mask], axis=0), axis=-1)
        # Drop selected edges
        edge_list = edge_list[:, edge_mask]
        # Also from main-edges, the "used" ones remain (if they were removed from 'edge_list')
        mask = main_edge_list.mask.any(0)
        mask[~mask] = edge_mask
        main_edge_list = main_edge_list[:, mask]
        main_edge_layer = main_edge_layer[mask]

        graph_list.append(par_nodes)

    # The main edge-list and graph must be unravelled, other graphs remain 1D
    main_edge_list = np.asarray(np.unravel_index(main_edge_list, alt_grid.shape))
    # The 'par_nodes' are coming from ravel_multi_index(), so use 'strict_unravel'
    graph_list[0] = reshape_graph(graph_list[0], alt_grid.shape, strict_unravel=True)

    print(f'Build-graph ends with {main_edge_list.shape[-1]} edges')
    return (main_edge_list[:, 0], main_edge_list[:, 1]), graph_list

#
#TODO: Move to pytest
#
def test_reshape_graph():
    """Test reshape/mask graph"""
    # Reshape 1D to 2D and back
    par_nodes = np.arange(12) + 1
    par_nodes[-1] = 0
    par_nodes = par_nodes[np.newaxis, :]
    new_par_nodes = reshape_graph(par_nodes, shape=(3,4))
    assert new_par_nodes.shape[1:] == (3,4)
    np.testing.assert_equal(reshape_graph(new_par_nodes, shape=(12,)), par_nodes)
    # Reshape 1D to 3D and back
    new_par_nodes = reshape_graph(par_nodes, shape=(2,3,2))
    assert new_par_nodes.shape[1:] == (2,3,2)
    np.testing.assert_equal(reshape_graph(new_par_nodes, shape=(12,)), par_nodes)
    # Reshape 3D to 2D and back to 1D
    new_par_nodes = reshape_graph(new_par_nodes, shape=(4,3))
    assert new_par_nodes.shape[1:] == (4,3)
    np.testing.assert_equal(reshape_graph(new_par_nodes, shape=(12,)), par_nodes)

    # Flatten graph
    new_par_nodes = reshape_graph(par_nodes, shape=(4,3))
    new_par_nodes = mask_graph(new_par_nodes, True)
    np.testing.assert_equal(new_par_nodes, par_nodes)
    # Mask graph
    new_par_nodes = reshape_graph(par_nodes, shape=(4,3))
    mask = np.zeros(shape=new_par_nodes.shape[1:] ,dtype=bool)
    mask[1:] = True
    new_par_nodes = mask_graph(new_par_nodes, mask)
    assert new_par_nodes.shape == (1, np.count_nonzero(mask))

def test_isolate_graphtrees():
    """Test identification of various tree-types and loop cutting"""
    #
    # Various graph shapes
    #
    par_nodes = np.arange(18) + 1
    par_nodes[8] = 0        # 0..8: O-shape / circle
    par_nodes[12] = 10      # 9..12: 9-shape
    par_nodes[[14,16]] = 16 # 13..16: 1-shape
    par_nodes[17] = 17      # 17..17: .-shape / leaf-seed
    par_nodes = par_nodes[np.newaxis, :]
    tree_idx, seed_mask = isolate_graphtrees(par_nodes)
    np.testing.assert_equal(tree_idx,  [0,0,0,0,0,0,0,0,0, 1,1,1,1, 2,2,2,2, 3])
    np.testing.assert_equal(seed_mask, [1,1,1,1,1,1,1,1,1, 0,1,1,1, 0,0,0,1, 1])

    # Test loop-cutting toward last element
    cut_par_nodes = cut_graph_loops(par_nodes, tree_idx, seed_mask, sort_keys=-np.arange(par_nodes.size))
    np.testing.assert_equal(np.nonzero(cut_par_nodes != par_nodes)[1], [8, 12])
    tree_idx, seed_mask = isolate_graphtrees(cut_par_nodes)
    assert np.count_nonzero(seed_mask) == tree_idx.max()+1, 'Must have single seed per graph-trees'

    #
    # Single graph-tree with loop
    #
    par_nodes = np.arange(16) + 1
    par_nodes[14] = 8       # 8..14: loop
    par_nodes[15] = 8       # 15: single node branch, base 8
    par_nodes[4] = 14       # 0..4: 5 node branch, base 14
    # leftover:             # 5..7: 3 node branch, base 8
    par_nodes = par_nodes[np.newaxis, :]
    tree_idx, seed_mask = isolate_graphtrees(par_nodes)
    np.testing.assert_equal(tree_idx,  0)
    np.testing.assert_equal(seed_mask, [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1, 0])

    # Test loop-cutting at a middle element
    cut_par_nodes = cut_graph_loops(par_nodes, tree_idx, seed_mask,
            sort_keys=abs(11 - np.arange(par_nodes.size)))
    np.testing.assert_equal(np.nonzero(cut_par_nodes != par_nodes)[1], 11)
    tree_idx, seed_mask = isolate_graphtrees(cut_par_nodes)
    assert np.count_nonzero(seed_mask) == tree_idx.max()+1, 'Must have single seed per graph-tree'

def test_graph_combined():
    """Combined graph tests"""
    # Use negtive indices, 2 looped graph-trees
    par_nodes = np.arange(8) - 2
    tree_idx, seed_mask = isolate_graphtrees(par_nodes[np.newaxis, :])
    np.testing.assert_equal(seed_mask,  True, 'All elements must be part of a loop')
    np.testing.assert_equal(tree_idx[::2], 0, 'Even elements must be from the first tree')
    np.testing.assert_equal(tree_idx[1::2], 1, 'Odd elements must be from the second tree')

    # Build graph with two loops and multiple branches
    par_nodes = np.arange(2*3*5) - 1
    par_nodes[20] = -1      # 20..29: loop
    par_nodes[10] = -2      # 19..10: branch, base 28
    par_nodes[5] = 5        # 9..5: simple branch, self-pointing/loop base 5
    # leftover:             # 5..0: branch, base 29
    par_nodes = par_nodes[np.newaxis, :]
    tree_idx, seed_mask = isolate_graphtrees(par_nodes)
    assert tree_idx.max() == 1, 'Must have two graph-trees'
    ref_mask = np.zeros_like(seed_mask)
    ref_mask[5] = ref_mask[20:] = True
    np.testing.assert_equal(seed_mask,  ref_mask, 'Element 5 and after 20 must be part of a loop')
    # Reshape to 3D shape
    new_par_nodes = reshape_graph(par_nodes, shape=(5,3,2))
    tree_idx, seed_mask = isolate_graphtrees(new_par_nodes)
    assert tree_idx.max() == 1, 'Must have two graph-trees'
    np.testing.assert_equal(np.count_nonzero(seed_mask), 11, '11 elements must be part of a loop')
    # Leave loop elements only
    new_par_nodes = mask_graph(new_par_nodes, seed_mask)
    tree_idx, seed_mask = isolate_graphtrees(new_par_nodes)
    assert tree_idx.max() == 1, 'Must have two graph-trees'
    np.testing.assert_equal(seed_mask, True, 'All elements must be part of a loop')

    # Test strict ravel_multi_index()/unravel_index() compatibility
    def assert_unravel(res_nodes, flat_nodes):
        """Validate after ravel the result"""
        nodes = np.ravel_multi_index(res_nodes, res_nodes.shape[1:])
        nodes = nodes[np.unravel_index(range(res_nodes[0].size), res_nodes.shape[1:])]
        np.testing.assert_equal(nodes[np.newaxis], flat_nodes % flat_nodes.shape[1])
    # Reshape 1D to 3D
    new_par_nodes = reshape_graph(par_nodes, shape=(2,5,3), strict_unravel=True)
    assert_unravel(new_par_nodes, par_nodes)
    np.testing.assert_equal(new_par_nodes, reshape_graph(par_nodes, shape=(2,5,3), strict_unravel=False),
            'CHECKME: Unexpected result between "strict_unravel" modes')
    # Reshape 3D to 2D
    new_par_nodes = reshape_graph(new_par_nodes, shape=(6,5), strict_unravel=True)
    assert_unravel(new_par_nodes, par_nodes)
    np.testing.assert_equal(new_par_nodes, reshape_graph(par_nodes, shape=(6,5), strict_unravel=False),
            'CHECKME: Unexpected result between "strict_unravel" modes')

if __name__ == '__main__':
    # pytest-s
    test_reshape_graph()
    test_isolate_graphtrees()
    test_graph_combined()

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
        main_edge_list = None
    else:
        if USE_DISTANCE:
            distance = USE_DISTANCE(dem_band)
            dist = lambda b,n: distance.get_distance(b.T, n.T, flat=True).T
        main_edge_list, graph_list = build_graph_layers(lla_grid[...,-1], distance=dist)
        # Keep layers where each edge is used
        main_edge_layer = find_main_edge_layer(np.stack(main_edge_list, 1), graph_list, lla_grid.shape[:-1])
        # Combine node-addresses
        main_edge_text = np.apply_along_axis(
                lambda adr: np.asarray('.'.join(map(str, adr)), dtype=object), 0,
                tuple(get_node_address(np.stack(main_edge_list, 1), graph_list, lla_grid.shape[:-1]))[::-1])
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

        def create_edges(bases_arr, neighbor_arr, **kwargs):
            """Figure edge generation"""
            kwargs['name'] += f' ({bases_arr.shape[-1]})'
            bases_arr = lla_grid[*bases_arr]
            neighbor_arr = lla_grid[*neighbor_arr]
            neighbor_arr = (neighbor_arr - bases_arr) * .7 + bases_arr
            if 'text' not in kwargs:
                kwargs['text'] = np.arange(bases_arr.size)//3   # 3 points per line
            return visualize.figarg_create_lines((bases_arr, neighbor_arr)) | kwargs

        # All possible edges
        yield create_edges(*build_edge_list(lla_grid[...,-1], distance=dist)[:2],
                name='All edges', mode='lines', visible='legendonly')
        # Main-graph
        yield visualize.figarg_create_graph_lines(lla_grid, graph_list[0]) | dict(
                name='Main-graph', mode='lines', visible='legendonly')

        # Individual trees inside main-graph
        def create_graphtree_coverage(tree_idx, seed_mask):
            """Figure polygons generation"""
            id_list = np.arange(tree_idx.max() + 1)
            for id in id_list:
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
        current_layer = 0
        layer_slider_pos = [len(fig.data)]
        tree_idx_seed = []
        # Precache graph-trees (slow operation)
        for par_nodes in graph_list:
            t_idx, s_mask = isolate_graphtrees(par_nodes)
            print(f'  Processing layer of {par_nodes[0].size} nodes: {t_idx.max()+1} in next, {np.count_nonzero(s_mask)} seeds')
            tree_idx_seed.append((t_idx, s_mask))
        # Generate coverage masks
        for layer, (tree_idx, seed_mask) in enumerate(tree_idx_seed):
            print(f'Generating graph-layer {layer}')
            for t_idx, s_mask in reversed(tree_idx_seed[:layer]):
                tree_idx = tree_idx[t_idx]
                seed_mask = seed_mask[t_idx]
            for s in create_graphtree_coverage(tree_idx, seed_mask):
                yield s
            # Force default slider position
            layer_slider_pos.append(len(fig.data))
            if layer != current_layer:
                for scat in fig.data[layer_slider_pos[-2]:layer_slider_pos[-1]]:
                    scat.visible = False
        del tree_idx_seed

        # Separate main-edges by the layer, where are in use
        if main_edge_list:
            for lay in range(main_edge_layer.max()+1):
                mask = main_edge_layer == lay
                yield create_edges(main_edge_list[0][:, mask], main_edge_list[1][:, mask],
                        name=f'Main-edges {lay}', mode='lines+markers', text=main_edge_text[:, mask].T.flat)

        # Experimental layout slider
        num_steps = len(layer_slider_pos)-1
        visible = np.ones((num_steps, len(fig.data)), dtype=object)
        # Take default visibility
        for idx, scat in enumerate(fig.data):
            visible[:, idx] = scat.visible
        visible[:, layer_slider_pos[0]:layer_slider_pos[-1]] = False
        steps = []
        for idx in range(num_steps):
            visible[idx, layer_slider_pos[idx]:layer_slider_pos[idx+1]] = True
            steps.append(dict(method='restyle', args=[{'visible': visible[idx]}], label=f'Layer {idx}'))
        fig.update_layout(sliders=[dict(steps=steps)])

    visualize.figure_show(lla_grid, figarg_gen)
