"""Pytest for topo_graph.py"""
import itertools
import numpy as np
import pytest
from . import conftest  # Imported by pytest anyway
from terrain_ridges import topo_graph, build_full_graph
from terrain_ridges.topo_graph import T_Graph, T_IndexArray


@pytest.fixture(params=([5], [7,3], [3,2,5]),
                ids=lambda s: 'shape:'+'x'.join(map(str, s)))
def complete_graph(request):
    shape = request.param
    """Graph with edges between all nodes, including self-pointing"""
    graph_edges = itertools.product(np.ndindex(tuple(shape)), repeat=2)
    return np.fromiter(graph_edges, dtype=(int, (2, len(shape)))).T

@pytest.fixture
def altitude_graph(altitude_grid):
    """Graph generated from altitude based grid, using build_full_graph"""
    graph_edges, _ = build_full_graph.build_graph_edges(altitude_grid)
    return graph_edges

def test_filter_treegraph(complete_graph):
    """Test creation of tree-graph"""
    shape = topo_graph.graph_node_min_shape(complete_graph)
    num_nodes = np.prod(shape)
    # Check test-data assumptions
    assert complete_graph[0, 0].size == num_nodes*num_nodes, 'Unexpected number of test edges'
    assert (complete_graph[:, 0] == complete_graph[:, 1]).all(0).sum() == num_nodes, \
            'Unexpected number of test self-edges'

    # Check if the graph is modified inplace
    res_edges, edge_mask = topo_graph.filter_treegraph(complete_graph)
    np.testing.assert_equal(res_edges, complete_graph[..., edge_mask],
                            'Source edges must be modified in-place')
    # Every node must have single edge
    assert res_edges[0,0].size == num_nodes, \
            f'Number of result edges vs. nodes mismatch: {res_edges[0,0].size} / {num_nodes}'
    # Check source-node uniqueness
    vals, counts = np.unique(res_edges[:, 0], axis=1, return_counts=True)
    assert vals.shape == res_edges[:, 0].shape, 'Unique source-nodes shape mismatch'
    np.testing.assert_equal(counts, 1, 'Result source-nodes are NOT unique')
    # Check the number of self-edges (single one at the tree root)
    assert (res_edges[:, 0] == res_edges[:, 1]).all(0).sum() == 1, \
            'Result must include a single self-edge'

def test_isolate_graph_sinks(complete_graph):
    """Test graph sink-node detection"""
    # Drop self-edges from the complete-graph, this allows creation of sinks by filter_treegraph()
    complete_graph = complete_graph[..., ~(complete_graph[:, 0] == complete_graph[:, 1]).all(0)]
    sink_mask = topo_graph.isolate_graph_sinks(complete_graph)
    np.testing.assert_equal(sink_mask, False, 'No sinks must be detected (graph is complete)')

    # Reduce edges by making it tree-graph
    graph_edges, _ = topo_graph.filter_treegraph(complete_graph)
    sink_mask = topo_graph.isolate_graph_sinks(graph_edges)
    assert np.count_nonzero(sink_mask) == 1, \
            'Single sink must be detected (graph is a single tree)'
    leaf_mask = topo_graph.isolate_graph_sinks(graph_edges[:, ::-1])
    assert np.count_nonzero(leaf_mask) > 2, \
            'Multiple leaves must be detected (graph is a single tree)'

def test_accumulate_src_vals(complete_graph):
    """Test node-value accumulation"""
    node_vals = topo_graph.accumulate_src_vals(complete_graph, 3)
    np.testing.assert_equal(node_vals, 3*node_vals.size,
                            'Accumulated values must match number of nodes (graph is complete)')

    # Reduce edges by making it tree-graph, obtain leaves
    graph_edges, _ = topo_graph.filter_treegraph(complete_graph)
    leaf_mask = topo_graph.isolate_graph_sinks(graph_edges[:, ::-1])

    # Accumulate along known tree-graph
    node_vals = topo_graph.accumulate_src_vals(graph_edges, 5)
    np.testing.assert_equal(node_vals[leaf_mask], 0, 'Leaf-nodes mut have no accumulated value')
    assert node_vals.sum() == 5 * node_vals.size, \
            'Accumulation must not change total sum (graph is a tree)'

def test_equalize_subgraph_vals(complete_graph):
    """Test node-value equalization"""
    shape = topo_graph.graph_node_min_shape(complete_graph)
    node_vals = np.arange(np.prod(shape)).reshape(shape) + 5
    res_vals = topo_graph.equalize_subgraph_vals(complete_graph, node_vals)
    np.testing.assert_equal(res_vals, 5, 'All nodes must have the same value (graph is complete)')
    assert res_vals is node_vals, 'Must be in-place operation'

def test_equalize_subgraph_vals2(altitude_graph, altitude_grid):
    """Test node-value equalization on altitude based graph"""
    node_vals = np.full(altitude_grid.shape, -7)
    val_data_mask = np.isfinite(altitude_grid)
    node_vals[val_data_mask] = np.arange(np.count_nonzero(val_data_mask)) + 5
    res_vals = topo_graph.equalize_subgraph_vals(altitude_graph, node_vals.copy())
    edge_res_vals = res_vals[*altitude_graph]
    np.testing.assert_equal(edge_res_vals[0], edge_res_vals[1],
                            'Node-values for each edge must be equalized')
    np.testing.assert_equal(res_vals[~val_data_mask], -7,
                            'Unreferenced nodes must keep its value')
    assert (res_vals[val_data_mask].min() == 5) and (res_vals.max() < node_vals.max()), \
            'Referenced nodes must have value in initial range'
    assert res_vals[val_data_mask].min() == 5, 'The minimum value must persist'

def test_propagate_mask(complete_graph):
    """Test node-mask propagation"""
    # Contract-propagate mask on all nodes
    node_mask = topo_graph.propagate_mask(complete_graph, True)
    np.testing.assert_equal(node_mask, True, 'All nodes must be selected (graph is complete)')

    # Expand-propagate mask from single node (at center)
    node_mask[...] = False
    node_mask[*(np.asarray(node_mask.shape) // 2)] = True
    res_mask = topo_graph.propagate_mask(complete_graph, node_mask, operation=np.logical_or)
    np.testing.assert_equal(res_mask, True, 'All nodes must be selected (graph is complete)')

    # Contract-propagate mask from single node
    res_mask = topo_graph.propagate_mask(complete_graph, node_mask, operation=np.logical_and)
    np.testing.assert_equal(res_mask, node_mask, 'Mask must not change (graph is complete)')

    #
    # Similar tests, but on a tree-graph
    #
    graph_edges, _ = topo_graph.filter_treegraph(complete_graph)

    # Expand-propagate mask from single node (at center)
    res_mask = topo_graph.propagate_mask(graph_edges, node_mask, operation=np.logical_or)
    assert not res_mask.all(), 'Some nodes must be NOT be selected'
    assert (~node_mask | res_mask).all(), 'Mask must expand'
    assert (node_mask != res_mask).any(), 'Mask must NOT remain the same'
    # Same expand-propagate mask in reverse
    res_mask = topo_graph.propagate_mask(graph_edges[:, ::-1], node_mask,
                                         operation=np.logical_or)
    assert not res_mask.all(), 'Some nodes must be NOT be selected'
    assert (~node_mask | res_mask).all(), 'Mask must expand (can remain the same)'

def test_propagate_mask2(altitude_graph):
    """Test node-mask propagation on altitude based graph"""
    # Run on filtered tree-graph
    graph_edges, _ = topo_graph.filter_treegraph(altitude_graph)

    # Loop-based tests
    loop_mask = topo_graph.propagate_mask(graph_edges, True)
    loop_mask = topo_graph.propagate_mask(graph_edges[:, ::-1], loop_mask)
    np.testing.assert_equal(loop_mask, False, 'Reference data must have no loops')
    sink_mask = topo_graph.isolate_graph_sinks(graph_edges)
    np.testing.assert_equal(sink_mask & loop_mask, False, 'Sink and loop nodes must not overlap')
    leaf_mask = topo_graph.isolate_graph_sinks(graph_edges[:, ::-1])
    assert np.count_nonzero(leaf_mask) > 1, 'Multiple leaves must be detected'
    np.testing.assert_equal(leaf_mask & sink_mask, False, 'Leaf and sink nodes must not overlap')
    np.testing.assert_equal(leaf_mask & loop_mask, False, 'Leaf and loop nodes must not overlap')

def test_isolate_subgraphs(complete_graph):
    """Test sub-graph identification"""
    parent_ids = topo_graph.isolate_subgraphs(complete_graph)
    np.testing.assert_equal(parent_ids, 0,
                            'All nodes must have the same parent (graph is complete)')

    # Reduce edges by making it tree-graph
    graph_edges, _ = topo_graph.filter_treegraph(complete_graph)

    # Use larger node-grid shape
    node_shape = np.asarray(parent_ids.shape)
    parent_ids = topo_graph.isolate_subgraphs(graph_edges, node_shape=node_shape + 1)
    assert np.count_nonzero(parent_ids == 0) == node_shape.prod(), \
            'All used node must have the same parent'
    assert np.count_nonzero(parent_ids == -1) == parent_ids.size - node_shape.prod(), \
            'Extra nodes must be marked as unused ones'

def test_isolate_subgraphs2(altitude_graph, altitude_grid):
    """Test sub-graph identification on altitude based graph"""
    # Run on initial edge-list
    parent_id = topo_graph.isolate_subgraphs(altitude_graph)
    if altitude_grid.ndim > 1:
        np.testing.assert_equal(parent_id, np.where(np.isfinite(altitude_grid), 0, -1),
                                'Initial edge-list, must have single sub-graph')
    else:   # 1D - relaxed rule (holes in 1D grid can split into sub-graphs)
        np.testing.assert_equal(parent_id >= 0, np.isfinite(altitude_grid),
                                'Unused nodes must match the holes in the grid')

    # Run on filtered tree-graph
    tree_edges, _ = topo_graph.filter_treegraph(altitude_graph)
    parent_id = topo_graph.isolate_subgraphs(tree_edges)
    parent_edges = parent_id[*tree_edges]
    np.testing.assert_equal(parent_edges[0], parent_edges[1],
                            'Edges can not cross between sub-graphs')

#
# Legacy graph-tree implementation
#
type T_TreeGraph = T_IndexArray
def graph_to_edge_list(tgt_nodes: T_TreeGraph, *, self_edges: bool=False) -> T_Graph:
    """Convert legacy tree-graph to edge list style graph

    Parameters
    ----------
    tgt_nodes : (target-node-idx, node-idx0, node-idx1, ...) ndarray of int
        Source tree-graph
    self_edges : bool
        Create self-pointing edges (where target-node is the same as node)

    Returns
    -------
    graph_edges : (node-idx, 2, edge-idx) ndarray of int
    """
    grid_shape = tgt_nodes.shape[1:]
    # `self_edges == False` - skip edges, where target and source nodes are the same
    normal_mask = (tgt_nodes != np.indices(grid_shape)).any(0)
    mask = np.broadcast_to(True, tgt_nodes.shape[1:]) if self_edges else normal_mask
    return np.stack((np.nonzero(mask), tgt_nodes[:, mask]), axis=1)

def wrap_isolate_subgraphs(tgt_nodes: T_IndexArray):
    """Wrapper of isolate_subgraphs() to use legacy tree-graph structure"""
    graph_edges = graph_to_edge_list(tgt_nodes, self_edges=True)
    parent_ids = topo_graph.isolate_subgraphs(graph_edges)
    return graph_edges, parent_ids

def test_combined():
    """Test identification of various tree-types"""
    #
    # Various graph shapes
    #
    tgt_nodes = np.arange(18) + 1
    tgt_nodes[8] = 0        # 0..8: O-shape / circle
    tgt_nodes[12] = 10      # 9..12: 9-shape
    tgt_nodes[[14,16]] = 16 # 13..16: 1-shape
    tgt_nodes[17] = 17      # 17..17: .-shape / leaf-root
    tgt_nodes = tgt_nodes[np.newaxis, :]
    graph_edges, parent_ids = wrap_isolate_subgraphs(tgt_nodes)
    np.testing.assert_equal(parent_ids,  [0,0,0,0,0,0,0,0,0, 1,1,1,1, 2,2,2,2, 3])
    loop_mask = topo_graph.propagate_mask(graph_edges, True)
    loop_mask = topo_graph.propagate_mask(graph_edges[:, ::-1], loop_mask)
    np.testing.assert_equal(loop_mask, [1,1,1,1,1,1,1,1,1, 0,1,1,1, 0,0,0,1, 1])
    np.testing.assert_equal(topo_graph.isolate_graph_sinks(graph_edges), False)
    leaf_mask = topo_graph.isolate_graph_sinks(graph_edges[:, ::-1])
    np.testing.assert_equal(leaf_mask, [0,0,0,0,0,0,0,0,0, 1,0,0,0, 1,0,1,0, 0])

    #
    # Single graph-tree with loop
    #
    tgt_nodes = np.arange(16) + 1
    tgt_nodes[14] = 8       # 8..14: loop
    tgt_nodes[15] = 8       # 15: single node branch, base 8
    tgt_nodes[4] = 14       # 0..4: 5 node branch, base 14
    # leftover:             # 5..7: 3 node branch, base 8
    tgt_nodes = tgt_nodes[np.newaxis, :]
    graph_edges, parent_ids = wrap_isolate_subgraphs(tgt_nodes)
    np.testing.assert_equal(parent_ids,  0)
    loop_mask = topo_graph.propagate_mask(graph_edges, True)
    loop_mask = topo_graph.propagate_mask(graph_edges[:, ::-1], loop_mask)
    np.testing.assert_equal(loop_mask, [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1, 0])
    np.testing.assert_equal(topo_graph.isolate_graph_sinks(graph_edges), False)
    leaf_mask = topo_graph.isolate_graph_sinks(graph_edges[:, ::-1])
    np.testing.assert_equal(leaf_mask, [1,0,0,0,0,1,0,0, 0,0,0,0,0,0,0, 1])
