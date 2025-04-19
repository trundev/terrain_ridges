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
    orig_graph = complete_graph.copy()
    edge_mask = topo_graph.filter_treegraph(complete_graph)
    assert np.count_nonzero(np.any(orig_graph != complete_graph, axis=(0, 1))) > 1, \
            'Some source edges must be modified in-place'
    # Every node must have single edge
    assert np.count_nonzero(edge_mask) == num_nodes, \
            f'Number of result edges vs. nodes mismatch: {edge_mask.sum()} / {num_nodes}'
    # Check source-node uniqueness
    res_edges = complete_graph[..., edge_mask]
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
    edge_mask = topo_graph.filter_treegraph(complete_graph)
    graph_edges = complete_graph[..., edge_mask]
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
    edge_mask = topo_graph.filter_treegraph(complete_graph)
    graph_edges = complete_graph[..., edge_mask]
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
    edge_mask = topo_graph.filter_treegraph(complete_graph)
    graph_edges = complete_graph[..., edge_mask]

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
    edge_mask = topo_graph.filter_treegraph(altitude_graph)
    graph_edges = altitude_graph[..., edge_mask]

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

def isolate_subgraphs_test(graph_edges: T_Graph, *args, **kwargs):
    """isolate_subgraphs() wrapper to fully test its result, esp. against safe one"""
    parent_ids = topo_graph.isolate_subgraphs(graph_edges, *args, **kwargs)
    parent_edges = parent_ids[*graph_edges[..., topo_graph.valid_node_edges(graph_edges)]]
    np.testing.assert_equal(parent_edges[0], parent_edges[1],
                            'Edges can not cross between sub-graphs')
    assert parent_ids.min() in (0, -1), 'Parent IDs must start from 0 or -1'
    uniq_ids = np.unique(parent_ids)
    assert uniq_ids.size == uniq_ids.max() - uniq_ids.min() + 1, 'Parent IDs must be consecutive'

    # Compare against result from reference/safe implementation
    ref_res = topo_graph.isolate_subgraphs_safe(graph_edges, *args, **kwargs)
    # Map `parent_ids` values to `ref_res`
    # (arrays may not be identical, but IDs must have 1-to-1 correspondence)
    _, uniq_idx, uniq_inv = np.unique(parent_ids, return_index=True, return_inverse=True)
    adj_res = ref_res.flat[uniq_idx][uniq_inv]
    np.testing.assert_equal(adj_res, ref_res, 'isolate_subgraphs() result do NOT match reference')
    return parent_ids

def test_isolate_subgraphs(complete_graph):
    """Test sub-graph identification"""
    parent_ids = isolate_subgraphs_test(complete_graph)
    np.testing.assert_equal(parent_ids, 0,
                            'All nodes must have the same parent (graph is complete)')

    # Reduce edges by making it tree-graph
    edge_mask = topo_graph.filter_treegraph(complete_graph)
    graph_edges = complete_graph[..., edge_mask]

    # Use larger node-grid shape
    node_shape = np.asarray(parent_ids.shape)
    parent_ids = isolate_subgraphs_test(graph_edges, node_shape=node_shape + 1)
    assert np.count_nonzero(parent_ids == 0) == node_shape.prod(), \
            'All used node must have the same parent'
    assert np.count_nonzero(parent_ids == -1) == parent_ids.size - node_shape.prod(), \
            'Extra nodes must be marked as unused ones'

def test_isolate_subgraphs2(altitude_graph, altitude_grid):
    """Test sub-graph identification on altitude based graph"""
    # Run on initial edge-list
    parent_id = isolate_subgraphs_test(altitude_graph)
    if altitude_grid.ndim > 1:
        np.testing.assert_equal(parent_id, np.where(np.isfinite(altitude_grid), 0, -1),
                                'Initial edge-list, must have single sub-graph')
    else:   # 1D - relaxed rule (holes in 1D grid can split into sub-graphs)
        np.testing.assert_equal(parent_id >= 0, np.isfinite(altitude_grid),
                                'Unused nodes must match the holes in the grid')

    # Run on filtered tree-graph
    edge_mask = topo_graph.filter_treegraph(altitude_graph)
    parent_id = isolate_subgraphs_test(altitude_graph[..., edge_mask])

#
# Various graph shapes with expected results
#
@pytest.mark.parametrize('test_data', (
    # Various graph shapes
    dict(graph_edges=np.array([
            # 0..8: O-shape / circle
            [0, 1], [ 1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 0],
            # 9..12: 9-shape
            [9, 10], [10, 11], [11, 12], [12, 10],
            # 13..16: 1-shape
            [13, 14], [14, 16], [15, 16], [16, 16],
            # 17..17: .-shape / leaf-root
            [17, 17]], dtype=int).T[np.newaxis],
        parent_ids=[0,0,0,0,0,0,0,0,0, 1,1,1,1, 2,2,2,2, 3],
        loop_mask=[1,1,1,1,1,1,1,1,1, 0,1,1,1, 0,0,0,1, 1],
        sink_mask=False,
        leaf_mask=[0,0,0,0,0,0,0,0,0, 1,0,0,0, 1,0,1,0, 0]),
    # Single graph-tree with loop
    dict(graph_edges=np.array([
            # 0..4: 5 node branch, base 14
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 14],
            # 5..7: 3 node branch, base 8
            [5, 6], [6, 7], [7, 8],
            # 8..14: loop
            [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14,  8],
            # 15: single node branch, base 8
            [15,  8]], dtype=int).T[np.newaxis],
        parent_ids=0,
        loop_mask=[0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1, 0],
        sink_mask=False,
        leaf_mask=[1,0,0,0,0,1,0,0, 0,0,0,0,0,0,0, 1]),
    # Even more generic sub-graphs (2D)
    dict(graph_edges=np.array([
            # Straight line from leaf (0,0) to root (0,2)
            [(0,0), (0,1)], [(0,1), (0,2)],
            # Loop with incoming and outgoing branches
            [(1,0), (1,1)], [(1,1), (1,0)], # loop
            [(1,1), (1,2)], [(1,2), (1,3)], # root (1,3)
            [(1,5), (1,4)], [(1,4), (1,0)], # leaf (1,5)
            # Nodes with multiple targets from leaf (2,0)
            [(2,0), (2,1)], [(2,1), (2,2)], [(2,2), (2,3)], # root (2,3)
            [(2,1), (2,4)],                 # root (2,4)
            [(2,2), (2,5)], [(2,5), (2,6)], # root (2,6)
            # Invalid source-node to root (3,1)
            [(3,-1), (3,0)], [(3,0), (3,1)],
            # Invalid target-node from leaf (4,0)
            [(4,0), (4,1)], [(4,1), (4,-1)],
            # Invalid source and target-nodes
            [(-1,-1), (5,-1)],
        ], dtype=int).T,
        parent_ids=[
            (0,  0,  0, -1, -1, -1, -1),
            (1,  1,  1,  1,  1,  1, -1),
            (2,  2,  2,  2,  2,  2,  2),
            (3,  3, -1, -1, -1, -1, -1),
            (4,  4, -1, -1, -1, -1, -1),
            (-1,-1, -1, -1, -1, -1, -1)],
        loop_mask=[
            (0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0)],
        sink_mask=[
            (0, 0, 1, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0),
            (0, 0, 0, 1, 1, 0, 1),
            (0, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0)],
        leaf_mask=[
            (1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 0),
            (1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0)]),
    ))
def test_combined(test_data):
    """Test identification of various tree-types"""
    graph_edges = test_data['graph_edges']
    parent_ids = isolate_subgraphs_test(graph_edges)
    np.testing.assert_equal(parent_ids, test_data['parent_ids'])
    loop_mask = topo_graph.propagate_mask(graph_edges, True)
    loop_mask = topo_graph.propagate_mask(graph_edges[:, ::-1], loop_mask)
    np.testing.assert_equal(loop_mask, test_data['loop_mask'])
    np.testing.assert_equal(topo_graph.isolate_graph_sinks(graph_edges), test_data['sink_mask'])
    leaf_mask = topo_graph.isolate_graph_sinks(graph_edges[:, ::-1])
    np.testing.assert_equal(leaf_mask, test_data['leaf_mask'])
