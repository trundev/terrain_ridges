"""Pytest for build_full_graph.py"""
import numpy as np
import pytest
from .conftest import altitude_grid  # Clarity only, imported by pytest anyway
from terrain_ridges import build_full_graph


@pytest.mark.parametrize('batch_size', (10**i for i in range(2,5)))
def test_prepare_edges(altitude_grid, batch_size):
    """Test internal edge-prepare function"""
    for edges, keys in build_full_graph.prepare_edges(altitude_grid, batch_size=batch_size):
        # Test shapes
        assert edges.shape[0] == altitude_grid.ndim, 'First `edge` dimension must be node-xy'
        assert edges.shape[1] == 2, 'Second `edge` dimension must be source-target'
        assert keys.shape[0] == 2, 'First `keys` dimension must be slope-alt'
        assert edges.shape[-1] == keys.shape[-1], 'Number of edges and keys must match'
        assert edges.shape[-1] <= batch_size, 'Exceeded number of edges in a batch'
        # Test node-coordinates
        np.testing.assert_equal(edges >= 0, True, 'Negative node coordinate')
        np.testing.assert_equal(edges.T < altitude_grid.shape, True, 'Node coordinate outside the grid')
        # Test node altitude
        node_alt = altitude_grid[*edges]
        np.testing.assert_equal(np.isfinite(node_alt), True, 'Node altitudes must be valid')
        # Test if first (source) node is lower than second (target) one
        np.testing.assert_equal(node_alt[0] <= node_alt[1], True,
                                'Source node must be the lower one')
        # Test sort-keys
        np.testing.assert_equal(keys[0], node_alt[1] - node_alt[0],
                                'Slope key-component must be altitude difference (no `distance`)')
        np.testing.assert_equal(keys[1], node_alt[0],
                                'Altitude key-component must match the lower node')

def test_build_graph_edges(altitude_grid):
    """Test build_graph_edges edge/node order"""
    main_edge_list, _ = build_full_graph.build_graph_edges(altitude_grid)
    # Test if first (source) node is lower than second (target) one
    node_alt = altitude_grid[*main_edge_list]
    np.testing.assert_equal(node_alt[0] <= node_alt[1], True,
                            'Source node must be the lower one')

    # Test if list is in descending node-altitude order
    src_node_alt = altitude_grid[*main_edge_list[:, 0]]
    np.testing.assert_equal(src_node_alt[:-1] >= src_node_alt[1:], True,
                            'Edge list order must be descending source-node altitude')

    # Test if highest (ceil) points are edge target-nodes only
    # (redundant, but the approach is funny)
    ceil_nodes = np.asarray(np.nonzero(np.nanmax(altitude_grid) == altitude_grid))
    try:
        # This is memory inefficient, when ceil nodes and edges are too many
        src_ceil_mask = (main_edge_list[:, np.newaxis, 0].T == ceil_nodes.T).T.all(0).any(0)
    except MemoryError as ex:
        pytest.xfail(str(ex))
    # Ceil nodes can be at source-side, but only at the beginning (no gaps between such)
    assert not src_ceil_mask[np.count_nonzero(src_ceil_mask):].any(), \
            'Highest points(s) was found at a source-node of an edge'
