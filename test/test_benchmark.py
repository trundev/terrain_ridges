""""Benchmark test for individual steps"""
import numpy as np
from terrain_ridges import gdal_utils, build_full_graph, topo_graph
from terrain_ridges.topo_graph import T_Graph


def test_build_graph_edges(benchmark, dem_band):
    """Benchmark build_graph_edges"""
    def distance(src, tgt):
        return gdal_utils.geod_distance(dem_band).get_distance(src.T, tgt.T, flat=True).T
    graph_edges, _ = benchmark(build_full_graph.build_graph_edges,
                               dem_band.dem_buf, distance=distance)
    # From test_build_graph_edges()
    node_alt = dem_band.dem_buf[*graph_edges]
    np.testing.assert_equal(node_alt[0] <= node_alt[1], True,
                            'Source node must be the lower one')
    np.testing.assert_equal(node_alt[0, :-1] >= node_alt[0, 1:], True,
                            'Edge list order must be descending source-node altitude')

def test_filter_treegraph(benchmark, build_graph_edges: T_Graph):
    """Benchmark filter_treegraph"""
    edge_mask = benchmark(topo_graph.filter_treegraph, build_graph_edges)
    graph_edges = build_graph_edges[..., edge_mask]
    # Borrowed from test_topo_graph.test_filter_treegraph()
    assert np.unique(graph_edges[:, 0], axis=1).shape == graph_edges[:, 0].shape, \
            'Result source-nodes are NOT unique'
    np.testing.assert_equal((graph_edges[:, 0] == graph_edges[:, 1]).all(0), False,
                            'Unexpected self-edges')

def test_isolate_subgraphs(benchmark, tree_graph: T_Graph):
    """Run topo_graph algorithms for a DEM-file"""
    parent_ids = benchmark(topo_graph.isolate_subgraphs, tree_graph)
    # from test_isolate_subgraphs2()
    parent_edges = parent_ids[*tree_graph]
    np.testing.assert_equal(parent_edges[0], parent_edges[1],
                            'Edges can not cross between sub-graphs')
