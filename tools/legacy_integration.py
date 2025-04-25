"""Integration to legacy ridges.py code"""
import numpy as np
from terrain_ridges import topo_graph
from terrain_ridges.topo_graph import T_Graph, T_IndexArray


def graph_to_mgrid(graph_edges: T_Graph) -> T_IndexArray:
    """Convert graph to "ridges" tool mgrid_n_xy format"""

    node_shape = topo_graph.graph_node_min_shape(graph_edges)
    valid_edges = topo_graph.valid_node_edges(graph_edges)
    print(f'Grid shape: {node_shape}, edges: {graph_edges.shape[2:]} / {valid_edges.size}')

    # Ensure this is a tree-graph (number of target nodes)
    tgt_nodes = topo_graph.accumulate_src_vals(graph_edges[:, ::-1], 1)
    np.testing.assert_equal((tgt_nodes == 1) | (tgt_nodes == 0), True,
                            'The graph must be filtered tree-graph')
    print(f'Root nodes: {(tgt_nodes == 0).sum()} / {tgt_nodes.size}')

    # Need only edges where both nodes are valid
    graph_edges = graph_edges[..., valid_edges]

    # Create legacy mgrid_n_xy grid
    mgrid_n_xy = np.indices(tuple(node_shape))
    mgrid_n_xy[:, *graph_edges[:, 0]] = graph_edges[:, 1]
    # Legacy grid has coordinates in the last dimension
    return np.moveaxis(mgrid_n_xy, 0, -1)

def keep_as_mgrid_snaphot(graph_edges: T_Graph, fname: str, *, add_sfix: bool=False) -> None:
    """Export graph as "ridges" tool stage 1 snapshot (<name>-1-mgrid_n_xy.npy)

    The file can be imported by "ridges.py", via "--resume-from-snapshot=1" option to generate
    ridges. This replaces its first stage (trace ridges) processing.
    """
    if add_sfix:
        fname += '-1-mgrid_n_xy.npy'
    np.save(fname, graph_to_mgrid(graph_edges))
