""""Conversion of topo_graph structures to legacy snapshot format (not actual tests)
"""
import os
import argparse
import numpy as np
import pytest
from terrain_ridges.topo_graph import T_Graph, T_MaskArray
from terrain_ridges import ridges, topo_graph
# Borrow some fixures from test_main
from tools import legacy_integration


def flip_edge_chain(graph_edges: T_Graph, start_edges: T_MaskArray, *,
                    edge_mask: T_MaskArray|bool=True) -> T_MaskArray:
    """Swap edge chains, starting at given edge set and propagating toward their target nodes

    This modifies the graph in-place.

    Parameters
    ----------
    graph_edges : (node-coord, 2, edge-indices) ndarray of int
        Source graph
    start_edges : (edge-indices) ndarray of bool
        Mask of edges to begin flipping
    edge_mask : (edge-indices) ndarray of bool [optional]
        Limit where propagation can take place
        Graph inplace modifications must be seen by the caller (not possible on masked copy)

    Returns
    -------
    swap_mask : (edge-indices) ndarray of bool
        Mask of where edges were swapped
    """
    edge_mask = np.broadcast_to(edge_mask, graph_edges.shape[2:]).copy()
    node_mask: T_MaskArray = topo_graph.broadcast_node_vals(graph_edges[..., edge_mask], False).copy()
    node_mask[*graph_edges[:, 0, start_edges]] = True

    node_mask = topo_graph.expand_mask(graph_edges[..., edge_mask], node_mask)
    # Take only edges where expansion occurs
    swap_mask = edge_mask.copy()
    swap_mask[swap_mask] = node_mask[*graph_edges[:, 0, edge_mask]]
    graph_edges[..., swap_mask] = graph_edges[:, ::-1, swap_mask]
    return swap_mask

#
# Experiment with tree-merging algorithms
#
MERGE_ITERATIONS = 1
@pytest.fixture(scope='module')
def merge_treegraph(build_graph_edges: T_Graph) -> T_Graph:
    """Merged tree-graph from test DEM file"""
    tree_edge_mask = topo_graph.filter_treegraph(build_graph_edges)
    print(f'Initial edges: {build_graph_edges.shape[2:]}')
    print(f'- tree mask:   {tree_edge_mask.shape}, number {tree_edge_mask.sum()}')

    # Isolate edges between sub-graphs
    parent_ids = topo_graph.isolate_subgraphs(build_graph_edges[..., tree_edge_mask])
    print('Total sub-graphs', parent_ids.max() + 1)
    par_graph_edges = parent_ids[*build_graph_edges[..., ~tree_edge_mask]]
    print('Internal sub-graph edges (non-tree)', (par_graph_edges[0] == par_graph_edges[1]).sum())
    print('External sub-graph edges', (par_graph_edges[0] != par_graph_edges[1]).sum())

    # Drop sub-graph internal edges, but keep tree-edges inside
    mask = par_graph_edges[0] != par_graph_edges[1]
    par_graph_edges = par_graph_edges[..., mask]
    ext_edge_mask = np.zeros_like(tree_edge_mask)
    ext_edge_mask[~tree_edge_mask] = mask
    mask = tree_edge_mask | ext_edge_mask
    build_graph_edges = build_graph_edges[..., mask]
    tree_edge_mask = tree_edge_mask[mask]
    ext_edge_mask = ext_edge_mask[mask]
    del mask
    assert np.count_nonzero(ext_edge_mask) == par_graph_edges[0].size, 'Wrong internal mask filtering'
    print(f'Edges after filtering internal non-tree edges: {build_graph_edges.shape[2:]}')
    print(f'- tree mask:     {tree_edge_mask.shape}, number {tree_edge_mask.sum()}')
    print(f'- external mask: {ext_edge_mask.shape}, number: {ext_edge_mask.sum()}')
    print(f'- parent edges:  {par_graph_edges.shape}')

    # Drop duplicated sub-graph external edges (sort to include both edge directions)
    mask = topo_graph.unique_mask(np.sort(par_graph_edges, axis=0), axis=1)
    ext_edge_mask[ext_edge_mask] = mask
    del mask, par_graph_edges
    print(f'Unique external sub-graph edges: {ext_edge_mask.shape}, number: {ext_edge_mask.sum()}')

    # Repeat filtering, but on graph from parent nodes
    # Note: This swaps edges inside build_graph_edges array
    next_tree_mask = topo_graph.filter_treegraph(build_graph_edges, ext_edge_mask,
                                                 node_ids=parent_ids)
    print(f'- next level tree mask:   {next_tree_mask.shape}, number {next_tree_mask.sum()}')

    # Note:
    # ext_edge_mask includes the internal, external and tree edges for the newly created sub-graphs,
    # next_tree_mask contains the tree-edges of the next sub-graph

    # Flip all edge-chains that starts at `next_tree_mask` source-nodes
    flip_edge_chain(build_graph_edges, next_tree_mask, edge_mask=tree_edge_mask)

    return build_graph_edges[..., tree_edge_mask | next_tree_mask]

@pytest.fixture(params=range(2))
def more_tree_graph(request, tree_graph: T_Graph, merge_treegraph: T_Graph) -> T_Graph:
    """Pick different tree-graph filtering options"""
    return (tree_graph, merge_treegraph)[request.param]

def test_keep_as_mgrid_snaphot(more_tree_graph: T_Graph, dem_file_path: str):
    """Export topo_graph results to a legacy stage 1 snapshot file"""
    legacy_integration.keep_as_mgrid_snaphot(more_tree_graph, dem_file_path, add_sfix=True)

def test_generate_ridges(tree_graph: T_Graph, dem_file_path: str):
    """Export topo_graph results to a legacy stage 1 snapshot file"""
    legacy_integration.keep_as_mgrid_snaphot(tree_graph, dem_file_path, add_sfix=True)

    result_name = dem_file_path + '.kml'
    # Taken from test_main.test_real_dem
    args = argparse.Namespace(
            src_dem_file=dem_file_path,
            dst_ogr_file=result_name,
            dst_format=None,
            valleys=False,
            boundary_val=None,
            distance_method=list(ridges.DISTANCE_METHODS.keys())[-1],
            multi_layer=None,
            append=False,
            separated_branches=False,
            smoothen_geometry=False,
            resume_from_snapshot=1,
            keep_snapshots=False)
    # Delete possible result left-over
    try:
        os.unlink(result_name)
    except FileNotFoundError:
        pass

    # Generate ridges
    ret = ridges.main(args)
    assert ret == 0, f'ridges.main() failed, code {ret}'

    print(f'Result KML: {result_name}')
