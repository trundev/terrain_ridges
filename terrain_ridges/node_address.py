"""Assign addresses to each node, based on sub-graph hierarchy"""
import numpy as np
from typing import Iterator, Sequence
from .topo_graph import T_Graph, T_IndexArray
from . import topo_graph


NORMALIZE_ID_LIMIT = 4000

def ravel_node_address(node_addrs: T_IndexArray|Sequence[int], *, dims: T_IndexArray|None=None
                       ) -> tuple[T_IndexArray, T_IndexArray]:
    """Convert node addresses to scalars"""
    _node_addrs = np.asarray(node_addrs)
    if dims is None:
        _dims: T_IndexArray = _node_addrs.reshape(_node_addrs.shape[0], -1).max(1) + 1
    else:
        _dims = dims
    return np.ravel_multi_index(tuple(_node_addrs), dims=tuple(_dims),
                                order='F').astype(int), _dims

#
# Address generation
#
def normalize_id_simple(parent_ids: T_IndexArray) -> T_IndexArray:
    """Convert node ID-pair to indices (oversimplified version)

    Note:
    When `parent_ids` is large (>100K) and contains big number of unique values (>20K),
    this allocates huge arrays (for `mask` and `add.accumulate()` result).
    """
    mask = np.zeros(parent_ids.shape + (parent_ids.max() + 1,), dtype=bool)
    np.put_along_axis(mask, parent_ids[..., np.newaxis], True, axis=-1)
    return (np.add.accumulate(mask, axis=0) - 1)[mask]

def normalize_id(parent_ids: T_IndexArray) -> T_IndexArray:
    """Convert node/parent IDs to normalized indices

    The node ID is represented by the position of a parent ID in an array and the ID itself.
    Parent IDs must be already indices (consecutive unique numbers starting from zero), if not,
    this can be achieved by:
    >>> parent_ids = np.unique_inverse(parent_ids)[1]

    Note:
    The parent ID array must be one dimensional, other options are not fully explored yet.
    """
    # Make collapsed sub-graphs (invalid IDs) unique ones
    mask = parent_ids < 0
    if mask.any():
        first_id = max(parent_ids.max() + 1, 0)
        parent_ids[mask] = np.arange(np.count_nonzero(mask)) + first_id

    # Ensure parent IDs are normalized
    uniq_ids = np.unique(parent_ids)
    assert uniq_ids.size == uniq_ids.max() + 1, \
            'Parent IDs must be indices (consecutive unique numbers starting from zero)'
    assert parent_ids.ndim == 1, 'Only single dimensional arrays are currently supported'

    # Process node IDs in batches, limited by NORMALIZE_ID_LIMIT
    # this is to prevents from allocation of huge arrays (`mask` then `add.accumulate()`)
    num_parents = parent_ids.max() + 1
    normalized = np.empty_like(parent_ids, shape=0)
    last_acc_ids = np.full(num_parents, -1)
    mask = np.empty(parent_ids[:NORMALIZE_ID_LIMIT].shape + (num_parents,), dtype=bool)
    for idx in range(0, parent_ids.shape[0], mask.shape[0]):
        # Take the next batch of parent IDs
        pids = parent_ids[idx: idx + mask.shape[0], ..., np.newaxis]
        mask[...] = False
        np.put_along_axis(mask[:pids.shape[0]], pids, True, axis=-1)
        acc_ids = (np.add.accumulate(mask, axis=0) + last_acc_ids)
        # Keep where the accumulated ID has reached for the next iteration
        last_acc_ids = acc_ids[-1]
        normalized = np.concatenate((normalized, acc_ids[mask]), axis=0)

    # Combine the node ID pair
    return np.stack((normalized, parent_ids))

def node_parent_gen(graph_edges: T_Graph, *, node_shape: T_IndexArray|None=None
                    ) -> Iterator[T_IndexArray]:
    """Iterator to assign a parent-node/sub-graph IDs to each node

    Parameters
    ----------
    graph_edges : (node-coord, 2, edge-indices) ndarray of int
        The graph
    node_shape : list of int [optional]
        Shape of node-grid, also the shape of the result from first iteration

    Returns
    -------
    parent_ids : (node-indices) ndarray of int
        Unique parent node IDs for each graph-node, see `topo_graph.isolate_subgraphs()`.
        - First iteration returns array of shape `node_shape`
        - Next iterations return 1D array of elements for each unique ID
          returned by previous iteration
    """
    # Iterate until the graph collapses, each iteration groups the nodes from the
    # previous one into sub-graphs, which in turn are treated as nodes by the next one
    while graph_edges.size:
        tree_edge_mask = topo_graph.filter_treegraph(graph_edges)
        parent_ids = topo_graph.isolate_subgraphs(graph_edges[..., tree_edge_mask],
                                                  node_shape=node_shape)
        node_shape = parent_ids.max() + 1   # `isolate_subgraphs()` does this anyway
        yield parent_ids

        # Identify edges between sub-graphs, skip internal ones
        ext_edge_mask = ~tree_edge_mask
        par_graph_edges = parent_ids[*graph_edges[..., ext_edge_mask]]
        mask = par_graph_edges[0] != par_graph_edges[1]
        ext_edge_mask[ext_edge_mask] = mask
        # Select unique edges between sub-graphs, skip duplicated ones
        mask = topo_graph.unique_mask(np.sort(par_graph_edges[..., mask], axis=0), axis=1)
        ext_edge_mask[ext_edge_mask] = mask
        del mask, tree_edge_mask

        # Drop the internal and duplicated edges (as selected above)
        # Use sub-graph IDs as node ID
        graph_edges = parent_ids[np.newaxis, *graph_edges[..., ext_edge_mask]]

def generate_node_addresses(graph_edges: T_Graph) -> T_IndexArray:
    """Assign address to each node based on sub-graph hierarchy

    Parameters
    ----------
    graph_edges : (node-coord, 2, edge-indices) ndarray of int
        The graph

    Returns
    -------
    node_addr : (addr-comp, node-indices) ndarray of int
        Multi-component address for each node
    """
    # Group nodes into parent-nodes, then repeat.
    # Each iteration "adjusts" the node-IDs from previous one, according to the parents
    node_addr = None
    for parent_ids in node_parent_gen(graph_edges):
        # Append parent IDs to the address
        if node_addr is None:
            # This is the first address component
            node_addr = parent_ids[np.newaxis]
        else:
            # Normalize previous address components, using their parent IDs.
            # Keep both in `node_addr`
            norm_ids = normalize_id(parent_ids)
            norm_ids = norm_ids[:, node_addr[-1]]
            # Restore the invalid-node markers
            norm_ids[:, node_addr[-1] < 0] = -1
            node_addr = np.concatenate((node_addr[:-1], norm_ids), axis=0)
            del norm_ids

    # Dummy/empty address array, when the graph was empty
    if node_addr is None:
        return np.empty((0,) * (graph_edges.shape[0] + 1), dtype=int)
    return node_addr
