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

    The node IDs are represented by an array of unique elements for each node. The node ID is the
    index of the element, parent ID is its value. Parent IDs must be indices (consecutive unique
    numbers starting from zero) or `-1` for invalid nodes. If not, this can be achieved by:
    >>> parent_ids = np.unique_inverse(parent_ids)[1]

    Separate IDs will be assigned to every node where `parent_ids == -1`.

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

    # Return normalized child-node IDs
    return normalized

def normalize_node_addr(node_addr: T_IndexArray) -> T_IndexArray:
    """Normalize the result from generate_node_addresses()

    Convert all node-address components to indices (consecutive numbers starting from zero).
    This rule apply to every group of components with the same parent component, thus the
    different parent groups can have same IDs at lower levels.

    A separate valid addresses are assigned to every invalid node ID (`-1`).

    Parameters
    ----------
    node_addr : (addr-comp, node-indices) ndarray of int
        Non-normalized node addresses as returned by generate_node_addresses()

    Returns
    -------
    node_addr : (addr-comp, node-indices) ndarray of int
        Normalized multi-component address for each node
    """
    for comp in range(node_addr.shape[0]):
        _, uniq_idx, uniq_inv = np.unique(node_addr[comp], return_index=True,  return_inverse=True)
        # Note: Use fake (all zeros) parent component of top of the last one
        parent_ids = node_addr[comp + 1] if comp + 1 < node_addr.shape[0] else \
                np.zeros_like(node_addr[0])
        if True:    #HACK: Check if each component level is a subset of the next one
            np.testing.assert_equal(parent_ids.flat[uniq_idx][uniq_inv], parent_ids,
                                    'Different parent IDs for the same node ID')
        norm_ids = normalize_id(parent_ids.flat[uniq_idx])
        node_addr[comp] = norm_ids[uniq_inv]
    return node_addr

def node_parent_gen(graph_edges: T_Graph, *, node_shape: T_IndexArray) -> Iterator[T_IndexArray]:
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
    edge_mask = True
    node_ids = np.indices(tuple(node_shape))
    # Iterate until the graph collapses, each iteration groups the nodes from the
    # previous one into sub-graphs, which in turn are treated as nodes by the next one
    while np.any(edge_mask):
        tree_edge_mask = topo_graph.filter_treegraph(graph_edges, edge_mask=edge_mask,
                                                     node_ids=node_ids)
        parent_ids = topo_graph.isolate_subgraphs(node_ids[:, *graph_edges[..., tree_edge_mask]],
                                                  node_shape=node_shape)
        yield parent_ids

        # `isolate_subgraphs()` will do this, but only for nodes in parent-graph
        node_shape = parent_ids.max() + 1
        # Map parent IDs to node-grid
        node_ids = parent_ids[np.newaxis, *node_ids]

        # Identify edges between sub-graphs, skip internal ones
        edge_mask &= ~tree_edge_mask
        par_graph_edges = node_ids[:, *graph_edges[..., edge_mask]]
        mask = (par_graph_edges[:, 0] != par_graph_edges[:, 1]).any(0)
        edge_mask[edge_mask] = mask

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
    node_shape = topo_graph.graph_node_min_shape(graph_edges)
    node_addr = np.arange(np.prod(node_shape)).reshape(1, *node_shape)
    # Group nodes into parent-nodes, then repeat.
    # Each iteration "adjusts" the node-IDs from previous one, according to the parents
    for parent_ids in node_parent_gen(graph_edges, node_shape=node_shape):
        # Remap returned parent IDs to node-grid
        parent_ids = parent_ids.flat[node_addr[-1]]
        # Restore the invalid-node markers ????
        parent_ids[node_addr[-1] < 0] = -1
        node_addr = np.concatenate((node_addr, parent_ids[np.newaxis]), axis=0)

    #HACK: Return normalized addresses, but w/o the base component
    return normalize_node_addr(node_addr[1:])
