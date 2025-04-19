"""Topology graph implementation

Graph is represented by list of edges. Each edge consist of two node indices/IDs:
for source and target node. Node order is usually relevant, but can be swapped.

Node indices are multi-dimensional vectors, usually of 2 or 1 components:
- 2D is used when referring to points in original DEM grid
- 1D is used when referring to sub-graphs (point groups) isolated by previous step

Graph edge-list array dimensions:
1. Node coordinates, shape is typically of 1 or 2
   Array values are the the node index/ID. Typically used as indices in internal arrays,
   so values must be limited (can be -1 for invalid/external nodes).
2. Source-target node, shape is always 2
   Index 0 - source, 1 - target
3. Edge index, shape corresponds to the number of edges
   Multi-dimensional index may be allowed for future development

Note:
- The tree-graphs are defined as ones, where nodes have single associated target-node.
  In the graph theory therms, nodes have single (or none) incoming nodes (may be confusing).
- Sink-nodes are the ones with no target-node, which corresponds the the tree-roots (no
  incoming edge in the graph theory)
- Generic graphs can have loops, self-edges, duplicated edges, root- and leaf- nodes.
  Self-edges are typically not used, duplicates and loops are dropped by tree-graph
  filtering process.
"""
import numpy as np
import numpy.typing as npt
from typing import Callable


# Type aliases
type T_MaskArray = npt.NDArray[np.bool]
type T_IndexArray = npt.NDArray[np.int_]
#type T_Graph = np.ndarray[tuple[int, Literal[2], int], np.dtype[np.int_]]
type T_Graph = np.ndarray[tuple[int, ...], np.dtype[np.int_]]   # or just T_IndexArray
type T_NodeValues = npt.NDArray

#
# Generic utils
#
def unique_mask(arr: T_IndexArray, axis: int|None=None) -> T_MaskArray:
    """Returns the mask of unique elements in an array"""
    _, unique_idx = np.unique(arr, axis=axis, return_index=True)
    result = np.zeros(arr.size if axis is None else arr.shape[axis], dtype=bool)
    result[unique_idx] = True
    return result

#
# Graph manipulations
#
def assert_graph_shape(graph_edges: T_Graph) -> None:
    """Validate shape and indices of a graph representation"""
    assert graph_edges.ndim > 2, 'Missing edge index dimension(s)'
    assert graph_edges.shape[1] == 2, 'Incorrect edge source-target dimension'
    np.testing.assert_equal(graph_edges >= -1, True, 'Incorrect node index (-1 still allowed)')

def graph_node_min_shape(graph_edges: T_Graph) -> T_IndexArray:
    """Minimum ndarray shape to accommodate graph's node grid"""
    return graph_edges.reshape(graph_edges.shape[0], -1).max(axis=1) + 1

def broadcast_node_vals(graph_edges: T_Graph, node_vals: T_NodeValues|int) -> T_NodeValues:
    """Broadcast array of values to the shape of graph's node grid"""
    node_vals = np.asarray(node_vals)
    # Check if `node_vals` needs broadcasting
    if node_vals.ndim < graph_edges.shape[0]:
        node_shape = graph_node_min_shape(graph_edges)
        node_vals = np.broadcast_to(node_vals, node_shape)
    return node_vals

def valid_node_edges(graph_edges: T_Graph, both: bool=True) -> T_MaskArray:
    """Get mask of edges with valid nodes (all node-index components non-negative)

    Parameters
    ----------
    graph_edges : (node-coord, 2, edge-indices) ndarray of int
        Source graph
    both : bool
        When False, returns separate flags for source and target nodes, otherwise single flag

    Returns
    -------
    edge_mask : (edge-indices) or (2, edge-indices) ndarray of bool
    """
    return np.all(graph_edges >= 0, axis=(0, 1) if both else 0)

def filter_treegraph(graph_edges: T_Graph, edge_mask: T_MaskArray|bool=True, *,
                     node_ids: T_IndexArray|None=None) -> T_MaskArray:
    """Filter graph to a tree-style, by selecting edges with unique source-nodes only

    Edge selection process:
    - Select unique/fist nodes based on their ID or coordinates. Edge order takes precedence
      over node order within the edges. For example, the target node of the first edge is selected
      instead of the source node of the second edge.
    - The ID option is allows nodes from the same sub-graph to be treated as non-unique (first one
      is picked, others - ignored).
    - Select edges with at least one selected node
    - Edges, where only target-node is selected are swapped (`graph_edges` is modified inplace)

    Parameters
    ----------
    graph_edges : (node-coord, 2, edge-indices) ndarray of int
        Source graph, edges with smaller index are picked first, i.e.
        in case of nd-index, lower dim is prioritized - `[...,0,1]` before `[...,1,0]`
        (the array will be modified in-place, by swapping edges, where target-node was selected)
    edge_mask : (edge-indices) ndarray of bool [optional]
        Mask of edges to operate on, intended to:
        - Shape of the returned mask to match the actual graph
        - Graph inplace modifications to be seen by the caller (not possible on masked copy)
    node_ids : (node-indices) ndarray of int [optional]
        Node coordinate to ID map, if omitted the coordinates are used instead

    Returns
    -------
    edge_mask : (edge-indices) ndarray of bool
        Mask of selected edges from original graph
        Graph is inplace modified (swapped edges) for some of these entries
    """
    # Pick edges from `edge_mask`, convert node coordinates to IDs (if necessary)
    edge_mask = np.broadcast_to(edge_mask, graph_edges.shape[2:]).copy()
    real_edges = graph_edges[..., edge_mask]
    if node_ids is not None:
        real_edges = node_ids[..., *real_edges]

    # Select the first occurrence of a node (by ID) from either the source or target side
    # of each edge in the graph, edge order takes precedence.
    # Note:
    # Array transposing is to preserve node order when reshaping/masking:
    # both source/target nodes from the top edge remain on top
    edge_src_mask = unique_mask(real_edges.T.reshape(real_edges.shape[-1] * 2, -1), axis=0)
    edge_src_mask = edge_src_mask.reshape(real_edges.shape[-1], 2).T

    # Swap edges (in-place), where target-node was selected
    swap_mask = edge_mask.copy()
    swap_mask[swap_mask] = ~edge_src_mask[0] & edge_src_mask[1]
    graph_edges[..., swap_mask] = graph_edges[:, ::-1, swap_mask]

    # Convert the selected edges to original graph shape
    edge_mask[edge_mask] = edge_src_mask.any(0)
    return edge_mask

def isolate_graph_sinks(graph_edges: T_Graph, *, node_shape: T_IndexArray|None=None) -> T_MaskArray:
    """Identify sink-nodes inside graph (ones w/o target-nodes)

    Hint:
    To isolate tree-graph leaf-nodes, reverse the source-target axis
    >>> isolate_graph_sinks(graph_edges[:, ::-1])

    Parameters
    ----------
    graph_edges : (node-coord, 2, edge-indices) ndarray of int
        Source graph
    node_shape : list of int [optional]
        Shape of node-grid

    Returns
    -------
    sink_mask : (node-indices) ndarray of bool
        Mask of where graph-nodes are pointed to by edge target-nodes, but not by source one
    """
    if node_shape is None:
        node_shape = graph_node_min_shape(graph_edges)

    # Identify edge target-nodes w/o source one pointing to
    # Note: target-node from edge with invalid source-node (and vice-versa) are still counted
    sink_mask = np.zeros(node_shape, dtype=bool)
    valid_edges = valid_node_edges(graph_edges, both=False)
    sink_mask[*graph_edges[:, 1, valid_edges[1]]] = True
    sink_mask[*graph_edges[:, 0, valid_edges[0]]] = False
    return sink_mask

def accumulate_src_vals(graph_edges: T_Graph, node_vals: T_NodeValues|int, *,
                        inval_src: T_NodeValues|int=0) -> T_NodeValues:
    """Accumulate all source-node values at the target one

    This uses unbuffered in place operation, see "numpy.ufunc.at", which allows
    target index overlap (much slower).

    Parameters
    ----------
    graph_edges : (node-coord, 2, edge-indices) ndarray of int
        The graph
    node_vals : (node-indices) ndarray
        Grid of values for each node
    inval_src
        Value generated by edges with invalid source-node

    Returns
    -------
    node_vals : (node-indices) ndarray
    """
    assert_graph_shape(graph_edges)
    node_vals = broadcast_node_vals(graph_edges, node_vals)

    # Start by accumulating `inval_src` value, to target-nodes of edges with invalid sources
    res_vals = np.zeros_like(node_vals)
    valid_edges = valid_node_edges(graph_edges, both=False)
    np.add.at(res_vals, tuple(graph_edges[:, 1, valid_edges[1] & ~valid_edges[0]]), inval_src)

    # Then accumulate along edges where both nodes are valid
    graph_edges = graph_edges[..., valid_edges.all(0)]
    np.add.at(res_vals, tuple(graph_edges[:, 1]), node_vals[*graph_edges[:, 0]])
    return res_vals

def equalize_subgraph_vals(graph_edges: T_Graph, node_vals: T_NodeValues) -> T_NodeValues:
    """Equalize (spread) node values along sub-graphs

    Take minimum of node values on both sides of every edge iteratively.
    Can be used to assign unique IDs to sub-graphs.

    This uses unbuffered in place operation, see "numpy.ufunc.at", which allows
    target index overlap (much slower).

    Parameters
    ----------
    graph_edges : (node-coord, 2, edge-indices) ndarray of int
        The graph
    node_vals : (node-indices) ndarray
        Grid of values for each node
        (the array will be modified in-place)

    Returns
    -------
    node_vals : (node-indices) ndarray
        The equalized values
        (this is the exact `node_vals` input array)
    """
    # Need only edges where both nodes are valid
    graph_edges = graph_edges[..., valid_node_edges(graph_edges)]

    last_vals = np.zeros_like(node_vals)
    while (node_vals != last_vals).any():
        last_vals[...] = node_vals
        # Along the edges: source to target node
        np.minimum.at(node_vals, tuple(graph_edges[:, 1]), node_vals[*graph_edges[:, 0]])
        # In reverse: target to source node
        np.minimum.at(node_vals, tuple(graph_edges[:, 0]), node_vals[*graph_edges[:, 1]])
    return node_vals

def propagate_mask(graph_edges: T_Graph, node_mask: T_MaskArray|bool, *,
                   operation: Callable[[T_MaskArray, T_MaskArray], T_MaskArray]=np.logical_and,
                   inval_src: bool=False) -> T_MaskArray:
    """Propagate node-mask along graph edges

    This propagates only `True` mask values, to avoid use of unbuffered in place operation
    (much faster).

    Hint:
    To isolate graph-loop, run in both directions using contract-mode (`logical_and`):
    >>> loop_mask = propagate_mask(graph_edges, True)
    >>> loop_mask = propagate_mask(graph_edges[:, ::-1], loop_mask)

    Parameters
    ----------
    graph_edges : (node-coord, 2, edge-indices) ndarray of int
        The graph
    node_mask : (node-indices) ndarray of bool
        Mask to propagate
    operation : callable
        Operation to apply at each step:
        - `logical_and` contract the mask
        - `logical_or` expand the mask
    inval_src : bool
        Value generated by edges with invalid source-node

    Returns
    -------
    node_mask : (node-indices) ndarray
    """
    assert_graph_shape(graph_edges)
    node_mask = broadcast_node_vals(graph_edges, node_mask).copy()

    # Handle edges, where source-node index is invalid, but target is valid:
    # the `inval_src` value will be applied to their target-node as a first iteration
    valid_edges = valid_node_edges(graph_edges, both=False)
    node_mask[*graph_edges[:, 1, ~valid_edges[0] & valid_edges[1]]] = inval_src

    # Iterate until node-mask values are NOT changed,
    # using only edges where both nodes are valid
    graph_edges = graph_edges[..., valid_edges.all(0)]
    last_mask = False
    while np.any(node_mask != last_mask):
        last_mask = node_mask
        # Set edge's target-node mask, where source-node is set
        # (copy source to target, allowing index overlap)
        node_mask = np.zeros_like(node_mask)
        node_mask[*graph_edges[:, 1, last_mask[*graph_edges[:, 0]]]] = True
        # Apply operation to expand or contact the mask
        node_mask = operation(node_mask, last_mask)
    return node_mask

def isolate_subgraphs_safe(graph_edges: T_Graph, *, node_shape: T_IndexArray|None=None) -> T_IndexArray:
    """Identify isolated sub-graphs (parent nodes), non-optimized implementation

    Parameters
    ----------
    graph_edges : (node-coord, 2, edge-indices) ndarray of int
        Source graph
    node_shape : list of int [optional]
        Shape of node-grid

    Returns
    -------
    parent_node_ids : (node-indices) ndarray of int
        Unique parent node IDs for each graph-node, -1 for unused ones
    """
    # Start with unique parent IDs for each valid graph-node
    if node_shape is None:
        node_shape = graph_node_min_shape(graph_edges)
    valid_nodes = np.zeros(node_shape, dtype=bool)
    valid_edges = valid_node_edges(graph_edges, both=False)
    valid_nodes[*graph_edges[:, valid_edges]] = True
    parent_ids = np.full(node_shape, -1)    # `-1` will remain at unused nodes
    parent_ids[valid_nodes] = np.arange(np.count_nonzero(valid_nodes))

    parent_ids = equalize_subgraph_vals(graph_edges, parent_ids)

    # Convert parent IDs to indices (still keep the `-1`)
    parent_ids[valid_nodes] = np.unique_inverse(parent_ids[valid_nodes])[1]
    return parent_ids

def isolate_subgraphs(graph_edges: T_Graph, *, node_shape: T_IndexArray|None=None) -> T_IndexArray:
    """Identify isolated sub-graphs, optimized implementation"""
    #TODO: Replace this with an algorithm to run equalize_subgraph_vals on "core" nodes only
    return isolate_subgraphs_safe(graph_edges, node_shape=node_shape)
