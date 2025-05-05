"""Build list of all edges between neighbor nodes from a DEM grid

Edges are sorted by altitude and slope keys, i.e:
First, by altitude of the lower node, where these altitudes are equal - by the edge-slope.

The edge-slope requires external `distance` callback. With no such parameter,
the altitude difference is used.
"""
import itertools
from typing import Callable, Iterator
import numpy as np
import numpy.typing as npt
from .topo_graph import T_Graph, T_IndexArray, T_NodeValues


# Prepare edges, by batches of source-nodes (reduce memory consumption)
PREP_EDGE_LIMIT = 1_000_000     # ~1M edges prepared at once

def get_rel_neighbors(ndim: int, diag_lvl: int=1):
    """Relative indices of neighbor nodes (no opposites)

    - Order of returned relative neighbor indices (4 for 2D):
      | node  | node      | node       |
      |-------|-----------|------------|
      | _dup_ | 3: (0,+1) | 2: (+1,-1) |
      | _dup_ | **src**   | 1: (+1,0)  |
      | _dup_ | _dup_     | 0: (+1,+1) |
    """
    res = 1 - np.indices((3,) * ndim)           # Grid: 1, 0, -1
    res = res[:, res.any(0)]                    # Drop the origin
    res = res[:, :res.shape[1]//2]              # Drop opposites
    assert (res != -res.T[..., np.newaxis]).any(1).all(), 'Opposite directions remain'
    return res[:, np.count_nonzero(res, axis=0) <= diag_lvl+1]  # Drop higher diagonal levels

def prepare_edges(alt_grid: T_NodeValues, *, distance: Callable|None=None, batch_size: int
                  ) -> Iterator[tuple[T_Graph, T_NodeValues]]:
    """Prepare edge generator"""
    # Obtain neighbors of each point
    # shape is: (target-xy, edge-neighbor, source-x, source-y)
    tgt_rel_idxs = get_rel_neighbors(alt_grid.ndim)
    iter = np.ndindex(alt_grid.shape)
    while src_idx := itertools.islice(iter, batch_size // tgt_rel_idxs[0].size):
        # One bundle of nodes at a time
        src_idx = np.fromiter(src_idx, dtype=(int, alt_grid.ndim)).T
        if src_idx.size == 0:
            return      # All nodes are processed
        src_idx = src_idx[:, np.newaxis]
        tgt_idx = (src_idx.T + tgt_rel_idxs.T).T

        # Drop the edges, where target-node is out-of-bounds
        mask = (tgt_idx.T < alt_grid.shape).T.all(0)
        mask &= (tgt_idx >= 0).all(0)
        src_idx = np.broadcast_to(src_idx, shape=tgt_idx.shape)[:, mask]
        tgt_idx = tgt_idx[:, mask]
        del mask

        # Get source-node altitude and slope (or just altitude difference)
        edge_list: T_Graph = np.stack((src_idx, tgt_idx), axis=1)
        del src_idx, tgt_idx

        node_alt = alt_grid[*edge_list]
        # Drop the edges, involving NaN
        mask = ~np.isnan(node_alt).any(0)
        if not mask.all():
            node_alt = node_alt[:, mask]
            edge_list = edge_list[..., mask]
        slope = node_alt[1] - node_alt[0]
        node_alt = np.where(node_alt[1] > node_alt[0], node_alt[0], node_alt[1])
        # Use actual slope arctan(vert / hor)
        if distance is not None:
            # Distance between source and target nodes (must be "flat")
            edge_len = distance(edge_list[:,0], edge_list[:,1])
            slope = np.arctan2(slope, edge_len)

        # Swap descending edges (make all slopes positive)
        mask = slope < 0
        edge_list[..., mask] = edge_list[:, ::-1, mask]
        slope[mask] = np.negative(slope[mask])
        yield edge_list, np.stack((slope, node_alt))

def build_graph_edges(alt_grid: npt.NDArray, *, distance: Callable|None=None
                      ) -> tuple[T_Graph, T_IndexArray]:
    """Get sorted list of all edges between neighbors

    Parameters
    ----------
    alt_grid : (node-indices, 2, edge-indices) ndarray
        Node altitude grid
    distance : callable [optional]
        Function to calculate distance between nodes

    Returns
    -------
    graph_edges : (node-coord, 2, edge-index) ndarray of ints
        Edges between neighboring nodes
        - the first node (source) of each edge is the lower-altitude one
        - edges are in descending order by lower-altitude/source node
          (first edge is the one with highest lower node)
        - lower-altitude node is first
        - no duplications of opposites
        - no nodes of invalid altitude or outsize grid boundary

    lexsort_keys: (slope-alt, edge-index) ndarray of ints
        `numpy.lexsort()` keys for further list-merge
    """
    # Prepare edges in batches of limited size, then recombine
    edge_list: list[T_Graph] = []
    lexsort_keys = []
    for edges, keys in prepare_edges(alt_grid, distance=distance, batch_size=PREP_EDGE_LIMIT):
        edge_list.append(edges)
        lexsort_keys.append(keys)
    graph_edges: T_Graph = np.concatenate(edge_list, axis=-1)
    del edge_list
    lexsort_keys = np.concatenate(lexsort_keys, axis=-1)

    # Sort by descending source-node altitude
    # For the same point, the steepest edge must come first
    # > use lexsort() to sort by multiple keys: primary key in last column
    alt_lexsort = np.lexsort(lexsort_keys)[::-1]
    graph_edges = graph_edges[..., alt_lexsort]
    lexsort_keys = lexsort_keys[:, alt_lexsort]
    return graph_edges, lexsort_keys
