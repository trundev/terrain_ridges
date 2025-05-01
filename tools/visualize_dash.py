"""Visualization of internal graph structures (for development purposes)

See: https://dash.plotly.com/ and https://js.cytoscape.org/

[Hint] visualization one-liners:
- Run server
  >>> py tools/visualize_graph-server.py
- Visualize graph
  >>> from tools import visualize_dash as vis
  >>> vis.visualize_graph(graph_edges, passive_edges=True, node_class=True, node_parent=True)
"""
import sys
import itertools
import uuid
import urllib.parse
import numpy as np
import numpy.typing as npt
import requests
import plotly.colors as clrs
from terrain_ridges import topo_graph
from terrain_ridges.topo_graph import T_Graph, T_IndexArray, T_MaskArray


# Obtain node position (layout - preset) by scaling node ID
MODEL_POSITION_STEP = 120
# Parent node ID to assign to invalid nodes
INVALID_PARENT_ID = 'invalid'

def vals_to_colorscale(val: npt.NDArray[np.floating], colors: list[str]|None=None
                       ) -> npt.NDArray[np.str_]:
    """Select colors for individual values, must be in range [0, 1]"""
    if colors is None:
        colors = clrs.sequential.Viridis
    return np.asarray(colors)[np.round(val * (len(colors) - 1)).astype(int) % len(colors)]

def val_array_to_colorscale(vals: T_IndexArray, **kwargs) -> npt.NDArray[np.str_]:
    """Select colors for all values from an array"""
    return vals_to_colorscale((vals - vals.min()) / (vals.max() - vals.min()), **kwargs)

#
# Graph/edge visualization
#
def graph_to_elements(graph_edges: T_Graph, edge_mask: T_MaskArray|bool=True, *,
                      passive_edges: T_MaskArray|bool=False,
                      node_class: npt.NDArray[np.str_]|None=None,
                      node_parent: T_IndexArray|None=None) -> list[dict[str, str|int]]:
    """Create cytoscape elements list from graph (list of edges)

    Parameters
    ----------
    graph_edges : (node-coord, 2, edge-indices) ndarray of int
        Graph to visualize
    edge_mask : (edge-indices) ndarray of bool [optional]
        Mask of graph-edges to visualize, for correct edge index text-labels
    passive_edges : (edge-indices) ndarray of bool [optional]
        Mask of graph-edges to visualize using 'edge.passive' selector,
        where it overlaps `edge_mask`, 'edge.ghost' is used
    node_class : (node-index) ndarray of str [optional]
        Add class selector to nodes, to indicate node-type, currently available:
        'leaf', 'root', 'loop' and 'invalid'
    node_parent : (node-index) ndarray [optional]
        Attach nodes to "compound parent" nodes, see:
        https://js.cytoscape.org/#notation/compound-nodes
    """
    edge_mask = np.broadcast_to(edge_mask, graph_edges.shape[2:])
    passive_edges = np.broadcast_to(passive_edges, graph_edges.shape[2:])
    # Color nodes based on parent
    node_color = None
    if node_parent is not None:
        node_color = val_array_to_colorscale(node_parent)

    elements = []
    # Add nodes, structure:
    # - data:
    #   - id: <node-id>
    #   - label: <node label>
    #   - parent: <node-id>
    #   classes: [<class-id>]
    #   style: {background-color: <color>}
    for node_id in np.unique(graph_edges.reshape(graph_edges.shape[:1] + (-1,)), axis=1).T:
        data = dict(
            id=str(node_id),
            label=f'node: {node_id}')
        element = dict(data=data)
        # Parent node, use INVALID_PARENT_ID for invalid nodes
        if node_parent is not None:
            data |= dict(parent=str(node_parent[*node_id])
                                    if (node_id >= 0).all() else INVALID_PARENT_ID)
        # Regular vs. invalid node
        if (node_id >= 0).all():
            # Node type class
            if node_class is not None:
                element |= dict(classes=node_class.item(*node_id))
            # Assign colors to nodes
            if node_color is not None:
                element |= dict(style={'background-color': node_color.item(*node_id)})
        else:
            element |= dict(classes='invalid')
        # Preset position (avoid numpy types as they are not JSON serializable)
        if node_id.size >= 2:
            element |= dict(position={k: v.item(0) for k, v in
                                      zip('xy', node_id * MODEL_POSITION_STEP)})
        elements.append(element)
    # Add parent-nodes, if necessary the one for invalid nodes
    if node_parent is not None:
        parent_ids = np.unique(node_parent[node_parent >= 0])
        if (graph_edges < 0).any():
            parent_ids = itertools.chain(parent_ids, [INVALID_PARENT_ID])
        for node_id in parent_ids:
            elements.append(dict(
                    data=dict(
                        id=str(node_id),
                        label=f'parent: {node_id}'),
                    classes='parent'))

    # Add edges, structure:
    # - data:
    #   - source: <node-id>
    #   - target: <node-id>
    #   - label: <edge label>
    #   classes: [<class-id>]
    for idx in np.asarray(np.nonzero(edge_mask | passive_edges)).T:
        # Edge class is selected from `edge_mask` and `passive_edges`
        classes = ''
        if passive_edges.item(*idx):
            classes=('ghost', 'passive')[edge_mask.item(*idx)]
        elements.append(dict(
                data=dict(
                    source=str(graph_edges[:, 0, *idx]),
                    target=str(graph_edges[:, 1, *idx]),
                    label=f'edge: {np.asarray(idx)}'),
                classes=classes))
    return elements

#
# Client side code
#
GRAPH_ID = str(uuid.uuid4())
SERVER_URL = 'http://127.0.0.1:8050/cytoscape?'

def post_server_request(url_query: dict[str, str], data: list[dict]) -> str|None:
    """Post request to the server (visualize_dash-server.py)"""
    url = SERVER_URL + urllib.parse.urlencode(url_query)
    try:
        res = requests.post(url, json=data)
    except requests.exceptions.ConnectionError as ex:
        print(ex)
        print()
        print('Must run dash visualization server:')
        print('>>> python tools/visualize_dash-server.py')
        return None
    return res.text

def visualize_graph(graph_edges: T_Graph, *, id: str=GRAPH_ID, **kwargs) -> str|None:
    """Convert graph to cytoscape format and post it to the server

    See graph_to_elements()
    """
    # Auto-generate `passive_edges`, `node_class`, etc.
    if kwargs.get('passive_edges') is True:
        kwargs['passive_edges'] = ~topo_graph.valid_node_edges(graph_edges)
    if kwargs.get('node_class') is True:
        root_nodes = topo_graph.isolate_graph_sinks(graph_edges)
        leaf_nodes = topo_graph.isolate_graph_sinks(graph_edges[:, ::-1])
        loop_nodes = topo_graph.shrink_mask(graph_edges, np.broadcast_to(True, leaf_nodes.shape))
        loop_nodes = topo_graph.shrink_mask(graph_edges[:, ::-1], loop_nodes)
        kwargs['node_class'] = np.where(loop_nodes, 'loop',
                np.where(root_nodes, 'root', np.where(leaf_nodes, 'leaf', '')))
    if kwargs.get('node_parent') is True:
        parent_ids = topo_graph.isolate_subgraphs(graph_edges)
        kwargs['node_parent'] = parent_ids

    cyto_elements = graph_to_elements(graph_edges, **kwargs)
    return post_server_request(dict(id=id, title=f'Graph - edges: {graph_edges.shape[2:]}'),
                               data=cyto_elements)

#
# Main entry
#
def main(argv) -> int:
    """Standalone execution (TODO: import data from file)"""
    # Show some demo graph
    grid_edges = np.array((
            #
            # Straight line from leaf (0,0) to root (0,2)
            [(0,0), (0,1)], [(0,1), (0,2)],
            #
            # Loop
            [(1,0), (1,1)], [(1,1), (1,0)],
            # Loop toward root
            [(1,1), (1,2)], [(1,2), (1,3)], # root (1,3)
            # Toward the loop, leaf (1,5)
            [(1,5), (1,4)], [(1,4), (1,0)],
            #
            # Invalid source-node to root (2,1)
            [(2,-1), (2,0)], [(2,0), (2,1)],
            #
            # Invalid target-node from leaf (3,0)
            [(3,0), (3,1)], [(3,1), (3,-1)],
            #
            # Invalid source and target-nodes
            [(-1,-1), (4,-1)],
        ), dtype=int).T
    res = visualize_graph(grid_edges, passive_edges=True, node_class=True, node_parent=True)
    print(res)
    return 0

if __name__ == '__main__':
    exit(main(sys.argv[1:]))
