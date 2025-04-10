"""Visualization for test data, using dash-cytoscape (not actual tests)

WARNING: This will block in `dash_context()` after all tests are completes,
waiting for `dash.Dash.app` server.

Requires dash-cytoscape:
    pip install dash dash-cytoscape
"""
import numpy as np
import numpy.typing as npt
import pytest
from . import conftest  # Imported by pytest anyway
from terrain_ridges import build_edges, topo_graph
from terrain_ridges.topo_graph import T_Graph, T_IndexArray, T_MaskArray
#from test_visualize import visualize_graph


# Skip the whole module with "not visualize" marker (default)
pytestmark = pytest.mark.visualize

DASH_STYLE = dict(width='100%', height='600px')
DASH_STYLE_SHEET = [
    dict(selector='node', style={
            'content': 'data(label)',
            }),
    dict(selector='edge', style={
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            }),
    ]

@pytest.fixture(scope='session')    #TODO: Try 'package', 'module'
def dash_context():
    """Create dash application fixture (one per session)"""
    dash = pytest.importorskip('dash')
    app = dash.Dash(__name__)
    app.layout = [dash.html.H2(__name__)]
    yield app, dash, pytest.importorskip('dash_cytoscape')
    app.run(debug=False)    #TODO: Try True

@pytest.fixture(scope='function')
def cyto_elements(dash_context):
    """Create cytoscape element fixture (one per test)"""
    dash_app, dash, cyto = dash_context
    title = dash.html.Legend('Loading...')
    elements = []
    dash_app.layout.append(dash.html.Fieldset([title,
                cyto.Cytoscape(elements=elements,
                    style=DASH_STYLE,
                    stylesheet=DASH_STYLE_SHEET,
                ),
            ]))
    return title, elements

def visualize_edge_list(elements, edge_list: T_IndexArray, altitude_grid: npt.NDArray|None, *,
                        edge_src_mask: T_MaskArray|None=None) -> None:
    """Visualize edge list using dash_cytoscape"""
    for idx in np.ndindex(edge_list.shape[2:]):
        elements.append(dict(data=dict(
                    source=str(edge_list[:, 0, *idx]),
                    target=str(edge_list[:, 1, *idx]),
                    label=f'edge: {idx}',
                ),
            ))
    for node_id in np.unique(edge_list.reshape(edge_list.shape[:1] + (-1,)), axis=1).T:
        if altitude_grid is None:
            label = ''
        else:
            label = f', alt: {altitude_grid[*node_id]}'
        elements.append(dict(data=dict(
                id=str(node_id),
                label=f'node: {node_id}' + label,
            )))

def test_select_edges(cyto_elements, altitude_grid):
    """Visualize edge-selection from the initial edge list"""
    title, elements = cyto_elements
    edge_list, _ = build_edges.build_edge_list(altitude_grid)
    #TODO: select_edges() need "raveled" indices (mey need to change this later)
    edge_list_r = np.ravel_multi_index(tuple(edge_list), altitude_grid.shape).astype(int)
    edge_src_mask = topo_graph.select_edges(edge_list_r)
    mask = edge_src_mask.any(0)

    title.children = f'Selected edges - edges: {edge_list.shape[2:]}, grid: {altitude_grid.shape}'
    visualize_edge_list(elements, edge_list[..., mask], altitude_grid,
                        edge_src_mask=edge_src_mask[..., mask])

def test_edge_list_to_graph(cyto_elements, altitude_grid):
    """Visualize tree-graph created from initial edge list"""
    title, elements = cyto_elements
    edge_list, _ = build_edges.build_edge_list(altitude_grid)
    #TODO: select_edges() need "raveled" indices (mey need to change this later)
    edge_list_r = np.ravel_multi_index(tuple(edge_list), altitude_grid.shape).astype(int)
    edge_src_mask = topo_graph.select_edges(edge_list_r)
    tgt_nodes = topo_graph.edge_list_to_graph(edge_list_r, edge_src_mask)
    # Unravel node indices, same as reshape graph
    tgt_nodes = topo_graph.reshape_graph(tgt_nodes, altitude_grid.shape)

    title.children = f'Graph - edges: {edge_list.shape[2:]}, grid: {altitude_grid.shape}'
    visualize_graph(elements, dymmy_edge_list[...,2:], None)
