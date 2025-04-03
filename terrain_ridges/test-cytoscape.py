"""Tool to visualize node-graph using dash-cytoscape

See:
    https://dash.plotly.com/cytoscape

Requires dash-cytoscape:
    pip install dash dash-cytoscape
"""
import numpy as np
import numpy.typing as npt
import topo_graph
from dash import Dash, html
import dash_cytoscape as cyto


DASH_LAYOUT = {'name': 'preset'}
#DASH_LAYOUT = {'name': 'breadthfirst', 'spacingFactor': .8, 'roots': '.seed[layer=0]'}
DASH_STYLE = {'width': '100%', 'height': '600px'}
DASH_STYLE_SHEET = [
    # Group selectors
    {
        'selector': 'node',
        'style': {
            'content': 'data(id)',  #TODO: 'data(label)'
            'min-zoomed-font-size': 9,
        }
    },
    {
        'selector': 'edge',
        'style': {
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            'text-rotation': 'autorotate',
            'min-zoomed-font-size': 9,
        }
    },
    # Class selectors
    {
        'selector': '.tree',
        'style': {
            'opacity': '.7',
        }
    },
    # Seed/root nodes
    {
        'selector': '.seed[layer=0]',
        'style': {
            'background-color': 'red',
            'shape': 'ellipse',
        }
    },
    {
        'selector': '.seed:selected[layer=0]',
        'style': {
            'background-color': 'blue',
        }
    },
    {
        'selector': '.seed.tree',
        'style': {
            'border-color': 'red',
        }
    },
    # Leaf nodes
    {
        'selector': '.leaf[layer=0]',
        'style': {
            'background-color': 'green',
            'shape': 'round-triangle',
        },
    },
    {
        'selector': '.leaf:selected[layer=0]',
        'style': {
            'background-color': 'blue',
        }
    },
    {
        'selector': '.leaf.tree',
        'style': {
            'border-color': 'green',
        },
    },
    # Regular nodes
    {
        'selector': '.normal[layer=0]',
        'style': {
            'shape': 'round-rectangle',
        }
    },
    # Shadow (lower layer) edges
    {
        'selector': '.shadow',
        'style': {
            'line-style': 'dashed',
            'width': 1,
            'opacity': '.5',
        }
    },
    # Ghost (unused/future) edges
    {
        'selector': '.ghost',
        'style': {
            'target-arrow-shape': 'none',
            'line-style': 'dashed',
            'line-dash-pattern': [3,10],
            'opacity': '.5',
        }
    },
]

def visualize_graph_hierarchy(graph_list: list[topo_graph.T_Graph],
        edge_list: topo_graph.T_IndexArray, edge_src_mask: topo_graph.T_MaskArray, *,
        node_pos: npt.NDArray|None=None) -> None:
    """Visualize topo-graph chain using dash-cytoscape"""
    # Check argument consistency
    assert edge_src_mask.shape == edge_list.shape[1:], 'Mismatch between edge_list and edge_src_mask'
    for layer, par_nodes in enumerate(graph_list):
        topo_graph.assert_graph_shape(par_nodes)
        # Edges are ravelled
        e_list = np.asarray(np.unravel_index(edge_list[layer], shape=par_nodes.shape[1:]))
        assert (e_list.max(axis=(1,2)) < par_nodes.shape[1:]).all(), 'Edge indices exceed graph shape'

    # Print some info
    edge_layer = topo_graph.find_main_edge_layer(edge_list)
    print(f'Total: graphs {len(graph_list)}, edges {edge_list.shape[-1]} ({np.count_nonzero(edge_layer < 0)} unused)')
    for layer, par_nodes in enumerate(graph_list):
        edge_layer_mask = edge_layer == layer
        # Edges are ravelled
        e_list = np.asarray(np.unravel_index(edge_list[layer], shape=par_nodes.shape[1:]))
        print(f'Layer {layer}: graph shape: {par_nodes.shape[1:]}, edge_list range: {e_list.min(axis=(1,2))} to {e_list.max(axis=(1,2))}')
        print(f'   used edges: {np.count_nonzero(edge_layer_mask)}')

    # Start at non-first layer
    base_layer = 2      # HACK: limit complexity
    if node_pos is not None:
        node_pos = topo_graph.get_node_center(node_pos, (topo_graph.isolate_graphtrees(pn) for pn in graph_list[:base_layer]))
        # Try to fit in window height
        range = (node_pos.max(0) - node_pos.min(0))[:2].min()
        node_pos *= int(DASH_STYLE['height'].strip('px')) / range
    edge_list = edge_list[base_layer:]
    graph_list = graph_list[base_layer:]

    elements = []
    for layer, par_nodes in enumerate(graph_list):
        edge_layer_mask = edge_layer == base_layer + layer
        # Edges are ravelled
        e_list = np.asarray(np.unravel_index(edge_list[layer], shape=par_nodes.shape[1:]))
        elements += build_cytoscape_elements(par_nodes, e_list, edge_src_mask, edge_layer - (base_layer + layer),
                layer=layer, base_layer=base_layer, top_layer=layer + 1 >= len(graph_list), node_pos=node_pos)
        node_pos = None

    subtitle = ', '.join(f'{base_layer + layer}: {list(par_nodes.shape[1:])}'
            for layer, par_nodes in enumerate(graph_list))
    create_dash(f'Graph of {len(graph_list)} layers, starting at {base_layer}', subtitle, elements)

def build_cytoscape_elements(par_nodes: topo_graph.T_Graph, edge_list: topo_graph.T_IndexArray,
        edge_src_mask: topo_graph.T_MaskArray, edge_layer: topo_graph.T_IndexArray, *,
        layer:int, base_layer: int, top_layer: bool, node_pos: npt.NDArray|None=None) -> list[dict[str, dict]]:
    """Build Cytoscape.elements parameter for specific graph"""
    topo_graph.assert_graph_shape(par_nodes)

    tree_idx, seed_mask = topo_graph.isolate_graphtrees(par_nodes)
    num_child = topo_graph.num_node_children(par_nodes)

    id_prefix = f'{base_layer + layer}'
    elements = []
    # Node elements and self-pointing edges
    for idx in np.ndindex(par_nodes.shape[1:]):
        idx = np.asarray(idx)
        node_id = f'{id_prefix}{idx}'
        seed = seed_mask[*idx]
        leaf = num_child[*idx] == 0
        node_data = {'id': node_id, 'label': f'{idx}', 'layer': layer}
        if not top_layer:
            node_data['parent'] = f'{base_layer + layer + 1}[{tree_idx[*idx]}]'
        classes = 'seed' if seed else 'leaf' if leaf else 'normal'
        if layer:
            classes += ' tree'
        elements.append({'data': node_data, 'classes': classes})
        if node_pos is not None:
            elements[-1]['position'] = {'x': node_pos[*idx][0], 'y': node_pos[*idx][1]}
        # Self-pointing edges
        if (par_nodes[:, *idx] == idx).all():
            elements.append({'data': {'source': node_id, 'target': node_id}})

    # Edge elements (ghost edges emerge in the top layer)
    ghost_mask = ~np.any(edge_src_mask, axis=0) if top_layer else np.zeros_like(edge_src_mask[0])
    edge_idx = np.nonzero((edge_layer >= 0) | ghost_mask)[0]
    for idx in edge_idx:
        src_id = f'{id_prefix}{edge_list[:, 0, idx]}'
        tgt_id = f'{id_prefix}{edge_list[:, 1, idx]}'
        edge_data = {'source': src_id, 'target': tgt_id,}
        if top_layer or edge_layer[idx] == 0:
            edge_data['label'] = str(idx)
        classes = '' if edge_layer[idx] == 0 else 'ghost' if ghost_mask[idx] else 'shadow'
        # Forward or ghost edge
        if edge_src_mask[0, idx] or ghost_mask[idx]:
            elements.append({'data': edge_data, 'classes': classes})
        # Backward edge
        if edge_src_mask[1, idx]:
            edge_data = edge_data.copy()
            edge_data['source'], edge_data['target'] = edge_data['target'], edge_data['source']
            if 'label' in edge_data:
                edge_data['label'] = '~' + edge_data['label']
            elements.append({'data': edge_data, 'classes': classes})

    return elements

def create_dash(title:str, subtitle:str, elements: list[dict[str, dict]]) -> None:
    """Create dash-citoscape graph"""
    app = Dash(__name__)
    app.layout = html.Div([
        html.H3(title),
        cyto.Cytoscape(
            id='topo-graph',
            elements=elements,
            layout=DASH_LAYOUT,
            style=DASH_STYLE,
            stylesheet=DASH_STYLE_SHEET,
        ),
        html.P(subtitle),
    ])
    app.run(debug=True)

if __name__ == '__main__':
    import sys
    input = sys.argv[1]
    print(f'Loading {input}')
    graph_data = topo_graph.load_topo_graph(input)
    visualize_graph_hierarchy(**graph_data)
