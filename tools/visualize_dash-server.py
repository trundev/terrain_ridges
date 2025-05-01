"""Dash server for visualization of internal graph structures, uses dash-cytoscape

See: https://dash.plotly.com/ and https://js.cytoscape.org/
"""
import sys, os
import uuid
import json
import dash
import flask
import dash_cytoscape as cyto

# Title/elements to show at startup
DEFAULT_CYTO_TITLE = dash.html.I('Use visualize_dash.visualize_graph() to upload the graph')
DEFAULT_CYTO_ELEMENTS =[
        dict(data=dict(id='x', label='wait...'),
             classes='invalid',
            )]
DEFAULT_CYTO_TAPINFO = dash.html.I('Click on graph element')

CYTOSCAPE_LAYOUTS = ['preset', 'random', 'grid', 'circle', 'concentric', 'breadthfirst', 'cose']
CYTOSCAPE_LAYOUT = dict(name='preset')
DASH_STYLE = dict(width='100%', height='600px')
# Style sheet (https://js.cytoscape.org/#style)
DASH_STYLE_SHEET = [
    dict(selector='node', style={
            'label': 'data(label)',
            'border-width': '3px',
            }),
    dict(selector='edge', style={
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            }),
    # Classes
    dict(selector='edge.passive', style={
            'line-style': 'dashed',
            #'line-dash-pattern': [3,10],
            'opacity': '.7',
        }),
    dict(selector='edge.ghost', style={
            'line-style': 'dotted',
            'opacity': '.5',
        }),
    dict(selector='node.leaf', style={
            'background-color': 'green',
            'border-color': 'green',
            'shape': 'triangle',
        }),
    dict(selector='node.root', style={
            'background-color': 'cyan',
            'border-color': 'cyan',
            'shape': 'star',
        }),
    dict(selector='node.loop', style={
            'background-color': 'blue',
            'border-color': 'blue',
            'shape': 'square',
        }),
    dict(selector='node.invalid', style={
            'background-color': 'red',
            'shape': 'octagon',
        }),
    dict(selector='node.parent', style={
            'shape': 'square',  # Empty parent-node only
        }),
    ]

#
# Dash server related
#
def create_cytoscape(app: dash.Dash) -> tuple[
        dash.development.base_component.Component, cyto.Cytoscape,
        dash.development.base_component.Component]:
    """Create the Cytoscape object"""
    #
    # Cytoscape object and a layout dropdown
    #
    title_comp = dash.html.Label(DEFAULT_CYTO_TITLE)
    cyto_comp = cyto.Cytoscape(elements=DEFAULT_CYTO_ELEMENTS,
            id=str(uuid.uuid4()),
            layout=CYTOSCAPE_LAYOUT,
            style=DASH_STYLE,
            stylesheet=DASH_STYLE_SHEET)
    dropdown = dash.dcc.Dropdown(
            id=cyto_comp.id +'-dropdown',
            value=cyto_comp.layout.get('name'),
            clearable=False,
            options=[{'label': name, 'value': name} for name in CYTOSCAPE_LAYOUTS])
    cyto_tapinfo = dash.html.Pre(DEFAULT_CYTO_TAPINFO,
            id=cyto_comp.id +'-tapinfo')
    @app.callback(dash.Output(cyto_comp.id, 'layout'),
                  dash.Input(dropdown.id, 'value'))
    def update_layout(layout):
        return {'name': layout}
    @app.callback(dash.Output(cyto_tapinfo.id, 'children'),
                  dash.Input(cyto_comp.id, 'selectedNodeData'),
                  dash.Input(cyto_comp.id, 'selectedEdgeData'),
                  prevent_initial_call=True)
    def displayNodeEdgeData(nodeData, edgeData):
        data = {f'nodes ({len(nodeData)})': nodeData} if nodeData else {}
        data |= {f'edges ({len(edgeData)})': edgeData} if edgeData else {}
        return json.dumps(data, indent=2)

    #
    # Combine into Fieldset
    #
    return title_comp, cyto_comp, dash.html.Fieldset(
            [title_comp, cyto_comp, dropdown, cyto_tapinfo])

def run_dash() -> None:
    """Start the Dash server"""
    server = flask.Flask(__name__)
    app = dash.Dash(__name__, server=server)
    #
    # Cytoscape object
    #
    title_comp, cyto_comp, cyto_container = create_cytoscape(app)

    #
    # Handle POST/GET requests
    #
    @server.route('/cytoscape', methods=['POST', 'GET'])
    def handle_request() -> str:
        """Flask server """
        request = flask.request
        if request.method == 'POST':
            data = request.get_json()  # For JSON data
        elif request.method == 'GET':
            data = request.args.get('elements')
            if data is None:
                return f'GET request with no elements'
            data = json.loads(data)
        else:
            return f'Request method {request.method} not supported'

        title = request.args.get('title')
        print(f'Received POST title: "{title}", data (len {len(data)}):')
        if len(data) < 10:
            print(data)
        else:
            print(data[:10], '\n...')

        title_comp.children = title
        cyto_comp.elements = data
        return f'Received {request.method}, title {title}'

    #
    # Main layout
    #
    app.layout = [
            dash.html.H2(os.path.basename(__file__)),
            dash.html.Button('Refresh', id='refresh_layout'),
            cyto_container,
        ]

    #
    # Handle "Refresh" button
    #
    @app.callback(dash.Output(cyto_comp.id, 'elements'),
                  dash.Input('refresh_layout', 'n_clicks'),
                  dash.State(cyto_comp.id, 'elements'),
                  prevent_initial_call=True)
    def refresh(n_clicks, elements):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate
        return elements

    return app.run(debug=True, use_reloader=False)

#
# Main entry
#
def main(argv) -> None:
    """Standalone execution (TODO: import data from file)"""
    run_dash()

if __name__ == '__main__':
    main(sys.argv[1:])
