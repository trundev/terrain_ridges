"""Visualization of internal graph structures (for development purposes) using plotly

See: https://plotly.com/python/

[Hint] visualization one-liner:
- 2D: `go.Figure(data=go.Surface(z=altitude_grid)).show()`
- 3D: `go.Figure(data=go.Surface(z=altitude_grid[:, :, <idx>])).show()`,
    where `:,:,<idx>` is a 2D slice of the data to be visualized
"""
import numpy as np
from terrain_ridges.topo_graph import T_Graph, T_MaskArray, T_IndexArray, T_NodeValues
import plotly.graph_objects as go
from plotly import basedatatypes

# Slice axis to project 3D grids in 2D plane
SLICE_AXIS = 0      #TODO: Change this
SLICE_LABEL = 'xyz'[SLICE_AXIS]
ALTITUDE_COLORSCALE = 'viridis'
GRAPH_MARKERS = True

#
# Node value (altitude) visualization
#
def node_vals_to_trace(fig: go.Figure, node_vals: T_NodeValues, *, shape_dims: int|None=None) -> go.Figure:
    """Create plotly traces from node values/coordinates/altitude

    Parameters
    ----------
    fig : plotly.Figure
    node_vals : (node-indices, value-indices) ndaray of float
        Map from node index to a generic value associated with the node (usually array).
        First `shape_dims` dimensions are the node-index, the rest - indices for individual value
    shape_dims : int or None
        Number of node-grid dimensions, the rest are value dimensions.
        - If negative, indicates the number of value dimensions.
        - If `None`, the value is a scalar - all `node_vals` dimensions are the grid

    Returns
    -------
    fig : plotly.Figure
    """
    node_shape = node_vals.shape[:shape_dims]

    # Handle various dimensions of node-value sub-arrays
    trace_kwargs: dict[str, str|T_NodeValues] = dict()
    geo_mode = len(node_shape) != node_vals.ndim
    if geo_mode:
        # Value is an array - longitude/latitude, altitude is optional
        assert len(node_shape) == node_vals.ndim - 1, 'TODO: 1D value only - lon/lat/alt'
        if len(node_shape) == node_vals.shape[-1]:
            # longitude/latitude only
            alts = None
        else:
            # longitude/latitude and altitude
            assert len(node_shape) == node_vals.shape[-1] + 1, \
                'Nodes must have extra altitude coordinate'
            alts = node_vals[..., -1]
            node_vals = node_vals[..., :-1]
        trace_kwargs = dict(zip(('lon', 'lat'), node_vals))
    else:
        # Value is a scalar - altitude, coordinates are integer grid
        alts = node_vals
        trace_kwargs = dict(zip('xyz', np.indices(node_shape)))
        # Extra dimension from altitude (if available)
        if 'z' not in trace_kwargs:
            trace_kwargs['xyz'[len(node_shape)]] = alts
    trace_kwargs['name'] = 'Node vals'

    # Select type of plotly figure
    if geo_mode:
        #TODO: Fix this
        fig.add_scattermapbox(**trace_kwargs)
    else:
        if len(node_shape) == 1:
            # 1D node grid - 2D scatter
            fig.add_scatter(**trace_kwargs)
        elif len(node_shape) == 2:
            # 2D node grid - 3D surface
            fig.add_surface(**trace_kwargs, colorscale=ALTITUDE_COLORSCALE)
        elif len(node_shape) == 3:
            # 3D node grid - 3D figure, slices on `SLICE_AXIS`
            assert alts is not None
            trace_kwargs['colorscale'] = ALTITUDE_COLORSCALE
            del trace_kwargs['x'], trace_kwargs['y']
            index_exp: list[slice|int] = [np.s_[:]] * alts.ndim
            steps = []
            for idx in range(alts.shape[SLICE_AXIS]):
                index_exp[SLICE_AXIS] = idx
                trace_kwargs['z'] = alts[*index_exp]
                fig.add_surface(**trace_kwargs, visible=idx==0)
                visible = np.arange(alts.shape[SLICE_AXIS]) == idx
                steps.append(dict(method='restyle', args=[{'visible': visible}], label=f'{SLICE_LABEL}={idx}'))
            fig.update_layout(sliders=[dict(steps=steps)])
        else:
            assert False, 'TODO: Unsupported'
    return fig

def visualize_altutude(altitude_grid: T_NodeValues) -> go.Figure:
    """Visualize altitude grid"""
    fig = go.Figure()
    fig.update_layout(title_text=f'Elevation grid, shape {altitude_grid.shape}', showlegend=True)
    node_vals_to_trace(fig, altitude_grid, shape_dims=None).show()
    return fig

#
# Graph/edge visualization
#
def adjust_node_points(points: T_NodeValues, mask: T_MaskArray, scale: float=.05) -> T_NodeValues:
    """Pull back edge's target nodes by 5% (to NOT overlap actual target-node)"""
    assert mask.shape[0] == 2 and points.shape[1:] == mask.shape, 'Point vs. target-mask shape mismatch'
    points = points.astype(float)
    points[:, mask] = points[:, mask] * (1-scale) + points[:, ::-1][:, mask] * scale
    return points

def graph_to_trace(fig: go.Figure, graph_edges: T_Graph, altitude_grid: T_NodeValues|None, *,
                   edge_mask: T_MaskArray|None=None, node_color: T_IndexArray|None=None,
                   name: str|None=None) -> go.Figure:
    """Create plotly traces from graph (list of edges)

    Parameters
    ----------
    fig : plotly.Figure
    graph_edges : (node-coord, 2, edge-indices) ndarray of int
        Source graph
    altitude_grid : (node-index) [optional]
        Graph-node altitudes
    edge_mask : (edge-indices) ndarray of bool [optional]
        Mask to apply over source graph, for correct edge index text-labels
    node_color : (node-index) [optional]
        Graph-node colors

    Returns
    -------
    fig : plotly.Figure
    """
    if edge_mask is None:
        edge_mask = np.broadcast_to(True, graph_edges.shape[2:])
    points = graph_edges[..., edge_mask]

    # Handle scenarios with and w/o altitude
    if altitude_grid is None:
        alts = None
        # 1D case: same x and y values
        if points.shape[0] == 1:
            points = np.concatenate((points, points), axis=0)
    else:
        alts = altitude_grid[*points]
        # Add altitude as next axis (if dimensions allow)
        if points.shape[0] < 3:
            points = np.concatenate((points, alts[np.newaxis]), axis=0)

    # Pull back edge-target a bit (to NOT overlap actual target-node)
    edge_src_mask = np.broadcast_to(([True], [False]), points.shape[1:])
    points = adjust_node_points(points, ~edge_src_mask)
    # Separate individual edges with NaN-s (pad the second dimension)
    pad_width = [(0,0)] * points.ndim
    pad_width[1] = (0,1)
    points = np.pad(points, pad_width, constant_values=np.nan)

    # Combine  plotly.graph_objects.Scatter arguments
    # Convert coordinates to x=<arr>, y=<arr>, ... format
    kwargs = {k: v.T.flat for k, v in zip('xyz', points)}
    # Generate text and symbols for each marker (TODO: expected single dimension edge-index)
    text = np.nonzero(np.broadcast_to(edge_mask, (3,) + edge_mask.shape).T)
    kwargs['text'] = 'edge:' + text[0].astype(str) + '-' + text[1].astype(str)
    if GRAPH_MARKERS:
        kwargs['mode'] = 'lines+markers'
        # Symbols denote source/target node
        symbols = np.where(edge_src_mask, 'circle', 'circle-open')
        symbols = np.pad(symbols, pad_width[1], constant_values='x')
        symbols = symbols.T.flat
        kwargs['marker_symbol'] = symbols
        if node_color is not None:
            # Color markers by node-altitude
            marker_color = node_color[*graph_edges[..., edge_mask]]
            kwargs['marker_color'] = np.pad(marker_color, pad_width[1:]).T.flat
        elif alts is not None:
            # Color markers by node-altitude
            kwargs['marker_color'] = np.pad(alts, pad_width[1:]).T.flat
            kwargs['marker_colorscale'] = ALTITUDE_COLORSCALE
    elif GRAPH_MARKERS is False:
        kwargs['mode'] = 'lines'

    kwargs['name'] = 'Graph edges' if name is None else name
    if 'z' in kwargs:
        fig.add_scatter3d(**kwargs)
    else:
        fig.add_scatter(**kwargs)
    return fig

def visualize_graph(graph_edges: T_Graph, altitude_grid: T_NodeValues|None=None, **kwargs
                    ) -> go.Figure:
    """Visualize graph"""
    if altitude_grid is None:
        grid_shape = None
    else:
        grid_shape = altitude_grid.shape
    fig = go.Figure()
    fig.update_layout(title_text=f'Elevation grid, shape {grid_shape}', showlegend=True)
    graph_to_trace(fig, graph_edges, altitude_grid, **kwargs).show()
    return fig
