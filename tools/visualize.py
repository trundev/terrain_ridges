"""Vizualize various topology graphs
"""
import os
import argparse
import numpy as np
import plotly.graph_objects as go
import contourpy


DEF_MGRID_N_POSTFIX = '-1-mgrid_n_xy.npy'
DEF_MAPBOX_STYLE = 'mapbox://styles/mapbox/outdoors-v12'
MAPBOX_ACCESS_TOKEN = 'pk.eyJ1IjoidHJ1bmRldiIsImEiOiJja211ejdmdjMwMDVmMnZucWR0bXAydW5oIn0._cWi8O8hVesaH0m8ZEO1Cw'
REDUCE_DEM_POINTS = 4   # Show each n-th point
POLYGON_HOLES = 2       # 0: drop, 1: separate polygons, others: single self-instersecting


def figure_show(lla_grid: np.array, figarg_gen: callable) -> int:
    """Create and show plotly figure"""
    # Greate plotly figure
    fig = go.Figure()

    # Add scatters as requested by 'figarg_iter'
    for figarg in figarg_gen(fig, lla_grid):
        fig.add_scattermapbox(**figarg)

    bounds = np.array([lla_grid.min((0,1)), lla_grid.max((0,1))])
    # Select zoom and center
    zoom = (bounds[1] - bounds[0])[:2].max()
    zoom = np.log2(360 / zoom)  # zoom 0 is 360 degree wide
    center = bounds.mean(0)
    fig.update_layout({
            'showlegend': True,
            'legend': dict(
                x=0, y=1,   # Top-left overlapping
                groupclick='toggleitem',
            ),
            'mapbox': dict(
                accesstoken=MAPBOX_ACCESS_TOKEN,
                style=DEF_MAPBOX_STYLE, #TODO: args.mapbox_style,
                center={'lon': center[0], 'lat': center[1]},
                zoom=zoom,
            ),
            'geo': dict(
                fitbounds='geojson',
                center={'lon': center[0], 'lat': center[1]},
                resolution=50,
                showrivers=True,
                showlakes=True,
                showland=True,
            ),
        })
    return fig.show()

def figarg_create_demgrid(lla_grid: np.array, reduce_points: int or None=REDUCE_DEM_POINTS) -> dict:
    """Create lon/lat kwargs for altitude-colored DEM points"""
    mask = np.isfinite(lla_grid[...,2])
    if reduce_points is not None:
        # Mask-out points by index-step of 'REDUCE_DEM_POINTS'
        m = np.zeros_like(mask)
        m[(slice(0, None, reduce_points),) * m.ndim] = True
        mask &= m
        del m
    text_gen = (f'{x}x{y}: {a}' for x,y,a in zip(*np.nonzero(mask), lla_grid[mask, 2]))
    return dict(lon=lla_grid[mask, 0], lat=lla_grid[mask, 1],
            mode='markers', marker_color=lla_grid[mask, 2], text=tuple(text_gen))

def figarg_create_lines(lla_grid_list: list[np.array]) -> dict:
    """Create lon/lat kwargs for arrows/lines between equal number of points"""
    # Make individual lines by adding gaps: (start-point, end-point, nan)
    lines_arr = np.stack(np.broadcast_arrays(*lla_grid_list, np.nan))
    lines_arr = lines_arr.T.reshape(lines_arr.shape[-1], -1)
    return dict(lon=lines_arr[0], lat=lines_arr[1])

def figarg_create_graph_lines(lla_grid: np.array, mgrid: np.array, *,
        len_scale: float or None=.8, name: str or None=None) -> dict:
    """Create lon/lat kwargs for arrows in a tree-graph"""
    lines_arr = lla_grid[*mgrid]
    # Filter-out zero lines
    mask = (mgrid != np.indices(mgrid.shape[1:])).any(0)
    bases_arr = lla_grid[mask]
    lines_arr = lines_arr[mask]
    #if text_arr is not None:
    #    # Each line produces 3 points, place text at middle one
    #    text_arr = np.stack(np.broadcast_arrays(None, text_arr, None))[:,mask].T.flat
    # Rescale line lengths
    if len_scale is not None:
        lines_arr = lines_arr * len_scale + bases_arr * (1 - len_scale)

    kwargs = figarg_create_lines((bases_arr, lines_arr))
    if name is not None:
        kwargs['name'] = f'{name} ({np.count_nonzero(mask)})'
    return kwargs

def figarg_create_mask_polygon(lla_grid: np.array, mask: np.array, *,
        quad_as_tri: bool=False, cut_level: float=.4) -> dict:
    """Create lon/lat kwargs for polygon around mask"""
    cont_gen = contourpy.contour_generator(x=lla_grid[..., 0], y=lla_grid[..., 1],
            z=mask, quad_as_tri=quad_as_tri)
    filled = cont_gen.filled(cut_level, mask.max())

    # Combine polygons from 'filled' as single scatter
    comb_poly = np.empty((0, 2))
    for poly, sub_idx in zip(filled[0], filled[1]):
        sub_idx = sub_idx.astype(int)   # Strange insert() failure workaround
        if POLYGON_HOLES == 0:
            # Main poligon only
            poly = poly[:sub_idx[1]]
        elif POLYGON_HOLES == 1:
            # Split individual sub-polygons (holes) with NaNs (plotly trick)
            poly = np.insert(poly, tuple(sub_idx[1:-1]), np.nan, axis=0)
        else:
            # Add intermediare points to fully close the polygon
            poly = np.concatenate((poly, poly[sub_idx[-2::-1]]))
        # Combine polygons with NaNs in between
        if comb_poly.size == 0:
            comb_poly = poly
        else:
            comb_poly = np.concatenate((comb_poly, np.insert(poly, 0, np.nan, axis=0)))
    return dict(lon=comb_poly[...,0], lat=comb_poly[...,1], fill='toself')

def figarg_create_mask_line(lla_grid: np.array, mask: np.array, *,
        quad_as_tri: bool=False, cut_level: float=.5) -> dict:
    """Create lon/lat kwargs for countour line around mask"""
    cont_gen = contourpy.contour_generator(x=lla_grid[..., 0], y=lla_grid[..., 1],
            z=mask, quad_as_tri=quad_as_tri)
    lines = cont_gen.lines(cut_level)

    # Combine polylines from 'lines' as single scatter with NaNs in between (plotly trick)
    comb_poly = lines[0] if len(lines) else np.empty((0, 2))
    for poly in lines[1:]:
        comb_poly = np.concatenate((comb_poly, np.insert(poly, 0, np.nan, axis=0)))
    return dict(lon=comb_poly[...,0], lat=comb_poly[...,1])

#
# Slider helper
#
class Slider():
    """Figure slider helper"""
    fig_data_pos: list[int]
    fig: go.Figure

    def __init__(self, fig: go.Figure, current_pos: int):
        self.fig = fig
        self.fig_data_pos = [len(fig.data)]
        self.current_pos = current_pos

    def add_slider_pos(self) -> None:
        """Create a slider position from just added widgets"""
        self.fig_data_pos.append(len(self.fig.data))
        # Hide all just created widgets, if NOT at the current position
        if len(self.fig_data_pos) - 2 != self.current_pos:
            for scat in self.fig.data[self.fig_data_pos[-2]:]:
                scat.visible = False

    def update_layout(self) -> None:
        """Add the slider in figure layout, must call after all widgets are created"""
        assert len(self.fig_data_pos) > 1, 'Must call after all widgets are created'

        num_steps = len(self.fig_data_pos) - 1
        visible = np.ones((num_steps, len(self.fig.data)), dtype=object)
        # Take default visibility, all widgets in figure (bool, None, or 'legendonly')
        for idx, scat in enumerate(self.fig.data):
            visible[:, idx] = scat.visible
        visible[:, self.fig_data_pos[0]:self.fig_data_pos[-1]] = False
        steps = []
        for idx in range(num_steps):
            visible[idx, self.fig_data_pos[idx]:self.fig_data_pos[idx+1]] = True
            steps.append(dict(method='restyle', args=[{'visible': visible[idx]}], label=f'Layer {idx}'))
        self.fig.update_layout(sliders=[dict(steps=steps)])

#
# Direct invocation
#
def main(args: object) -> int:
    """Main function"""
    # Load input files - need PYTHONPATH to parent
    import gdal_utils
    print(f'Loading DEM-file: "{args.dem_file}"')
    dem_band = gdal_utils.dem_open(args.dem_file)
    dem_band.load()
    #print(f'  shape: {dem_band.shape}')

    # Load neighbor-grids
    mgrid_n_list = {}
    if args.mgrid_n is not None:
        for fname in args.mgrid_n:
            print(f'Loading neighbor-grid: "{args.mgrid_n}"')
            mgrid_n = np.load(fname)
            assert dem_band.shape == mgrid_n.shape[:-1], \
                    f'DEM vs neighbor-grid shape mismatch: {dem_band.shape}, {mgrid_n.shape[:-1]}'
            # Move the coordinates into the first dimension
            # (as in numpy convention)
            mgrid_n = np.moveaxis(mgrid_n, -1, 0)
            mgrid_n_list[os.path.basename(fname)] = mgrid_n
            print(f'  self pointing nodes: {np.count_nonzero((mgrid_n == np.indices(mgrid_n.shape[1:])).all(0))} / {mgrid_n[0,...].size}')

    # Obtain coordinates of each point
    lla_grid = dem_band.xy2lonlatalt(np.moveaxis(np.indices(dem_band.shape), 0, -1))
    del dem_band

    def figarg_gen(fig: go.Figure, lla_grid: np.array) -> dict:
        """Figure scatters kwargs generator"""
        # Points from the DEM-file (slow)
        yield figarg_create_demgrid(lla_grid) | dict(name='DEM')
        # Graphs from neighbor-grids
        for name, mgrid in mgrid_n_list.items():
            yield figarg_create_graph_lines(lla_grid, mgrid) | dict(name=name, mode='lines')
    return figure_show(lla_grid, figarg_gen)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Terrain ridges visualization')
    parser.add_argument('dem_file',
            help='Input DEM file, formats supported by https://gdal.org')
    parser.add_argument('--mgrid-n', action='append', nargs='?',
            help='Legacy neighbor grid file (intermediate ridges.py result), numpy.save() format.'
            f' If empty, append "{DEF_MGRID_N_POSTFIX}" to "dem_file"')
    parser.add_argument('--mapbox-style', default=DEF_MAPBOX_STYLE,
            help=f'Mapbox layout style, default: "{DEF_MAPBOX_STYLE}"')
    args = parser.parse_args()

    # Apply '--mgrid-n' default
    if args.mgrid_n and (None in args.mgrid_n):
        args.mgrid_n[args.mgrid_n.index(None)] = args.dem_file + DEF_MGRID_N_POSTFIX
        if None in args.mgrid_n:
            parser.exit(255, 'Error: Empty "--mgrid-n" option was used more than once')

    res = main(args)
    if res:
        exit(res)
