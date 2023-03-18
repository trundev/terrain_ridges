"""Terrain ridges visualization

Requires: numpy plotly gdal pyproj
"""
import os
import argparse
import numpy as np
import plotly.graph_objects as go
import gdal_utils

DEF_MGRID_N_POSTFIX = '-1-mgrid_n_xy.npy'
#DEF_MAPBOX_STYLE = 'open-street-map'
DEF_MAPBOX_STYLE = 'mapbox://styles/mapbox/outdoors-v12'
#DEF_MAPBOX_STYLE = 'mapbox://styles/trundev/ckpn5fzfm05zk17rfj1oz6j2t'
MAPBOX_ACCESS_TOKEN = 'pk.eyJ1IjoidHJ1bmRldiIsImEiOiJja211ejdmdjMwMDVmMnZucWR0bXAydW5oIn0._cWi8O8hVesaH0m8ZEO1Cw'


#
# From ridges.py
# (here 'mgrid_n' is with coordinates at first dimension)
#
def accumulate_by_mgrid(src_arr, mgrid_n_xy, mask=Ellipsis):
    """Accumulate array values into their next points in graph, esp. for graph-nodes"""
    res_arr = np.zeros_like(src_arr)
    src_arr = src_arr[mask]
    # To avoid '+=' overlapping, the accumulation is performed by using unbuffered in place
    # operation, see "numpy.ufunc.at".
    indices = mgrid_n_xy[mask if mask is Ellipsis else (...,mask)]
    np.add.at(res_arr, tuple(indices), src_arr)
    return res_arr

def get_n_num_seeds(mgrid_n_xy, *, leaf_seed_val: int or None=-1):
    """Count number of neighbors of each node-point"""
    # Helper self-pointing array
    mgrid_xy = np.indices(mgrid_n_xy.shape[1:])
    # Mask of self-pointing "seed" pixels
    seed_mask = (mgrid_n_xy == mgrid_xy).all(0)
    # Start with ones at each non-seed pixel
    n_num = np.asarray(seed_mask == 0, dtype=int)
    n_num = accumulate_by_mgrid(n_num, mgrid_n_xy)

    # Mark the leaf-seeds (invalid nodes, except "real" seeds)
    if leaf_seed_val is not None:
        assert leaf_seed_val < 0, f'Invalid leaf_seed_val of {leaf_seed_val}'
        n_num[(n_num == 0) & seed_mask] = leaf_seed_val
    return n_num, seed_mask

#
# Figure scatter helpers
#
def add_scatter_points(fig: go.Figure, lonlat_arr: np.array, mask: np.array, text_arr: np.array=None, **scatter_kwargs) -> go.Scattermapbox:
    """Add scatter with markers at each masked point"""
    # Marker text to include x,y coordinates and extra string
    text = np.indices(mask.shape)[:,mask]
    if text.size:
        format = '%d,%d'
        if text_arr is not None:
            text = np.concatenate((text, text_arr[np.newaxis,...]))
            format += ': %s'
        text = np.apply_along_axis(
                lambda xys: np.asarray(format%tuple(xys), dtype=object),
                0, text)
    fig.add_scattermapbox(lon=lonlat_arr[mask][...,0], lat=lonlat_arr[mask][...,1],
                          text=text, **scatter_kwargs)
    return fig.data[-1]

def add_scatter_lines(fig: go.Figure, lonlat_arr_list: list[np.array], **scatter_kwargs) -> go.Scattermapbox:
    """Add scatter with array of arrows/lines between fixed number of points"""
    # Make individual lines by adding gaps: (start-point, end-point, nan)
    lines_arr = np.stack(np.broadcast_arrays(*lonlat_arr_list, np.nan))
    lines_arr = lines_arr.T.reshape(lines_arr.shape[-1], -1)
    #lines_arr = lines_arr.T.reshape(lonlat_start.shape[-1], -1)
    fig.add_scattermapbox(lon=lines_arr[0], lat=lines_arr[1], **scatter_kwargs)
    return fig.data[-1]

def add_scatter_mgrid_n(fig: go.Figure, lonlat_arr: np.array, mgrid_n_xy: np.array, len_scale: float=None,
    text_arr: np.array=None, **scatter_kwargs) -> go.Scattermapbox:
    """Add scatter with arrows (graph-edges) toward neighbors"""
    lines_arr = lonlat_arr[tuple(mgrid_n_xy)]
    # Filter-out zero lines
    mask = (mgrid_n_xy != np.indices(mgrid_n_xy.shape[1:])).any(0)
    lonlat_arr = lonlat_arr[mask]
    lines_arr = lines_arr[mask]
    if text_arr is not None:
        # Each line produces 3 points, place text at middle one
        text_arr = np.stack(np.broadcast_arrays(None, text_arr, None))[:,mask].T.flat
    # Rescale line lengths
    if len_scale is not None:
        lines_arr = lines_arr * len_scale + lonlat_arr * (1 - len_scale)
    return add_scatter_lines(fig, (lonlat_arr, lines_arr), text=text_arr, **scatter_kwargs)

def get_gradient_mgrid(altitude: np.array) -> np.array:
    """Generate gradient mgrid_n"""
    mgrid = np.indices(altitude.shape)
    grad_n = mgrid.copy()
    altitude_n = altitude.copy()
    for sl_left, sl_right in [
            # along x
            [(slice(0, -1), ...), (slice(1, None), ...)],
            [(slice(1, None), ...), (slice(0, -1), ...)],
            # along y
            [(..., slice(0, -1)), (..., slice(1, None))],
            [(..., slice(1, None)), (..., slice(0, -1))],
            # along x\y
            [(slice(0, -1), slice(0, -1)), (slice(1, None), slice(1, None))],
            [(slice(1, None), slice(1, None)), (slice(0, -1), slice(0, -1))],
            # along x/y
            [(slice(0, -1), slice(1, None)), (slice(1, None), slice(0, -1))],
            [(slice(1, None), slice(0, -1)), (slice(0, -1), slice(1, None))],
        ]:
        mask = altitude_n[sl_left] < altitude[sl_right]
        altitude_n[sl_left][mask] = altitude[sl_right][mask]
        grad_n[:, *sl_left][:, mask] = mgrid[:, *sl_right][:, mask]
    return grad_n

def plot_figure(fig: go.Figure, dem_band: gdal_utils.gdal_dem_band, mgrid_n_list: list) -> None:
    """Create figure plot"""
    indices = np.moveaxis(np.indices(dem_band.shape), 0, -1)
    lla_arr = dem_band.xy2lonlatalt(indices)
    valid_mask = np.isfinite(lla_arr[...,2])

    # Markers at each valid grid-point
    altitude = lla_arr[valid_mask][...,2]
    data = add_scatter_points(fig, lla_arr, valid_mask,
                              # Show altitude
                              text_arr=altitude, marker_color=altitude,
                              mode='markers',
                              name='DEM')
    res = data,

    # Visualize node-graphs from 'mgrid_n_list'
    for idx, (name, mgrid_n, *info) in enumerate(mgrid_n_list):
        print(f'{idx}: {name}')
        # Arrows (graph-edges) toward neighbors (cut lines half-way)
        data = add_scatter_mgrid_n(fig, lla_arr, mgrid_n, len_scale=.5,
                                   mode='lines', text_arr=info[0] if info else None,
                                   name=f'mgrid_n', legendgroup=idx, legendgrouptitle_text=name)
        res = *res, data

        # Number of neighbors of each node-point
        n_num, seed_mask = get_n_num_seeds(mgrid_n)
        # Leafs
        print(f'  Leafs: {np.count_nonzero(n_num == 0)}')
        data = add_scatter_points(fig, lla_arr, n_num == 0,
                                  mode='markers', marker=dict(symbol='circle'),
                                  name=f'Leafs', legendgroup=idx)
        res = *res, data
        # Real-seeds (self-pointing, but not leafs)
        print(f'  Seeds: {np.count_nonzero((n_num > 0) & seed_mask)}, self-pointing: {np.count_nonzero(seed_mask)}')
        seed_mask &= n_num > 0
        data = add_scatter_points(fig, lla_arr, seed_mask,
                                  mode='markers', marker=dict(symbol='circle'),
                                  name=f'Seeds', legendgroup=idx)
        res = *res, data
        # Nodes
        node_mask = n_num > 1
        n_num_masked = n_num[node_mask]
        print(f'  Nodes: {n_num_masked.size}, max: {n_num_masked.max()}')
        data = add_scatter_points(fig, lla_arr, node_mask,
                                  text_arr=n_num_masked,
                                  mode='markers', marker=dict(symbol='circle', size=4*n_num_masked),
                                  name=f'Nodes', legendgroup=idx)
        res = *res, data

        # Straight-lines between non-leaf nodes
        start_xy = np.nonzero(node_mask)
        next_xy = mgrid_n[:,*start_xy]
        # "Cut" the grid at nodes, where to stop traversing
        mgrid_tmp = mgrid_n.copy()
        mgrid_tmp[:,node_mask] = np.indices(node_mask.shape)[:,node_mask]
        while True:
            prev_xy = next_xy
            next_xy = mgrid_tmp[:,*tuple(next_xy)]
            # Check if there is any change
            if (prev_xy == next_xy).all():
                break
        data = add_scatter_lines(fig, (lla_arr[start_xy], lla_arr[tuple(next_xy)]),
                                 mode='lines',
                                 name=f'Node-edges', legendgroup=idx)
        res = *res, data

    return res

def main(args):
    """Main finction"""
    # Load input files
    print(f'Loading DEM: "{args.dem_file}"')
    dem_band = gdal_utils.dem_open(args.dem_file)
    dem_band.load()

    mgrid_n_list = []
    if args.gradient:
        altitude = dem_band.get_elevation(True)
        mgrid_n_list.append(('Gradient', get_gradient_mgrid(altitude)))
    if args.mgrid_n is not None:
        for fname in args.mgrid_n:
            print(f'Loading neighbor-grid: "{args.mgrid_n}"')
            mgrid_n = np.load(fname)
            assert dem_band.shape == mgrid_n.shape[:-1], \
                    f'DEM vs neighbor-grid shape mismatch: {dem_band.shape}, {mgrid_n.shape[:-1]}'
            # Move the coordinates into the first dimension
            # (as in numpy convention)
            mgrid_n = np.moveaxis(mgrid_n, -1, 0)
            mgrid_n_list.append((os.path.basename(fname), mgrid_n))

    # Obtain boundaries
    bounds = np.asarray([(0,0), dem_band.shape])
    bounds[1] -= 1
    bounds = dem_band.xy2lonlatalt(bounds)
    print(f'DEM boundaries: {dem_band.shape}:')
    print(f'  Upper Left : {bounds[0]}')
    print(f'  Lower Right: {bounds[1]}')

    # Greate plotly figure
    fig = go.Figure()

    plot_figure(fig, dem_band, mgrid_n_list)

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
                style=args.mapbox_style,
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
    # Experimental: menus
    fig.update_layout(updatemenus=[
                dict(buttons=list([
                            dict(
                                args=["type", "scattermapbox"],
                                label="Scatter Mapbox",
                                method="restyle"
                            ),
                            dict(
                                args=["type", "scattergeo"],
                                label="Scatter Geo",
                                method="restyle"
                            )
                        ]),
                        x=1,
                    ),
            ])
    fig.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Terrain ridges visualization')
    parser.add_argument('dem_file',
            help='Input DEM file, formats supported by https://gdal.org')
    parser.add_argument('--mgrid-n', action='append', nargs='?',
            help='Neighbor grid file (intermediate terrain_ridges), numpy.save() format.'
            f' If empty, append "{DEF_MGRID_N_POSTFIX}" to "dem_file"')
    parser.add_argument('--mapbox-style', default=DEF_MAPBOX_STYLE,
            help=f'Mapbox layout style, default: "{DEF_MAPBOX_STYLE}"')
    parser.add_argument('--gradient', action='store_true',
            help=f'Generate gradient mgrid-n')
    args = parser.parse_args()

    # Apply '--mgrid-n' default
    if args.mgrid_n and (None in args.mgrid_n):
        args.mgrid_n[args.mgrid_n.index(None)] = args.dem_file + DEF_MGRID_N_POSTFIX
        if None in args.mgrid_n:
            parser.exit(255, 'Error: Empty "--mgrid-n" option was used more than once')

    res = main(args)
    if res:
        exit(res)
