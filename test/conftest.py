"""Test data fixtures to be used by other pytest modules"""
import os
from typing import Iterator, Any
import numpy as np
import numpy.typing as npt
import pytest
from terrain_ridges import gdal_utils, build_full_graph, topo_graph
from terrain_ridges.topo_graph import T_Graph


DEM_EXTERNSIONS = '.tif', '.hgt', '.dem'
DEM_SIZE_SLOW = 500_000     # Mark files of larger size as 'slow'

TEST_DIR = os.path.dirname(__file__)
SRC_DEM_DIR = os.path.join(TEST_DIR, 'DEM')
REF_RESULT_DIR = os.path.join(TEST_DIR, 'ref_results')

@pytest.fixture
def rng():
    """Random number generator fixture"""
    default_test_seed = 1  # the default seed to start pseudo-random tests
    return np.random.default_rng(default_test_seed)

#
# Data generators
#
@pytest.fixture(params=([31], [59, 53], [5, 47, 37]),
                ids=lambda s: 'shape:'+'x'.join(map(str, s)))
def dist_grid(request) -> npt.NDArray:
    """Distance from center grid fixture"""
    shape = np.asarray(request.param, dtype=int)
    center = shape / 2
    dist = ((np.indices(tuple(shape)).T - center)**2).T
    # First axis is hyperbolic (just for fun)
    dist[0] = -dist[0]
    return np.sqrt(np.abs(dist.sum(0)))

@pytest.fixture
def altitude_grid_base(dist_grid) -> npt.NDArray:
    """Altitude grid (clean) fixture"""
    shape = dist_grid.shape
    # A cosine function with a slope along last axis
    data = np.cos(dist_grid * 4 * np.pi / np.mean(shape))
    data += np.arange(shape[-1]) / shape[-1]
    # Round to provoke duplicate altitudes
    return data.round(1)

@pytest.fixture
def altitude_grid(rng, altitude_grid_base: npt.NDArray) -> npt.NDArray:
    """Altitude grid with NaNs fixture"""
    alt_grid = altitude_grid_base
    idxs = np.indices(alt_grid.shape).reshape(alt_grid.ndim, -1)
    # Add some NaN-s (between 1 and 1%)
    idxs = rng.choice(idxs.T, rng.integers(1, idxs.shape[-1] // 100 + 2)).T
    alt_grid[*idxs] = np.nan
    return alt_grid

#
# Data from DEM files, chained pre-processing
#
def all_dem_files() -> Iterator[str | Any]:
    """All test DEM files, larger ones have 'slow' mark"""
    for entry in os.scandir(SRC_DEM_DIR):
        if entry.is_file() and entry.name.lower().endswith(DEM_EXTERNSIONS):
            if entry.stat().st_size < DEM_SIZE_SLOW:
                yield entry.name
            else:
                yield pytest.param(entry.name, marks=pytest.mark.slow)

# `params` are file-names, instead of paths, just to make pytest-benchmark report cleaner
@pytest.fixture(params=all_dem_files(), scope='module')
def dem_file_path(request) -> str:
    """Test DEM file names fixture"""
    return os.path.join(SRC_DEM_DIR, request.param)

@pytest.fixture(scope='module')
def dem_band(dem_file_path: str) -> gdal_utils.gdal_dem_band:
    """Load test DEM band file fixture"""
    dem_band = gdal_utils.dem_open(dem_file_path)
    assert dem_band is not None, f'Unable to open source DEM "{dem_file_path}"'
    dem_band.load()
    return dem_band

@pytest.fixture(scope='module')
def build_graph_edges(dem_band) -> T_Graph:
    """All edges between neighbor points from test DEM file"""
    def distance(src, tgt):
        return gdal_utils.geod_distance(dem_band).get_distance(src.T, tgt.T, flat=True).T
    graph_edges, _ = build_full_graph.build_graph_edges(dem_band.dem_buf, distance=distance)
    return graph_edges

@pytest.fixture(scope='module')
def tree_graph(build_graph_edges: T_Graph) -> T_Graph:
    """Tree-graph from test DEM file"""
    edge_mask = topo_graph.filter_treegraph(build_graph_edges)
    return build_graph_edges[..., edge_mask]

@pytest.fixture(scope='module')
def sub_graphs(tree_graph: T_Graph) -> T_Graph:
    """Sub-graphs from tree-graph from test DEM file"""
    parent_ids = topo_graph.isolate_subgraphs(tree_graph)
    return parent_ids[*tree_graph]
