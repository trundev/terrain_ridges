"""Test data fixtures to be used by other pytest modules"""
import numpy as np
import numpy.typing as npt
import pytest


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
