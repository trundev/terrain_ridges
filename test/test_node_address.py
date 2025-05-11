"""Pytest for node_address.py"""
import os
import numpy as np
import pytest
from . import conftest  # Imported by pytest anyway
from terrain_ridges.topo_graph import T_Graph, T_IndexArray
from terrain_ridges import node_address

@pytest.fixture(scope='module')
def dem_node_addr(build_graph_edges: T_Graph) -> T_IndexArray:
    """Test node addresses for DEM file fixture"""
    return node_address.generate_node_addresses(build_graph_edges)

def test_generate(dem_node_addr: T_IndexArray):
    """Test generate_node_addresses results"""
    np.testing.assert_equal(dem_node_addr.min((1,2)), 0,
                            'Node address components must start from zero')
    # Check if all components are normalized
    for comp in range(dem_node_addr.shape[0]):
        par_addrs = dem_node_addr[comp + 1:].reshape(-1, dem_node_addr[0].size)
        for par_addr in np.unique(par_addrs, axis=1).T:
            # Extract components for single parent address, check if normalized
            mask = (par_addr == dem_node_addr[comp + 1:].T).T.all(0)
            uniq_ids = np.unique(dem_node_addr[comp, mask])
            assert uniq_ids.size == uniq_ids.max() + 1, \
                    f'Non-normalized address components in {par_addr}'

def test_ref_data(dem_node_addr: T_IndexArray, dem_file_path: str):
    """Test generate_node_addresses vs. reference results"""
    ref_path = dem_file_path + '-node_addr.npy'
    try:
        node_addr_ref = np.load(ref_path)
    except FileNotFoundError:
        if False:   #TODO: Change this to generate reference data
            np.save(ref_path, node_addr)
            np.save(dem_file_path + '-graph_edges.npy', build_graph_edges)
        pytest.xfail(f'Missing reference result: {ref_path}')

    np.testing.assert_equal(dem_node_addr, node_addr_ref, f'Reference result mismatch: {ref_path}')

def test_ravel_node_address(dem_node_addr: T_IndexArray):
    """Test ravel_node_address"""
    ravel_addr, addr_shape = node_address.ravel_node_address(dem_node_addr)
    np.testing.assert_equal(ravel_addr.shape, dem_node_addr[0].shape, 'Unexpected ravelled array shape')
    np.testing.assert_equal(ravel_addr.min(), 0, 'Node address components must start from zero')
    np.testing.assert_equal(addr_shape, dem_node_addr.max((1,2)) + 1, 'Unexpected ravel-address shape')
    np.testing.assert_equal(ravel_addr.max() < np.prod(addr_shape), True,
                            'Node address components must not exceed dimensions')
