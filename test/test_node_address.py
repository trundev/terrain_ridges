"""Pytest for node_address.py"""
import os
import numpy as np
import pytest
from . import conftest  # Imported by pytest anyway
from terrain_ridges.topo_graph import T_Graph, T_IndexArray
from terrain_ridges import node_address

#
# `generate_node_addresses()` fixture
# also separate the two result arrays: `node_addr` and `edge_levels`
#
@pytest.fixture(scope='module')
def node_addr_edge_levels(build_graph_edges: T_Graph) -> tuple[T_IndexArray, T_IndexArray]:
    """Test node addresses for DEM file fixture"""
    return node_address.generate_node_addresses(build_graph_edges)

@pytest.fixture(scope='module')
def dem_node_addr(node_addr_edge_levels: tuple[T_IndexArray, T_IndexArray]) -> T_IndexArray:
    """Test node addresses for DEM file fixture"""
    return node_addr_edge_levels[0]

@pytest.fixture(scope='module')
def dem_edge_levels(node_addr_edge_levels: tuple[T_IndexArray, T_IndexArray]) -> T_IndexArray:
    """Test node addresses for DEM file fixture"""
    return node_addr_edge_levels[1]

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

def test_edge_levels_ref_data(build_graph_edges: T_Graph, dem_edge_levels: T_IndexArray,
                              dem_file_path: str):
    """Test generate_node_addresses (edge-levels component) vs. reference results"""
    mask = dem_edge_levels > 0
    graph_edges = build_graph_edges[..., mask]
    edge_levels = dem_edge_levels[mask]

    def test_ref_file(data: T_IndexArray, ref_path: str):
        try:
            data_ref = np.load(ref_path)
        except FileNotFoundError:
            if False:   #TODO: Change this to generate reference data
                np.save(ref_path, data)
            pytest.xfail(f'Missing reference result: {ref_path}')

        np.testing.assert_equal(data, data_ref, f'Reference result mismatch: {ref_path}')

    test_ref_file(graph_edges, dem_file_path + '-graph_edges.npy')
    test_ref_file(edge_levels, dem_file_path + '-edge_levels.npy')

def test_edge_levels(build_graph_edges: T_Graph, dem_edge_levels: T_IndexArray, dem_file_path: str):
    """Test generate_node_addresses results"""
    mask = dem_edge_levels > 0
    graph_edges = build_graph_edges[..., mask]
    edge_levels = dem_edge_levels[mask]

    #
    # TODO: Move this in separate file
    #
    import itertools
    import argparse
    from tools import legacy_integration
    from terrain_ridges import ridges
    from .test_legacy import flip_edge_chain

    tree_edge_mask = False
    for level, next_level in itertools.pairwise(np.unique(edge_levels)):
        tree_edge_mask |= edge_levels == level
        next_tree_mask = edge_levels == next_level
        mask = flip_edge_chain(graph_edges, next_tree_mask, edge_mask=tree_edge_mask)
    legacy_integration.keep_as_mgrid_snaphot(graph_edges, dem_file_path, add_sfix=True)

    # This is takem from test_legacy.test_generate_ridges()
    result_name = dem_file_path + '.kml'
    # Taken from test_main.test_real_dem
    args = argparse.Namespace(
            src_dem_file=dem_file_path,
            dst_ogr_file=result_name,
            dst_format=None,
            valleys=False,
            boundary_val=None,
            distance_method=list(ridges.DISTANCE_METHODS.keys())[-1],
            multi_layer=None,
            append=False,
            separated_branches=False,
            smoothen_geometry=False,
            resume_from_snapshot=1,
            keep_snapshots=False)
    # Delete possible result left-over
    try:
        os.unlink(result_name)
    except FileNotFoundError:
        pass

    # Generate ridges
    ret = ridges.main(args)
    assert ret == 0, f'ridges.main() failed, code {ret}'

    print(f'Result KML: {result_name}')
