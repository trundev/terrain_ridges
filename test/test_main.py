""""Test terrain ridges generator using real data"""
import os
import argparse
import pytest
from .conftest import REF_RESULT_DIR
from terrain_ridges import ridges


RESULT_NAME = os.environ.get('DST_NAME', 'res.kml')

def compare_kmls(test_path, ref_path):
    """Compare test KML against reference one"""
    with open(test_path, 'rt') as fd:
        test_result = fd.read()
    with open(ref_path, 'rt') as fd:
        ref_result = fd.read()
    return test_result == ref_result

def test_real_dem(dem_file_path: str):
    """Run ridges.main() for a DEM-file"""

    # Tweak internal behavior
    ridges.ASSERT_LEVEL = 3
    args = argparse.Namespace(
            src_dem_file=dem_file_path,
            dst_ogr_file=RESULT_NAME,
            dst_format=None,
            valleys=False,
            boundary_val=None,
            distance_method=list(ridges.DISTANCE_METHODS.keys())[-1],
            multi_layer=None,
            append=False,
            separated_branches=False,
            smoothen_geometry=False,
            resume_from_snapshot=0,
            keep_snapshots=False)
    # Delete possible result left-over
    try:
        os.unlink(RESULT_NAME)
    except FileNotFoundError:
        pass

    # Generate ridges
    ret = ridges.main(args)
    assert ret == 0, f'ridges.main() failed, code {ret}'

    # Compare to reference result (if any)
    ref_path = os.path.join(REF_RESULT_DIR, os.path.basename(dem_file_path) + '.kml')
    if not os.path.exists(ref_path):
        pytest.xfail(f'Missing reference result: {ref_path}')

    print(f'PyTest compare result: {args.dst_ogr_file} to {ref_path}')
    assert compare_kmls(args.dst_ogr_file, ref_path), 'Compare to reference result failed'
