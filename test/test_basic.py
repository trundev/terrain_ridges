""""Basic terrain ridges generator test"""
import os
import argparse
import pytest


SRC_DEM_DIR = 'DEM'
REF_RESULT_DIR = 'ref_results'
RESULT_NAME = os.environ.get('DST_NAME', 'res.kml')


def compare_kmls(test_path, ref_path):
    """Compare test KML against reference one"""
    with open(test_path, 'rt') as fd:
        test_result = fd.read()
    with open(ref_path, 'rt') as fd:
        ref_result = fd.read()
    return test_result == ref_result

def test_main():
    """Run ridges.main() for all the DEMs"""
    import ridges

    # Tweek internal behaviour
    ridges.ASSERT_LEVEL = 3
    args = argparse.Namespace(
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

    src_cnt = ref_cnt = 0
    for entry in os.scandir(SRC_DEM_DIR):
        if entry.is_file():
            # Skip unwanted files
            if entry.name.startswith(('.', '!')) or entry.name.endswith('.npy'):
                continue

            src_cnt += 1
            print(f'\nPyTest {src_cnt}: {entry.path} to {args.dst_ogr_file}')

            # Generate ridges
            args.src_dem_file = entry.path
            ret = ridges.main(args)
            assert ret == 0, f'ridges.main() failed, code {ret}'

            # Compare to reference result (if any)
            ref_path = os.path.join(REF_RESULT_DIR, entry.name + '.kml')
            if os.path.exists(ref_path):
                ref_cnt += 1
                print(f'PyTest compare result {ref_cnt}: {args.dst_ogr_file} to {ref_path}')
                assert compare_kmls(args.dst_ogr_file, ref_path), 'Compare to reference result failed'
            else:
                print(f'PyTest missing reference result: {ref_path}')

    assert src_cnt, f'No source files to process in "{SRC_DEM_DIR}"'
    #assert ref_cnt, f'No matching reference results in "{REF_RESULT_DIR}"'

    print(f'PyTest total: {src_cnt} source DEMs, {ref_cnt} reference results')

#
# For non-pytest debugging
#
if __name__ == '__main__':
    res = test_main()
    if res:
        exit(res)
