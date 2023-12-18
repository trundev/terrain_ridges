"""Pytest for topo_graph.py"""
import numpy as np
from terrain_ridges import topo_graph


def test_reshape_graph():
    """Test reshape/mask graph"""
    # Reshape 1D to 2D and back
    tgt_nodes = np.arange(12) + 1
    tgt_nodes[-1] = 0
    tgt_nodes = tgt_nodes[np.newaxis, :]
    new_tgt_nodes = topo_graph.reshape_graph(tgt_nodes, shape=(3,4))
    assert new_tgt_nodes.shape[1:] == (3,4)
    np.testing.assert_equal(topo_graph.reshape_graph(new_tgt_nodes, shape=(12,)), tgt_nodes)
    # Reshape 1D to 3D and back
    new_tgt_nodes = topo_graph.reshape_graph(tgt_nodes, shape=(2,3,2))
    assert new_tgt_nodes.shape[1:] == (2,3,2)
    np.testing.assert_equal(topo_graph.reshape_graph(new_tgt_nodes, shape=(12,)), tgt_nodes)
    # Reshape 3D to 2D and back to 1D
    new_tgt_nodes = topo_graph.reshape_graph(new_tgt_nodes, shape=(4,3))
    assert new_tgt_nodes.shape[1:] == (4,3)
    np.testing.assert_equal(topo_graph.reshape_graph(new_tgt_nodes, shape=(12,)), tgt_nodes)

    # Flatten graph
    new_tgt_nodes = topo_graph.reshape_graph(tgt_nodes, shape=(4,3))
    new_tgt_nodes = topo_graph.mask_graph(new_tgt_nodes, True)
    np.testing.assert_equal(new_tgt_nodes, tgt_nodes)
    # Mask graph
    new_tgt_nodes = topo_graph.reshape_graph(tgt_nodes, shape=(4,3))
    mask = np.zeros(shape=new_tgt_nodes.shape[1:] ,dtype=bool)
    mask[1:] = True
    new_tgt_nodes = topo_graph.mask_graph(new_tgt_nodes, mask)
    assert new_tgt_nodes.shape == (1, np.count_nonzero(mask))

def test_isolate_graphtrees():
    """Test identification of various tree-types and loop cutting"""
    #
    # Various graph shapes
    #
    tgt_nodes = np.arange(18) + 1
    tgt_nodes[8] = 0        # 0..8: O-shape / circle
    tgt_nodes[12] = 10      # 9..12: 9-shape
    tgt_nodes[[14,16]] = 16 # 13..16: 1-shape
    tgt_nodes[17] = 17      # 17..17: .-shape / leaf-seed
    tgt_nodes = tgt_nodes[np.newaxis, :]
    tree_idx, seed_mask = topo_graph.isolate_graphtrees(tgt_nodes)
    np.testing.assert_equal(tree_idx,  [0,0,0,0,0,0,0,0,0, 1,1,1,1, 2,2,2,2, 3])
    np.testing.assert_equal(seed_mask, [1,1,1,1,1,1,1,1,1, 0,1,1,1, 0,0,0,1, 1])

    # Test loop-cutting toward last element
    cut_tgt_nodes = topo_graph.cut_graph_loops(tgt_nodes, tree_idx, seed_mask, sort_keys=-np.arange(tgt_nodes.size))
    np.testing.assert_equal(np.nonzero(cut_tgt_nodes != tgt_nodes)[1], [8, 12])
    tree_idx, seed_mask = topo_graph.isolate_graphtrees(cut_tgt_nodes)
    assert np.count_nonzero(seed_mask) == tree_idx.max()+1, 'Must have single seed per graph-trees'

    #
    # Single graph-tree with loop
    #
    tgt_nodes = np.arange(16) + 1
    tgt_nodes[14] = 8       # 8..14: loop
    tgt_nodes[15] = 8       # 15: single node branch, base 8
    tgt_nodes[4] = 14       # 0..4: 5 node branch, base 14
    # leftover:             # 5..7: 3 node branch, base 8
    tgt_nodes = tgt_nodes[np.newaxis, :]
    tree_idx, seed_mask = topo_graph.isolate_graphtrees(tgt_nodes)
    np.testing.assert_equal(tree_idx,  0)
    np.testing.assert_equal(seed_mask, [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1, 0])

    # Test loop-cutting at a middle element
    cut_tgt_nodes = topo_graph.cut_graph_loops(tgt_nodes, tree_idx, seed_mask,
            sort_keys=abs(11 - np.arange(tgt_nodes.size)))
    np.testing.assert_equal(np.nonzero(cut_tgt_nodes != tgt_nodes)[1], 11)
    tree_idx, seed_mask = topo_graph.isolate_graphtrees(cut_tgt_nodes)
    assert np.count_nonzero(seed_mask) == tree_idx.max()+1, 'Must have single seed per graph-tree'

def test_merge_graphtrees():
    """Merge graph-trees tests"""
    # Build graph of two trees
    tgt_nodes = np.arange(10) + 1
    tgt_nodes[1] = 4        # 0..1, 2..3: branches, base 4
    tgt_nodes[7] = 6        # 0..7: tree, seed is loop 6..7
    tgt_nodes[9] = 9        # 8..9: tree, seed is self-pointing base 9
    tgt_nodes = tgt_nodes[np.newaxis, :]
    tree_idx, seed_mask = topo_graph.isolate_graphtrees(tgt_nodes)
    assert np.count_nonzero(seed_mask) == 3 and tree_idx.max() == 1, 'Unexpected input array'

    # Test first-to-second tree merging
    new_tgt_nodes, flip_mask = topo_graph.merge_graphtrees(tgt_nodes, np.asarray([[2, 8]]))
    tree_idx, seed_mask = topo_graph.isolate_graphtrees(new_tgt_nodes)
    assert np.count_nonzero(seed_mask) == 1 and tree_idx.max() == 0, 'Must have single graph-tree'
    np.testing.assert_equal(flip_mask, [0,0, 1,1, 1,1,1,1, 0,0], 'Unexpected flip_mask')

    # Test second-to-first tree merging
    new_tgt_nodes, flip_mask = topo_graph.merge_graphtrees(tgt_nodes, np.asarray([[8, 2]]))
    tree_idx, seed_mask = topo_graph.isolate_graphtrees(new_tgt_nodes)
    assert np.count_nonzero(seed_mask) == 2 and tree_idx.max() == 0, 'Must have single graph-tree'
    np.testing.assert_equal(flip_mask, [0,0, 0,0, 0,0,0,0, 1,1], 'Unexpected flip_mask')

    # Test two-edge merge: 0->2 and 8->2
    edge_list = np.stack(([[0, 2]], [[8, 2]]), axis=-1)
    edge_list_orig = edge_list.copy()
    new_tgt_nodes, flip_mask = topo_graph.merge_graphtrees(tgt_nodes, edge_list)
    np.testing.assert_equal(edge_list, edge_list_orig, 'The edge_list argument was changed')
    tree_idx, seed_mask = topo_graph.isolate_graphtrees(new_tgt_nodes)
    assert tree_idx.max() == 0, 'Must have single graph-tree'
    np.testing.assert_equal(seed_mask, [1,1, 1,1, 1,0,0,0, 0,0], 'Unexpected loop location')
    np.testing.assert_equal(flip_mask, [1,1, 0,0, 1,1,1,1, 1,1], 'Unexpected flip_mask')

def test_graph_combined():
    """Combined graph tests"""
    # Use negative indices, 2 looped graph-trees
    tgt_nodes = np.arange(8) - 2
    tree_idx, seed_mask = topo_graph.isolate_graphtrees(tgt_nodes[np.newaxis, :])
    np.testing.assert_equal(seed_mask,  True, 'All elements must be part of a loop')
    np.testing.assert_equal(tree_idx[::2], 0, 'Even elements must be from the first tree')
    np.testing.assert_equal(tree_idx[1::2], 1, 'Odd elements must be from the second tree')

    # Build graph with two loops and multiple branches
    tgt_nodes = np.arange(2*3*5) - 1
    tgt_nodes[20] = -1      # 20..29: loop
    tgt_nodes[10] = -2      # 19..10: branch, base 28
    tgt_nodes[5] = 5        # 9..5: simple branch, self-pointing/loop base 5
    # leftover:             # 4..0: branch, base 29
    tgt_nodes = tgt_nodes[np.newaxis, :]
    tree_idx, seed_mask = topo_graph.isolate_graphtrees(tgt_nodes)
    assert tree_idx.max() == 1, 'Must have two graph-trees'
    ref_mask = np.zeros_like(seed_mask)
    ref_mask[5] = ref_mask[20:] = True
    np.testing.assert_equal(seed_mask,  ref_mask, 'Element 5 and after 20 must be part of a loop')
    # Reshape to 3D shape
    new_tgt_nodes = topo_graph.reshape_graph(tgt_nodes, shape=(5,3,2))
    tree_idx, seed_mask = topo_graph.isolate_graphtrees(new_tgt_nodes)
    assert tree_idx.max() == 1, 'Must have two graph-trees'
    np.testing.assert_equal(np.count_nonzero(seed_mask), 11, '11 elements must be part of a loop')
    # Leave loop elements only
    new_tgt_nodes = topo_graph.mask_graph(new_tgt_nodes, seed_mask)
    tree_idx, seed_mask = topo_graph.isolate_graphtrees(new_tgt_nodes)
    assert tree_idx.max() == 1, 'Must have two graph-trees'
    np.testing.assert_equal(seed_mask, True, 'All elements must be part of a loop')

    # Test strict ravel_multi_index()/unravel_index() compatibility
    def assert_unravel(res_nodes, flat_nodes):
        """Validate after ravel the result"""
        nodes = np.ravel_multi_index(res_nodes, res_nodes.shape[1:])
        nodes = nodes[np.unravel_index(range(res_nodes[0].size), res_nodes.shape[1:])]
        np.testing.assert_equal(nodes[np.newaxis], flat_nodes % flat_nodes.shape[1])
    # Reshape 1D to 3D
    new_tgt_nodes = topo_graph.reshape_graph(tgt_nodes, shape=(2,5,3), strict_unravel=True)
    assert_unravel(new_tgt_nodes, tgt_nodes)
    np.testing.assert_equal(new_tgt_nodes, topo_graph.reshape_graph(tgt_nodes, shape=(2,5,3), strict_unravel=False),
            'CHECKME: Unexpected result between "strict_unravel" modes')
    # Reshape 3D to 2D
    new_tgt_nodes = topo_graph.reshape_graph(new_tgt_nodes, shape=(6,5), strict_unravel=True)
    assert_unravel(new_tgt_nodes, tgt_nodes)
    np.testing.assert_equal(new_tgt_nodes, topo_graph.reshape_graph(tgt_nodes, shape=(6,5), strict_unravel=False),
            'CHECKME: Unexpected result between "strict_unravel" modes')

    # Test 3D shape merge (single-edge merge, then reshape back)
    new_tgt_nodes = topo_graph.reshape_graph(tgt_nodes, shape=(5,3,2), strict_unravel=True)
    edge_list = np.asarray(np.unravel_index([4, 5], shape=new_tgt_nodes.shape[1:]))
    new_tgt_nodes, flip_mask = topo_graph.merge_graphtrees(new_tgt_nodes, edge_list)
    np.testing.assert_equal(flip_mask[*edge_list], [True, False], 'Merged edges must be in flip_mask, target ones must NOT')
    new_tgt_nodes = topo_graph.reshape_graph(new_tgt_nodes, shape=tgt_nodes.shape[1:], strict_unravel=True)
    #  Reference from merged 1D graph (adjust negative indices)
    new_tgt_nodes_ref, flip_mask_ref = topo_graph.merge_graphtrees(tgt_nodes, np.asarray([[4, 5]]))
    new_tgt_nodes_ref %= tgt_nodes.shape[1:]
    assert np.count_nonzero(flip_mask) == np.count_nonzero(flip_mask_ref), 'Unexpected number of flips'
    np.testing.assert_equal(new_tgt_nodes, new_tgt_nodes_ref, 'Unexpected 3D merge result')

#
# For non-pytest debugging
#
if __name__ == '__main__':
    test_reshape_graph()
    test_isolate_graphtrees()
    test_merge_graphtrees()
    test_graph_combined()
