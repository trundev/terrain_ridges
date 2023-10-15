"""Pytest for topo_graph.py"""

def test_module():
    """Run all 'test_*' functions from topo_graph.py"""
    from tools import topo_graph

    for name, fn in topo_graph.__dict__.items():
        if callable(fn) and name.startswith('test_'):
            print('')
            print(f'# Invoking topo_graph.{name}()...')
            fn()
