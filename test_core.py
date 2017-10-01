import pytest
from graph_tool import Graph
from core import (TreeNotFound,
                  build_closure_with_order,
                  find_tree_by_closure)
from gt_utils import extract_edges


@pytest.fixture
def g():
    g = Graph(directed=False)
    g.add_vertex(4)
    g.add_edge_list([(0, 1), (1, 2), (0, 3)])
    return g


@pytest.fixture
def cand_source():
    return 0


@pytest.fixture
def terminals():
    return {0, 1, 2}


@pytest.fixture
def infection_times():
    return {0: 0, 1: 1, 2: 2, 3: -1}


@pytest.fixture
def infection_times_invalid():
    return {0: 0, 1: 2, 2: 1, 3: -1}


def test_build_closure_with_order(g, cand_source, terminals,
                                  infection_times, infection_times_invalid):
    # just a simple test: 4 node tree
    cg, eweight, _ = build_closure_with_order(g, cand_source, terminals, infection_times, debug=True)

    assert set(map(int, cg.vertices())) == {0, 1, 2}
    assert set(extract_edges(cg)) == {(0, 1), (1, 2)}
    edges_and_weights = [(0, 1, 1), (1, 2, 1)]
    for u, v, weight in edges_and_weights:
        assert eweight[cg.edge(u, v)] == weight

    # no feasible solution
    cg, eweight, _ = build_closure_with_order(g, cand_source, terminals,
                                              infection_times_invalid, debug=True)

    assert set(map(int, cg.vertices())) == {0, 1, 2}
    assert set(extract_edges(cg)) == {(0, 1), (2, 1)}
    edges_and_weights = [(0, 1, 1),
                         (2, 1, 1)]  # notice that it's not (1, 2)
    for u, v, weight in edges_and_weights:
        assert eweight[cg.edge(u, v)] == weight


def test_find_tree_by_closure(g, cand_source, terminals,
                              infection_times, infection_times_invalid):
    t = find_tree_by_closure(
        g, cand_source, infection_times, terminals,
        closure_builder=build_closure_with_order,
        strictly_smaller=True,
        return_closure=False,
        k=-1,
        debug=False,
        verbose=False)
    
    assert set(map(int, t.vertices())) == {0, 1, 2}
    assert set(extract_edges(t)) == {(0, 1), (1, 2)}

    with pytest.raises(TreeNotFound):
        t = find_tree_by_closure(
            g, cand_source, infection_times_invalid, terminals,
            closure_builder=build_closure_with_order,
            strictly_smaller=True,
            return_closure=False,
            k=-1,
            debug=True,
            verbose=False)