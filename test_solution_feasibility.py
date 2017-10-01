import numpy as np
import pytest
from graph_tool.all import load_graph

from steiner_tree_greedy import steiner_tree_greedy
from steiner_tree import get_steiner_tree
from core import find_tree_by_closure

from temporal_bfs import temporal_bfs
from utils import earliest_obs_node
from feasibility import is_feasible
from cascade import gen_nontrivial_cascade
from gt_utils import is_arborescence


QS = np.linspace(0.1, 1.0, 10)
MODELS = ['si', 'ct']

P = 0.5
K = 10


@pytest.fixture
def cascades_on_grid():
    cascades = []
    g = load_graph('data/grid/2-6/graph.gt')
    for model in MODELS:
        for q in QS:
            for i in range(K):
                ret = gen_nontrivial_cascade(
                    g, P, q, model=model, return_tree=True,
                    source_includable=True)
                ret = (g, ) + ret + (model, q, i)
                cascades.append(ret)  # g, infection_times, source, obs_nodes, true_tree
    return cascades
        

def test_greedy(cascades_on_grid):
    for g, infection_times, source, obs_nodes, true_tree, model, q, i in cascades_on_grid:
        print(model, q, i)
        root = earliest_obs_node(obs_nodes, infection_times)
        tree = steiner_tree_greedy(
            g, root, infection_times, source, obs_nodes,
            debug=False,
            verbose=True
        )
        assert is_feasible(tree, root, obs_nodes, infection_times)


def test_closure(cascades_on_grid):
    for g, infection_times, source, obs_nodes, true_tree, model, q, i in cascades_on_grid:
        print(model, q, i)
        root = earliest_obs_node(obs_nodes, infection_times)
        tree = find_tree_by_closure(
            g, root, infection_times, obs_nodes,
            strictly_smaller=False,
            debug=False,
            verbose=False,
        )
        assert is_feasible(tree, root, obs_nodes, infection_times)


def test_temporal_bfs(cascades_on_grid):
    for g, infection_times, source, obs_nodes, true_tree, model, q, i in cascades_on_grid:
        print(model, q, i)
        root = earliest_obs_node(obs_nodes, infection_times)
        tree = temporal_bfs(
            g, root, infection_times, source, obs_nodes,
            debug=False,
            verbose=False,
        )
        assert is_feasible(tree, root, obs_nodes, infection_times)


def test_vanilla_steiner_tree(cascades_on_grid):
    for g, infection_times, source, obs_nodes, true_tree, model, q, i in cascades_on_grid:
        print(model, q, i)
        root = earliest_obs_node(obs_nodes, infection_times)
        pred_tree = get_steiner_tree(
            g, root, obs_nodes,
            debug=False,
            verbose=False,
        )
        # it's undirected, so test is a bit different
        assert is_arborescence(pred_tree)

        for o in obs_nodes:
            assert pred_tree.vertex(o) is not None
