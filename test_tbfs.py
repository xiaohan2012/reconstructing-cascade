import numpy as np
from ic import gen_nontrivial_cascade
from utils import tree_sizes_by_roots, get_rank_index
from fixtures import grid_and_cascade, tree_and_cascade


def test_best_tree_sizes_grid_tbfs(grid_and_cascade):
    g, _, infection_times, source, obs_nodes = grid_and_cascade
    scores = tree_sizes_by_roots(g, obs_nodes, infection_times, source,
                                 method='tbfs')
    assert get_rank_index(scores, source) <= 10


def test_full_observation_tree_tbfs(tree_and_cascade):
    g = tree_and_cascade[0]
    for p in np.arange(0.2, 1.0, 0.1):
        infection_times, source, obs_nodes = gen_nontrivial_cascade(g, p, 1.0)
        scores = tree_sizes_by_roots(g, obs_nodes, infection_times, source,
                                     method='tbfs')
        assert get_rank_index(scores, source) == 0


def test_full_observation_grid_tbfs(grid_and_cascade):
    g = grid_and_cascade[0]
    for p in np.arange(0.5, 1.0, 0.1):
        print('p={}'.format(p))
        infection_times, source, obs_nodes = gen_nontrivial_cascade(g, p, 1.0)
        scores = tree_sizes_by_roots(g, obs_nodes, infection_times, source,
                                     method='tbfs')
        assert get_rank_index(scores, source) <= 1.0
