import pytest
import networkx as nx
from synthetic_data import load_data_by_gtype, GRID, PL_TREE
from graph_generator import add_p_and_delta
from cascade import gen_nontrivial_cascade
from graph_tool.all import load_graph


@pytest.fixture
def simulated_cascade_summary():
    return load_data_by_gtype(GRID, '10')


@pytest.fixture
def tree_and_cascade():
    g = load_graph('data/balanced-tree/2-6/graph.gt')
    c, s, o = gen_nontrivial_cascade(g, 0.8, 0.5)
    return g, c, s, o


@pytest.fixture
def grid_and_cascade():
    g = load_graph('data/grid/2-6/graph.gt')
    c, s, o = gen_nontrivial_cascade(g, 0.8, 0.5)
    return g, c, s, o


def setup_function(module):
    import random
    import numpy as np
    seed = 123456
    random.seed(seed)
    np.random.seed(seed)
