import numpy as np
from graph_tool import Graph
from evaluate import evaluate_performance


def test_evaluate():
    g = Graph(directed=False)
    g.add_vertex(5)
    root = source = 0
    true_edges = [(0, 1), (1, 2), (2, 3)]

    # perfect case
    pred_edges = true_edges
    obs_nodes = [0, 3]
    infection_times = np.array([0, 1, 2, 3, -1])

    (n_prec, n_rec, obj, e_prec, e_rec,
     rank_corr, order_accuracy) = evaluate_performance(
         g, root, source, pred_edges, obs_nodes,
         infection_times,
         true_edges)
    assert np.isclose(n_prec, 1)
    assert np.isclose(n_rec, 1)
    assert obj == 3
    assert np.isclose(e_prec, 1)
    assert np.isclose(e_rec, 1)
    assert np.isclose(rank_corr,  1)
    assert np.isclose(order_accuracy, 1)

    # non-perfect case
    pred_edges = [(0, 1), (1, 4), (4, 2), (2, 3)]
    
    (n_prec, n_rec, obj, e_prec, e_rec,
     rank_corr, order_accuracy) = evaluate_performance(
         g, root, source, pred_edges, obs_nodes,
         infection_times,
         true_edges)
    assert np.isclose(n_prec, 0.8)
    assert np.isclose(n_rec, 1)
    assert obj == 4
    assert np.isclose(e_prec, 0.5)
    assert np.isclose(e_rec, 2 / 3.0)
    assert rank_corr < 1.0
    assert np.isclose(order_accuracy, 1.0)  # only (0, 1), (2, 3) are considered
