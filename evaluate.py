import os
import pandas as pd
import pickle as pkl
from graph_tool.all import load_graph
from glob import glob
from tqdm import tqdm
from utils import edges2graph
from infer_time import fill_missing_time

from scipy.stats import kendalltau
from feasibility import is_arborescence


def edge_order_accuracy(pred_edges, infection_times):
    n_correct_edges = sum(1
                          for u, v in pred_edges
                          if infection_times[u] <= infection_times[v])
    return n_correct_edges / len(pred_edges)
    

# @profile
def evaluate_performance(g, root, source, pred_edges, obs_nodes, infection_times,
                         true_edges):
    # change -1 to infinity (for order comparison)
    # infection_times[infection_times == -1] = float('inf')

    true_nodes = {i for e in true_edges for i in e}
    pred_nodes = {i for e in pred_edges for i in e}

    common_nodes = true_nodes.intersection(pred_nodes)
    n_prec = len(common_nodes) / len(pred_nodes)
    n_rec = len(common_nodes) / len(true_nodes)
    obj = len(pred_edges)

    pred_tree = edges2graph(g, pred_edges)

    root = next(v
                for v in pred_tree.vertices()
                if v.in_degree() == 0 and v.out_degree() > 0)

    assert is_arborescence(pred_tree)
    
    pred_times = fill_missing_time(g, pred_tree, root, obs_nodes,
                                   infection_times, debug=False)
    
    # consider only predicted nodes that are actual infections
    nodes = list(common_nodes)
    rank_corr = kendalltau(pred_times[nodes], infection_times[nodes])[0]

    common_edges = set(pred_edges).intersection(true_edges)

    e_prec = len(common_edges) / len(pred_edges)
    e_rec = len(common_edges) / len(true_edges)

    # order accuracy on edge
    edges = [e for e in pred_edges
             if (e[0] in common_nodes and
                 e[1] in common_nodes)]
    if len(edges) > 0:
        order_accuracy = edge_order_accuracy(edges, infection_times)
    else:
        order_accuracy = 0.0

    return (n_prec, n_rec, obj, e_prec, e_rec, rank_corr, order_accuracy)


def evaluate_from_result_dir(g, result_dir, qs):
    for q in tqdm(qs):
        rows = []
        for p in glob(result_dir + "/{}/*.pkl".format(q)):
            # print(p)
            # TODO: add root
            infection_times, source, obs_nodes, true_edges, pred_edges = pkl.load(open(p, 'rb'))
            
            root = None

            try:
                scores = evaluate_performance(g, root, source, pred_edges, obs_nodes,
                                              infection_times, true_edges)
            except AssertionError:
                import sys
                print(p)
                print(sys.exc_info()[0])
                raise

            rows.append(scores)
        path = result_dir + "/{}.pkl".format(q)
        if rows:
            df = pd.DataFrame(rows, columns=['n.prec', 'n.rec',
                                             'obj',
                                             'e.prec', 'e.rec',
                                             'rank-corr',
                                             'order accuracy'
            ])
            yield (path, df)
        else:
            if os.path.exists(path):
                os.remove(path)
            yield None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gtype', required=True)
    parser.add_argument('-l', '--model', required=True)
    parser.add_argument('-m', '--method', required=True)
    parser.add_argument('-q', '--qs', type=float, nargs="+")
    parser.add_argument('-o', '--output_dir', default='outputs/paper_experiment')

    args = parser.parse_args()
    gtype = args.gtype
    qs = args.qs
    method = args.method
    model = args.model
    output_dir = args.output_dir

    print("""graph: {}
model: {}
qs: {}
method: {}""".format(gtype, model, qs, method))

    result_dir = "{output_dir}/{gtype}/{model}/{method}/qs".format(
        output_dir=output_dir,
        gtype=gtype,
        model=model,
        method=method)

    g = load_graph('data/{}/graph.gt'.format(gtype))

    for r in evaluate_from_result_dir(g, result_dir, qs):
        if r:
            path, df = r
            print('writing to {}'.format(path))
            df.describe().to_pickle(path)
        else:
            print('not result.')
