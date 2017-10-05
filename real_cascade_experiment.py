import numpy as np
import pandas as pd
import pickle as pkl

from tqdm import tqdm
from glob import glob

from graph_tool import load_graph
from paper_experiment import get_tree
from cascade import observe_cascade
from gt_utils import extract_edges
from evaluate import edge_order_accuracy


def run_k_runs(g, q, infection_times, method,
               k, result_dir,
               verbose=False):
    for i in range(k):
        obs = observe_cascade(infection_times, source=None, q=q)
        tree = get_tree(g, infection_times, source=None, obs_nodes=obs, method=method, verbose=verbose)

        pred_edges = extract_edges(tree)
        pkl.dump(pred_edges,
                 open(result_dir + '/{}.pkl'.format(i), 'wb'))


def evaluate(pred_edges, infection_times):
    pred_nodes = set([i for e in pred_edges for i in e])
    true_nodes = set(np.nonzero(infection_times >= 0)[0])

    prec = len(pred_nodes.intersection(true_nodes)) / len(pred_nodes)
    rec = len(pred_nodes.intersection(true_nodes)) / len(true_nodes)

    order_acc = edge_order_accuracy(pred_edges, infection_times)
    return prec, rec, order_acc


def evaluate_from_result_dir(result_dir, infection_times):
    rows = []
    for p in glob(result_dir + "/*.pkl"):
        # print(p)
        # TODO: add root
        pred_edges = pkl.load(open(p, 'rb'))

        scores = evaluate(pred_edges, infection_times)
        rows.append(scores)
    path = result_dir + "/{}.pkl".format(q)
    if rows:
        df = pd.DataFrame(rows, columns=['n.prec', 'n.rec',
                                         'order accuracy'])
        return (path, df)
    else:
        if os.path.exists(path):
            os.remove(path)
        return None


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', required=True)
    parser.add_argument('-q', '--report_proba', type=float, default=0.1)
    parser.add_argument('-k', '--repeat_times', type=int, default=100)
    parser.add_argument('-o', '--output_dir', default='output/real_cascade')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--evaluate', type=bool, default=False)

    args = parser.parse_args()
    
    g = load_graph('data/digg/cascade_graph.gt')
    infection_times = pkl.load(open('data/digg/cascade.pkl', 'rb'))
    
    q = args.report_proba
    k = args.repeat_times
    method = args.method
    output_dir = args.output_dir

    result_dir = os.path.join(output_dir, method, "{}".format(q))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print(args.evaluate)
    if not args.evaluate:
        print('run experiment...')
        run_k_runs(g, q, infection_times, method, k, result_dir, verbose=args.verbose)
    else:
        print('evaluate...')
        path, df = evaluate_from_result_dir(result_dir,
                                            infection_times=infection_times)
        print(path)
        summary = df.describe()
        print(summary)
        print('writing to {}'.format(path))
        summary.to_pickle(path)
