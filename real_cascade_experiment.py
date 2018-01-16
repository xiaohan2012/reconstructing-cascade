import numpy as np
import pandas as pd
import pickle as pkl

from tqdm import tqdm
from graph_tool import load_graph
from graph_tool.topology import label_largest_component
from joblib import Parallel, delayed
    
from paper_experiment import get_tree
from cascade import observe_cascade
from gt_utils import extract_edges
from utils import cascade_size
from evaluate import edge_order_accuracy


def one_run(g, q, result_dir, i,
            verbose):
    obs = observe_cascade(infection_times, source=None, q=q)
    tree = get_tree(g, infection_times, source=None, obs_nodes=obs, method=method, verbose=verbose)

    pred_edges = extract_edges(tree)
    pkl.dump((obs, pred_edges),
             open(result_dir + '/{}.pkl'.format(i), 'wb'))


def run_k_runs(g, q, infection_times, method,
               k, result_dir,
               verbose=False):
    Parallel(n_jobs=-1)(delayed(one_run)(g, q, result_dir, i,
                                         verbose)
                        for i in tqdm(range(k), total=k))


def evaluate(pred_edges, infection_times):
    pred_nodes = set([i for e in pred_edges for i in e])
    true_nodes = set(np.nonzero(infection_times >= 0)[0])

    correct_nodes = pred_nodes.intersection(true_nodes)
    # prec = len(correct_nodes) / len(pred_nodes)
    # rec = len(correct_nodes) / len(true_nodes)

    n_correct_edges, n_pred_edges = edge_order_accuracy(pred_edges, infection_times, return_count=True)

    return (len(correct_nodes), len(pred_nodes), len(true_nodes),
            n_correct_edges, n_pred_edges)


def evaluate_from_result_dir(result_dir, infection_times, k):
    rows = []
    paths = [result_dir + "/{}.pkl".format(i) for i in range(k)]
    for p in paths:
        print(p)
        # TODO: add root
        try:
            pred_edges = pkl.load(open(p, 'rb'))
            scores = evaluate(pred_edges, infection_times)
            rows.append(scores)
        except FileNotFoundError:
            print(p, ' not found')

    path = result_dir + ".pkl"  # {q}.pkl
    if rows:
        df = pd.DataFrame(rows, columns=['n.correct_nodes',
                                         'n.pred_nodes',
                                         'n.true_nodes',
                                         'n.correct_edges',
                                         'n.pred_edges'])
        return (path, df)
    else:
        if os.path.exists(path):
            os.remove(path)
        return None


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--cascade_id', required=True)
    parser.add_argument('-m', '--method', required=True)
    parser.add_argument('-q', '--report_proba', type=float, default=0.1)
    parser.add_argument('-k', '--repeat_times', type=int, default=100)
    parser.add_argument('-o', '--output_dir', default='output/real_cascade')
    parser.add_argument('-s', '--small_cascade', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--evaluate', type=bool, default=False)

    args = parser.parse_args()
    
    g = load_graph('data/digg/graph.gt')
    if args.small_cascade:
        cascade_path = 'data/digg/small_cascade_{}.pkl'.format(args.cascade_id)
    else:
        cascade_path = 'data/digg/cascade_{}.pkl'.format(args.cascade_id)
    print('cascade_path: ', cascade_path)

    infection_times = pkl.load(open(cascade_path,
                                    'rb'))
    print('cascade size: ', len(np.nonzero(infection_times > 0)[0]))
    
    q = args.report_proba
    k = args.repeat_times
    method = args.method
    output_dir = args.output_dir

    result_dir = os.path.join(output_dir, method, "{}".format(q))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if not args.evaluate:
        print('run experiment...', 'q=', q, ', method=', method, 'cascade: ', args.cascade_id,
              'cascade size: ', cascade_size(infection_times))
        print(g)
        print(sum(label_largest_component(g).a))
        run_k_runs(g, q, infection_times, method, k, result_dir, verbose=args.verbose)
    else:
        print('evaluate...')
        path, df = evaluate_from_result_dir(result_dir,
                                            infection_times=infection_times,
                                            k=k)
        print(path)
        summary = df.describe()
        print(summary)
        print('writing to {}'.format(path))
        summary.to_pickle(path)
