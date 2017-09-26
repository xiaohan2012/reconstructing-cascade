
import argparse
import pickle as pkl
import networkx as nx
import numpy as np
from networkx.generators.random_graphs import random_powerlaw_tree
from graph_generator import kronecker_random_graph, grid_2d, \
    P_peri ,P_hier, P_rand, add_p_and_delta


KRONECKER_RAND = 'kr-rand'
KRONECKER_PERI = 'kr-peri'
KRONECKER_HIER = 'kr-hier'
GRID = 'grid'
PL_TREE = 'pl-tree'
B_TREE = 'balanced-tree'
ER = 'er'
BARABASI = 'barabasi'
CLIQUE = 'clique'
LINE = 'line'
all_graph_types = [KRONECKER_RAND,
                   KRONECKER_PERI,
                   KRONECKER_HIER,
                   GRID,
                   PL_TREE,
                   B_TREE,
                   ER,
                   BARABASI,
                   CLIQUE,
                   LINE]


INF_TIME_PROBA_FILE = 'inf_time_proba_matrix'
NODE2ID_FILE = 'node2id'
ID2NODE_FILE = 'id2node'
REWARD_TABLE_NAME = 'edge_reward_tables'

TIMES_FILE_SUFFIX = 'source2times'
SP_LEN_NAME = 'sp_len'


def extract_larges_CC(g):
    nodes = max(nx.connected_components(g), key=len)
    return g.subgraph(nodes)


def gen_kronecker(P, k=8, n_edges=512):
    g = kronecker_random_graph(k, P, n_edges=n_edges, directed=False)
    return extract_larges_CC(g)


def load_data_by_gtype(gtype, size_param_str):
    g = nx.read_gpickle('data/{}/{}/graph.gpkl'.format(gtype, size_param_str))
    try:
        dir_tbl, inf_tbl = pkl.load(open('data/{}/{}/{}.pkl'.format(
            gtype, size_param_str,
            REWARD_TABLE_NAME), 'rb'))
    except IOError:
        dir_tbl, inf_tbl = None, None

    try:
        sp_len = np.load('data/{}/{}/{}.npz.npy'.format(
            gtype, size_param_str,
            SP_LEN_NAME))
    except IOError:
        sp_len = None

    try:
        time_probas = pkl.load(open('data/{}/{}/{}.pkl'.format(gtype, size_param_str,
                                                               INF_TIME_PROBA_FILE), 'rb'))
    except IOError:
        time_probas = None

    try:
        node2id = pkl.load(open('data/{}/{}/{}.pkl'.format(gtype, size_param_str,
                                                           NODE2ID_FILE), 'rb'))
        id2node = pkl.load(open('data/{}/{}/{}.pkl'.format(gtype, size_param_str,
                                                           ID2NODE_FILE), 'rb'))
    except IOError:
        node2id, id2node = None, None
    return g, time_probas, dir_tbl, inf_tbl, sp_len, node2id, id2node


def main():
    import os
    
    p = 0.7
    delta = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', choices=all_graph_types,
                        help='graph type')
    parser.add_argument('-s', '--size', type=int,
                        default=0,
                        help="size of graph")
    parser.add_argument('-e', '--size_exponent', type=int,
                        default=1,
                        help="exponent of the size")
    parser.add_argument('-b', '--exponent_base', type=int,
                        default=10,
                        help="base of the size exponent")
    parser.add_argument('-n', '--n_rounds', type=int,
                        default=100,
                        help="number of simulated cascades")

    args = parser.parse_args()
    gtype = args.type
    if args.size:
        size = args.size
        output_dir = 'data/{}/{}'.format(gtype, size)
    else:
        size = args.exponent_base ** args.size_exponent
        output_dir = 'data/{}/{}-{}'.format(gtype, args.exponent_base,
                                            args.size_exponent)
    if gtype == KRONECKER_HIER:
        g = gen_kronecker(P=P_hier, k=args.size_exponent, n_edges=2**args.size_exponent * 3)
    elif gtype == KRONECKER_PERI:
        g = gen_kronecker(P=P_peri, k=args.size_exponent, n_edges=2**args.size_exponent * 3)
    elif gtype == KRONECKER_RAND:
        g = gen_kronecker(P=P_rand, k=args.size_exponent, n_edges=2**args.size_exponent * 3)
    elif gtype == PL_TREE:
        p = 0.88
        g = random_powerlaw_tree(size, tries=999999)
    elif gtype == B_TREE:
        g = nx.balanced_tree(args.exponent_base, args.size_exponent-1)
    elif gtype == ER:
        g = extract_larges_CC(nx.fast_gnp_random_graph(size, 0.1))
    elif gtype == BARABASI:
        g = extract_larges_CC(nx.barabasi_albert_graph(size, 5))
    elif gtype == GRID:
        g = grid_2d(int(np.sqrt(size)))
    elif gtype == CLIQUE:
        g = nx.complete_graph(size)
    elif gtype == LINE:
        g = nx.path_graph(size)
    else:
        raise ValueError('unsupported graph type {}'.format(gtype))

    g.remove_edges_from(g.selfloop_edges())
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))

    if gtype == GRID:
        mapping = {(i, j): int(np.sqrt(size)) * i + j for i, j in g.nodes_iter()}
        g = nx.relabel_nodes(g, mapping)
    else:
        g = nx.convert_node_labels_to_integers(g)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('graph type: {}'.format(gtype))
    # g = add_p_and_delta(g, p, delta)
    output_path = '{}/graph.graphml'.format(output_dir, gtype)
    print('saving to {}'.format(output_path))
    nx.write_graphml(g, output_path)
    nx.write_gpickle(g, '{}/graph.gpkl'.format(output_dir, gtype))

    if False:
        pkl.dump(time_probas,
                 open('{}/{}.pkl'.format(output_dir, INF_TIME_PROBA_FILE), 'wb'))

        pkl.dump(node2id,
                 open('{}/{}.pkl'.format(output_dir, NODE2ID_FILE), 'wb'))
        pkl.dump(id2node,
                 open('{}/{}.pkl'.format(output_dir, ID2NODE_FILE), 'wb'))
    
if __name__ == "__main__":
    main()
