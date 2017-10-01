import numpy as np
from tqdm import tqdm

from graph_tool import Graph, GraphView
from graph_tool.search import cpbfs_search, bfs_iterator
from pyedmond import find_minimum_branching

from utils import init_visitor, extract_edges_from_pred
from errors import TreeNotFound


def get_edges(dist, root, terminals):
    """the set of edges (root, terminal) where root reaches terminal
    """
    return ((root, t, dist[t])
            for t in terminals
            if dist[t] != -1 and t != root)


def build_closure_with_order(
        g, cand_source, terminals, infection_times, k=-1,
        strictly_smaller=True,
        debug=False,
        verbose=False):
    """
    build transitive closure with infection order constraint

    g: gt.Graph(directed=False)
    cand_source: int
    terminals: list of int
    infection_times: dict int -> float

    build a clojure graph in which cand_source + terminals are all connected to each other.
    the number of neighbors of each node is determined by k

    the larger the k, the denser the graph

    note that vertex ids are preserved (without re-mapping to consecutive integers)

    return:

    gt.Graph(directed=True)
    """
    r2pred = {}
    edges = {}
    terminals = list(terminals)

    # from cand_source to terminals
    vis = init_visitor(g, cand_source)
    cpbfs_search(g, source=cand_source, visitor=vis, terminals=terminals,
                 forbidden_nodes=terminals,
                 count_threshold=k)
    r2pred[cand_source] = vis.pred
    for u, v, c in get_edges(vis.dist, cand_source, terminals):
        edges[(u, v)] = c

    if debug:
        print('cand_source: {}'.format(cand_source))
        print('#terminals: {}'.format(len(terminals)))
        print('edges from cand_source: {}'.format(edges))

    if verbose:
        terminals_iter = tqdm(terminals)
        print('building closure graph')
    else:
        terminals_iter = terminals

    # from terminal to other terminals
    for root in terminals_iter:

        if strictly_smaller:
            late_terminals = [t for t in terminals
                              if infection_times[t] > infection_times[root]]
        else:
            # respect what the paper presents
            late_terminals = [t for t in terminals
                              if infection_times[t] >= infection_times[root]]

        late_terminals = set(late_terminals) - {cand_source}  # no one can connect to cand_source
        if debug:
            print('root: {}'.format(root))
            print('late_terminals: {}'.format(late_terminals))
        vis = init_visitor(g, root)
        cpbfs_search(g, source=root, visitor=vis, terminals=list(late_terminals),
                     forbidden_nodes=list(set(terminals) - set(late_terminals)),
                     count_threshold=k)
        r2pred[root] = vis.pred
        for u, v, c in get_edges(vis.dist, root, late_terminals):
            if debug:
                print('edge ({}, {})'.format(u, v))
            edges[(u, v)] = c

    if verbose:
        print('returning closure graph')

    gc = Graph(directed=True)

    gc.add_vertex(g.num_vertices())

    vfilt = gc.new_vertex_property('bool')
    vfilt.a = False
    
    for (u, v) in edges:
        gc.add_edge(u, v)
        vfilt[u] = vfilt[v] = True

    eweight = gc.new_edge_property('int')
    eweight.set_2d_array(np.array(list(edges.values())))
    gc.set_vertex_filter(vfilt)
    return gc, eweight, r2pred


def find_tree_by_closure(
        g, root, infection_times, terminals,
        closure_builder=build_closure_with_order,
        strictly_smaller=True,
        return_closure=False,
        k=-1,
        debug=False,
        verbose=True):
    """find the steiner tree by trainsitive closure
    
    """
    gc, eweight, r2pred = closure_builder(g, root, terminals,
                                          infection_times,
                                          strictly_smaller=strictly_smaller,
                                          k=k,
                                          debug=debug,
                                          verbose=verbose)

    # get the minimum spanning arborescence
    # graph_tool does not provide minimum_spanning_arborescence
    if verbose:
        print('getting mst')
    tree_edges = find_minimum_branching(gc,  [root], weights=eweight)
    
    efilt = gc.new_edge_property('bool')
    efilt.a = False
    for u, v in tree_edges:
        efilt[gc.edge(u, v)] = True

    mst_tree = GraphView(gc, efilt=efilt)

    if verbose:
        print('extract edges from original graph')

    # extract the edges from the original graph

    # sort observations by time
    # and also topological order
    # why doing this: we want to start collecting the edges
    # for nodes with higher order
    topological_index = {}
    for i, e in enumerate(bfs_iterator(mst_tree, source=root)):
        topological_index[int(e.target())] = i

    print('mst_tree', mst_tree)
    print('infection_times', infection_times)
    print('topological_index', topological_index)

    try:
        sorted_obs = sorted(
            set(terminals) - {root},
            key=lambda o: (infection_times[o], topological_index[o]))
    except KeyError:
        raise TreeNotFound("it's likely that the input cannot produce a feasible solution" +
                           "because the topological sort on terminals does not visit all terminals")

    # next, we start reconstructing the minimum steiner arborescence
    tree_nodes = {root}
    tree_edges = set()
    # print('root', root)
    for u in sorted_obs:
        if u in tree_nodes:
            if debug:
                print('{} covered already'.format(u))
            continue
        # print(u)
        v, u = map(int, next(mst_tree.vertex(u).in_edges()))  # v is ancestor
        tree_nodes.add(v)

        late_nodes = [n for n in terminals if infection_times[n] > infection_times[u]]
        vis = init_visitor(g, u)
        # from child to any tree node, including v

        cpbfs_search(g, source=u, terminals=list(tree_nodes),
                     forbidden_nodes=late_nodes,
                     visitor=vis,
                     count_threshold=1)
        # dist, pred = shortest_distance(g, source=u, pred_map=True)
        node_set = {v for v, d in vis.dist.items() if d > 0}
        reachable_tree_nodes = node_set.intersection(tree_nodes)
        ancestor = min(reachable_tree_nodes, key=vis.dist.__getitem__)

        edges = extract_edges_from_pred(g, u, ancestor, vis.pred)
        edges = {(j, i) for i, j in edges}  # need to reverse it
        if debug:
            print('tree_nodes', tree_nodes)
            print('connecting {} to {}'.format(v, u))
            print('using ancestor {}'.format(ancestor))
            print('adding edges {}'.format(edges))
        tree_nodes |= {u for e in edges for u in e}

        tree_edges |= edges

    t = Graph(directed=True)
    t.add_vertex(g.num_vertices())

    for u, v in tree_edges:
        t.add_edge(t.vertex(u), t.vertex(v))

    tree_nodes = {u for e in tree_edges for u in e}
    vfilt = t.new_vertex_property('bool')
    vfilt.a = False
    for v in tree_nodes:
        vfilt[t.vertex(v)] = True

    t.set_vertex_filter(vfilt)

    if return_closure:
        return t, gc, mst_tree
    else:
        return t
