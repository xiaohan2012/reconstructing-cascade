"""finding minimum steiner arborescence
respecting time order
"""

import numpy as np
from tqdm import tqdm
from graph_tool import Graph, GraphView
from graph_tool.search import cpbfs_search, bfs_iterator
from pyedmond import find_minimum_branching
from utils import (extract_edges_from_pred,
                   init_visitor)



def steiner_tree_mst(g, root, infection_times, source, terminals,
                     closure_builder=build_closure,
                     strictly_smaller=True,
                     return_closure=False,
                     k=-1,
                     debug=False,
                     verbose=True):
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
        
    sorted_obs = sorted(
        set(terminals) - {root},
        key=lambda o: (infection_times[o], topological_index[o]))

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
