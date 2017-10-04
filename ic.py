import math
import itertools
import networkx as nx
import numpy as np
import random

from graph_tool.all import (shortest_distance, Graph, GraphView, label_largest_component)

from utils import get_infection_time


def sample_graph_from_infection(g):
    rands = np.random.rand(g.number_of_edges())
    active_edges = [(u, v) for (u, v), r in zip(g.edges_iter(), rands) if g[u][v]['p'] >= r]
    induced_g = nx.Graph()
    induced_g.add_nodes_from(g.nodes())
    induced_g.add_edges_from(active_edges)
    for u, v in induced_g.edges_iter():
        induced_g[u][v]['d'] = g[u][v]['d']
    return induced_g


def sample_graph_by_p(g, p):
    """
    graph_tool version of sampling a graph
    mask the edge according to probability p and return the masked graph"""
    flags = (np.random.random(g.num_edges()) <= p)
    p = g.new_edge_property('bool')
    p.set_2d_array(flags)
    return GraphView(g, efilt=p)


def simulate_cascade(g, p, source=None, return_tree=False):
    """
    graph_tool version of simulating cascade
    return np.ndarray on vertices as the infection time in cascade
    uninfected node has dist -1
    """
    gv = sample_graph_by_p(g, p)

    if source is None:
        # consider the largest cc
        infected_nodes = np.nonzero(label_largest_component(gv).a)[0]
        source = np.random.choice(infected_nodes)
    
    times = get_infection_time(gv, source)

    if return_tree:
        # get the tree edges
        _, pred_map = shortest_distance(gv, source=source, pred_map=True)
        edges = [(pred_map[i], i) for i in infected_nodes if i != source]

        # create tree
        tree = Graph(directed=True)
        tree.add_vertex(g.num_vertices())
        for u, v in edges:
            tree.add_edge(int(u), int(v))
            vfilt = tree.new_vertex_property('bool')
            vfilt.a = False
        for v in set(itertools.chain(*edges)):
            vfilt[v] = True
        tree.set_vertex_filter(vfilt)

    if return_tree:
        return source, times, tree
    else:
        return source, times
