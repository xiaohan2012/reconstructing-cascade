import numpy as np
from graph_tool import GraphView
from graph_tool.all import pbfs_search, label_components
from utils import extract_edges_from_pred, init_visitor
from gt_utils import get_roots


def is_order_respected(tree, root, obs_nodes, infection_times):
    tree = GraphView(tree)
    obs_set = set(obs_nodes)
    vfilt = tree.new_vertex_property('bool')
    vfilt.a = True
    tree.set_vertex_filter(vfilt)

    leaves = [o for o in obs_nodes if tree.vertex(o).out_degree() == 0]
    vis = init_visitor(tree, root)
    pbfs_search(tree, root, terminals=leaves, visitor=vis, count_threshold=-1)
    for l in leaves:
        edges = extract_edges_from_pred(tree, root, l, vis.pred)
        edges = edges[::-1]
        path = list(edges[0]) + [u for _, u in edges[1:]]
        useful_nodes_on_path = [v for v in path if v in obs_set]

        for i in range(len(useful_nodes_on_path)-1):
            u, v = useful_nodes_on_path[i: i+2]
            if infection_times[u] > infection_times[v]:
                return False
    return True


def is_arborescence(tree):
    # is tree?
    l, _ = label_components(GraphView(tree, directed=False))
    if not np.all(np.array(l.a) == 0):
        print('not connected')
        print(np.array(l.a))
        return False

    in_degs = np.array([v.in_degree() for v in tree.vertices()])
    if in_degs.max() > 1:
        print('in_degree.max() > 1')
        return False
    if np.sum(in_degs == 1) != (tree.num_vertices() - 1):
        print('should be: only root has no parent')
        return False

    roots = get_roots(tree)
    assert len(roots) == 1, '>1 roots'
    
    return True


def is_feasible(tree, root, obs_nodes, infection_times):
    if not is_arborescence(tree):
        print('not arborescence')
        return False
    for o in obs_nodes:
        try:
            tree.vertex(o)
        except ValueError:
            print('does not span {}'.format(o))
            return False
    if not is_order_respected(tree, root, obs_nodes, infection_times):
        print('not respecting order')
        return False
    return True
