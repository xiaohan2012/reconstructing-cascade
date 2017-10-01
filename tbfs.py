import numpy as np

from graph_tool.topology import shortest_distance
from collections import defaultdict

from utils import remove_redundant_edges_from_tree, extract_edges


def temporal_bfs(g, r, D, infection_times, source, obs_nodes, debug=False):
    """return the tree covering obs_nodes"""
    queue = [r]
    t_lower = np.ones(g.num_vertices(), dtype=np.int32) * -1  # hidden nodes has lower bound -1
    t_lower[obs_nodes] = infection_times[obs_nodes]
    t_lower[r] = D
    visited = np.zeros(g.num_vertices(), dtype=bool)
    tree = []
    while len(queue) > 0 and np.any(visited[obs_nodes] == 0):
        v = queue.pop(0)

        if debug:
            print('visiting {}'.format(v))

        visited[v] = True
        for u in g.vertex(v).all_neighbours():
            u = int(u)
            if debug:
                print('trying its nbr {}'.format(u))

            if visited[u] == 0:
                if debug:
                    print('{} is not visited'.format(u))
                    print('t_l[u]={}, t_l[v]={}'.format(t_lower[u], t_lower[v]))
                visitable = False

                if t_lower[u] >= t_lower[v]:
                    if debug:
                        print('first case')
                    visitable = True

                if t_lower[u] == -1:
                    if debug:
                        print('second case')
                    visitable = True
                    t_lower[u] = t_lower[v]

                if visitable:
                    if debug:
                        print('add {} to queue'.format(u))
                    queue.append(u)
                    tree.append((v, u))
                    visited[u] = True
    if np.any(visited[obs_nodes] == 0):
        # some terminal is uncovered
        return None
    else:
        return remove_redundant_edges_from_tree(g, tree, r, obs_nodes)


def temporal_bfs_sync(g, r, infection_times, source, obs_nodes, debug=False):
    t_lower = np.ones(g.num_vertices(), dtype=np.int32) * -1  # hidden nodes has lower bound -1
    t_lower[obs_nodes] = infection_times[obs_nodes]
    t_lower[r] = infection_times[obs_nodes].min() - 1
    visited = np.zeros(g.num_vertices(), dtype=bool)
    tree = []

    obs_by_time = defaultdict(list)
    for o in obs_nodes:
        obs_by_time[infection_times[o]].append(o)
    obs_times = list(sorted(set(infection_times[obs_nodes])))

    success = True
    
    queue = [r]
    for cur_t in obs_times:
        banned_nodes = {v for v in obs_nodes if infection_times[v] != cur_t}
        target_nodes = [v for v in obs_nodes if infection_times[v] == cur_t]
        
        if debug:
            print('---- current time = {}'.format(cur_t))
            print('targets {}'.format(target_nodes))
        # cover nodes of level t
        while len(queue) > 0:
            if np.all(visited[target_nodes] == 1):
                if debug:
                    print('covered all targets')
                break
            v = queue.pop(0)
            for u in g.vertex(v).all_neighbours():
                u = int(u)
                if u not in banned_nodes and visited[u] == 0:
                    if debug and u in target_nodes:
                        print('cover target {}'.format(u))
                    if debug:
                        print('add edge {}'.format((v, u)))
                    if u in target_nodes:
                        if debug:
                            print('adding {} to baned list'.format(u))
                        banned_nodes.add(u)
                    else:
                        queue.append(u)
                    tree.append((v, u))
                    visited[u] = 1

        if np.all(visited[target_nodes] == 1):  # all targets covered
            if True:
                # remove redundant edges
                # construct the tree from used edges
                terminals = [o for o in obs_nodes if infection_times[o] <= cur_t]
                if debug:
                    print('terminals to cover: {}'.format(terminals))

                min_tree = remove_redundant_edges_from_tree(g, tree, r, terminals)
                if debug:
                    print('size of min tree: {}'.format(min_tree.num_edges()))
                tree = extract_edges(min_tree)
                if debug:
                    print('current tree edges {}'.format(tree))

                # update visited table
                visited.fill(0)
                covered_nodes = {u for nodes in tree for u in nodes}
                sorted_by_time = list(sorted(
                    covered_nodes,
                    key=lambda v: shortest_distance(min_tree, source=r, target=v),
                    reverse=False))
                if debug:
                    print('covered nodes: {}'.format(sorted_by_time))
                queue = []
                for v in sorted_by_time:
                    visited[v] = 1
                    queue.append(v)
                if debug:
                    print('current queue: {}'.format(queue))
            continue
        else:
            if debug:
                print('failed to cover targets')
            success = False
            break
    if success:
        return remove_redundant_edges_from_tree(g, tree, r, obs_nodes)
    else:
        return None
