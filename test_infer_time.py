from graph_tool import Graph, GraphView
from infer_time import fill_missing_time


def test_fill_missing_time():
    """simple chain graph test
    """
    g = Graph(directed=False)
    g.add_vertex(4)
    g.add_edge_list([(0, 1), (1, 2), (2, 3)])

    t = GraphView(g, directed=True)
    efilt = t.new_edge_property('bool')
    efilt.a = True
    efilt[t.edge(2, 3)] = False
    t.set_edge_filter(efilt)
    vfilt = t.new_vertex_property('bool')
    vfilt.a = True
    vfilt[3] = False
    t.set_vertex_filter(vfilt)
    
    root = 0
    obs_nodes = {0, 2}
    infection_times = [0, 1.5, 3, -1]
    
    pt = fill_missing_time(g, t, root, obs_nodes,
                           infection_times,
                           debug=False)

    for i in range(4):
        assert pt[i] == infection_times[i]
