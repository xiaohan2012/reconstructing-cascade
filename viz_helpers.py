import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

from collections import OrderedDict
from graph_tool.draw import graph_draw

from helpers import cascade_source, infected_nodes


SIZE_ZERO = 0
SIZE_SMALL = 10
SIZE_MEDIUM = 20
SIZE_LARGE = 30

COLOR_BLUE = (31/255, 120/255, 180/255, 1.0)
COLOR_RED = (1.0, 0, 0, 1.0)
COLOR_YELLOW = (255/255, 217/255, 47/255, 1.0)
COLOR_WHITE = (255/255, 255/255, 255/255, 1.0)
COLOR_ORANGE = (252/255, 120/255, 88/255, 1.0)
COLOR_PINK = (1.0, 20/255, 147/255, 1.0)
COLOR_GREEN = (50/255, 205/255, 50/255, 1.0)

SHAPE_CIRCLE = 'circle'
SHAPE_PENTAGON = 'pentagon'
SHAPE_HEXAGON = 'hexagon'
SHAPE_SQUARE = 'square'
SHAPE_TRIANGLE = 'triangle'
SHAPE_PENTAGON = 'pentagon'


def lattice_node_pos(g, shape):
    pos = g.new_vertex_property('vector<float>')
    for v in g.vertices():
        r, c = int(int(v) / shape[1]), int(v) % shape[1]
        pos[v] = np.array([r, c])
    return pos


def default_plot_setting(g, c, X,
                         size_multiplier=1.0, edge_width_multiplier=1.0,
                         deemphasize_hidden_infs=False):
    source = cascade_source(c)
    inf_nodes = infected_nodes(c)
    hidden_infs = set(inf_nodes) - set(X)

    node_color_info = OrderedDict()
    node_color_info[tuple(X)] = COLOR_BLUE
    if not deemphasize_hidden_infs:
        node_color_info[tuple(hidden_infs)] = COLOR_YELLOW
    node_color_info[(source, )] = COLOR_GREEN
    node_color_info['default'] = COLOR_WHITE

    node_shape_info = OrderedDict()
    node_shape_info[tuple(X)] = SHAPE_SQUARE
    node_shape_info['default'] = SHAPE_CIRCLE
    node_shape_info[(source, )] = SHAPE_PENTAGON

    node_size_info = OrderedDict()

    node_size_info[tuple(X)] = 15 * size_multiplier
    node_size_info[(source, )] = 20 * size_multiplier
    if not deemphasize_hidden_infs:
        node_size_info[tuple(hidden_infs)] = 12.5 * size_multiplier
    node_size_info['default'] = 5 * size_multiplier

    node_text_info = {'default': ''}
    
    edge_color_info = {
        'default': 'white'
    }
    edge_pen_width_info = {
        'default': 2.0 * edge_width_multiplier
    }
    return {
        'node_color_info': node_color_info,
        'node_shape_info': node_shape_info,
        'node_size_info': node_size_info,
        'edge_color_info': edge_color_info,
        'edge_pen_width_info': edge_pen_width_info,
        'node_text_info': node_text_info
    }


def visualize(g, pos,
              node_color_info={},
              node_shape_info={},
              node_size_info={},
              edge_color_info={},
              edge_pen_width_info={},
              node_text_info={},
              color_map=mpl.cm.Reds,
              ax=None,
              output=None):

    def populate_property(dtype, info, on_edge=False):
        if on_edge:
            prop = g.new_edge_property(dtype)
        else:
            prop = g.new_vertex_property(dtype)
            
        prop.set_value(info['default'])
        del info['default']
        
        for entries, v in info.items():
            if on_edge:
                for n in entries:
                    prop[g.edge(*n)] = v
            else:
                if dtype not in {'int', 'float'}:
                    for n in entries:
                        prop[n] = v
                else:
                    prop.a[list(entries)] = v

        return prop

    # vertex color is a bit special
    # can pass both ndarray and RGB
    # for ndarray, it converted to cm.Reds

    # vertex_fill_color.set_value(node_color_info['default'])
    if 'default' in node_color_info:
        del node_color_info['default']

    # colormap to convert to rgb
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    m = cm.ScalarMappable(norm=norm, cmap=color_map)

    if not isinstance(node_color_info, np.ndarray):
        assert isinstance(node_color_info, dict)
        vertex_fill_color = g.new_vertex_property('vector<float>')
        for entries, v in node_color_info.items():
            if isinstance(v, np.ndarray):
                assert len(entries) == len(v)
                for e, vv in zip(entries, v):
                    # convert to RGB
                    vertex_fill_color[e] = m.to_rgba(vv)
            else:
                for e in entries:
                    vertex_fill_color[e] = v
    else:
        vertex_fill_color = g.new_vertex_property('float')
        vertex_fill_color.a = node_color_info

    vertex_size = populate_property('int', node_size_info)
    vertex_shape = populate_property('string', node_shape_info)
    vertex_text = populate_property('string', node_text_info)
    
    edge_color = populate_property('string', edge_color_info, True)
    edge_pen_width = populate_property('float', edge_pen_width_info, True)
    
    graph_draw(g, pos=pos,
               vertex_fill_color=vertex_fill_color,
               vertex_size=vertex_size,
               vertex_shape=vertex_shape,
               edge_color=edge_color,
               edge_pen_width=edge_pen_width,
               vertex_text=vertex_text,
               mplfig=ax,
               vcmap=color_map,
               bg_color=[256, 256, 256, 256],
               output=output)
