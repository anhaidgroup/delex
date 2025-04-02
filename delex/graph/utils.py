import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from .algorithms import find_all_nodes
import numpy as np
from .node import Node
from typing import Iterable
import re
import io

def _dfs_longest_path(G, node, path_lens, depth):
    path_lens[node] = max(path_lens.get(node, 0), depth)
    for n in G.pred[node]:
        _dfs_longest_path(G, n, path_lens, depth + 1)

def _single_source_longest_paths(G):
    path_lens = {}
    sinks = [node for node in G.nodes if G.out_degree(node) == 0]
    if len(sinks) != 1:
        raise ValueError('multiple sinks')
    _dfs_longest_path(G, sinks[0], path_lens, 0)
    return path_lens

#def _create_graph_pos(G):
#    #
#    #return nx.spring_layout(G)
#    #return nx.kamada_kawai_layout(G)
#
#    return nx.nx_agraph.graphviz_layout(G, prog='dot')
#    #return {n : p[::-1]  for n,p in nx.kamada_kawai_layout(G).items()}

def _create_graph_pos(G):

    path_lens = _single_source_longest_paths(G)
    column_counts = {v : 0 for v in set(path_lens.values())}
    pos = {}
    for node, path_len in path_lens.items():
        # reflect over the y axis so that source nodes are on the left
        pos[node] = np.array([-path_len, column_counts[path_len]])
        column_counts[path_len] += 1

    return pos


_seps = re.compile(r'[(].+[)]')
def _to_node_label(n):
    return _seps.sub(lambda x : '\n' + x.group(0) + '\n', str(n))

def _nodes_to_networkx(nodes):
    if isinstance(nodes, Node):
        nodes = find_all_nodes(nodes)
    G = nx.DiGraph()
    node_to_label = {n : f'n_{i}\n{_to_node_label(n)}' for i,n in enumerate(nodes)}
    for n in nodes:
        G.add_node(node_to_label[n])

    for n in nodes:
        for o in n._out_edges:
            G.add_edge(node_to_label[n], node_to_label[o])

    return G

def _networkx_to_dot_str(G):
    buffer = io.StringIO()
    nx.drawing.nx_pydot.write_dot(G, buffer)
    buffer.seek(0)
    return buffer.read()

def nodes_to_dot(nodes : Node | Iterable[Node]) -> str:
    """
    convert a graph to a dot string representation
    
    Parameters
    ----------
    nodes : Node | Iterable[Node]
        the nodes to be converted to a string

    Returns
    -------
    str
        a dot graph string of nodes
    """
    G = _nodes_to_networkx(nodes)
    return _networkx_to_dot_str(G)

def _print_debug_graph(nodes, output_file):
    if isinstance(nodes, Node):
        nodes = find_all_nodes(nodes)
    G = _nodes_to_networkx(nodes)
    pos = _create_graph_pos(G)
    
    max_depth = max(abs(x[0]) for x in pos.values())
    max_height = max(abs(x[1]) for x in pos.values())
    #scale = np.array([2, 1.0])
    scale = np.array([max_depth * 2.5, max_height])
    fig, ax = plt.subplots(figsize=5*scale)
    ax.margins(.5)
    options = {
        "font_size": 10,
        "font_weight" : 'bold',
        "node_size": 1,
        "node_shape" : 'o',
        "node_color": "white",
        "edgecolors": "black",
        #"linewidths": 5,
        #"alpha" : .5,
        #"width": 5,
    }
    nx.draw(G, ax=ax, pos=pos, with_labels=True, **options)
    arcs = {}
    for c in ax.get_children():
        if isinstance(c, mpl.text.Text):
            text = c.get_text()
            kwargs = {'facecolor' : 'white', 'alpha' : .75}
            if text in G.nodes and len(G.pred[text]) == 0:
                kwargs = {'facecolor' : 'tab:blue', 'alpha' : .5}
            
            c.set_bbox(kwargs)
        elif isinstance(c, mpl.patches.FancyArrowPatch):
            # where the arrow head is
            #k =  c._posA_posB[1]
            
            k = (c._posA_posB[0][1], c._posA_posB[1])
            arc = arcs.get(k, 0.1)
            arcs[k] = arc + .1
            if c._posA_posB[0][1] < c._posA_posB[1][1]:
                arc = -arc
            c.set_connectionstyle('arc3', rad=arc)

            #breakpoint()
    fig.savefig(output_file)

