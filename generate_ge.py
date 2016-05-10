import igraph
import os
import numpy as np
import pandas as pd
from visualize import plot_degree_dist

def generate_gn_graph(n_node, n_com, p_in, p_out):
    labels = np.zeros(n_node).astype(int)
    slice_ = n_node // n_com
    for c, i in enumerate(range(n_com)):
        start = i*slice_
        stop = (i+1) * slice_
        labels[start:stop] = c
    labels = list(labels)
    b_sizes = [labels.count(c) for c in range(n_com)]
    pref = np.zeros(shape=[n_com, n_com]) + p_out
    for c in range(n_com):
        pref[c,c] = p_in
    pref = pref.tolist()
    graph = igraph.GraphBase.SBM(n_node, pref, b_sizes)
    elist = graph.get_edgelist()
    return elist, labels


if __name__=='__main__':
    n = 1000
    c = 4
    p_in = 0.01
    p_out = 0.001

    name = "gn_%s_%s"%(str(n),str(c))
    elist_path = os.path.join("data/", name + "_edge.pkl")
    label_path = os.path.join("data/", name + "_label.pkl")
    elist, labels = generate_gn_graph(n,c,p_in,p_out)
    
    pd.to_pickle(elist, elist_path)
    pd.to_pickle(labels, label_path)
    print(len(elist))

    plot_degree_dist(elist)

