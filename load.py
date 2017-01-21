import pandas as pd
import csv
import os
import numpy as np
from itertools import chain
import igraph
import networkx as nx

def load_gml(path, attr_name="value"):
    g = igraph.read(path)
    elist = g.get_edgelist()
    min_id = min(chain.from_iterable(elist))
    if min_id == 1:
        offset = True
    else:
        offset = False
    if offset:
        elist = [(i-1, j-1) for i, j in elist]
    try:
        labels = [int(v) for v in g.vs.get_attribute_values(attr_name)]
    except:
        labels = [v for v in g.vs.get_attribute_values(attr_name)]
    return elist, labels

def load_graphml(path, attr_name="ground truth"):
    g = nx.read_graphml(path)
    elist_ = g.edges()
    elist = [(int(i), int(j)) for i, j in elist_]
    label_dict = nx.get_node_attributes(g, attr_name)
    n_nodes = len(label_dict)
    labels = [label_dict[str(n)] for n in range(n_nodes)]
    return elist, labels

def load_friendship(link_path, label_path):
    reader = csv.reader(open(link_path), delimiter=" ")
    reader2 = csv.reader(open(label_path), delimiter="\t")
    sid_dict = {}
    labels = []
    edge_list = []
    for sid, com, _ in reader2:
        sid_dict[sid] = len(sid_dict)
        labels.append(com)
    for i, j in reader:
        i2 = sid_dict[i]
        j2 = sid_dict[j]
        edge_list.append((i2,j2))
    return edge_list, labels


def load_dblp(link_path):
    reader = csv.reader(open(link_path), delimiter="\t")
    edge_list = []
    for i, j in reader:
        edge_list.append((int(i),int(j)))
    return edge_list


def save_elist(data_name, elist):
    path = os.path.join("data",data_name+"_edge.pkl")
    pd.to_pickle(elist, path)

def save_labels(data_name, labels):
    path = os.path.join("data", data_name+"_label.pkl")
    pd.to_pickle(labels, path)

if __name__ == '__main__':
    data_name = "football"
    gml_file = "data/football.gml"
    #edge_file = "data/dolphins_edge.csv"
    #edge_file = "data/dolphins_edge.txt"

    #elist, labels = load_graphml(gml_file, attr_name="ground truth")

    elist, labels = load_gml(gml_file)

    save_elist(data_name, elist)
    save_labels(data_name, labels)
    #elist, labels = load_friendship(edge_file, label_file)
    #elist = load_dblp(edge_file)

    #save_elist(data_name, elist)

    #save_elist(data_name, elist)
    #save_labels(data_name, labels)
