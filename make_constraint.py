import sys
import networkx as nx
import csv
import random
import numpy as np
import os
import time
from itertools import combinations
import pandas as pd
from sscd import SSCD
from ssnmf import SSNMF
import scipy.sparse as ssp
from sklearn.metrics import normalized_mutual_info_score

def impose_constraint(correct_label, max_density=1.0):
    correct_label = list(correct_label)
    nodes_to_com = {}
    for n, c in enumerate(correct_label):
        nodes_to_com.setdefault(c,set())
        nodes_to_com[c].add(n)
    all_n_pairs = 0
    for nodes in nodes_to_com.values():
        all_n_pairs += len(nodes) * (len(nodes)-1) / 2
    n_pairs = int(all_n_pairs * max_density)
    count = 0
    nid_coms = list(enumerate(correct_label))
    combi = list(combinations(nid_coms, 2))
    random.shuffle(combi)
    const_pairs = []
    for nc1, nc2 in combi:
        n1, c1 = nc1
        n2, c2 = nc2
        if c1 == c2:
            const_pairs.append((n1,n2))
            count += 1
        if count > n_pairs: break
    return const_pairs, all_n_pairs


def export_const(path, c_pairs):
    writer = csv.writer(open(path, "w"))
    writer.writerows(c_pairs)

#data_label = "dolphins"
#data_label = "polbooks"
#data_label = "polblogs"
#data_label = "football"
#data_label = "dolphins"
#data_label = "karate"
#data_label = "gn_32_4"
#data_label = "LRF"
#data_label = "friendship"

def impose_constraint_by_degree(edge_list, correct_label, ratio):
    edge_list = np.array(edge_list)
    assert edge_list.min() == 0

    degrees = []
    for i in range(len(correct_label)):
        degrees.append(-(edge_list == i).sum())

    n_const = int(len(correct_label) * ratio)

    constrained_nodes = np.argsort(degrees)[:n_const]

    node_for_label = {}
    for node in constrained_nodes:
        label = correct_label[node]

        node_for_label.setdefault(label, [])

        node_for_label[label].append(node)

    const_pairs = []
    for label, nodes in node_for_label.items():
        for i, j in combinations(nodes, 2):
            const_pairs.append((i, j))

    return const_pairs

    #degrees_for_label = {}
    #for node in reversed(node_sorted_by_degree):
    #    label = correct_label[node]
    #    degrees_for_label.setdefault(label, [])
    #
    #    degrees_for_label[label].append(node)


def make_constraints(data_label):
    data_path = "data/%s_edge.pkl"%(data_label)
    label_path = "data/%s_label.pkl"%(data_label)

    edge_list = pd.read_pickle(data_path)
    correct_label = pd.read_pickle(label_path)

    #const_pairs, all_n_pairs = impose_constraint(correct_label, 0.5)
    #densities = list(range(15+1)) + [20,30]
    densities = [0, 0.05, 0.10, 0.15, 0.2, 0.25, 0.30]

    for d in densities:
        const_pairs = impose_constraint_by_degree(edge_list, correct_label, d)
        path = "data/const/degree_"+data_label+"_"+str(d)+".pkl"
        #c_pairs = const_pairs[:n_c]
        pd.to_pickle(const_pairs, path)


if __name__=="__main__":
    make_constraints(sys.argv[1])
