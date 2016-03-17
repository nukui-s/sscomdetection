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
    return const_pairs

def export_const(path, c_pairs):
    writer = csv.writer(open(path, "w"))
    writer.writerows(c_pairs)

#data_label = "polbooks"
data_label = "karate"
data_path = "data/%s_edge.pkl"%(data_label)
label_path = "data/%s_label.pkl"%(data_label)

edge_list = pd.read_pickle(data_path)
correct_label = pd.read_pickle(label_path)

const_pairs = impose_constraint(correct_label)
densities = range(10)


for d in densities:
    n_c = int(max(1, len(const_pairs)*d / 5))
    path = "data/const_"+data_label+"_"+str(d)+".pkl"
    c_pairs = const_pairs[:n_c]
    pd.to_pickle(c_pairs, path)

