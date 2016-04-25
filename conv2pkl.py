import pandas as pd
import sys
import os
import csv

def read_network(path):
    elist = []
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, j in reader:
            i = int(i) -1
            j = int(j) - 1
            elist.append((i,j))
    return elist

def read_community(path):
    coms = []
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for n, c in reader:
            coms.append(c)
    return coms

def conv2pkl(name):
    elist = read_network("network.dat")
    coms = read_community("community.dat")

    elist_path = os.path.join("binary_networks/data", name+"_edge.pkl")
    coms_path = os.path.join("binary_networks/data", name+"_label.pkl")

    pd.to_pickle(elist, elist_path)
    pd.to_pickle(coms, coms_path)

