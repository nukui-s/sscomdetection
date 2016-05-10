import networkx as nx
import numpy as np
import os
import time
import pandas as pd
from sscd_s import SSCDS
from ssnmf import SSNMF
import scipy.sparse as ssp


"""Functions definitions"""
def generate_synthetic_graph(n_node_par_com, n_com=2, p_in=0.1, p_out=0.001):
    sizes = [n_node_par_com] * n_com
    g = nx.random_partition_graph(sizes, p_in, p_out)
    edges = g.edges()
    return edges

def make_matrix_from_elist(elist, nn):
    mat = ssp.lil_matrix((nn,nn))
    ind = np.array(elist)
    mat[ind[:,0],ind[:,1]] = 1
    return mat.toarray()

def calculate_loss(edge_list, W, H, mlambda, const):
    nn = W.shape[0]
    X = make_matrix_from_elist(edge_list, nn)
    O = make_matrix_from_elist(const, nn)
    lse = np.sum(np.power(X - np.matmul(W,H.T),2))
    reg = np.matmul(H.T, np.matmul(O, H)).trace()
    loss = lse + mlambda * reg
    return loss

def export_cost(path, cost_list):
    with open(path, "w") as f:
        for c in cost_list:
            f.write(str(c) + "\n")

mlambda = 1.0
threads = 8
lr_adam = 0.1
K = 10
abs_adam = SSCDS(K, method="adam", positivate="abs", mlambda=mlambda, learning_rate=lr_adam, threads=threads)

model = abs_adam

start =- time.time()
dblp = pd.read_pickle("data/dblp_edge.pkl")
const = [(1,2)]
W, H, mean_cost, cost_list = model.fit_and_transform(dblp, const)
print("Takes " + str(time.time()-start) + "[s]")


