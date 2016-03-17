import networkx as nx
import numpy as np
import os
import time
import pandas as pd
from sscd import SSCD
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


n_nodes = [100,1000]
mlambda = 1.0
abs_adam = SSCD(2, method="abs_adam", mlambda=mlambda,learning_rate=0.1)
abs_sgd = SSCD(2, optimizer="sgd",method="abs_adam", mlambda=mlambda,learning_rate=0.1)
clipped_adam = SSCD(2, method="clipped_adam", mlambda=mlambda,learning_rate=0.1)
clipped_sgd = SSCD(2, optimizer="sgd",method="clipped_adam", mlambda=mlambda,learning_rate=0.1)
update_rule = SSNMF(2, mlambda=mlambda)
models = {"abs_adam":abs_adam, "clipped_adam":clipped_adam,
        "update_rule":update_rule, "abs_sgd":abs_sgd, "clipped_sgd":clipped_sgd}

abs_adam_times = []
clipepd_adam_times = []
update_rule_times = []

const = [(0,1)]

#models = {"abs_adam":abs_adam}
for nn in n_nodes:
    edge_list = generate_synthetic_graph(nn)
    for name, model in models.items():
        print("******************************")
        print(name)
        start = time.time()
        H, W, cost_list = abs_adam.fit_and_transform(edge_list, const, threshold=0.001,steps=1000)
        elapsed = time.time() - start
        loss = calculate_loss(edge_list, W, H, mlambda, const)
        path = "cost/" + name + "_" + str(nn) + ".csv"
        export_cost(path, cost_list)
        print("Time: " + str(elapsed))
        print("Loss: " + str(loss))

        
