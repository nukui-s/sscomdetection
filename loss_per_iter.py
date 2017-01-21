import sys
import networkx as nx
import csv
import random
import numpy as np
import numpy
import pandas
import os
import time
from itertools import combinations
import pandas as pd
from sscd import SSCD
from ssnmf import SSNMF
import scipy.sparse as ssp
from sklearn.metrics import normalized_mutual_info_score
from ssscd import SynmetricSSCD, KLSynmetricSSCD
from settings import *


np.random.seed(42)


"""Functions definitions"""
def make_matrix_from_elist(elist, nn):
    mat = ssp.lil_matrix((nn,nn))
    ind = np.array(elist)
    mat[ind[:,0],ind[:,1]] = 1
    return mat.toarray()

def expand_synmetric(elist):
    elist = elist + [(j, i) for i, j in elist]
    return elist

def calculate_loss(edge_list, W, H, mlambda, const):
    nn = H.shape[0]
    edge_list = expand_synmetric(edge_list)
    const = expand_synmetric(const)
    X = make_matrix_from_elist(edge_list, nn)
    O = make_matrix_from_elist(const, nn)
    D = np.diag(O.sum(axis=1))
    L = D - O
    H = np.mat(H)
    W = np.mat(W)
    L = np.mat(L)
    lse = np.sum(np.power(X - H*H.T,2))
    reg_H = (H.T * L * H).trace()
    reg_W = (W.T * L * W).trace()
    reg = reg_W + reg_H
    loss = lse + mlambda * reg
    return loss

def calculate_nmi(H, correct_label):
    com = H.argmax(axis=1)
    nmi = normalized_mutual_info_score(com, correct_label)
    return nmi

def export_cost(path, cost_list):
    with open(path, "w") as f:
        for c in cost_list:
            f.write(str(c) + "\n")

if __name__=="__main__":
    os.system("rm -rf log")
    for density in densities:
        dense_label = str(int(density))
        data_path = "data/%s_edge.pkl"%(data_label)
        label_path = "data/%s_label.pkl"%(data_label)
        const_path = "data/const/degree_{}_{:.1f}.pkl".format(data_label, density)

        edge_list = pd.read_pickle(data_path)
        correct_label = pd.read_pickle(label_path)
        const = pd.read_pickle(const_path)

        if len(const) == 0:
            const = None

        abs_adam = SynmetricSSCD(K, method="adam", positivate="abs", mlambda=mlambda, learning_rate=lr_adam, threads=threads)
        kl_adam = KLSynmetricSSCD(K, method="adam", positivate="abs", mlambda=mlambda, learning_rate=lr_adam, threads=threads)
        clip_adam = SynmetricSSCD(K, method="adam", positivate="clip", mlambda=mlambda, learning_rate=lr_adam, threads=threads)
        abs_sgd = SynmetricSSCD(K, method="sgd", positivate="abs", mlambda=mlambda, learning_rate=lr_sgd, threads=threads)
        clip_sgd = SynmetricSSCD(K, method="sgd", positivate="clip", mlambda=mlambda, learning_rate=lr_sgd, threads=threads)
        #abs_adam = SSCD(K, method="adam", positivate="abs", mlambda=mlambda, learning_rate=lr_adam, threads=threads)
        #clip_adam = SSCD(K, method="adam", positivate="clip", mlambda=mlambda, learning_rate=lr_adam, threads=threads)
        #abs_sgd = SSCD(K, method="sgd", positivate="abs", mlambda=mlambda, learning_rate=lr_sgd, threads=threads)
        #clip_sgd = SSCD(K, method="sgd", positivate="clip", mlambda=mlambda, learning_rate=lr_sgd, threads=threads)
        update_rule = SSCD(K, mlambda=mlambda, method="mult", threads=threads)
        all_models = {"update_rule":update_rule, "abs_adam": abs_adam, "abs_sgd": abs_sgd,
                    "clip_adam":clip_adam, "clip_sgd":clip_sgd, "kl_adam": kl_adam}
        #models = {"update_rule":update_rule, "abs_adam": abs_adam, "clip_adam":clip_adam}
        #models = {"abs_adam": abs_adam, "update_rule":update_rule}
        models = {name: all_models[name] for name in used_models}
        costs = {}
        nmis = {}

        for name, model in models.items():
            print("******************************")
            print(name)

            start = time.time()

            W, H, best_cost, cost_list, H_list = model.fit_and_transform(edge_list, const,
                    threshold=threshold, steps=max_iters)
            nmi_list = [calculate_nmi(H, correct_label) for H in H_list]

            costs[name] = cost_list
            nmis[name] = nmi_list

        #pd.to_pickle(costs, "costs_per_iter.pkl")
