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


###########################################
density = 0
trials = 1
##########################################

os.system("rm -rf log")

"""Functions definitions"""
def make_matrix_from_elist(elist, nn):
    mat = ssp.lil_matrix((nn,nn))
    ind = np.array(elist)
    mat[ind[:,0],ind[:,1]] = 1
    return mat.toarray()

def expand_synmetric(elist):
    elist = elist + [(j, i) for i, j in elist]
    return elist

def calculate_loss(edge_list, H, mlambda, const):
    nn = H.shape[0]
    edge_list = expand_synmetric(edge_list)
    const = expand_synmetric(const)
    X = make_matrix_from_elist(edge_list, nn)
    O = make_matrix_from_elist(const, nn)
    D = np.diag(O.sum(axis=1))
    L = D - O
    lse = np.sum(np.power(X - np.matmul(H,H.T),2))
    reg = np.matmul(H.T, np.matmul(L, H)).trace()
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


dense_label = str(int(density))
data_label = "polbooks"
data_path = "data/%s_edge.pkl"%(data_label)
label_path = "data/%s_label.pkl"%(data_label)
const_path = "data/const_%s_%d.pkl"%(data_label, density)

edge_list = pd.read_pickle(data_path)
correct_label = pd.read_pickle(label_path)
mlambda = 1.0
const = pd.read_pickle(const_path)


abs_adam = SSCD(2, method="adam", positivate="abs",mlambda=mlambda,learning_rate=0.1)
abs_sgd = SSCD(2, method="sgd", positivate="abs", mlambda=mlambda,learning_rate=0.01)
clipped_adam = SSCD(2, method="adam", positivate="clip", mlambda=mlambda,learning_rate=0.1)
clipped_sgd = SSCD(2, method="sgd",positivate="clip", mlambda=mlambda,
                    learning_rate=0.01)
update_rule = SSNMF(2, mlambda=mlambda)
models = {"abs_adam":abs_adam, "clipped_adam":clipped_adam,
        "update_rule":update_rule, "abs_sgd":abs_sgd, "clipped_sgd":clipped_sgd}
#models = {"abs_adam":abs_adam}
#models = {"update_rule":update_rule}
times = {}
nmis = {}

for _ in range(trials):
    for name, model in models.items():
        print("******************************")
        print(name)
        times.setdefault(name, [])
        nmis.setdefault(name, [])
        start = time.time()
        H, cost_list = model.fit_and_transform(edge_list, const,
                                                    threshold=0.001,steps=2000)
        elapsed = time.time() - start
        loss = calculate_loss(edge_list, H, mlambda, const)
        nmi = calculate_nmi(H, correct_label)
        path = "cost/" + name + "_" + data_label + ".csv"
        export_cost(path, cost_list)
        print("Time: " + str(elapsed))
        print("Loss: " + str(loss))
        print("NMI: " + str(nmi))
        times[name].append(elapsed)
        nmis[name].append(nmi)
        import pdb; pdb.set_trace()

result_path = "result/result_"+data_label+"_" + dense_label+".csv"
with open(result_path, "w") as f:
    writer = csv.writer(f)
    header = ["model", "mean time", "std time", "mean nmi", "std nmi"]
    writer.writerow(header)
    for name in models.keys():
        mean_time = np.mean(times[name])
        std_time = np.std(times[name])
        mean_nmi = np.mean(nmis[name])
        std_nmi = np.std(nmis[name])
        result = [name, mean_time, std_time, mean_nmi, std_nmi]
        writer.writerow(result)


