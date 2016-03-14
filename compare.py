import csv
from sscd import SSCD
from ssnmf import SSNMF
import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
import tensorflow as tf
import pdb
import os
import sys
import time
from constrain import constrain_label


def evaluate_label(A, H, W, corr, K):
    label = H.argmax(axis=1)
    km = KMeans(K)
    label2 = km.fit_predict(H)
    nmi = normalized_mutual_info_score(label, corr)
    nmi2 = normalized_mutual_info_score(label2, corr)
    print("NMI by argmax: " + str(nmi))
    print("NMI by kmeans: " + str(nmi2))
    A = np.matrix(A)
    W = np.matrix(W)
    H = np.matrix(H)
    loss = np.power(A - W * H.T, 2).sum()
    print(loss)
    return nmi, nmi2, loss

if __name__ == '__main__':
    os.system("rm -rf log")
    edge_list = pd.read_pickle("data/karate.pkl")
    correct = pd.read_pickle("data/karate_com.pkl")
    #edge_list = pd.read_pickle("data/polbooks_edge.pkl")
    #correct = pd.read_pickle("data/polbooks_label.pkl")
    n_const = 15
    const = constrain_label(correct, edge_list, n_const)
    const += [(0,0)]
    K = 2
    trials = 100

    sum_nmi_gd = 0
    sum_nmi_arg_gd = 0
    nmi_arg_list_gd = []
    sum_loss_gd = 0
    sum_time_gd = 0

    sum_nmi_ur = 0
    sum_nmi_arg_ur = 0
    sum_loss_ur = 0
    sum_time_ur = 0
    nmi_arg_list_ur = []

    for i in range(trials):
        #Gradient based method
        model = SSCD(K=K, learning_rate=0.1, mlambda=1.0, threads=1)
        start = time.time()
        H = model.fit_and_transform(edge_list, const, steps=100)
        constime = time.time() - start
        W = model.get_W()
        A = model.get_A()
        nmi, nmi_arg, loss = evaluate_label(A, H, W, correct, K)
        print("Time: " + str(constime))
        sum_nmi_gd += nmi
        sum_nmi_arg_gd += nmi_arg
        sum_loss_gd += loss
        sum_time_gd += constime
        nmi_arg_list_gd.append(nmi_arg)

        #Update rule method
        ssnmf = SSNMF(K, mlambda=1.0)
        start = time.time()
        H = ssnmf.fit_and_transform(edge_list, const, steps=50)
        W = ssnmf.get_W()
        constime = time.time() - start
        evaluate_label(A, H, W, correct, K)
        nmi, nmi_arg, loss = evaluate_label(A, H, W, correct, K)
        print("Time: " + str(constime))
        sum_nmi_ur += nmi
        sum_nmi_arg_ur += nmi_arg
        sum_loss_ur += loss
        sum_time_ur += constime
        nmi_arg_list_ur.append(nmi_arg)

    print("Num of const nodes: " + str(n_const))
    
    mean_nmi_gd = sum_nmi_gd / trials
    mean_nmi_arg_gd = sum_nmi_arg_gd / trials
    mean_loss_gd = sum_loss_gd / trials
    mean_time_gd = sum_time_gd / trials
    std = np.std(nmi_arg_list_gd)
    print("Mean NMI: " + str(mean_nmi_gd))
    print("Mean NMI arg: " + str(mean_nmi_arg_gd))
    print("Std NMI arg: " + str(std))
    print("Mean loss: " + str(mean_loss_gd))
    print("Mean time: " + str(mean_time_gd))

    

    mean_nmi_ur = sum_nmi_ur / trials
    mean_nmi_arg_ur = sum_nmi_arg_ur / trials
    mean_loss_ur = sum_loss_ur / trials
    mean_time_ur = sum_time_ur / trials
    std = np.std(nmi_arg_list_ur)

    print("Mean NMI: " + str(mean_nmi_ur))
    print("Mean NMI arg: " + str(mean_nmi_arg_ur))
    print("Std NMI arg: " + str(std))
    print("Mean loss: " + str(mean_loss_ur))
    print("Mean time: " + str(mean_time_ur))

    result = {}
    result["MeanNMIKMGD"] = mean_nmi_gd
    result["MeanNMIArgGD"] = mean_nmi_arg_gd
    result["StdNMIGD"] = np.std(nmi_arg_list_gd)
    result["MeanLossGD"] = mean_loss_gd
    result["MeanTimeGD"] = mean_time_gd

    result["MeanNMIKMUR"] = mean_nmi_ur
    result["MeanNMIARGUR"] = mean_nmi_arg_ur
    result["StdNMIUR"] = np.std(nmi_arg_list_ur)
    result["MeanLossUR"] = mean_loss_ur
    result["MeanTimeUR"] = mean_time_ur

    with open("results/" + str(n_const) + ".csv", "w") as f:
        writer = csv.writer(f)
        for k, v in result.items():
            writer.writerow((k,v))


