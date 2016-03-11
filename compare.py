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

if __name__ == '__main__':
    os.system("rm -rf log")
    #edge_list = pd.read_pickle("data/karate.pkl")
    #correct = pd.read_pickle("data/karate_com.pkl")
    edge_list = pd.read_pickle("data/polbooks_edge.pkl")
    correct = pd.read_pickle("data/polbooks_label.pkl")
    const = [(1,2), (32,33)]
    K = 4
    
    #Gradient based method
    model = SSCD(K=K, learning_rate=0.1, mlambda=1.0, threads=4)
    start = time.time()
    H = model.fit_and_transform(edge_list, const, steps=100)
    constime = time.time() - start
    W = model.get_W()
    A = model.get_A()
    evaluate_label(A, H, W, correct, K)
    print("Time: " + str(constime))


    #Update rule method
    ssnmf = SSNMF(K, mlambda=1.0)
    start = time.time()
    H = ssnmf.fit_and_transform(edge_list, const, steps=20)
    W = ssnmf.get_W()
    print("Time: " + str(time.time() - start))
    evaluate_label(A, H, W, correct, K)

