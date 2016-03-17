import os
import sys
import time
from itertools import chain
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import pdb


class SSCD(object):
    """Unified Semi-Supervised Community Detection"""

    def __init__(self, K, mlambda=1.0, method="adam",learning_rate=0.01, 
                threads=8, positivate="abs"):
        self.K = K
        self.mlambda = mlambda
        self.lr = learning_rate
        self.threads = threads
        self.method = method
        self.positivate = positivate
        if method == "sgd":
            self.optimizer = tf.train.GradientDescentOptimizer
        elif method == "adam":
            self.optimizer = tf.train.AdamOptimizer
        else:
            self.optimizer = None

    def fit_and_transform(self, edge_list, const_pairs=None, weights=None,
                          const_weights=None, steps=2000, log_dir="log",
                          threshold=0.001):
        edge_list = list(set(edge_list))
        self.n_nodes = n_nodes = max(chain.from_iterable(edge_list)) + 1
        self.n_edges = n_edges = len(edge_list)
        if weights is None:
            weights = np.ones(n_edges).astype(np.float32)
        edge_list = [(e[0],e[1],w) for e, w in zip(edge_list, weights)]
        edge_list = edge_list + [(j, i, w) for i, j, w in edge_list]
        self.edge_list = sorted(edge_list, key=lambda x: (x[0], x[1]))
        if const_pairs is None:
            const_pairs = []
        if const_weights is None:
            const_weights = np.ones(len(const_pairs)).astype(np.float32)
        const_pairs = [(c[0], c[1], w) for c, w
                            in zip(const_pairs, const_weights)]
        const_pairs = const_pairs + [(j, i, w) for i, j, w in const_pairs]
        self.const_pairs = sorted(const_pairs, key=lambda x: (x[0], x[1]))
        self.prepare_calculation()
        self.sess.run(self.init_op)
        self.writer = tf.train.SummaryWriter(log_dir, self.sess.graph_def)
        pre_cost = -1
        cost_list = []
        for s in range(steps):
            cost, sm, _ = self.sess.run([self.cost, self.summary, self.opt])
            self.writer.add_summary(sm, s)
            if abs(cost - pre_cost) < threshold:
                break
            pre_cost = cost
            cost_list.append(cost)
        print("Steps: " + str(s + 1))
        H = self.get_H()
        self.best_cost = cost
        print("Best Cost: " + str(cost))
        return H, cost_list

    def prepare_calculation(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            K = self.K
            n_nodes = self.n_nodes
            edge_list = [(i,j) for i,j,w in self.edge_list]
            weights = np.array([w for i,j,w in self.edge_list],
                               dtype=np.float32)
            const_pairs = [(i,j) for i,j,w in self.const_pairs]
            const_weights = np.array([w for i,j,w in self.const_pairs],
                                      dtype=np.float32)
            mlambda = self.mlambda

            self.A = A = tf.sparse_to_dense(output_shape=[n_nodes, n_nodes],
                                            sparse_indices=edge_list,
                                            sparse_values=weights)
            self.O = O = tf.sparse_to_dense(output_shape=[n_nodes, n_nodes],
                                            sparse_indices=const_pairs,
                                            sparse_values=const_weights)
            self.D = D = tf.diag(tf.reduce_sum(O, reduction_indices=1))
            self.L = L = D - O
            scaler = np.sqrt(weights.sum())
            initializer = tf.random_uniform_initializer(maxval=1/scaler)
            self.H_var = H_var = tf.get_variable("H_var", shape=[n_nodes, K],
                                                  initializer=initializer)
            self.H = H = self.get_prosessed_H(H_var)
            self.W = W = H
            self.loss = loss = self.loss_LSE(A, W, H)
            self.sup_term = sup_term = self.supervisor_term(H, L)

            self.cost = cost = loss + mlambda * sup_term

            tf.scalar_summary("loss", loss)
            tf.scalar_summary("sup_term", sup_term)
            tf.scalar_summary("cost", cost)
            self.summary = tf.merge_all_summaries()
            if self.method == "mult":
                self.prepare_multiplicative(H_var, A, O, D)
            else:
                self.prepare_gradient(H_var, cost)

            config = tf.ConfigProto(inter_op_parallelism_threads=self.threads,
                                  intra_op_parallelism_threads=self.threads)
            self.sess = tf.Session(config=config)
            self.init_op = tf.initialize_all_variables()

    def prepare_gradient(self, H_var, cost):
        opt = self.optimizer(self.lr).minimize(cost)
        if self.positivate != "clip":
            self.opt = opt
            return
        with tf.control_dependencies([opt]):
            clipped = tf.maximum(H_var,0)
            clip_H = H_var.assign(clipped)
        self.opt = tf.group(opt, clip_H)

    def prepare_multiplicative(self, H_var, A, O, D):
        H_tmp = tf.get_variable("H_tmp",
                            initializer=tf.zeros_initializer(H_var.get_shape()))
        self.H = H = H_var
        lmd = self.mlambda
        AH = tf.matmul(A, H, a_is_sparse=True)
        OH = tf.matmul(O, H, a_is_sparse=True)
        HHH = tf.matmul(H, tf.matmul(H, H, transpose_a=True))
        DH = tf.matmul(D, H, a_is_sparse=True)
        H_new = H * tf.div(AH + lmd*OH, HHH + lmd*DH + 10e-9)
        save_H = H_tmp.assign(H_new)
        with tf.control_dependencies([save_H]):
            update_H = H_var.assign(H_tmp)
        self.opt = tf.group(save_H, update_H)

    def get_prosessed_H(self, H_var):
        if self.positivate == "abs":
            H = tf.abs(H_var, "H")
        elif self.positivate == "square":
            H = tf.pow(H_var, 2, "H")
        else:
            H = H_var
        return H

    def get_latent_vectors(self):
        return self.sess.run(self.H)

    def get_H(self):
        return self.sess.run(self.H)

    def get_W(self):
        return self.sess.run(self.W)

    def get_A(self):
        return self.sess.run(self.A)

    def get_O(self):
        return self.sess.run(self.O)

    def get_L(self):
        return self.sess.run(self.L)

    @classmethod
    def loss_LSE(cls, A, W, H):
        WH = tf.matmul(W, H, transpose_b=True, name="WH")
        loss = tf.nn.l2_loss(A - WH, name="LSE")
        return loss

    @classmethod
    def supervisor_term(cls, H, L):
        HLH = tf.matmul(tf.matmul(H, L, transpose_a=True, b_is_sparse=True), H)
        mask = tf.diag(tf.ones([H.get_shape()[1]]))
        sup_term = tf.reduce_sum(HLH * mask)
        return sup_term

if __name__ == '__main__':
    os.system("rm -rf log")
    edge_list = pd.read_pickle("data/karate.pkl")
    correct = pd.read_pickle("data/karate_com.pkl")
    const = [(1,2),(2,3),(32,33),(19,21)]

    model = SSCD(K=2, learning_rate=1.0)
    start = time.time()
    H = model.fit_and_transform(edge_list, const, steps=100)
    constime = time.time() - start
    W = np.matrix(model.get_W())
    H = np.matrix(H)
    A = model.get_A()
    loss = np.power(A - W * H.T, 2).sum()
    print(loss)

    km = KMeans(2)
    H = model.get_latent_vectors()
    labels = km.fit_predict(H)
    labels2 = H.argmax(axis=1)
    
    nmi = normalized_mutual_info_score(labels, correct)
    nmi2 = normalized_mutual_info_score(labels2, correct)

    print("NMI by KMeans: " + str(nmi))
    print("NMI by argmax: " + str(nmi2))
    print("Time: " + str(constime))
