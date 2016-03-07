import os
import sys
import time
from itertools import chain
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans

class SSCD(object):
    """Unified Semi-Supervised Community Detection"""

    def __init__(self, K, mlambda=1.0, learning_rate=0.01):
        self.K = K
        self.mlambda = mlambda
        self.lr = learning_rate

    def fit_and_transform(self, edge_list, const_pairs=None, weights=None,
                          const_weights=None, steps=100, log_dir="log"):
        self.n_nodes = n_nodes = max(chain.from_iterable(edge_list)) + 1
        self.n_edges = n_edges = len(edge_list)
        if weights is None:
            weights = np.ones(n_edges).astype(np.float32)
        edge_list = [(e[0],e[1],w) for e, w in zip(edge_list, weights)]
        edge_list = [(j, i, w) for i, j, w in edge_list]
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
        for s in range(steps):
            cost, _ = self.sess.run([self.cost, self.opt])
            

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


            initializer = tf.truncated_normal_initializer(mean=0.0,
                                                          stddev=1.0)
            self.W_root = W_root = tf.get_variable("W_root", shape=[n_nodes, K],
                                                   initializer=initializer)
            self.H_root = H_root = tf.get_variable("H_root", shape=[n_nodes, K],
                                                   initializer=initializer)
            self.W = W = tf.mul(W_root, W_root, name="W")
            self.H = H = tf.mul(H_root, H_root, name="H")

            self.loss = loss = self.loss_LSE(A, W, H)
            self.reg_term = reg_term = self.regulation_term(H, L)

            self.cost = cost = loss + mlambda * reg_term

            tf.scalar_summary("loss", loss)
            tf.scalar_summary("reg_term", reg_term)
            tf.scalar_summary("cost", cost)

            self.summary = tf.merge_all_summaries()

            self.opt = tf.train.AdamOptimizer(self.lr).minimize(cost)

            self.sess = tf.Session()
            self.init_op = tf.initialize_all_variables()

    def get_latent_vectors(self):
        return self.sess.run(self.H)

    @classmethod
    def loss_LSE(cls, A, W, H):
        WH = tf.matmul(W, H, transpose_b=True, name="WH")
        loss = tf.nn.l2_loss(A - WH, name="LSE")
        return loss

    @classmethod
    def regulation_term(cls, H, L):
        reg_term = tf.matmul(tf.matmul(H, L, transpose_a=True), H)
        return reg_term

if __name__ == '__main__':
    os.system("rm -rf log")
    edge_list = pd.read_pickle("data/karate.pkl")
    correct = pd.read_pickle("data/karate_com.pkl")
    const = [(6,7), (30,31)]

    model = SSCD(2)

    model.fit_and_transform(edge_list, const)
    km = KMeans(2)
    H = model.get_latent_vectors()
