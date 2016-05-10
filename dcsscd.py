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
from sscd import SSCD

import pdb

class DegreeCorrectedSSCD(SSCD):

    def prepare_calculation(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            K = self.K
            mlambda = self.mlambda
            n_nodes = self.n_nodes
            deg_vec = self.get_degree_vector()
            edge_list, weights = self.seperate_nodeid_and_weight(
                                                            self.edge_list)
            const_pairs, const_weights = self.seperate_nodeid_and_weight(
                                                            self.const_pairs)

            pdb.set_trace()
            self.A = A = tf.sparse_to_dense(output_shape=[n_nodes, n_nodes],
                                            sparse_indices=edge_list,
                                            sparse_values=weights)
            self.O = O = tf.sparse_to_dense(output_shape=[n_nodes, n_nodes],
                                            sparse_indices=const_pairs,
                                            sparse_values=const_weights)
            self.P = P = tf.constant(self.get_degree_matrix(O))
            self.L = L = P - O

            degrees = self.get_degree_vector()
            self.U = U = tf.Variable(self.get_initial_U(degrees, K),
                                     name="U")
            self.Z = Z = tf.Variable(self.get_initial_Z(degrees, K),
                                     name="Z")
            U_norm = self.normalize_U(U)
            Z_norm = self.get_positive_variable(Z)

            Y = tf.matmul(U_norm, tf.matmul(Z_norm, U_norm, transpose_b=True))
            self.loss = loss = tf.nn.l2_loss(A - Y)
            adam = tf.AdamOptimizer(self.lr)
            self.opt = adam.minimize(loss)
            self.setup_session()

    def get_degree_vector(self):
        n_nodes = np.array(self.edge_list).shape[0]
        degrees = [0] * n_nodes
        for i, j, w in self.edge_list:
            degrees[i] += w
            degrees[j] += w
        return np.array(degrees, dtype=np.float32)

    def normalize_U(self,U):
        if self.positivate == "abs":
            U_pos = tf.abs(U)
        elif self.positivate == "square":
            U_pos = tf.pow(U)
        else:
            raise AttributeError("positivate must be in 'abs' or 'square'")
        U_sum_col = self.diagonalize(tf.reduce_sum(U_pos, reduction_indices=0))
        U_norm = tf.matmul(U, U_sum_col)
        return U_norm

    @staticmethod
    def diagonalize(v):
        I = tf.diag(tf.ones_like(v))
        V = tf.expand_dims(v, 1)
        D = tf.matmul(I, V)
        return D

    def get_initial_U(self, degrees, K):
        n_nodes = degrees.shape[0]
        deg_root = np.expand_dims(np.sqrt(degrees),0)
        sum_deg_root = np.sum(deg_root)
        msr = deg_root / sum_deg_root
        U = np.repeat(msr, K, axis=0).T + np.random.normal(0,0.1,size=[n_nodes, K])
        if self.positivate == "square":
            U = np.sqrt(U)
        return U

    def get_initial_Z(self, degrees, K):
        sum_degrees = np.sum(degrees)
        mean_deg = sum_degrees / K
        Z = np.random.uniform(0, 2*mean_deg)
        if self.positivate == "square":
            Z = np.sqrt(Z)
        return Z

if __name__ == "__main__":
    elist = [(0,0), (0,1), (1,2)]
    cpairs = [(0,0), (0,1), (1,2)]
    model = DegreeCorrectedSSCD(2)
    model.fit_and_transform(elist)





