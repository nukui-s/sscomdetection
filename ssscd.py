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


class SynmetricSSCD(SSCD):

    def prepare_calculation(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            K = self.K
            n_nodes = self.n_nodes
            edge_list, weights = self.seperate_nodeid_and_weight(
                                                            self.edge_list)
            const_pairs, const_weights = self.seperate_nodeid_and_weight(
                                                            self.const_pairs)
            mlambda = self.mlambda

            self.A = A = tf.sparse_to_dense(output_shape=[n_nodes, n_nodes],
                                            sparse_indices=edge_list,
                                            sparse_values=weights)
            self.O = O = tf.sparse_to_dense(output_shape=[n_nodes, n_nodes],
                                            sparse_indices=const_pairs,
                                            sparse_values=const_weights)

            self.D = D = self.get_degree_matrix(O)
            self.L = L = D - O
            scaler = 2 * np.sqrt(weights.sum() / (n_nodes * n_nodes * K))
            initializer = tf.random_uniform_initializer(maxval=scaler)
            self.H_var = H_var = tf.get_variable("H_var", shape=[n_nodes, K],
                                                  initializer=initializer)
            self.W_var = W_var = tf.get_variable("W_var", shape=[n_nodes, K],
                                                 initializer=initializer,
                                                 trainable=(not self.synmetric))

            #Positivate H
            self.H = H = self.get_positive_variable(H_var)
            self.W = H

            H_norm = self.normalize_H(H, n_nodes)

            self.loss = loss = self.loss_LSE(A, H)
            self.sup_term = sup_term = self.supervisor_term(H_norm, L)

            self.cost = cost = loss + mlambda * sup_term

            self.define_tfsummary()

            if self.optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(self.lr, epsilon=0.1)
            else:
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            opt = optimizer.minimize(cost)
            if self.positivate != "clip":
                self.opt = opt
            else:
                with tf.control_dependencies([opt]):
                    clipped = tf.maximum(H_var,0)
                    clip_H = H_var.assign(clipped)
                self.opt = tf.group(opt, clip_H)

            config = tf.ConfigProto(inter_op_parallelism_threads=self.threads,
                                  intra_op_parallelism_threads=self.threads)
            self.sess = tf.Session(config=config)
            self.init_op = tf.global_variables_initializer()


    @classmethod
    def loss_LSE(cls, A, H):
        HH = tf.matmul(H, H, transpose_b=True, name="HH")
        loss = tf.nn.l2_loss(A - HH, name="LSE")
        return loss

    @classmethod
    def supervisor_term(cls, H, L):
        HLH = tf.matmul(H, tf.matmul(L, H), transpose_a=True)
        mask = tf.diag(tf.ones([H.get_shape()[1]]))
        sup_term = tf.reduce_sum(HLH * mask)
        return sup_term

    @classmethod
    def normalize_H(cls, H, n_nodes):
        #Normalize H along to row
        I = tf.diag(tf.ones([n_nodes]))
        H_diag_inv = tf.matmul(I,
                                tf.expand_dims(
                                    1/tf.reduce_sum(H, reduction_indices=1),
                                    1))
        H_norm = H_diag_inv * H
        return H_norm


class KLSynmetricSSCD(SynmetricSSCD):

    @classmethod
    def supervisor_term(cls, H, L):
        loss = L * tf.matmul(H, tf.log(H + 1e-8), transpose_b=True)
        return tf.reduce_sum(loss)


class L2SynmetricSSCD(SynmetricSSCD):

    @classmethod
    def normalize_H(cls, H, n_nodes):
        # normalize H along row
        return tf.nn.l2_normalize(H, dim=1)
