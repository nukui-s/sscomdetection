import numpy as np
import scipy.sparse as ssp
import tensorflow as tf
from Update import UpdateElem
import pandas as pd

from itertools import chain
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

class SSNMF(object):
    """A wrapper class for UpdateElem"""

    def __init__(self, K, mlambda=1.0):
        self.K = K
        self.mlambda = mlambda

    def fit_and_transform(self, edge_list, const_pairs=None, weights=None,
                          const_weights=None, steps=2000, log_dir="log",
                          threshold=0.001):
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
        self.A = A = self.convert_edge_list_into_dense(edge_list, n_nodes)
        self.O = O = self.convert_edge_list_into_dense(const_pairs, n_nodes)
        D = np.diag(O.sum(axis=1))
        self.updater = updater = UpdateElem()
        graph = tf.Graph()
        with graph.as_default():
            updater.add_semi_supervised_rule(A, O, D, self.K, self.mlambda)
            init_op = tf.initialize_all_variables()
            self.sess = tf.Session()
            self.sess.run(init_op)
        pre_cost = -1
        cost_list = []
        for s in range(steps):
            self.sess.run(updater.assign_W_node())
            self.sess.run(updater.assign_H_node())
            cost = self.sess.run(updater.cost)
            if abs(cost - pre_cost) < threshold:
                break
            pre_cost = cost
            cost_list.append(cost)
            print(cost)
        print("Steps: " + str(s+1))
        H = self.get_H()
        W = self.get_W()
        return H, cost_list

    @classmethod
    def convert_edge_list_into_dense(cls, edge_list, n_nodes):
        """Construct a adjacency matrix A from a list of (i, j, w)"""
        ind1 = [i for i, j, w in edge_list]
        ind2 = [j for i, j, w in edge_list]
        values = [w for i, j, w in edge_list]
        A = ssp.lil_matrix((n_nodes, n_nodes))
        A[ind1, ind2] = values
        return A.toarray()

    def get_W(self):
        return np.array(self.sess.run(self.updater._W))

    def get_H(self):
        return np.array(self.sess.run(self.updater._H))

    def get_A(self):
        return np.array(self.A)

if __name__ == '__main__':
    edge_list = pd.read_pickle("data/karate.pkl")
    correct = pd.read_pickle("data/karate_com.pkl")
    const = [(6,7), (30,31)]

    model = SSNMF(2)

    H = model.fit_and_transform(edge_list, const, steps=20)
    W = np.matrix(model.get_W())
    H = np.matrix(H)
    A = model.get_A()
    loss = np.power(A - W * H.T, 2).sum()
    km = KMeans(2)
    H = model.get_H()
    labels = km.fit_predict(H)
    labels2 = H.argmax(axis=1)
