# Author : Hoang NT
# Date : 2016-03-07
#
# Update module to handle fractorized matrix elements update.

import tensorflow as tf
import numpy as np

# A warper of a collection of update functions
# These functions will add the element update
# computation node to a graph passed to it.
class UpdateElem(object) :

  # Create object with a tensor flow graph
  def __init__(self) :
    self._A = None
    self._W = None
    self._H = None
    self._O = None
    self._D = None
    self._semi_supervised = None
    self._lambda = None
    self._updateW = None
    self._updateH = None
    self._assignW = None
    self._assignH = None

  # Add semi supervised updating rules to the graph
  # Input: 3 numpy.ndarray(s), 1 integer, 1 float
  ## A : Adjacency matrix - np.ndarray
  ## O, D : Matrix encoding the prior info and its diagonal matrix D - np.ndarray
  ## k : Number of latent communities, shape of W is N x k, H is N x k
  # Output: Tuple containing W and H (not transposed)
  def add_semi_supervised_rule(self, A, O, D, k, l) :
    # Check for data type
    assert type(A) is np.ndarray
    assert type(O) is np.ndarray
    assert type(D) is np.ndarray
    assert type(k) is int
    assert type(l) is float
    # ... and shape
    assert O.shape == D.shape == A.shape
    # Add computational node to the graph
    self._A = tf.constant(A, dtype=tf.float32, name="A")
    self._O = tf.constant(O, dtype=tf.float32, name="0")
    self._D = tf.constant(D, dtype=tf.float32, name="D")
    self._lambda = tf.constant(l, dtype=tf.float32, name="lambda")
    # Intialize W, H with truncated normal distribution values
    # The mean is 1.0 and stddev is 0.5 to ensure only positive number is choosen
    N = A.sum()
    self._W = tf.Variable(tf.random_uniform([A.shape[0],k], minval=0.0, maxval=2/N ), name="W")
    self._H = tf.Variable(tf.random_uniform([A.shape[0],k], minval=0.0, maxval=2/N), name="H")
    self._W_tmp = tf.Variable(tf.random_uniform([A.shape[0],k], minval=0.0, maxval=2/N ), name="W_tmp")
    self._H_tmp = tf.Variable(tf.random_uniform([A.shape[0],k], minval=0.0, maxval=2/N), name="H_tmp")
    # Add update nodes compute denominator and numerator separately
    ## Update W
    w_nume = tf.matmul(self._A, self._H, a_is_sparse=True)
    w_deno_temp = tf.matmul(self._H, self._H, transpose_a = True)
    w_deno = tf.matmul(self._W, w_deno_temp)
    w_update_factor = tf.div(w_nume, w_deno + 10e-8)
    W_new = tf.mul(w_update_factor, self._W)
    self._updateW = self._W_tmp.assign(W_new)
    self._assignW = self._W.assign(self._W_tmp)
    ## Update H
    h_nume_a = tf.matmul(self._A, self._W, transpose_a = True,
                         a_is_sparse=True)
    h_nume_b_temp = tf.matmul(self._O, self._H, a_is_sparse=True)
    h_nume_b = tf.mul(self._lambda, h_nume_b_temp)
    h_nume = tf.add(h_nume_a, h_nume_b)
    h_deno_a_temp = tf.matmul(self._W, self._W, transpose_a = True)
    h_deno_a = tf.matmul(self._H, h_deno_a_temp)
    h_deno_b_temp = tf.matmul(self._D, self._H, a_is_sparse=True)
    h_deno_b = tf.mul(self._lambda, h_deno_b_temp)
    h_deno = tf.add(h_deno_a, h_deno_b)
    h_update_factor = tf.div(h_nume, h_deno + 10e-8)
    H_new = tf.mul(h_update_factor, self._H)
    self._updateH = self._H_tmp.assign(H_new)
    self._assignH = self._H.assign(self._H_tmp)

    self.loss = tf.nn.l2_loss(self._A - tf.matmul(self._W, self._H,
                                                  transpose_b=True))
    L = self._D - self._O
    self.reg = self.supervisor_term(self._H, L)
    self.cost = self.loss + self._lambda * self.reg
                                                  
  # Return the node that calculates new H value
  def update_H_node (self) :
      return self._updateH

  # Return the node that calculates new W value
  def update_W_node (self) :
      return self._updateW

  # Return the node that assign H to the new value
  def assign_H_node (self) :
      return self._assignH

  # Return the node that assign W to the new value
  def assign_W_node (self) :
      return self._assignW

  @classmethod
  def supervisor_term(cls, H, L):
      HLH = tf.matmul(tf.matmul(H, L, transpose_a=True, b_is_sparse=True), H)
      mask = tf.diag(tf.ones([H.get_shape()[1]]))
      sup_term = tf.reduce_sum(HLH * mask)
      return sup_term

