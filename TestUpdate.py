# Author : Hoang NT
# Date : 2016-03-08
#
# Simple test case for Updater.py

import tensorflow as tf
import numpy as np
import Update as ud

# Made-up matrix for testing
# A is 5-by-5 adj matrix
# O is 5-by-5 prior knowledge matrix
# D is 5-by-5 diagonal matrix of O
# k = 4 - Suppose we have 4 clustering
# lambda = 0.5 - Consider trade off between topology (A) and prior knowledge (O)

A = np.array([[0, 1, 1, 0, 1],
       [1, 0, 1, 1, 0],
       [1, 1, 0, 0, 1],
       [0, 1, 0, 0, 1],
       [1, 0, 1, 1, 0]])
O = np.array([[1, 1, 1, 1, 1],
       [1, 0, 1, 1, 1],
       [1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 0, 1, 1]])
D = np.diag(O.sum(axis=1))
k = 4
l = 0.5 # lambda

iterations = 100

# Create a tensorflow graph and add nodes
graph = tf.Graph()
updater = ud.UpdateElem()
# Add updating rule computing node to graph
# Look at UpdateElem class (Update.py) for more detail
updater.add_semi_supervised_rule(A, O, D, k, l)
# Create a session to run
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
# Check for initial values of W and H
print(sess.run(updater._W))
print(sess.run(updater._H))
# Update the matrices
for _ in range(iterations) :
  # Get the new value for W and W
  sess.run([updater.update_H_node(), updater.update_W_node()])
  # Assign W and H to the new values
  sess.run([updater.assign_H_node(), updater.assign_W_node()])
# Print results
print('Final result for %d iterations' % iterations)
print(sess.run(updater._W))
print(sess.run(updater._H))
