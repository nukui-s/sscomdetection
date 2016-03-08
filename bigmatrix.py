import numpy as np
import pandas as pd
import tensorflow as tf
from random import randint
from itertools import chain

enum = 100000
nnum = 10000
K = 10
print("check")
indices = [(randint(0,nnum-1), randint(0,nnum-1)) for _ in range(enum)]
print("check")
nnum = max(set(chain.from_iterable(indices))) + 1
print("check")
indices = sorted(list(set(indices)), key=lambda x: (x[0], x[1]))
print("check")
enum = len(indices)
print("check")
values = np.ones(enum, dtype=np.float32)
print("check")

spmat = tf.SparseTensor(indices,values,[nnum, nnum])

print("check")
var = tf.Variable(np.random.rand(nnum,K).astype(np.float32))
mat = tf.constant(np.random.rand(K,K).astype(np.float32))
print("check")

mul2 = tf.sparse_matmul(spmat, var)

targets = tf.placeholder(name="target", dtype="bool", shape=[nnum])

spmat_t = tf.boolean_mask(spmat, targets)
var_t = tf.boolean_mask(var, targets)

mul = tf.matmul(var_t, tf.matmul(spmat_t, var_t, a_is_sparse=True, transpose_a=True), transpose_b=True)

print("check")
loss = tf.nn.l2_loss(mul)
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
import pdb; pdb.set_trace()

opt = tf.train.AdamOptimizer(0.1).minimize(loss)
for _ in range(1000):
    vec = np.array([np.random.randint(0,100)==0 for _ in range(nnum)])
    #sess.run(opt, feed_dict={targets: vec})
    print(sess.run(mul2))
