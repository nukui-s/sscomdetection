#conding: utf-8
import numpy as np
from itertools import combinations
from itertools import chain

def constrain_label(correct_labels, edge_list, c_num):
    node_deg = {}
    for e1, e2 in edge_list:
        node_deg[e1] = node_deg.get(e1,0) + 1
        node_deg[e2] = node_deg.get(e2,0) + 1
    deg_tuples = list(node_deg.items())
    deg_tuples.sort(key=lambda x: x[1], reverse=True)
    indices = [k for k, v in deg_tuples]
    correct_labels = np.array(correct_labels)
    n_nodes = correct_labels.size
    targets = indices[:c_num]
    print("Target:" + str(targets))
    print("Labels: " + str(correct_labels[targets]))
    const_pairs = []
    for i, j in combinations(targets, 2):
        c1 = correct_labels[i]
        c2 = correct_labels[j]
        if c1 == c2:
            const_pairs.append((i,j))
    return const_pairs

if __name__=="__main__":
    labels = [1,1,1,2,2,2,1,2,3,3,3]
    pairs = constrain_label(labels, 4)
    print(labels)
    print(pairs)



