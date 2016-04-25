import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from itertools import chain


def plot_degree_dist(edge_list):
    deg_dict = {}
    for n, n_ in edge_list:
        deg_dict[n] = deg_dict.get(n,0) + 1
    degrees = deg_dict.values()
    max_k = max(degrees)
    ddist = [0] * max_k
    for d in degrees:
        ddist[d-1] += 1
    plt.bar(range(max_k), ddist)
    plt.show()


if __name__=="__main__":
    elist = [(0,1), (1,2), (1,0)]
    plot_degree_dist(elist)
