import numpy as np
import tensorflow as tf
import sys
import os
import pandas as pd
from sscd import SSCD

if __name__ == '__main__':
    edge_list = pd.read_pickle("data/karate.pkl")
    correct = pd.read_pickle("data/karate_com.pkl")
    const = [(6,7), (30,31)]

    model = SSCD(2)

    model.fit_and_transform(edge_list, const)

    
