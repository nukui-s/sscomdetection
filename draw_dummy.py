#!-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import os
import sys
from string import Template
from settings import *
import pandas

colors = {"abs_adam":"r", "abs_sgd":"g", "clip_sgd":"c", "clip_adam":"m","update_rule":"b",
          "kl_adam": 'k'}
markers = {"abs_adam":"o", "abs_sgd":"v", "clip_sgd":"^", "clip_adam":"8","update_rule":"*",
          "kl_adam": "<"}


label_dict = {"abs_adam":"Adam(abs)", "clip_adam":"Adam(PG)", 
              "abs_sgd":"GD(abs)","clip_sgd":"GD(PG)","update_rule":"Mult",
              "kl_adam": "KL/Adam(abs)"}


def draw_line(model, clr, mkr, densities, nmi_list, std_list):
    print(densities)
    print(nmi_list)
    plt.plot(densities, nmi_list, marker=mkr,markersize=12,
             color=clr, linewidth=2.0, label=model)


def load_model_nmi(data_name, model, densities):
    nmi_list = []
    time_list = []
    std_list = []
    d_list = []
    for d in densities:
        path = os.path.join("result","result_"+data_name+"_"+str(d)+".csv")
        reader = csv.reader(open(path))
        #for name, meantime, stdtime, nmi, std, best_cost, best_nmi in reader:
        for name, meantime, stdtime, nmi, std, best_cost, best_nmi, best_time in reader:
            if name == model:
                nmi_list.append(float(best_nmi))
                std_list.append(float(std))
                d_list.append(d/100)
                time_list.append(float(best_time))
                continue
    return d_list, nmi_list, std_list, time_list


def load_dummy_data(nmi_file):
    nmi_map = {}
    df = pandas.read_csv(nmi_file, header=None)
    for line in df.values:
        model = line[0]
        nmi_list = line[1:]
        nmi_map[model] = nmi_list
    return nmi_map

if __name__=="__main__":

    fig = plt.figure(1)
    #nmi_file = "dummy/polbooks_nmi.csv"
    #nmi_file = "dummy/karate_nmi.csv"
    #nmi_file = "dummy/friendship_nmi.csv"
    nmi_file = "dummy/gn_lambda01_nmi.csv"
    dummy_data = load_dummy_data(nmi_file)

    d_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    std_list = [0, 0, 0, 0, 0, 0]
    #models = ["clip_sgd", "abs_sgd", "clip_adam", "abs_adam", "kl_adam", "update_rule"]
    models = ["abs_adam", "update_rule"]
    for name in models:
        nmi_list = dummy_data[name]

        label = label_dict[name]
        clr = colors[name]
        mkr = markers[name]
        draw_line(label, clr, mkr, d_list, list(nmi_list), std_list)

    margin = 0.005
    plt.xlabel("used priors", fontsize=26)
    plt.ylabel("NMI", fontsize=26)
    plt.xlim([-margin, max(d_list)+margin])
    plt.ylim([-0.05,1.05])
    plt.legend(loc="lower right", prop={"size":18})
    plt.savefig("figs/%s_nmi.png"%(data_label))
    plt.show()

    plt.clf()
