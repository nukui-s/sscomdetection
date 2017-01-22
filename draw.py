#!-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import os
import sys
from string import Template
from settings import *

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


def load_model_nmi(exp_name, model, densities):
    nmi_list = []
    time_list = []
    std_list = []
    d_list = []
    for d in densities:
        path = os.path.join("result", exp_name+"_"+str(d)+".csv")
        reader = csv.reader(open(path))
        #for name, meantime, stdtime, nmi, std, best_cost, best_nmi in reader:
        for name, meantime, stdtime, nmi, std, best_cost, best_nmi, best_time in reader:
            if name == model:
                nmi_list.append(float(best_nmi))
                std_list.append(float(std))
                d_list.append(d)
                time_list.append(float(best_time))
                continue
    return d_list, nmi_list, std_list, time_list


if __name__=="__main__":

    fig = plt.figure(1)
    for name in used_models:
        d_list, nmi_list, std_list, time_list = load_model_nmi(exp_name, name, densities)
        if len(d_list) == 0: continue
        label = label_dict[name]
        clr = colors[name]
        mkr = markers[name]
        draw_line(label, clr, mkr, d_list, nmi_list, std_list)

    margin = 0.00
    plt.xlabel("used priors", fontsize=26)
    plt.ylabel("NMI", fontsize=26)
    plt.xlim([-margin, max(densities)+margin])
    plt.ylim([-0.0,1.05])
    plt.legend(loc="best", prop={"size":18})
    plt.savefig("figs/%s_nmi.png"%(data_label))
    plt.show()

    plt.clf()

    fig2 = plt.figure(2)
    for name in used_models:
        d_list, nmi_list, std_list, time_list = load_model_nmi(exp_name, name, densities)
        if len(d_list) == 0: continue
        label = label_dict[name]
        clr = colors[name]
        mkr = markers[name]
        draw_line(label, clr, mkr, d_list, time_list, std_list)

    plt.xlabel("used priors", fontsize=26)
    plt.ylabel("Elapsed Time", fontsize=26)
    plt.xlim([-margin,max(densities) + margin])
    plt.legend(loc="best", prop={"size":18})
    plt.savefig("figs/%s_time.png"%(data_label))
    plt.show()

