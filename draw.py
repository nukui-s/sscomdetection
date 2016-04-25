#!-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import os
import sys
from string import Template
from settings import *
import seaborn

colors = {"abs_adam":"r", "abs_sgd":"g", "clip_sgd":"c", "clip_adam":"m","update_rule":"b"}
markers = {"abs_adam":"o", "abs_sgd":"v", "clip_sgd":"^", "clip_adam":"8","update_rule":"*"}


label_dict = {"abs_adam":"Adam_abs", "clip_adam":"Adam_proj", 
                "abs_sgd":"GD_abs","clip_sgd":"GD_proj","update_rule":"Mult"}


def draw_line(model, clr, mkr, densities, nmi_list, std_list):
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
        for name, meantime, stdtime, nmi, std, best_cost, best_nmi, best_time in reader:
            if name == model:
                nmi_list.append(float(best_nmi))
                std_list.append(float(std))
                d_list.append(d/100)
                time_list.append(float(best_time))
                continue
    return d_list, nmi_list, std_list, time_list


if __name__=="__main__":

    fig = plt.figure(1)
    for name in used_models:
        d_list, nmi_list, std_list, time_list = load_model_nmi(data_label, name, densities)
        if len(d_list) == 0: continue
        label = label_dict[name]
        clr = colors[name]
        mkr = markers[name]
        draw_line(label, clr, mkr, d_list, nmi_list, std_list)

    plt.xlabel("used priors", fontsize=26)
    plt.ylabel("NMI", fontsize=26)
    plt.xlim([0,max(densities)*0.01])
    plt.ylim([-0.05,1.05])
    plt.legend(loc="best", prop={"size":20})
    plt.savefig("figs/%s_nmi.png"%(data_label))
    #plt.show()

    plt.clf()


    fig2 = plt.figure(2)
    for name in used_models:
        d_list, nmi_list, std_list, time_list = load_model_nmi(data_label, name, densities)
        if len(d_list) == 0: continue
        label = label_dict[name]
        clr = colors[name]
        mkr = markers[name]
        draw_line(label, clr, mkr, d_list, time_list, std_list)

    plt.xlabel("used priors", fontsize=26)
    plt.ylabel("Elapsed Time", fontsize=26)
    plt.xlim([0,max(densities)*0.01])
    plt.legend(loc="best", prop={"size":20})
    plt.savefig("figs/%s_time.png"%(data_label))
    #plt.show()

