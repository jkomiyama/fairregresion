# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import time
import datetime
import sys
import os
import copy
import itertools
import math
import random

adir = "plot/"

def read_data(filename):
  import ast,re
  lines = open(filename, "r").readlines()
  data = []
  for line in lines:
    if line.find("###Result:")!=-1:
      #print line
      title_str = line[10:].split("md=")[0]
      if False: #title_str.find("unfair")!=-1:
        continue
      else:
        #print title_str
        title = ast.literal_eval(title_str)
        md = re.search(r"md=\[(.+?)\]", line).group(1).split(",")
        corr = re.search(r"corr=\[(.+?)\]", line).group(1).split(",")
        #print md
        auc = re.search(r"auc=\[(.+?)\]", line).group(1).split(",")
        #print auc
        rmse = float(re.search(r"rmse=([0-9\.]+)", line).group(1))
        #print rmse
        if title["dataset"] == "test" and title["eps"] != "unfair":
          data.append((title, md, corr, auc, rmse))
  return data

def plot_data(out_name, data_list):

  plt.switch_backend('agg')
  plt.figure(figsize=(3, 2.5))
  plt.subplots_adjust(left=0.3, bottom=0.2)

  eps_list, rmse_list, md_list, corr_list, auc_list = [], [], [], [], []
  for (title, mds, corrs, aucs, rmse) in data_list:
    eps = title["eps"]
    eps_list.append(eps); rmse_list.append(rmse); md_list.append(mds); corr_list.append(corrs), auc_list.append(aucs)

  d_s = len(data_list[0][1])
  for j in range(d_s):
    plt.xlabel("Corr.Coef")
    plt.ylabel("RMSE")
    plt.xlim([0,1])
    plt.plot([corr[j] for corr in corr_list], rmse_list, marker="x", markersize=7)
    #plt.legend(loc="upper right")
    plt.savefig(os.path.join("plot/"+out_name+"_corr"+str(j)+"_rmse.png"))
    plt.clf()

if __name__ == '__main__':
  filename = sys.argv[1]
  #print filenames;sys.exit()
  data_list = read_data(filename)
  for elem in data_list:
    print (elem)
  out_name = sys.argv[2]
  plot_data(out_name, data_list)

