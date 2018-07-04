# -*- coding: utf-8 -*-
"""
convert compas dataset into S, X1, X2, y quadraple
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import time
import datetime
import sys
import os
import copy
import itertools
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import pandas as pd
#from statsmodels.discrete import discrete_model
import math
import random
from io import open
import conf

def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text

def read_compas(filename = os.path.join(conf.datadir, "compas-analysis/compas-scores-two-years.csv"), smlfeatures=False, return_all=False, single_S=False): #read compas dataset file (numeric ver)
  lines = [line for line in open(filename, "r").readlines() if line.find("?")==-1]
  fo = open(filename, "w")
  for line in lines:
    fo.write(line)
  fo.close()
  #pd.set_option("display.max_rows", 100)
  #pd.set_option("display.max_colwidth", 100)
  #print dir(pd)
  data = pd.read_csv(filename, sep=',')

  int_values = ["age","juv_fel_count","decile_score","juv_misd_count","juv_other_count","v_decile_score","priors_count"] #,"is_recid"
  #string_values = ["sex","race","two_year_recid","c_charge_degree","c_charge_desc"]
  string_values = ["sex","two_year_recid","type_of_assessment","v_type_of_assessment"]#,"r_charge_desc"]
  date_values=["c_jail_in","c_jail_out","c_offense_date","screening_date","in_custody","out_custody"]

  my_attrs = []
  for int_val in int_values:
    my_attrs.append(data[int_val])
  for string_val in string_values:
    my_attrs.append( pd.get_dummies(data[string_val], prefix=string_val, drop_first=True) )
  for date_val in date_values:
    temp = pd.to_datetime(data[date_val])
    t_min, t_max = min(temp), max(temp)
    my_attrs.append( (temp-t_min)/(t_max-t_min) )
  new_data = pd.concat(my_attrs, axis=1)
  new_data["African-American"] = (data["race"] == "African-American")
  new_data = new_data.dropna()
  if return_all:
    return new_data
  new_data.insert(0, "intercept", 1)

  corr_akey = []
  for akey in new_data.keys():
    corr_akey.append((np.corrcoef(new_data[akey], new_data["two_year_recid_1"])[0,1], akey))

  if single_S: 
    S_keys = ["sex_Male"]
  else:
    S_keys = ["sex_Male", "African-American"]
  #race_Native American race_Asian race_Other race_Hispanic race_Caucasian
  S = np.transpose([list(new_data[i]) for i in S_keys])
  #S = np.array(S, dtype=np.int_)*2-1
  y = [v*2.0-1.0 for v in new_data["two_year_recid_1"]]
  X_keys = set(new_data.keys()).difference([]+S_keys)
  X_keys_nonrace = set()
  for akey in X_keys:
    if akey.find("race") != 0:
      X_keys_nonrace.add(akey)
  X_keys = X_keys_nonrace
  print("X_keys=",len(X_keys),X_keys)
  #print list(race.keys())
  #X2_keys = set()
  X2_keys = set(["intercept"]).intersection(X_keys)
  print("X2 keys=",X2_keys)
  X2 = np.transpose([list(new_data[i]) for i in X2_keys])
  #print("X2=",str(X2))
  X2 = np.array(X2).reshape([len(new_data),len(X2_keys)])
  #print "X2=",X2.shape
  #print "X2=",X2
  X1_keys = X_keys.difference(X2_keys.union(set(["two_year_recid_1"])))
  if smlfeatures:
    X1_keys = X1_keys.difference(set(["out_custody","decile_score","in_custody","c_jail_out","c_jail_in","screening_date","v_decile_score"]))
  X1 = np.transpose([list(new_data[i]) for i in X1_keys])
  print("X1 keys=",X1_keys)
  #sys.exit()
  #print "S=",S[:10]
 
  return np.array(S), np.array(X1), np.array(X2), np.array(y)

if __name__ == '__main__':
 read_compas()
