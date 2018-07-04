# -*- coding: utf-8 -*-
"""
convert lsac dataset into S, X1, X2, y quadraple
http://www2.law.ucla.edu/sander/Systemic/Data.htm
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
import math
import random
from io import open
from sas7bdat import SAS7BDAT
import conf

def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text


def read_lsac(filename = os.path.join(conf.datadir, "law/lsac.sas7bdat"), regression=True, single_S = False): #read lsac dataset file
  #lines = [line for line in open(filename, "r").readlines() if line.find("?")==-1]
  #fo = open(filename, "w")
  #for line in lines:
  #  fo.write(line)
  #fo.close()
  #data = pd.read_sas(filename)
  data = f=SAS7BDAT(filename).to_data_frame()
#  print data.describe();sys.exit(0)
  data = data.dropna()
#  for akey in sorted(list(data.keys())):
#    if akey != "ID":
#      print "support for key " + str(akey) + " =",set(data[akey])
#  sys.exit(0)

  real_values = ["DOB_yr","age","decile1","decile1b","decile3","fam_inc","lsat","ugpa"] #real (or int) values
  string_values = ["gender","race1","cluster","fulltime"] #binary/categorical values

  my_attrs = []
  for real_val in real_values:
    my_attrs.append(data[real_val])
  for string_val in string_values:
    my_attrs.append( pd.get_dummies(data[string_val], prefix=string_val, drop_first=True) )
  new_data = pd.concat(my_attrs, axis=1)
  #print "bar=",data["bar"];sys.exit()
  new_data["bar"] = data["bar"] == "a Passed 1st time"
  new_data.insert(0, "intercept", 1)
  new_data = new_data.dropna()
  #sys.exit(0) 
  #print "data=",new_data.describe()

  if single_S:
    S_keys = ["race1_black"]
  else:
    S_keys = ["race1_black","age"]
  S = np.transpose([list(new_data[i]) for i in S_keys])
  #S = np.array(S, dtype=np.int_)*2-1
  if not regression:
    y = new_data["bar"]
  else:
    y = new_data["ugpa"]
  #print "y=",y;sys.exit()
  print("keys=",len(new_data.keys()))
  X_keys = set(new_data.keys()).difference([]+S_keys)
  print("X_keys=",X_keys)
  #sys.exit()
  #print list(race.keys())
  #X2_keys = set()
  X2_keys = set(["intercept"]).intersection(X_keys)
  print("X2 keys=",X2_keys)
  X2 = np.transpose([list(new_data[i]) for i in X2_keys])
  print("X2=",str(X2))
  X2 = np.array(X2).reshape([len(new_data),len(X2_keys)])
  #print "X2=",X2.shape
  #print "X2=",X2
  if not regression:
    X1_keys = X_keys.difference(X2_keys.union(\
       set(["bar","bar_b Passed 2nd time","bar_c Failed", 'race1_black', 'race1_hisp', 'race1_other', 'race1_white'])))
  else:
    X1_keys = X_keys.difference(X2_keys.union(\
       set(["bar","bar_b Passed 2nd time","bar_c Failed", 'race1_black', 'race1_hisp', 'race1_other', 'race1_white', 'ugpa'])))
  X1 = np.transpose([list(map(float, list(new_data[i]))) for i in X1_keys])
  print("X1 keys=",X1_keys)
  #print "S=",S[:10]

  #print ("S=",S.shape)
  #print ("X=",new_data.shape)
  #print ("X1=",X1.shape)
  #print ("X2=",X2.shape)
  #print ("y=",y.shape)
 
  return np.array(S), np.array(X1), np.array(X2), np.array(y)

if __name__ == '__main__':
 read_lsac()
