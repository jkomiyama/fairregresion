# -*- coding: utf-8 -*-
"""
convert the NLSY79 dataset into S, X1, X2, y quadraple
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

def ketastr(i):
  v = str(i)
  if len(v)<2:
    v = "0"+v
  return v

def read_nlsy(filename = os.path.join(conf.datadir, "nlsy79/data.csv"), use_explanatory=False, contS=False, return_all=False, single_S = False): #read nlsy dataset file (numeric ver)
  global names, numeric_values, string_values
  lines = [line for line in open(filename, "r").readlines()]
  #fo = open(filename, "w")
  #for line in lines:
  #  fo.write(line)
  #fo.close()
  data = pd.read_csv(filename, sep=',')
  age = data["R0000600"]
  race = data["R0009600"] #categorical
  gender = data["R0214800"] #binary
  grade90 = data["R3401501"] 
  income06 = data["T0912400"]
  income96 = data["R5626201"]
  income90 = data["R3279401"]
  partner = data["R2734200"] #binary
  height = data["R0481600"] 
  weight = data["R1774000"] 
  famsize = data["R0217502"] 
  genhealth = data["H0003400"] 
  illegalact = data["R0304900"] #categorical
  charged = data["R0307100"] 
  jobsnum90 = data["R3403500"] 
  afqt89 = data["R0618300"] 
  typejob90 = data["R3127300"] 
  #data = data[data.R3127500 >= 0] 
  #classjob90 = data["R3127500"] 
  jobtrain90 = data["R3146100"] 
  #data = data[data.R0304900 >= 0]
  
  my_attrs = [gender,income90,genhealth,illegalact,age,charged,grade90,jobsnum90,afqt89,jobtrain90]#,height,weight]
  #  my_attrs.append( pd.get_dummies(industories90[i], prefix="industory"+ketastr(i), drop_first=False) ) 
  #my_attrs.append( pd.get_dummies(classjob90, prefix="classjob90", drop_first=True) ) 
  #my_attrs.append( pd.get_dummies(illegalact, prefix="illegalact", drop_first=True) ) 
  new_data = pd.concat(my_attrs, axis=1)
  new_data["job_agri"] = [int(10 <= j <= 39) for j in typejob90]
  new_data["job_mining"] = [int(40 <= j <= 59) for j in typejob90]
  new_data["job_construction"] = [int(60 <= j <= 69) for j in typejob90]
  new_data["job_manuf"] = [int(100 <= j <= 399) for j in typejob90]
  new_data["job_transp"] = [int(400 <= j <= 499) for j in typejob90]
  new_data["job_wholesale"] = [int(500 <= j <= 579) for j in typejob90]
  new_data["job_retail"] = [int(580 <= j <= 699) for j in typejob90]
  new_data["job_fin"] = [int(700 <= j <= 712) for j in typejob90]
  new_data["job_busi"] = [int(721 <= j <= 760) for j in typejob90]
  new_data["job_personal"] = [int(761 <= j <= 791) for j in typejob90]
  new_data["job_enter"] = [int(800 <= j <= 811) for j in typejob90]
  new_data["job_pro"] = [int(812 <= j <= 892) for j in typejob90]
  new_data["job_pub"] = [int(900 <= j <= 932) for j in typejob90]
  new_data = new_data.rename(columns={"R0000600":"age"})
  new_data = new_data.rename(columns={"R0214800":"gender"})
  new_data["gender"] = new_data["gender"]-1 #1,2->0,1
  new_data = new_data.rename(columns={"R3279401":"income"})
  new_data = new_data[new_data.income >= 0]
  new_data = new_data.rename(columns={"R3401501":"grade90"})
  new_data = new_data[new_data.grade90 >= 0]
  #new_data = new_data.rename(columns={"R2734200":"partner"})
  #new_data = new_data[new_data.partner >= 0]
  #new_data = new_data.rename(columns={"R0217502":"famsize"})
  #new_data = new_data[new_data.famsize >= 0]
  new_data = new_data.rename(columns={"H0003400":"genhealth"})
  new_data = new_data[new_data.genhealth >= 0]
  new_data = new_data.rename(columns={"R0304900":"illegalact"})
  new_data = new_data[new_data.illegalact >= 0]
  new_data = new_data.rename(columns={"R0307100":"charged"})
  new_data = new_data[new_data.charged >= 0]
  new_data = new_data.rename(columns={"R3403500":"jobsnum90"})
  new_data = new_data[new_data.jobsnum90 >= 0]
  new_data = new_data.rename(columns={"R0618300":"afqt89"})
  new_data = new_data[new_data.afqt89 >= 0]
  new_data = new_data.rename(columns={"R3146100":"jobtrain90"})
  new_data = new_data[new_data.jobtrain90 >= 0]
  #new_data = new_data.rename(columns={"R0481600":"height"})
  #new_data = new_data[new_data.height >= 0]
  #new_data = new_data.rename(columns={"R1774000":"weight"})
  #new_data = new_data[new_data.weight >= 0]

  #new_data = new_data.dropna() #this does not work in nlsy
  new_data.insert(0, "intercept", 1)

  new_data["income"] = new_data["income"]/10000.0
  y = list(new_data["income"])
 
  if single_S:
    S_keys = ["gender"]
  else:
    S_keys = ["gender","age"]
  S = np.transpose([list(new_data[i]) for i in S_keys])
  X_keys = set(new_data.keys()).difference(["income"]+S_keys)
  print("X_keys=",X_keys)
  X2_keys = set(["intercept"]).intersection(X_keys)
  print("X2 keys=",X2_keys)
  X2 = np.transpose([list(new_data[i]) for i in X2_keys])

  #print("X2=",str(X2))
  #X2 = np.array(X2).reshape([len(new_data),len(X2_keys)])
  #print "X2=",X2.shape
  #print "X2=",X2
  X1_keys = X_keys.difference(X2_keys)
  X1 = np.transpose([list(new_data[i]) for i in X1_keys])
  print("X1 keys=",X1_keys)
  #print "S=",S[:10]
 
  return np.array(S), np.array(X1), np.array(X2), np.array(y)

if __name__ == '__main__':
 read_nlsy()

