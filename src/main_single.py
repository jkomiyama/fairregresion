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
import argparse
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import feature_selection
from sklearn import naive_bayes
from scipy import stats
import pandas as pd
#from statsmodels.discrete import discrete_model
import math
import random
from joblib import Parallel, delayed

import compasdata, communitydata, lsacdata, nlsydata
#import germandata, germandata_numeric
import synthdata
from util import Dataset, Result, ResultRep
import preprocessing

def remove_var0_attrs(trainS, trainX1, validX1, testX1, use_boxcox=False): #removing feature X in X1 s.t. Var[X|S=1 or 0]=0
  #print "trainX1.shape=",trainX1.shape
  #print "validX1.shape=",validX1.shape
  #print "testX1.shape=",testX1.shape
  new_train_X1_tmp = []
  new_valid_X1_tmp = []
  new_test_X1_tmp = []
  NumData, X1_size = trainX1.shape[0], trainX1.shape[1]
  NumValidData = validX1.shape[0]
  NumTestData = testX1.shape[0]
  for j in range(X1_size):
    is_regressionFeature = len(set(trainX1[:,j]))>=5
    if is_regressionFeature: #normalization
      tmp = np.concatenate((trainX1[:,j],validX1[:,j],testX1[:,j]))
      u = max(tmp)
      l = min(tmp)
      trainX1[:,j] = (trainX1[:,j]-l)/(u-l)
      validX1[:,j] = (validX1[:,j]-l)/(u-l)
      testX1[:,j] = (testX1[:,j]-l)/(u-l)
    #stddev1 = np.std([trainX1[i,j] for i in range(NumData) if trainS[i][0]==1])
    #stddev0 = np.std([trainX1[i,j] for i in range(NumData) if trainS[i][0]==0])
    #if stddev1 > 0.0 and stddev0 > 0.0:
    if np.std(trainX1[:,j])>0:
      if use_boxcox and is_regressionFeature: #box-cox
        tmp = np.concatenate((trainX1[:,j],validX1[:,j],testX1[:,j]))
        tmp, _ = stats.boxcox(tmp+1-min(tmp))
        trainX1[:,j] = tmp[:NumData]
        validX1[:,j] = tmp[NumData:NumData+NumValidData]
        testX1[:,j]  = tmp[NumData+NumValidData:]
      new_train_X1_tmp.append(trainX1[:,j])
      new_valid_X1_tmp.append(validX1[:,j])
      new_test_X1_tmp.append(testX1[:,j])
    else:
      print ("feature",j,"dropped")
  new_train_X1 = np.array(new_train_X1_tmp).transpose()
  new_valid_X1 = np.array(new_valid_X1_tmp).transpose()
  new_test_X1 = np.array(new_test_X1_tmp).transpose()
  #print "ntrainX1.shape=",new_train_X1.shape
  #print "nvalidX1.shape=",new_valid_X1.shape
  #print "ntestX1.shape=",new_test_X1.shape
  return new_train_X1, new_valid_X1, new_test_X1

def data_shuffle(S, X1, X2, Y, seed):
  l = len(Y)
  arr = list(range(l))
  np.random.seed(seed=seed)
  np.random.shuffle(arr)
  Snew, X1new, X2new, Ynew = copy.deepcopy(S), copy.deepcopy(X1), copy.deepcopy(X2), copy.deepcopy(Y)
  for i in range(l):
    Snew[i]  = S[arr[i]]
    X1new[i] = X1[arr[i]]
    X2new[i] = X2[arr[i]]
    Ynew[i]  = Y[arr[i]]
  return Snew, X1new, X2new, Ynew

def fold_split(S, X1, X2, Y, n, k, validation = False):
  if k<0 or k>=n:
    print ("Error: k is not in [0,...,n-1]");sys.exit(0)
  l = len(Y)
  test_idx = list(range(int((l*k)/n), int((l*(k+1))/n)))
  if k == n-1:
    valid_idx = list(range(int((l*0)/n), int((l*(0+1))/n)))
  else:
    valid_idx = list(range(int((l*(k+1))/n), int((l*(k+2))/n)))
  valid_idx_set = set(valid_idx)
  test_idx_set = set(test_idx)
  if validation:
    training_idx = [i for i in range(l) if (not i in test_idx_set and not i in valid_idx_set)]
  else:
    training_idx = [i for i in range(l) if not i in test_idx_set]
  if validation:
    trainS, validS, testS = S[training_idx], S[valid_idx], S[test_idx]
    trainX1, validX1, testX1 = X1[training_idx], X1[valid_idx], X1[test_idx]
    trainX2, validX2, testX2 = X2[training_idx], X2[valid_idx], X2[test_idx]
    trainY, validY, testY = Y[training_idx], Y[valid_idx], Y[test_idx]
    return trainS, trainX1, trainX2, trainY, validS, validX1, validX2, validY, testS, testX1, testX2, testY
  else:
    trainS, testS = S[training_idx], S[test_idx]
    trainX1, testX1 = X1[training_idx], X1[test_idx]
    trainX2, testX2 = X2[training_idx], X2[test_idx]
    trainY, testY = Y[training_idx], Y[test_idx]
    return trainS, trainX1, trainX2, trainY, testS, testX1, testX2, testY

def experiment(p, trainData, validData, testData, eps, hparams, run, resultRep):
  trainS, trainX1, trainX2, trainY = copy.deepcopy(trainData)
  validS, validX1, validX2, validY = copy.deepcopy(validData)
  testS, testX1, testX2, testY = copy.deepcopy(testData)
  dataset = Dataset(trainS, trainX1, trainX2, trainY)
  dataset.add_validdata(validS, validX1, validX2, validY)
  dataset.add_testdata(testS, testX1, testX2, testY)
  avails = []
  for j in range(len(testS[0])):
    vals = set()
    vals = vals.union(set([s[j] for s in trainS]))
    #print "trainS=",trainS
    vals = vals.union(set([s[j] for s in validS]))
    #print "validS=",trainS
    vals = vals.union(set([s[j] for s in testS]))
    #print "testS=",trainS
    avails.append(vals == set([0,1]))
    #print "j,vals=",j,vals

  result_unfair_train, result_unfair_valid, result_unfair_test = dataset.Unfair_Prediction(p.kernel or p.rff, hparams["lmd"], hparams["gamma"], avails)
  title = {}
  result_train, result_valid, result_test = dataset.EpsFair_Prediction(p.dataset, eps, hparams, avails, p)
  title["hparam"] = hparams
  title["preprocessing"] = p.preprocessing
  title["run"]=run
  if p.kernel:
    title["kernel"]="kernel"
  elif p.rff and p.nonlinears:
    title["kernel"]="rff-ns"
  elif p.rff:
    title["kernel"]="rff"
  else:
    title["kernel"]="no"
  title["eps"]=eps
  title["dataset"]="train"
  resultRep.add_run(copy.deepcopy(title), result_train)
  title["dataset"]="valid"
  resultRep.add_run(copy.deepcopy(title), result_valid)
  title["dataset"]="test"
  resultRep.add_run(copy.deepcopy(title), result_test)
  title["eps"]="unfair"
  title["dataset"]="train"
  resultRep.add_run(copy.deepcopy(title), result_unfair_train)
  title["dataset"]="valid"
  resultRep.add_run(copy.deepcopy(title), result_unfair_valid)
  title["dataset"]="test"
  resultRep.add_run(copy.deepcopy(title), result_unfair_test)

def main_single(p, S, X1, X2, Y, eps, hparams, n, k):
  resultRep = ResultRep()
  trainS, trainX1, trainX2, trainY, validS, validX1, validX2, validY, testS, testX1, testX2, testY \
    = fold_split(S, X1, X2, Y, n, k, validation=True)
  trainS, trainX1, trainX2, trainY, validS, validX1, validX2, validY, testS, testX1, testX2, testY = copy.deepcopy( (trainS, trainX1, trainX2, trainY, validS, validX1, validX2, validY, testS, testX1, testX2, testY) )
  trainX1, validX1, testX1 = remove_var0_attrs(trainS, trainX1, validX1, testX1)
  trainData = [trainS, trainX1, trainX2, trainY]
  validData = [validS, validX1, validX2, validY]
  testData  = [testS, testX1, testX2, testY]
  experiment(p, trainData, validData, testData, eps, hparams, k, resultRep) 
  return resultRep

def main(p):
  dataset = p.dataset
  kernel = p.kernel
  n = p.fold
  
  #eps_list = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
  #if p.rff:
  #  #gamma_list = [1.0]
  #  gamma_list = [0.1, 1.0, 10.0, 100.0]
  #  lmd_list = [1.0, 10.0, 100.0]
  #elif kernel:
  #  #gamma_list = [1.0]
  #  gamma_list = [0.1, 1.0, 10.0, 100.0]
  #  lmd_list = [0.01, 0.1, 1.0, 10.0]
  #else:
  #  gamma_list = [1.0]
  #  lmd_list = [1.0, 10.0, 100.0]
  #eps_k_hparams = []
  S, X1, X2, Y = read_data(p)
#  print ("S.shape=",S.shape)
#  print ("X1.shape=",X1[:,0].shape)
#  for j in range(X2.shape[1]):
#    print ("corr before=",np.corrcoef(S[:,0], X2[:,j])[0,1])
  if p.preprocessing == "quantile":
#    print ("conducting quantile transformation")
    X1, X2 = preprocessing.quantile(S, X1, X2)
#  for j in range(X2.shape[1]):
#    print ("corr after=",np.corrcoef(S[:,0], X2[:,j])[0,1])
   
  resultRep_splits = []
  for k in range(n): 
    eps, gamma, lmd = p.eps, p.gamma, p.lmd
    hparams = {}
    hparams["gamma"] = gamma; hparams["lmd"]=lmd
    resultRep_splits.append(main_single(p, S, X1, X2, Y, eps, hparams, n, k))
  resultRep = ResultRep()
  for r in resultRep_splits:
    resultRep.merge(r)
  print (resultRep)
  print (resultRep.str_pretty())

def read_data(p):
  dataset = p.dataset
  if dataset == "cc":
    print ("C&C data")
    S, X1, X2, Y = communitydata.read_community()
  elif dataset == "nlsy":
    if p.single_s:
      print ("NLSY data (single s)")
      S, X1, X2, Y = nlsydata.read_nlsy(single_S = True)
    else:
      print ("NLSY data")
      S, X1, X2, Y = nlsydata.read_nlsy(single_S = False)
  elif dataset == "synth":
    print ("synth data")
    S, X1, X2, Y = synthdata.Simulate_Data()
  elif dataset == "lsac":
    if p.single_s:
      print ("lsac data (single s)")
      S, X1, X2, Y = lsacdata.read_lsac(single_S=True)
    else:
      print ("lsac data")
      S, X1, X2, Y = lsacdata.read_lsac()
  elif dataset == "compas":
    print ("compas data")
    S, X1, X2, Y = compasdata.read_compas()
  else:
    print ("unknown dataset")
    sys.exit(0)
  S, X1, X2, Y = data_shuffle(S, X1, X2, Y, p.seed)
  S, X1, X2, Y = S.astype(np.float64), X1.astype(np.float64), X2.astype(np.float64), Y.astype(np.float64)
  return S, X1, X2, Y


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='process some integers.') 
  parser.add_argument('-d', '--dataset', \
        action='store', \
        nargs='?', \
        const=None, \
        default="cc", \
        type=str, \
        choices=None, \
        help='dataset', \
        metavar=None)
  parser.add_argument('-p', '--preprocessing', action='store', 
        default="", \
        type=str, \
        choices=None)
  parser.add_argument('-k', '--kernel', action='store_true')
  parser.add_argument('-r', '--rff', action='store_true')
  parser.add_argument('-n', '--nonlinears', action='store_true')
  parser.add_argument('-s', '--seed', action='store', type=int, default=1) #rand seed
  parser.add_argument('-u', '--single_s', action='store_true', default=False) #if true dim(s)=1
  parser.add_argument('-f', '--fold', action='store', type=int, default=5) #n-fold
  parser.add_argument('-e', '--eps', action='store', type=float, default=0.1) #eps (fairness parameter)
  parser.add_argument('-g', '--gamma', action='store', type=float, default=1.0) #gamma (hyperparameter)
  parser.add_argument('-l', '--lmd', action='store', type=float, default=1.0) #lambda (hyperparameter)
  p = parser.parse_args()
  main(p) #C&C no explanatory vars


