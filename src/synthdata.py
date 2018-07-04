# -*- coding: utf-8 -*-
"""
create synthetic S, X1, X2, y quadraple
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

def Simulate_Data(n=1000, d_x = 3, d_s = 2):
  #model y = (1,1)^\top (s, s*s) + (1,1,1)^\top (x, x*x) + eps
  eps = np.random.normal(size=n)
  X = np.random.normal(size=(n,d_x))
  S = np.random.normal(size=(n,d_s))
  y = np.zeros(n)
  for i in range(n):
    y[i] = eps[i]
    for l in range(d_s):
      y[i] += abs(S[i,l])
    for l in range(d_x):
      y[i] += abs(X[i,l])
  X1 = X
  X2 = np.array([[1] for i in range(n)])
  y = y/np.std(y)
   
  return np.array(S), np.array(X1), np.array(X2), np.array(y)

if __name__ == '__main__':
  Simulate_Data()
