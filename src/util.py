# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import time
from scipy import sparse
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
from sklearn import neural_network
from sklearn import gaussian_process
from sklearn import kernel_approximation
from sklearn import kernel_ridge
from scipy import stats
#from fastFM import als, sgd, mcmc
import pandas as pd
#from statsmodels.discrete import discrete_model
import math
import random

#result class
class Result:
  def __init__(self, Y, Yhat, S, avails):
    self.md = self.getMD(Yhat, S, avails)
    self.corr = self.getCorr(Yhat, S)
    self.auc = self.getAUC(Yhat, S, avails)
    self.rmse = self.getRMSE(Y, Yhat)
  def elems(self):
    return self.md,self.corr,self.auc,self.rmse
  def __str__(self):
    return "Result: md="+str(self.md)+" corr="+str(self.corr)+" auc="+str(self.auc)+" rmse="+str(self.rmse)
  def getMD(self, Yhat, S, avails):
    if len(Yhat) != len(S):
      print ("Error: len(Yhat) != len(S)");sys.exit()
    d_s = len(S[0]) #num of sensitive features
    l = len(Yhat)
    mds = []
    for j in range(d_s):
      srange = set([S[i,j] for i in range(l)])
      if avails[j]:
        Y1 = [Yhat[i] for i in range(l) if S[i,j]==1]
        Y0 = [Yhat[i] for i in range(l) if S[i,j]==0]
        md = abs(np.mean(Y1)-np.mean(Y0))
        mds.append(md)
      else:
        mds.append(False)
    return mds
  def getCorr(self, Yhat, S):
    if len(Yhat) != len(S):
      print ("Error: len(Yhat) != len(S)");sys.exit()
    d_s = len(S[0]) #num of sensitive features
    l = len(S)
    corrs = []
    for j in range(d_s):
      corr = abs( np.corrcoef(Yhat, S[:,j])[0,1] ) #mdというか相関
      corrs.append(corr)
    return corrs
  def getAUC(self, Yhat, S, avails):
    if len(Yhat) != len(S):
      print ("Error: len(Yhat) != len(S)");sys.exit()
    d_s = len(S[0]) #num of sensitive features
    l = len(Yhat)
    aucs = []
    for j in range(d_s):
      srange = set([S[i,j] for i in range(l)])
      #print "srange=",srange
      if avails[j]:
        Y1 = [Yhat[i] for i in range(l) if S[i,j]==1]
        Y0 = [Yhat[i] for i in range(l) if S[i,j]==0]
        if len(Y1)*len(Y0)==0:
          auc = 0
        else:
          count = 0
          #slow (O(N^2)...)
          for y1i in Y1:
            for y0j in Y0:
              if y1i>y0j:
                count+=1
          auc = count/float(len(Y1)*len(Y0))
        aucs.append(auc)
      else:
        aucs.append(False)
    return aucs
  def getRMSE(self, Y, Yhat):
    if len(Yhat) != len(Y):
      print ("Error: len(Yhat) != len(S)");sys.exit()
    return np.mean([(Y[i]-Yhat[i])**2 for i in range(len(Y))])**0.5

#merges several Result classes
class ResultRep:
  def __init__(self):
    self.results, self.runnames = [], []
  def add_run(self, runname, result):
    self.results.append(result)
    self.runnames.append(runname)
  def merge(self, resultRep):
    for i in range(len(resultRep.results)):
      result, runname = resultRep.results[i], resultRep.runnames[i]
      self.results.append(result)
      self.runnames.append(runname)
  def __str__(self):
    strs = []
    for i,_ in enumerate(self.results):
      strs.append("Title:"+str(self.runnames[i])+" "+str(self.results[i]))
    return "\n".join(strs)
  def str_pretty(self):
    astr = ""
    md_avgs, corr_avgs, auc_avgs, rmse_avg = {}, {}, {}, {}
    for i,_ in enumerate(self.results):
      run = self.runnames[i]
      conf = copy.deepcopy(self.runnames[i])
      conf.pop('run', None)
      conf = str(conf) #for using conf as a key 
      #print ("conf=",conf)
      if not conf in md_avgs:
        md_avgs[conf], corr_avgs[conf], auc_avgs[conf], rmse_avg[conf] = [], [], [], []
      result = self.results[i]
      md_avgs[conf].append(result.md)
      corr_avgs[conf].append(result.corr)
      auc_avgs[conf].append(result.auc)
      rmse_avg[conf].append(result.rmse)
    for conf in md_avgs.keys():
      l = len(md_avgs[conf][0])
      mds = [np.mean([md_avgs[conf][i][j] for i in range(len(md_avgs[conf]))]) for j in range(l)]
      corrs = [np.mean([corr_avgs[conf][i][j] for i in range(len(md_avgs[conf]))]) for j in range(l)]
      aucs = [np.mean([auc_avgs[conf][i][j] for i in range(len(md_avgs[conf]))]) for j in range(l)]
      rmse = np.mean(rmse_avg[conf])
      astr += "###Result:"+str(conf) + " md=" + str(mds) + " corr=" + str(corrs) + " auc=" + str(aucs) + " rmse=" + str(rmse) + "\n"
      #strs.append("Title:"+self.runnames[i]+" "+str(self.results[i]))
    return astr

class Dataset:
  def __init__(self, S, X1, X2, y):
    self.has_validdata = False
    self.has_testdata = False
    self.fstStageRegressor = linear_model.Ridge(fit_intercept=True)
    self.trainS, self.trainX1, self.trainX2, self.trainY = S, X1, X2, y
  def set_traindata(self, S, X1, X2, y):
    self.trainS, self.trainX1, self.trainX2, self.trainY = S, X1, X2, y
  def add_validdata(self, S, X1, X2, y):
    self.validS, self.validX1, self.validX2, self.validY = S, X1, X2, y
    self.has_validdata = True
  def add_testdata(self, S, X1, X2, y):
    self.testS, self.testX1, self.testX2, self.testY = S, X1, X2, y
    self.has_testdata = True
  def getValidationData(self):
    if self.has_validdata:
      return copy.deepcopy((self.validS, self.validX1, self.validX2, self.validY))
    else:
      print ("Error: validation data not found");sys.exit(0)
  def getPredictData(self):
    if self.has_testdata:
      return copy.deepcopy((self.testS, self.testX1, self.testX2, self.testY))
    else:
      print ("Error: test data not found");sys.exit(0)
      return copy.deepcopy((self.trainS, self.trainX1, self.trainX2, self.trainY))
  def Unfair_Prediction_Noarg(self, lmd):
    X = np.c_[self.trainX1, self.trainX2]
    lr = linear_model.Ridge(alpha=lmd, fit_intercept=True)
    lr.fit( X, self.trainY )
    #validS, validX1, validX2, validY = self.getValidationData()
    testS, testX1, testX2, testY = self.getPredictData()
    predictX_train = np.c_[self.trainX1, self.trainX2] #use X1, S, and X2
    #predictX_valid = np.c_[validX1, validX2] #use X1, S, and X2
    predictX_test = np.c_[testX1, testX2] #use X1, S, and X2
    yhat_train = lr.predict(predictX_train).flatten()
    #yhat_valid = lr.predict(predictX_valid).flatten()
    yhat_test = lr.predict(predictX_test).flatten()
    y_pred_error_unfair = testY - yhat_test
    print ("genvar=",np.mean([(testY[i])**2 for i in range(len(testY))])**0.5)
    #print "unfair genavg=",np.mean([(testY[i]-yhat_test[i])**2 for i in range(len(testY))])**0.5
    return yhat_test, np.mean([y_pred_error_unfair**2 for i in range(len(testY))])**0.5
  def Unfair_Prediction(self, kernel, lmd, gamma, avails, use_S = False):
    if use_S:
      X = np.c_[self.trainX1, self.trainS]
    else:
      X = self.trainX1
    if not kernel: #linear
      lr = linear_model.Ridge(alpha=lmd, fit_intercept=True)
      #lr = linear_model.LinearRegression(fit_intercept=True)
    else:
      lr = kernel_ridge.KernelRidge(alpha=lmd, kernel="rbf", gamma=gamma)
    lr.fit( X, self.trainY )
    validS, validX1, validX2, validY = self.getValidationData()
    testS, testX1, testX2, testY = self.getPredictData()
    if use_S:
      predictX_train = np.c_[self.trainX1, self.trainS] 
      predictX_valid = np.c_[validX1, validS] 
      predictX_test = np.c_[testX1, testS] 
    else:
      predictX_train = self.trainX1
      predictX_valid = validX1 
      predictX_test = testX1
    yhat_train = lr.predict(predictX_train).flatten()
    yhat_valid = lr.predict(predictX_valid).flatten()
    yhat_test = lr.predict(predictX_test).flatten()
    #print ("genvar=",np.mean([(testY[i])**2 for i in range(len(testY))])**0.5)
    #print ("unfair genavg=",np.mean([(testY[i]-yhat_test[i])**2 for i in range(len(testY))])**0.5)
    return Result(self.trainY, yhat_train, self.trainS, avails), Result(validY, yhat_valid, validS, avails), Result(testY, yhat_test, testS, avails)
  def train_X1_resid(self, trainX1, trainS_X2, trainS, lr1, use_X2=True):
    X1_size = len(trainX1[0])
    NumData = len(trainX1)
    trainS_X2_resX1 = trainS_X2
    #self.stddevs = []
    for i in range(X1_size): #train models
      #print "train",i,trainS_X2_resX1.shape
      if use_X2:
        X1i = np.array([x[i] for x in trainX1])
        lr1[i].fit(trainS_X2_resX1, X1i)
        resid = X1i - lr1[i].predict(trainS_X2_resX1)
      else:
        X1i = np.array([x[i] for x in trainX1])
        lr1[i].fit(trainS, X1i)
        resid = X1i - lr1[i].predict(trainS)
      #stddevS1 = np.std([resid[j] for j in range(NumData) if trainS[j][0]==1])
      #stddevS0 = np.std([resid[j] for j in range(NumData) if trainS[j][0]==0])
      #if stddevS1*stddevS0 <= 0: #cannot correct variance
      #  print "error: var0 attr,i",i;sys.exit(0)
      #self.stddevs.append([stddevS1, stddevS0])
  def get_X1_resid(self, lrs, X1, S, S_X2, use_X2=True): #note that lrs are classifiers/regressors
    X1_size = len(X1[0])
    NumData = len(X1)
    #print "X1size,NumData=",X1_size,NumData,X1.shape
    X1_resid_tmp = []
    S_X2_resX1 = S_X2
    for i in range(X1_size):
      if use_X2:
        X1_resid_tmp.append( X1[:,i] - lrs[i].predict(S_X2_resX1) )
      else:
        X1_resid_tmp.append( X1[:,i] - lrs[i].predict(S) )
      #print "correcting variance heteroscadecity"
      #stddevS1, stddevS0 = self.stddevs[i]
      #stddev = np.std(X1_resid_tmp[i])
      #for j in range(NumData):
      #  X1_resid_tmp[i][j] /= stddev
        #if S[j][0]==1:
        #  X1_resid_tmp[i][j] /= stddevS1
        #else:
        #  X1_resid_tmp[i][j] /= stddevS0
    X1_resid = np.array([[X1_resid_tmp[i][j] for i in range(X1_size)] for j in range(NumData)])
    return X1_resid
  def save(self, S, X1, X2, Y, filename): #obsolate
    fo = file(filename, "w")
    fo.write("#"+str(S.shape[1])+","+str(X1.shape[1])+","+str(X2.shape[1])+","+str(1)+"\n") #header
    for i in xrange(len(Y)): #main
      fo.write(\
        ",".join([str(S[i,j]) for j in range(max(1,S.shape[1]))])+","+\
        ",".join([str(X1[i,j]) for j in range(X1.shape[1])])+","+\
        ",".join([str(X2[i,j]) for j in range(max(1,X2.shape[1]))])+","\
        +str(Y[i])+"\n")
    fo.close()
  def seeResidMean(self, x, S): #see whether E[x|S=1]=E[x|S=0]
    print ("E[x|S=1]=",np.mean([x[i] for i in range(len(x)) if S[i]==1]),"E[x|S=0]=",np.mean([x[i] for i in range(len(x)) if S[i]==0]),"E[x]=",np.mean([x[i] for i in range(len(x))]))
    print ("stddev[x|S=1]=",np.std([x[i] for i in range(len(x)) if S[i]==1]),"stddev[x|S=0]=",np.std([x[i] for i in range(len(x)) if S[i]==0]),"stddev[x]=",np.std([x[i] for i in range(len(x))]))
  def Fair_Prediction_Optimization(self, eps, lmd_n, Vs, Vx, vs, vx):
    #calculate alpha, beta as an optimization problem
    # min  a.T Vx a + b.T Vy b - a.T E[sy] - b.t E[xy]
    # s.t. (1-eps)a.T Vs a - eps b.T Vx b
    import fairopt
    if not (0<=eps<=1):
      print ("Error: eps must be in [0,1]");sys.exit(0)
    #def solveCQP(Q, q, c, epsVal): 
    #  #Solve 1QCQP whose objective function is convex.
    #  #min  x'*setQ{1}*x+2*setq{1}'*x+setc{1}
    #  #s.t. x'*setQ{2}*x+2*setq{2}'*x+setc{2} <= 0
    ds,dx = len(vs),len(vx)
    Q = [[],[]]
    Q[0] = np.zeros((ds+dx, ds+dx))
    Q[0][:ds,:ds] = Vs
    Q[0][ds:ds+dx,ds:ds+dx] = Vx[:dx,:dx]
    Q[0] += lmd_n * np.identity(ds+dx)
#    print "eps,lmd_n,Q=",eps,lmd_n,Q
    Q[1] = np.zeros((ds+dx, ds+dx))
    Q[1][:ds,:ds] = (1-eps)*Vs
    Q[1][ds:ds+dx,ds:ds+dx] = -eps*Vx[:dx,:dx]
    q = [[], []]
    q[0] = np.concatenate((-vs,-vx)).reshape(-1,1)[:ds+dx,:]
    q[1] = np.zeros((ds+dx,1))
    c = np.array([0,0])
    np.set_printoptions(threshold='nan')
    sol_cqp, val_cqp = fairopt.solveCQP( Q, q, c, eps, core = 2 )
#    print "sol=",sol_cqp
    return sol_cqp
    #sol, val, valfst = fairopt.qcqpSDP_mosek( Q, q, c )
    #return sol.flatten()
  def Fair_Prediction_Optimization_Correlated(self, eps, S, X, vs, vx):
    #calculate alpha, beta as an optimization problem
    # min  a.T Vx a + b.T Vy b - a.T E[sy] - b.t E[xy]
    # s.t. (1-eps)a.T Vs a - eps b.T Vx b
    import fairopt
    if not (0<=eps<=1):
      print ("Error: eps must be in [0,1]");sys.exit(0)
    Vs = np.array(np.cov(S.T)).reshape((1,-1))
    Vsx = np.matmul(S.T,X)/len(S)
    #xxxxs = np.matmul(X, np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T),S))
    #VsD = np.array(np.cov((S-xxxxs).T)).reshape((1,-1))
    Vx = np.cov(X.T)
    #print "Vs,Vx,Vsx norm=",np.linalg.norm(Vs),np.linalg.norm(Vx),np.linalg.norm(Vsx)
    #print vs.shape,vx.shape
    ds,dx = len(vs),len(vx)
    Q = [[],[]]
    Q[0] = np.zeros((ds+dx, ds+dx))
    Q[0][:ds,:ds] = Vs
    Q[0][ds:ds+dx,ds:ds+dx] = Vx
    Q[0][:ds,ds:] = Vsx
    Q[0][ds:,:ds] = Vsx.T
    #print "Q[0] svd =",np.linalg.svd(Q[0])[1]
    Q[1] = np.zeros((ds+dx, ds+dx))
    Q[1][:ds,:ds] -= eps*Vs
    Q[1][:ds,:ds] += Vs
    Q[1][:ds,ds:] += (0.5-eps)*Vsx
    Q[1][ds:,:ds] += (0.5-eps)*Vsx.T
    Q[1][ds:ds+dx,ds:ds+dx] -= eps*Vx[:dx,:dx]
    q = [[], []]
    q[0] = np.concatenate((-vs,-vx)).reshape(-1,1)[:ds+dx,:]
    q[1] = np.zeros((ds+dx,1))
    c = np.array([0,0])
    np.set_printoptions(threshold='nan')
    sol_cqp, val_cqp = fairopt.solveCQP( Q, q, c, eps, core=2 )
    return sol_cqp  
  def Fair_Prediction_Kernel_Optimization(self, eps, lmd, Ks, Kx, S, X, Y):
    #kernel version (optimization)
    import fairopt
    if not (0<=eps<=1):
      print ("Error: eps must be in [0,1]");sys.exit(0)
    Q = [[],[]]
    q = [[],[]]
    n = len(Y)
    Q[0] = np.zeros((2*n, 2*n))
    Q[1] = np.zeros((2*n, 2*n))
    q[0] = np.zeros((2*n, 1))
    q[1] = np.zeros((2*n, 1))
    c = np.array([0,0])
    #print "preparing data";sys.stdout.flush()
    Ks2, Kx2 = np.matmul(Ks,Ks), np.matmul(Kx,Kx)
    In = np.eye(n, n)
    #print "Ks2,Kx2 norm=",np.linalg.norm(Ks2),np.linalg.norm(Kx2)
    #print "matrix prepared";sys.stdout.flush()
    for i in range(n):
      for j in range(n):
        q[0][j] -= Y[i] * Ks[i,j]
        q[0][j+n] -= Y[i] * Kx[i,j]
        Q[0][i,j] += Ks2[i,j]
        Q[0][i+n,j+n] += Kx2[i,j]
        Q[0][i,j] += lmd * Ks[i,j] #Ks[i,j]
        Q[0][i+n,j+n] += lmd * Kx[i,j]
    for i in range(n):
      for j in range(n):
        Q[1][i,j] += (1.0 - eps) * (Ks2[i,j] + lmd * Ks[i,j]) #note: lambda * In (for making  Q PSD)
        Q[1][i+n,j+n] += - eps * (Kx2[i,j] + lmd * Kx[i,j])
    def mysvd(A): #note: np.svd sometimes fails
      try:
        X, Y, Z = np.linalg.svd(A)
      except:
        try:
          A2 = np.dot(A.T, A)
          X2, Y2, Z = np.linalg.svd(A2)
          Y = np.sqrt(Y2)
          X = np.dot(A, Z.T); X = np.dot(X, np.linalg.inv(np.diag(Y)))
        except:
          try:
            print ("svd try2")
            w,v = np.linalg.eigh(np.dot(A.T ,A))
            w = w[::-1]; v = v[:,::-1]
            Y = np.sqrt(w)
            X = np.dot(A,v); X = np.dot(X,np.diag(Y**(-1))); Z = v.T
          except:
            try:
              print ("svd try3")
              n = A.shape[0]
              Ad = A + np.identity(n)*0.01
              A2 = np.dot(Ad.T, Ad)
              X2, Y2, Z = np.linalg.svd(A2)
              Y = np.sqrt(Y2)
              X = np.dot(Ad, Z.T); X = np.dot(X, np.linalg.inv(np.diag(Y)))
            except:
              print ("svd try4")
              n = A.shape[0]
              Ad = A + np.identity(n)*0.1
              A2 = np.dot(Ad.T, Ad)
              X2, Y2, Z = np.linalg.svd(A2)
              Y = np.sqrt(Y2)
              X = np.dot(Ad, Z.T); X = np.dot(X, np.linalg.inv(np.diag(Y)))
      return X,Y,Z
    print ("calling optimizer");sys.stdout.flush()
    try:
      sol, val_cqp = fairopt.solveCQP( Q, q, c, eps )
    except:
      print ("Warning: cqp failed. trying SDP")
      sol, val_sdp, val_sdp_fst = fairopt.qcqpSDP_mosek( Q, q, c )
      sol = sol.flatten()
    return sol
  def subsample_from_levscore(self, ks, kx, S, X, gamma, p_ratio, ratio):
    import leveragescore
    n = X.shape[0]
    p = int(n*p_ratio)
    p_ids = np.random.choice(range(n), p, False)
    C = np.zeros((n, p))
    W = np.zeros((p, p))
    for i in range(n):
      for j in range(p):
        C[i,j] = kx(X[i], X[p_ids[j]])
    for i in range(p):
      for j in range(p):
        W[i,j] = kx(X[p_ids[i]], X[p_ids[j]])
    W = W + 0.01 * np.identity(p) #for numerical stability
    B = np.matmul(C, np.linalg.pinv(np.linalg.cholesky(W)))
    lmd = gamma/2.0
    BtBpNlI_inv = np.linalg.inv(np.matmul(B.T,B)+n*lmd*np.identity(p))
    lx = []
    for i in range(n):
      l = np.matmul(np.matmul(B[i,:].T, BtBpNlI_inv),B[i,:])
      lx.append(l)
    #ls = leveragescore.leverage_score(Ks, gamma/2.0)
    #lx = leveragescore.leverage_score(Kx, gamma/2.0)
    dx_eff = sum(lx)
    return np.random.choice(range(n), int(n*ratio), False, lx/dx_eff)
  def EpsFair_Prediction(self, filename, eps, hparams, avails, p):
    is_kernel = p.kernel
    rff = p.rff
    lmd = hparams["lmd"]
    gamma = hparams["gamma"]
    if is_kernel and rff:
      print ("Error: either rff or kernel needs to be false");sys.exit()

    NTrain = len(self.trainX1)
    transform_s = p.nonlinears
    trainS, trainX1 = copy.deepcopy(self.trainS), copy.deepcopy(self.trainX1)

    if rff:
      if not transform_s:
        print ("random fourier feature")
      else:
        print ("random fourier feature (full ns)")
      ds_new = len(self.trainS[0])*10 
      dx_new =  len(self.trainX1[0])*10 
      sys.stdout.flush()
      if transform_s:
        sampler_s = kernel_approximation.RBFSampler(gamma = hparams["gamma"], n_components = ds_new)
        sampler_s.fit(trainS)
        trainS = sampler_s.transform(trainS)
      sampler_x = kernel_approximation.RBFSampler(gamma = hparams["gamma"], n_components = dx_new)
      sampler_x.fit(self.trainX1)
      trainX1 = sampler_x.transform(self.trainX1)
    else:
      trainX1 = self.trainX1
    S_std = [np.std(trainS[:,j]) for j in range(len(trainS[0]))]
    for j in range(len(trainS[0])):
      trainS[:,j] = trainS[:,j] / S_std[j]
    X1_size = len(trainX1[0])
    lr1 = [] #stage1 regressor/classifiers
    for i in range(X1_size):
      lr1.append(copy.deepcopy(self.fstStageRegressor))
      lr1[-1].set_params(alpha = lmd)
    trainS_X2 = np.c_[trainS, self.trainX2] #use S and X2 (not used currently...)
    X1_hat_tmp = []
    self.train_X1_resid(trainX1, trainS_X2, trainS, lr1, use_X2 = False)
    train_X1_resid = self.get_X1_resid(lr1, trainX1, trainS, trainS_X2, use_X2 = False)
    X1_std = [np.std(train_X1_resid[:,j]) for j in range(len(self.trainX1[0]))]
    for j in range(len(self.trainX1[0])):
      train_X1_resid[:,j] = train_X1_resid[:,j] / X1_std[j]
    trainX_rn = copy.deepcopy(train_X1_resid) #self.trainX1

    for i in range(trainX_rn.shape[1]):
      trainX_rn[:,i] = trainX_rn[:,i] - np.mean(train_X1_resid[:,i])
    trainS_n = copy.deepcopy(trainS)
    for j in range(len(trainS[0])):
      trainS_n[:,j] = trainS[:,j] - np.mean(trainS[:,j])
    trainY_n = self.trainY - np.mean(self.trainY)

    validS, validX1, validX2, validY = self.getValidationData()
    if rff:
      validX1 = sampler_x.transform(validX1)
      if transform_s:
        validS = sampler_s.transform(validS)
    for j in range(len(trainS[0])):
      validS[:,j] = validS[:,j] / S_std[j]
    validS_X2 = np.c_[validS, validX2] #use S and X2
    valid_X1_resid = self.get_X1_resid(lr1, validX1, validS, validS_X2, use_X2 = False)
    valid_n = len(validS)
    for j in range(len(self.trainX1[0])):
      valid_X1_resid[:,j] = valid_X1_resid[:,j] / X1_std[j]
    validX_rn = valid_X1_resid #self.trainX1
    for i in range(train_X1_resid.shape[1]):
      validX_rn[:,i] = validX_rn[:,i] - np.mean(train_X1_resid[:,i])
    validS_n = copy.deepcopy(validS)
    for j in range(len(trainS[0])):
      validS_n[:,j] = validS[:,j] - np.mean(trainS[:,j])
    validY_n = validY - np.mean(self.trainY)
    testS, testX1, testX2, testY = self.getPredictData()
    if rff:
      testX1 = sampler_x.transform(testX1)
      if transform_s:
        testS = sampler_s.transform(testS)
    for j in range(len(trainS[0])):
      testS[:,j] = testS[:,j] / S_std[j]
    testS_X2 = np.c_[testS, testX2] #use S and X2
    test_X1_resid = self.get_X1_resid(lr1, testX1, testS, testS_X2, use_X2 = False)
    for j in range(len(self.trainX1[0])):
      test_X1_resid[:,j] = test_X1_resid[:,j] / X1_std[j]
    test_n = len(testS)
    testX_rn = test_X1_resid #self.trainX1
    for i in range(train_X1_resid.shape[1]):
      testX_rn[:,i] = testX_rn[:,i] - np.mean(train_X1_resid[:,i])
    testS_n = copy.deepcopy(testS)
    for j in range(len(trainS[0])):
      testS_n[:,j] = testS[:,j] - np.mean(trainS[:,j])
    testY_n = testY - np.mean(self.trainY)

    Vs = np.cov(trainS_n.T)
    Vx = np.cov(trainX_rn.T)
    vs = np.matmul(trainS_n.T, trainY_n)/NTrain
    vx = np.matmul(trainX_rn.T, trainY_n)/NTrain
    def linearKernel(): return ( lambda x,y:np.dot(x,y) )
    def rbfKernel(gamma): return ( lambda x,y:math.exp(-gamma*np.inner(x-y,x-y)) )
    def polyKernel(gamma): return ( lambda x,y:(gamma*np.inner(x,y)+1.0)**3 )
    if not is_kernel: #Linear
      sol = self.Fair_Prediction_Optimization(eps, lmd/NTrain, Vs, Vx, vs, vx) #main optimization
      train_S_X1_resid = np.c_[trainS_n, trainX_rn]
      trainYhat = [np.dot(sol,train_S_X1_resid[i]) for i in range(NTrain)]
      valid_S_X1_resid = np.c_[validS_n, validX_rn]
      validYhat = [np.dot(sol,valid_S_X1_resid[i]) for i in range(valid_n)]
      test_S_X1_resid = np.c_[testS_n, testX_rn]
      testYhat = [np.dot(sol,test_S_X1_resid[i]) for i in range(test_n)]

      result_train = Result(self.trainY, trainYhat + np.mean(self.trainY), self.trainS, avails) 
      result_valid = Result(validY, validYhat + np.mean(self.trainY), self.validS, avails) 
      result_test  = Result(testY, testYhat + np.mean(self.trainY), self.testS, avails) 
    else: #Kernel
      ks = rbfKernel(gamma)
      kx = rbfKernel(gamma)
      n = NTrain
      subsampling_ratio = 1.0 #0.1
      n_sub = int(n*subsampling_ratio)
      if subsampling_ratio < 1.0:
        sample_ids = self.subsample_from_levscore(ks, kx, trainS_n, trainX_rn, gamma, 0.05, subsampling_ratio)
      else:
        sample_ids = [i for i in range(n)]

      Ks, Kx = np.zeros((n_sub, n_sub)), np.zeros((n_sub, n_sub))
      trainS_n_sub = trainS_n[sample_ids]
      trainX_rn_sub = trainX_rn[sample_ids]
      trainY_n_sub = trainY_n[sample_ids]
      for i in range(n_sub):
        for j in range(n_sub):
          Ks[i,j], Kx[i,j] = ks(trainS_n_sub[i],trainS_n_sub[j]), kx(trainX_rn_sub[i],trainX_rn_sub[j])
      sol = self.Fair_Prediction_Kernel_Optimization(eps, lmd, Ks, Kx, trainS_n_sub, trainX_rn_sub, trainY_n_sub)

      trainYhat = np.matmul(Ks, sol[:n_sub]) + np.matmul(Kx, sol[n_sub:])
      valid_n = len(validS)
      Ks_valid, Kx_valid = np.zeros((valid_n, n_sub)), np.zeros((valid_n, n_sub))
      for i in range(valid_n):
        for j in range(n_sub):
          Ks_valid[i,j], Kx_valid[i,j]\
            = ks(validS_n[i],trainS_n_sub[j]), kx(validX_rn[i],trainX_rn_sub[j])
      validYhat = np.matmul(Ks_valid, sol[:n_sub]) + np.matmul(Kx_valid, sol[n_sub:])
      test_n = len(testS)
      Ks_test, Kx_test = np.zeros((test_n, n_sub)), np.zeros((test_n, n_sub))
      for i in range(test_n):
        for j in range(n_sub):
          Ks_test[i,j], Kx_test[i,j]\
            = ks(testS_n[i],trainS_n_sub[j]), kx(testX_rn[i],trainX_rn_sub[j])
      testYhat = np.matmul(Ks_test, sol[:n_sub]) + np.matmul(Kx_test, sol[n_sub:])

      result_train = Result(self.trainY[sample_ids], trainYhat + np.mean(self.trainY[sample_ids]), self.trainS[sample_ids], avails) 
      result_valid = Result(validY, validYhat + np.mean(self.trainY), validS, avails) 
      result_test  = Result(testY, testYhat + np.mean(self.trainY), testS, avails) 

    return result_train, result_valid, result_test



