# -*- coding: utf-8 -*-

import sys, math
import numpy as np
import scipy
import scipy.optimize as optim #l-bfgs
import random 
import copy

#  return np.array(S), np.array(X1), np.array(X2), np.array(y)
def save_file(S, X1, X2, y, filename):
  l = len(y)
  if S.shape[0] != l or X1.shape[0] != l or X2.shape[0] != l:
    print("Error: size does not match in save_file")
    sys.exit()

  #header
  astr = ""
  for i in range(S.shape[1]):
    astr+="s"+[i]+","
  for i in range(X1.shape[1]):
    astr+="X1_"+[i]+","
  for i in range(X2.shape[1]):
    astr+="X2_"+[i]+","
  astr += "y" 
  print(astr)

  #datapoints
  for i in range(l):
    print( ",".join(map(str, S[i,:]))+","+",".join(map(str, X1[i,:]))+","+",".join(map(str, X2[i,:]))+","+y[i] )

def batchFileProcessing(datasets, filenames, adir="data_preprocessed"):
  for i in range(len(datasets)):
    S,X1,X2,y = datasets[i]
    filename = filenames[i]
    base_file = os.path.join(adir, filename)
    save_file(S, X1, X2, y, base_file)
    out_file_bba = os.path.join(adir, "bba_"+filename)
    cmd = 'BlackBoxAuditing-repair '+base_file+' '+out_file_bba+' 0.5 True p="s1"'
    subprocess.call(cmd)

#learning fair representations
#K = # of prototypes
#Note, S,X1,X2,y = train
def lfr(S, X1, X2, Y, X1test, X2test, K):
  #print "S_vars=",set([S[i,0] for i in range(S.shape[0])])
  if S.shape[1] != 1:
    print ("Error in LFR: S is not univariate");sys.exit()
  elif len(set([S[i,0] for i in range(S.shape[0])])) != 2:
    print ("Error in LFR: S is not binary");sys.exit()
  X = np.c_[X1, X2]
  N, D = X.shape
  Y = copy.deepcopy(Y)
  for i in range(N):
    if Y[i]==-1:
      Y[i] = 0 #{0,1}
  #normalization
  D1, D2 = X1.shape[1], X2.shape[1]
  stds = [np.std(X[:,j]) for j in range(D)]
  for i in range(N):
    for j in range(D):
      if stds[j]>0.001:
        X[i,j] = X[i,j] / stds[j]
  #optimization
  A_X, A_Y, A_Z = 0.01, 1, 50 #follows supplementary material of Zemel et al. 
  #A_X, A_Y, A_Z = 0.0001, 0.1, 1000 #follows https://github.com/zjelveh/learning-fair-representations/
  #A_X, A_Y, A_Z = 0.01, 1, 50
  def dist(x1, x2): #l2-dist
    d = len(x1)
    #print "x1=",x1
    #print "x2=",x2
    s = 0
    for t in xrange(d):
      s += (x1[t]-x2[t])**2
    return s**0.5
  def distl1(x1, x2): #l1-dist
    d = len(x1)
    s = 0
    for t in xrange(d):
      s += abs(x1[t]-x2[t])
    return s
  def Mnk(vk, Xi):
    pk = np.zeros(K)
    for j in range(K):
      pk[j] = math.exp(-dist(vk[j], Xi))
    #print ("pk=",pk)
    s_pk = sum(pk)
    pk = pk/s_pk
    return pk
  def LFR_optim(params, S, X, Y):
    S_vars = list(set([S[i,0] for i in range(S.shape[0])]))
    vk = np.zeros([K, D])
    for i in range(K):
      for j in range(D):
        vk[i,j] = params[i*D+j]
    wk = params[K*D:K*(D+1)]
    Mkp, Mkm = np.zeros(K), np.zeros(K)
    cp, cm = 0, 0
    Mnk_vec = np.zeros([N,K])
    for i in xrange(N):
      Xi = X[i,:]
      Mnk_vec[i] = Mnk(vk, Xi)
      if S[i,0]==S_vars[0]:
        Mkp += Mnk_vec[i]
        cp += 1
      elif S[i,0]==S_vars[1]:
        Mkm += Mnk_vec[i]
        cm += 1
    Mkp /= cp
    Mkm /= cm
    L_Z = distl1(Mkp, Mkm)
    #L_Z /= K 
    L_X = 0
    for i in xrange(N):
      pk = Mnk_vec[i]
      hatXi = np.zeros(D)
      hatXi += np.matmul(vk.T, pk)
      L_X += dist(X[i], hatXi)**2 
      #print "X[i],hatXi=",Xi,hatXi
    #L_X /= N
    L_Y = 0
    yhat = np.zeros(N)
    for i in xrange(N):
      pk = Mnk_vec[i]
      #print "pk,wk[j]=",pk,wk[j]
      r = np.inner(pk, wk)
      r = min(max(0.0000001, r), 0.9999999)
      #print "Y[i]=",Y[i]
      #print "pk,wk,r=",pk,wk,r
      yhat[i] = r
      L_Y -= Y[i]*math.log(r) + (1-Y[i])*math.log(1-r)
    
    #L_Y /= N
    L = A_Z * L_Z + A_X * L_X + A_Y * L_Y
    #print "vk=",vk
    #print "yhat (avg,std),loss=",np.mean(yhat),np.std(yhat),A_Z*L_Z,A_X*L_X,A_Y*L_Y,L
    return L
  #optimize params (vk (K x D), wk (K))
  bnd = [(None, None) for i in range(K*D)] + [(0,1) for i in range(K)]
  params = optim.fmin_l_bfgs_b(LFR_optim, x0=np.random.rand(K*(D+1)), epsilon=1e-5, 
                          args=(S, X, Y), 
                          bounds = bnd, approx_grad=True, maxfun=10000, 
                          maxiter=20)
  vk = np.zeros([K, D])
  #print ("params=",params)
  for i in range(K):
    for j in range(D):
      #print ("i,j=",i,j)
      vk[i,j] = params[0][i*D+j]
  wk = params[0][K*D:K*(D+1)]
  #allocate yhat in accordance w/ softmax (Eqn. (2) in Zemel+)
  X1_hat, X2_hat = np.zeros([N, D1]), np.zeros([N, D2])
  for i in range(N):
    pk = Mnk(vk, X[i])
    #r = np.inner(pk, wk)
    for k in range(K):
      for j in range(D1):
        X1_hat[i,j] += np.inner(vk[:,j], pk)
      for j in range(D2):
        X2_hat[i,j] += np.inner(vk[:,j+D1], pk)
  Nt = X1test.shape[0]
  X1test_hat, X2test_hat = np.zeros([Nt, D1]), np.zeros([Nt, D2])
  for i in range(Nt):
    pk = Mnk(vk, X[i])
    #r = sum([pk[j]*wk[j] for j in range(K)])
    for k in range(K):
      for j in range(D1):
        X1test_hat[i,j] += np.inner(vk[:,j], pk)
      for j in range(D2):
        X2test_hat[i,j] += np.inner(vk[:,j+D1], pk)
  return X1_hat, X2_hat, X1test_hat, X2test_hat

#Certifying and Removing Disparate Impact
def quantile(S, X1, X2):
  if S.shape[1] != 1:
    print ("Error in QT: S is not univariate");sys.exit()
  elif len(set([S[i,0] for i in range(S.shape[0])])) != 2:
    print ("Error in QT: S is not binary");sys.exit() 
  S_vars = list(set([S[i,0] for i in range(S.shape[0])]))
  X = np.c_[X1, X2]
  N, D = X.shape
  X_hat = copy.deepcopy(X)
  D1, D2 = X1.shape[1], X2.shape[1]
  for d in range(D):
    #d_vars_size = len(set(X[:,d]))
    #if d_vars_size <=2:continue
    temp_s0, temp_s1 = [], []
    for i in range(N):
      if S[i,0]==S_vars[0]:
        i_s = len(temp_s0)
        temp_s0.append((X[i,d],i,i_s))
      elif S[i,0]==S_vars[1]:
        i_s = len(temp_s1)
        temp_s1.append((X[i,d],i,i_s))
    temp_s0 = sorted(temp_s0)
    temp_s1 = sorted(temp_s1)
    for (v, i, i_s) in temp_s0:
      a_i = int(i_s*len(temp_s1)/float(len(temp_s0)))
      #print "a_i,len(temp_s0),len(temp_s1)=",a_i,len(temp_s0),len(temp_s1)
      X_hat[i,d] = (v + temp_s1[a_i][0])/2.
    for (v, i, i_s) in temp_s1:
      a_i = int(i_s*len(temp_s0)/float(len(temp_s1)))
      X_hat[i,d] = (v + temp_s0[a_i][0])/2.
  X1_hat = X_hat[:,:D1]
  X2_hat = X_hat[:,D1:]

  return X1_hat, X2_hat
  
#Optimized Pre-Processing for Discrimination Prevention
#https://github.com/fair-preprocessing/nips2017/
def calmon(S, X1, X2, Y, X1test, X2test, K):
  import mosek
  X = np.c_[X1, X2]
  N, D = X.shape
  Xtest = np.c_[X1test, X2test]
  Nt = Xtest.shape[0]
  Nall = N + Nt
  Xall = np.concatenate((X, Xtest), axis=0)
  Xall_new = copy.deepcopy(Xall)
  #supports = []
  for j in range(D):
    support = set(Xall[:,j])
    if support > 2: #divide into 3 regions
      lq, hq = np.percentile(Xall[:,j],100/3.), np.percentile(Xall[:,j],200/3.)
      print ("feature",j,"quantized: ")
      for i in range(Nall):
        if Xall[i,j]>hq:
          Xall_new[i,j]=2
        elif Xall[i,j]>lq:
          Xall_new[i,j]=1
        else:
          Xall_new[i,j]=0
  X_new, Xtest_new = Xall_new[:N,:], Xall_new[N:,:]
  #objective
  #constraint
  #optimization

