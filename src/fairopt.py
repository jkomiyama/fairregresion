#!/usr/bin/python
# coding:utf-8

#import picos as pic
#import cvxopt as cvx
import gurobipy as grb
import numpy as np
import sys,time

def solveCQP(Q, q, c, epsVal, core=-1): 
  #Solve 1QCQP whose objective function is convex.
  #min  x'*setQ{1}*x+2*setq{1}'*x+setc{1}
  #s.t. x'*setQ{2}*x+2*setq{2}'*x+setc{2} <= 0
  k = len(Q)
  n = len(Q[0])
  lams = epsVal
  tesQ = Q[1]+lams*Q[0]
  #problem(1).f = f;
  model = grb.Model("qcp")
  #x = np.zeros((n+1))
  var_x = []
  for i in range(n+1):
    var_x.append( model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY, name = "x"+str(i) ) )
  #print var_x
  obj = var_x[n]
  model.setObjective(obj, grb.GRB.MINIMIZE)
  model.setParam('OutputFlag', 0)
  if core != -1:
    #model.params.Threads = core
    model.params.method = 1 #0/1 primal/dual

  bigQ = np.zeros((n+1,n+1))
  bigQ[0:n,0:n] = tesQ
  #problem(1).qc(1).Q = bigQ;
  bigq = np.zeros((n+1,1))
  bigq[0:n] = q[1]+lams*q[0]
  bigq[n] = -lams/2 #0?
  #probl(em(1).qc(1).a = 2*bigq;
  #problem(1).qc(1).rhs = -c{2}-lams*c{1};
  constr1 = 0
  for i in range(n+1):
    for j in range(n+1):
      constr1 += var_x[i]*var_x[j]*bigQ[i,j]
  for i in range(n+1):
    constr1 += 2*var_x[i]*bigq[i]
    #constr1 += grb.abs_(var_x[i])
  model.addConstr(constr1 <= - lams * c[0] - c[1])

  bigQ2 = np.zeros((n+1,n+1))
  bigQ2[0:n,0:n] = Q[0]
  #problem(1).qc(2).Q = bigQ;
  bigq2 = np.zeros((n+1,1))
  bigq2[0:n] = q[0]
  bigq2[n] = -.5
  #poblem(1).qc(2).a = 2*bigq;
  #problem(1).qc(2).rhs = -c{1};
  constr2 = 0
  for i in range(n+1):
    for j in range(n+1):
      constr2 += var_x[i]*var_x[j]*bigQ2[i,j]
  for i in range(n+1):
    constr2 += 2*var_x[i]*bigq2[i]
  model.addConstr(constr2 <= - c[1])

  model.setParam("DualReductions", 0) #c.f. http://www.gurobi.com/documentation/6.5/refman/optimization_status_codes.html
  model.optimize()
  print ("gurobi statuscode",model.status," (2 is optimal)")
  if model.status != grb.GRB.Status.OPTIMAL:
    print ("Warning: solution model might not be optimal")
  if model.status == grb.GRB.Status.INF_OR_UNBD:
    print ("reoptimizing...")
    model.optimize()
  xs = [0 for i in range(n)] #Note: not n+1
  for v in model.getVars():
    #print('%s %g' % (v.varName, v.x))
    i = int(v.varName[1:])
    #print i
    if i<n:
      xs[i] = v.x 
  print ("CQP sol norm=",np.linalg.norm(xs))
  #print "CQP sol",xs
  return xs, model.objVal
  #[sol,optval,exitflag] = cplexqcp(problem);

def qcqpSDP_mosek(Q, q, c, core = -1):
  import mosek #need mosek lib to call this function
  if core != -1:
    mosek.MSK_IPAR_NUM_THREADS = core
  #Obtain the Lagrangean dual (SDP relaxation) of
  #   min  x' Q1 x+ 2q1' x +c1
  #   s.t. x' Qi x+ 2qi' x +ci <= 0, i=2,3,...
  #as
  #   max  y0 
  #   s.t. M0 + y0*cM + y1*M1 + y2*M2 >= 0
  #Add more valid constraints if rlt=1
  m = len(Q)-1 # num of constraints
  k = len(Q)   # num of matrices
  n = len(Q[0]) # size of a matrix
  #if size(lb,1) == n
  #    Aine = [Aine;eye(n);-eye(n)];
  #    bine = [bine;ub;-lb];
  O = np.zeros((n+1,m));
  M = []
  Msize = 1+n+m
  M0_mat = np.zeros((Msize, Msize))
  M0_mat[0,0] = c[0]
  M0_mat[0:1,1:1+n] = q[0].T
  #print q[0].shape,q[0].T.shape
  M0_mat[1:1+n,0:1] = q[0]
  M0_mat[1:1+n,1:1+n] = Q[0]
  M.append(M0_mat) # [[c{1} q{1}';q{1} Q{1}] O; O' zeros(m) ] #M0 objective
  for i in range(1,k):
    Oi = np.zeros((m,m))
    Oi[i-1,i-1] = 1
    Mi_mat = np.zeros((Msize, Msize))
    Mi_mat[0,0] = c[i]
    Mi_mat[0:1,1:1+n] = q[i].T
    Mi_mat[1:1+n,0:1] = q[i]
    Mi_mat[1:1+n,1:1+n] = Q[i]
    Mi_mat[1+n:1+n+m,1+n:1+n+m] = Oi
    M.append(Mi_mat)
    #M{i} = [[c{i} q{i}';q{i} Q{i}] O; O' Oi ] #ok<*AGROW> todo

  cM = np.zeros((n+1+m,n+1+m))
  cM[0,0] = -1
  bt = np.zeros((k,1))  # coeff of the objective func in SDP
  bt[0] = 1
  ct = M[0]
  At = np.zeros( (n+1+m, n+1+m,k) )
  At[:,:,0] = -cM # y0
  for i in range(1,k):
      At[:,:,i] = -M[i]
  #K.s = size(M[0],1);

  #mosek code
  def streamprinter(msg):
    #sys.stdout.write(msg)
    #sys.stdout.flush()
    pass
  with mosek.Env() as env:
    with env.Task() as task:
      task.set_Stream(mosek.streamtype.log, streamprinter)
      task.appendbarvars((n+1+m,n+1+m))
      #denseall = [i for i in range(n+1+m)]*(n+1+m)
      ind_l, ind_r, At0v, At1v, ctv  = [], [], [], [], []
      for i in range(n+1+m):
        for j in range(i, n+1+m):
          ind_l.append(i)
          ind_r.append(j)
          At0v.append(At[i,j,0])
          At1v.append(At[i,j,1])
          ctv.append(ct[i,j])
      #print zip(ind_l,ind_r)
      Ap0 = task.appendsparsesymmat(n+1+m, ind_r, ind_l, At0v )
      Ap1 = task.appendsparsesymmat(n+1+m, ind_r, ind_l, At1v )
      cp  = task.appendsparsesymmat(n+1+m, ind_r, ind_l, ctv)
      #Ap0 = pic.new_param("A0",At[:,:,0])
      #Ap1 = pic.new_param("A1",At[:,:,1])
      numcon = 2
      blb = bt.T[0]
      bub = bt.T[0]
      #print "blb=",blb
      task.appendcons(numcon)
      for i in range(numcon):
        task.putconbound(i, mosek.boundkey.fx, blb[i], bub[i])
      #print At[:,:,0].shape
      task.putbaraij(0, 0, [Ap0], [1.0])
      task.putbaraij(1, 0, [Ap1], [1.0])
      #cp = pic.new_param("c",ct)
      #prob = pic.Problem()
      #xp = prob.add_variable("x", (n+1+m, n+1+m), vtype="symmetric")
      # prob.add_constraint(xp >> 0) #PSD constraint
      #prob.add_constraint(Ap0|xp == bp[0]) 
      #prob.add_constraint(Ap1|xp == bp[1]) 
      task.putbarcj(0, [cp], [1.0])
      #prob.set_objective("min", cp|xp)
      #print "x=",x.reshape((n+1+m,n+1+m))
      #[x, y, info] = sedumi(At,bt,ct,K,pars);
      
      task.putobjsense(mosek.objsense.minimize)
      task.optimize()
      #print task.solutionsummary(mosek.streamtype.msg)
      solsta = task.getsolsta(mosek.soltype.itr)
      if solsta not in [mosek.solsta.optimal, mosek.solsta.near_optimal]:
        print ("Warning: mosek might not be optimal")
        #sys.exit(0)
      #print solsta
      xx_L = [0 for i in range((n+1+m)*(n+1+m+1)/2)] #lower triangular
      task.getbarxj(mosek.soltype.itr, 0, xx_L)
      #print xx_L
      xx = np.zeros((n+1+m,n+1+m))
      l=0
      for j in range(n+1+m):
        for i in range(j, n+1+m):
          xx[i,j] = xx_L[l]
          xx[j,i] = xx_L[l]
          l = l + 1
      #print "Mosek mat norm=",np.linalg.norm(xx)
      svd = np.linalg.svd(xx)
      fstNorm = np.linalg.norm(svd[1][0])**0.5
      fstVector = (-fstNorm*svd[2][0][1:n+1]).reshape((-1,1))
      #print "mosek sol",-fstNorm*svd[2][0][1:n+1]
      #print "Mosek svd 1st norm=",fstNorm
      #x' Q1 x+ 2q1' x +c1
      #print Q[0].shape,fstVector.shape
      fstVal = np.matmul(np.matmul(fstVector.T, Q[0]),fstVector) + 2 * np.dot(q[0].T, fstVector) + c[0]
      return fstVector,task.getprimalobj(mosek.soltype.itr), fstVal[0,0]

if __name__ == '__main__':
  #Solve 1QCQP whose objective function is convex.
  #min x'*setQ{1}*x+2*setq{1}'*x+setc{1}
  #s.t. x'*setQ{2}*x+2*setq{2}'*x+setc{2} <= 0

  obj = 1 #if the objective is convex, set 1. Otherwise, set 0.

  ns = [10,30,100,300,1000,3000,10000]
  ex = int(sys.argv[1]) #trial num
  epsVal=0.1

  for n in ns:
    elapsed_time_cqp = 0
    elapsed_time_sdp = 0
    better_cqp = 0
    better_cqp_over_rank1sdp = 0
    better_sdp = 0
    sol_ratios_cqp = []
    sol_ratios_sdp = []
    sol_diffs = []
    fstval_ratios = []

    nx = n/10
    ny = n-n/10

    ro = [0 for x in range(ex)]
    rt = [0 for x in range(ex)]
    so = [0 for x in range(ex)]
    st = [0 for x in range(ex)]
    #ratio_val = [0 for x in range(ex)]
    #soldiff = [0 for x in range(ex)]

    for num in range(ex):
      setq = []
      setQ = []
      setc = []
      
      DATASIZE = n*10
      x = np.random.normal(size=(DATASIZE,nx))
      y = np.random.normal(size=(DATASIZE,ny))
      z = np.random.normal(size=DATASIZE)
      cx = np.random.random(size=nx)
      cy = np.random.random(size=ny)*0.01
      for i in range(DATASIZE):
        z[i] += np.dot(cx, x[i,:])+np.dot(cy, y[i,:])
      #z = z / np.std(z)
      vx = np.cov(x.T)
      vy = np.cov(y.T)
      q = np.zeros((n,1))
      for i in range(DATASIZE):
        q[:nx,0] -= x[i,:]*z[i]     
        q[nx:n,0] -= y[i,:]*z[i]     
      q = q/DATASIZE

      # objective func.
      setq.append( q ) #unifrnd(-5,5,n,1);
      setQ.append( np.zeros((n,n)) ) #zeros(n,n);
      #randX = np.random.random((nx,nx))*10-5 #unifrnd(-5,5,nx,nx);
      setQ[0][0:nx,0:nx] = vx #np.matmul(randX.T, randX)
      #randY = np.random.random((ny,ny))*10-5 #unifrnd(-5,5,ny,ny);
      setQ[0][nx:n,nx:n] = vy #np.matmul(randY.T, randY) #randY'*randY;
      setc.append(0)

      # nonconvex constraint
      setq.append( np.zeros((n,1)) )
      setQ.append( np.zeros((n,n)) )
      setQ[1][0:nx,0:nx] = (1-epsVal)*setQ[0][0:nx,0:nx]
      setQ[1][nx:n,nx:n] = -epsVal*setQ[0][nx:n,nx:n]
      setc.append(0)

      for i in range(2):
        for j in range(n):
          for l in range(j,n):
            setQ[i][j,l] = setQ[i][l,j]
      #print setQ[0].shape,setQ[1].shape,setq[0].shape,setq[1].shape

      start = time.time()
      sol_cqp, val_cqp = solveCQP( setQ, setq, setc, epsVal )
      elapsed_time_cqp += time.time() - start
      print ("N=",n,"CQP calculated v_cqp=",val_cqp)
      sys.stdout.flush()
      start = time.time()
      sol_sdp, val_sdp, val_sdp_fst = qcqpSDP_mosek( setQ, setq, setc )
      elapsed_time_sdp += time.time() - start
      print ("N=",n,"SDP calculated v_sdp=",val_sdp," v_sdp_fst=",val_sdp_fst)
      fstval_ratios.append(val_sdp_fst/val_sdp)
      if val_cqp < val_sdp * 1.001:
        better_cqp += 1
        sol_ratios_cqp.append(val_cqp/val_sdp)
      if abs(val_sdp - val_sdp_fst) / abs(val_sdp) > 0.001: #実際は制約をviolateしているfstのほうが値が良いことが多そう
        better_cqp_over_rank1sdp += 1
      if val_sdp < val_cqp * 1.01:
        better_sdp += 1
        sol_ratios_sdp.append(val_sdp/val_cqp)
      sys.stdout.flush()
      print ("N,val_cqp,val_sdp,val_ratio,sol_diff=",n,val_cqp,val_sdp,val_cqp/val_sdp,np.linalg.norm(sol_cqp-sol_sdp))
      sol_diffs.append( np.linalg.norm(sol_cqp-sol_sdp)/np.linalg.norm(sol_cqp) )
      sys.stdout.flush()

      #ro(1,num) = relopt;
      #rt(1,num) = reltime;
      #so(1,num) = sdpopt;
      ##st(1,num) = sdptime;
      #ratio_val(1,num)  = relopt/sdpopt;
      #soldiff(1,num)  =  norm(relsol_x-sdpsol_x);

    print ("elapsed_time_cqp=",elapsed_time_cqp)
    print ("elapsed_time_sdp=",elapsed_time_sdp)
    print ("better_cqp, better_sdp = ", better_cqp, better_sdp)
    print ("med(sol_ratios_cqp)=",np.median(sol_ratios_cqp))
    print ("med(sol_ratios_sdp)=",np.median(sol_ratios_sdp))
    print ("sol_diffs=",sorted(sol_diffs))
    print ("fstval_ratios=",sorted(fstval_ratios))
    print ("#features, #trial, RT (CQP), RT (SDP), diff obj, diff obj rank-1")
    print ("##",n,ex,elapsed_time_cqp,elapsed_time_sdp,better_cqp/float(n),better_cqp_over_rank1sdp/float(n))
#fprintf('********\nn= %d, numSolvedProb= %d\n',n, ex);
#fprintf('SDP:        val= %f,   time= %f\n',mean(so),mean(st));
#fprintf('proposed:   val= %f,   time= %f\n',mean(ro),mean(rt));
#fprintf('ave(proposed.val/SDP.val)= %f,  soldiff(SDP,CQP) = %f\n',mean(ratio_val),mean(soldiff));
#ratio_val
#soldiff






