# KLIEP algorithm for sequential change point detection from the papers 
# “Direct importance estimation for covariate shift adaptation” (Annals of the Institute of Statistical Mathematics, 2008)
# by M. Sugiyama, T. Suzuki, S. Nakajima, H. Kashima, P. von Bunau, and M. Kawanabe
# and
# "Change-point detection in time-series data by relative density-ratio estimation" (Neural Networks, 2013)
# by S. Liu, M. Yamada, N. Collier, and M. Sugiyama

import numpy as np
import cvxpy as cvx
from sklearn.metrics.pairwise import pairwise_distances
import math


def kliep(X_te, X_re, sigma):
    
    # Test sample size
    n_te = X_te.shape[0]
    # Reference sample size
    n_re = X_re.shape[0]
    
    # Compute pairwise distances
    te_te_dist = pairwise_distances(X_te)
    re_te_dist = pairwise_distances(X_re, X_te)
    
    # Compute kernel matrices
    te_te_kernel = np.exp(-0.5 * (te_te_dist / sigma)**2)
    re_te_kernel = np.exp(-0.5 * (re_te_dist / sigma)**2)
    
    # Initialize a vector of coefficients
    theta = cvx.Variable(n_te)
    
    # Objective
    obj = cvx.Maximize(cvx.sum(cvx.log(te_te_kernel @ theta)))
    
    # Constraints
    constraints = [cvx.sum(re_te_kernel @ theta) == n_re, theta >= 0]
    
    # Problem
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver='SCS', eps=1e-2)
    
    return obj.value


def compute_test_stat_kliep(X, window_size=10, sigma=0.1, threshold=math.inf):
    
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    
    # More convenient notation
    b = window_size
    
    # Sample size
    n = X.shape[0]
    
    # Initialization
    T = np.zeros(n)
    stopping_time = -1
    
    for t in range(2*b + 1, n):
        
        # Test sample
        X_te = X[t-b:t]
        # Reference sample
        X_re = X[t-2*b:t-b]
        
        T[t] = kliep(X_te, X_re, sigma)
        
        if T[t] > threshold:
            stopping_time = t
            break
    
    # Array of test statistics
    if stopping_time != -1:
        T = T[:stopping_time + 1]
    
    return T, stopping_time