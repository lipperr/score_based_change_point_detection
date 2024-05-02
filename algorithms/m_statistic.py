# An online change point detection procedure based on M-statistic from the paper 
# "M-statistic for kernel change-point detection” (NIPS, 2015)
# by S. Li, Y. Xie, H. Dai, and L. Song

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import math

def mmd_squared(X, Y, sigma):
    
    # Sample size
    n = X.shape[0]
    
    # Compute pairwise distances
    xx_dist = pairwise_distances(X)
    xy_dist = pairwise_distances(X, Y)
    yy_dist = pairwise_distances(Y)
    
    # Compute kernel matrices
    xx_kernel = np.exp(-0.5 * (xx_dist / sigma)**2) - np.identity(n)
    xy_kernel = np.exp(-0.5 * (xy_dist / sigma)**2) - np.identity(n)
    yy_kernel = np.exp(-0.5 * (yy_dist / sigma)**2) - np.identity(n)
    
    # Compute the U-statistic
    u_stat = (np.sum(xx_kernel) - 2 * np.sum(xy_kernel) + np.sum(yy_kernel)) / n / (n - 1)
    
    return u_stat
    
    
def h_squared(X, sigma):
    
    # Sample size
    n = X.shape[0]
    
    # Divide the array into four equal parts
    n_max = 4 * (n // 4)
    X_1 = X[0:n_max:4]
    X_2 = X[1:n_max:4]
    X_3 = X[2:n_max:4]
    X_4 = X[3:n_max:4]
    
    K_12 = np.exp(-0.5 * (np.linalg.norm(X_1 - X_2, axis=1) / sigma)**2)
    K_13 = np.exp(-0.5 * (np.linalg.norm(X_1 - X_3, axis=1) / sigma)**2)
    K_24 = np.exp(-0.5 * (np.linalg.norm(X_2 - X_4, axis=1) / sigma)**2)
    K_34 = np.exp(-0.5 * (np.linalg.norm(X_3 - X_4, axis=1) / sigma)**2)
    
    return np.mean((K_12 - K_13 - K_24 + K_34)**2)


# Compute the first term in the variance estimate
def h_cov(X, sigma):
    
    # Sample size
    n = X.shape[0]
    
    # Divide the array into six equal parts
    n_max = 6 * (n // 6)
    X_1 = X[0:n_max:6]
    X_2 = X[1:n_max:6]
    X_3 = X[2:n_max:6]
    X_4 = X[3:n_max:6]
    X_5 = X[4:n_max:6]
    X_6 = X[5:n_max:6]
    
    K_12 = np.exp(-0.5 * (np.linalg.norm(X_1 - X_2, axis=1) / sigma)**2)
    K_13 = np.exp(-0.5 * (np.linalg.norm(X_1 - X_3, axis=1) / sigma)**2)
    K_24 = np.exp(-0.5 * (np.linalg.norm(X_2 - X_4, axis=1) / sigma)**2)
    K_34 = np.exp(-0.5 * (np.linalg.norm(X_3 - X_4, axis=1) / sigma)**2)
    
    K_56 = np.exp(-0.5 * (np.linalg.norm(X_5 - X_6, axis=1) / sigma)**2)
    K_53 = np.exp(-0.5 * (np.linalg.norm(X_5 - X_3, axis=1) / sigma)**2)
    K_64 = np.exp(-0.5 * (np.linalg.norm(X_6 - X_4, axis=1) / sigma)**2)
    K_34 = np.exp(-0.5 * (np.linalg.norm(X_3 - X_4, axis=1) / sigma)**2)
    
    # Compute the second term in the variance estimate
    h_1234 = K_12 - K_13 - K_24 + K_34
    h_5634 = K_56 - K_53 - K_64 + K_34
    
    return np.mean((h_1234 - np.mean(h_1234)) * (h_5634 - np.mean(h_5634)))


# Variance estimate under the null hypothesis
def estimate_variance(X, window_size, sigma):
    
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
        
    # Sample size
    n = X.shape[0]
    
    # Compute the first term in the variance estimate
    h2 = h_squared(X, sigma) 
    
    # Compute the second term in the variance estimate
    h_c = h_cov(X, sigma)
    
    # Variance estimate
    var = 2 * (h2 + h_c) / window_size / (window_size - 1)
        
    return np.maximum(var, 1e-5)


# Compute MMD test statistic
def compute_test_stat_mmd(X, window_size=10, sigma=0.1, threshold=math.inf):
    
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
        X_te = X[t-b:t, :]
        # Reference sample
        X_re = X[t-2*b:t-b, :]
        
        MMD_2 = mmd_squared(X_re, X_te, sigma)
        var = estimate_variance(X_re, window_size, sigma)
        
        T[t] = MMD_2 / np.sqrt(var)
        
        if T[t] > threshold:
        
            stopping_time = t
            break
    
    # Array of test statistics
    if stopping_time != -1:
        T = T[:stopping_time + 1]
    
    return T, stopping_time