# Online contrastive change point detection algorithm from the paper 
# "A Contrastive Approach to Online Change Point Detection" (arXiv:2206.10143)
# by A. Goldman, N. Puchkin, V. Shcherbakova, and U. Vinogradova

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
import math

torch.manual_seed(1)

#----------------------------------------------------------------------------------------------------------------
# A version of the algorithm with a linear class
#----------------------------------------------------------------------------------------------------------------

# Auxiliary function
# Computation of design matrix based on (1, x, x**2, ..., x**(p-1))
#
# X -- array of univariate observations
#
# p -- positive integer
#
def compute_design_poly(X, p):
    
    n = X.shape[0]
    Psi = np.power(np.outer(X, np.ones(p)), np.outer(np.ones(n), np.arange(p)))
    
    return Psi
    
    
# Auxiliary function
# Computation of design matrix
#
# X -- array of univariate observations
#
# p -- positive integer
#
def compute_design_Fourier(X, p):
    res = np.zeros((p, X.shape[0]))
    res[0] = np.ones(X.shape[0]) / np.sqrt(2)
    T = 1
    for i in range(1, p):
        if (i // 2 == 0):
            res[i] = np.sin(X * 2 * np.pi * i / T) / np.sqrt(T / 2)
        else:
            res[i] = np.cos(X * (2 * np.pi * i) / T) / np.sqrt(T / 2)
    return res.T


# Auxiliary function
# Computation of design matrix based on a linear class
#
# X -- array of multivariate observations
#
# p -- positive integer
#
def compute_design_multivariate(X):
    
    Psi = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    
    return Psi
    
    
# Auxiliary function
# Computation of the best fitting parameter theta
# under the hypothesis that tau is the true change point
#
# Psi -- (n x p)-array, design matrix 
#
# tau -- change point candidate
#
def compute_theta(Psi, tau):
    
    # Sample size
    t = Psi.shape[0]
    
    # Create "virtual" labels
    Y = np.append(np.ones(tau), -np.ones(t - tau))
    
    lr = LogisticRegression(penalty='none', fit_intercept=False, tol=1e-2,\
                            solver='lbfgs', class_weight='balanced', n_jobs=-1)
    lr.fit(Psi, Y)
    theta = (lr.coef_).reshape(-1)
    
    return theta
    
    
# Computation of the test statistic
#
# X -- array of univariate observations
#
# p -- positive integer (used for basis construction)
#
def compute_test_stat_linear(X, p, t_min=20, n_out_min=10, B=10, delta_max=150, design="poly", threshold=math.inf):
    
    # Sample size
    n = X.shape[0]
    
    # Compute design matrix
    if design == "poly":
        Psi = compute_design_poly(X, p)
    elif design == "fourier":
        Psi = compute_design_Fourier(X, p)
    elif design == "multivariate":
        Psi = compute_design_multivariate(X)
        p = X.shape[1] + 1
    else:
        raise ValueError()
    
    # Initialization
    T = np.zeros((n, n))
    S = np.zeros(n)
    
    stopping_time = -1

    for t in range(t_min, n):
        
        D = np.zeros(t)
        
        for tau in range(np.maximum(n_out_min, t - n_out_min - delta_max), t-n_out_min):
            
            # Compute the best fitting parameter theta
            theta = compute_theta(Psi[:t, :], tau)
            Z = Psi[:t, :] @ theta
            
            # Use thresholding to avoid numerical issues
            Z = np.minimum(Z, B)
            Z = np.maximum(Z, -B)
            
            D[:tau] = 2 / (1 + np.exp(-Z[:tau]))
            D[tau:] = 2 / (1 + np.exp(Z[tau:]))
            D = np.log(D)
            
            # Compute statistics for each t
            # and each change point candidate tau
            T[tau, t] = tau * (t - tau) / t * (np.mean(D[:tau]) + np.mean(D[tau:]))
            
        # Check whether the test statistic exceeded the threshold
        S[t] = np.max(T[:, t])
        if S[t] > threshold:
        
            stopping_time = t
            break
            
    # Array of test statistics
    if stopping_time != -1:
        S = S[:stopping_time + 1]
    
    return S, stopping_time


#------------------------------------------------------------------------------------------------------------------
# A version of the algorithm with neural networks
#------------------------------------------------------------------------------------------------------------------

# Defines the architecture of a neural network
#
class NN(nn.Module):
    def __init__(self, n_in, n_out):
        
        super(NN, self).__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(n_in, 2 * n_in)
        self.fc2 = nn.Linear(2 * n_in, 3 * n_in)
        self.fc3 = nn.Linear(3 * n_in, n_out)        
    
    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        
        return x
        
        
# Computation of the test statistic
#
# X -- array of univariate observations
#
# p -- positive integer (used for basis construction)
#
def compute_test_stat_nn(X, t_min=20, n_out_min=10, B=10, n_epochs=200, delta_max=150, model=NN, threshold=math.inf):
    
    X = X.reshape(-1, 1)
    
    # Sample size
    n = X.shape[0]
    
    # Initialization
    T = np.zeros((n, n))
    S = np.zeros(n)
    
    stopping_time = -1

    for t in range(t_min, n):
    
        for tau in range(np.maximum(n_out_min, t - n_out_min - delta_max), t-n_out_min):
            
            # Initialize neural network
            f = model(n_in=1, n_out=1)
            
            # Parameters of the optimizer
            opt = torch.optim.Adam(f.parameters(), lr=1e-1)
            
            X_t = torch.tensor(X[:t, :], dtype=torch.float32, requires_grad=True)
            
            # weights
            W = torch.cat((torch.ones(tau) * (t - tau), torch.ones(t - tau) * tau)).reshape(-1, 1)
            
            # Create "virtual" labels
            Y_t = torch.cat((torch.ones(tau), torch.zeros(t - tau))).reshape(-1, 1)
    
            # Loss function    
            loss_fn = nn.BCEWithLogitsLoss(weight=W)
            
            # Neural network training
            for epoch in range(n_epochs):
                
                loss = loss_fn(f(X_t), Y_t).mean()
                loss.backward()
                opt.step()
                opt.zero_grad()
                
            Z = f(X_t).detach().numpy().reshape(-1)
            
            # Use thresholding to avoid numerical issues
            Z = np.minimum(Z, B)
            Z = np.maximum(Z, -B)
            
            D = np.zeros(t)
            D[:tau] = 2 / (1 + np.exp(-Z[:tau]))
            D[tau:] = 2 / (1 + np.exp(Z[tau:]))
            D = np.log(D)
            
            # Compute statistics for each t
            # and each change point candidate tau
            T[tau, t] = tau * (t - tau) / t * (np.mean(D[:tau]) + np.mean(D[tau:]))
            
        # Check whether the test statistic exceeded the threshold
        S[t] = np.max(T[:, t])
        if S[t] > threshold:
        
            stopping_time = t
            break
            
    # Array of test statistics
    if stopping_time != -1:
        S = S[:stopping_time + 1]
        
    return S, stopping_time