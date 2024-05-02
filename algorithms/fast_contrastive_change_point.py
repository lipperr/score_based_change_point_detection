# Fast online contrastive change point detection algorithm from the paper 
# "A Contrastive Approach to Online Change Point Detection" (arXiv:2206.10143)
# by A. Goldman, N. Puchkin, V. Shcherbakova, and U. Vinogradova


import numpy as np
from scipy.special import expit, log_expit
from scipy.linalg import schur
import math


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
# Computation of design matrix based on Hermite polynomials
#
# X -- array of univariate observations
#
# p -- positive integer, p <= 11
#
def compute_design_Hermite(X, p):
    
    if p > 11:
        raise ValueError()
    
    n = X.shape[0]
    
    Y = X[:].reshape(-1, 1)
    # First 10 Hermite polynomials
    Psi = np.ones((n, 1))
    Psi = np.append(Psi, Y, axis=1)
    Psi = np.append(Psi, Y**2 - 1, axis=1)
    Psi = np.append(Psi, Y**3 - 3 * Y, axis=1)
    Psi = np.append(Psi, Y**4 - 6 * Y**2 + 3, axis=1)
    Psi = np.append(Psi, Y**5 - 10 * Y**3 + 15 * Y, axis=1)
    Psi = np.append(Psi, Y**6 - 15 * Y**4 + 45 * Y**2 - 15, axis=1)
    Psi = np.append(Psi, Y**7 - 21 * Y**5 + 105 * Y**3 - 105 * Y, axis=1)
    Psi = np.append(Psi, Y**8 - 28 * Y**6 + 210 * Y**4 - 420 * Y**2 + 105, axis=1)
    Psi = np.append(Psi, Y**9 - 36 * Y**7 + 378 * Y**5 - 1260 * Y**3 + 945 * Y, axis=1)
    Psi = np.append(Psi, Y**10 - 45 * Y**8 + 630 * Y**6 - 3150 * Y**4 + 4725 * Y**2 - 945, axis=1)
    
    
    return Psi[:, :p]
    
    
# Auxiliary function
# Computation of design matrix based on Legendre polynomials
#
# X -- array of univariate observations
#
# p -- positive integer, p <= 11
#
def compute_design_Legendre(X, p):
    
    if p > 11:
        raise ValueError()
    
    n = X.shape[0]
    
    Y = X[:].reshape(-1, 1)
    # First 10 Legendre polynomials
    Psi = np.ones((n, 1))
    Psi = np.append(Psi, Y, axis=1)
    Psi = np.append(Psi, (3 * Y**2 - 1) / 2, axis=1)
    Psi = np.append(Psi, (5 * Y**3 - 3 * Y) / 2, axis=1)
    Psi = np.append(Psi, (35 * Y**4 - 30 * Y**2 + 3) / 8, axis=1)
    Psi = np.append(Psi, (63 * Y**5 - 70 * Y**3 + 15 * Y) / 8, axis=1)
    Psi = np.append(Psi, (231 * Y**6 - 315 * Y**4 + 105 * Y**2 - 5) / 16, axis=1)
    Psi = np.append(Psi, (429 * Y**7 - 693 * Y**5 + 315 * Y**3 - 35) / 16, axis=1)
    Psi = np.append(Psi, (6435 * Y**8 - 12012 * Y**6 + 6930 * Y**4 - 1260 * Y**2 + 35) / 128, axis=1)
    Psi = np.append(Psi, (12155 * Y**9 - 25740 * Y**7 + 18018 * Y**5 - 4620 * Y**3 + 315 * Y) / 128, axis=1)
    Psi = np.append(Psi, (46189 * Y**10 - 109395 * Y**8 + 90090 * Y**6 - 30030 * Y**4 + 3465 * Y**2 - 63) / 256, axis=1)
    
    
    return Psi[:, :p]
    
    
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
    

# An auxiliary function to compute projections
#
def project_induced_lm(x, A, R):
        if np.linalg.norm(x) <= R:
            return x

        # projection procedure in induced norm described by Lagrangian method
        # supports zero diagonal entrance
        # not implemented for when A is not (almost) symmetric and if not (almost) positive semi definite
        if not np.allclose(A, A.T):
            raise ValueError("Matrix is not symmetric")
        D, U = schur(A, output='real')

        U = U.T

        # we allow for numeric instability and consider x~0 if np.abs(x) < 1e-4 (even for negative values)
        if np.any(np.diag(D) <= -1e-4):
            raise ValueError("Matrix is not positive semi definite")

        p = A.shape[0]
        cv = np.zeros(p)
        uv = U @ x
        for j in range(p):
            if np.abs(D[j, j]) < 1e-4:
                # zero diagonal element case (repeating code for numeric stability)
                cv[j] = 0
            else:
                cv[j] = uv[j]

        if np.linalg.norm(cv) <= R:
            # check if we projected after zeroing coordinates
            return U.T @ cv

        l = 1e-8
        r = R * np.max(np.abs(D))

        for i in range(50):
            m = (r+l)/2
            for j in range(p):
                if np.abs(D[j, j]) < 1e-4:
                    # zero diagonal element case
                    cv[j] = 0
                else:
                    cv[j] = uv[j]*D[j, j]/(D[j, j]+m)
            if np.linalg.norm(cv) <= R:
                r = m
            else:
                l = m
            if r - l < 1e-3:
                break

        for j in range(p):
            if np.abs(D[j, j]) < 1e-4:
                # zero diagonal element case
                cv[j] = 0
            else:
                cv[j] = uv[j]*D[j, j]/(D[j, j]+r) # it projects inside ball, but maybe not on its border
        return U.T @ cv
        
        
def compute_test_stat_ftal(X, p, beta=None, R=10, t_min=20, n_out_min=10, delta_max=150, design="hermite", threshold=math.inf):
    
    # Sample size
    n = X.shape[0]

    # Compute design matrix
    if design == "poly":
        Psi = compute_design_poly(X, p)
    elif design == "fourier":
        Psi = compute_design_Fourier(X, p)
    elif design == "hermite":
        Psi = compute_design_Hermite(X, p)
    elif design == "legendre":
        Psi = compute_design_Legendre(X, p)
    elif design == "multivariate":
        Psi = compute_design_multivariate(X)
        p = X.shape[1] + 1
    else:
        raise ValueError()

    # Initialization
    if beta is None:
        beta = 1 / ((2 * R) * n)  # not fully according to formula

    T = np.zeros((n, 1))  # current values
    S = np.zeros(n)
    varphi = np.zeros((n, 1))
    thetas = np.zeros((n, p))  # current parameters
    grads = np.zeros((n, p))
    seq = np.arange(1, n + 1)
    A_hes = np.zeros((n, p, p))
    b_vec = np.zeros((n, p))
    A_hes_inv = np.zeros((n, p, p))
    was_here = np.zeros((n,))
    proj_rate = np.zeros((n,))
    
    stopping_time = -1

    for t in range(t_min, n):

        # consider time intervals [0;t) and [0;tau), thus tau is the first point in the new distribution

        for tau in range(max(n_out_min, t - n_out_min - delta_max), t - n_out_min):

            grads[tau] = expit(- thetas[tau][None, :] @ Psi[:tau].T) @ Psi[:tau] - tau * expit( thetas[tau][None, :] @ Psi[t - 1][:, None]) * Psi[t - 1][None, :]

            A_hes[tau] += grads[tau][:, None] @ grads[tau][None, :]


            A_hes_inv[tau] = np.linalg.pinv(A_hes[tau], hermitian=True)
            b_vec[tau] += (grads[tau] @ thetas[tau] + 1/beta) * grads[tau]
            y = A_hes_inv[tau] @ b_vec[tau]


            thetas[tau] = project_induced_lm(y, A_hes[tau], R)

            if not np.allclose(y, thetas[tau]):
                proj_rate[tau] += 1

            varphi[tau] = log_expit(thetas[tau][None, :] @ Psi[:tau].T).sum() + tau * log_expit( -thetas[tau][None, :] @ Psi[t - 1][:, None]) + 2 * np.log(2) * (tau)
            T[tau] = (t - 1) / t * T[tau] + 1 / (t) * varphi[tau]
            
    
        # Check whether the test statistic exceeded the threshold
        S[t] = np.max(T[:t])
        if S[t] > threshold:
        
            stopping_time = t
            break
    
    # Array of test statistics
    if stopping_time != -1:
        S = S[:stopping_time + 1]
    
    return S, stopping_time