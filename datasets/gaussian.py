import numpy as np
from numpy.random import normal as randn
from typing import Union, List
from collections.abc import Iterable
import matplotlib.pyplot as plt

class Gaussian:
    def __init__(self, mu=Union[float, List[float], np.ndarray], sigma=Union[float, List[float], np.ndarray], shift=2, shift_type="mean", seed=1, runs=1, cps=[-1], size=300, dim=1):
        self.type = "independent"
        self.name = "gaussian"
        self.mu = mu
        self.sigma = sigma
        self.shift = shift
        self.shift_type = shift_type
        self.runs = runs
        self.cps = np.full(runs, cps)
        self.size = size
        self.dim = dim
        np.random.seed(seed)
        self.data_cp = np.array([self.gen_data(self.size, self.dim, self.cps[i]) for i in range(runs)])
        self.data_stat = self.data_cp[:, :self.cps[0], ...]

    def gen_data(self, size, dim, tau=-1):
        mu_vec = self._as_array(self.mu, dim)
        std_vec = self._as_array(self.sigma, dim)
        X = randn(size=size*dim).reshape((size, dim)) 
        X[:tau, :] = mu_vec + std_vec * X[:tau, :]
        if tau > -1:
            if self.shift_type == "mean": 
                X[tau:, :] = (self._as_array(self.shift, dim) * std_vec).reshape(1, dim) + mu_vec + std_vec * X[tau:, :]
            else: 
                X[tau:, :] = mu_vec + self._as_array(self.shift, self.dim).reshape(1, dim) * std_vec * X[tau:, :] 

        return X
    
    def _as_array(self, x, dim):
        if isinstance(x, Iterable):
            arr = np.array(x, dtype=float)
            assert arr.shape[0] == dim, f"Length of parameter must equal {dim}"
            return arr
        return np.full(dim, float(x))
    
    def get_data_stat(self):
        return self.data_stat

    def get_data_cp(self):
        return self.data_cp

    def get_cps(self):
        return self.cps
    
    def get_cps_plot(self):
        cps_for_plot = []
        ls = 0
        for i in range(self.runs):
            cps_for_plot.append(self.cps[i] + ls)
            ls += len(self.data_cp[i])
        return cps_for_plot
    
    def display_data(self, save_path_plot="results/gaussian.png"):
        for i in range(self.dim):
            plt.subplot(2, 2, i+1)
            plt.plot(self.data_cp[0, :, i])

        plt.savefig(save_path_plot)