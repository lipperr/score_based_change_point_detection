import numpy as np
from scipy.special import logsumexp
from algorithms.ew import ExpWeightedForecaster

class FS(ExpWeightedForecaster):
    def __init__(self, alpha=1, ew_params={}, threshold=np.inf):
        self.alpha = alpha
        self.threshold = threshold

        super().__init__(**ew_params)


    def compute_test_stat(self, samples):
        super().restart()
        self._logZ = dict() 
        self._logV = dict()
        self.FS_cumloss = np.zeros(1) 
        self.FS_predictions = []

        samples = np.array(samples).reshape(len(samples), self.basis.xdim, 1)
        n_samples = len(samples)
        stopping_time = -1

        for t in range(n_samples):
            x = samples[t]
            super().ewstep(x)
            FS_pred = self.predict_FS(t+1)
            FS_loss = super().compute_loss(FS_pred)
            self.FS_cumloss = np.append(self.FS_cumloss, self.FS_cumloss[-1] + FS_loss)
            self.FS_predictions.append(FS_pred)
            if self.EW_cumloss[-1] - self.FS_cumloss[-1] > self.threshold:
                stopping_time = t
                break

        self.test_statistic = np.array(self.EW_cumloss[1:]) - np.array(self.FS_cumloss[1:]).flatten() 
        return self.test_statistic, stopping_time

    def predict_FS(self, t):
        """
        :t: - step
        """
        if t == 1:
            return np.zeros((self.dim, 1))

        summands = np.array([(t - 2) * np.log(1 - self.alpha) + self.logZ(1, t - 1)])
        summands = np.insert(summands, 0, np.log(self.alpha) + np.array(
            [s * np.log(1 - self.alpha) + self.logV(t - 2 - s) + self.logZ(t - 1 - s, t - 1) for s in range(t - 2)]))
        summands += np.log(1 - self.alpha) - self.logV(t - 1)
        scale = np.array([self.predict_EW(t - 1 - s, t - 1).flatten() for s in range(t - 1)])
        theta = np.zeros((self.dim, 1))

        for i in range(self.dim):
            th, sign = logsumexp(summands, b=scale[:, i], return_sign=True)
            theta[i] = np.exp(th) * sign

        return theta
    
    def logZ(self, s, t):
        """
        compute and return the natural logarithm of Z_s:t
        """
        mat = self.A(s, t) + (self.lambda_ / self.eta(self.t)) * np.eye(self.dim)
        under_exp = self.b(s, t).T @ np.linalg.pinv(mat) @ self.b(s, t)
        if (s,t) not in self._logZ:
            z = 0.5 * self.dim * np.log(self.lambda_ / self.eta(self.t)) - 0.5 * np.log(np.linalg.det(mat)) + 0.5 * self.eta(self.t) * \
                under_exp[0, 0]
            self._logZ[(s,t)] = z

        return self._logZ[(s,t)]

    def logV(self, t):
        """
        compute and return the natural logarithm of V_s:t
        """
        if t <= 1:
            self._logV[t] = 0

        if t not in self._logV:
            summands = np.zeros(t)
            summands[0] = (t - 1) * np.log(1 - self.alpha) + self.logZ(1, t)
            summands[1:] = np.log(self.alpha) + np.array(
                [s * np.log(1 - self.alpha) + self.logV(t - 1 - s) + self.logZ(t - s, t) for s in range(0, t - 1)])
            self._logV[t] = logsumexp(summands)

        return self._logV[t]
    
    