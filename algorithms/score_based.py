# Score-based change point detection algorithm from the paper
# "Score-based change point detection via tracking the best of infinitely many experts"
# (https://arxiv.org/abs/2408.14073)
# by A. Markovich, N. Puchkin

import numpy as np
from scipy.special import logsumexp


def get_iterator(obj):
    try:
        iterator = iter(obj)
        return iterator
    except TypeError:
        return None


class PolyBasis:

    def __init__(self, degree, dim_x=1, include_cross=False):
        if include_cross and dim_x == 1:
            raise Exception('No pairwise products with 1-dimentional input')
        self.name = 'poly'
        self.degree = degree
        self.dim_x = dim_x
        self.include_cross = include_cross
        self.dim = self.dimention()

    def dimention(self):
        # calculate dimention of the parameter vector based on the dimention of the basis function
        d = self.degree * self.dim_x
        if self.include_cross:
            d += (self.dim_x * (self.dim_x - 1)) // 2
            if self.degree >= 3:
                d += (self.dim_x * (self.dim_x - 1) * (self.dim_x - 2)) // 6
        return d

    def make_poly_grad(self, x, degree):
        # gradient of the vector of monomials (diagonal jacobi matrix)
        poly = []
        for i in range(degree):
            poly.append(np.eye(self.dim_x) * (x ** i * (i + 1)))

        if degree == 1:
            return poly[0]
        return np.vstack(poly)

    def jacobi2(self, x):
        # jacobi matrix for pairwise products
        size = (self.dim_x * (self.dim_x - 1)) // 2
        J = np.empty((size, self.dim_x))
        j = 0
        for i in range(self.dim_x - 1):  # block idx
            cursize = self.dim_x - i - 1  # block height
            J[j:j + cursize, :] = np.hstack([np.zeros((cursize, i)), x[i + 1:].reshape(-1, 1), np.eye(cursize) * x[i]])
            j += cursize
        return J

    def jacobi3(self, x):
        # jacobi matrix for products of three
        size = (self.dim_x * (self.dim_x - 1) * (self.dim_x - 2)) // 6
        J = np.zeros((size, self.dim_x))
        J2 = self.jacobi2(x)
        j = 0
        m = 1
        for i in range(self.dim_x - 2):
            cur2sbstart = m + self.dim_x - 2 - i
            cursubblocksize = 0
            for k in range(self.dim_x - i - 2):
                cur2sbstart += cursubblocksize
                cursubblocksize = self.dim_x - i - k - 2
                J[j:j + cursubblocksize] = x[i + k + 1] * J2[m + k:m + k + cursubblocksize] + x[i] * J2[
                                                                                                     cur2sbstart:cur2sbstart + cursubblocksize]
                J[j:j + cursubblocksize, self.dim_x - cursubblocksize:] /= 2
                j += cursubblocksize
            m += self.dim_x - i - 1
        return J

    def compute_jacobi(self, x):
        # combine parts of jacobi matrix
        grad = self.make_poly_grad(x, self.degree)
        if self.include_cross:
            if self.degree >= 2:
                grad = np.vstack([grad, self.jacobi2(x)])
            if self.degree >= 3:
                grad = np.vstack([grad, self.jacobi3(x)])
        return grad

    def compute_laplacian(self, x):
        if self.degree == 1:
            return np.zeros(self.dim_x)

        laplacian = np.repeat((np.arange(1, self.degree) + 1), self.dim_x).reshape(-1, 1) * self.make_poly_grad(x,
                                                                                                                self.degree - 1)
        laplacian = laplacian.sum(axis=1)

        # laplacian of products of 2 and 3 coordinates equals zero
        laplacian = np.concatenate(
            [np.zeros(self.dim_x), laplacian, np.zeros(self.dim - laplacian.shape[0] - self.dim_x)])

        return laplacian.reshape(self.dim)


class ChangePointDetector:
    def __init__(self, d_dim, x_dim, lambda_, alpha, eta=1, basis=None, reference_score=None, threshold=np.inf):
        self.dim = d_dim
        self.dim = basis.dimention() if basis is not None else d_dim
        self.lambda_ = lambda_
        self.alpha = alpha
        self.eta = eta
        self.basis = basis
        self.threshold = threshold
        self.reference_score = lambda x: np.zeros((x_dim, 1)) if reference_score is None else reference_score(
            x)  # \nabla log ref_p : func

        # initialize

        self.cumsum_A = [np.zeros((self.dim, self.dim))]
        self.cumsum_b = [np.zeros((self.dim, 1))]

        self.EW_cumloss = np.zeros(1)
        self.FS_cumloss = np.zeros(1)

        self.EW_predictions = []
        self.FS_predictions = []

        self.current_grad = None
        self.current_laplacian = None

        self._logZ = None
        self._logV = None
        self._Y = None

        self.samples = None
        self.test_statistic = []

    def run(self, X, eta, diff=False, recalc=False):
        """
        :X: - input time series
        :eta: - scalar or np.array, value of the hyperparameter η
        :diff: - if set to True, the algorithm will run on the first difference of X
        :recalc: - if set to True, the values of V, Z, Y will be recalculated on each step (used with variable η)
        return: array of test statistic and the number of iteration of change point occurrence
        """

        # initialize
        self.samples = np.zeros(X.shape)
        n_samples = X.shape[0]

        # get η as an array of values for each step
        eta_iter = get_iterator(eta)
        if eta_iter is None and isinstance(eta, (int, float)):
            eta_iter = iter(np.ones(n_samples) * eta)
        elif eta_iter is None and not isinstance(eta, (int, float)):
            print('eta must be either a number or an iterable')
            return

        self._logZ = np.full((n_samples + 1, n_samples + 1), None)
        self._logV = np.full((n_samples + 1), None)
        self._Y = np.full((n_samples + 1, n_samples + 1, self.dim), None)

        stopping_time = -1
        pred = X[0] if n_samples > 0 else 0

        for i in range(n_samples):
            self.eta = next(eta_iter)
            x = X[i]
            if recalc:
                self._logZ = np.full((n_samples + 1, n_samples + 1), None)
                self._logV = np.full((n_samples + 1), None)
                self._Y = np.full((n_samples + 1, n_samples + 1, self.dim), None)

            if diff:
                x -= pred
                pred = X[i]

            stopping_time = self.step(x, i + 1)
            if stopping_time > 0:
                break

        self.test_statistic = np.array(self.EW_cumloss[1:]) - np.array(self.FS_cumloss[1:]).flatten()
        return self.test_statistic, stopping_time

    def step(self, x, t):
        """
        :x: - current observation
        :t: - number of iteration
        return: t, if change point is detected, else -1
        """
        self.samples[t - 1] = x

        self.compute_A(x)
        self.compute_b(x)

        EW_pred = self.predict_EW()
        EW_loss = self.compute_loss(EW_pred)
        self.EW_cumloss = np.append(self.EW_cumloss, self.EW_cumloss[-1] + EW_loss)
        self.EW_predictions.append(EW_pred)

        FS_pred = self.predict_FS(t)
        FS_loss = self.compute_loss(FS_pred)
        self.FS_cumloss = np.append(self.FS_cumloss, self.FS_cumloss[-1] + FS_loss)
        self.FS_predictions.append(FS_pred)

        if self.EW_cumloss[-1] - self.FS_cumloss[-1] > self.threshold:
            return t + 1

        return -1

    def compute_A(self, x):
        """
        :x: - current observation
        compute A = \nabla Psi(x) @ \nabla Psi(x).T 
        update cumulative A_s
        """

        self.current_grad = self.basis.compute_jacobi(x)
        A = self.current_grad @ self.current_grad.T
        self.cumsum_A.append(self.cumsum_A[-1] + A)
        return A

    def compute_b(self, x):
        """
        :x: - current observation
        self.reference_score = \nabla log p_0(x) 
        compute b = - \Delta Psi(x) - \nabla Psi(x) @ reference_score(x)
        update cumulative b_s
        """

        self.current_laplacian = -self.basis.compute_laplacian(x) - self.current_grad @ self.reference_score(
            x).flatten()
        self.current_laplacian = self.current_laplacian.reshape(-1, 1)
        self.cumsum_b.append(self.cumsum_b[-1] + self.current_laplacian)
        return self.current_laplacian

    def compute_loss(self, theta):
        """
        :theta: - np.array((self.dim, 1)) - forecasted params for sample x_t
        compute and return: loss(x, theta) = 0.5 ||\nabla Psi(x) * theta ||^2 - b(x) * theta
        """
        u = self.current_grad.T @ theta
        loss = 0.5 * u.T @ u - self.current_laplacian.T @ theta
        return loss

    def predict_EW(self):
        theta = np.linalg.pinv(self.cumsum_A[-2] + (self.lambda_ / self.eta) * np.eye(self.dim)) @ self.cumsum_b[-2]
        return theta

    def A(self, s, t):
        """
        return: sub-section sum of A
        """
        if s > t:
            return 0
        return self.cumsum_A[t] - self.cumsum_A[s - 1]

    def b(self, s, t):
        """
        return: sub-section sum of b
        """
        if s > t:
            return 0
        return self.cumsum_b[t] - self.cumsum_b[s - 1]

    def logZ(self, s, t):
        """
        compute and return the natural logarithm of Z_s:t
        """
        mat = self.A(s, t) + (self.lambda_ / self.eta) * np.eye(self.dim)
        under_exp = self.b(s, t).T @ np.linalg.pinv(mat) @ self.b(s, t)
        if self._logZ[s, t] is None:
            z = 0.5 * self.dim * np.log(self.lambda_ / self.eta) - 0.5 * np.log(np.linalg.det(mat)) + 0.5 * self.eta * \
                under_exp[0, 0]
            self._logZ[s, t] = z

        return self._logZ[s, t]

    def logV(self, t):
        """
        compute and return the natural logarithm of V_s:t
        """
        if t <= 1:
            self._logV[t] = 0

        if self._logV[t] is None:
            summands = np.zeros(t)
            summands[0] = (t - 1) * np.log(1 - self.alpha) + self.logZ(1, t)
            summands[1:] = np.log(self.alpha) + np.array(
                [s * np.log(1 - self.alpha) + self.logV(t - 1 - s) + self.logZ(t - s, t) for s in range(0, t - 1)])
            self._logV[t] = logsumexp(summands)

        return self._logV[t]

    def Y(self, s, t):
        """
        compute and return:  EW prediction for s:t
        """
        if s > t:
            return np.zeros((self.dim, 1))

        if self._Y[s, t, 0] is None:
            self._Y[s, t] = (np.linalg.pinv(self.A(s, t) + (self.lambda_ / self.eta) * np.eye(self.dim)) @ self.b(s,
                                                                                                                  t)).flatten()
        return self._Y[s, t]

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
        scale = np.array([self.Y(t - 1 - s, t - 1) for s in range(t - 1)])
        theta = np.zeros((self.dim, 1))

        for i in range(self.dim):
            th, sign = logsumexp(summands, b=scale[:, i], return_sign=True)
            theta[i] = np.exp(th) * sign

        return theta
