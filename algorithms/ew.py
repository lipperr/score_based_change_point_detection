import numpy as np 
from scipy.special import logsumexp

def ref_score(x):
    if isinstance(x, (int, float)):
        return np.array([-x]).reshape(-1, 1)
    return -x.reshape(-1, 1)

class PolyBasis:

    def __init__(self, degree, xdim=1, include_cross=False):
        if include_cross and xdim == 1:
            raise Exception('No pairwise products with 1-dimentional input')
        self.name = 'poly'
        self.degree = degree
        self.xdim = xdim
        self.include_cross = include_cross
        self.dim = self.dimention()

    def dimention(self):
        # calculate dimention of the parameter vector based on the dimention of the basis function
        d = self.degree * self.xdim
        if self.include_cross:
            d += (self.xdim * (self.xdim - 1)) // 2
            if self.degree >= 3:
                d += (self.xdim * (self.xdim - 1) * (self.xdim - 2)) // 6
        return d

    def make_poly_grad(self, x, degree):
        # gradient of the vector of monomials (diagonal jacobi matrix)
        poly = []
        for i in range(degree):
            poly.append(np.eye(self.xdim) * (x ** i * (i + 1)))

        if degree == 1:
            return poly[0]
        return np.vstack(poly)

    def jacobi2(self, x):
        # jacobi matrix for pairwise products
        size = (self.xdim * (self.xdim - 1)) // 2
        J = np.empty((size, self.xdim))
        j = 0
        for i in range(self.xdim - 1):  # block idx
            cursize = self.xdim - i - 1  # block height
            J[j:j + cursize, :] = np.hstack([np.zeros((cursize, i)), x[i + 1:].reshape(-1, 1), np.eye(cursize) * x[i]])
            j += cursize
        return J

    def jacobi3(self, x):
        # jacobi matrix for products of three
        size = (self.xdim * (self.xdim - 1) * (self.xdim - 2)) // 6
        J = np.zeros((size, self.xdim))
        J2 = self.jacobi2(x)
        j = 0
        m = 1
        for i in range(self.xdim - 2):
            cur2sbstart = m + self.xdim - 2 - i
            cursubblocksize = 0
            for k in range(self.xdim - i - 2):
                cur2sbstart += cursubblocksize
                cursubblocksize = self.xdim - i - k - 2
                J[j:j + cursubblocksize] = x[i + k + 1] * J2[m + k:m + k + cursubblocksize] + x[i] * J2[
                                                                                                     cur2sbstart:cur2sbstart + cursubblocksize]
                J[j:j + cursubblocksize, self.xdim - cursubblocksize:] /= 2
                j += cursubblocksize
            m += self.xdim - i - 1
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
            return np.zeros(self.xdim)

        laplacian = np.repeat((np.arange(1, self.degree) + 1), self.xdim).reshape(-1, 1) * self.make_poly_grad(x,
                                                                                                                self.degree - 1)
        laplacian = laplacian.sum(axis=1)

        # laplacian of products of 2 and 3 coordinates equals zero
        laplacian = np.concatenate(
            [np.zeros(self.xdim), laplacian, np.zeros(self.dim - laplacian.shape[0] - self.xdim)])

        return laplacian.reshape(self.dim)


class ExpWeightedForecaster:
    def __init__(self, xdim=1, lambda_=1, eta=-1, gamma=1e-6, basis=None, reference_score=False, eta_type="const"):
        if basis is None:
            raise ValueError("Specify basis")
        self.dim = basis.dim 
        self.lambda_ = lambda_

        self.eta_type = eta_type
        self._eta = lambda t: eta
        if eta_type == "var":
            self._eta = lambda t: 1/np.sqrt(t)
        self.gamma = gamma
        self.basis = basis

        self.reference_score = lambda x: np.zeros((xdim, 1)) if reference_score is False else ref_score(
            x)  # \nabla log ref_p : func

        self.restart()

    @property
    def eta(self):
        return self._eta
    
    @eta.setter
    def eta(self, new_value):
        self._eta = lambda t: new_value
        if self.eta_type == "var":
            self._eta = lambda t: 1/np.sqrt(t)

    def restart(self):
        self.cumsum_A = [np.zeros((self.dim, self.dim))]
        self.cumsum_b = [np.zeros((self.dim, 1))]

        self.EW_cumloss = np.zeros(1)
        self.EW_predictions = []

        self.current_grad = None
        self.current_laplacian = None

        self.test_statistic = []
        self.samples = []

        self._thetas = dict()
        
        self.t = 1


    def ewstep(self, x):
        """
        :x: - current observation
        :t: - number of iteration
        return: t, if change point is detected, else -1
        """

        self.samples.append(x)
        self.compute_A(x)
        self.compute_b(x)

        EW_pred = self.predict_EW()
        EW_loss = self.compute_loss(EW_pred)
        self.EW_cumloss = np.append(self.EW_cumloss, self.EW_cumloss[-1] + EW_loss)
        self.EW_predictions.append(EW_pred)

        self.t += 1
        return EW_loss, EW_pred
    
    def predict_EW(self, s=1, t=-1):
        if t == -1:
            t = self.t - 1

        if s > t:
            return np.zeros((self.dim, 1))
        
        if (s, t) not in self._thetas:
            self._thetas[(s, t)] = np.linalg.solve(self.A(s, t) + (self.lambda_ / self.eta(t)) * np.eye(self.dim), self.b(s,t))
        return self._thetas[(s, t)]
    
    def compute_loss(self, theta):
        """
        :theta: - np.array((self.dim, 1)) - forecasted params for sample x_t
        compute and return: loss(x, theta) = 0.5 ||\nabla Psi(x) * theta ||^2 - b(x) * theta
        """
        u = self.current_grad.T @ theta
        loss = 0.5 * u.T @ u - self.current_laplacian.T @ theta
        return loss
    
    def compute_A(self, x):
        """
        :x: - current observation
        compute A = \nabla Psi(x) @ \nabla Psi(x).T 
        update cumulative A_s
        """

        self.current_grad = self.basis.compute_jacobi(x)
        A = self.current_grad @ self.current_grad.T + self.gamma * np.eye(self.dim)
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

    def A(self, s, t):
        """
        return: sub-section sum of A
        """
        if s > t:
            return np.zeros_like(self.cumsum_A[0])
        return self.cumsum_A[t] - self.cumsum_A[s - 1]

    def b(self, s, t):
        """
        return: sub-section sum of b
        """
        if s > t:
            return np.zeros_like(self.cumsum_b[0])
        return self.cumsum_b[t] - self.cumsum_b[s - 1]






