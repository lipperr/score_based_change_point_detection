import numpy as np
from numpy.random import normal as randn, multivariate_normal as randmn
from scipy.special import logsumexp
from scipy.linalg import fractional_matrix_power



def get_iterator(obj):
    try:
        iterator = iter(obj)
        return iterator
    except TypeError:
        return None



class TrigBasis:
    def __init__(self, dim):
        self.dim = dim

    def compute_jacobian_m(self, x):

        def grad_i(x, d):
            coef = (d+1) // 2
            sign = 1 if d % 2 else -1
            func = np.cos if d % 2 else np.sin
            return sign * coef * func(coef * x)
            
        current_grad = np.zeros(self.dim)
        if self.dim == 1:
            current_grad[0] = grad_i(x, 1)
        else:
            for i in range(self.dim):
                current_grad[i] = grad_i(x[i], i)

        return current_grad.reshape((self.dim, 1))

    def compute_laplacian(self, x):

        def second_derivative(x, d):
            coef = (d+1) // 2
            func = np.sin if d % 2 else np.cos
            return (-1) * coef**2 * func(coef * x)
        
        current_laplacian = np.zeros(self.dim) 
        if self.dim == 1:
            current_laplacian[0] = second_derivative(x, 1) 
        else:
            for i in range(self.dim):
                current_laplacian[i] = second_derivative(x[i], i)

        return current_laplacian.reshape((self.dim, 1))


class PolyBasis:
    """ dim_x = 1
    x, x^2, x^3, ... x^d

    dim_x = k
    x_1, ..., x_k, x_1^2, ..., x_k^2, ..., x_1^k, ..., x_k^k
    """
    def __init__(self, dim, dim_x=1):
        self.dim = dim
        self.dim_x = dim_x
        self.name = 'poly'

    def _make_poly_grad(self, x, dim):
        poly = np.empty((dim, x.shape[0], 1))
        for i in range(dim):
            poly[i] = x ** i * (i+1)
        return poly.reshape(-1, 1)
    
    def compute_jacobian_m(self, x):
        grad = self._make_poly_grad(x, self.dim)
        return grad.reshape((self.dim * x.shape[0], 1))
        

    def compute_laplacian(self, x):
        laplacian = np.repeat((np.arange(1, self.dim) + 1), x.shape[0]).reshape(-1, 1) * self._make_poly_grad(x, self.dim-1)
        laplacian = np.concatenate([np.zeros((x.shape[0], 1)), laplacian])

        return laplacian.reshape((self.dim * x.shape[0], 1))
        




class ChangePointDetector:
    def __init__(self, d_dim, lambda_, alpha, basis=None, reference_score=None, threshold=np.inf):
        self.dim = d_dim
        self.lambda_ = lambda_
        self.alpha = alpha
        self.basis = basis 
        self.reference_score = lambda x: np.zeros((self.dim, 1)) if reference_score is None else reference_score(x) # \nabla log ref_p : func

        self.cumsum_A = [np.zeros((self.dim, self.dim))]
        self.cumsum_b = [np.zeros((self.dim, 1))]

        self.EW_cumloss = np.zeros(1)
        self.FS_cumloss = np.zeros(1)

        self.EW_predictions = []
        self.FS_predictions = []

        self.current_grad = None
        self.current_laplacian = None

        self._Z = None
        self._V = None
        self._Y = None

        self._logZ = None
        self._logV = None
        self._Yo = None

        self.samples = None
        self.threshold = threshold

        self.test_statistic = []

    def run(self, X, eta):
        self.samples = X
        n_samples = X.shape[0]

        eta_iter = get_iterator(eta)
        if eta_iter is None and isinstance(eta, (int, float)):
            eta_iter = iter(np.ones(n_samples) * eta)
        elif eta_iter is None and not isinstance(eta, (int, float)):
            print('eta must be either a number or an iterable')
            return

        

        self._Z = np.full((n_samples+1, n_samples+1), None)
        self._V = np.full((n_samples+1), None)
        self._Y = np.full((n_samples+1, n_samples+1, self.dim), None)


        self._logZ = np.full((n_samples+1, n_samples+1), None)
        self._logV = np.full((n_samples+1), None)
        self._Yo = np.full((n_samples+1, n_samples+1, self.dim), None)


        stopping_time = -1
        for i, x in enumerate(self.samples):
            self.eta = next(eta_iter)
            stopping_time = self.step(x, i+1)
            if stopping_time > 0:
                break

        self.test_statistic = np.array(self.EW_cumloss[1:]) - np.array(self.FS_cumloss[1:]).flatten()
        return self.test_statistic, stopping_time

    def step(self, x, t):

        self.compute_A(x)
        self.compute_b(x)
        
        self.eta = np.sqrt(1/(t+1))

        
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
        compute A = \nabla Psi(x) @ \nabla Psi(x).T 
        update cumulative A_s
        """
        self.current_grad = self.basis.compute_jacobian_m(x)
        A = np.outer(self.current_grad, self.current_grad)
        self.cumsum_A.append(self.cumsum_A[-1] + A)
        return A
    

    def compute_b(self, x):
        """ 
        self.reference_score = \nabla log p_0(x) 
        compute b = - \Delta Psi(x) - \nabla Psi(x) * reference_score(x)
        update cumulative b_s
        """

        self.current_laplacian = -self.basis.compute_laplacian(x) - self.current_grad * self.reference_score(x)
        self.cumsum_b.append(self.cumsum_b[-1] + self.current_laplacian)
        return self.current_laplacian
    
    
    def compute_loss(self, theta):
        """
        :theta: np.array((self.dim, 1)) - forecasted params for sample x_t
        compute loss(x, theta) = 0.5 ||\nabla Psi(x) * theta ||^2 - b(x) * theta
        """
        # print('--', self.current_grad, self.current_laplacian,  theta, sep='\n')
        u = self.current_grad * theta
        loss = 0.5 * u.T @ u - self.current_laplacian.T @ theta
        # print(loss)
        return loss

    
    def predict_EW(self):
        theta = np.linalg.pinv(self.cumsum_A[-2] + (self.lambda_ / self.eta) * np.eye(self.dim)) @ self.cumsum_b[-2]
        return theta
    
    
    def A(self, s, t):
        """
        sub-section sum of A
        """
        if s > t:
            return 0
        return self.cumsum_A[t] - self.cumsum_A[s-1]
    
    def b(self, s, t):
        """
        sub-section sum of b
        """
        if s > t:
            return 0
        return self.cumsum_b[t] - self.cumsum_b[s-1]
    
    # def Z(self, s, t):
    #     coef1 = (self.lambda_ / self.eta) ** (self.dim / 2)
    #     mat = self.A(s, t) + (self.lambda_ / self.eta) * np.eye(self.dim)
    #     if self._Z[s, t] is None:
    #         a = self.b(s, t).T @ np.linalg.pinv(mat) @ self.b(s, t)
    #         self._Z[s, t] = coef1 * np.sqrt(1 / np.linalg.det(mat)) * np.exp(self.eta / 2 * a[0, 0])


    #     return self._Z[s, t]
    


    # def V(self, t):
    #     if t <= 1:
    #         self._V[t] = 1
    #     elif self._V[t] is None: 
    #         self._V[t] = (1-self.alpha)**(t-1) * self.Z(1, t)
    #         self._V[t] += self.alpha * sum([(1-self.alpha)**s * self.V(t-1-s) * self.Z(t-s, t) for s in range (0, t-1)])

    #     return self._V[t]

    
    # def Y(self, s, t):
    #     if s > t:
    #         return np.zeros((self.dim, 1))

    #     if self._Y[s, t, 0] is None:
    #         self._Y[s, t] = (self.Z(s, t) * np.linalg.pinv(self.A(s, t) + (self.lambda_ / self.eta) * np.eye(self.dim)) @ self.b(s,t)).flatten()

    #     return self._Y[s, t]
    
    def logZ(self, s, t):
        mat = self.A(s, t) + (self.lambda_ / self.eta) * np.eye(self.dim)
        under_exp = self.b(s, t).T @ np.linalg.pinv(mat) @ self.b(s, t)
        if self._logZ[s, t] is None:
            z = 0.5 * self.dim * np.log(self.lambda_ / self.eta) - 0.5*np.log(np.linalg.det(mat)) + 0.5 * self.eta * under_exp[0, 0]
            self._logZ[s, t] = z

        return self._logZ[s, t]
    
    def logV(self, t):
        if t <= 1:
            self._logV[t] = 0

        if self._logV[t] is None: 
            summands = np.zeros(t)
            summands[0] = (t-1) * np.log(1-self.alpha)+self.logZ(1, t)
            summands[1:] = np.log(self.alpha) + np.array([s * np.log(1-self.alpha) + self.logV(t-1-s) + self.logZ(t-s, t) for s in range (0, t-1)])
            self._logV[t] = logsumexp(summands)

        return self._logV[t]
    


    def Yo(self, s, t):
        if s > t:
            return np.zeros((self.dim, 1))

        if self._Yo[s, t, 0] is None:
            self._Yo[s, t] =  (np.linalg.pinv(self.A(s, t) + (self.lambda_ / self.eta) * np.eye(self.dim)) @ self.b(s,t)).flatten()
        return self._Yo[s, t]

    def predict_FS(self, t):
        """
        :t: - step
        """

        if t == 1:
            return np.zeros((self.dim, 1))
        
        
        summands = np.array([(t-2) * np.log(1-self.alpha) + self.logZ(1, t-1)])
        summands = np.insert(summands, 0, np.log(self.alpha) + np.array([s * np.log(1-self.alpha) + self.logV(t-2-s) + self.logZ(t-1-s, t-1) for s in range(t-2)]))
        summands += np.log(1-self.alpha) - self.logV(t-1)
        scale = np.array([self.Yo(t-1-s, t-1) for s in range(t-1)])
        theta = np.zeros((self.dim, 1))
        # print('t:' , t)
        
        # print('self.logZ(1, t-1)', self.logZ(1, t-1), '\n','self.logV(t-1)', self.logV(t-1), '\n', 's', [s for s in range(t-2)], '\n', 'logV', np.array([self.logV(t-2-s) for s in range(t-2)]), '\n', 'logZ', np.array([self.logZ(t-1-s, t-1) for s in range(t-2)]), '\n', 'scale', scale)
        # theta = (1 - self.alpha) / np.exp(self.logV(t-1)) * (1-self.alpha)**(np.max(t-2, 0)) * np.exp(self.logZ(1, t-1)) * self.Yo(1, t-1)
        # theta += (1 - self.alpha) / np.exp(self.logV(t-1)) * self.alpha * np.sum([(1-self.alpha)**s * np.exp(self.logV(t-2-s)) * np.exp(self.logZ(t-1-s, t-1) )* self.Yo(t-1-s, t-1) for s in range(t-2)])

        for i in range(self.dim):
            th, sign = logsumexp(summands, b=scale[:, i], return_sign=True)
            theta[i] = np.exp(th) * sign

        # return theta

        # thetaT = (1 - self.alpha) / self.V(t-1) * (1-self.alpha)**(np.max(t-2, 0)) * self.Y(1, t-1)
        # thetaT += (1 - self.alpha) / self.V(t-1) * self.alpha * np.sum([(1-self.alpha)**s * self.V(t-2-s) * self.Y(t-1-s, t-1) for s in range(t-2)])
        # print('theta', '\n', 'self.V(t-1)', self.V(t-1), '\n', 'self.Y(1, t-1)', self.Y(1, t-1), '\n', 's', [s for s in range(t-2)], '\n', 'self.V(t-2-s)', [self.V(t-2-s) for s in range(t-2)], '\n', 'self.Y(t-1-s, t-1)',[self.Y(t-1-s, t-1) for s in range(t-2)] )

        # print('final')
        # print(thetaT, theta)

        return theta
    


        # return thetaT.reshape(-1, 1)



