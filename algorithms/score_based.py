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
        if include_cross and dim_x==1:
            raise Exception('No cross products with 1-dimentional input')
        self.degree = degree
        self.dim_x = dim_x
        self.name = 'poly'
        self.include_cross=include_cross
        self.dim = self.dimention()

    def dimention(self):
        d = self.degree * self.dim_x
        if self.include_cross:
            d += (self.dim_x*(self.dim_x-1))//2
            if self.degree >= 3:
                d += (self.dim_x*(self.dim_x-1)*(self.dim_x-2))//6
        return d

    def _make_poly_grad(self, x, degree):
        poly = []
        for i in range(degree):
            poly.append(np.eye(self.dim_x) * (x ** i * (i+1)))

        if degree == 1:
            return poly[0]
        return np.vstack(poly)
    

    # jacobian, cross product of two
    def jacobi2(self, x):
        size = (self.dim_x*(self.dim_x-1))//2
        J = np.empty((size, self.dim_x))
        j=0
        for i in range(self.dim_x-1): # block idx
            cursize = self.dim_x-i-1 # block height
            J[j:j+cursize, :] = np.hstack([np.zeros((cursize, i)), x[i+1:].reshape(-1, 1), np.eye(cursize)*x[i]])
            j += cursize
        return J


    # jacobian, cross product of three
    def jacobi3(self, x):
        size = (self.dim_x*(self.dim_x-1)*(self.dim_x-2))//6
        J = np.zeros((size, self.dim_x))
        J2 = self.jacobi2(x)
        j=0 #start idx
        l = 1
        for i in range(self.dim_x-2): # block idx
            cur2sbstart = l + self.dim_x-2 - i
            cursubblocksize = 0
            for k in range(self.dim_x-i-2): # subblock idx
                cur2sbstart += cursubblocksize 
                cursubblocksize = self.dim_x-i-k-2
                J[j:j+cursubblocksize] = x[i+k+1]*J2[l+k:l+k+cursubblocksize] + x[i]*J2[cur2sbstart:cur2sbstart+cursubblocksize]
                J[j:j+cursubblocksize, self.dim_x - cursubblocksize:] /= 2
                j += cursubblocksize
            l += self.dim_x - i - 1
        return J


    def compute_jacobi(self, x):
        grad = self._make_poly_grad(x, self.degree)
        if self.include_cross:
            if self.degree >= 2:
                grad = np.vstack([grad, self.jacobi2(x)])
            if self.degree >= 3:
                grad = np.vstack([grad, self.jacobi3(x)])
        return grad
        

    def compute_laplacian(self, x):
        if self.degree == 1:
            return np.zeros(self.dim_x)
        
        laplacian = np.repeat((np.arange(1, self.degree) + 1), self.dim_x).reshape(-1, 1) * self._make_poly_grad(x, self.degree-1)
        laplacian = laplacian.sum(axis=1)
        laplacian = np.concatenate([np.zeros(self.dim_x), laplacian, np.zeros(self.dim - laplacian.shape[0] - self.dim_x)])

        return laplacian.reshape(self.dim)


class ChangePointDetector:
    def __init__(self, d_dim, x_dim, lambda_, alpha, gamma=0, basis=None, reference_score=None, threshold=np.inf):
        self.dim = d_dim
        self.dim = basis.dimention() if basis is not None else d_dim
        self.lambda_ = lambda_
        self.alpha = alpha
        self.gamma = gamma
        self.basis = basis 
        self.reference_score = lambda x: np.zeros((x_dim, 1)) if reference_score is None else reference_score(x) # \nabla log ref_p : func

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
        self._Yo = None

        self.samples = None
        self.threshold = threshold

        self.test_statistic = []

    def run(self, X, eta, transform=False, diff=False, recalc=False):
        self.samples = np.zeros(X.shape)
        n_samples = X.shape[0]
        dim_x = X.shape[1]

        eta_iter = get_iterator(eta)
        if eta_iter is None and isinstance(eta, (int, float)):
            eta_iter = iter(np.ones(n_samples) * eta)
        elif eta_iter is None and not isinstance(eta, (int, float)):
            print('eta must be either a number or an iterable')
            return


        self._logZ = np.full((n_samples+1, n_samples+1), None)
        self._logV = np.full((n_samples+1), None)
        self._Yo = np.full((n_samples+1, n_samples+1, self.dim), None)


        stopping_time = -1
        summ = np.zeros((dim_x, 1))
        pred = X[0] if n_samples>0 else 0

        for i in range(n_samples):
            self.eta = next(eta_iter)
            x = X[i]
            if recalc:
                self._logZ = np.full((n_samples+1, n_samples+1), None)
                self._logV = np.full((n_samples+1), None)
                self._Yo = np.full((n_samples+1, n_samples+1, self.dim), None)
            
            
            if diff:
                x -= pred
                pred = X[i]


            if transform:
                summ += X[i]
                x = X[i]-summ/(i+1)
                
            stopping_time = self.step(x, i+1)
            if stopping_time > 0:
                break

        self.test_statistic = np.array(self.EW_cumloss[1:]) - np.array(self.FS_cumloss[1:]).flatten()
        return self.test_statistic, stopping_time

    def step(self, x, t):
        self.samples[t-1] = x

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
        compute A = \nabla Psi(x) @ \nabla Psi(x).T 
        update cumulative A_s
        """

        self.current_grad = self.basis.compute_jacobi(x)
        A = self.current_grad @ self.current_grad.T
        A += self.gamma * np.eye(A.shape[0])
        self.cumsum_A.append(self.cumsum_A[-1] + A)
        return A
    

    def compute_b(self, x):
        """ 
        self.reference_score = \nabla log p_0(x) 
        compute b = - \Delta Psi(x) - \nabla Psi(x) @ reference_score(x)
        update cumulative b_s
        """

        self.current_laplacian = -self.basis.compute_laplacian(x) - self.current_grad @ self.reference_score(x).flatten()
        self.current_laplacian = self.current_laplacian.reshape(-1, 1)
        self.cumsum_b.append(self.cumsum_b[-1] + self.current_laplacian)
        return self.current_laplacian
    
    
    def compute_loss(self, theta):
        """
        :theta: np.array((self.dim, 1)) - forecasted params for sample x_t
        compute loss(x, theta) = 0.5 ||\nabla Psi(x) * theta ||^2 - b(x) * theta
        """
        u = self.current_grad.T @ theta
        loss = 0.5 * u.T @ u - self.current_laplacian.T @ theta
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

        for i in range(self.dim):
            th, sign = logsumexp(summands, b=scale[:, i], return_sign=True)
            theta[i] = np.exp(th) * sign

        return theta




# implementation without logsumexp
        # self._Z = None
        # self._V = None
        # self._Y = None

        # self._Z = np.full((n_samples+1, n_samples+1), None)
        # self._V = np.full((n_samples+1), None)
        # self._Y = np.full((n_samples+1, n_samples+1, self.dim), None)


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



    # def predict_FS(self, t):
    #     """
    #     :t: - step
    #     """

    #     if t == 1:
    #         return np.zeros((self.dim, 1))
        
    #     thetaT = (1 - self.alpha) / self.V(t-1) * (1-self.alpha)**(np.max(t-2, 0)) * self.Y(1, t-1)
    #     thetaT += (1 - self.alpha) / self.V(t-1) * self.alpha * np.sum([(1-self.alpha)**s * self.V(t-2-s) * self.Y(t-1-s, t-1) for s in range(t-2)])
    #     return thetaT.reshape(-1, 1)



# for jacobi cross


# [1]
# x2 x1 0 0 0 0    
# x3 0 x1 0 0 0 
# x4 0 0 x1 0 0
# x5 0 0 0 x1 0
# x6 0 0 0 0 x1

# [2]
# 0 x3 x2 0 0 0    
# 0 x4 0 x2 0 0
# 0 x5 0 0 x2 0
# 0 x6 0 0 0 x2

# [3]
# 0 0 x4 x3 0 0
# 0 0 x5 0 x3 0
# 0 0 x6 0 0 x3

# [4]
# 0 0 0 x5 x4 0
# 0 0 0 x6 0 x4

# [5]
# 0 0 0 0 x6 x5




# dim

# in pairs:

# fdim = dim*(dim-1)//2 

# blocks: dim-1 

# block sizes: dim-1, dim-2, dim-3 ... 1


# in threesomes:

# fdim = dim*(dim-1)*(dim-2)//6

# blocks: dim-2

# block sizes: binoms of dim-1, dim-2, dim-3 ... 1
# subblock sizes: as in pairs

# {1}: 4 blocks from 5 in pairs: (1) - 4, (2) - 3, (3) - 2, (4) - 1
# (1) = x2*[1][1:] + x1[2]
# (1)[size([2]):] /= 2


# (2) = x3*[1][2:] + x1*[3]
# (2)[size([3]):] /= 2


# (3) = x4*[1][3:] + x1*[4]
# (3)[size([4]:)] /= 2


# (4) = x5*[1][4:] + x1*[5]
# (4)[size([5]:)] /= 2


# {1}
# ----
# (1)
# x2x3 x1x3 x1x2 0 0 0
# x2x4 x1x4 0 x1x2 0 0
# x2x5 x1x5 0 0 x1x2 0
# x2x6 x1x6 0 0 0 x1x2

# (2)
# x3x4 0 x1x4 x1x3 0 0
# x3x5 0 x1x5 0 x1x3 0
# x3x6 0 x1x6 0 0 x1x3

# (3)
# x4x5 0 0 x1x5 x1x4 0
# x4x6 0 0 x1x6 0 x1x4

# (4)
# x5x6 0 0 0 x1x6 x1x5
# ----



# {2}: 3 blocks from pairs: (1) - 3, (2) - 2, (3) - 1

# (1) = x*3[2][1:] + x2*[3]
# (1)[size([3]:)] /= 2

# (2) = x*4[2][2:] + x2*[4]
# (2)[size([4]:)] /= 2

# (3) = x*5[2][3:] + x2*[5]
# (3)[size([5]:)] /= 2

# {2}
# ---
# (1)
# 0 x3x4 x2x4 x2x3 0 0
# 0 x3x5 x2x5 0 x2x3 0
# 0 x3x6 x2x6 0 0 x2x3

# (2)
# 0 x4x5 0 x2x5 x2x4 0
# 0 x4x6 0 x2x6 0 x2x4

# (3)
# 0 x5x6 0 0 x2x6 x2x5
# ---


# {3}
# ---
# 0 0 x4x5 x3x5 x3x4 0
# 0 0 x4x6 x3x6 0 x3x4

# 0 0 x5x6 0 x3x6 x3x5
# ---

# {4}
# ---
# 0 0 0 x5x6 x4x6 x4x5
# ---



# class TrigBasis:
#     def __init__(self, dim):
#         self.dim = dim

#     def compute_jacobi(self, x):

#         def grad_i(x, d):
#             coef = (d+1) // 2
#             sign = 1 if d % 2 else -1
#             func = np.cos if d % 2 else np.sin
#             return sign * coef * func(coef * x)
            
#         current_grad = np.zeros(self.dim)
#         if self.dim == 1:
#             current_grad[0] = grad_i(x, 1)
#         else:
#             for i in range(self.dim):
#                 current_grad[i] = grad_i(x[i], i)

#         return current_grad.reshape((self.dim, 1))

#     def compute_laplacian(self, x):

#         def second_derivative(x, d):
#             coef = (d+1) // 2
#             func = np.sin if d % 2 else np.cos
#             return (-1) * coef**2 * func(coef * x)
        
#         current_laplacian = np.zeros(self.dim) 
#         if self.dim == 1:
#             current_laplacian[0] = second_derivative(x, 1) 
#         else:
#             for i in range(self.dim):
#                 current_laplacian[i] = second_derivative(x[i], i)

#         return current_laplacian.reshape((self.dim, 1))




# without cross and wrong (vector gradient instead of jacobi matrix)

# class PolyBasis:
#     """ dim_x = 1
#     x, x^2, x^3, ... x^d

#     dim_x = k
#     x_1, ..., x_k, x_1^2, ..., x_k^2, ..., x_1^k, ..., x_k^k
#     """
#     def __init__(self, dim, dim_x=1):
#         self.dim = dim
#         self.dim_x = dim_x
#         self.name = 'poly'
#         self.include_cross=include_cross

#     def _make_poly_grad(self, x, dim):
#         poly = np.empty((dim, self.dim_x, 1))
#         for i in range(dim):
#             poly[i:i+self.dim_x] = np.diag(x ** i * (i+1))
#         return poly.reshape(-1, 1)
    
#     def compute_jacobi(self, x):
#         grad = self._make_poly_grad(x, self.dim)
#         return grad.reshape((self.dim, x.shape[0], 1))
        

#     def compute_laplacian(self, x):
#         laplacian = np.repeat((np.arange(1, self.dim) + 1), x.shape[0]).reshape(-1, 1) * self._make_poly_grad(x, self.dim-1)
#         laplacian = np.concatenate([np.zeros((x.shape[0], 1)), laplacian])

#         return laplacian.reshape((self.dim * x.shape[0], 1))
        
