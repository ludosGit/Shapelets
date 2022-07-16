import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers

# https://pythonprogramming.net/soft-margin-kernel-cvxopt-svm-machine-learning-tutorial/
# https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
## kernel functions
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

class polynomial_kernel():
    def __init__(self, p=3):
        self.p = p
    
    def kernel(self, x, y):
        return (1 + np.dot(x, y)) ** self.p

class gaussian_kernel():
    def __init__(self, sigma=3):
        self.sigma = sigma
    
    def kernel(self, x, y):
        return np.exp(-linalg.norm(x-y)**2 / (2 * (self.sigma ** 2)))

# g = gaussian_kernel(sigma=2).kernel
# g(2,3)



class SVDD(object):

    def __init__(self, kernel='linear', C=0.9, zero_center = False, p=3, sigma=5, tol=1e-6, verbose=True):
        '''
        Class of SVDD
        @param kernel:
        @param C:
        @param zero_center: whether to fix the center = 0
        @param p: only for poly kernel
        @param sigma: only for rbf kernel
        @param tol: tolerance for considering the lagrange multipliers alpha > 0 or < C
        '''
        if kernel == 'linear':
            self.kernel = linear_kernel

        if kernel == 'poly':
            self.kernel = polynomial_kernel(p).kernel

        if kernel == 'rbf':
            self.kernel = gaussian_kernel(sigma).kernel

        self.C = C
        if self.C is not None: self.C = float(self.C)

        self.zero_center = zero_center
        self.tol = tol
        self.gram = None
        self.radius = None
        self.center = None
        self.verbose = verbose

    def compute_radius(self):
        '''RADIUS (squared)
        # take any of the boundary observations
        # should be equal for each of the support vectors with 0 < alpha < C
        '''
        # take one support vector on the boundary
        b = self.boundary_sv[0]
        tmp1 = 0
        tmp2 = 0
        for i in range(len(self.alpha)):
            tmp1 += self.alpha[i] * self.kernel(self.sv[i],b)
            for j in range(len(self.alpha)):
                tmp2 += self.alpha[i] * self.alpha[j] * self.kernel(self.sv[i],self.sv[j])
        radius = self.kernel(b, b) - 2*tmp1 + tmp2
        return radius

    def fit(self, X):
        cvxopt.solvers.options['show_progress'] = False
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])
        self.gram = K

        ####### Standard SVDD
        if not self.zero_center:
            P = cvxopt.matrix(2 * K)
            q = cvxopt.matrix(np.diag(K) * -1)
            # tmp is a temporary variable
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
            # last condition is sum(alpha_i) = 1
            A = cvxopt.matrix(np.ones((1, n_samples)))
            b = cvxopt.matrix(1.0)
    
            # solve QP problem
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    
            # Lagrange multipliers
            alpha = np.ravel(solution['x'])
            self.total_alpha = alpha
            # Support vectors have non zero lagrange multipliers
            sv = alpha > self.tol
            self.alpha = alpha[sv]
            self.sv = X[sv]
    
            # NOTE: alpha has length equal to number of data points
            # self.alpha saves only the support vectors!!!
    
            # save also the indices of the support vectors
            self.sv_index = np.arange(len(alpha))[sv]
    
            # do the same for boundary support vectors
            boundary_sv = np.logical_and(alpha < self.C - self.tol, alpha > self.tol)
            self.boundary_sv = X[boundary_sv]
            self.boundary_sv_index = np.arange(len(alpha))[boundary_sv]
    
            # CENTER
            self.center = np.zeros(n_features)
            # consider only support vectors!!
            for i in range(len(self.alpha)):
                self.center += self.alpha[i] * self.sv[i]
            
            # RADIUS
            self.radius = self.compute_radius()
            if self.verbose:
                print("%d support vectors out of %d points" % (len(self.alpha), n_samples))
                print(f'Solution found with center in {self.center} and radius {np.sqrt(self.radius)}')

        ###### Modified SVDD with the center in the ORIGIN
        else:
            c = cvxopt.matrix(np.diag(K) * -1)
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
            # last condition is sum(alpha_i) = 1
            A = cvxopt.matrix(np.ones((1, n_samples)))
            b = cvxopt.matrix(1.0)

            # Solve LP problem
            solution = cvxopt.solvers.lp(c, G, h, A, b)
            
            # Lagrange multipliers
            alpha = np.ravel(solution['x'])
            self.total_alpha = alpha
            # Support vectors have non zero lagrange multipliers
            sv = alpha > self.tol
            self.alpha = alpha[sv]
            self.sv = X[sv]
    
            # NOTE: total_alpha has length equal to number of data points
            # self.alpha saves only the support vectors multipliers!!!
    
            # save also the indices of the support vectors
            self.sv_index = np.arange(len(alpha))[sv]
    
            # do the same for boundary support vectors
            boundary_sv = np.logical_and(alpha < (self.C - self.tol), alpha > self.tol)
            self.boundary_sv = X[boundary_sv]
            self.boundary_sv_index = np.arange(len(alpha))[boundary_sv]
    
            # CENTER is zero
            self.center = 0.0
            
            # RADIUS is equal to the norm of each of boundary sv in this case
            b = self.boundary_sv[0]
            # remind: the radius is squared!!
            self.radius = self.kernel(b, b)
            if self.verbose:
                print("%d support vectors out of %d points" % (len(self.alpha), n_samples))
                print(f'Solution found with center in {self.center} and radius {np.sqrt(self.radius)}')
        
    def decision_function(self, X):
        '''
        Signed distance to the separating hypersphere
        '''
        # assume the kernel function does matrix computations
        # aka can take in input a batch of vectors stacked by column
        n_samples, _ = X.shape
        if not self.zero_center:
            tmp1 = np.zeros(n_samples)
            tmp2 = np.zeros(n_samples)
            for i in range(n_samples):
                tmp1[i] = self.kernel(X[i],self.center)
                tmp2[i] = self.kernel(X[i], X[i])
            return self.radius - tmp2 + 2 * tmp1 - self.kernel(self.center, self.center)

        else:
            tmp1 = np.zeros(n_samples)
            for i in range(n_samples):
                tmp1[i] = self.kernel(X[i], X[i])
            return self.radius - tmp1

    def predict(self, X):
        y = np.sign(self.decision_function(X))
        y[y==0] = 1 # set the boundary points as anomalies
        return y



# ########## TEST FUNCTIONS

# def compute_radius(b, alpha, sv, kernel=linear_kernel):
#     '''RADIUS (squared)
#         b: boundary support vector
#         # take any of the boundary observations
#         # should be equal for each of the support vectors with 0 < alpha < C
#     '''
#     tmp1 = 0
#     tmp2 = 0
#     for i in range(len(alpha)):
#         tmp1 += alpha[i] * kernel(sv[i],b)
#         for j in range(len(alpha)):
#             tmp2 += alpha[i] * alpha[j] * kernel(sv[i],sv[j])
#     print('kernel b', kernel(b, b))
#     print('tmp1', tmp1)
#     print('tmp2', tmp2)
#     radius = kernel(b, b) - 2*tmp1 + tmp2
#     return radius

# b = np.array([[1,2,3], [1,2,3]])

# alpha = np.array([0.5, 0.5])
# sv = np.array([[1,2,4], [2,3,4]])
# compute_radius(b, alpha, sv)
# ## should be ok
