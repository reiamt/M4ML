import numpy as np
from cvxopt import matrix, solvers
from kernels import linear, gaussian


class hard_SVM(object):
    def __init__(self, kernel=linear):
        self.kernel = kernel

    def fit(self, X, y, alpha_tol):
        N, D = X.shape

        #calc outer product of y
        y_tmp = np.zeros((y.shape[0],y.shape[0]))
        for i in range(0,y.shape[0]):
            for j in range(0,y.shape[0]):
                y_tmp[i,j] = y[i]*y[j]

        #calc Gram matrix
        Gram = np.zeros((y.shape[0], y.shape[0]))
        for i in range(0,X.shape[0]):
            for j in range(0,X.shape[0]):
                Gram[i,j] = self.kernel(X[i],X[j])
        
        P = matrix(y_tmp*Gram) #positive defininte matrix which will be minimized
        q = matrix(-np.ones([N, 1])) #np array with ones

        G = matrix(-np.eye(N))
        h = matrix(np.zeros(N))

        A = matrix(y.reshape(1,-1))
        b = matrix(np.zeros(1))

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b) #solve the optimization problem
        alphas = np.array(solution['x']).flatten()

        #find support vectors
        support_vecs = (alphas > alpha_tol)

        self.alphas = alphas[support_vecs] #save support vectors
        self.sv = X[support_vecs] #save coordinates of support vectors
        self.sv_y = y[support_vecs] #save labels of support vectors

        #calc weight vector
        w = np.dot(X.T, alphas * y)
        print(f'weights are {w}')

        #calc biases
        biases = y[support_vecs] - np.dot(X[support_vecs, :], w)
        b = np.sum(alphas[support_vecs]*biases) / np.sum(alphas[support_vecs])
        print(f'bias is {b}')
        return alphas, w, b


class soft_SVM(object):
    def __init__(self, kernel=linear, C=1.0):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y, alpha_tol):
        N, D = X.shape

        #calc outer product of y
        y_tmp = np.zeros((y.shape[0],y.shape[0]))
        for i in range(0,y.shape[0]):
            for j in range(0,y.shape[0]):
                y_tmp[i,j] = y[i]*y[j]

        #calc Gram matrix
        Gram = np.zeros((y.shape[0], y.shape[0]))
        for i in range(0,X.shape[0]):
            for j in range(0,X.shape[0]):
                Gram[i,j] = self.kernel(X[i],X[j])
        
        P = matrix(y_tmp*Gram) #positive defininte matrix which will be minimized
        q = matrix(-np.ones([N, 1])) #np array with ones

        G = matrix(np.vstack((-np.eye(N),np.eye(N))))
        h = matrix(np.hstack((np.zeros(N),np.ones(N)*self.C)))
        
        A = matrix(y.reshape(1,-1))
        b = matrix(np.zeros(1))

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b) #solve the optimization problem
        alphas = np.array(solution['x']).flatten()

        #find support vectors
        support_vecs = (alphas > alpha_tol)

        self.alphas = alphas[support_vecs] #save support vectors
        self.sv = X[support_vecs] #save coordinates of support vectors
        self.sv_y = y[support_vecs] #save labels of support vectors

        #calc weight vector
        if self.kernel == linear:
            w = np.dot(X.T, alphas * y)
            print(f'weights are {w}')
        else:
            w = None

        #calc biases
        biases = y[support_vecs] - np.dot(X[support_vecs, :], w)
        b = np.sum(alphas[support_vecs]*biases) / np.sum(alphas[support_vecs])
        print(f'bias is {b}')

        return alphas, w, b

    def project(self, X):
        return np.dot(X, self.w) + self.b
        

    def predict(self, X):
        return np.sign(self.project(X))


class sgd_SVM(object):
    def __init__(self, C=1.0):
        self.C = C
        self.w = 0
        self.b = 0

    #define the hingeloss function, which we want to optimize
    def hingeLoss(self, w, b, X, y):
        hloss = 0.0
        hloss = hloss + 0.5*np.dot(w, w.T)
        N, D = X.shape
        for i in range(N):
            #print(f'w is {w} and X_i transpose {X.T[1]}')
            hloss = self.C *max(0, (1-y[i] * (np.dot(w, X[i]) + b))) +hloss
        return hloss[0][0]
    
    def fit(self, X, y, batch_size=96, lr=0.001, iterations=5000):
        #get number of samples, number of features and regularization parameter C
        N, D = X.shape
        C = self.C
        #inizialize weight vector and bias
        w = np.zeros((1, D))
        b = 0
        #initialize loss list
        hlosses = []
        for i in range(iterations):
        #Training Loop
            ids = np.arange(N)
            np.random.shuffle(ids)
            l = self.hingeLoss(w, b, X, y)
            hlosses.append(l)
            #Gradient descent
            for batch in range(0, N, batch_size):
                #initialize gradients for minibach
                dw = 0
                db = 0
                #enter minibach optimization
                for j in range(batch, batch + batch_size):
                    if j < N: 
                        i = ids[j]
                        if y[i] * (np.dot(w, X[i].T) + b) <= 1:
                            #gradient of hingeloss wrt w
                            dw = dw + C * y[i] * X[i]
                            #gradient of hingeloss wrt b
                            db = db + C * y[i]
                #update parameters
                w = w - lr*w + lr*dw
                b = b + lr*db
        return w.flatten(), b

class soft_SVM_gauss(object):
    def __init__(self, kernel=linear, C=1.0):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y, alpha_tol):
        N, D = X.shape

        #calc outer product of y
        y_tmp = np.zeros((y.shape[0],y.shape[0]))
        for i in range(0,y.shape[0]):
            for j in range(0,y.shape[0]):
                y_tmp[i,j] = y[i]*y[j]

        #calc Gram matrix
        Gram = np.zeros((y.shape[0], y.shape[0]))
        for i in range(0,X.shape[0]):
            for j in range(0,X.shape[0]):
                Gram[i,j] = self.kernel(X[i],X[j])
        
        P = matrix(y_tmp*Gram) #positive defininte matrix which will be minimized
        q = matrix(-np.ones([N, 1])) #np array with ones

        G = matrix(np.vstack((-np.eye(N),np.eye(N))))
        h = matrix(np.hstack((np.zeros(N),np.ones(N)*self.C)))
        
        A = matrix(y.reshape(1,-1))
        b = matrix(np.zeros(1))

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b) #solve the optimization problem
        self.alphas = np.array(solution['x']).flatten()

        #find support vectors
        support_vecs = (self.alphas > alpha_tol)
        ind = np.arange(len(self.alphas))[support_vecs]
        self.alphas = self.alphas[support_vecs] #save support vectors
        self.sv = X[support_vecs] #save coordinates of support vectors
        self.sv_y = y[support_vecs] #save labels of support vectors

    
        self.w = None

        #calc biases
        self.bias = 0
        for n in range(len(self.alphas)):
            self.bias = self.bias + self.sv_y[n]
            self.bias = self.bias - np.sum(self.alphas * self.sv_y * Gram[ind[n],support_vecs])
        self.bias = self.bias / len(self.alphas)

        return self.alphas, self.w, self.bias

    def project(self, X):
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                tmp = 0
                for alpha, sv_y, sv in zip(self.alphas, self.sv_y, self.sv):
                    tmp = tmp + alpha * sv_y * self.kernel(X[i], sv)
                y_predict[i] = tmp
            return y_predict + self.bias