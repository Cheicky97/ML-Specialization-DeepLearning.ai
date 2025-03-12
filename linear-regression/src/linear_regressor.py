import numpy as np

class LinearRegressor:
    """
    Linear regressor
    """
    def __init__(self):
        """
        """
        pass

    def error(self,  phi:np.ndarray, theta:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        -------------
        Parameters:
        theta : (D, 1) np.ndarray of parameters (that is coefficients and intercept)
                D being the dimension of the dataset
                Note : with bias D -> D + 1
        phi : (N, D) matrix of input features
        y: (N,1 np.ndarray of the target variable
        -------------
        returns
            (N,1) np.array of the matrix product of phi and theta
        """
        return y - phi @ theta
    
    def loss(self,  phi:np.ndarray, theta:np.ndarray, y:np.ndarray, N:int) -> np.ndarray:
        """
        Total lost per sample
        """
        return (self.error(phi, theta, y).T @ self.error(phi, theta, y))[0] / (2 * N)
    
    def dlossdtheta(self, phi:np.ndarray, theta:np.ndarray, y:np.ndarray, N) -> np.ndarray:
        """
        derivative of the loss per sample
        """
        return -1.0 * (self.error(phi, theta, y).T @ phi) / N
    
    def minibatch(self, size:int, msk:list, phi:np.ndarray, theta:np.ndarray, y:np.ndarray) -> np.ndarray:
        np.random.shuffle(msk)
        msk[:size]
        return self.dlossdtheta(phi=phi[msk,:], theta=theta, y=y, N=msk.shape[0])
    
    def stoc_dloss(self, high:int, phi:np.ndarray, theta:np.ndarray, y:np.ndarray) -> np.ndarray:
        idx = np.random.randint(low=0, high=high)
        return self.dlossdtheta(phi=phi[idx,:], theta=theta, y=high, N=1)
    
    def backtracking(self, func, args, grad:np.ndarray, deltax:np.ndarray, step_size:float, beta:float=0.5, alpha:float=0.1)->float:
        """
        Line search algorithm (See chapter 3 of the book 
        "Convex Optimization" by Stephen Boyd & Lieven Vandenberghe)
        ----------
        Parameter
        args:
            arguments of function to minimize
        grad: np.ndarray
            the gradient vector
        deltax: np.ndarray
            a steepest direction
        step_size: float or np.ndarray
            the steepest step size
        beta: float or np.ndarray
            factor scaling step_size
            is often chosen to be between 0.1 (which corresponds
            to a very crude search) and 0.8 (which corresponds
            to a less crude search).
            See page 466 of "Convex Optimization" by Stephen Boyd & Lieven Vandenberghe
        alpha: float or np.ndarray
            a parameter of the backtracking
            Values of alpha typically range between 0.01 and 0.3
            The default is 0.1
        """
        while func((args + step_size * deltax)) > func(args) + alpha * step_size * grad.T @ deltax:
            step_size *= beta
        return step_size
    
    def gradient_descent(self, args, func, jac, learn_rate:float=0.5,
                          eps_conv:float=1e4, max_iter:int=50, history=False,
                         backtrack:bool=False, beta_bt:float=0.5, alpha_bt:float=0.1):
        """
        simple gradient descent, i.e fixed learn_rate and no momemtum
        ---------
        parameters:
        args: np.ndarray
            (D,) (or (D+1,) if bias) of parameter to optimize
        func: function
            the total loss only function of these parameters
        jac = np.ndarray
            same shape as args, contains gradient of func with respect to args
        learn_rate: float
            step_size, to be chosen wisely
        eps_conv: float
            convergence criterion on the norm of the gradient 
        max_iter: int
            maximum step of iteration
        history: bool
            if true will record and return args and tot_loss all along gradient descent
        """
        J_history = []
        theta_history = []
        if backtrack:
            learn_rate = 1.0
        for i in range(max_iter):
            if history:
                J_history.append(func(args))
                theta_history.append(args)
            #line search
            grad = jac(args).reshape(-1, 1)
            if backtrack:
                learn_rate = self.backtracking(func=func, args=args, grad=grad,
                                  deltax=-grad, step_size=learn_rate, beta=beta_bt, alpha=alpha_bt)

            if np.linalg.norm(grad) > eps_conv:
                args -= learn_rate * grad / np.linalg.norm(grad)
            else:
                print(f"converged at step {i}")
                break
        return args, func(args), J_history, theta_history

    def fit(self, X:np.ndarray, y, seed_num:int=0, method:int=0, mb_size=10, bias:bool=False,
        amp:float=1.0, learn_rate:float=0.5, eps_conv:float=1e-4, max_iter:int=50,
        backtrack=False, beta_bt=0.5, alpha_bt=0.1, history=False
        ):
        """
        X: np.ndarray
            (N,D) array training dataset with N samples and D features
        y: np.ndarray
            (N,) 1D array of the label data
        seed_num : int, fix the seed, so the experience is repeatible
        method:int
            method to be employed for GD
            0 -> for batch GD
            1 -> for minibatch GD
            2 -> for stochastic GD
            The default is 0.
        mb_size: int
            size of the minibatch.
            Note that 1 < mb_size < N
            since mb_size <-> batch GD and mb_size = 1 <-> stochastic GD.
            The default is 10.
        bias: bool
            if true, bias in the linear model,
            i.e. take into accound the intercept as parameter to optimize.
            The default is False.
        amp: float
            amplititude of the random initialization of parameters for GD
            The default is 1.0
        learn_rate: float
            step_size, to be chosen wisely. The default is 0.5.
        eps_conv: float
            convergence criterion on the norm of the gradient.
            The default is 1e4
        max_iter: int
            maximum step of iteration.
            The default is 50.
        backtrack: bool
            if True backtracking line search is activated
            The default is False.
        beta_bt: float,
            Scale the step size while the Armijo's condition is fullfilled
            The default is 0.5
        alpha_bt: float
            
            The default is 0.1
        history: bool
            if true will record and return args and tot_loss all along gradient descent.
            The default is False
        """
        np.random.seed(seed_num)
        N = X.shape[0]
        if sum(X.shape) == N:
            X = X.reshape((N,1))
            D=1
        else:
            D = X.shape[1]
        y = y.reshape(-1,1)
        if bias:
            X = np.hstack((X, np.ones(N).reshape(-1,1)))
            theta = amp * np.random.randn(D+1).reshape(-1,1)
        else:
            theta = amp * np.random.randn(D).reshape(-1,1)

        if method == 0:
            jac = lambda params: self.dlossdtheta(phi=X, theta=params, y=y, N=N)
        elif method == 1:
            msk = [i for i in range(N)]
            jac = lambda params : self.minibatch(size=mb_size, msk=msk, phi=X, theta=params, y=y)
        elif method == 2:
            jac = lambda params : self.stoc_dloss(high=N, phi=X, theta=params, y=y)
        
        J = lambda params : self.loss(phi=X, theta=params, y=y, N=N) #total Loss function redefined

        min_args, min_val, J_hist, args_hist = self.gradient_descent(args=theta, func=J, jac=jac, learn_rate=learn_rate,
                                                                    eps_conv=eps_conv, max_iter=max_iter, 
                                                                     backtrack=backtrack, alpha_bt=alpha_bt, beta_bt=beta_bt,
                                                                     history=history)
        self.coef_ = min_args[:D]
        self.intercept_ = min_args[-1][0]

        if history:
            return min_args, min_val, J_hist, args_hist

    def predict(self, X):
        return X @ self.coef_ + self.intercept_