import numpy as np

class LinearRegressor:
    """
    Linear regressor
    """
    def __init__(norm=2):
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
    
    def loss(self,  phi:np.ndarray, theta:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        The lost per sample
        """
        return self.error(phi, theta, y) @ self.error(phi, theta, y)
    
    def total_loss(self,  phi:np.ndarray, theta:np.ndarray, y:np.ndarray, N:int) -> np.ndarray:
        """
        The total loss
        """
        return sum(self.loss(phi, theta, y)) / (2 * N)
    
    def dlossdtheta(self, phi:np.ndarray, theta:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        derivative of the loss per sample
        """
        return -2 * self.error(phi, theta, y).T @ phi
    
    def dtot_loss(self,  phi:np.ndarray, theta:np.ndarray, y:np.ndarray, N:int) -> np.ndarray:
        """
        derivative of the total loss
        """
        return sum(self.dlossdtheta(phi, theta, y)) / (2 * N)
    
    def minibatch(self, size:int, msk:list, phi:np.ndarray, theta:np.ndarray, y:np.ndarray) -> np.ndarray:
        np.random.shuffle(msk)
        msk[:size]
        return self.dtot_loss(phi=phi[msk,:], theta=theta, y=y)
    
    def stoc_dloss(self, high:int, phi:np.ndarray, theta:np.ndarray, y:np.ndarray) -> np.ndarray:
        idx = np.random.randint(low=0, high=high)
        return self.dlossdtheta(phi=phi[idx,:], theta=theta, y=y)
    
    def gradient_descent(self, args, func, jac:function, learn_rate:float=0.5,
                          eps_conv:float=1e4, max_iter:int=50, history=False):
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
        for i in range(max_iter):
            if history:
                J_history.append(func(args))
                theta_history.append(args)
            grad = jac(args)
            if np.linalg.norm(grad) > eps_conv:
                args = args - learn_rate * grad
            else:
                print(f"converged at step {i}")
        return args, func(args), J_history, theta_history

    def fit(self, X:np.ndarray, y, seed_num:int=0, method:int=0, mb_size=10, bias:bool=False,
        amp:float=1.0, learn_rate:float=0.5, eps_conv:float=1e4, max_iter:int=50, history=False
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
        history: bool
            if true will record and return args and tot_loss all along gradient descent.
            The default is False
        """
        np.random.seed(seed_num)
        N, D = X.shape
        if bias:
            X = np.hstack((X, np.ones(N)).reshape(-1,1))
            theta = amp * np.random.randn(D+1)
        else:
            theta = amp * np.random.randn(D)

        if method.lower() in ("batch or b-gd"):
            jac = lambda params: self.dtot_loss(phi=X, theta=params, y=y, N=N)
        elif method.lower() in ("mb-gd","mb"):
            msk = [i for i in range(N)]
            jac = lambda params : self.minibatch(size=mb_size, msk=msk, phi=X, theta=params, y=y)
        elif method.lower() in ("sgd", "s-gd"):
            jac = lambda params : self.stoc_dloss(high=N, phi=X, theta=params, y=y)
        
        J = lambda params : self.total_loss(phi=X, theta=params, y=y, N=N) #total Loss function redefined
        
        min_args, min_val, J_hist, args_hist = self.gradient_descent(self, args=theta, func=J, jac=jac, learn_rate=learn_rate,
                                                                    eps_conv=eps_conv, max_iter=max_iter, history=history)
        self.coef_ = min_args[:D]
        self.intercept_ = min_args[-1]

        if history:
            return min_args, min_val, J_hist, args_hist

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
