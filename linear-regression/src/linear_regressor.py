import numpy as np

class LinearRegressor:
    """
    Linear regressor
    """

    # important functions
    @staticmethod
    def lin_func(phi:np.ndarray, theta:np.ndarray) -> np.ndarray:
        """
        Linear function
        ---------
        Parameters
        phi: (N, D) np.ndarray of the data set
            N samples, D features
        theta: (D, 1) np.ndarray of the data set
        ---------
        return
            (N, 1) np.ndarray
        """
        return phi @ theta

    def cost(self, phi:np.ndarray, theta:np.ndarray, y:np.ndarray):
        """
        Compute the total cost
        -------------
        Parameters:
        phi : (N, D) matrix of input features.More precisely phi is
            the basis function of our linear model.
        theta : (D, 1) np.ndarray of parameters (that is coefficients and intercept)
                D being the dimension of the dataset
                Note : with bias D -> D + 1
        y: (N,1 np.ndarray of the target variable
        N:int,
            number of row in phi
        -------------
        returns
            float (the total cost)
        """
        loss_ = np.square(self.lin_func(phi, theta) - y)
        return np.mean(loss_)

    @staticmethod
    def dJdtheta(phi:np.ndarray, f_wb:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        derivative of the total cost
        --------
        
        phi: (N, D) np.ndarray of the data set (features only)
        f_wb: (N, 1) np.ndarray of the base function (the linear function)
        y: (N, 1) np.ndarray of the target
        --------
        return
            (D, 1) np.array
        """
        return 2 * phi.T @ (f_wb - y) / f_wb.shape[0]
    
    def minibach_gd(self, phi:np.ndarray, f_wb:np.ndarray, y:np.ndarray,
                    msk:np.ndarray, size:int) -> np.ndarray:
        """
        Compute mini-batch gradient
        --------
        phi: (N, D) np.ndarray of the data set (features only)
        f_wb: (N, 1) np.ndarray of the base function (sigmoid here)
        y: (N, 1) np.ndarray of the target
        msk: list of integer from 0 to N-1
        size: int, chunck size of the array phi on which to compute the gradient.
        --------
        return
            (D, 1) np.array
        """
        msk_ = np.random.choice(msk, size, replace=False)
        return - self.dJdtheta(phi[msk_], f_wb[msk_], y[msk_])
    
    def stoc_gd(self, phi:np.ndarray, f_wb:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        Compute stochastic gradient by randomly select a sample in the data set
        --------
        phi: (N, D) np.ndarray of the data set (features only)
        f_wb: (N, 1) np.ndarray of the base function (sigmoid here)
        y: (N, 1) np.ndarray of the target
        return
            (D, 1) np.array
        """
        i = np.random.randint(0, y.shape[0])
        return - self.dJdtheta(phi[i:i+1], f_wb[i:i+1], y[i:i+1])


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
        f_wb = func(args)
        while func((args + step_size * deltax)) > f_wb + alpha * step_size * grad.T @ deltax:
            step_size *= beta
        return step_size

    def gradient_descent(self, args, func, jac, learn_rate:float=0.5,
                          eps_conv:float=1e4, max_iter:int=50, history=False, step_traj:int=100,
                         backtrack:bool=False, beta_bt:float=0.5, alpha_bt:float=0.1):
        """
        simple gradient descent, i.e fixed learn_rate and no momemtum
        ---------
        parameters:
        args: np.ndarray
            (D,) (or (D+1,) if bias) of parameter to optimize
        func: function
            the redifined cost function with only one variable (np.array of parameters)
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
        J_hist = []
        theta_hist = []
        if backtrack:
            learn_rate = 1.0
        for i in range(max_iter):
            if history and i%(step_traj) == 0:
                J_hist.append(func(args))
                theta_hist.append(args)
            #line search
            grad = jac(args).reshape(-1, 1)
            norm_grad = np.linalg.norm(grad)
            if backtrack:
                # put -grad instead of grad because - in the def (here) of jac 
                learn_rate = self.backtracking(func=func, args=args, grad=-grad,
                                               deltax=grad, step_size=learn_rate, beta=beta_bt,
                                               alpha=alpha_bt)

            args += learn_rate * grad
            if norm_grad <= eps_conv:
                print(f"converged at step {i} grad {norm_grad}")
                break
            elif i == max_iter - 1:
                print(f"Max iteration reached, no convergence : {norm_grad}")

        return args, func(args), J_hist, theta_hist

    def fit(self, X:np.ndarray, y, seed_num:int=0, method:int=0, mb_size=10, bias:bool=False,
        amp:float=1.0, learn_rate:float=0.5, eps_conv:float=1e-4, max_iter:int=50,
        backtrack=False, beta_bt=0.5, alpha_bt=0.1, history=False, step_traj:int=10
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
            The default is 1e-4
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
        step_traj:int
            will save J and theta each step_traj iteration
        ------------
        return
            if history then return
            np.ndarray, float, list, list
            min_args, min_val, J_hist, args_hist
            optimal values of parameters, minimum values of the cost,
            trajectory of the cost, and trajectory of the parameters.
        """
        np.random.seed(seed_num)
        N = X.shape[0]
        # important deal with case of dataset with only 1 feature
        # also make that .T operation are possible
        if sum(X.shape) == N:
            X = X.reshape((N,1))
            D=1
        else:
            D = X.shape[1]
        if sum(y.shape) == N:
            y = y.reshape(-1,1)

        # incorporate bias in the parameters
        # hence add a new col in X full with 1.0
        if bias:
            X = np.hstack((X, np.ones(N).reshape(-1,1)))
            theta = amp * np.random.randn(D+1).reshape(-1,1)
        else:
            theta = amp * np.random.randn(D).reshape(-1,1)
    
        f_wb = lambda params : self.lin_func(X, params) #lin func defined as only func of theta
        
        # selection of the approach to compute the gradient
        if method == 0:
            jac = lambda params: - self.dJdtheta(X, f_wb(params), y) # batch GD
        elif method == 1:
            msk = [i for i in range(N)]
            jac = lambda params: self.minibach_gd(X, f_wb(params), y, msk, mb_size) # mini batch
        else :
            jac = lambda params : self.stoc_gd(X, f_wb(params), y) # stochastic
        
        J = lambda params : self.cost(phi=X, theta=params, y=y) #total Loss function redefined

        min_args, min_val, J_hist, args_hist = self.gradient_descent(args=theta, func=J, jac=jac, learn_rate=learn_rate,
                                                                     eps_conv=eps_conv, max_iter=max_iter, 
                                                                     backtrack=backtrack, alpha_bt=alpha_bt, beta_bt=beta_bt,
                                                                     history=history, step_traj=step_traj)
        self.coef_ = min_args[:D]
        self.intercept_ = min_args[-1][0]

        if history:
            return min_args, min_val, J_hist, args_hist

    def predict(self, X):
        return X @ self.coef_ + self.intercept_