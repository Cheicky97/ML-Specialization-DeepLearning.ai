import numpy as np

class LogisticRegression:
    def __init__(self):
        pass

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

    @staticmethod
    def sigmoid(z) -> np.ndarray:
        """
        Sigmoid of z
        ------
        Parameter
        z : (N, 1) np.ndarray
        ------
        return
            (N, 1) np.ndarray
        """
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def loss(f_wb:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        Total cost function
        -------
        Parameters
        f_wb : (N, 1) np.ndarray is the sigmoid function
        y: (N, 1) np.ndarray is the target
        -------
        return
            (N, 1) np.array
        """
        return -y * np.log(f_wb) - (1 - y) * np.log(1 - f_wb)

    def cost(self, phi:np.ndarray, theta:np.ndarray, y:np.ndarray) -> float:
        """
        The total cost function
        ------
        Parameters
        phi: (N, D) np.ndarray of the data set (features only)
        theta: (D,1) np.ndarray of the parameters
        loss: (N, 1) np.ndarray is the loss
        ------
        return
            float
        """
        # logistic function
        f_wb = self.sigmoid(self.lin_func(phi,theta))
        # loss (i.e. individual cost)
        loss_ = self.loss(f_wb, y)
        return (loss_.T @ loss_)[0] / loss_.shap[0]

    @staticmethod
    def dJdtheta(phi:np.ndarray, f_wb:np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        derivative of the total cost
        --------
        
        phi: (N, D) np.ndarray of the data set (features only)
        f_wb: (N, 1) np.ndarray of the base function (sigmoid here)
        y: (N, 1) np.ndarray of the target
        --------
        return
            (D, 1) np.array
        """
        return phi.T @ (f_wb - y) / f_wb.shape[0]

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
        np.random.shuffle(msk)
        return - self.dJdtheta(phi[msk[:size], :], f_wb, y)

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
        return - self.dJdtheta(phi[i, :], f_wb, y)

    def gradient_descent(self, args:np.ndarray, phi:np.ndarray, y:np.ndarray, gra_dir,
                        gamma:float, max_iter:int, eps_conv:float,
                            history:bool=False) -> np.ndarray:
        """
        Compute gradient descent with fixed step_size
        ---------
        Parameters
        args: (D, 1) parameter to optimize.
        phi: (N, D) the data set of features.
        y: (N,1) the target variable.
        grad_dir: function of args that compute the gradient descent direction.
        gamma: float, the learning rate.
        max_iter: int, maximum iteration.
        eps_conv: float, convergence criterion on the gradient
        history: bool, activate (or not) the trajectory recordind of the parameter
                and the cost, all along the gradient descent
        ----------
        return
            np.ndarray (optimized parameters),
            float (the minimum of the cost),
            np.ndarray (the sigmoid for optimized parameters),
            list (trajectory of the parameters),
            list (trajectory of the cost).
        """
        theta_hist = []
        J_hist = []
        for i in range(max_iter):
            f_wb = self.sigmoid(self.lin_func(phi=phi, theta=args))
            dx = gra_dir(f_wb)
            if history:
                theta_hist.append(args)
                J_hist.append(self.cost(phi, f_wb, y))
            if np.linalg.norm(dx) > eps_conv:
                args += gamma * dx # type: ignore
            else:
                print(f"Converged at iteration {i}")
        return args, self.cost(phi, f_wb, y), self.base_func(phi=phi, theta=args),  theta_hist, J_hist

    @staticmethod
    def p_labels(f_wb:np.ndarray, threshold:float):
        """
        attribute 1 if f_wb >= threshold and 0 otherwise
        -------
        return
            np.array of int 
        """
        N = f_wb.shape[0]
        labels = np.zeros(N, dtype="int")
        for i in range(N):
            if f_wb[i, 1] >= threshold:
                labels[i] = 1
            else:
                labels[i] = 0
        return labels

    def fit(self, X_train:np.ndarray, y_train:np.ndarray, bias:bool=False, method:int=0, threshold:float=0.5,
            seed_num:int=97, amp:float=1.0, size_mb:int=10, gamma:float=0.5,
            max_iter:int=100, eps_conv:float=1e-4, history:bool=False):
        """
        Compute gradient descent with fixed step_size
        ---------
        Parameters
        X_train: (N, D) the training data set of features.
        y_train: (N,) the training target variable.
        bias: bool
            if true, bias in the linear model,
            i.e. take into accound the intercept as parameter to optimize.
            The default is False.
        method:int
            method to be employed for GD
            0 -> for batch GD
            1 -> for minibatch GD
            2 -> for stochastic GD
            The default is 0.
        threshold: float
            define the position of the decision boundary. The default is 0.5
        mb_size: int
            size of the minibatch.
            Note that 1 < mb_size < N
            since mb_size <-> batch GD and mb_size = 1 <-> stochastic GD.
            The default is 10.
        seed_num : int, fix the seed, so the experience is repeatible.
        amp: float
            amplititude of the random initialization of parameters for GD
            The default is 1.0.
        mb_size: int
            size of the minibatch.
            Note that 1 < mb_size < N
            since mb_size <-> batch GD and mb_size = 1 <-> stochastic GD.
            The default is 10.
        learn_rate: float
            step_size, to be chosen wisely. The default is 0.5.
        max_iter: int
            maximum step of iteration.
            The default is 50.
        eps_conv: float
            convergence criterion on the norm of the gradient.
            The default is 1e-4
        history: bool, activate (or not) the trajectory recordind of the parameter
                and the cost, all along the gradient descent
        ----------
        return
            only if history then
            list (trajectory of the parameters),
            list (trajectory of the cost).
        """
        np.random.seed(seed_num)
        N = X.shape[0]
        # important deal with case of dataset with only 1 feature
        if sum(X.shape) == N:
            X = X.reshape((N,1))
            D=1
        else:
            D = X.shape[1]
        if sum(y.shape) == N:
            y = y.reshape(-1,1)
        if bias:
            X = np.hstack((X, np.ones(N).reshape(-1,1)))
            theta = amp * np.random.randn(D+1).reshape(-1,1)
        else:
            theta = amp * np.random.randn(D).reshape(-1,1)

        if method == 0:
            grad_dir = lambda f_wb : - self.dJdtheta(X_train, f_wb, y_train)
        elif method == 1:
            msk = [i for i in range(N)]
            grad_dir = lambda f_wb : self.minibach_gd(X_train, f_wb, y_train, msk, size_mb)
        else:
            grad_dir = lambda f_wb : self.stoc_gd(X_train, f_wb, y_train)
        
        opt_theta, _, f_wb_opt, theta_hist, J_hist = self.gradient_descent(theta, X_train, y_train,
                                                                        grad_dir, gamma, max_iter,
                                                                          eps_conv, history)
        self.theta = opt_theta
        self.f_wb = f_wb_opt
        self.threshold = threshold
        self.labels = self.p_labels(self.f_wb, threshold)
        if history:
            return theta_hist, J_hist
        
    def predict(self, X_test):
        z = self.lin_func(X_test, self.theta)
        f_wb = self.sigmoid(z)
        labels = self.p_labels(f_wb, self.threshold)
        return f_wb, labels