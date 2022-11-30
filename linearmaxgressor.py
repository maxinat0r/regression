import logging
from abc import ABC, abstractmethod

import numpy as np

import constants as c

LOGGER = logging.getLogger(__name__)


class Regressor(ABC):
    """
    Abstract Base Class for Regressors.
    """

    def __init__(
        self,
        solver="ols",
        include_constant=True,
        n_iterations=500,
        learning_rate=0.1,
    ):
        self.solver = solver
        self.include_constant = include_constant
        self.constant_ = None
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    @staticmethod
    def learning_schedule(t):
        """
        Learning schedule starting at 0.1 and decreasing for t
        Arguments: t
        Returns: eta
        """
        t0 = 10
        t1 = 100
        eta = t0 / (t + t1)
        return eta

    def _initatialize_coefficients(self, X, low=-1, high=1):
        """
        Initialize a random coefficients for each feature by sampling
        uniformly fromhe half-open interval [low, high) (includes low, but excludes high)
        """
        n = X.shape[1]
        self.coefficients_ = np.random.uniform(low, high, n)

    def _batch_gradient_descent(self, X, y):
        """
        Use Batch Gradient Descent to solve for least squares.
        Loss function is Mean Squared Errors.

        Args:
            n_iterations: The amount of interations used in gradient descent
            alpha:
        """

        m = X.shape[0]

        for i in range(self.n_iterations):
            yhat = X @ self.coefficients_
            error = y - yhat
            mse = np.mean(error**2)

            # Calculate the gradient using the partial derivative of
            # the loss function (MSE) with respect to coefficients
            gradient = -2 / m * ((error) @ X) + (self.alpha * self.coefficients_)

            # Get eta, which decreases for each iteration
            eta = self.learning_schedule(i)

            # Update coefficients using the new gradient
            self.coefficients_ -= eta * gradient
            if i % 100 == 0:
                LOGGER.info(
                    f"[Gradient Descent] Iteration {i}. Eta: {eta}. MSE: {mse:,.0f}"
                )

    def _solve_bgd(self, X, y):
        self._initatialize_coefficients(X)
        self._batch_gradient_descent(X, y)

    def _calculate_constant(self, X, y):
        self.constant_ = np.mean(y - (X @ self.coefficients_))

    @abstractmethod
    def fit(self, X, y):
        """Fit the model using training data."""

    def predict(self, X):
        """Make predictions using a fitted model."""
        if self.coefficients_ is None:
            LOGGER.error(
                "[LinearMaxregressor] Model not fit yet. Call fit to fit the model."
            )
        return X @ self.coefficients_ + self.constant_


class LinearRegressor(Regressor):
    def _solve_ols(self, X, y):
        """
        Use the normal equation to find the coefficient vector of the least-squares hyperplane.
        The normal equation is defined as:

        beta_hat = (X.T @ X)^-1 @ (X.T @ y)

        Gram matrix = X.T @ X
        moment matrix = X.T @ y

        In words: Beta hat is the dot-product of moment matrix and the inverse Gram matrix.
        """
        moment_matrix = X.T @ y
        gram_matrix = X.T @ X
        inverse_gram_matrix = np.linalg.inv(gram_matrix)
        self.coefficients_ = inverse_gram_matrix @ moment_matrix

    def _solve_svd(self, X, y):
        """
        Self linear regression using Singular Value Decomposition.
        """
        # Use Singular Value Decomposition to decompose the X
        # The returned Sigma is a vector containing only the diagonals
        U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)

        # Get Moore-Penrose pseudoinverse of Sigma by taking the
        # reciprocal of its nonzero elements.
        Sigma_pinv = np.divide(1, Sigma, out=np.zeros_like(Sigma), where=Sigma != 0)

        self.coefficients_ = Sigma_pinv * (U.T @ y) @ Vt

    def fit(self, X, y):
        LOGGER.info("[OrdinaryLeastSquares] Fitting starting")

        if self.solver not in ["ols", "svd"]:
            raise ValueError(
                f"""Known methods are {c.known_methods}. Got "{self.solver}"."""
            )
        if self.solver == "ols":
            self._solve_ols(X, y)
        elif self.solver == "svd":
            self._solve_svd(X, y)

        self._calculate_constant(X, y)
        LOGGER.info("[OrdinaryLeastSquares] Fitting finished")


class RidgeRegressor(Regressor):
    """
    Linear Regression with L2 regularization, also known as Ridge.

    This can be solved in multiple ways. This class has two options:
    (1) Singular Value Decomposition
    (2) Batch Gradient Descent
    """

    def _solve_svd(self, X, y):
        """
         Use Singular Value Decomposition to minimize the following objective function:
        ||y - X*Beta||^2_2 + alpha * ||Beta||^2_2
        In words: the L2-norm  of the error plus alpha times the L2-norm of the beta coefficients.
        The L2-norm is also known as Euclidean norm because it measures the distance
        between vectors in terms of the Euclidean distance.
        The L2-norm of a vector x is calculated by summing the squares of its absolute values.
        ||x||^2_2 = SUM(|x|^2)|i=1 to i=n
        𝛽̂ =(𝑋𝑡𝑋+𝜆𝐼)−1 𝑋𝑡𝑦
        𝑋=𝑈𝐷𝑉−1
        𝛽̂ =𝑉(𝐷**2 + 𝜆𝐼)−1 𝐷𝑈𝑡𝑦
        """
        # Use Singular Value Decomposition to decompose the X
        # The returned Sigma is a vector containing only the diagonals
        U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)

        # Get Moore-Penrose pseudoinverse of Sigma by dividing sigma by its square,
        # returning 0 when Sigma square is 0.
        Sigma_pinv = np.divide(
            Sigma,
            Sigma**2 + self.alpha,
            out=np.zeros_like(Sigma),
            where=Sigma**2 + self.alpha != 0,
        )
        self.coefficients_ = Sigma_pinv * (U.T @ y) @ Vt

    def fit(self, X, y):
        LOGGER.info("[RidgeRegressor] Fitting starting")
        known_methods = ["svd", "sgd"]
        if self.solver not in known_methods:
            raise ValueError(
                f"""Known methods are {known_methods}. Got "{self.method}"."""
            )
        if self.method == "svd":
            self._solve_svd(X, y)
        elif self.method == "sgd":
            self._solve_sgd(X, y)
        self._calculate_constant(X, y)
        LOGGER.info("[RidgeRegressor] Fitting finished")
