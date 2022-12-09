import logging
from abc import ABC, abstractmethod

import numpy as np

LOGGER = logging.getLogger(__name__)


class Regressor(ABC):
    """
    Abstract Base Class for Regressors.

    This class provides a common interface for several types of regression models,
    and defines shared methods and attributes used by these models.

    Attributes:
        solver: The solver used to fit the regression model. Can be "ols" for
            ordinary least squares or "bgd" for batch gradient descent.
        include_constant: A boolean indicating whether to include a constant term
            in the regression model.
        constant_: The constant term of the regression model.
        n_iterations: The number of iterations to use for batch gradient descent.
        learning_rate: The learning rate to use for batch gradient descent.
        alpha: The regularization parameter for the regression model.
        l1_ratio: The mixing parameter between L1 and L2 regularization.

    Methods:
        fit: Fit the regression model to data. Must be implemented by subclasses.
        predict: Make predictions using a fitted model.
    """

    def __init__(
        self,
        solver="ols",
        include_constant=True,
        n_iterations=500,
        learning_rate=0.1,
        alpha=0,
    ):
        self.solver = solver
        self.include_constant = include_constant
        self.constant_ = None
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.alpha = alpha

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
        Initialize the coefficients of the regression model to random values.

        Args:
            X: The feature matrix.
            low: The lower bound of the random coefficients.
            high: The upper bound of the random coefficients.

        Returns:
            A vector of random coefficients for each feature in X,
            sampled uniformly from the half-open interval [low, high).
        """
        n = X.shape[1]
        self.coefficients_ = np.random.uniform(low, high, n)

    def _batch_gradient_descent(self, X, y):
        """
        Use Batch Gradient Descent to solve for least squares.
        Loss function is Mean Squared Errors.

        Args:
            X: The feature matrix.
            y: The target vector.

        Returns:
            The fitted coefficients of the regression model.
        """
        self._initatialize_coefficients(X)
        m = X.shape[0]

        for i in range(self.n_iterations):
            yhat = X @ self.coefficients_
            error = y - yhat
            mse = np.mean(error**2)

            # Calculate the gradient using the partial derivative of
            # the loss function (MSE) with respect to coefficients
            gradient = -2 / m * ((error) @ X) + (self.alpha * self.coefficients_)
            gradient += (1 - self.l1_ratio) * self.alpha * self.coefficients_
            gradient += self.l1_ratio * self.alpha

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
        known_solvers = ["ols", "svd"]
        if self.solver not in known_solvers:
            raise ValueError(
                f"""Known solvers are {known_solvers}. Got "{self.solver}"."""
            )
        if self.solver == "ols":
            self._solve_ols(X, y)
        elif self.solver == "svd":
            self._solve_svd(X, y)

        self._calculate_constant(X, y)
        LOGGER.info("[OrdinaryLeastSquares] Fitting finished")


class RidgeRegressor(Regressor):
    """
    Ridge Regressor.

    This class fits a ridge regularized linear regression model to data using
    either ordinary least squares or batch gradient descent. Ridge regression
    uses L2 regularization to prevent overfitting and improve model
    generalization.

    Attributes:
        alpha: The regularization parameter for the ridge regression model.

    Methods:
        fit: Fit the ridge regression model to data.
        predict: Make predictions using a fitted model.

    This can be solved in multiple ways. This class has two options:
    (1) Singular Value Decomposition
    (2) Batch Gradient Descent
    """

    def _solve_svd(self, X, y):
        """
        Use Singular Value Decomposition to minimize the following objective function:

            ||y - X*Beta||^2_2 + alpha * ||Beta||^2_2

        where X and y are matrices, Beta is a vector of coefficients, and alpha is a scalar value.

        The objective function measures the L2-norm (also known as the Euclidean norm)
        of the error between the actual values in y and the predicted values from X and Beta,
        plus the L2-norm of the Beta coefficients multiplied by alpha.

        The L2-norm of a vector x is calculated by summing the squares of its absolute values:
        ||x||^2_2 = SUM(|x|^2)|i=1 to i=n

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
        known_solvers = ["svd", "bgd"]
        if self.solver not in known_solvers:
            raise ValueError(
                f"""Known solvers are {known_solvers}. Got "{self.solver}"."""
            )
        if self.solver == "svd":
            self._solve_svd(X, y)
        elif self.solver == "bgd":
            self._batch_gradient_descent(X, y)
        self._calculate_constant(X, y)
        LOGGER.info("[RidgeRegressor] Fitting finished")


class Elasticnet(Regressor):
    """
    Elastic Net Regressor.

    This class fits an elastic net regularized linear regression model to data
    using either ordinary least squares or batch gradient descent.

    Elastic net regression is a type of linear regression that uses both L1 and
    L2 regularization to prevent overfitting and improve model generalization.
    L1 regularization adds a penalty proportional to the absolute value of the
    coefficients, while L2 regularization adds a penalty proportional to the
    square of the coefficients. Elastic net regression uses a mixing parameter
    l1_ratio to determine the relative weight of L1 and L2 regularization.
    For example, if l1_ratio=0.5, the regularization penalty is a mix of L1
    and L2 regularization with equal weight.

    In general, elastic net regression can improve the performance of linear
    regression by selecting a more parsimonious set of features and preventing
    overfitting. It is particularly useful when there are a large number of
    features in the data, as it can help to select a smaller subset of relevant
    features and reduce the model complexity.

    Attributes:
        l1_ratio: The mixing parameter between L1 and L2 regularization.

    Methods:
        fit: Fit the elastic net regression model to data.
        predict: Make predictions using a fitted model.
    """

    def __init__(self, l1_ratio=0.5, **kwargs):
        super().__init__(**kwargs)
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        if self.solver == "ols":
            self._solve_ols(X, y)
        elif self.solver == "bgd":
            self._solve_bgd(X, y)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        if self.include_constant:
            self._calculate_constant(X, y)
