import logging

import numpy as np

import constants as c

LOGGER = logging.getLogger(__name__)


class LinearMaxregressor:
    """
    Fit the LinearMaxregressor
    """

    def __init__(
        self,
        method="ols",
        include_constant=True,
        alpha=0,
        n_iterations=500,
        learning_rate=0.1,
    ):
        self.method = method
        self.include_constant = include_constant
        self.alpha = alpha
        self.constant_ = None
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def learning_schedule(self, t):
        """
        Learning schedule starting at 0.1 and decreasing for t
        Arguments: t
        Returns:
        """
        t0 = 5
        t1 = 50
        eta = t0 / (t + t1)
        return eta

    def _calculate_coeffients_batch_gradient_descent(self, X, y):
        """
        Use Gradient Descent to solve for least squares.
        Loss function is Mean Squared Errors.
        """
        X_normed = X / X.max(axis=0)
        m = X.shape[0]
        n = X.shape[1]

        # Initialize random coefficients between -1 and 1
        self.coefficients_ = np.random.uniform(-1, 1, n)

        # Do batch gradient descent 'n_iteration' times
        for i in range(self.n_iterations):
            yhat = X_normed @ self.coefficients_
            error = y - yhat
            mse = np.mean(error**2)

            # Calculate the gradient using the partial derivative of
            # the loss function (MSE) with respect to coefficients
            gradient = -2 / m * ((error) @ X_normed) + (self.alpha * self.coefficients_)

            # Update coefficients using the new gradient
            eta = self.learning_schedule(i + i)
            self.coefficients_ -= eta * gradient
            if i % 100 == 0:
                LOGGER.info(f"[Gradient Descend] Iteration {i}. MSE: {mse:,.0f}")

    def _calculate_coefficients_svd(self, X, y):
        """ """
        # Use Singular Value Decomposition to decompose the X
        # The returned Sigma is a vector containing only the diagonals
        U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)

        # Get Moore-Penrose pseudoinverse of Sigma by dividing 1 by Sigma, returning 0 when Sigma is 0.
        Sigma_pinv = np.divide(1, Sigma, out=np.zeros_like(Sigma), where=Sigma != 0)

        self.coefficients_ = Sigma_pinv * (U.T @ y) @ Vt

    def _calculate_coefficients_ridge_svd(self, X, y):
        """
        Linear Regression with L2 regularization, also known as Ridge.

        Use Singular Value Decomposition to minimize the following objective function:

        ||y - X*Beta||^2_2 + alpha * ||Beta||^2_2

        In words: the L2-norm  of the error plus alpha times the L2-norm of the beta coefficients.

        The L2-norm is also known as Euclidean norm because it measures the distance
        between vectors in terms of the Euclidean distance.

        The L2-norm of a vector x is calculated by summing the squares of its absolute values.
        ||x||^2_2 = SUM(|x|^2)|i=1 to i=n
        """
        # Use Singular Value Decomposition to split the X matrix into three components
        # U and Vt are orthogonal matrices
        # Sigma is a rectangular diagonal matrix with non-negative real numbers on the diagonal
        U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)

        # Get Moore-Penrose pseudoinverse of Sigma by dividing sigma by its square, returning 0 when Sigma square is 0.
        Sigma_pinv = np.divide(
            1,
            Sigma**2 + self.alpha,
            out=np.zeros_like(Sigma),
            where=Sigma**2 + self.alpha != 0,
        )

        self.coefficients_ = Sigma_pinv * (U.T @ y) @ Vt

    def _calculate_coefficients_ols(self, X, y):
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

    def _calculate_constant(self, X, y):
        self.constant_ = np.mean(y - (X @ self.coefficients_))

    def fit(self, X, y):
        """
        Fit the LinearMaxregressor
        """
        if self.method not in c.known_methods:
            raise ValueError(
                f"""Known methods are {c.known_methods}. Got "{self.method}"."""
            )
        LOGGER.info(f"[LinearMaxregressor] Method: {self.method}")
        if self.method == "ols":
            self._calculate_coefficients_ols(X, y)
        elif self.method == "svd":
            self._calculate_coefficients_svd(X, y)
        elif self.method == "ridge_svd":
            self._calculate_coefficients_ridge_svd(X, y)
        elif self.method == "gradient_descent":
            self._calculate_coeffients_batch_gradient_descent(X, y)

        LOGGER.info(f"[LinearMaxregressor] Coefficients: {self.coefficients_}")

        if self.include_constant:
            self._calculate_constant(X, y)

        LOGGER.info(f"[LinearMaxregressor] Constant (intercept): {self.constant_}")

        LOGGER.info("[LinearMaxregressor] Fitting finished")

    def predict(self, X):
        if self.coefficients_ is None:
            LOGGER.error(
                "[LinearMaxregressor] Model not fit yet. Call fit to fit the model."
            )

        yhat = X @ self.coefficients_

        if self.constant_ is None:
            return yhat
        else:
            return yhat + self.constant_
