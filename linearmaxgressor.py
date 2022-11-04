import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class LinearMaxregressor:
    """
    Fit the LinearMaxregressor
    """

    def __init__(self, method="ols", include_constant=True, alpha=1):
        self.method = method
        self.include_constant = include_constant
        self.alpha = alpha
        self.constant_ = None

    def _calculate_coefficients_svd(self, X, y):
        # Use Singular Value Decomposition to split the X matrix into three components
        # U and Vt are orthogonal matrices
        # Sigma is a rectangular diagonal matrix with non-negative real numbers on the diagonal
        U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)

        # Get Moore-Penrose pseudoinverse of Sigma by dividing 1 by Sigma, returning 0 when Sigma is 0.
        Sigma_pinv = np.divide(1, Sigma, out=np.zeros_like(Sigma), where=Sigma != 0)

        self.coefficients_ = Sigma_pinv * (U.T @ y) @ Vt

    def _calculate_coefficients_svd_l2(self, X, y):
        """
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
        LOGGER.info(f"[LinearMaxregressor] Coefficients are {self.coefficients_}")

    def _calculate_constant(self, X, y):
        self.constant_ = np.mean(y - (X @ self.coefficients_))
        LOGGER.info(f"[LinearMaxregressor] Constant (intercept) is {self.constant_}")

    def fit(self, X, y):
        """
        Fit the LinearMaxregressor
        """
        known_methods = ["ols", "svd", "svd_l2"]
        if self.method not in (known_methods):
            raise ValueError(
                f"""Known methods are {known_methods}. Got "{self.method}"."""
            )
        if self.method == "ols":
            self._calculate_coefficients_ols(X, y)
        elif self.method == "svd":
            self._calculate_coefficients_svd(X, y)
        elif self.method == "svd_l2":
            self._calculate_coefficients_svd_l2(X, y)
        else:
            LOGGER.error(
                f"""[LinearMaxregressor] Specified method "
                {self.method}" is not known."""
            )

        if self.include_constant:
            self._calculate_constant(X, y)

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
