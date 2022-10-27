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

    def _calculate_coefficients_svd(self, X,y):
        # Use SVD
        U, Sigma, Vt = np.linalg.svd(X.T @ X)
        V = Vt.T
        # Get Moore-Penrose pseudoinverse of X squared
        Sigma_pinv = np.linalg.pinv(np.diag(Sigma))
        X_squared_pinv = V @ Sigma_pinv @ U.T
        self.coefficients_ = np.array(X_squared_pinv @ X.T @ y)

    def _calculate_coefficients_svd_l2(self, X, y):
        U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
        denominator = Sigma ** 2 + self.alpha
        d = np.divide(Sigma, denominator, out=np.zeros_like(Sigma), where=denominator != 0)
        self.coefficients_ = np.array(d * (U.T @ y) @ Vt)

    def _calculate_coefficients_ols(self, X, y):
        self.coefficients_ = np.linalg.inv(X.T @ X) @ (X.T @ y)
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
