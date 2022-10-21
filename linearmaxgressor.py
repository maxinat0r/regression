import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class LinearMaxregressor:
    """
    Fit the LinearMaxregressor
    """

    def __init__(self, method="ols", include_constant=True):
        self.method = method
        self.include_constant = include_constant

    def _calculate_coeffients(self, X, y):
        LOGGER.info(f"[LinearMaxregressor] Using the {self.method} method")
        if self.method == "ols":
            self.coefficients_ = np.linalg.inv(X.T @ X) @ (X.T @ y)

        elif self.method == "svd":
            # Use SVD
            U, Sigma, Vt = np.linalg.svd(X.T @ X)
            V = Vt.T
            # Get Moore-Penrose pseudoinverse of X squared
            Sigma_pinv = np.linalg.pinv(np.diag(Sigma))
            X_squared_pinv = V @ Sigma_pinv @ U.T
            self.coefficients_ = np.array(X_squared_pinv @ X.T @ y)

        else:
            LOGGER.error(
                f"""[LinearMaxregressor] Specified method "
                {self.method}" is not known."""
            )

        LOGGER.info(f"[LinearMaxregressor] Coefficients are {self.coefficients_}")

    def _calculate_constant(self, X, y):
        self.constant_ = np.mean(y - (X @ self.coefficients_))
        LOGGER.info(f"[LinearMaxregressor] Constant (intercept) is {self.constant_}")

    def fit(self, X, y):
        """
        Fit the LinearMaxregressor
        """

        LOGGER.info("[LinearMaxregressor] Fitting started")
        self._calculate_coeffients(X, y)

        if self.include_constant:
            self._calculate_constant(X, y)

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
