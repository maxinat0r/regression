import logging
import os

import pandas as pd

from linearmaxgressor import LinearMaxregressor
from metrics.metrics import mean_absolute_error

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    current_directory = os.path.dirname(__file__)
    path_to_data = os.path.join(current_directory, "./data/house_prices.csv")
    df = pd.read_csv(path_to_data)

    feature_selection = ["LotFrontage", "LotArea", "BedroomAbvGr"]
    target_selection = "SalePrice"
    df.dropna(subset=feature_selection, inplace=True)
    feature_data = df[feature_selection]
    target_data = df[target_selection]

    regressor_ols = LinearMaxregressor(method="ols", include_constant=True)
    regressor_ols.fit(X=feature_data, y=target_data)

    regressor_svd = LinearMaxregressor(
        method="svd", include_constant=True
    )
    regressor_svd.fit(X=feature_data, y=target_data)

    regressor_ridge_svd = LinearMaxregressor(
        method="ridge_svd", include_constant=True, alpha=25000
    )
    regressor_ridge_svd.fit(X=feature_data, y=target_data)

    y_hat_ols = regressor_ols.predict(X=feature_data)
    y_hat_svd = regressor_svd.predict(X=feature_data)
    y_hat_ridge_svd = regressor_ridge_svd.predict(X=feature_data)

    LOGGER.info(
        f"[OLS] Mean Absolute error is {mean_absolute_error(target_data, y_hat_ols):3f}"
    )
    LOGGER.info(
        f"[OLS] Coefficients {regressor_ols.coefficients_}"
    )

    LOGGER.info(
        f"[SVD] Mean Absolute error is {mean_absolute_error(target_data, y_hat_svd):3f}"
    )
    LOGGER.info(
        f"[SVD] Coefficients {regressor_svd.coefficients_}"
    )

    LOGGER.info(
        f"[Ridge SVD] Mean Absolute error is {mean_absolute_error(target_data, y_hat_ridge_svd):3f}"
    )
    LOGGER.info(
        f"[Ridge SVD] Coefficients {regressor_ridge_svd.coefficients_}"
    )
