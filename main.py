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

    regressor_svd_l2 = LinearMaxregressor(
        method="svd_l2", include_constant=True, alpha=1
    )
    regressor_svd_l2.fit(X=feature_data, y=target_data)

    y_hat_ols = regressor_ols.predict(X=feature_data)
    y_hat_svd = regressor_svd.predict(X=feature_data)
    y_hat_svd_l2 = regressor_svd_l2.predict(X=feature_data)

    LOGGER.info(
        f"[OLS] Mean Absolute error is {mean_absolute_error(target_data, y_hat_ols)}"
    )

    LOGGER.info(
        f"[SVD] Mean Absolute error is {mean_absolute_error(target_data, y_hat_svd)}"
    )

    LOGGER.info(
        f"[SVD L2] Mean Absolute error is {mean_absolute_error(target_data, y_hat_svd_l2)}"
    )
