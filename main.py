import logging
import os

import pandas as pd
import constants as c
from linearmaxgressor import LinearMaxregressor
from metrics.metrics import mean_absolute_error

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
def main():
    current_directory = os.path.dirname(__file__)
    path_to_data = os.path.join(current_directory, "./data/house_prices.csv")
    df = pd.read_csv(path_to_data)

    feature_selection = ["LotFrontage", "LotArea", "BedroomAbvGr"]
    target_selection = "SalePrice"
    df.dropna(subset=feature_selection, inplace=True)
    feature_data = df[feature_selection]
    target_data = df[target_selection]

    for method in c.known_methods:
        regressor = LinearMaxregressor(method=method)
        regressor.fit(X=feature_data, y=target_data)
        y_hat = regressor.predict(X=feature_data)
        LOGGER.info(
            f"[{method}] Mean Absolute error is {mean_absolute_error(target_data, y_hat):,.0f}"
        )
        LOGGER.info(f"[{method}] Coefficients {regressor.coefficients_}")

if __name__ == "__main__":
    main()
