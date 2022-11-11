import logging
import os

import pandas as pd
import constants as c
from linearmaxgressor import LinearMaxregressor
from metrics.metrics import mean_absolute_error
from utils import train_test_split
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

    train_df, test_df = train_test_split(df)
    X_train = train_df[feature_selection]
    y_train = train_df[target_selection]
    X_test = test_df[feature_selection]
    y_test = test_df[target_selection]

    for method in c.known_methods:
        regressor = LinearMaxregressor(method=method)
        regressor.fit(X=X_train, y=y_train)
        y_hat = regressor.predict(X=X_test)
        LOGGER.info(
            f"[{method}] Mean Absolute error is {mean_absolute_error(y_test, y_hat):,.0f}"
        )
        LOGGER.info(f"[{method}] Coefficients {regressor.coefficients_}")

if __name__ == "__main__":
    main()
