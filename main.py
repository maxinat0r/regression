import logging
import pandas as pd
import numpy as np
from linearmaxgressor import LinearMaxregressor

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":

    df = pd.read_csv("/Users/max.delavieter/Documents/regression/data/house_prices.csv")
    feature_selection = ["LotFrontage", "LotArea", "BedroomAbvGr"]
    target_selection = "SalePrice"
    df.dropna(subset=feature_selection, inplace=True)
    feature_data = df[feature_selection]
    target_data = df[target_selection]

    regressor = LinearMaxregressor(method="normal equation")
    regressor.fit(X=feature_data, y=target_data)

    y_hat = regressor.predict(X=feature_data)
    LOGGER.info(
        f"[Main] Mean Absolute error is {np.mean(np.abs(target_data-y_hat))}"
    )

