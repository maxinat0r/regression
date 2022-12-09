import logging
import os

import pandas as pd


import constants as c
from linearmaxgressor import *
from metrics.metrics import mean_absolute_error, mean_squared_error
from normalizer import MinMaxilizer
from utils import train_test_split

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    current_directory = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(current_directory, c.data_path_from_root))
    df.dropna(subset=c.feature_selection, inplace=True)

    train_df, test_df = train_test_split(df)
    X_train = train_df[c.feature_selection]
    y_train = train_df[c.target_selection]
    X_test = test_df[c.feature_selection]
    y_test = test_df[c.target_selection]

    normalizer = MinMaxilizer()
    normalizer.fit(X_train)
    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)

    regressor_list = []
    ols_model = LinearRegressor(solver="ols")
    regressor_list.append(ols_model)
    svd_model = LinearRegressor(solver="svd")
    regressor_list.append(svd_model)
    ridge_svd_model = RidgeRegressor(solver="svd", alpha=5)
    regressor_list.append(ridge_svd_model)
    ridge_bgd_model = RidgeRegressor(solver="bgd", alpha=5)
    regressor_list.append(ridge_bgd_model)

    for model in regressor_list:
        model.fit(X=X_train, y=y_train)
        model.fit(X=X_train, y=y_train)
        y_hat = model.predict(X=X_test)
        mse = mean_squared_error(y_test, y_hat)
        print(model.__class__.__name__, model.solver, mse)

    result_out = pd.DataFrame()
    for alpha in range(0, 200, 10):
        ridge_svd_model = RidgeRegressor(solver="svd", alpha=alpha)
        ridge_svd_model.fit(X=X_train, y=y_train, )
        y_hat = ridge_svd_model.predict(X=X_test)
        mse = mean_squared_error(y_test, y_hat)
        result = pd.DataFrame(model.coefficients_).T
        result["mse"] = mse
        result["alpha"] = alpha
        result_out = pd.concat([result_out, result], ignore_index=True)






if __name__ == "__main__":
    main()
