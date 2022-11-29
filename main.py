import logging
import os

import pandas as pd
from plotnine import aes, facet_grid, geom_line, ggplot, labs, scales, theme

import constants as c
from linearmaxgressor import LinearMaxregressor
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

    ols_model = LinearMaxregressor(method="ols")
    ols_model.fit(X=X_train, y=y_train)

    svd_model = LinearMaxregressor(method="svd")
    svd_model.fit(X=X_train, y=y_train)

    result_out = pd.DataFrame()
    for alpha in range(0, 25):
        model = LinearMaxregressor(method="gradient_descent", alpha=alpha)
        model.fit(X=X_train, y=y_train)
        y_hat = model.predict(X=X_test)
        mse = mean_squared_error(y_test, y_hat)
        result = pd.DataFrame(model.coefficients_).T
        result["mse"] = mse
        result["alpha"] = alpha
        result_out = pd.concat([result_out, result], ignore_index=True)

    coefficient_df = pd.melt(
        result_out,
        id_vars=["alpha"],
        value_vars=["mse"],
        var_name="feature",
        value_name="coefficient",
    )

    coefficient_plot = (
            ggplot(coefficient_df)
            + aes(x="alpha", y="coefficient", colour="feature")
            + geom_line()
            + theme(legend_position="top", figure_size=(10, 12))
    )

    mse_df = pd.melt(
        result_out,
        id_vars=["alpha"],
        value_vars=["mse"],
        var_name="feature",
        value_name="mse",
    )

    mse_plot = (
            ggplot(mse_df)
            + aes(x="alpha", y="mse")
            + geom_line()
            + theme(figure_size=(8, 8))
    )

    print(coefficient_plot)
    print(mse_plot)


if __name__ == "__main__":
    main()
