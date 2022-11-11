import logging
import os
from plotnine import ggplot, aes, geom_line, labs, theme, scales, facet_grid

import pandas as pd
import constants as c
from linearmaxgressor import LinearMaxregressor
from metrics.metrics import mean_absolute_error, mean_squared_error
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

    for method in c.known_methods:
        model = LinearMaxregressor(method=method)
        model.fit(X=X_train, y=y_train)
        y_hat = model.predict(X=X_test)
        LOGGER.info(
            f"[{method}] Mean Absolute error is {mean_absolute_error(y_test, y_hat):,.0f}"
        )
        LOGGER.info(f"[{method}] Coefficients {model.coefficients_}")

    result_out = pd.DataFrame()
    for alpha in range(0,50):
        model = LinearMaxregressor(method="gradient_descent", alpha=alpha)
        model.fit(X=X_train, y=y_train)
        y_hat = model.predict(X=X_test)
        mse = mean_squared_error(y_test, y_hat)
        result = pd.DataFrame(model.coefficients_).T
        result["mse"] = mse
        result["alpha"] = alpha
        result_out = pd.concat([result_out, result], ignore_index=True)

    molten_df = pd.melt(result_out,
                        id_vars=["alpha", "mse"],
                        value_vars=c.feature_selection,
                        var_name="feature",
                        value_name="coefficient")

    plot = (
            ggplot(molten_df)
            + aes(x="alpha", y="coefficient")
            + geom_line(colour="feature")
            + theme(legend_position="top", figure_size=(10, 12))
    )



    print(plot)
if __name__ == "__main__":
    main()
