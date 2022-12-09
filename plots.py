import pandas as pd
from plotnine import aes, facet_grid, geom_line, ggplot, labs, scales, theme

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
print(coefficient_plot)

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
print(mse_plot)
