import logging
from typing import Optional

import polars as pl
from scipy.stats import zscore

from b2b_pricing_model.utils.ds_utils import (
    ModelOptimizer,
)

logger = logging.getLogger(__name__)


def create_master_base(
    params: dict, df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # Extract parameters
    target_feature = params["features"]["target"]
    z_thresh = params["z_score_threshold"]
    q_lower = params["base_quantile"]["lower_quantile"]
    q_upper = params["base_quantile"]["upper_quantile"]
    min_perc_week = params["frequent_clients"]["min_perc_week_threshold"]
    days_since_last_trx = params["frequent_clients"]["days_since_last_trx_threshold"]

    # Split regular vs. non-regular clients
    regular_clients = df.filter(
        (pl.col("transaction_week_ratio") >= min_perc_week)
        & (pl.col("c_days_since_last_trx") <= days_since_last_trx)
    )
    non_regular_clients = df.filter(
        (pl.col("transaction_week_ratio") < min_perc_week)
        | (pl.col("c_days_since_last_trx") > days_since_last_trx)
    )

    # Compute z-score of log_c_yearly_margin
    margin_series = regular_clients.select("log_c_yearly_margin").to_series().to_numpy()
    z_scores = zscore(margin_series, nan_policy="omit")

    regular_clients = regular_clients.with_columns(
        pl.Series(name="z_score_log_c_yearly_margin", values=z_scores)
    )

    # Flag outliers by z-score
    zscore_outliers = regular_clients["z_score_log_c_yearly_margin"].abs() > z_thresh

    # Flag outliers by target quantiles
    lower_quantile = regular_clients[target_feature].quantile(q_lower)
    upper_quantile = regular_clients[target_feature].quantile(q_upper)
    quantile_outliers = (regular_clients[target_feature] < lower_quantile) | (
        regular_clients[target_feature] > upper_quantile
    )

    # Add outlier column
    regular_clients = regular_clients.with_columns(
        (zscore_outliers | quantile_outliers).alias("is_outlier")
    )

    # Split regular clients into base and outliers
    regular_base_clients = regular_clients.filter(
        (~pl.col("is_outlier")) & (pl.col("c_yearly_margin_per_liter") > 0)
    )
    base_ids = regular_base_clients["customer_id"].unique()
    regular_outlier_clients = regular_clients.filter(
        ~pl.col("customer_id").is_in(base_ids)
    )

    # Logging for debug
    logger.debug(f"Non-regular clients: {non_regular_clients.shape}")
    logger.debug(f"Regular base clients: {regular_base_clients.shape}")
    logger.debug(f"Regular outlier clients: {regular_outlier_clients.shape}")

    return non_regular_clients, regular_base_clients, regular_outlier_clients


def get_best_hyperparameters(
    params: dict,
    data: pl.DataFrame,
    default_hyperparameters: Optional[list[str]] = None,
):
    """
    Get the best hyperparameters for the two-stage regression model.

    Args:
        params (dict): Parameters for the regression model.
        data (pd.DataFrame): Data with clusters.
        default_hyperparameters (dict): Default hyperparameters.

    Returns:
        dict: Best hyperparameters.
    """
    data_pl = data.to_pandas()

    optimizer = ModelOptimizer(params=params, random_state=42)
    results = optimizer.get_model_performance(data=data_pl)

    if results == {}:
        results = default_hyperparameters

    breakpoint()

    return results
