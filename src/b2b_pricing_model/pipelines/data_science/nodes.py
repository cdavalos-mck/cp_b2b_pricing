import logging
from typing import Any, Optional

import pandas as pd
import polars as pl
from scipy.stats import zscore

from b2b_pricing_model.utils.ds_utils import ModelOptimizer, RegressionModel

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

    return results


def train_model(
    params: dict,
    best_hyperparameters: dict,
    data: pl.DataFrame,
):
    """
    Train the model with the best hyperparameters.

    Args:
        params (dict): Parameters for the regression model.
        best_hyperparameters (dict): Best hyperparameters.
        top_performers (pd.DataFrame): Data of top performers.

    Returns:
        tuple: DataFrames of cross-validation results and metrics.
    """
    regression_model = RegressionModel(
        params=params, best_hyperparameters=best_hyperparameters, random_state=42
    )
    cv_results, pipeline_list = regression_model.train_model(data=data.to_pandas())

    # Step 1: Compute the summary per model
    summary = {}
    for model, df in cv_results.items():
        df_copy = df.copy()
        df_copy["test_mae"] = df_copy["test_mae"].abs()
        df_copy["train_mae"] = df_copy["train_mae"].abs()
        summary[model] = df_copy.mean()

    summary_df = pd.DataFrame(summary).T

    # Step 2: Select the model with the lowest test_mape
    best_model = summary_df["test_mae"].idxmin()

    combined_df = pd.concat(
        [df.assign(model=model) for model, df in cv_results.items()], ignore_index=True
    )

    logger.info(
        f"Best model selected: {best_model} with test MAE: {summary_df.loc[best_model, 'test_mae']:.4f}"
    )

    return combined_df, summary_df, pipeline_list[best_model]


def predict_first_stage(
    params: dict[str, Any],
    best_first_stage_model,
    train_data: pl.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Predict using the first stage model and classify customers as top or under performers.

    Args:
        params (Dict[str, Any]): Configuration parameters containing:
            - threshold (float): Relative error threshold for top performer classification
            - use_minimum_margin (bool, optional): Whether to apply minimum margin condition
            - min_margin_threshold (float, optional): Maximum margin threshold when use_minimum_margin=True
            - min_volume_threshold (float, optional): Minimum volume threshold when use_minimum_margin=True
        best_first_stage_model: Trained regression model with predict method
        data (pd.DataFrame): Input data with features for prediction
        train_data (pd.DataFrame): Training data used to determine valid customer IDs

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - Complete data with predictions and performance labels
            - Under performers subset
            - Top performers subset

    Raises:
        KeyError: If required parameters are missing
        ValueError: If data doesn't contain required columns
    """

    # Extract parameters
    threshold = params["filter"]["threshold"]
    use_minimum_margin = params["filter"]["use_minimum_margin"]["enabled"]

    # Make a copy to avoid modifying the original data
    train_data_pd = train_data.to_pandas()
    result_data = train_data_pd.copy()

    # Generate predictions
    result_data["predicted_value"] = best_first_stage_model.predict(X=result_data)

    # Calculate relative error
    result_data["relative_error"] = (
        result_data["predicted_value"] - result_data["c_yearly_margin_per_liter"]
    ) / result_data["c_yearly_margin_per_liter"]

    # Primary condition: relative error within threshold
    cond_error = result_data["relative_error"] <= threshold

    # Secondary condition: minimum margin (configurable)
    cond_minimum_margin = pd.Series([False] * len(result_data), index=result_data.index)

    if use_minimum_margin:
        min_margin_threshold = params["filter"]["use_minimum_margin"][
            "min_margin_threshold"
        ]
        min_volume_threshold = params["filter"]["use_minimum_margin"][
            "min_volume_threshold"
        ]

        cond_minimum_margin = (
            result_data["c_yearly_margin_per_liter"] < min_margin_threshold
        ) & (result_data["c_yearly_network_volumen"] > min_volume_threshold)

    # Get valid customer IDs from training data
    train_customer_ids = set(train_data_pd["customer_id"].drop_duplicates())
    cond_in_training = result_data["customer_id"].isin(train_customer_ids)

    # Initialize all as under performers
    result_data["performance_label"] = "Under Performer"
    # Apply top performer conditions
    top_performer_mask = (cond_error | cond_minimum_margin) & cond_in_training
    result_data.loc[top_performer_mask, "performance_label"] = "Top Performer"

    # Split into performance categories
    under_performers = result_data[
        result_data["performance_label"] == "Under Performer"
    ].copy()
    top_performers = result_data[
        result_data["performance_label"] == "Top Performer"
    ].copy()

    return result_data, under_performers, top_performers


def predict_second_stage(
    params: dict,
    best_first_stage_model,
    under_performers: pd.DataFrame,
    top_performers: pd.DataFrame,
):
    """
    Predict using the first stage model.

    Args:
        params (dict): Parameters for the regression model.
        best_first_stage_model (RegressionModel): The best first stage model.
        data_with_clusters (pd.DataFrame): Data with clusters.

    Returns:
        tuple: DataFrames of cross-validation results and metrics.
    """

    under_performers["predicted_value"] = best_first_stage_model.predict(
        X=under_performers
    )
    top_performers["predicted_value"] = best_first_stage_model.predict(X=top_performers)

    all_data = pd.concat([under_performers, top_performers], ignore_index=True)

    return all_data
