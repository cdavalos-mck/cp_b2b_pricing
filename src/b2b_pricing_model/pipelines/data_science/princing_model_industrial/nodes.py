import logging

import pandas as pd
import polars as pl
import shap
from scipy.stats import zscore

logger = logging.getLogger(__name__)


def create_master_base(
    params: dict, df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Extract parameters
    target_feature = params["features"]["target"]
    z_thresh = params["z_score_threshold"]
    q_lower = params["base_quantile"]["lower_quantile"]
    q_upper = params["base_quantile"]["upper_quantile"]

    # Compute z-score of log_c_yearly_margin
    margin_series = df.select("log_revenue").to_series().to_numpy()
    z_scores = zscore(margin_series, nan_policy="omit")

    df = df.with_columns(pl.Series(name="z_score_log_c_yearly_margin", values=z_scores))

    # Flag outliers by z-score
    zscore_outliers = df["z_score_log_c_yearly_margin"].abs() > z_thresh

    # Flag outliers by target quantiles
    lower_quantile = df[target_feature].quantile(q_lower)
    upper_quantile = df[target_feature].quantile(q_upper)
    quantile_outliers = (df[target_feature] < lower_quantile) | (
        df[target_feature] > upper_quantile
    )

    # Add outlier column
    df = df.with_columns((zscore_outliers | quantile_outliers).alias("is_outlier"))

    # Split regular clients into base and outliers
    regular_base_clients = df.filter(
        (~pl.col("is_outlier")) & (pl.col("c_yearly_margin_per_liter") > 0)
    )
    base_ids = regular_base_clients["customer_id"].unique()
    regular_outlier_clients = df.filter(~pl.col("customer_id").is_in(base_ids))

    # Logging for debug
    logger.debug(f"Regular base clients: {df.shape}")
    logger.debug(f"Regular outlier clients: {regular_outlier_clients.shape}")

    return regular_base_clients, regular_outlier_clients


def predict_second_stage(
    params: dict,
    best_first_stage_model,
    under_performers: pl.DataFrame,
    top_performers: pl.DataFrame,
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

    under_performers_pd = under_performers.to_pandas()
    top_performers_pd = top_performers.to_pandas()

    under_performers_pd["predicted_value"] = best_first_stage_model.predict(
        X=under_performers_pd
    )
    top_performers_pd["predicted_value"] = best_first_stage_model.predict(
        X=top_performers_pd
    )

    all_data = pd.concat(
        [under_performers_pd, top_performers_pd],
        ignore_index=True,
    )

    all_data["corrected_predicted_value"] = all_data.apply(
        lambda row: row["predicted_value"]
        if row["predicted_value"] > row["c_yearly_margin_per_liter"]
        else row["c_yearly_margin_per_liter"],
        axis=1,
    )

    all_data["margin_change"] = (
        all_data["corrected_predicted_value"] - all_data["c_yearly_margin_per_liter"]
    )

    model = best_first_stage_model.named_steps["regressor"]
    explainer = shap.TreeExplainer(model)
    preprocessor = best_first_stage_model.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(all_data)
    shap_values = explainer.shap_values(X_transformed)
    customer_id = all_data["customer_id"]
    feature_names = preprocessor.get_feature_names_out()
    cleaned = [name.split("__")[1] for name in feature_names]
    X = all_data[cleaned]

    return (
        pl.from_pandas(all_data),
        shap_values,
        customer_id,
        feature_names,
        explainer,
        X,
    )
