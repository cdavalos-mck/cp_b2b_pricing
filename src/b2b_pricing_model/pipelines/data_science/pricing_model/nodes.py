import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import polars as pl
import shap
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
    margin_series = regular_clients.select("log_revenue").to_series().to_numpy()
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


def filter_ood_clients(
    params: dict,
    top_performers: pl.DataFrame,
    non_regular_clients: pl.DataFrame,
    under_performers: pl.DataFrame,
    regular_outlier_clients: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # Filter out clients with out-of-distribution (OOD) characteristics
    categorical_features = params["features"]["categorical"]
    if not categorical_features:
        categorical_features = []
    numerical_features = params.get("features", {}).get("numerical", {}).get(
        "comma_columns", []
    ) + params.get("features", {}).get("numerical", {}).get("percent_columns", [])
    X_train = top_performers[numerical_features + categorical_features]
    X_regular = under_performers[numerical_features + categorical_features]
    X_non_regular = non_regular_clients[numerical_features + categorical_features]
    X_regular_outlier = regular_outlier_clients[
        numerical_features + categorical_features
    ]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_regular_scaled = scaler.transform(X_regular)
    X_non_regular_scaled = scaler.transform(X_non_regular)
    X_regular_outlier_scaled = scaler.transform(X_regular_outlier)

    distances_regular = compute_knn_distance(X_train_scaled, X_regular_scaled)
    distances_regular_scaled = (distances_regular - distances_regular.min()) / (
        distances_regular.max() - distances_regular.min()
    )

    distances_non_regular = compute_knn_distance(X_train_scaled, X_non_regular_scaled)
    distances_non_regular_scaled = (
        distances_non_regular - distances_non_regular.min()
    ) / (distances_non_regular.max() - distances_non_regular.min())

    distances_regular_outlier = compute_knn_distance(
        X_train_scaled, X_regular_outlier_scaled
    )
    distances_regular_outlier_scaled = (
        distances_regular_outlier - distances_regular_outlier.min()
    ) / (distances_regular_outlier.max() - distances_regular_outlier.min())

    outlier_regular_flags = percentile_outlier_flags(
        X_train.to_pandas(), X_regular.to_pandas()
    )
    n_flags = outlier_regular_flags["n_outlier_flags"].values
    n_flags_scaled = n_flags / n_flags.max()

    outlier_non_regular_flags = percentile_outlier_flags(
        X_train.to_pandas(), X_non_regular.to_pandas()
    )
    n_flags_non_regular = outlier_non_regular_flags["n_outlier_flags"].values
    n_flags_non_regular_scaled = n_flags_non_regular / n_flags_non_regular.max()

    outlier_regular_outlier_flags = percentile_outlier_flags(
        X_train.to_pandas(), X_regular_outlier.to_pandas()
    )
    n_flags_regular_outlier = outlier_regular_outlier_flags["n_outlier_flags"].values
    n_flags_regular_outlier_scaled = (
        n_flags_regular_outlier / n_flags_regular_outlier.max()
    )

    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(X_train_scaled)
    scaler = MinMaxScaler()

    # Score for new clients
    scores_regular = -iso.decision_function(X_regular_scaled)  # Higher = more outlier
    scaled_iso_forest_score = scaler.fit_transform(
        scores_regular.reshape(-1, 1)
    ).flatten()

    scores_non_regular = -iso.decision_function(X_non_regular_scaled)
    scaled_iso_forest_score_non_regular = scaler.fit_transform(
        scores_non_regular.reshape(-1, 1)
    ).flatten()

    scores_regular_outlier = -iso.decision_function(X_regular_outlier_scaled)
    scaled_iso_forest_score_regular_outlier = scaler.fit_transform(
        scores_regular_outlier.reshape(-1, 1)
    ).flatten()

    risk_score_regular = (
        0.5 * distances_regular_scaled
        + 0.3 * n_flags_scaled
        + 0.2 * scaled_iso_forest_score
    )

    risk_score_non_regular = (
        0.5 * distances_non_regular_scaled
        + 0.3 * n_flags_non_regular_scaled
        + 0.2 * scaled_iso_forest_score_non_regular
    )

    risk_score_regular_outlier = (
        0.5 * distances_regular_outlier_scaled
        + 0.3 * n_flags_regular_outlier_scaled
        + 0.2 * scaled_iso_forest_score_regular_outlier
    )

    safe_threshold = params.get("risk_thresholds", {}).get("safe", 0.4)
    ood_threshold = params.get("risk_thresholds", {}).get("ood", 0.75)

    risk_category_regular = classify(risk_score_regular, safe_threshold, ood_threshold)
    risk_category_non_regular = classify(
        risk_score_non_regular, safe_threshold, ood_threshold
    )
    risk_category_regular_outlier = classify(
        risk_score_regular_outlier, safe_threshold, ood_threshold
    )

    # Add to under_performers
    under_performers_with_scores = under_performers.with_columns(
        [
            pl.Series(name="risk_score", values=risk_score_regular),
            pl.Series(name="risk_category", values=risk_category_regular),
        ]
    )

    # Add to non_regular_clients
    non_regular_clients_with_scores = non_regular_clients.with_columns(
        [
            pl.Series(name="risk_score", values=risk_score_non_regular),
            pl.Series(name="risk_category", values=risk_category_non_regular),
        ]
    )

    # Add to regular_outlier_clients
    regular_outlier_clients_with_scores = regular_outlier_clients.with_columns(
        [
            pl.Series(name="risk_score", values=risk_score_regular_outlier),
            pl.Series(name="risk_category", values=risk_category_regular_outlier),
        ]
    )

    return (
        under_performers_with_scores,
        non_regular_clients_with_scores,
        regular_outlier_clients_with_scores,
    )


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
    (
        cv_results,
        pipeline_list,
        shap_list,
        customer_ids,
        feature_names,
        explainer_list,
        X,
    ) = regression_model.train_model(data=data.to_pandas())

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

    return (
        pl.from_pandas(combined_df),
        pl.from_pandas(summary_df),
        pipeline_list[best_model],
    )


def predict_first_stage(
    params: dict[str, Any],
    best_first_stage_model,
    train_data: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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

    return (
        pl.from_pandas(result_data),
        pl.from_pandas(under_performers),
        pl.from_pandas(top_performers),
    )


def predict_second_stage(
    params: dict,
    best_first_stage_model,
    under_performers: pl.DataFrame,
    top_performers: pl.DataFrame,
    non_regular_clients_tct: pl.DataFrame,
    regular_outlier_clients: pl.DataFrame,
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
    non_regular_clients_tct = non_regular_clients_tct.to_pandas()
    regular_outlier_clients = regular_outlier_clients.to_pandas()

    non_regular_clients_tct["c_yearly_network_volumen"] = non_regular_clients_tct[
        "c_annualized_network_monthly_volume"
    ]

    under_performers_pd["predicted_value"] = best_first_stage_model.predict(
        X=under_performers_pd
    )
    top_performers_pd["predicted_value"] = best_first_stage_model.predict(
        X=top_performers_pd
    )
    top_performers_pd["risk_category"] = "top_performer"

    non_regular_clients_tct["predicted_value"] = best_first_stage_model.predict(
        X=non_regular_clients_tct
    )

    regular_outlier_clients["predicted_value"] = best_first_stage_model.predict(
        X=regular_outlier_clients
    )

    non_regular_clients_tct["performance_label"] = "Non Regular Client"
    regular_outlier_clients["performance_label"] = "Outlier Client"

    all_data = pd.concat(
        [
            under_performers_pd,
            top_performers_pd,
            non_regular_clients_tct,
            regular_outlier_clients,
        ],
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

    all_data_pl = pl.from_pandas(all_data)

    return (
        all_data_pl,
        shap_values,
        customer_id,
        feature_names,
        explainer,
        X,
    )


def compute_knn_distance(X_train, X_new, n_neighbors=5):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X_train)
    distances, _ = knn.kneighbors(X_new)
    return distances.mean(axis=1)  # one value per client


def percentile_outlier_flags(X_train, X_new, lower_q=0.01, upper_q=0.99):
    flags = pd.DataFrame(index=X_new.index)
    for col in X_train.columns:
        low = X_train[col].quantile(lower_q)
        high = X_train[col].quantile(upper_q)
        flags[col + "_is_outlier"] = (X_new[col] < low) | (X_new[col] > high)
    flags["n_outlier_flags"] = flags.sum(axis=1)
    return flags


def compute_client_risk_scores(
    X_train: np.ndarray, X_new: np.ndarray, X_new_raw: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # KNN distance
    distances = compute_knn_distance(X_train, X_new)
    distances_scaled = (distances - distances.min()) / (
        distances.max() - distances.min()
    )

    # Outlier flags (on raw features)
    outlier_flags = percentile_outlier_flags(pd.DataFrame(X_train), X_new_raw)
    n_flags = outlier_flags["n_outlier_flags"].values
    n_flags_scaled = n_flags / n_flags.max()

    # Isolation Forest
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(X_train)
    scores = -iso.decision_function(X_new)
    iso_scaled = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten()

    # Combined risk
    risk_score = 0.5 * distances_scaled + 0.3 * n_flags_scaled + 0.2 * iso_scaled
    return risk_score, distances_scaled, n_flags_scaled


def classify(score_array, safe_threshold, ood_threshold):
    return np.select(
        [
            score_array <= safe_threshold,
            (score_array > safe_threshold) & (score_array <= ood_threshold),
            score_array > ood_threshold,
        ],
        ["safe", "unsafe", "ood"],
        default="unsafe",  # <-- Fix: explicitly provide a string fallback
    )
