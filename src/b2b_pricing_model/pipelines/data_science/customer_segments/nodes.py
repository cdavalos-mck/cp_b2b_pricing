import pandas as pd
import polars as pl

from b2b_pricing_model.utils.ds_utils import (
    ClusteringPipeline,
    explain_neighbors_original_scale,
)


def create_cluster_pipeline(params: dict) -> ClusteringPipeline:
    """
    Create a pipeline for clustering analysis.

    Returns:
    --------
    Pipeline
        A Kedro pipeline for clustering analysis.
    """
    return ClusteringPipeline(
        algorithm=params["algorithm"],
        min_cluster_size=params["min_cluster_size"],
        k_range=params["k_range"],
    )


def preprocess_columns(params: dict, df: pl.DataFrame, pipeline: ClusteringPipeline):
    """
    Preprocess the specified columns in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    columns : list
        List of columns to preprocess.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with preprocessed columns.
    """

    df = df.sort("customer_id")

    comma_columns = params["features"]["numerical"]["comma_columns"]
    percent_columns = params["features"]["numerical"]["percent_columns"]

    numeric_columns = comma_columns + percent_columns
    categorical_columns = params["features"]["categorical"]

    if categorical_columns is None:
        df_pandas = df.select(numeric_columns).to_pandas()
    if numeric_columns is None:
        df_pandas = df.select(categorical_columns).to_pandas()
    if numeric_columns is not None and categorical_columns is not None:
        df_pandas = df.select(numeric_columns + categorical_columns).to_pandas()

    X = df_pandas.copy()
    X_scaled = pipeline.preprocess_data(X=X, method="standard")

    return X, X_scaled


def get_k_nearest_neighbors(
    params: dict,
    master_trx_customer_tct: pl.DataFrame,
    X_scaled: pl.DataFrame,
    pipeline: ClusteringPipeline,
):
    """
    Get the k-nearest neighbors for the given DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    pipeline : ClusteringPipeline
        The clustering pipeline to use for preprocessing.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with k-nearest neighbors.
    """

    master_trx_customer_tct = master_trx_customer_tct.sort("customer_id")

    pipeline.X_scaled = X_scaled

    # Assuming the pipeline has a method to get k-nearest neighbors
    # knn_results = pipeline.get_k_nearest_neighbors(
    #     k=params["k_neighbors"],
    #     customer_ids=master_trx_customer_tct["customer_id"].to_list(),
    # )

    knn_results = pipeline.get_k_nearest_top_performers(
        k=params["k_neighbors"],
        customer_ids=master_trx_customer_tct["customer_id"].to_list(),
        performance_labels=master_trx_customer_tct["performance_label"].to_list(),
    )

    return knn_results


def summarize_closest_customers(params, df_original, neighbors_df, X_scaled):
    """
    Summarize differences in features between each customer and their nearest neighbors.

    Parameters
    ----------
    params : dict
        Clustering parameters, e.g. which columns to use.
    df_original : pd.DataFrame
        Original unscaled dataframe with customer IDs and features.
    neighbors_df : pd.DataFrame
        Output from get_k_nearest_neighbors: contains customer_id, neighbor_id, rank, distance.

    Returns
    -------
    pd.DataFrame
        Flat dataframe with feature differences between each customer and its neighbors.
    """

    df_original = df_original.sort("customer_id")

    df_original_pd = df_original.to_pandas()

    categorical_features = params["features"]["categorical"]
    if not categorical_features:
        categorical_features = []
    numerical_features = params.get("features", {}).get("numerical", {}).get(
        "comma_columns", []
    ) + params.get("features", {}).get("numerical", {}).get("percent_columns", [])

    feature_cols = categorical_features + numerical_features

    X_scaled_pd = pd.DataFrame(X_scaled, columns=feature_cols)
    X_scaled_pd["customer_id"] = df_original["customer_id"].to_list()

    customers_id_list = neighbors_df["customer_id"].unique().tolist()
    explanation_list = []
    neighbors_raw_list = []

    for customer_id in customers_id_list:
        neighbor_indices = neighbors_df[neighbors_df["customer_id"] == customer_id][
            "neighbor_id"
        ].unique()

        explanation, neighbors_original = explain_neighbors_original_scale(
            customer_idx=customer_id,
            neighbor_indices=neighbor_indices,
            X_scaled=X_scaled_pd,
            X_original=df_original_pd,
            feature_names=feature_cols,
        )

        # Add customer_id so we can track later
        explanation["customer_id"] = customer_id
        neighbors_original = neighbors_original.copy()
        neighbors_original["source_customer_id"] = customer_id

        explanation_list.append(explanation)
        neighbors_raw_list.append(neighbors_original)

    # Combine everything into final DataFrames
    explanation_df = pd.concat(explanation_list, ignore_index=True)
    neighbors_detail_df = pd.concat(neighbors_raw_list)

    return explanation_df, neighbors_detail_df
