import polars as pl

from b2b_pricing_model.utils.ds_utils import ClusteringPipeline


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

    comma_columns = params["features"]["numeric"]["comma_columns"]
    percent_columns = params["features"]["numeric"]["percent_columns"]

    numeric_columns = comma_columns + percent_columns
    categorical_columns = params["features"]["categorical"]

    if categorical_columns is None:
        df_pandas = df.select(numeric_columns).to_pandas()
    if numeric_columns is None:
        df_pandas = df.select(categorical_columns).to_pandas()
    if numeric_columns is not None and categorical_columns is not None:
        df_pandas = df.select(numeric_columns + categorical_columns).to_pandas()

    X = df_pandas.copy()
    X_scaled = pipeline.preprocess_data(X=X, method=params["standardize_method"])

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

    pipeline.X_scaled = X_scaled

    # Assuming the pipeline has a method to get k-nearest neighbors
    knn_results = pipeline.get_k_nearest_neighbors(
        k=params["k_neighbors"],
        customer_ids=master_trx_customer_tct["customer_id"].to_list(),
    )

    return knn_results
