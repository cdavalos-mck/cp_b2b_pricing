import pandas as pd
import polars as pl

from b2b_pricing_model.utils.dp_utils import (
    clean_polars_column_names,
)


def preprocess_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    TODO: Add docstring for process_base_cupon_electronico function.
    """
    # Clean column names

    if isinstance(df, pd.DataFrame):
        for col in df.select_dtypes(include=["object"]).columns:
            # Try to ensure consistent types
            df[col] = df[col].astype(str)  # type: ignore

        df = pl.from_pandas(df)

    df = clean_polars_column_names(df)
    if "customer_id" in df.columns:
        df = df.with_columns(pl.col("customer_id").cast(pl.Int64).cast(pl.String))

    return df
