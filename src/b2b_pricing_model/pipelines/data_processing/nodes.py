import logging

import pandas as pd
import polars as pl

from b2b_pricing_model.utils.dp_utils import (
    clean_polars_column_names,
)

logger = logging.getLogger(__name__)

VALOR_IVA = 1.19  # Valor del IVA utilizado en los cálculos


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


def process_base_tae_tct(df: pl.DataFrame) -> pl.DataFrame:
    """
    TODO: Add docstring for process_base_cupon_electronico function.
    """

    key_columns = [
        "id_cliente",
        "tipo_descuento",
        "estacion",
        "fecha_transaccion",
        "ano_mes",
        "producto",
        "codigo_region",
        "negocio_final",
    ]

    grouped_columns = {
        "volumen": "sum",
        "monto": "sum",
        "monto_cobrado": "sum",
        "descuento": "sum",
        "precio_totem": "mean",
        "comision": "sum",
        "flete_unitario": "mean",
        "m1_total_trx": "sum",
        "flete_total_trx": "sum",
        "m2_total_trx": "sum",
    }

    # Prepare the aggregation expressions
    agg_exprs = [
        getattr(pl.col(col), func)().alias(col) for col, func in grouped_columns.items()
    ]

    # Apply groupby and aggregation
    df_ = df.group_by(key_columns).agg(agg_exprs)

    # Convert columns to appropriate data types
    df_ = df_.with_columns(
        margen_diagnostico=pl.col("m1_total_trx")
        + pl.col("m2_total_trx")
        + pl.col("flete_total_trx")
    ).with_columns(
        margen_neto=pl.col("m1_total_trx")
        + pl.col("m2_total_trx")
        - (pl.col("descuento") / 1.19)
        - pl.col("comision")
    )

    df_ = df_.with_columns(
        fecha_transaccion=pl.col("fecha_transaccion")
        .cast(pl.Utf8)  # First convert integer to string
        .str.strptime(pl.Datetime, format="%Y%m%d")  # Then parse string as datetime
    )

    df_ = df_.with_columns(
        [
            ("id_" + pl.col("id_cliente").cast(pl.Int64).cast(pl.String)).alias(
                "id_cliente"
            )
        ]
    )

    has_duplicates = df_.is_duplicated().any()

    # Ensure df does not have duplicate rows
    if has_duplicates:
        logger.error(f"Duplicated rows found:\n{df_.is_duplicated()}")
        raise Exception("Duplicated rows found")

    return df_


def process_base_cupon_electronico(df: pl.DataFrame) -> pl.DataFrame:
    """
    TODO: Add docstring for process_base_cupon_electronico function.
    """

    key_columns = [
        "id_cliente",
        "estacion",
        "fecha_transaccion",
        "ano_mes",
        "producto",
        "codigo_region",
        "negocio_final",
    ]

    grouped_columns = {
        "volumen": "sum",
        "monto": "sum",
        "monto_cobrado": "sum",
        "descuento": "sum",
        "comision": "sum",
        "flete_unitario": "mean",
        "m1_total_trx": "sum",
        "flete_total_trx": "sum",
        "m2_total_trx": "sum",
    }

    # Prepare the aggregation expressions
    agg_exprs = [
        getattr(pl.col(col), func)().alias(col) for col, func in grouped_columns.items()
    ]

    # Apply groupby and aggregation
    df_ = df.group_by(key_columns).agg(agg_exprs)

    # Convert columns to appropriate data types
    df_ = df_.with_columns(
        tipo_descuento=pl.when(pl.col("descuento") > 0)
        .then(pl.lit("precio_fijo"))
        .otherwise(pl.lit("sin_descuento"))
    )

    df_ = df_.with_columns(
        fecha_transaccion=pl.col("fecha_transaccion")
        .cast(pl.Utf8)  # First convert integer to string
        .str.strptime(pl.Datetime, format="%Y%m%d")  # Then parse string as datetime
    )

    df_ = df_.with_columns(
        [
            ("id_" + pl.col("id_cliente").cast(pl.Int64).cast(pl.String)).alias(
                "id_cliente"
            )
        ]
    )

    has_duplicates = df_.is_duplicated().any()

    # Ensure df does not have duplicate rows
    if has_duplicates:
        logger.error(f"Duplicated rows found:\n{df_.is_duplicated()}")
        raise Exception("Duplicated rows found")

    return df_


def process_base_empresas(df: pl.DataFrame) -> pl.DataFrame:
    """
    TODO: Add docstring for process_base_cupon_electronico function.
    """
    # Convert columns to appropriate data types
    df = df.with_columns(
        margen_unitario=pl.col("m1_unitario")
        + pl.col("m2_unitario")
        - pl.col("flete_unitario")
        - pl.col("comision") / pl.col("volumen")
        - pl.col("descuento") / VALOR_IVA / pl.col("volumen")
    )

    df = df.with_columns(margen=pl.col("margen_unitario") * pl.col("volumen"))

    df = df.with_columns(
        fecha_transaccion=pl.lit(None),
        tipo_descuento=pl.lit("zero_discount"),
    )

    df = df.filter(pl.col("descuento") > 0)

    has_duplicates = df.is_duplicated().any()

    # Ensure df does not have duplicate rows
    if has_duplicates:
        logger.error(f"Duplicated rows found:\n{df.is_duplicated()}")
        raise Exception("Duplicated rows found")

    return df


def process_maestro_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    TODO: Add docstring for process_contratos_tct function.
    """

    # Se carga un CSV maestro que mapea clientes a grupos. Se limpian nombres de columnas,
    # y se prepara la clave 'id_cliente' e 'id_grupo' para posteriormente unir con la tabla de transacciones.
    df = df.with_columns(
        pl.when(pl.col("rut").str.contains("-"))
        .then(pl.col("rut").str.split("-").list.get(0))
        .otherwise(pl.col("rut"))
        .str.replace_all(" ", "_")
        .alias("id_cliente")
    )

    # Select required columns
    df = df.select(
        ["id_cliente", "codigo_grupo", "razon_social", "vendedor", "nombre_grupo"]
    )

    # Add prefix "id_" to `id_cliente`
    df = df.with_columns((pl.lit("id_") + pl.col("id_cliente")).alias("id_cliente"))

    # Add prefix "idg_" to `codigo_grupo`, stored as `id_grupo`
    df = df.with_columns(
        (pl.lit("idg_") + pl.col("codigo_grupo").cast(pl.Utf8)).alias("id_grupo")
    )

    # Final selection of unique combinations
    df = df.unique(subset=["id_cliente", "id_grupo"]).select(["id_cliente", "id_grupo"])

    has_duplicates = df.is_duplicated().any()

    # Ensure df does not have duplicate rows
    if has_duplicates:
        logger.error(f"Duplicated rows found:\n{df.is_duplicated()}")
        raise Exception("Duplicated rows found")

    return df


def process_competitiveness_index(df: pl.DataFrame) -> pl.DataFrame:
    """
    TODO: Add docstring for process_competitiveness_index function.
    """

    df = df.select(
        [
            "id_estacion",
            "latitud",
            "longitud",
            "codigo_comuna",
            "indicador_competencia",
            "indicador_competencia_normalizado",
        ]
    )

    # Add "id_" prefix to `id_estacion`
    df = df.with_columns(
        (pl.lit("id_") + pl.col("id_estacion").cast(pl.Int64).cast(pl.Utf8)).alias(
            "id_estacion"
        )
    )

    has_duplicates = df.is_duplicated().any()

    # Ensure df does not have duplicate rows
    if has_duplicates:
        logger.error(f"Duplicated rows found:\n{df.is_duplicated()}")
        raise Exception("Duplicated rows found")

    return df
