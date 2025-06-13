import logging
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

from b2b_pricing_model.utils.dp_utils import (
    _clean_id,
    _validate_customer_id,
    calculate_weekly_transaction,
    clean_polars_column_names,
)

logger = logging.getLogger(__name__)

VALOR_IVA = 1.19  # Valor del IVA utilizado en los cálculos
PREFIX_ID = "id_"
FILTER_VALUE_QUITAR = "Quitar"
CUTOFF_DATE_RECENT = datetime(2026, 1, 1)
CUTOFF_DATE_UPPER = datetime(2025, 1, 1)
CUTOFF_DATE_LOWER = datetime(2024, 1, 1)


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
        pl.col("rut")
        .str.replace_all(r"\.", "")  # Remove dots
        .str.extract(r"(\d+)(?:-\d+)?")
        .str.replace_all(" ", "_")
        .alias("id_cliente")  # Extract digits before optional dash
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


def create_clients_to_remove(
    df_tae: pl.DataFrame, df_tct: pl.DataFrame, df_additional_tae: pl.DataFrame
) -> pl.DataFrame:
    """
    TODO: Add docstring for create_clients_to_remove function.
    """
    # Processing to_drop_tct
    df_tct = df_tct.drop("unnamed_4")
    df_tct = df_tct.filter(pl.col("que_hacer") == FILTER_VALUE_QUITAR)
    df_tct = df_tct.with_columns(
        id_cliente=(pl.lit(PREFIX_ID) + pl.col("rut").cast(pl.Int64).cast(pl.Utf8))
    )
    to_drop_tct = df_tct.select("id_cliente").unique()

    # Processing df_tae
    df_tae = df_tae.select(["rut", "vencimiento_contrato", "razon_social"])
    df_tae = df_tae.with_columns(
        pl.col("rut")
        .str.replace_all(r"\.", "")  # Remove dots
        .str.extract(r"(\d+)(?:-\d+)?")  # Extract digits before optional dash
        .alias("rut")
    )
    df_tae = df_tae.filter(
        (pl.col("rut").is_not_null())
        & (~pl.col("rut").str.to_lowercase().is_in(["nan", "none", "null", "na"]))
    ).filter(~pl.col("vencimiento_contrato").is_in(["Sin info"]))

    # Convert to datetime
    df_tae = df_tae.with_columns(
        pl.col("vencimiento_contrato").str.strptime(
            pl.Datetime, format="%Y-%m-%d %H:%M:%S"
        )
    )

    # Sort by vencimiento_contrato in descending order
    df_tae = df_tae.sort("vencimiento_contrato", descending=True)

    # Define contract expiration condition
    # "Rule from Dario for customer elimination"
    contract_expiration_condition = (
        pl.col("vencimiento_contrato") < CUTOFF_DATE_UPPER
    ) & (pl.col("vencimiento_contrato") > CUTOFF_DATE_LOWER)

    # Filter active customers
    to_drop_tae = df_tae.filter(
        (pl.col("vencimiento_contrato") > CUTOFF_DATE_RECENT)
        | contract_expiration_condition
    )

    # Apply _clean_id to create id_cliente column
    to_drop_tae = to_drop_tae.with_columns(
        id_cliente=pl.col("rut").map_elements(_clean_id)
    )

    # Get unique customer IDs

    to_drop_tae = to_drop_tae.select("id_cliente").unique()

    # Drop rows with null values in the "rut" column
    df_additional_tae = df_additional_tae.drop_nulls(subset=["rut"])

    # Select only the needed columns
    df_additional_tae = df_additional_tae.select(["rut", "que_hacer"])

    df_additional_tae = df_additional_tae.filter(
        (pl.col("rut").is_not_null())
        & (~pl.col("rut").str.to_lowercase().is_in(["nan", "none", "null", "na"]))
    )

    # Create id_cliente column by transforming the rut
    df_additional_tae = df_additional_tae.with_columns(
        pl.col("rut")
        .map_elements(lambda x: "id_" + str(int(str(x).split("-")[0].replace(".", ""))))
        .alias("id_cliente")
    )

    # Get list of id_cliente values where que_hacer equals "Quitar"
    to_drop_additional_tae = (
        df_additional_tae.filter(pl.col("que_hacer") == "Quitar")
        .select("id_cliente")
        .unique()
    )

    to_drop = pl.concat(
        [to_drop_tae, to_drop_tct, to_drop_additional_tae], how="diagonal"
    ).unique()

    return to_drop


def create_master_transactions(
    prm_base_tct_tae: pl.DataFrame,
    prm_base_cupon_electronico: pl.DataFrame,
    prm_maestro_data: pl.DataFrame,
    prm_competitiveness_index: pl.DataFrame,
    df_clients_to_remove: pl.DataFrame,
) -> pl.DataFrame:
    """
    TODO: Add docstring for create_master_transactions function.
    """
    # Merge the DataFrames

    # Step 1: Define the replacement mapping

    replace_dscto = {
        "T02": "sin_descuento",
        "D04": "precio_fijo",
        "D08": "dscto_fijo",
        "D07": "dscto_fijo",
        "D06": "dscto_fijo",
        "D02": "precio_fijo",
        "D03": "precio_fijo",
        "T01": "dscto_fijo",
        "0": "sin_descuento",
    }

    df_ = pl.concat([prm_base_tct_tae, prm_base_cupon_electronico], how="diagonal")

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
        [pl.col("tipo_descuento").replace(replace_dscto).alias("tipo_descuento")]
    )

    df_ = df_.with_columns(
        [
            pl.col("fecha_transaccion")
            .map_elements(lambda d: d.isocalendar()[0], return_dtype=pl.Int64)
            .alias("iso_year"),
            pl.col("fecha_transaccion")
            .map_elements(lambda d: d.isocalendar()[1], return_dtype=pl.Int64)
            .alias("iso_week"),
            pl.col("fecha_transaccion")
            .map_elements(
                lambda d: f"{d.isocalendar()[0]}-{d.isocalendar()[1]:02d}",
                return_dtype=pl.String,
            )
            .alias("year_week"),
        ]
    )

    df_ = df_.filter(
        ~pl.col("id_cliente").is_in(df_clients_to_remove["id_cliente"].to_list())
    )
    df_ = df_.join(prm_maestro_data, on="id_cliente", how="left")
    df_ = df_.with_columns(
        pl.col("id_grupo")
        .fill_null("sin_grupo")
        .str.replace("idg_SIN INFORMACION", "sin_grupo")
    )

    df_ = df_.with_columns(
        pl.struct(["id_cliente", "id_grupo"])
        .map_elements(
            lambda x: _validate_customer_id(x["id_cliente"], x["id_grupo"]),
            return_dtype=pl.String,  # Specify return type
        )
        .alias("customer_id")
    )

    df_ = df_.with_columns(
        customer_id=pl.when(pl.col("customer_id").is_null())
        .then(pl.col("id_cliente"))
        .otherwise(pl.col("customer_id"))
    )

    # Step 5: Convert `estacion` to string with prefix
    df_ = df_.with_columns(
        id_estacion=pl.lit("id_") + pl.col("estacion").cast(pl.Int64).cast(pl.Utf8)
    )

    df_ = df_.join(prm_competitiveness_index, on="id_estacion", how="left")

    df_ = df_.with_columns(
        [
            (pl.col("indicador_competencia") * pl.col("volumen")).alias(
                "indicador_competencia_volumen"
            )
        ]
    )

    df_ = df_.with_columns(
        codigo_region=pl.col("codigo_region").cast(pl.String),
    )

    return df_


def create_master_transactions_tct(
    mst_transactions: pl.DataFrame,
) -> pl.DataFrame:
    """
    TODO: Add docstring for create_master_transactions_tct function.
    """

    master_transactions_tct = mst_transactions.filter(
        pl.col("negocio_final").is_in(["TCT"])
    )

    return master_transactions_tct


def create_tct_spine_clientes(df_transactions: pl.DataFrame) -> pl.DataFrame:
    """
    TODO: Add docstring for create_spine_clientes function.
    """

    spine_clientes = df_transactions.select("customer_id").unique()

    return spine_clientes


def create_master_year_tct_trx(
    params: dict, mst_transactions: pl.DataFrame
) -> pl.DataFrame:
    """
    TODO: Add docstring for create_master_year_tct_CS_trx function.
    """
    key = params["key"]
    prefix = params["prefix"]

    max_total_trx_day = (
        mst_transactions.select("fecha_transaccion").max().to_series().to_list()[0]
    )
    # Create a unique DataFrame with year and customer-station combinations
    master_year_tct_trx = mst_transactions.group_by(key).agg(
        pl.col("volumen").sum().alias("yearly_volumen"),
        pl.col("monto_cobrado").sum().alias("yearly_charged_amount"),
        pl.col("margen_neto").sum().alias("yearly_margin"),
        pl.col("fecha_transaccion").count().alias("yearly_trx_count"),
        pl.col("codigo_region").n_unique().alias("yearly_regions_count"),
        pl.col("id_estacion").n_unique().alias("yearly_stations_count"),
        days_since_last_trx=(
            max_total_trx_day - pl.col("fecha_transaccion").max()
        ).dt.total_days(),
    )

    master_year_tct_trx = master_year_tct_trx.with_columns(
        yearly_avg_volume_per_trx=pl.col("yearly_volumen") / pl.col("yearly_trx_count"),
        yearly_charged_amount_per_liter=pl.col("yearly_charged_amount")
        / pl.col("yearly_volumen"),
        yearly_margin_per_liter=pl.col("yearly_margin") / pl.col("yearly_volumen"),
    )

    # Add prefix to all columns except the key columns
    key_cols = key if isinstance(key, list) else [key]
    master_year_tct_trx = master_year_tct_trx.rename(
        {
            col: f"{prefix}{col}"
            for col in master_year_tct_trx.columns
            if col not in key_cols
        }
    )

    weekly_transaction = calculate_weekly_transaction(df=mst_transactions)

    master_year_tct_trx = master_year_tct_trx.join(
        weekly_transaction, on=["customer_id"], how="left"
    )

    return master_year_tct_trx


def create_master_year_network_trx(
    params: dict, mst_transactions: pl.DataFrame
) -> pl.DataFrame:
    """TODO: Add docstring for create_master_year_network_trx function."""

    key = params["key"]
    prefix = params["prefix"]

    # Create a unique DataFrame with year and customer-station combinations
    master_year_network_trx = mst_transactions.group_by(key).agg(
        pl.col("volumen").sum().alias("yearly_network_volumen")
    )

    # Add prefix to all columns except the key columns
    key_cols = key if isinstance(key, list) else [key]
    master_year_network_trx = master_year_network_trx.rename(
        {
            col: f"{prefix}{col}"
            for col in master_year_network_trx.columns
            if col not in key_cols
        }
    )

    return master_year_network_trx


def create_master_year_region_tct_trx(
    params: dict, mst_transactions: pl.DataFrame, grouped_volume: pl.DataFrame
) -> pl.DataFrame:
    """TODO: Add docstring for create_master_year_region_tct_trx function."""
    df = mst_transactions.with_columns(
        codigo_region=pl.col("codigo_region").cast(pl.String),
    )
    # 1. Number of unique regions used per customer
    region_usage = df.group_by("customer_id").agg(
        number_of_regions_used=pl.col("codigo_region").n_unique()
    )

    # 2. Region diversity (entropy) per customer
    entropy_expr = (
        df.group_by("customer_id", "codigo_region")
        .agg(region_count=pl.col("fecha_transaccion").count())
        .group_by("customer_id")
        .agg(
            region_diversity_index=(
                pl.col("region_count") / pl.col("region_count").sum()
            ).map_elements(lambda p: -(p * np.log2(p)).sum())
        )
    )

    # 3. Volume by customer-region
    region_vol = df.group_by(["customer_id", "codigo_region"]).agg(
        region_volume=pl.col("volumen").sum()
    )

    # 4. Home region: region with highest volume per customer
    home_region = (
        region_vol.sort("region_volume", descending=True)
        .group_by("customer_id")
        .agg(home_region_volume=pl.col("region_volume").first())
    )

    home_region_ratio = (
        home_region.join(
            grouped_volume.select("customer_id", "c_yearly_volumen"),
            on="customer_id",
        )
        .with_columns(
            (pl.col("home_region_volume") / pl.col("c_yearly_volumen")).alias(
                "home_region_ratio"
            )
        )
        .select(["customer_id", "home_region_ratio"])
    )

    # 5. Pivoted region features

    tmp_volume_pivot = region_vol.pivot(
        values="region_volume",
        index="customer_id",
        on="codigo_region",
        aggregate_function="first",
    ).fill_null(pl.lit(0))

    # 5.1: Pivot - total volume per region per customer
    region_volume_pivot = tmp_volume_pivot.rename(
        {
            col: f"total_volume_region_{col}"
            for col in region_vol.select("codigo_region").unique().to_series().to_list()
        }
    )

    # 5.2: Share of customer’s volume in region over total volume of that customer
    share_customer_region = (
        region_vol.join(grouped_volume, on="customer_id")
        .with_columns(
            (pl.col("region_volume") / pl.col("c_yearly_volumen")).alias(
                "share_customer_region"
            )
        )
        .pivot(
            values="share_customer_region",
            index="customer_id",
            on="codigo_region",
            aggregate_function="first",
        )
        .rename(
            {
                col: f"share_customer_region_{col}"
                for col in region_vol.select("codigo_region")
                .cast(pl.String)
                .unique()
                .to_series()
                .to_list()
            }
        )
        .fill_null(pl.lit(0))
    )

    # 5.3: Share of customer's volume in region over total volume of that region (i.e., region’s total across all customers)
    region_total = df.group_by("codigo_region").agg(
        region_total_volume=pl.col("volumen").sum()
    )

    share_region_customer = (
        region_vol.join(region_total, on="codigo_region")
        .with_columns(
            (pl.col("region_volume") / pl.col("region_total_volume")).alias(
                "share_region_customer"
            )
        )
        .pivot(
            values="share_region_customer",
            index="customer_id",
            on="codigo_region",
            aggregate_function="first",
        )
        .rename(
            {
                col: f"share_region_customer_{col}"
                for col in region_vol.select("codigo_region")
                .unique()
                .to_series()
                .to_list()
            }
        )
        .fill_null(pl.lit(0))
    )

    region_features = (
        region_usage.join(entropy_expr, on="customer_id", how="left")
        .join(home_region_ratio, on="customer_id", how="left")
        .join(region_volume_pivot, on="customer_id", how="left")
        .join(share_customer_region, on="customer_id", how="left")
        .join(share_region_customer, on="customer_id", how="left")
    )

    return region_features


def create_master_monthly_tct_trx(
    params: dict, mst_transactions: pl.DataFrame
) -> pl.DataFrame:
    """
    TODO: Add docstring for create_master_year_tct_CS_trx function.
    """
    key = params["key"]
    prefix = params["prefix"]

    # Create a unique DataFrame with year and customer-station combinations
    master_monthly_tct_CS_trx = mst_transactions.group_by(key + ["ano_mes"]).agg(
        pl.col("volumen").sum().alias("monthly_volume"),
        pl.col("monto_cobrado").sum().alias("monthly_charged_amount"),
        pl.col("margen_neto").sum().alias("monthly_margin"),
        pl.col("fecha_transaccion").count().alias("monthly_trx_count"),
        pl.col("codigo_region").n_unique().alias("monthly_regions_count"),
        pl.col("id_estacion").n_unique().alias("monthly_stations_count"),
    )

    master_monthly_tct_CS_trx = master_monthly_tct_CS_trx.with_columns(
        perc_of_volume_potential=pl.col("monthly_volume")
        / pl.col("monthly_volume").max().over(key),
    )

    master_avg_monthly_tct_CS_trx = master_monthly_tct_CS_trx.group_by(key).agg(
        pl.col("monthly_volume").mean().alias("monthly_avg_volume"),
        pl.col("monthly_volume").std().alias("monthly_std_volume"),
        pl.col("monthly_trx_count").mean().alias("monthly_avg_trx_count"),
        pl.col("monthly_trx_count").std().alias("monthly_std_trx_count"),
        pl.col("monthly_regions_count").mean().alias("monthly_avg_regions_count"),
        pl.col("monthly_regions_count").std().alias("monthly_std_regions_count"),
        pl.col("monthly_stations_count").mean().alias("monthly_avg_stations_count"),
        pl.col("monthly_stations_count").std().alias("monthly_std_stations_count"),
        pl.col("perc_of_volume_potential")
        .mean()
        .alias("monthly_avg_perc_of_volume_potential"),
        pl.col("perc_of_volume_potential")
        .std()
        .alias("monthly_std_perc_of_volume_potential"),
    )

    # Add prefix to all columns except the key columns
    key_cols = key if isinstance(key, list) else [key]
    master_avg_monthly_tct_CS_trx = master_avg_monthly_tct_CS_trx.rename(
        {
            col: f"{prefix}{col}"
            for col in master_avg_monthly_tct_CS_trx.columns
            if col not in key_cols
        }
    )

    return master_avg_monthly_tct_CS_trx


def create_master_patentes(
    df: pl.DataFrame,
    int_patentes: pl.DataFrame,
) -> pl.DataFrame:
    """
    TODO: Add docstring for create_master_patentes function.
    """

    group_id_patentes = df.select("id_cliente", "customer_id").unique()
    group_id_patentes = group_id_patentes.join(
        int_patentes, on="id_cliente", how="left"
    )
    group_id_patentes = group_id_patentes.group_by("customer_id").agg(
        fleet_size=pl.col("num_patentes").sum()
    )

    return group_id_patentes


def create_master_yearly_station_tct_trx(
    params: dict, mst_transactions: pl.DataFrame
) -> pl.DataFrame:
    """TODO: Add docstring for create_master_yearly_station_tct_trx function."""
    key = params["key"]
    prefix = params["prefix"]

    master_station = mst_transactions.with_columns(
        indicador_competencia_volumen=pl.col("indicador_competencia")
        * pl.col("volumen")
    )

    master_station = master_station.group_by(key).agg(
        pl.col("volumen").sum().alias("yearly_volume"),
        pl.col("indicador_competencia_volumen")
        .sum()
        .alias("yearly_competitive_index_volume"),
    )

    master_station = master_station.with_columns(
        weighted_competitive_index=pl.col("yearly_competitive_index_volume")
        / pl.col("yearly_volume"),
    )

    master_station = master_station.select("customer_id", "weighted_competitive_index")

    # Add prefix to all columns except the key columns
    key_cols = key if isinstance(key, list) else [key]
    master_station = master_station.rename(
        {col: f"{prefix}{col}" for col in master_station.columns if col not in key_cols}
    )

    return master_station


def create_master_trx_customer_tct(  # noqa: PLR0913
    customer_spine: pl.DataFrame,
    master_year_tct_C_trx: pl.DataFrame,
    master_year_region_tct_trx: pl.DataFrame,
    master_month_tct_C_trx: pl.DataFrame,
    master_patentes: pl.DataFrame,
    master_yearly_station_tct_trx: pl.DataFrame,
    master_year_network_C_trx: pl.DataFrame,
) -> pl.DataFrame:
    """
    TODO:
    Add docstring for create_master_tct_customer function.
    """

    master_tct_customer = customer_spine.join(
        master_year_tct_C_trx, on=["customer_id"], how="left"
    )
    master_tct_customer = master_tct_customer.join(
        master_patentes, on="customer_id", how="left"
    )
    master_tct_customer = master_tct_customer.join(
        master_year_region_tct_trx, on="customer_id", how="left"
    )

    master_tct_customer = master_tct_customer.join(
        master_year_network_C_trx, on=["customer_id"], how="left"
    )

    master_tct_customer = master_tct_customer.join(
        master_month_tct_C_trx,
        on=["customer_id"],
        how="left",
    )

    master_tct_customer = master_tct_customer.join(
        master_yearly_station_tct_trx,
        on=["customer_id"],
        how="left",
    )

    master_tct_customer = master_tct_customer.with_columns(
        c_tct_yearly_perc_volume=pl.col("c_yearly_volumen")
        / pl.col("c_yearly_network_volumen")
    )

    master_tct_customer = master_tct_customer.with_columns(
        log_c_yearly_margin=pl.when(pl.col("c_yearly_margin") < 0)
        .then(pl.lit(0))
        .otherwise(pl.col("c_yearly_margin"))
        .map_elements(
            lambda p: np.log1p(
                p,
            ),
            return_dtype=pl.Float64,
        )
    )

    return master_tct_customer
