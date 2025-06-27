import logging
import re
import unicodedata

import inflection
import polars as pl

logger = logging.getLogger(__name__)


def clean_polars_column_names(polars_df: pl.DataFrame) -> pl.DataFrame:
    """
    Refactor pandas DataFrame column names including lower casing,
    and replacement of non-alpha characters with underscores or words.
    Args:
        polars_df: a dataframe

    Returns:
        pl.DataFrame with columns names in low
    """
    return polars_df.rename({col: clean_column_name(col) for col in polars_df.columns})


def clean_column_name(column_name: str) -> str:
    """
    Refactor column names including lowercasing,
    and replacing non-alphanumeric characters with underscores or words.

    Args:
        column_name (str): A 'dirty' column name.

    Returns:
        str: A 'clean' column name.
    """
    _old_column_name = column_name
    column_new = inflection.underscore(column_name.strip())
    column_new = (
        unicodedata.normalize("NFKD", column_new)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    column_new = column_new.replace("(%)", "_percent")
    column_new = re.sub(r"[ :_\.,;{}()'\n\t=]+", "_", column_new)
    column_new = re.sub(r"%", "percent", column_new)
    column_new = column_new.replace("?", "question")
    column_new = re.sub(r"[-]+", "minus", column_new)
    column_new = re.sub(r"[/]+", "by", column_new)
    column_new = re.sub(r"#", "number", column_new)
    column_new = re.sub(r"[&+]+", "and", column_new)
    column_new = re.sub(r"[|,;]+", "or", column_new)

    logger.debug("%s -> %s", _old_column_name, column_new)

    return column_new


def clean_string_value(value: str) -> str:
    """
    Standardizes string values by:
    - Lowercasing
    - Replacing spaces and special characters with underscores or meaningful text
    - Removing accents and normalizing unicode

    Args:
        value (str): Original string.

    Returns:
        str: Cleaned string.
    """
    original_value = value

    # Underscore formatting
    value = inflection.underscore(value.strip())

    # Remove accents and normalize
    value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("utf-8")
    )

    # Custom replacements
    value = value.replace("(%)", "_percent")
    value = re.sub(r"%", "percent", value)
    value = value.replace("?", "question")
    value = re.sub(r"[-]+", "minus", value)
    value = re.sub(r"[/]+", "by", value)
    value = re.sub(r"#", "number", value)
    value = re.sub(r"[&+]+", "and", value)
    value = re.sub(r"[|,;]+", "or", value)
    value = re.sub(r"[ :_\.,;{}()'\n\t=]+", "_", value)

    # Clean up multiple/trailing underscores
    value = re.sub(r"__+", "_", value)
    value = value.strip("_")

    logger.debug("%s -> %s", original_value, value)

    return value


def _clean_id(client_id):
    """
    Limpia el ID de un cliente removiendo guiones y puntos,
    y lo regresa como cadena con el prefijo 'id_'.

    Args:
        client_id (str): Valor numérico del ID del cliente.

    Returns:
        str: Cadena formateada como 'id_XXXXX'.
    """
    client_id = str(client_id)
    if "-" in client_id:
        client_id = client_id.split("-")[0].replace(".", "")
    final_str = "id_" + client_id.strip()
    final_str = final_str.replace(" ", "").replace("\xa0", "")
    return final_str


def clean_number(s):
    return re.sub(r"\D", "", s)  # \D matches any non-digit character


def _validate_customer_id(customer_id, group_id):
    """
    Retorna el ID final del cliente, usando el ID de grupo si existe,
    o el ID del cliente individual si no existe grupo.
    """
    if group_id != "sin_grupo":
        return group_id
    else:
        return customer_id


def calculate_weekly_transaction(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates the percentage of weeks each customer had transactions relative to the total number of weeks.

    Args:
        df (pl.DataFrame): Input DataFrame with columns 'customer_id' and 'year_week'.

    Returns:
        pl.DataFrame: A DataFrame with columns:
                      - 'customer_id'
                      - 'weeks_transacted'
                      - 'transaction_week_ratio' (float between 0 and 1)
    """
    # Step 1: Get all unique weeks in the dataset
    all_weeks = df.select("year_week").unique()

    # Step 2: Count total number of weeks in the period
    total_weeks = all_weeks.height

    # Step 3: Count number of unique weeks per customer
    weeks_per_customer = (
        df.select("customer_id", "year_week")
        .unique()
        .group_by("customer_id")
        .len()
        .rename({"len": "weeks_transacted"})
    )

    # Step 4: Compute percentage of weeks each customer transacted
    result = weeks_per_customer.with_columns(
        [(pl.col("weeks_transacted") / total_weeks).alias("transaction_week_ratio")]
    )

    return result
