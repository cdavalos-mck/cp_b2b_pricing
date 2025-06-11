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
