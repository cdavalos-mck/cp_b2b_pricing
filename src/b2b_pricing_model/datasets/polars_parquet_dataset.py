from pathlib import Path
from typing import Any, Dict

import polars as pl
from kedro.io.core import AbstractDataset


class PolarsParquetDataSet(AbstractDataset):
    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
    ):
        self._filepath = Path(filepath)
        self._load_args = load_args or {}
        self._save_args = save_args or {}

    def _load(self) -> pl.DataFrame:
        return pl.read_parquet(self._filepath, **self._load_args)

    def _save(self, data: pl.DataFrame) -> None:
        data.write_parquet(self._filepath, **self._save_args)

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=str(self._filepath))
