from pathlib import Path
from typing import Any

import polars as pl
from kedro.io.core import AbstractVersionedDataset, Version


class PolarsParquetDataset(AbstractVersionedDataset):
    def __init__(
        self,
        filepath: str,
        version: Version = None,
        load_args: dict[str, Any] = None,
        save_args: dict[str, Any] = None,
    ):
        self._load_args = load_args or {}
        self._save_args = save_args or {}

        # Initialize the parent class with filepath and version
        super().__init__(filepath, version)

    def _load(self) -> pl.DataFrame:
        # Use the versioned filepath from the parent class
        load_path_str = self._get_load_path()
        load_path = Path(load_path_str)
        return pl.read_parquet(load_path, **self._load_args)

    def _save(self, data: pl.DataFrame) -> None:
        # Use the versioned filepath from the parent class
        save_path_str = self._get_save_path()
        save_path = Path(save_path_str)
        # Ensure the directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data.write_parquet(save_path, **self._save_args)

    def _describe(self) -> dict[str, Any]:
        return dict(
            filepath=str(self._filepath),
            version=str(self._version) if self._version else None,
            load_args=self._load_args,
            save_args=self._save_args,
        )
