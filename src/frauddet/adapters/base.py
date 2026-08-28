"""Read-only adapters over the raw files.

An adapter knows where a dataset's files live (bare csv or zip member), reads
them with the dtypes declared in the contract, and exposes dataset-specific
audit checks. It never writes, never renames columns, never derives values.
Zip archives are streamed in place; nothing is extracted into the repository.
"""

from __future__ import annotations

import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

import pandas as pd

from ..contracts import DatasetContract, FileSpec, Kind
from ..findings import Finding


class RawAdapter:
    contract: DatasetContract

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    # -- files ------------------------------------------------------------------
    def path(self, key: str) -> Path:
        fs = self.contract.file(key)
        return self.data_dir / (fs.container or fs.member)

    def available(self, key: str) -> bool:
        return self.path(key).exists()

    @contextmanager
    def open(self, key: str) -> Iterator[IO[bytes]]:
        fs: FileSpec = self.contract.file(key)
        if fs.container:
            with zipfile.ZipFile(self.data_dir / fs.container) as zf, zf.open(fs.member) as fh:
                yield fh
        else:
            with open(self.data_dir / fs.member, "rb") as fh:
                yield fh

    def header(self, key: str) -> list[str]:
        with self.open(key) as fh:
            return list(pd.read_csv(fh, nrows=0).columns)

    # -- loading ----------------------------------------------------------------
    def load(self, key: str, usecols: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
        """Load a file with contract dtypes. Unexpected columns are read as-is (object)."""
        header = self.header(key)
        specs, _unexpected = self.contract.resolve(header)
        wanted = set(usecols) if usecols else set(header)
        dtypes = {s.name: s.pandas_dtype for s in specs if s.pandas_dtype and s.name in wanted}
        dates = [s.name for s in specs if s.kind is Kind.DATETIME and s.name in wanted]
        with self.open(key) as fh:
            df = pd.read_csv(fh, dtype=dtypes, usecols=usecols, nrows=nrows, parse_dates=dates or None)
        return df

    # -- dataset-specific audit checks -------------------------------------------
    def checks(self, frames: dict[str, pd.DataFrame]) -> list[Finding]:
        """Suspicious-field / consistency checks specific to the dataset. Read-only."""
        return []
