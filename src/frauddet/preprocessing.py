"""Foundational preprocessing (Phase 1A.4).

Row-wise, reproducible transformations only — no history, no resampling, no
feature selection. Every step is either stateless or learns its state from the
frame it is *fitted* on (the training part), and every step serialises to plain
JSON, so the fitted pipeline can be reloaded at serving time and applied to a
single event with results identical to the batch run (tests/test_preprocessing.py
checks this parity).

Per-dataset pipelines are assembled by ``build_pipeline`` from the frozen
contracts; the last step (``FinalizeFeatures``) removes every column whose role
is not an input role and fixes the output column order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .contracts import INPUT_ROLES, DatasetContract

DAY = 86_400
WEEK = 7 * DAY
MA2026_PROTOCOL = "stratified_ma2026"


# ------------------------------------------------------------------------------ base
class Step:
    """A serialisable transformation. Subclasses set ``params`` in __init__ and, if they learn,
    implement ``_fit`` (returning the learned state dict) and use ``self.state`` in ``transform``."""

    params: dict[str, Any]
    state: dict[str, Any]

    def __init__(self, **params: Any):
        self.params = params
        self.state = {}
        self.fitted = False

    def _fit(self, df: pd.DataFrame) -> dict[str, Any]:
        return {}

    def fit(self, df: pd.DataFrame) -> "Step":
        self.state = self._fit(df)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {"kind": type(self).__name__, "params": self.params, "state": self.state}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Step":
        obj = STEP_TYPES[d["kind"]](**d["params"])
        obj.state, obj.fitted = d["state"], True
        return obj


# ------------------------------------------------------------------------------ stateless steps
class DropColumns(Step):
    def __init__(self, columns: list[str]):
        super().__init__(columns=list(columns))

    def transform(self, df):
        return df.drop(columns=[c for c in self.params["columns"] if c in df.columns])


class StripPrefix(Step):
    """Cosmetic normalisation (Sparkov merchant names all start with 'fraud_')."""

    def __init__(self, column: str, prefix: str):
        super().__init__(column=column, prefix=prefix)

    def transform(self, df):
        c, p = self.params["column"], self.params["prefix"]
        out = df.copy()
        out[c] = out[c].astype("string").str.removeprefix(p)
        return out


class LogAmount(Step):
    def __init__(self, column: str, output: str | None = None):
        super().__init__(column=column, output=output or f"{column}_log1p")

    def transform(self, df):
        out = df.copy()
        out[self.params["output"]] = np.log1p(np.clip(out[self.params["column"]].astype(float), 0, None))
        return out


class AmountDecimals(Step):
    """Number of decimals (0..3) of the shipped amount. A property of the value as recorded
    (IEEE: 3-decimal amounts are currency-converted) — no semantics invented."""

    def __init__(self, column: str, output: str | None = None):
        super().__init__(column=column, output=output or f"{column}_decimals")

    def transform(self, df):
        x = df[self.params["column"]].astype(float).to_numpy()
        dec = np.full(len(x), 3, dtype=np.int8)
        for d in (2, 1, 0):
            dec[np.isclose(x, np.round(x, d), atol=1e-9)] = d
        out = df.copy()
        out[self.params["output"]] = dec
        return out


class CalendarFeatures(Step):
    """Calendar features from a wall-clock timestamp (Sparkov trans_date_trans_time). The timezone
    is unknown: these are 'dataset clock' values, never local time."""

    def __init__(self, column: str, prefix: str = "cal"):
        super().__init__(column=column, prefix=prefix)

    def transform(self, df):
        c, p = self.params["column"], self.params["prefix"]
        ts = pd.to_datetime(df[c])
        out = df.copy()
        hour, dow, month = ts.dt.hour, ts.dt.dayofweek, ts.dt.month
        out[f"{p}_hour"] = hour.astype("int8")
        out[f"{p}_dow"] = dow.astype("int8")
        out[f"{p}_month"] = month.astype("int8")
        frac = (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second) / DAY
        out[f"{p}_day_sin"] = np.sin(2 * np.pi * frac).astype("float32")
        out[f"{p}_day_cos"] = np.cos(2 * np.pi * frac).astype("float32")
        out[f"{p}_dow_sin"] = np.sin(2 * np.pi * dow / 7).astype("float32")
        out[f"{p}_dow_cos"] = np.cos(2 * np.pi * dow / 7).astype("float32")
        return out


class AgeAtEvent(Step):
    def __init__(self, dob: str, event_time: str, output: str = "age_years"):
        super().__init__(dob=dob, event_time=event_time, output=output)

    def transform(self, df):
        out = df.copy()
        delta = pd.to_datetime(df[self.params["event_time"]]) - pd.to_datetime(df[self.params["dob"]])
        out[self.params["output"]] = (delta.dt.days / 365.25).astype("float32")
        return out


class HaversineDistance(Step):
    def __init__(self, lat: str, lon: str, lat2: str, lon2: str, output: str = "dist_km"):
        super().__init__(lat=lat, lon=lon, lat2=lat2, lon2=lon2, output=output)

    def transform(self, df):
        p = self.params
        la1, lo1 = np.radians(df[p["lat"]].astype(float)), np.radians(df[p["lon"]].astype(float))
        la2, lo2 = np.radians(df[p["lat2"]].astype(float)), np.radians(df[p["lon2"]].astype(float))
        a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        out = df.copy()
        out[p["output"]] = (2 * 6371.0088 * np.arcsin(np.sqrt(a))).astype("float32")
        return out


class CyclicClock(Step):
    """sin/cos of the phase of a relative clock within ``period_seconds`` (+ optional integer bin).

    ``anchor`` documents what the phase means: 'anchored' (IEEE: TransactionDT mod 86400 reproduces
    D9·24 exactly, so the daily phase is Vesta's own hour-of-day) or 'relative' (ULB: hour of the
    24 h cycle counted from the dataset start; IEEE week: day index of a 7-day cycle, weekday unknown).
    """

    def __init__(self, column: str, period_seconds: int, prefix: str, bins: int | None = None,
                 anchor: str = "relative"):
        super().__init__(column=column, period_seconds=period_seconds, prefix=prefix, bins=bins, anchor=anchor)

    def transform(self, df):
        p = self.params
        t = df[p["column"]].astype(float).to_numpy()
        phase = np.mod(t, p["period_seconds"]) / p["period_seconds"]
        out = df.copy()
        out[f"{p['prefix']}_sin"] = np.sin(2 * np.pi * phase).astype("float32")
        out[f"{p['prefix']}_cos"] = np.cos(2 * np.pi * phase).astype("float32")
        if p["bins"]:
            out[f"{p['prefix']}_bin"] = np.floor(phase * p["bins"]).astype("int16")
        return out


class CategoricalTyper(Step):
    """Fixes a train-learned category set and emits a pandas ``category`` column. Missing -> "<NA>",
    unseen -> "<UNK>". The integer codes carry NO ordinal meaning: model views decide whether to use
    native categorical support (tree) or one-hot (linear)."""

    def __init__(self, columns: list[str]):
        super().__init__(columns=list(columns))

    def _fit(self, df):
        cats = {}
        for c in self.params["columns"]:
            vc = df[c].astype("string").dropna().value_counts()
            cats[c] = ["<NA>", "<UNK>"] + [str(v) for v in vc.index]
        return {"categories": cats}

    def transform(self, df):
        out = df.copy()
        for c, cats in self.state["categories"].items():
            s = df[c].astype("string").fillna("<NA>")
            s = s.where(s.isin(cats), "<UNK>")
            out[c] = pd.Categorical(s, categories=cats)
        return out


class FrequencyEncoder(Step):
    """Adds <col>_freq = relative frequency of the value in the training part (unseen/missing -> 0)."""

    def __init__(self, columns: list[str]):
        super().__init__(columns=list(columns))

    def _fit(self, df):
        freq = {}
        for c in self.params["columns"]:
            vc = df[c].astype("string").dropna().value_counts(normalize=True)
            freq[c] = {str(k): float(v) for k, v in vc.items()}
        return {"freq": freq}

    def transform(self, df):
        out = df.copy()
        for c, fr in self.state["freq"].items():
            out[f"{c}_freq"] = df[c].astype("string").map(fr).astype("float32").fillna(0.0)
        return out


class TokenSplit(Step):
    """Value-derived family of a string column: the token at ``index`` after splitting on ``pattern``
    (e.g. email provider 'gmail' from 'gmail.com', OS family 'Windows' from 'Windows 10')."""

    def __init__(self, column: str, output: str, pattern: str = r"[.\s/]", index: int = 0):
        super().__init__(column=column, output=output, pattern=pattern, index=index)

    def transform(self, df):
        p = self.params
        out = df.copy()
        parts = df[p["column"]].astype("string").str.split(p["pattern"], regex=True)
        out[p["output"]] = parts.str[p["index"]].astype("string")
        return out


class RowMissingCount(Step):
    """Number of missing values per row among the columns matching ``pattern`` (a property of the
    record, e.g. how many V columns are absent)."""

    def __init__(self, pattern: str, output: str):
        super().__init__(pattern=pattern, output=output)

    def _fit(self, df):
        import re
        return {"columns": [c for c in df.columns if re.fullmatch(self.params["pattern"], c)]}

    def transform(self, df):
        out = df.copy()
        cols = [c for c in self.state["columns"] if c in df.columns]
        out[self.params["output"]] = df[cols].isna().sum(axis=1).astype("int16")
        return out


# ------------------------------------------------------------------------------ learned steps
class CategoricalEncoder(Step):
    """Ordinal codes learned on the training part: categories ranked by frequency get 1..K;
    unseen values and missing map to 0. Output replaces the column (int32)."""

    def __init__(self, columns: list[str]):
        super().__init__(columns=list(columns))

    def _fit(self, df):
        cats = {}
        for c in self.params["columns"]:
            s = df[c].astype("string")
            vc = s.dropna().value_counts()
            cats[c] = vc.index.tolist()
        return {"categories": cats}

    def transform(self, df):
        out = df.copy()
        for c, cats in self.state["categories"].items():
            mapping = {v: i + 1 for i, v in enumerate(cats)}
            out[c] = df[c].astype("string").map(mapping).fillna(0).astype("int32")
        return out


class MissingIndicator(Step):
    """Adds <col>__isna for every column that had at least one null in the training part."""

    def __init__(self, columns: list[str] | None = None):
        super().__init__(columns=list(columns) if columns else None)

    def _fit(self, df):
        cols = self.params["columns"] or list(df.columns)
        return {"columns": [c for c in cols if c in df.columns and bool(df[c].isna().any())]}

    def transform(self, df):
        cols = self.state["columns"]
        if not cols:
            return df.copy()
        ind = pd.DataFrame({f"{c}__isna": df[c].isna().astype("int8") for c in cols}, index=df.index)
        return pd.concat([df, ind], axis=1)


class MedianImputer(Step):
    """Medians learned on the training part for every numeric column given (or all numeric), so any
    null seen at serving time is imputable even if the column had no nulls in training."""

    def __init__(self, columns: list[str] | None = None):
        super().__init__(columns=list(columns) if columns else None)

    def _fit(self, df):
        cols = self.params["columns"] or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c].dtype)]
        med = {}
        for c in cols:
            m = df[c].median()
            med[c] = float(m) if pd.notna(m) else 0.0
        return {"medians": med}

    def transform(self, df):
        med = {c: m for c, m in self.state["medians"].items() if c in df.columns}
        return df.fillna(value=med)


class Standardize(Step):
    """(x - mean) / std with statistics from the training part (std 0 -> 1)."""

    def __init__(self, columns: list[str]):
        super().__init__(columns=list(columns))

    def _fit(self, df):
        stats = {}
        for c in self.params["columns"]:
            x = df[c].astype(float)
            sd = float(x.std(ddof=0))
            stats[c] = {"mean": float(x.mean()), "std": sd if sd > 0 else 1.0}
        return {"stats": stats}

    def transform(self, df):
        cols = list(self.state["stats"])
        mean = np.array([self.state["stats"][c]["mean"] for c in cols], dtype=float)
        std = np.array([self.state["stats"][c]["std"] for c in cols], dtype=float)
        # one vectorised block instead of a per-column pandas loop (identical values; ~30x faster on 1-row frames)
        z = ((df[cols].to_numpy(dtype=float) - mean) / std).astype("float32")
        out = df.copy()
        out[cols] = pd.DataFrame(z, index=df.index, columns=cols)
        return out


class FinalizeFeatures(Step):
    """Keeps only input-role columns and derived columns; fixes the output order at fit time and
    enforces it at transform time (serving parity). ``contract_name`` resolves roles."""

    def __init__(self, contract_name: str, extra_exclude: list[str] | None = None,
                 keep_order_key: bool = False):
        super().__init__(contract_name=contract_name, extra_exclude=list(extra_exclude or []),
                         keep_order_key=keep_order_key)

    def _fit(self, df):
        from .adapters import ADAPTERS
        contract = ADAPTERS[self.params["contract_name"]].contract
        keep = []
        for c in df.columns:
            if c in self.params["extra_exclude"]:
                continue
            s = contract.spec_for(c)
            if s is None or s.role in INPUT_ROLES:       # derived (no spec) or input role
                keep.append(c)
            elif self.params["keep_order_key"] and c == contract.order_key:
                keep.append(c)                            # Ma-2026 benchmark only (DECISIONS D11)
        return {"feature_columns": keep}

    def transform(self, df):
        cols = self.state["feature_columns"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"serving frame lacks feature columns: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        return df[cols].copy()


STEP_TYPES = {cls.__name__: cls for cls in (
    DropColumns, StripPrefix, LogAmount, AmountDecimals, CalendarFeatures, AgeAtEvent, HaversineDistance,
    CyclicClock, CategoricalTyper, FrequencyEncoder, TokenSplit, RowMissingCount, CategoricalEncoder,
    MissingIndicator, MedianImputer, Standardize, FinalizeFeatures)}


# ------------------------------------------------------------------------------ pipeline
@dataclass
class Pipeline:
    dataset: str
    contract_version: str
    protocol: str
    steps: list[Step]
    fitted_on: dict[str, Any] | None = None

    @property
    def feature_columns(self) -> list[str]:
        return self.steps[-1].state["feature_columns"]

    def fit(self, train: pd.DataFrame, order_key: str | None = None) -> "Pipeline":
        df = train
        for step in self.steps:
            df = step.fit(df).transform(df)
        self.fitted_on = {"rows": int(len(train)),
                          "order_key_range": ([float(train[order_key].min()), float(train[order_key].max())]
                                              if order_key and order_key in train else None),
                          "n_features": len(df.columns)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.fitted_on is None:
            raise RuntimeError("pipeline not fitted")
        for step in self.steps:
            df = step.transform(df)
        return df

    def to_dict(self) -> dict[str, Any]:
        return {"frauddet_version": __version__, "dataset": self.dataset, "contract_version": self.contract_version,
                "protocol": self.protocol, "fitted_on": self.fitted_on, "steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Pipeline":
        return cls(d["dataset"], d["contract_version"], d["protocol"], [Step.from_dict(s) for s in d["steps"]],
                   d["fitted_on"])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Pipeline":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ------------------------------------------------------------------------------ per-dataset pipelines
def build_pipeline(contract: DatasetContract, protocol: str = "temporal") -> Pipeline:
    name = contract.name
    if name == "sparkov":
        steps = [
            StripPrefix("merchant", "fraud_"),
            CalendarFeatures("trans_date_trans_time", prefix="cal"),          # calendar from the datetime
            AgeAtEvent("dob", "trans_date_trans_time"),
            HaversineDistance("lat", "long", "merch_lat", "merch_long"),
            LogAmount("amt"),
            FrequencyEncoder(["merchant", "category", "state"]),
            CategoricalTyper(["merchant", "category", "state", "gender"]),   # typed, not ordinal
            FinalizeFeatures(name),      # drops unix_time (order only), event time, PII, excluded, target
        ]
    elif name == "ieee":
        string_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo",
                       *[f"M{i}" for i in range(1, 10)],
                       *[f"id_{i:02d}" for i in (12, 15, 16, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38)]]
        derived_cat = ["P_email_provider", "P_email_tld", "R_email_provider", "os_family", "browser_family",
                       "device_family"]
        steps = [
            AmountDecimals("TransactionAmt"),
            LogAmount("TransactionAmt"),
            CyclicClock("TransactionDT", DAY, prefix="clk_day", bins=24, anchor="anchored: equals D9*24"),
            CyclicClock("TransactionDT", WEEK, prefix="clk_week", bins=7, anchor="relative: weekday unknown"),
            TokenSplit("P_emaildomain", "P_email_provider", index=0),
            TokenSplit("P_emaildomain", "P_email_tld", index=-1),
            TokenSplit("R_emaildomain", "R_email_provider", index=0),
            TokenSplit("id_30", "os_family", index=0),
            TokenSplit("id_31", "browser_family", index=0),
            TokenSplit("DeviceInfo", "device_family", index=0),
            RowMissingCount(r"V\d+", "n_missing_V"),
            RowMissingCount(r"D\d+", "n_missing_D"),
            RowMissingCount(r"id_\d\d", "n_missing_id"),
            MissingIndicator(),                              # every column with a null in train (reversible)
            FrequencyEncoder(string_cols + derived_cat),     # train frequencies, unseen -> 0
            CategoricalTyper(string_cols + derived_cat),     # typed categoricals; NO imputation of numerics
            FinalizeFeatures(name),                          # drops TransactionDT/ID, target
        ]
    elif name == "ulb" and protocol == "temporal":
        v = [f"V{i}" for i in range(1, 29)]
        steps = [
            LogAmount("Amount"),
            CyclicClock("Time", DAY, prefix="clk_day", bins=24, anchor="relative: phase unknown, 2 cycles"),
            Standardize(v + ["Amount_log1p"]),
            FinalizeFeatures(name, extra_exclude=["Amount"]),   # Time (order only) dropped by role
        ]
    elif name == "ulb" and protocol == MA2026_PROTOCOL:
        steps = [
            Standardize(["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]),   # Ma: StandardScaler on train
            FinalizeFeatures(name, keep_order_key=True),   # Ma use Time as an input — benchmark only
        ]
    else:
        raise ValueError(f"no pipeline for {name}/{protocol}")
    return Pipeline(name, contract.contract_version, protocol, steps)
