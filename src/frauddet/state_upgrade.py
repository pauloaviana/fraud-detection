"""Derive the v2 entity-state snapshot for a frozen stateful bundle (Phase 3A).

The v2 store keeps running totals so that online scoring reproduces the batch arithmetic bit for bit.
A frozen 1A bundle ships a v1 snapshot; this tool rebuilds the state from the SAME training rows
(membership.csv) with ``snapshot_from_frame`` and writes ``history-state.v2.json`` next to it, plus
``serving-extras.json`` with its sha256. No existing 1A/1B file (bundle.json included) is modified.

    python -m frauddet.state_upgrade --dataset sparkov --protocol temporal
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import __version__
from .adapters import get_adapter
from .history import STATE_FORMAT, snapshot_from_frame
from .prepare import prepare_frame
from .serving import FeatureBundle

V2_FILE = "history-state.v2.json"
EXTRAS = "serving-extras.json"


def upgrade(dataset: str, protocol: str, data_dir="data", artifacts="artifacts") -> dict:
    adir = Path(artifacts) / dataset / protocol
    bundle = FeatureBundle.load(adir, require_state=False)
    if bundle.history_spec is None:
        raise SystemExit(f"{dataset}/{protocol} is stateless; nothing to upgrade")
    df, _ = prepare_frame(get_adapter(dataset, data_dir), protocol)
    member = pd.read_csv(adir / "membership.csv")
    train_rows = member.loc[member["part"] == member["part"].iloc[0], "row"].to_numpy()
    train = df.iloc[train_rows]
    st = snapshot_from_frame(train, bundle.history_spec, bundle.contract.row_id)
    st.save(adir / V2_FILE)
    sha = hashlib.sha256((adir / V2_FILE).read_bytes()).hexdigest()
    extras = {"frauddet_version": __version__, "format": STATE_FORMAT, "derived_from": "training part (membership.csv)",
              "rows": int(len(train)), "entities": len(st.last), "buffered_events": sum(len(v) for v in st.events.values()),
              "files": {V2_FILE: sha}}
    (adir / EXTRAS).write_text(json.dumps(extras, indent=1))
    return extras


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", default="temporal")
    a = p.parse_args(argv)
    print(json.dumps(upgrade(a.dataset, a.protocol)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
