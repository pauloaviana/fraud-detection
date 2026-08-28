"""Causal per-entity history features (Phase 1A.5) — Sparkov only (the only dataset with an entity key).

Correa Bahnsen et al. (2016) methodology, restricted to information available *before* the event:

* agg1  — per card, number and amount of transactions in the previous tp hours,
          tp ∈ {1, 3, 6, 12, 18, 24, 72, 168} (paper's windows).
* agg2  — the same, restricted to prior transactions with the same category / same merchant.
* last  — hours since the previous transaction, previous amount, ratio to it.
* deviation — amount relative to the card's 168 h mean / z-score, and to its same-category 168 h mean.
* geo   — distance from the previous merchant location, implied speed, distance-from-home relative to the
          card's 168 h mean.
* periodic (von Mises) — for prior transactions in the last 7 / 30 days (paper: ≥ 7 days), the circular
          mean and concentration of the time of day; a flag whether the current time lies inside the
          α = 0.9 interval (the paper's demonstrated α), the circular distance to the mean (hours), the
          mean resultant length R and the number of prior events used.

"Prior" means strictly earlier in the chronological event order (order key, then file order for ties);
an event never sees itself or anything later. No labels are used. Because the features are computed on
the whole chronological frame, an event in the validation or holdout part sees the earlier *observed*
validation/holdout transactions of the same card as well as the training-period ones — exactly what a
live sequential scorer sees, since transactions (not labels) are observed as they arrive. Only labels are
withheld, and no label ever enters these features. Nothing is fitted: windows and α are
configuration, so the batch computation (``compute_history``) and the serving-side ``EntityStateStore``
produce identical values (tests check parity).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import vonmises

HOUR = 3600.0
DAY = 86400.0
EARTH_KM = 6371.0088


@dataclass(frozen=True)
class HistorySpec:
    entity: str = "cc_num"
    order: str = "unix_time"                   # seconds; ordering and deltas only
    event_time: str = "trans_date_trans_time"  # wall-clock; time-of-day angle for the periodic features
    amount: str = "amt"
    conds: tuple[str, ...] = ("category", "merchant")
    lat: str = "lat"
    lon: str = "long"
    merch_lat: str = "merch_lat"
    merch_lon: str = "merch_long"
    windows_h: tuple[int, ...] = (1, 3, 6, 12, 18, 24, 72, 168)
    deviation_window_h: int = 168
    vm_windows_d: tuple[int, ...] = (7, 30)
    vm_alpha: float = 0.9
    vm_min_prior: int = 3
    speed_min_gap_s: float = 60.0              # speed undefined below this gap (same-minute events)

    @property
    def max_window_s(self) -> float:
        return max(max(self.windows_h) * HOUR, max(self.vm_windows_d) * DAY, self.deviation_window_h * HOUR)

    def feature_names(self) -> list[str]:
        names = ["h_n_prior"]
        for w in self.windows_h:
            names += [f"h_cnt_{w}h", f"h_amt_{w}h"]
        for c in self.conds:
            for w in self.windows_h:
                names += [f"h_{c}_cnt_{w}h", f"h_{c}_amt_{w}h"]
        names += ["h_hours_since_last", "h_prev_amt", "h_amt_ratio_prev",
                  f"h_amt_over_mean_{self.deviation_window_h}h", f"h_amt_z_{self.deviation_window_h}h"]
        names += [f"h_amt_over_{c}_mean_{self.deviation_window_h}h" for c in self.conds]
        names += ["h_dist_prev_km", "h_speed_kmh", f"h_dist_home_mean_{self.deviation_window_h}h",
                  f"h_dist_home_ratio_{self.deviation_window_h}h"]
        for d in self.vm_windows_d:
            names += [f"vm{d}_n", f"vm{d}_R", f"vm{d}_dist_h", f"vm{d}_in_ci{int(self.vm_alpha * 100)}"]
        return names


# ------------------------------------------------------------------------------ helpers
def haversine_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (np.asarray(lat1, float), np.asarray(lon1, float),
                                          np.asarray(lat2, float), np.asarray(lon2, float)))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def kappa_from_R(R):
    """Best & Fisher approximation of the von Mises concentration from the mean resultant length."""
    R = np.clip(np.asarray(R, float), 0.0, 0.9999)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(R < 0.53, 2 * R + R ** 3 + 5 * R ** 5 / 6,
                     np.where(R < 0.85, -0.4 + 1.39 * R + 0.43 / (1 - R), 1 / (R ** 3 - 4 * R ** 2 + 3 * R)))
    return np.clip(np.nan_to_num(k, nan=0.0, posinf=1e3), 0.0, 1e3)


_VM_GRID: dict[float, tuple[np.ndarray, np.ndarray]] = {}
_VM_KAPPA_GRID = np.concatenate([np.linspace(1e-6, 1.0, 400, endpoint=False), np.geomspace(1.0, 1e3, 1200)])


def vm_halfwidth(alpha: float, kappa):
    """Half-width q (radians) of the central interval containing probability alpha under VM(0, kappa).

    scipy solves the von Mises ppf numerically per element; for millions of rows that is minutes, so q is
    tabulated once on a fixed kappa grid (1,600 points, exact scipy values) and linearly interpolated.
    Deterministic and shared by the batch computation and the serving store, so parity is unaffected.
    """
    if alpha not in _VM_GRID:
        _VM_GRID[alpha] = (_VM_KAPPA_GRID, vonmises.interval(alpha, _VM_KAPPA_GRID)[1])
    kg, qg = _VM_GRID[alpha]
    kappa = np.asarray(kappa, float)
    out = np.full(kappa.shape, np.pi)
    ok = kappa > 1e-6
    if ok.any():
        out[ok] = np.interp(np.clip(kappa[ok], kg[0], kg[-1]), kg, qg)
    return out


def circ_dist(a, b):
    d = np.mod(np.asarray(a, float) - np.asarray(b, float) + np.pi, 2 * np.pi) - np.pi
    return np.abs(d)


def time_of_day_angle(event_time: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(event_time)
    sec = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    return (2 * np.pi * sec / DAY).to_numpy(dtype=float)


def _window_sums(tk: np.ndarray, w: float, *cums: np.ndarray):
    """For sorted per-group keys tk (group*BIG + t), prior events j < i with tk_j > tk_i - w.
    Returns count and, for each exclusive cumsum given, the windowed sum."""
    i = np.arange(len(tk))
    lo = np.searchsorted(tk, tk - w, side="right")
    cnt = i - lo
    return (cnt, *[c[i] - c[lo] for c in cums])


def _excl_cumsum(x: np.ndarray) -> np.ndarray:
    c = np.zeros(len(x) + 1)
    np.cumsum(x, out=c[1:])
    return c


def _excl_cumsum_by_group(x: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Exclusive cumulative sums restarted at every group boundary (``group`` sorted, contiguous), with the
    same sequential float64 accumulation as ``np.cumsum`` inside each group. Entity-local accumulation
    keeps an entity's features independent of other entities' rounding and lets an online store reproduce
    the batch arithmetic exactly. Returns length len(x)+1 like ``_excl_cumsum`` (only [:-1] is used)."""
    n = len(x)
    c = np.zeros(n + 1)
    if n == 0:
        return c
    starts = np.r_[0, np.flatnonzero(group[1:] != group[:-1]) + 1, n]
    for a, b in zip(starts[:-1], starts[1:]):
        c[a + 1:b + 1] = np.cumsum(x[a:b])
    incl = c[1:].copy()
    excl = np.empty(n)
    excl[1:] = incl[:-1]
    excl[0] = 0.0
    excl[starts[1:-1]] = 0.0
    c[:-1] = excl
    return c


# ------------------------------------------------------------------------------ batch computation
def compute_history(df: pd.DataFrame, spec: HistorySpec = HistorySpec()) -> pd.DataFrame:
    """Return ``df`` (original row order) with the history features appended. Uses only prior events."""
    n = len(df)
    if n == 0:
        return df.copy()
    t_all = df[spec.order].to_numpy(dtype=float)
    ent_codes, _ = pd.factorize(df[spec.entity], sort=False)
    file_pos = np.arange(n)
    amt_all = df[spec.amount].to_numpy(dtype=float)
    theta_all = time_of_day_angle(df[spec.event_time])
    dist_home_all = haversine_km(df[spec.lat], df[spec.lon], df[spec.merch_lat], df[spec.merch_lon])
    span = float(t_all.max() - t_all.min())
    BIG = span + spec.max_window_s + 2.0
    out: dict[str, np.ndarray] = {}

    def grouped(group_codes: np.ndarray):
        order = np.lexsort((file_pos, t_all, group_codes))     # group, then time, then file order (ties)
        tk = group_codes[order] * BIG + (t_all[order] - t_all.min())
        return order, tk

    # ---- entity-level windows, last-transaction, deviation, geo, periodic
    order, tk = grouped(ent_codes)
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)
    a = amt_all[order]
    g = ent_codes[order]
    ca, ca2 = _excl_cumsum_by_group(a, g), _excl_cumsum_by_group(a * a, g)     # entity-local prefix sums
    dh = dist_home_all[order]
    cdh = _excl_cumsum_by_group(dh, g)
    start = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    grp_start = np.repeat(start, np.diff(np.r_[start, n]))
    out["h_n_prior"] = (np.arange(n) - grp_start)[inv]
    for w in spec.windows_h:
        cnt, s = _window_sums(tk, w * HOUR, ca)
        out[f"h_cnt_{w}h"], out[f"h_amt_{w}h"] = cnt[inv], s[inv]
    # last transaction (same entity, strictly prior in the sequence)
    has_prev = np.arange(n) > grp_start
    prev = np.where(has_prev, np.arange(n) - 1, 0)
    t_s = t_all[order]
    gap = np.where(has_prev, t_s - t_s[prev], np.nan)
    prev_amt = np.where(has_prev, a[prev], np.nan)
    out["h_hours_since_last"] = (gap / HOUR)[inv]
    out["h_prev_amt"] = prev_amt[inv]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["h_amt_ratio_prev"] = np.where(prev_amt > 0, a / prev_amt, np.nan)[inv]
        # deviation vs the entity's window
        W = spec.deviation_window_h
        cnt, s1, s2, sd = _window_sums(tk, W * HOUR, ca, ca2, cdh)
        mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
        var = np.where(cnt > 1, (s2 - cnt * mean ** 2) / np.maximum(cnt - 1, 1), np.nan)
        std = np.sqrt(np.clip(var, 0, None))
        out[f"h_amt_over_mean_{W}h"] = np.where(mean > 0, a / mean, np.nan)[inv]
        out[f"h_amt_z_{W}h"] = np.where(std > 0, (a - mean) / std, np.nan)[inv]
        dmean = np.where(cnt > 0, sd / np.maximum(cnt, 1), np.nan)
        out[f"h_dist_home_mean_{W}h"] = dmean[inv]
        out[f"h_dist_home_ratio_{W}h"] = np.where(dmean > 0, dh / dmean, np.nan)[inv]
        # geo: distance from previous merchant location, implied speed
        mlat, mlon = df[spec.merch_lat].to_numpy(float)[order], df[spec.merch_lon].to_numpy(float)[order]
        dprev = np.where(has_prev, haversine_km(mlat[prev], mlon[prev], mlat, mlon), np.nan)
        out["h_dist_prev_km"] = dprev[inv]
        out["h_speed_kmh"] = np.where(has_prev & (gap >= spec.speed_min_gap_s), dprev / (gap / HOUR), np.nan)[inv]
        # periodic (von Mises) over prior events of the last d days
        th = theta_all[order]
        cc, cs = _excl_cumsum_by_group(np.cos(th), g), _excl_cumsum_by_group(np.sin(th), g)
        for d in spec.vm_windows_d:
            cnt, C, S = _window_sums(tk, d * DAY, cc, cs)
            valid = cnt >= spec.vm_min_prior
            R = np.where(valid, np.sqrt(C ** 2 + S ** 2) / np.maximum(cnt, 1), np.nan)
            mu = np.arctan2(S, C)
            q = vm_halfwidth(spec.vm_alpha, kappa_from_R(np.nan_to_num(R)))
            dist = circ_dist(th, mu)
            tag = int(spec.vm_alpha * 100)
            out[f"vm{d}_n"] = cnt[inv]
            out[f"vm{d}_R"] = R[inv]
            out[f"vm{d}_dist_h"] = np.where(valid, dist * 24 / (2 * np.pi), np.nan)[inv]
            out[f"vm{d}_in_ci{tag}"] = np.where(valid, (dist <= q).astype(float), np.nan)[inv]

    # ---- grouped windows (same category / same merchant)
    for c in spec.conds:
        cond_codes, _ = pd.factorize(df[c].astype("string").fillna("<NA>"), sort=False)
        cc_codes, _ = pd.factorize(ent_codes.astype(np.int64) * (cond_codes.max() + 1) + cond_codes, sort=False)
        order2, tk2 = grouped(cc_codes)
        inv2 = np.empty(n, dtype=int)
        inv2[order2] = np.arange(n)
        a_c = amt_all[order2]
        ca_c = _excl_cumsum_by_group(a_c, cc_codes[order2])                   # (entity, context)-local
        for w in spec.windows_h:
            cnt, s = _window_sums(tk2, w * HOUR, ca_c)
            out[f"h_{c}_cnt_{w}h"], out[f"h_{c}_amt_{w}h"] = cnt[inv2], s[inv2]
        W = spec.deviation_window_h
        cnt, s = _window_sums(tk2, W * HOUR, ca_c)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_c = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
            out[f"h_amt_over_{c}_mean_{W}h"] = np.where(mean_c > 0, a_c / mean_c, np.nan)[inv2]

    feats = pd.DataFrame({k: out[k] for k in spec.feature_names()}, index=df.index).astype("float32")
    return pd.concat([df, feats], axis=1)


# ------------------------------------------------------------------------------ serving-side state
class OutOfOrderEvent(ValueError):
    """An event arrived with an order key earlier than the entity's last processed event."""


class DuplicateEvent(ValueError):
    """An event with the entity's last processed row id was submitted again."""


STATE_FORMAT = "frauddet.entity_state.v2"
# buffered event layout: [t, amt, cond_0..cond_{nc-1}, mlat, mlon, theta, dist_home,
#                         T_amt, T_amt2, T_dh, T_cos, T_sin, Tc_0..Tc_{nc-1}]  (T_* = running totals BEFORE the event)


class EntityStateStore:
    """Per-entity state for serving. ``process(event)`` returns the history features for the event using
    only what was seen before, then records it. Snapshot/restore are plain JSON.

    Bit-parity with the batch reference (``compute_history``) is by construction: every windowed sum is
    the difference of running totals accumulated in the same sequential float64 order as ``np.cumsum``
    (per entity, and per entity × context), and the derived quantities use the batch formulas verbatim.

    Serving contract (see serving.ServingContract): per-entity non-decreasing order-key order
    (OutOfOrderEvent otherwise unless ``strict_order=False``); resubmitting the entity's last row id raises
    DuplicateEvent; features are computed BEFORE the event is recorded; an unseen entity gets cold-start
    values (counts 0, everything else NaN)."""

    def __init__(self, spec: HistorySpec = HistorySpec(), strict_order: bool = True):
        self.spec = spec
        self.strict_order = strict_order
        self.events: dict[str, list[list[Any]]] = {}      # entity -> buffered rows (time-ordered, pruned)
        self.last: dict[str, list[Any]] = {}              # entity -> [last row, count, last row id]
        self.totals: dict[str, list[float]] = {}          # entity -> [T_amt, T_amt2, T_dh, T_cos, T_sin]
        self.ctotals: dict[str, list[dict[str, float]]] = {}   # entity -> per cond: value -> running amt total
        self.processed: int = 0
        self._cache: dict[str, tuple] = {}

    # -- event layout ---------------------------------------------------------------
    def _row(self, ev: dict[str, Any]) -> list[Any]:
        s = self.spec
        theta = time_of_day_angle(pd.Series([ev[s.event_time]]))[0]
        dist_home = float(haversine_km(ev[s.lat], ev[s.lon], ev[s.merch_lat], ev[s.merch_lon]))
        conds = [str(ev[c]) if pd.notna(ev[c]) else "<NA>" for c in s.conds]
        return [float(ev[s.order]), float(ev[s.amount]), *conds, float(ev[s.merch_lat]), float(ev[s.merch_lon]),
                float(theta), dist_home]

    def _arrays(self, key: str, buf: list[list[Any]]):
        c = self._cache.get(key)
        if c is not None and c[0] is buf and c[1] == len(buf):
            return c[2]
        nc = len(self.spec.conds)
        if buf:
            num = np.array([[e[0], e[1], e[2 + nc], e[3 + nc], e[4 + nc], e[5 + nc],
                             e[6 + nc], e[7 + nc], e[8 + nc], e[9 + nc], e[10 + nc], *e[11 + nc:11 + 2 * nc]] for e in buf],
                           dtype=float)
            conds = [np.array([e[2 + i] for e in buf], dtype=object) for i in range(nc)]
        else:
            num, conds = np.zeros((0, 11 + nc)), [np.zeros(0, dtype=object) for _ in range(nc)]
        arrs = (num, conds)
        self._cache[key] = (buf, len(buf), arrs)
        return arrs

    # -- scoring --------------------------------------------------------------------
    def process(self, ev: dict[str, Any], row_id: Any = None) -> dict[str, float]:
        s = self.spec
        key = str(ev[s.entity])
        row = self._row(ev)
        t, amt = row[0], row[1]
        prev = self.last.get(key)
        if prev is not None:
            if row_id is not None and len(prev) > 2 and prev[2] is not None and str(row_id) == str(prev[2]):
                raise DuplicateEvent(f"entity {key}: row id {row_id!r} already processed")
            if self.strict_order and t < prev[0][0]:
                raise OutOfOrderEvent(f"entity {key}: event at {t} precedes last processed event at {prev[0][0]}")
        nc = len(s.conds)
        buf = self.events.get(key, [])
        cutoff = t - s.max_window_s
        if buf and buf[0][0] <= cutoff:                     # prune events outside every window (buffer is time-ordered)
            k = 0
            while k < len(buf) and buf[k][0] <= cutoff:
                k += 1
            buf = buf[k:]
        self.events[key] = buf
        tot = self.totals.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0])
        ctot = self.ctotals.setdefault(key, [dict() for _ in range(nc)])
        num, conds = self._arrays(key, buf)
        tcol = num[:, 0]
        n = len(buf)
        T_AMT, T_AMT2, T_DH, T_COS, T_SIN = 6, 7, 8, 9, 10          # numeric-array columns (see _arrays)

        def lo_of(w_seconds: float) -> int:                  # first buffered event with t > t - w (batch: searchsorted right)
            return int(np.searchsorted(tcol, t - w_seconds, side="right"))

        def wsum(lo: int, col: int, total: float) -> float:  # batch: c[i] - c[lo]
            return total - (num[lo, col] if lo < n else total)

        f: dict[str, float] = {"h_n_prior": float(prev[1]) if prev is not None else 0.0}
        for w in s.windows_h:
            lo = lo_of(w * HOUR)
            f[f"h_cnt_{w}h"] = float(n - lo)
            f[f"h_amt_{w}h"] = wsum(lo, T_AMT, tot[0])
        for ci, c in enumerate(s.conds):
            v = row[2 + ci]
            same = conds[ci] == v
            ctv = ctot[ci].get(v, 0.0)
            for w in s.windows_h:
                lo = lo_of(w * HOUR)
                idx = np.flatnonzero(same[lo:])
                if len(idx):
                    j = lo + int(idx[0])
                    f[f"h_{c}_cnt_{w}h"] = float(len(idx))
                    f[f"h_{c}_amt_{w}h"] = ctv - num[j, 11 + ci]
                else:
                    f[f"h_{c}_cnt_{w}h"], f[f"h_{c}_amt_{w}h"] = 0.0, 0.0
        if prev is None:
            f.update(h_hours_since_last=np.nan, h_prev_amt=np.nan, h_amt_ratio_prev=np.nan, h_dist_prev_km=np.nan,
                     h_speed_kmh=np.nan)
        else:
            le = prev[0]
            gap = t - le[0]
            f["h_hours_since_last"] = gap / HOUR
            f["h_prev_amt"] = le[1]
            f["h_amt_ratio_prev"] = amt / le[1] if le[1] > 0 else np.nan
            dprev = float(haversine_km(le[2 + nc], le[3 + nc], row[2 + nc], row[3 + nc]))
            f["h_dist_prev_km"] = dprev
            f["h_speed_kmh"] = dprev / (gap / HOUR) if gap >= s.speed_min_gap_s else np.nan
        W = s.deviation_window_h
        lo = lo_of(W * HOUR)
        cnt = n - lo
        s1, s2, sd = wsum(lo, T_AMT, tot[0]), wsum(lo, T_AMT2, tot[1]), wsum(lo, T_DH, tot[2])
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = s1 / max(cnt, 1) if cnt > 0 else np.nan
            var = (s2 - cnt * (mean * mean)) / max(cnt - 1, 1) if cnt > 1 else np.nan   # numpy `**2` == x*x
            std = float(np.sqrt(np.clip(var, 0, None))) if cnt > 1 else np.nan
            f[f"h_amt_over_mean_{W}h"] = amt / mean if cnt > 0 and mean > 0 else np.nan
            f[f"h_amt_z_{W}h"] = (amt - mean) / std if cnt > 1 and std > 0 else np.nan
            for ci, c in enumerate(s.conds):
                v = row[2 + ci]
                idx = np.flatnonzero((conds[ci] == v)[lo:])
                if len(idx):
                    j = lo + int(idx[0])
                    cc = len(idx)
                    sc = ctot[ci].get(v, 0.0) - num[j, 11 + ci]
                    mean_c = sc / max(cc, 1)
                    f[f"h_amt_over_{c}_mean_{W}h"] = amt / mean_c if mean_c > 0 else np.nan
                else:
                    f[f"h_amt_over_{c}_mean_{W}h"] = np.nan
            dmean = sd / max(cnt, 1) if cnt > 0 else np.nan
            f[f"h_dist_home_mean_{W}h"] = dmean
            f[f"h_dist_home_ratio_{W}h"] = row[5 + nc] / dmean if cnt > 0 and dmean > 0 else np.nan
            tag = int(s.vm_alpha * 100)
            for d in s.vm_windows_d:
                lo = lo_of(d * DAY)
                k = n - lo
                f[f"vm{d}_n"] = float(k)
                if k >= s.vm_min_prior:
                    C, S = wsum(lo, T_COS, tot[3]), wsum(lo, T_SIN, tot[4])
                    R = float(np.sqrt(C * C + S * S) / max(k, 1))
                    mu = float(np.arctan2(S, C))
                    q = float(vm_halfwidth(s.vm_alpha, kappa_from_R([R]))[0])
                    dist = float(circ_dist(row[4 + nc], mu))
                    f[f"vm{d}_R"], f[f"vm{d}_dist_h"], f[f"vm{d}_in_ci{tag}"] = R, dist * 24 / (2 * np.pi), float(dist <= q)
                else:
                    f[f"vm{d}_R"] = f[f"vm{d}_dist_h"] = f[f"vm{d}_in_ci{tag}"] = np.nan
        # record AFTER scoring: totals before this event travel with it (point-in-time)
        stored = [*row, tot[0], tot[1], tot[2], tot[3], tot[4], *[ctot[ci].get(row[2 + ci], 0.0) for ci in range(nc)]]
        buf.append(stored)
        self._cache.pop(key, None)
        tot[0] += amt; tot[1] += amt * amt; tot[2] += row[5 + nc]
        tot[3] += float(np.cos(row[4 + nc])); tot[4] += float(np.sin(row[4 + nc]))
        for ci in range(nc):
            ctot[ci][row[2 + ci]] = ctot[ci].get(row[2 + ci], 0.0) + amt
        self.last[key] = [row, f["h_n_prior"] + 1, None if row_id is None else str(row_id)]
        self.processed += 1
        return f

    def replay(self, df: pd.DataFrame, row_id_col: str | None = None) -> pd.DataFrame:
        """Warm-up / streaming simulation: process a chronological frame event by event; returns features."""
        d = df.sort_values([self.spec.order], kind="stable")
        ids = d[row_id_col].to_numpy() if row_id_col else [None] * len(d)
        rows = [self.process(r, rid) for r, rid in zip(d.to_dict("records"), ids)]
        return pd.DataFrame(rows, index=d.index)[self.spec.feature_names()].astype("float32")

    # -- io -------------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {"format": STATE_FORMAT, "spec": asdict(self.spec), "strict_order": self.strict_order,
                "processed": self.processed, "events": self.events, "totals": self.totals, "ctotals": self.ctotals,
                "last": {k: [v[0], v[1], v[2] if len(v) > 2 else None] for k, v in self.last.items()}}

    @classmethod
    def restore(cls, d: dict[str, Any]) -> "EntityStateStore":
        if d.get("format") != STATE_FORMAT:
            raise ValueError(f"unsupported entity-state format {d.get('format')!r}; expected {STATE_FORMAT}")
        spec = HistorySpec(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d["spec"].items()})
        st = cls(spec, strict_order=d.get("strict_order", True))
        st.events = {k: [list(e) for e in v] for k, v in d["events"].items()}
        st.totals = {k: list(map(float, v)) for k, v in d["totals"].items()}
        st.ctotals = {k: [dict(x) for x in v] for k, v in d["ctotals"].items()}
        st.last = {k: [list(v[0]), float(v[1]), v[2] if len(v) > 2 else None] for k, v in d["last"].items()}
        st.processed = int(d.get("processed", 0))
        return st

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.snapshot()))

    @classmethod
    def load(cls, path: str | Path) -> "EntityStateStore":
        return cls.restore(json.loads(Path(path).read_text()))


def snapshot_from_frame(df: pd.DataFrame, spec: HistorySpec = HistorySpec(),
                        row_id_col: str | None = None) -> EntityStateStore:
    """The serving state a sequential replay of ``df`` would leave behind, built with the batch's own
    per-group sequential cumulative sums (so a later online continuation is bit-identical to the batch)."""
    st = EntityStateStore(spec)
    n = len(df)
    t_all = df[spec.order].to_numpy(float)
    order = np.lexsort((np.arange(n), t_all, pd.factorize(df[spec.entity], sort=False)[0]))
    d = df.iloc[order]
    t = d[spec.order].to_numpy(float)
    amt = d[spec.amount].to_numpy(float)
    theta = time_of_day_angle(d[spec.event_time])
    dh = haversine_km(d[spec.lat], d[spec.lon], d[spec.merch_lat], d[spec.merch_lon])
    conds = [d[c].astype("string").fillna("<NA>").to_numpy().astype(object) for c in spec.conds]
    ml, mo = d[spec.merch_lat].to_numpy(float), d[spec.merch_lon].to_numpy(float)
    ent = d[spec.entity].astype(str).to_numpy()
    ids = d[row_id_col].to_numpy() if row_id_col and row_id_col in d.columns else None
    t_end = float(t.max())
    nc = len(spec.conds)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # per-entity exclusive cumulative sums, sequential like np.cumsum (identical to compute_history)
    starts = np.r_[0, np.flatnonzero(ent[1:] != ent[:-1]) + 1, n]
    ex = {name: np.zeros(n) for name in ("amt", "amt2", "dh", "cos", "sin")}
    for a, b in zip(starts[:-1], starts[1:]):
        for name, vals in (("amt", amt), ("amt2", amt * amt), ("dh", dh), ("cos", cos_t), ("sin", sin_t)):
            ex[name][a:b] = _excl_cumsum(vals[a:b])[:-1]
    # per (entity, cond) exclusive cumulative amt
    exc = [np.zeros(n) for _ in range(nc)]
    for ci in range(nc):
        groups = pd.Series(np.arange(n)).groupby([ent, conds[ci]], sort=False).indices
        for idx in groups.values():
            idx = np.sort(idx)
            exc[ci][idx] = _excl_cumsum(amt[idx])[:-1]
    for a, b in zip(starts[:-1], starts[1:]):
        key = ent[a]
        st.totals[key] = [float(ex["amt"][b - 1] + amt[b - 1]), float(ex["amt2"][b - 1] + amt[b - 1] ** 2),
                          float(ex["dh"][b - 1] + dh[b - 1]), float(ex["cos"][b - 1] + cos_t[b - 1]),
                          float(ex["sin"][b - 1] + sin_t[b - 1])]
        ct: list[dict[str, float]] = [dict() for _ in range(nc)]
        for ci in range(nc):
            seg = conds[ci][a:b]
            for v in np.unique(seg):
                m = np.flatnonzero(seg == v)
                j = a + m[-1]
                ct[ci][str(v)] = float(exc[ci][j] + amt[j])
        st.ctotals[key] = ct
        rows = []
        for i in range(a, b):
            if t[i] > t_end - spec.max_window_s:
                rows.append([float(t[i]), float(amt[i]), *[str(c[i]) for c in conds], float(ml[i]), float(mo[i]),
                             float(theta[i]), float(dh[i]), float(ex["amt"][i]), float(ex["amt2"][i]), float(ex["dh"][i]),
                             float(ex["cos"][i]), float(ex["sin"][i]), *[float(exc[ci][i]) for ci in range(nc)]])
        if rows:
            st.events[key] = rows
        j = b - 1
        last_row = [float(t[j]), float(amt[j]), *[str(c[j]) for c in conds], float(ml[j]), float(mo[j]), float(theta[j]), float(dh[j])]
        st.last[key] = [last_row, float(b - a), None if ids is None else str(ids[j])]
    st.processed = n
    return st
