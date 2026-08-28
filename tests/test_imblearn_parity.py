"""Behavioural parity of the custom SMOTE / ENN implementations against imbalanced-learn.
Skipped when imbalanced-learn is not installed (it is in the Nix dev shell)."""

import numpy as np
import pandas as pd
import pytest

imblearn = pytest.importorskip("imblearn")
from imblearn.combine import SMOTEENN                       # noqa: E402
from imblearn.over_sampling import SMOTE                     # noqa: E402
from imblearn.under_sampling import EditedNearestNeighbours  # noqa: E402

from frauddet.imbalance import ImbalanceSpec, resample_training_fold  # noqa: E402


def _data(n=1500, p=0.08, seed=3):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < p).astype(np.int8)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    X.loc[y == 1, ["a", "b"]] += 1.5
    return X, y


def _rows(arr):
    return {tuple(np.round(r, 9)) for r in np.asarray(arr, float)}


@pytest.mark.parametrize("kind_sel", ["all", "mode"])
def test_enn_retained_rows_match_imblearn(kind_sel):
    X, y = _data()
    ours, yo, meta = resample_training_fold(X, y, ImbalanceSpec("enn", enn_neighbors=3, enn_kind_sel=kind_sel))
    ref = EditedNearestNeighbours(n_neighbors=3, kind_sel=kind_sel)      # sampling_strategy='auto' = majority class
    Xr, yr = ref.fit_resample(X.to_numpy(float), y)
    assert _rows(ours) == _rows(Xr) and int(yo.sum()) == int(yr.sum()) and len(yo) == len(yr)
    assert meta["removed_by_enn"] == len(y) - len(yr)


def test_enn_all_classes_matches_imblearn():
    X, y = _data()
    ours, yo, _ = resample_training_fold(X, y, ImbalanceSpec("enn", enn_classes="all", enn_kind_sel="all"))
    Xr, yr = EditedNearestNeighbours(n_neighbors=3, kind_sel="all", sampling_strategy="all").fit_resample(X.to_numpy(float), y)
    assert _rows(ours) == _rows(Xr) and len(yo) == len(yr)


def _on_minority_segments(X_orig, y_orig, X_new, k=5, atol=1e-6):
    """Every synthetic point lies on a segment between a minority point and one of its k minority neighbours."""
    from sklearn.neighbors import NearestNeighbors
    pos = X_orig[y_orig == 1]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(pos)
    _, idx = nn.kneighbors(pos)
    # all (i, neighbour j) segments as arrays: a = pos[i], ab = pos[j] - pos[i]
    I = np.repeat(np.arange(len(pos)), k)
    J = idx[:, 1:].reshape(-1)
    A, AB = pos[I], pos[J] - pos[I]
    ab2 = np.maximum((AB * AB).sum(axis=1), 1e-12)
    ok = 0
    for s in X_new:
        t = ((s - A) * AB).sum(axis=1) / ab2
        proj = A + t[:, None] * AB
        on = (t >= -atol) & (t <= 1 + atol) & (np.linalg.norm(proj - s, axis=1) < 1e-5)
        ok += bool(on.any())
    return ok / max(len(X_new), 1)


def test_smote_counts_and_segment_property_match_imblearn():
    X, y = _data()
    ratio = 0.5
    ours, yo, meta = resample_training_fold(X, y, ImbalanceSpec("smote", sampling_ratio=ratio, k_neighbors=5))
    Xr, yr = SMOTE(k_neighbors=5, sampling_strategy=ratio, random_state=0).fit_resample(X.to_numpy(float), y)
    assert len(yo) == len(yr) and int(yo.sum()) == int(yr.sum())                     # identical counts
    assert (yo[: len(y)] == y).all() and np.allclose(ours.to_numpy(float)[: len(y)], X.to_numpy(float))
    ours_new = ours.to_numpy(float)[len(y):]
    ref_new = Xr[len(y):]
    assert _on_minority_segments(X.to_numpy(float), y, ours_new) > 0.99
    assert _on_minority_segments(X.to_numpy(float), y, ref_new) > 0.99                # same geometric semantics


def test_smote_enn_size_is_comparable_to_imblearn():
    X, y = _data()
    ours, yo, meta = resample_training_fold(X, y, ImbalanceSpec("smote_enn", sampling_ratio=1.0))
    Xr, yr = SMOTEENN(random_state=0, smote=SMOTE(k_neighbors=5, random_state=0)).fit_resample(X.to_numpy(float), y)
    assert abs(len(yo) - len(yr)) / len(yr) < 0.05 and abs(int(yo.sum()) - int(yr.sum())) / int(yr.sum()) < 0.05
    assert meta["synthetic_added"] > 0 and meta["removed_by_enn"] > 0
