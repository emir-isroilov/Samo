# -*- coding: utf-8 -*-
"""mpm.explain testlari: SHAP shakllari (BUG-01), tree_model orqali SHAP (BUG-03/12), MDI,
permutation importance (tabular va patch), summarize."""
from __future__ import annotations

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

from mpm import explain as E
from mpm.base import ModelWrapper
from mpm.common import CancelledError, CancelToken
from mpm.config import default_hyperparams
from mpm.models import make_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = {"RandomForest": {"n_estimators": 30}, "SVM": {}, "XGBoost": {"n_estimators": 25}}
P = 6
NAMES = [f"f{i}" for i in range(P)]


def hp(name):
    d = dict(default_hyperparams()[name])
    d.update(FAST[name])
    return d


def make_xy(n=100, n_pos=20, p=P, seed=0):
    """0-feature'da kuchli signal; musbatlar BIRINCHI (loyiha konvensiyasi)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = np.r_[np.ones(n_pos), np.zeros(n - n_pos)].astype(np.int8)
    X[:n_pos, 0] += 2.5
    return X, y


@pytest.fixture(scope="module")
def xy():
    return make_xy()


@pytest.fixture(scope="module")
def zoo(xy):
    """{(model, calibrate): o'qitilgan wrapper}"""
    X, y = xy
    out = {}
    for name in ("RandomForest", "XGBoost", "SVM"):
        for cal in (True, False):
            out[(name, cal)] = make_model(name, hp(name), calibrate=cal, seed=5).fit(X, y)
    return out


# ---------------------------------------------------------------------------
# normalize_shap_values (BUG-01)
# ---------------------------------------------------------------------------
def test_normalize_all_layouts():
    rng = np.random.default_rng(0)
    c0, c1 = rng.normal(size=(7, 4)), rng.normal(size=(7, 4))
    nv = E.normalize_shap_values
    assert np.array_equal(nv([c0, c1]), c1)                                  # eski shap: sinflar ro'yxati
    assert np.array_equal(nv((c0, c1), class_index=0), c0)
    assert np.array_equal(nv([c0]), c0)
    assert np.array_equal(nv(c1), c1)                                        # (n, p): XGBoost
    assert np.array_equal(nv(np.stack([c0, c1], axis=-1)), c1)               # (n, p, 2): yangi shap RF
    assert np.array_equal(nv(np.stack([c0, c1], axis=-1), class_index=0), c0)
    assert np.array_equal(nv(np.stack([c0, c1], axis=0)), c1)                # (2, n, p)
    assert np.array_equal(nv(c1[:, :, None]), c1)                            # (n, p, 1)
    assert np.array_equal(nv(c1[None]), c1)                                  # (1, n, p)
    assert np.array_equal(nv(c1[:, :, None], class_index=1), c1)             # bitta chiqish: indeks e'tiborsiz
    assert np.array_equal(nv(c1.tolist()), c1)                               # oddiy ichma-ich ro'yxat
    out = nv(np.stack([c0, c1], axis=-1).astype(np.float32))
    assert out.shape == (7, 4) and out.dtype == np.float64 and out.flags["C_CONTIGUOUS"]


def test_normalize_explanation_object():
    class Expl:
        values = np.arange(24, dtype=float).reshape(4, 3, 2)

    out = E.normalize_shap_values(Expl())
    assert np.array_equal(out, Expl.values[:, :, 1])


def test_normalize_ambiguous_with_hints():
    """p=2 bo'lganda (C, n, p) va (n, p, C) farqlanmaydi - n_samples/n_features aniq hal qiladi."""
    rng = np.random.default_rng(1)
    c0, c1 = rng.normal(size=(7, 2)), rng.normal(size=(7, 2))
    cnp = np.stack([c0, c1], axis=0)                                         # (2, 7, 2)
    assert np.array_equal(E.normalize_shap_values(cnp, n_samples=7, n_features=2), c1)
    npc = np.stack([c0, c1], axis=-1)                                        # (7, 2, 2)
    assert np.array_equal(E.normalize_shap_values(npc, n_samples=7, n_features=2), c1)


@pytest.mark.parametrize("bad", [
    np.zeros(5), np.float64(1.0), np.zeros((2, 3, 4, 5)), np.zeros((7, 4, 3)),      # 0/1/4 o'lcham, sinf o'qi yo'q
    [], [np.zeros((3, 2)), np.zeros((4, 2))], [np.zeros((3, 2, 2))], [["a", "b"], ["c", "d"]],
    object(), "salom", {}, {"values": 1},
])
def test_normalize_invalid_shapes(bad):
    with pytest.raises(ValueError):
        E.normalize_shap_values(bad)


def test_normalize_invalid_class_index_and_hints():
    a = np.zeros((7, 4, 2))
    with pytest.raises(ValueError, match="class_index"):
        E.normalize_shap_values(a, class_index=2)
    with pytest.raises(ValueError, match="class_index"):
        E.normalize_shap_values([np.zeros((3, 2)), np.zeros((3, 2))], class_index=5)
    with pytest.raises(ValueError, match="kutilgan"):
        E.normalize_shap_values(np.zeros((7, 4)), n_samples=7, n_features=5)
    with pytest.raises(ValueError):
        E.normalize_shap_values(np.zeros((7, 4, 2)), n_samples=3, n_features=4)


# ---------------------------------------------------------------------------
# compute_shap_summary: haqiqiy shap (RF + XGBoost, kalibrlangan va kalibrlanmagan)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("calibrate", [True, False])
def test_shap_rf_xgb_real(calibrate, xy, zoo):
    pytest.importorskip("shap")
    X, y = xy
    fm = {"RandomForest": [zoo[("RandomForest", calibrate)]], "XGBoost": [zoo[("XGBoost", calibrate)]],
          "SVM": [zoo[("SVM", calibrate)]]}
    logs = []
    res = E.compute_shap_summary(fm, X, NAMES, max_background=30, seed=3, log_fn=logs.append)
    assert set(res) == {"RandomForest", "XGBoost"}                           # SVM tashlanadi
    for name, r in res.items():
        sv = r["shap_values"]
        assert sv.shape == (30, P) and sv.ndim == 2 and sv.dtype == np.float64   # 3D qolib ketmagan
        assert r["mean_abs_shap"].shape == (P,)
        assert np.allclose(r["mean_abs_shap"], np.abs(sv).mean(axis=0))
        assert r["X_background"].shape == (30, P) and r["feature_names"] == NAMES
        assert int(np.argmax(r["mean_abs_shap"])) == 0                       # signal feature
        assert np.all(np.isfinite(sv)) and r["units"] in ("ehtimollik", "log-odds")
        # SHAP aynan o'qitilgan bazaviy daraxtni tushuntiradi: yig'indi + kutilgan qiymat = xom chiqish
        tree = fm[name][0].tree_model()
        xb = np.asarray(r["X_background"], dtype=np.float32)
        ref = tree.predict_proba(xb)[:, 1] if name == "RandomForest" else tree.predict(xb, output_margin=True)
        assert np.allclose(sv.sum(axis=1) + r["expected_value"], ref, atol=1e-4)
    assert sum("SHAP hisoblandi" in m for m in logs) == 2


def test_shap_background_random_seeded_and_capped(xy, zoo):
    pytest.importorskip("shap")
    X, y = xy
    fm = {"RandomForest": [zoo[("RandomForest", True)]]}
    a = E.compute_shap_summary(fm, X, NAMES, max_background=25, seed=1)["RandomForest"]
    b = E.compute_shap_summary(fm, X, NAMES, max_background=25, seed=1)["RandomForest"]
    c = E.compute_shap_summary(fm, X, NAMES, max_background=25, seed=2)["RandomForest"]
    assert np.array_equal(a["X_background"], b["X_background"]) and np.array_equal(a["shap_values"], b["shap_values"])
    assert not np.array_equal(a["X_background"], c["X_background"])
    assert len(np.unique(a["X_background"], axis=0)) == 25                    # almashtirishsiz
    big = E.compute_shap_summary(fm, X, NAMES, max_background=10_000)["RandomForest"]
    assert big["shap_values"].shape == (len(X), P)
    one = E.compute_shap_summary(fm, X, NAMES, max_background=1)["RandomForest"]
    assert one["shap_values"].shape == (1, P) and one["mean_abs_shap"].shape == (P,)


def test_shap_uses_first_draw_model(xy):
    """Ro'yxatdagi birinchi model ishlatiladi (qolganlari emas)."""
    pytest.importorskip("shap")
    X, y = xy
    m1 = make_model("RandomForest", hp("RandomForest"), seed=1).fit(X, y)
    m2 = make_model("RandomForest", hp("RandomForest"), seed=2).fit(X, y)
    r12 = E.compute_shap_summary({"RandomForest": [m1, m2]}, X, NAMES, max_background=20)["RandomForest"]
    r1 = E.compute_shap_summary({"RandomForest": [m1]}, X, NAMES, max_background=20)["RandomForest"]
    r2 = E.compute_shap_summary({"RandomForest": [m2]}, X, NAMES, max_background=20)["RandomForest"]
    assert np.array_equal(r12["shap_values"], r1["shap_values"])
    assert not np.array_equal(r12["shap_values"], r2["shap_values"])


def test_shap_none_cases(monkeypatch, xy, zoo):
    X, y = xy
    fm = {"RandomForest": [zoo[("RandomForest", True)]]}
    logs = []
    monkeypatch.setattr(E, "get_shap", lambda: None)                         # shap yo'q
    assert E.compute_shap_summary(fm, X, NAMES, log_fn=logs.append) is None
    assert any("shap" in m.lower() for m in logs)
    monkeypatch.undo()
    pytest.importorskip("shap")
    logs.clear()
    assert E.compute_shap_summary({"SVM": [zoo[("SVM", True)]]}, X, NAMES, log_fn=logs.append) is None
    assert logs
    assert E.compute_shap_summary({}, X, NAMES) is None
    assert E.compute_shap_summary(None, X, NAMES) is None
    assert E.compute_shap_summary({"RandomForest": []}, X, NAMES) is None
    assert E.compute_shap_summary(fm, X[:0], NAMES) is None


def test_shap_invalid_inputs(xy, zoo):
    pytest.importorskip("shap")
    X, y = xy
    fm = {"RandomForest": [zoo[("RandomForest", True)]]}
    with pytest.raises(ValueError, match="2 o'lchamli"):
        E.compute_shap_summary(fm, X[:, 0], NAMES)
    with pytest.raises(ValueError, match="feature_names"):
        E.compute_shap_summary(fm, X, NAMES[:-1])


class _FakeShap:
    """shap'ning turli versiyalarini taqlid qiladi (haqiqiy shap talab qilinmaydi)."""

    def __init__(self, mode):
        self.mode = mode
        outer = self

        class TreeExplainer:
            expected_value = np.array([0.4, 0.6])

            def __init__(self, model, **kw):
                if outer.mode == "old_api" and "model_output" in kw:
                    raise TypeError("model_output qo'llab-quvvatlanmaydi")
                self.n_p = model.n_features_in_

            def shap_values(self, X, **kw):
                n = len(X)
                if outer.mode == "additivity" and kw.get("check_additivity", True):
                    raise RuntimeError("Additivity check failed in TreeExplainer")
                if outer.mode == "boom":
                    raise RuntimeError("portladi")
                if outer.mode == "bad_shape":
                    return np.ones((n + 1, self.n_p))
                if outer.mode == "cube":
                    return np.stack([np.zeros((n, self.n_p)), np.ones((n, self.n_p))], axis=-1)
                if outer.mode == "nan_col":                 # bitta ustun butunlay NaN
                    a = np.ones((n, self.n_p))
                    a[:, 1] = np.nan
                    return [np.zeros((n, self.n_p)), a]
                return [np.zeros((n, self.n_p)), np.ones((n, self.n_p))]       # eski API: ro'yxat

        self.TreeExplainer = TreeExplainer


@pytest.mark.parametrize("mode", ["list", "old_api", "additivity", "cube"])
def test_shap_version_robustness(monkeypatch, mode, xy, zoo):
    X, y = xy
    monkeypatch.setattr(E, "get_shap", lambda: _FakeShap(mode))
    logs = []
    res = E.compute_shap_summary({"RandomForest": [zoo[("RandomForest", True)]]}, X, NAMES, max_background=12,
                                 log_fn=logs.append)["RandomForest"]
    assert res["shap_values"].shape == (12, P) and np.all(res["shap_values"] == 1.0)    # musbat sinf
    assert np.array_equal(res["mean_abs_shap"], np.ones(P)) and res["expected_value"] == 0.6
    if mode == "additivity":
        assert any("additivity" in m.lower() for m in logs)


def test_shap_all_nan_column_no_runtime_warning(monkeypatch, xy, zoo):
    """Regressiya (BUG-14): butunlay NaN SHAP ustuni np.nanmean'da "Mean of empty slice" RuntimeWarning berar,
    qat'iy ogohlantirish rejimida butun SHAP natijasi yo'qolar edi. Endi: ogohlantirishsiz, ustun NaN."""
    X, y = xy
    monkeypatch.setattr(E, "get_shap", lambda: _FakeShap("nan_col"))
    logs = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = E.compute_shap_summary({"RandomForest": [zoo[("RandomForest", True)]]}, X, NAMES, max_background=12,
                                     log_fn=logs.append)
    mean_abs = res["RandomForest"]["mean_abs_shap"]
    assert np.isnan(mean_abs[1]) and np.all(np.delete(mean_abs, 1) == 1.0)
    assert any("NaN" in m for m in logs)


@pytest.mark.parametrize("mode", ["boom", "bad_shape"])
def test_shap_failure_does_not_crash(monkeypatch, mode, xy, zoo):
    X, y = xy
    monkeypatch.setattr(E, "get_shap", lambda: _FakeShap(mode))
    logs = []
    assert E.compute_shap_summary({"RandomForest": [zoo[("RandomForest", True)]]}, X, NAMES,
                                  log_fn=logs.append) is None
    assert any("hisoblanmadi" in m for m in logs)


def test_shap_cancel(xy, zoo):
    pytest.importorskip("shap")
    X, y = xy
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledError):
        E.compute_shap_summary({"RandomForest": [zoo[("RandomForest", True)]]}, X, NAMES, cancel=tok)


# ---------------------------------------------------------------------------
# mdi_importance (BUG-03)
# ---------------------------------------------------------------------------
def test_mdi_importance(zoo):
    for name in ("RandomForest", "XGBoost"):
        for cal in (True, False):
            m = zoo[(name, cal)]
            imp = E.mdi_importance(m)
            assert imp.shape == (P,) and imp.dtype == np.float64
            assert np.array_equal(imp, m.tree_model().feature_importances_.astype(np.float64))
            assert imp.sum() > 0 and int(np.argmax(imp)) == 0                 # o'qitilgan model (nol emas)
    assert E.mdi_importance(zoo[("SVM", True)]) is None
    assert E.mdi_importance(None) is None


# ---------------------------------------------------------------------------
# permutation_importance_auc (tabular)
# ---------------------------------------------------------------------------
def test_perm_signal_feature_highest(zoo):
    Xv, yv = make_xy(n=120, n_pos=30, seed=11)                                # model ko'rmagan ma'lumot
    for name in ("RandomForest", "XGBoost", "SVM"):
        imp = E.permutation_importance_auc(zoo[(name, True)], Xv, yv, n_repeats=5, seed=0)
        assert imp.shape == (P,) and imp.dtype == np.float64 and np.all(np.isfinite(imp))
        assert int(np.argmax(imp)) == 0 and imp[0] > 0.1
        assert np.all(imp[1:] < imp[0] / 2)


def test_perm_deterministic_and_inputs_untouched(zoo):
    Xv, yv = make_xy(n=80, n_pos=20, seed=4)
    X0, y0 = Xv.copy(), yv.copy()
    m = zoo[("RandomForest", True)]
    a = E.permutation_importance_auc(m, Xv, yv, n_repeats=4, seed=9)
    b = E.permutation_importance_auc(m, Xv, yv, n_repeats=4, seed=9)
    c = E.permutation_importance_auc(m, Xv, yv, n_repeats=4, seed=10)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.array_equal(Xv, X0) and np.array_equal(yv, y0)                  # kirish o'zgarmagan


def test_perm_chunking_equivalence(monkeypatch, zoo):
    """Takrorlarni bitta bashoratga jamlash natijani o'zgartirmaydi."""
    Xv, yv = make_xy(n=60, n_pos=15, seed=5)
    m = zoo[("RandomForest", False)]
    a = E.permutation_importance_auc(m, Xv, yv, n_repeats=6, seed=1)
    monkeypatch.setattr(E, "_PERM_MAX_ELEMS", 1)                              # har chaqiruvda 1 takror
    b = E.permutation_importance_auc(m, Xv, yv, n_repeats=6, seed=1)
    assert np.array_equal(a, b)


def test_perm_single_class_and_validation(zoo):
    Xv, yv = make_xy(n=40, n_pos=10, seed=6)
    m = zoo[("RandomForest", True)]
    for y1 in (np.zeros(40, dtype=int), np.ones(40, dtype=int)):
        out = E.permutation_importance_auc(m, Xv, y1)
        assert out.shape == (P,) and np.all(np.isnan(out))
    with pytest.raises(ValueError, match="n_repeats"):
        E.permutation_importance_auc(m, Xv, yv, n_repeats=0)
    with pytest.raises(ValueError, match="teng emas"):
        E.permutation_importance_auc(m, Xv[:-1], yv)
    with pytest.raises(ValueError, match="0/1"):
        E.permutation_importance_auc(m, Xv, yv + 2)
    with pytest.raises(ValueError, match="X"):
        E.permutation_importance_auc(m, None, yv)
    with pytest.raises(ValueError, match="batch_size"):
        E.permutation_importance_auc(m, Xv, yv, batch_size=0)
    assert E.permutation_importance_auc(m, Xv, yv, n_repeats=1).shape == (P,)


class _Spy(ModelWrapper):
    """predict_proba_pos chaqiruvlarini yozib boruvchi soxta model."""

    def __init__(self, input_kind="tabular", fn=None):
        super().__init__("Spy", {})
        self.input_kind = input_kind
        self.fn = fn
        self.calls = []
        self.is_fitted = True

    def fit(self, X, y, patches=None, cancel=None, log_fn=None):
        return self

    def predict_proba_pos(self, X=None, patches=None, batch_size=8192):
        self.calls.append({"X": None if X is None else np.array(X), "patches": None if patches is None
                           else np.array(patches), "batch_size": batch_size})
        return np.asarray(self.fn(X, patches), dtype=np.float64)


def test_perm_efficient_calls_and_batch_size():
    Xv, yv = make_xy(n=50, n_pos=10, seed=7)
    spy = _Spy(fn=lambda X, patches: X[:, 0])
    imp = E.permutation_importance_auc(spy, Xv, yv, n_repeats=5, seed=1, batch_size=1234)
    assert len(spy.calls) == 1 + P                                            # bazaviy + har feature uchun bitta
    assert all(c["batch_size"] == 1234 for c in spy.calls)
    assert imp[0] > 0.3 and np.all(imp[1:] == 0.0)                            # faqat 0-feature'ga bog'liq
    assert all(c["patches"] is None for c in spy.calls)


def test_perm_nonfinite_prediction_gives_nan():
    Xv, yv = make_xy(n=30, n_pos=8, seed=8)
    spy = _Spy(fn=lambda X, patches: np.where(np.arange(len(X)) == 3, np.nan, X[:, 0]))
    assert np.all(np.isnan(E.permutation_importance_auc(spy, Xv, yv)))


def test_perm_cancel_and_progress(zoo):
    Xv, yv = make_xy(n=40, n_pos=10, seed=9)
    m = zoo[("RandomForest", True)]
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledError):
        E.permutation_importance_auc(m, Xv, yv, cancel=tok)
    fr = []
    E.permutation_importance_auc(m, Xv, yv, n_repeats=2, progress_fn=lambda f, msg="": fr.append(f))
    assert len(fr) == P and fr[-1] == 1.0 and fr == sorted(fr)


# ---------------------------------------------------------------------------
# permutation_importance_auc (patch modeli: kanal butun patchlar bo'yicha aralashtiriladi)
# ---------------------------------------------------------------------------
def make_patches(n=40, n_pos=10, w=5, p=3, seed=0):
    rng = np.random.default_rng(seed)
    patches = rng.normal(size=(n, w, w, p)).astype(np.float32)
    y = np.r_[np.ones(n_pos), np.zeros(n - n_pos)].astype(np.int8)
    patches[:n_pos, w // 2 - 1:w // 2 + 2, w // 2 - 1:w // 2 + 2, 0] += 2.0
    return patches, y


def test_perm_patch_channel_shuffled_whole_patches():
    patches, y = make_patches(n=24, n_pos=8, w=5, p=3, seed=1)
    orig = patches.copy()
    spy = _Spy("patch", fn=lambda X, pt: pt[:, 2, 2, 0])
    imp = E.permutation_importance_auc(spy, None, y, patches=patches, n_repeats=3, seed=4, batch_size=64)
    assert imp.shape == (3,) and imp[0] > 0.3 and np.all(imp[1:] == 0.0)
    assert np.array_equal(patches, orig)                                      # kirish o'zgarmagan
    assert all(c["X"] is None and c["batch_size"] == 64 for c in spy.calls)
    n = len(y)
    assert len(spy.calls) == 1 + 3                                            # takrorlar bitta chaqiruvga jamlangan
    for j in range(3):
        call = spy.calls[1 + j]["patches"]
        assert call.shape == (3 * n,) + orig.shape[1:]
        for t in range(3):
            blk = call[t * n:(t + 1) * n]
            other = [c for c in range(3) if c != j]
            assert np.array_equal(blk[..., other], orig[..., other])          # boshqa kanallar o'zgarmagan
            # j-kanal: asl patchlarning (butun w x w) permutatsiyasi - bir xil permutatsiya barcha pikselda
            src = [int(np.flatnonzero((orig[..., j] == blk[i, ..., j]).all(axis=(1, 2)))[0]) for i in range(n)]
            assert sorted(src) == list(range(n))
            assert np.array_equal(blk[..., j], orig[src, ..., j])


def test_perm_patch_requirements():
    patches, y = make_patches()
    spy = _Spy("patch", fn=lambda X, pt: pt[:, 2, 2, 0])
    with pytest.raises(ValueError, match="patches"):
        E.permutation_importance_auc(spy, np.zeros((len(y), 3)), y)
    with pytest.raises(ValueError, match="4 o'lchamli"):
        E.permutation_importance_auc(spy, None, y, patches=patches[:, :, :, 0])
    out = E.permutation_importance_auc(spy, None, np.zeros(len(y), dtype=int), patches=patches)
    assert out.shape == (3,) and np.all(np.isnan(out))
    # tabular model patches ni e'tiborsiz qoldiradi
    tab = _Spy("tabular", fn=lambda X, pt: X[:, 0])
    Xv, yv = make_xy(n=30, n_pos=8)
    E.permutation_importance_auc(tab, Xv, yv, patches=np.zeros((30, 3, 3, 2)), n_repeats=1)
    assert all(c["patches"] is None for c in tab.calls)


@pytest.mark.slow
def test_perm_patch_real_cnn():
    pytest.importorskip("tensorflow")
    from mpm.cnn import CNNModel
    patches, y = make_patches(n=80, n_pos=20, w=5, p=3, seed=2)
    params = dict(default_hyperparams()["CNN"], window=5, filters1=8, filters2=0, dense_units=8, batch_size=16,
                  epochs=20, patience=20, learning_rate=0.01, val_fraction=0.2)
    m = CNNModel(params, seed=3).fit(None, y, patches=patches)
    pv, yv = make_patches(n=60, n_pos=15, w=5, p=3, seed=12)
    imp = E.permutation_importance_auc(m, None, yv, patches=pv, n_repeats=3, seed=0, batch_size=32)
    assert imp.shape == (3,) and int(np.argmax(imp)) == 0 and imp[0] > 0.1


# ---------------------------------------------------------------------------
# summarize_perm_importance
# ---------------------------------------------------------------------------
def test_summarize_basic():
    folds = [np.array([0.2, 0.0, 0.1]), np.array([0.4, 0.1, 0.1]), np.array([0.3, -0.1, 0.1])]
    s = E.summarize_perm_importance(folds)
    assert set(s) >= {"mean", "std", "per_fold", "n_folds"}
    assert s["n_folds"] == 3 and s["per_fold"].shape == (3, 3)
    assert np.allclose(s["mean"], [0.3, 0.0, 0.1]) and np.allclose(s["std"], np.std(np.array(folds), axis=0))
    assert np.array_equal(s["per_fold"], np.vstack(folds))


def test_summarize_nan_handling_no_warnings():
    folds = [np.array([0.2, np.nan, np.nan]), np.array([np.nan, np.nan, 0.3]), np.array([0.4, np.nan, 0.5]), None]
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                        # "Mean of empty slice" ham xato
        s = E.summarize_perm_importance(folds)
    assert s["n_folds"] == 3 and s["per_fold"].shape == (3, 3)
    assert np.allclose(s["mean"][[0, 2]], [0.3, 0.4]) and np.isnan(s["mean"][1]) and np.isnan(s["std"][1])
    assert s["n_valid_folds"] == 3
    allnan = E.summarize_perm_importance([np.full(3, np.nan), np.full(3, np.nan)])
    assert np.all(np.isnan(allnan["mean"])) and allnan["n_folds"] == 2 and allnan["n_valid_folds"] == 0


def test_summarize_edge_cases():
    s = E.summarize_perm_importance([])
    assert s["n_folds"] == 0 and s["mean"].shape == (0,) and s["per_fold"].shape[0] == 0
    assert E.summarize_perm_importance(None)["n_folds"] == 0
    one = E.summarize_perm_importance([np.array([0.5, 0.1])])
    assert one["n_folds"] == 1 and np.allclose(one["std"], 0.0)
    with pytest.raises(ValueError, match="uzunligi"):
        E.summarize_perm_importance([np.zeros(3), np.zeros(4)])
    lst = E.summarize_perm_importance([[0.1, 0.2], [0.3, 0.4]])               # ro'yxatlar ham qabul qilinadi
    assert np.allclose(lst["mean"], [0.2, 0.3])


# ---------------------------------------------------------------------------
# Import qoidalari
# ---------------------------------------------------------------------------
def test_heavy_libs_not_imported_at_module_level():
    code = ("import sys; import mpm.explain, mpm.tuning; "
            "bad = [m for m in ('tensorflow', 'shap', 'xgboost') if m in sys.modules]; assert not bad, bad")
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
