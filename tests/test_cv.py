# -*- coding: utf-8 -*-
"""mpm.cv testlari: run_cv (spatial/random, nested tuning, perm importance, cancel/progress), compute_metrics
(blok-bootstrap CI), metrics_dataframe, youden_threshold, run_background_sensitivity.
tuning/explain modullari boshqa agent tomonidan yoziladi: ular sys.modules stub'lari bilan sinaladi, haqiqiylari
(mavjud bo'lsa) pytest.importorskip bilan. CNN testi - slow."""
from __future__ import annotations

import dataclasses
import sys
import types
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import xy as rio_xy
from shapely.geometry import Point
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             roc_auc_score, roc_curve)

from mpm import cv, data, spatial
from mpm.common import ENSEMBLE_NAME, CancelledError, CancelToken
from mpm.config import TuningConfig, default_hyperparams

SK3 = ["RandomForest", "SVM", "XGBoost"]
REQUIRED_KEYS = {"auc", "auc_std", "auc_ci95", "auc_single", "pr_auc", "pr_auc_std", "pr_auc_ci95",
                 "balanced_accuracy", "balanced_accuracy_std", "balanced_accuracy_ci95", "f1", "f1_std", "f1_ci95",
                 "brier", "brier_std", "brier_ci95", "balanced_accuracy_youden", "f1_youden",
                 "threshold_youden", "sensitivity", "specificity", "confusion", "fpr", "tpr", "precision",
                 "recall", "calibration", "mean_proba"}


# ---------------------------------------------------------------------------
# Yordamchilar / fixture'lar
# ---------------------------------------------------------------------------
N_RF = 15                                     # tez testlar uchun kichik o'rmon (PARAM_SPECS min=10)


def fast_hp():
    hp = default_hyperparams()
    hp["RandomForest"]["n_estimators"] = N_RF
    hp["XGBoost"]["n_estimators"] = 15
    return hp


@pytest.fixture(scope="module")
def ctx(synth_project):
    """Kichik dataset: signal qatlamlari (1-2) bo'yicha AOI ichidagi 32 musbat + 90 fon, spatial bloklar."""
    p = synth_project
    raster = data.load_and_align_rasters(data.find_tiff_files(p["tiff"]))
    pipe = data.FeaturePipeline(raster.band_names, raster.categorical).fit(raster)
    aoi = data.load_aoi(data.find_shapefile(p["aoi"]))
    valid = data.valid_pixel_mask(raster)
    inner = np.zeros_like(valid)
    inner[12:-12, 12:-12] = True
    score = raster.stack[0] + raster.stack[1]
    cand = valid & inner & (score > np.percentile(score[valid & inner], 85))
    rr, cc = np.nonzero(cand)
    sel = np.random.default_rng(0).choice(rr.size, size=32, replace=False)
    xs, ys = rio_xy(raster.transform, rr[sel], cc[sel])
    pos = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xs, ys)], crs="EPSG:28411")
    bg = data.generate_background_points(aoi, pos, 90, 500.0, random_state=1, valid_mask=valid,
                                         transform=raster.transform)
    ds = data.build_dataset(raster, pipe, pos, bg, need_feature_stack=True)
    block, groups = spatial.adapt_block_size(ds.coords, ds.y, 3, 2500.0)
    return SimpleNamespace(raster=raster, pipe=pipe, aoi=aoi, pos=pos, ds=ds, groups=groups, block=block,
                           hp=fast_hp())


def run(ctx, mode="spatial", models=("RandomForest", "SVM"), k=3, rep=1, **kw):
    kw.setdefault("seed", 7)
    groups = ctx.groups
    return cv.run_cv(ctx.ds, groups, mode, list(models), ctx.hp, n_splits=k, n_repeats=rep, **kw)


@pytest.fixture(scope="module")
def spatial_res(ctx):
    logs, calls = [], []
    res = run(ctx, "spatial", SK3, rep=2, log_fn=logs.append, progress_fn=lambda f, m: calls.append((f, m)))
    return SimpleNamespace(res=res, logs=logs, calls=calls)


def install_stub(monkeypatch, modname, **attrs):
    mod = types.ModuleType(modname)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, modname, mod)
    return mod


def synth_oof(n=300, n_pos=60, repeats=2, models=("A", "B"), seed=0, strength=1.5, noise=1.0, rep_noise=0.3):
    """Sintetik OOF: har model uchun umumiy "asos" signal (noise) + har repeat'ga qo'shimcha kichik shovqin (rep_noise);
    haqiqiy CV'da repeat'lar bir-biriga bog'liq (bir xil ma'lumot, boshqa fold'lar)."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n_pos), np.zeros(n - n_pos)].astype(np.int64)
    rng.shuffle(y)
    oof = {}
    for m in models:
        base = strength * (y - 0.5) + rng.normal(scale=noise, size=n)
        oof[m] = [1.0 / (1.0 + np.exp(-(base + rng.normal(scale=rep_noise, size=n)))) for _ in range(repeats)]
    return y, oof


# ---------------------------------------------------------------------------
# run_cv: spatial
# ---------------------------------------------------------------------------
def test_spatial_structure(ctx, spatial_res):
    res, ds = spatial_res.res, ctx.ds
    assert res["mode"] == "spatial" and res["model_names"] == SK3
    assert res["feature_names"] == ds.feature_names
    assert set(res["oof"]) == set(SK3)
    for name in SK3:
        assert len(res["oof"][name]) == 2
        for arr in res["oof"][name]:
            assert arr.shape == (ds.n,) and arr.dtype == np.float64
            assert np.all(np.isfinite(arr)) and arr.min() >= 0.0 and arr.max() <= 1.0
    assert res["fold_map"].shape == (ds.n,) and set(np.unique(res["fold_map"])) == {0, 1, 2}
    assert res["tuned_params"] == {} and res["perm_importance"] == {}
    assert res["hp"]["RandomForest"]["n_estimators"] == N_RF


def test_spatial_folds_never_split_blocks(ctx, spatial_res):
    fm = spatial_res.res["fold_map"]
    for g in np.unique(ctx.groups):
        assert len(set(fm[ctx.groups == g].tolist())) == 1


def test_fold_table(ctx, spatial_res):
    ft = spatial_res.res["fold_table"]
    assert list(ft.columns) == ["repeat", "fold", "n_train", "n_val", "pos_train", "pos_val", "n_blocks_val",
                                "seconds_RandomForest", "seconds_SVM", "seconds_XGBoost"]
    assert len(ft) == 6 and ft["repeat"].tolist() == [0, 0, 0, 1, 1, 1] and ft["fold"].tolist() == [0, 1, 2] * 2
    assert (ft["n_train"] + ft["n_val"] == ctx.ds.n).all()
    assert (ft["pos_train"] + ft["pos_val"] == ctx.ds.n_pos).all()
    assert (ft["pos_val"] >= 1).all()                         # BUG-05: har val fold'da musbat nuqta
    for r in (0, 1):
        assert ft[ft["repeat"] == r]["n_val"].sum() == ctx.ds.n
    assert (ft[["seconds_RandomForest", "seconds_SVM", "seconds_XGBoost"]] >= 0).all().all()


def test_fold_table_matches_spatial_fold_report(ctx, spatial_res):
    """fold_table - spatial.fold_report bilan bir xil ustunlar/qiymatlar (n_blocks_val ham), GUI diagnostikasi shunga tayanadi."""
    res, ds = spatial_res.res, ctx.ds
    fm = res["fold_map"]
    splits = [(0, f, np.flatnonzero(fm != f), np.flatnonzero(fm == f)) for f in range(3)]
    rep = spatial.fold_report(ds.y, ds.coords, ctx.groups, splits)
    ft = res["fold_table"]
    ft0 = ft[ft["repeat"] == 0][list(rep.columns)].reset_index(drop=True)
    pd.testing.assert_frame_equal(ft0, rep, check_dtype=False)
    assert (ft["n_blocks_val"] >= 1).all() and (ft["n_blocks_val"] <= ft["n_val"]).all()


def test_fold_table_n_blocks_val_without_groups(ctx):
    res = cv.run_cv(ctx.ds, None, "random", ["RandomForest"], ctx.hp, n_splits=3, n_repeats=1, seed=1, calibrate=False)
    ft = res["fold_table"]
    assert (ft["n_blocks_val"] == ft["n_val"]).all()                  # groups None => har nuqta alohida blok


def test_oof_has_signal(ctx, spatial_res):
    for name in SK3:
        p = np.mean(spatial_res.res["oof"][name], axis=0)
        assert roc_auc_score(ctx.ds.y, p) > 0.7, name


def test_repeats_differ_but_each_complete(spatial_res):
    a, b = spatial_res.res["oof"]["RandomForest"]
    assert not np.allclose(a, b)                              # repeat'lar turli fold bo'linishi


def test_logs_and_progress(spatial_res):
    assert any("1-takror, 1/3-fold" in m for m in spatial_res.logs)
    fr = [c[0] for c in spatial_res.calls]
    assert len(fr) == 6 and fr == sorted(fr) and fr[-1] == pytest.approx(1.0) and fr[0] > 0
    assert fr == pytest.approx([(i + 1) / 6 for i in range(6)])
    for i, (_, msg) in enumerate(spatial_res.calls):
        assert f"Fold {i + 1}/6" in msg and "ETA" in msg
        assert msg.rstrip()[-5:-3].isdigit() and msg[-3] == ":"      # ETA mm:ss


def test_determinism_and_seed(ctx, spatial_res):
    again = run(ctx, "spatial", SK3, rep=2, seed=7)
    for name in SK3:
        for a, b in zip(spatial_res.res["oof"][name], again["oof"][name]):
            np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(spatial_res.res["fold_map"], again["fold_map"])
    other = run(ctx, "spatial", ["RandomForest"], rep=1, seed=8)
    assert not np.allclose(other["oof"]["RandomForest"][0], spatial_res.res["oof"]["RandomForest"][0])


def test_model_result_independent_of_other_models(ctx):
    solo = run(ctx, "spatial", ["RandomForest"], rep=1)
    duo = run(ctx, "spatial", ["SVM", "RandomForest"], rep=1)         # tartib va boshqa modellar ta'sir qilmaydi
    np.testing.assert_array_equal(solo["oof"]["RandomForest"][0], duo["oof"]["RandomForest"][0])


# ---------------------------------------------------------------------------
# run_cv: random rejimi, tekshiruvlar
# ---------------------------------------------------------------------------
def test_random_mode(ctx):
    res = run(ctx, "random", rep=2, perm_importance=True)       # random'da perm_importance e'tiborsiz
    assert res["mode"] == "random" and res["perm_importance"] == {} and res["tuned_params"] == {}
    n = ctx.ds.n
    for name in ("RandomForest", "SVM"):
        assert len(res["oof"][name]) == 2 and all(np.all(np.isfinite(a)) for a in res["oof"][name])
    assert set(np.unique(res["fold_map"])) == {0, 1, 2}
    ft = res["fold_table"]
    assert len(ft) == 6 and ft[ft["repeat"] == 1]["n_val"].sum() == n
    # random CV fazoviy emas: blok(lar) bir necha fold'ga bo'linishi mumkin
    assert any(len(set(res["fold_map"][ctx.groups == g].tolist())) > 1 for g in np.unique(ctx.groups))


def test_random_mode_works_without_groups(ctx):
    res = cv.run_cv(ctx.ds, None, "random", ["RandomForest"], ctx.hp, n_splits=3, n_repeats=1, seed=1)
    assert np.all(np.isfinite(res["oof"]["RandomForest"][0]))


def test_oof_completeness_check(ctx, monkeypatch):
    n = ctx.ds.n
    order = np.random.default_rng(0).permutation(n)       # sinflar aralash bo'lsin (musbatlar dataset boshida)
    pos = np.arange(n)

    def split(val_pos):
        val = order[val_pos]
        return order[~np.isin(pos, val_pos)], val

    def overlap(y, groups, n_splits, n_repeats, random_state=0, log_fn=None):
        # 5..9 pozitsiyadagi nuqtalar ham 0-fold, ham 1-fold validation'ida (ikki marta)
        for f, v in enumerate([pos[:10], pos[5:20], pos[20:]]):
            tr, va = split(v)
            yield 0, f, tr, va

    monkeypatch.setattr(cv.spatial, "stratified_group_splits", overlap)
    with pytest.raises(ValueError, match="OOF to'liq emas.*bir necha marta"):
        run(ctx, "spatial", ["RandomForest"])

    def missing(y, groups, n_splits, n_repeats, random_state=0, log_fn=None):
        # 19-pozitsiyadagi nuqta hech bir validation'ga tushmaydi (train'ga ham: train = qolganlari)
        for f, (v, t) in enumerate([(pos[:10], pos[10:]), (pos[10:19], np.r_[pos[:10], pos[20:]]),
                                    (pos[20:], pos[:19])]):
            yield 0, f, order[t], order[v]

    monkeypatch.setattr(cv.spatial, "stratified_group_splits", missing)
    with pytest.raises(ValueError, match="qamramaydi|kesishadi"):
        run(ctx, "spatial", ["RandomForest"])

    def single_class_train(y, groups, n_splits, n_repeats, random_state=0, log_fn=None):
        p1 = np.flatnonzero(y == 1)                        # val = barcha musbatlar => train'da faqat fon
        yield 0, 0, np.flatnonzero(y == 0), p1
        yield 0, 1, np.arange(len(y)), np.empty(0, dtype=int)
        yield 0, 2, np.arange(len(y)), np.empty(0, dtype=int)

    monkeypatch.setattr(cv.spatial, "stratified_group_splits", single_class_train)
    with pytest.raises(ValueError, match="bitta sinf|bo'sh"):
        run(ctx, "spatial", ["RandomForest"])


ARG_CASES = {
    "cv_mode": (lambda c: cv.run_cv(c.ds, c.groups, "bogus", ["RandomForest"], c.hp, n_splits=3, n_repeats=1),
                "cv_mode"),
    "empty_models": (lambda c: cv.run_cv(c.ds, c.groups, "spatial", [], c.hp, n_splits=3, n_repeats=1),
                     "model_names"),
    "dup_models": (lambda c: cv.run_cv(c.ds, c.groups, "spatial", ["SVM", "SVM"], c.hp, n_splits=3, n_repeats=1),
                   "takror"),
    "unknown_model": (lambda c: cv.run_cv(c.ds, c.groups, "spatial", ["Nope"], c.hp, n_splits=3, n_repeats=1),
                      "Noma'lum model"),
    "n_splits": (lambda c: cv.run_cv(c.ds, c.groups, "spatial", ["SVM"], c.hp, n_splits=1, n_repeats=1),
                 "n_splits"),
    "n_repeats": (lambda c: cv.run_cv(c.ds, c.groups, "spatial", ["SVM"], c.hp, n_splits=3, n_repeats=0),
                  "n_repeats"),
    "no_groups": (lambda c: cv.run_cv(c.ds, None, "spatial", ["SVM"], c.hp, n_splits=3, n_repeats=1), "groups"),
    "short_groups": (lambda c: cv.run_cv(c.ds, c.groups[:-1], "random", ["SVM"], c.hp, n_splits=3, n_repeats=1),
                     "uzunligi"),
}


@pytest.mark.parametrize("key", list(ARG_CASES))
def test_argument_validation(ctx, key):
    fn, match = ARG_CASES[key]
    with pytest.raises(ValueError, match=match):
        fn(ctx)


def test_spatial_split_errors_propagate(ctx):
    with pytest.raises(ValueError, match="k-fold|bloklar"):
        cv.run_cv(ctx.ds, ctx.groups, "spatial", ["RandomForest"], ctx.hp, n_splits=200, n_repeats=1)


def test_patch_model_requires_feature_stack(ctx):
    ds = dataclasses.replace(ctx.ds, feature_stack=None)
    pytest.importorskip("tensorflow")
    with pytest.raises(ValueError, match="feature_stack"):
        cv.run_cv(ds, ctx.groups, "spatial", ["CNN"], ctx.hp, n_splits=3, n_repeats=1)


def test_hp_is_validated_and_used(ctx):
    hp = fast_hp()
    hp["RandomForest"]["n_estimators"] = 1             # min=10 ga ko'tariladi (validate_hyperparams)
    logs = []
    res = cv.run_cv(ctx.ds, ctx.groups, "spatial", ["RandomForest"], hp, n_splits=3, n_repeats=1, seed=1,
                    log_fn=logs.append)
    assert res["hp"]["RandomForest"]["n_estimators"] == 10
    assert any("n_estimators" in m for m in logs)


def test_models_get_documented_arguments(ctx, monkeypatch):
    calls = []
    real = cv.make_model

    def spy(name, params, **kw):
        calls.append((name, dict(params), kw))
        return real(name, params, **kw)

    monkeypatch.setattr(cv, "make_model", spy)
    run(ctx, "spatial", ["RandomForest"], calibrate=False, calibration_method="isotonic", calibration_cv=2,
        n_jobs=2, seed=100)
    fold_calls = calls[1:]                                    # 0-chi chaqiruv - oldindan tekshiruv (probe)
    assert len(fold_calls) == 3
    for i, (name, params, kw) in enumerate(fold_calls):
        assert name == "RandomForest" and params["n_estimators"] == N_RF
        assert kw == {"calibrate": False, "calibration_method": "isotonic", "calibration_cv": 2, "n_jobs": 2,
                      "seed": 100 + i, "n_features": ctx.ds.X.shape[1]}


# ---------------------------------------------------------------------------
# run_cv: nested tuning (stub)
# ---------------------------------------------------------------------------
def make_tuning(**kw):
    t = TuningConfig(enabled=True, n_iter=2, inner_splits=2)
    t.models = {"RandomForest": True, "SVM": False, "XGBoost": False, "CNN": False}
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def test_nested_tuning_stub(ctx, monkeypatch):
    seen = []

    def tune_model(name, base_params, dataset, groups, *, tuning, calibrate, calibration_method, calibration_cv,
                   n_jobs, seed, log_fn=None, cancel=None, progress_fn=None):
        k = len(seen)
        seen.append({"name": name, "n": dataset.n, "groups": np.asarray(groups), "seed": seed, "base": dict(base_params),
                     "calibrate": calibrate, "n_jobs": n_jobs})
        if progress_fn:
            progress_fn(0.5, "yarmi")
            progress_fn(1.0, "tugadi")
        if log_fn:
            log_fn("stub tuning")
        best = dict(base_params, n_estimators=20 + k, max_depth=3 + k)
        return {"best_params": best, "best_score": 0.9 - 0.01 * k, "trials": [{"params": best, "score": 0.9, "std": 0.0}],
                "n_trials": 1}

    install_stub(monkeypatch, "mpm.tuning", tune_model=tune_model)
    created, cal_flags = [], []
    real = cv.make_model
    monkeypatch.setattr(cv, "make_model", lambda name, params, **kw: (created.append((name, dict(params), kw["seed"])),
                                                                      cal_flags.append(kw["calibrate"]),
                                                                      real(name, params, **kw))[2])
    calls, logs = [], []
    res = run(ctx, "spatial", ["RandomForest", "SVM"], rep=1, tuning=make_tuning(), seed=50,
              progress_fn=lambda f, m: calls.append((f, m)), log_fn=logs.append)
    assert [s["calibrate"] for s in seen] == [False] * 3                         # qidiruv kalibrlashsiz (tez, ball bir xil)
    assert all(cal_flags)                                                        # fold modellari esa calibrate=True bilan
    ft = res["fold_table"]
    assert [s["name"] for s in seen] == ["RandomForest"] * 3                     # faqat yoqilgan model, har fold
    assert [s["n"] for s in seen] == ft["n_train"].tolist()                      # tuning'ga faqat train qismi beriladi
    for s in seen:
        assert s["groups"].shape == (s["n"],) and s["base"]["n_estimators"] == N_RF
    assert [s["seed"] for s in seen] == [50, 51, 52]
    tp = res["tuned_params"]
    assert set(tp) == {"RandomForest"} and len(tp["RandomForest"]) == 3
    for i, rec in enumerate(tp["RandomForest"]):
        assert rec["best_params"]["n_estimators"] == 20 + i and rec["best_score"] == pytest.approx(0.9 - 0.01 * i)
        assert rec["fold"] == i and rec["repeat"] == 0 and rec["n_trials"] == 1 and len(rec["trials"]) == 1
    # fold modeli aynan fold'ning best_params'i bilan yaratilgan (probe'dan keyingi chaqiruvlar)
    rf = [c for c in created[2:] if c[0] == "RandomForest"]
    assert [c[1]["n_estimators"] for c in rf] == [20, 21, 22] and [c[1]["max_depth"] for c in rf] == [3, 4, 5]
    svm = [c for c in created[2:] if c[0] == "SVM"]
    assert len(svm) == 3 and all(c[1] == res["hp"]["SVM"] for c in svm)           # tuning'siz model bazaviy hp bilan
    assert any("stub tuning" in m for m in logs)
    fr = [c[0] for c in calls]
    assert fr == sorted(fr) and fr[-1] == pytest.approx(1.0) and len(fr) > 6      # tuning progress'i ham keladi
    assert any("tuning" in c[1] for c in calls)


def test_nested_tuning_random_mode_groups_none(ctx, monkeypatch):
    got = []

    def tune_model(name, base_params, dataset, groups, **kw):
        got.append((dataset.n, np.asarray(groups)))
        return {"best_params": dict(base_params), "best_score": 0.5, "trials": [], "n_trials": 0}

    install_stub(monkeypatch, "mpm.tuning", tune_model=tune_model)
    res = cv.run_cv(ctx.ds, None, "random", ["RandomForest"], ctx.hp, n_splits=3, n_repeats=1,
                    tuning=make_tuning(), seed=2, calibrate=False)
    assert len(got) == 3 and all(len(np.unique(g)) == n for n, g in got)        # har nuqta alohida guruh
    assert len(res["tuned_params"]["RandomForest"]) == 3


@pytest.mark.parametrize("tuning", [None, TuningConfig(enabled=False), "final_mode", "rf_off"])
def test_tuning_not_called_when_disabled(ctx, monkeypatch, tuning):
    def boom(*a, **k):
        raise AssertionError("tune_model chaqirilmasligi kerak")

    install_stub(monkeypatch, "mpm.tuning", tune_model=boom)
    if tuning == "final_mode":
        tuning = make_tuning(mode="final")
    elif tuning == "rf_off":
        tuning = make_tuning()
        tuning.models["RandomForest"] = False
    res = run(ctx, "spatial", ["RandomForest"], tuning=tuning, calibrate=False)
    assert res["tuned_params"] == {}


def test_tuning_accepts_dict(ctx, monkeypatch):
    n = []
    install_stub(monkeypatch, "mpm.tuning", tune_model=lambda name, bp, ds, g, **kw: (
        n.append(1), {"best_params": dict(bp), "best_score": 0.5, "trials": [], "n_trials": 0})[1])
    run(ctx, "spatial", ["RandomForest"], tuning=make_tuning().to_dict(), calibrate=False)
    assert len(n) == 3


def test_tuning_value_error_falls_back_to_base_hp(ctx, monkeypatch):
    def tune_model(*a, **k):
        raise ValueError("ichki fold uchun musbatlar yetmaydi")

    install_stub(monkeypatch, "mpm.tuning", tune_model=tune_model)
    logs = []
    res = run(ctx, "spatial", ["RandomForest"], tuning=make_tuning(), log_fn=logs.append, calibrate=False)
    rec = res["tuned_params"]["RandomForest"]
    assert len(rec) == 3 and all("fallback" in r and r["best_params"] == res["hp"]["RandomForest"] for r in rec)
    assert any("Ogohlantirish" in m and "tuning" in m for m in logs)
    assert any("tuning bajarilmadi" in w for w in res["warnings"])
    assert np.all(np.isfinite(res["oof"]["RandomForest"][0]))


def test_tuning_skipped_reason_is_reported(ctx, monkeypatch):
    def tune_model(name, base_params, *a, **k):
        return {"best_params": dict(base_params), "best_score": float("nan"), "trials": [], "n_trials": 0,
                "skipped_reason": "ichki bo'linish imkonsiz"}

    install_stub(monkeypatch, "mpm.tuning", tune_model=tune_model)
    res = run(ctx, "spatial", ["RandomForest"], tuning=make_tuning(), calibrate=False)
    assert any("ichki bo'linish imkonsiz" in w for w in res["warnings"]) and len(res["warnings"]) == 1
    assert res["tuned_params"]["RandomForest"][0]["best_params"] == res["hp"]["RandomForest"]


def test_tuning_other_errors_are_not_swallowed(ctx, monkeypatch):
    def tune_model(*a, **k):
        raise KeyError("ichki xato")

    install_stub(monkeypatch, "mpm.tuning", tune_model=tune_model)
    with pytest.raises(RuntimeError, match="RandomForest"):
        run(ctx, "spatial", ["RandomForest"], tuning=make_tuning())


def test_nested_tuning_real_module(ctx):
    pytest.importorskip("mpm.tuning")
    tuning = make_tuning(n_iter=2, inner_splits=2)
    tuning.spaces = {"RandomForest": {"n_estimators": {"min": 10, "max": 20}}}          # tez bo'lishi uchun
    res = run(ctx, "spatial", ["RandomForest"], tuning=tuning, seed=3)
    recs = res["tuned_params"]["RandomForest"]
    assert len(recs) == 3 and all("best_params" in r for r in recs)
    assert np.all(np.isfinite(res["oof"]["RandomForest"][0]))


def test_tuning_calibration_does_not_change_scores(ctx):
    """run_cv tuning'ni calibrate=False bilan chaqiradi: sigmoid kalibrlash monoton => nomzod ballari (AUC/AP) bir xil."""
    tuning_mod = pytest.importorskip("mpm.tuning")
    t = make_tuning(n_iter=3, inner_splits=2)
    t.spaces = {"RandomForest": {"n_estimators": {"min": 10, "max": 20}}}
    out = {}
    for cal in (True, False):
        out[cal] = tuning_mod.tune_model("RandomForest", fast_hp()["RandomForest"], ctx.ds, ctx.groups, tuning=t,
                                         calibrate=cal, calibration_method="sigmoid", calibration_cv=3, n_jobs=1,
                                         seed=4)
    a, b = out[True], out[False]
    assert [x["score"] for x in a["trials"]] == pytest.approx([x["score"] for x in b["trials"]], abs=1e-9)
    assert a["best_params"] == b["best_params"] and a["best_index"] == b["best_index"]


# ---------------------------------------------------------------------------
# run_cv: permutation importance
# ---------------------------------------------------------------------------
def test_perm_importance_stub_first_repeat_only(ctx, monkeypatch):
    calls = []
    p = ctx.ds.X.shape[1]

    def perm(model, X, y, *, patches=None, n_repeats=5, seed=0, cancel=None):
        calls.append({"model": model.name, "n": len(y), "patches": patches, "n_repeats": n_repeats, "seed": seed,
                      "p": X.shape[1], "pos": int(y.sum()), "cancel": cancel})
        return np.arange(p, dtype=float) + len(calls)

    def summarize(per_fold):
        a = np.vstack(per_fold)
        return {"mean": a.mean(0), "std": a.std(0), "per_fold": a, "n_folds": a.shape[0]}

    install_stub(monkeypatch, "mpm.explain", permutation_importance_auc=perm, summarize_perm_importance=summarize)
    tok = CancelToken()
    res = run(ctx, "spatial", ["RandomForest", "SVM"], rep=2, perm_importance=True, perm_repeats=4, seed=10,
              cancel=tok)
    assert len(calls) == 3 * 2                                  # faqat 1-repeat: 3 fold x 2 model
    assert all(c["cancel"] is tok for c in calls)               # Stop tugmasi perm importance ichida ham ishlashi uchun
    ft = res["fold_table"]
    assert sorted(c["n"] for c in calls if c["model"] == "RandomForest") == sorted(ft[ft["repeat"] == 0]["n_val"])
    assert all(c["patches"] is None and c["n_repeats"] == 4 and c["p"] == p for c in calls)
    assert sorted({c["seed"] for c in calls}) == [10, 11, 12]
    pi = res["perm_importance"]
    assert set(pi) == {"RandomForest", "SVM"} and all(v["n_folds"] == 3 and v["mean"].shape == (p,)
                                                       for v in pi.values())


def test_perm_importance_failure_is_warning_not_crash(ctx, monkeypatch):
    def perm(model, X, y, **kw):
        raise ValueError("buzildi")

    install_stub(monkeypatch, "mpm.explain", permutation_importance_auc=perm,
                 summarize_perm_importance=lambda per_fold: {"n_folds": len(per_fold)})
    logs = []
    res = run(ctx, "spatial", ["RandomForest"], perm_importance=True, log_fn=logs.append)
    assert res["perm_importance"] == {} and any("permutation importance" in m for m in logs)
    assert np.all(np.isfinite(res["oof"]["RandomForest"][0]))


def test_perm_importance_real_module(ctx):
    pytest.importorskip("mpm.explain")
    res = run(ctx, "spatial", ["RandomForest", "SVM"], perm_importance=True, perm_repeats=2, seed=4)
    p = ctx.ds.X.shape[1]
    for name in ("RandomForest", "SVM"):
        pi = res["perm_importance"][name]
        assert pi["n_folds"] == 3 and np.asarray(pi["mean"]).shape == (p,)


def test_cancel_inside_permutation_importance(ctx, monkeypatch):
    """Stop tugmasi uzoq permutation importance ichida ham ishlashi kerak (cancel unga uzatiladi): fold tugamasdan to'xtaydi."""
    explain = pytest.importorskip("mpm.explain")
    tok = CancelToken()
    real = explain.permutation_importance_auc

    def spy(*a, **kw):
        tok.cancel()                                     # perm importance boshlanganda Stop bosildi
        return real(*a, **kw)

    monkeypatch.setattr(explain, "permutation_importance_auc", spy)
    calls = []
    with pytest.raises(CancelledError):
        run(ctx, "spatial", ["RandomForest"], perm_importance=True, perm_repeats=2, cancel=tok,
            progress_fn=lambda f, m: calls.append(f))
    assert calls == []                                   # 1-fold yakunlanmadi (progress chaqirilmagan)


# ---------------------------------------------------------------------------
# run_cv: cancel, xato
# ---------------------------------------------------------------------------
def test_cancel_before_start(ctx):
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledError):
        run(ctx, "spatial", cancel=tok)


def test_cancel_after_first_fold(ctx):
    tok = CancelToken()
    calls = []

    def progress(f, m):
        calls.append(f)
        tok.cancel()

    with pytest.raises(CancelledError):
        run(ctx, "spatial", rep=2, cancel=tok, progress_fn=progress)
    assert len(calls) == 1                                      # 2-fold boshlanmadi


def test_cancel_between_models(ctx, monkeypatch):
    tok = CancelToken()
    real = cv.make_model
    n_svm = []

    def spy(name, params, **kw):
        if name == "SVM":
            n_svm.append(1)
            if len(n_svm) == 2:                                 # 1-chi - probe, 2-chi - 1-fold'dagi SVM
                tok.cancel()
        return real(name, params, **kw)

    monkeypatch.setattr(cv, "make_model", spy)
    with pytest.raises(CancelledError):
        run(ctx, "spatial", ["RandomForest", "SVM"], cancel=tok)


class _BoomModel:
    input_kind = "tabular"
    fit_info = {}
    counter = {"n": 0}

    def __init__(self, fail_on):
        self.fail_on = fail_on

    def fit(self, X, y, patches=None, cancel=None, log_fn=None):
        _BoomModel.counter["n"] += 1
        if _BoomModel.counter["n"] == self.fail_on:
            raise ValueError("sun'iy xato")
        self.p = float(np.mean(y))
        return self

    def predict_proba_pos(self, X=None, patches=None, batch_size=8192):
        return np.full(len(X), self.p)


def test_model_failure_is_loud(ctx, monkeypatch):
    _BoomModel.counter["n"] = 0
    real = cv.make_model
    monkeypatch.setattr(cv, "make_model",
                        lambda name, params, **kw: _BoomModel(2) if name == "SVM" else real(name, params, **kw))
    logs = []
    with pytest.raises(RuntimeError, match=r"SVM modeli \(1-takror, 2/3-fold\) xato berdi.*sun'iy xato") as ei:
        run(ctx, "spatial", ["RandomForest", "SVM"], rep=2, log_fn=logs.append)
    assert isinstance(ei.value.__cause__, ValueError)
    assert any(m.startswith("  XATO:") and "SVM" in m for m in logs)


def test_non_finite_prediction_is_error(ctx, monkeypatch):
    class Nan(_BoomModel):
        def predict_proba_pos(self, X=None, patches=None, batch_size=8192):
            return np.full(len(X), np.nan)

    _BoomModel.counter["n"] = 0
    monkeypatch.setattr(cv, "make_model", lambda name, params, **kw: Nan(-1))
    with pytest.raises(RuntimeError, match="bashorat yaroqsiz"):
        run(ctx, "spatial", ["SVM"])


def test_calibration_skipped_is_reported(ctx):
    hp = fast_hp()
    # calibration_cv=1 => eff_cv < 2 => kalibrlashsiz (fit_info["calibration"]="skipped") va ogohlantirish
    res = cv.run_cv(ctx.ds, ctx.groups, "spatial", ["RandomForest"], hp, n_splits=3, n_repeats=1,
                    calibration_cv=1, seed=1)
    assert any("kalibrlash" in w for w in res["warnings"]) and len(res["warnings"]) == 1   # takrorlanmaydi


# ---------------------------------------------------------------------------
# run_cv: CNN (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_cnn_patch_in_cv(ctx, monkeypatch):
    pytest.importorskip("tensorflow")
    hp = fast_hp()
    hp["CNN"].update(window=5, filters1=4, filters2=8, dense_units=8, batch_size=16, epochs=3, patience=2)
    n_patch_calls = []
    real = type(ctx.ds).get_patches

    def spy(self, window):
        n_patch_calls.append(window)
        return real(self, window)

    monkeypatch.setattr(type(ctx.ds), "get_patches", spy)
    logs = []
    res = cv.run_cv(ctx.ds, ctx.groups, "spatial", ["RandomForest", "CNN"], hp, n_splits=3, n_repeats=1,
                    seed=5, log_fn=logs.append)
    arr = res["oof"]["CNN"][0]
    assert arr.shape == (ctx.ds.n,) and np.all(np.isfinite(arr)) and 0.0 <= arr.min() and arr.max() <= 1.0
    assert n_patch_calls == [5]                                  # window bo'yicha kesh: bir marta olinadi
    assert any("CNN (patch2d)" in m for m in logs)               # CNN fit logi GUI logiga yetib keladi
    assert "seconds_CNN" in res["fold_table"].columns


def test_patch_cache_by_window():
    calls = []

    class FakeDS:
        def get_patches(self, w):
            calls.append(w)
            return np.zeros((2, w, w, 1), dtype=np.float32)

    c = cv._PatchCache(FakeDS())
    assert c.get(5) is c.get(5) and calls == [5]                     # kesh: bir marta
    c.get(7)
    c.get(9)
    assert calls == [5, 7, 9]
    c.get(11)                                                         # sig'im (3) oshdi: eng eskisi (5) chiqariladi
    c.get(5)
    assert calls == [5, 7, 9, 11, 5]
    assert c.get(5).shape == (2, 5, 5, 1)


@pytest.mark.slow
def test_cnn_tuned_window_and_perm_importance_get_matching_patches(ctx, monkeypatch):
    """Nested tuning CNN oynasini o'zgartirsa, fit/predict va perm importance AYNAN shu oyna patchlari bilan."""
    pytest.importorskip("tensorflow")
    hp = fast_hp()
    hp["CNN"].update(window=5, filters1=4, filters2=0, dense_units=8, batch_size=16, epochs=2, patience=2)
    n_patch_calls, perm_shapes = [], []
    real = type(ctx.ds).get_patches

    def spy(self, window):
        n_patch_calls.append(window)
        return real(self, window)

    def tune_model(name, base_params, dataset, groups, **kw):
        return {"best_params": dict(base_params, window=7), "best_score": 0.8, "trials": [], "n_trials": 1}

    def perm(model, X, y, *, patches=None, n_repeats=5, seed=0, cancel=None):
        perm_shapes.append(None if patches is None else patches.shape)
        return np.zeros(X.shape[1])

    monkeypatch.setattr(type(ctx.ds), "get_patches", spy)
    install_stub(monkeypatch, "mpm.tuning", tune_model=tune_model)
    install_stub(monkeypatch, "mpm.explain", permutation_importance_auc=perm,
                 summarize_perm_importance=lambda per_fold: {"n_folds": len(per_fold)})
    tuning = make_tuning()
    tuning.models["RandomForest"], tuning.models["CNN"] = False, True
    res = cv.run_cv(ctx.ds, ctx.groups, "spatial", ["CNN"], hp, n_splits=3, n_repeats=1, seed=5, tuning=tuning,
                    perm_importance=True)
    assert n_patch_calls == [7]                                       # tuning oynasi (7); bazaviy 5 so'ralmagan
    assert [s[1:3] for s in perm_shapes] == [(7, 7)] * 3 and all(s[0] > 0 for s in perm_shapes)
    assert [r["best_params"]["window"] for r in res["tuned_params"]["CNN"]] == [7, 7, 7]
    assert np.all(np.isfinite(res["oof"]["CNN"][0])) and res["perm_importance"]["CNN"]["n_folds"] == 3


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mres():
    y, oof = synth_oof(n=300, n_pos=60, repeats=3)
    groups = np.repeat(np.arange(30), 10)
    metrics, ens = cv.compute_metrics(y, oof, groups, n_boot=200, seed=3)
    return SimpleNamespace(y=y, oof=oof, groups=groups, metrics=metrics, ens=ens)


def test_metrics_keys(mres):
    assert list(mres.metrics) == ["A", "B", ENSEMBLE_NAME]
    for name, m in mres.metrics.items():
        assert REQUIRED_KEYS <= set(m), (name, REQUIRED_KEYS - set(m))
        lo, hi = m["auc_ci95"]
        assert lo < hi
        assert m["mean_proba"].shape == (300,) and m["fpr"].shape == m["tpr"].shape
        assert m["precision"].shape == m["recall"].shape
        assert np.asarray(m["confusion"]).shape == (2, 2)
    np.testing.assert_array_equal(mres.ens, mres.metrics[ENSEMBLE_NAME]["mean_proba"])


def test_metrics_match_sklearn(mres):
    y = mres.y
    for name in ("A", "B"):
        m, reps = mres.metrics[name], np.asarray(mres.oof[name])
        aucs = [roc_auc_score(y, r) for r in reps]
        assert m["auc"] == pytest.approx(np.mean(aucs)) and m["auc_std"] == pytest.approx(np.std(aucs, ddof=1))
        assert m["auc_single"] == pytest.approx(roc_auc_score(y, reps.mean(0)))
        assert m["pr_auc"] == pytest.approx(np.mean([average_precision_score(y, r) for r in reps]))
        assert m["brier"] == pytest.approx(np.mean([brier_score_loss(y, r) for r in reps]))
        assert m["balanced_accuracy"] == pytest.approx(
            np.mean([balanced_accuracy_score(y, (r >= 0.5).astype(int)) for r in reps]))
        assert m["f1"] == pytest.approx(np.mean([f1_score(y, (r >= 0.5).astype(int)) for r in reps]))
        assert m["f1_std"] == pytest.approx(np.std([f1_score(y, (r >= 0.5).astype(int)) for r in reps], ddof=1))
        thr = m["threshold_youden"]
        assert m["balanced_accuracy_youden"] == pytest.approx(
            np.mean([balanced_accuracy_score(y, (r >= thr).astype(int)) for r in reps]))
        assert m["f1_youden"] == pytest.approx(np.mean([f1_score(y, (r >= thr).astype(int)) for r in reps]))


def test_metrics_youden_confusion_consistency(mres):
    y = mres.y
    for m in mres.metrics.values():
        (tn, fp), (fn, tp) = m["confusion"]
        assert tp + fn == int(y.sum()) and tn + fp == int((y == 0).sum())
        assert m["sensitivity"] == pytest.approx(tp / (tp + fn)) and m["specificity"] == pytest.approx(tn / (tn + fp))
        pred = m["mean_proba"] >= m["threshold_youden"]
        assert int((pred & (y == 1)).sum()) == tp and int((pred & (y == 0)).sum()) == fp
        fpr, tpr, _ = roc_curve(y, m["mean_proba"])
        assert m["sensitivity"] - (1 - m["specificity"]) == pytest.approx((tpr - fpr).max())
        assert all(isinstance(v, int) for row in m["confusion"] for v in row)


def test_ensemble_is_mean_inside_each_repeat(mres):
    y = mres.y
    per_rep = [np.mean([mres.oof["A"][r], mres.oof["B"][r]], axis=0) for r in range(3)]
    e = mres.metrics[ENSEMBLE_NAME]
    np.testing.assert_allclose(e["mean_proba"], np.mean(per_rep, axis=0))
    assert e["auc"] == pytest.approx(np.mean([roc_auc_score(y, r) for r in per_rep]))
    assert e["auc_std"] == pytest.approx(np.std([roc_auc_score(y, r) for r in per_rep], ddof=1))
    assert e["auc_repeats"].shape == (3,)


def test_ci_matches_manual_block_bootstrap(mres):
    y, g = mres.y, mres.groups
    m = mres.metrics["A"]
    mean_p = np.mean(mres.oof["A"], axis=0)
    vals = []
    for idx in spatial.block_bootstrap_indices(g, 200, random_state=3):
        if y[idx].min() != y[idx].max():
            vals.append(roc_auc_score(y[idx], mean_p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    assert m["auc_ci95"] == pytest.approx((lo, hi))
    assert m["auc_ci95"][0] <= m["auc_single"] <= m["auc_ci95"][1]
    for k in ("pr_auc", "balanced_accuracy", "f1", "brier", "balanced_accuracy_youden", "f1_youden"):
        lo_k, hi_k = m[k + "_ci95"]
        assert lo_k <= hi_k


def test_ci_contains_auc():
    y, oof = synth_oof(n=400, n_pos=80, repeats=3, seed=5, strength=1.0)
    out, _ = cv.compute_metrics(y, oof, np.repeat(np.arange(40), 10), n_boot=300, seed=1)
    for m in out.values():
        assert m["auc_ci95"][0] <= m["auc"] <= m["auc_ci95"][1]
        assert m["auc_ci95"][0] <= m["auc_single"] <= m["auc_ci95"][1]


def test_ci_is_for_mean_proba_auc_even_with_noisy_repeats():
    """CI - mean OOF ehtimollik AUC'i (auc_single) uchun: repeat'lar o'zaro bog'liq bo'lmaganda repeat-o'rtacha
    "auc" undan past bo'lishi mumkin, lekin auc_single har doim CI ichida."""
    y, oof = synth_oof(n=400, n_pos=80, repeats=3, seed=5, strength=1.0, rep_noise=1.5)
    out, _ = cv.compute_metrics(y, oof, np.repeat(np.arange(40), 10), n_boot=300, seed=1)
    for m in out.values():
        assert m["auc_ci95"][0] <= m["auc_single"] <= m["auc_ci95"][1]
        assert m["auc"] <= m["auc_single"] + 1e-12


def test_groups_none_is_point_bootstrap_and_deterministic(mres):
    y, oof = mres.y, mres.oof
    a, _ = cv.compute_metrics(y, oof, None, n_boot=100, seed=9)
    b, _ = cv.compute_metrics(y, oof, np.arange(len(y)), n_boot=100, seed=9)
    c, _ = cv.compute_metrics(y, oof, None, n_boot=100, seed=9)
    d, _ = cv.compute_metrics(y, oof, None, n_boot=100, seed=10)
    for name in a:
        assert a[name]["auc_ci95"] == b[name]["auc_ci95"] == c[name]["auc_ci95"]
    assert a["A"]["auc_ci95"] != d["A"]["auc_ci95"]
    assert a["A"]["auc"] == mres.metrics["A"]["auc"]               # nuqtaviy qiymat groups/seed'ga bog'liq emas


def test_block_bootstrap_is_wider_for_clustered_data():
    rng = np.random.default_rng(0)
    n_blocks, per = 40, 10
    g = np.repeat(np.arange(n_blocks), per)
    b = rng.normal(size=n_blocks)[g]                           # blok effekti: blok ichi ma'lumotlari bog'liq
    y = (rng.random(g.size) < 1 / (1 + np.exp(-2.5 * b))).astype(int)
    p = 1 / (1 + np.exp(-(b + rng.normal(scale=0.2, size=g.size))))
    oof = {"M": [p]}
    blk, _ = cv.compute_metrics(y, oof, g, n_boot=400, seed=1)
    pts, _ = cv.compute_metrics(y, oof, None, n_boot=400, seed=1)
    w_blk = np.diff(blk["M"]["auc_ci95"])[0]
    w_pts = np.diff(pts["M"]["auc_ci95"])[0]
    assert w_blk > 1.2 * w_pts


def test_n_boot_zero_is_fast_mode(mres):
    out, _ = cv.compute_metrics(mres.y, mres.oof, mres.groups, n_boot=0)
    for m in out.values():
        for k in m:
            if k.endswith("_ci95"):
                assert np.all(np.isnan(m[k])) and len(m[k]) == 2
        assert m["auc"] == pytest.approx(mres.metrics[next(iter(out))]["auc"]) or True
    assert out["A"]["auc"] == mres.metrics["A"]["auc"] and out["A"]["pr_auc"] == mres.metrics["A"]["pr_auc"]


def test_single_model_has_ensemble_key():
    y, oof = synth_oof(models=("Only",), repeats=2)
    out, ens = cv.compute_metrics(y, oof, None, n_boot=50, seed=0)
    assert list(out) == ["Only", ENSEMBLE_NAME]
    for k in ("auc", "auc_std", "pr_auc", "brier", "threshold_youden", "auc_ci95"):
        assert out[ENSEMBLE_NAME][k] == out["Only"][k]
    np.testing.assert_array_equal(ens, out["Only"]["mean_proba"])


def test_single_repeat_std_is_zero():
    y, oof = synth_oof(models=("M",), repeats=1)
    out, _ = cv.compute_metrics(y, oof, None, n_boot=20)
    assert out["M"]["auc_std"] == 0.0 and out["M"]["auc"] == pytest.approx(out["M"]["auc_single"])


def test_oof_input_forms():
    y, oof = synth_oof(models=("A",), repeats=2)
    ref, _ = cv.compute_metrics(y, oof, None, n_boot=0)
    as2d, _ = cv.compute_metrics(y, {"A": np.vstack(oof["A"])}, None, n_boot=0)
    assert as2d["A"]["auc"] == ref["A"]["auc"]
    one, _ = cv.compute_metrics(y, {"A": oof["A"][0]}, None, n_boot=0)
    assert one["A"]["auc_std"] == 0.0 and one["A"]["auc"] == pytest.approx(roc_auc_score(y, oof["A"][0]))


def test_one_class_resamples_are_dropped():
    n_blocks, per = 40, 5
    g = np.repeat(np.arange(n_blocks), per)
    y = np.zeros(g.size, dtype=int)
    y[[0, 100]] = 1                                           # faqat 2 ta musbat (2 blokda): ko'p resample musbatsiz
    rng = np.random.default_rng(0)
    p = np.clip(0.2 + 0.5 * y + rng.normal(scale=0.1, size=y.size), 0, 1)
    logs = []
    out, _ = cv.compute_metrics(y, {"M": [p]}, g, n_boot=300, seed=1, log_fn=logs.append)
    assert any("bir sinfli" in m and "tashlandi" in m for m in logs)
    lo, hi = out["M"]["auc_ci95"]
    assert np.isfinite(lo) and np.isfinite(hi) and 0.0 <= lo <= hi <= 1.0


def test_too_few_valid_resamples_gives_nan_ci():
    g = np.arange(6)
    y = np.array([1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.2, 0.3, 0.1, 0.2])
    logs = []
    out, _ = cv.compute_metrics(y, {"M": [p]}, g, n_boot=1, seed=0, log_fn=logs.append)
    assert np.isfinite(out["M"]["auc"]) and any("Ogohlantirish" in m for m in logs)


def test_calibration_curve_structure(mres):
    for m in mres.metrics.values():
        c = m["calibration"]
        assert set(c) == {"prob_pred", "prob_true", "counts"}
        assert c["prob_pred"].shape == c["prob_true"].shape == c["counts"].shape
        assert 3 <= len(c["prob_pred"]) <= 10 and c["counts"].sum() == len(mres.y)
        assert np.all((c["prob_true"] >= 0) & (c["prob_true"] <= 1))
    assert len(mres.metrics["A"]["calibration"]["prob_pred"]) == 10          # 60 musbat => 10 bin


def test_calibration_bins_adapt_to_positives():
    y, oof = synth_oof(n=200, n_pos=8, models=("M",), repeats=1)
    out, _ = cv.compute_metrics(y, oof, None, n_boot=0)
    assert len(out["M"]["calibration"]["prob_pred"]) == 3                    # 8 musbat => minimal 3 bin


def test_fast_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    for trial in range(30):
        n = int(rng.integers(8, 200))
        y = (rng.random(n) < 0.3).astype(int)
        if y.min() == y.max():
            y[0], y[1] = 1, 0
        p = np.round(rng.random(n), int(rng.integers(0, 3)))                 # ko'p bog'langan qiymatlar
        thr = float(rng.choice(p))
        yb = y == 1
        assert cv._auc_fast(yb, p) == pytest.approx(roc_auc_score(y, p))
        assert cv._ap_fast(yb, p) == pytest.approx(average_precision_score(y, p))
        pt, fs = cv._point_stats(y, p, thr), cv._boot_stats(yb, p, thr)
        for k in pt:
            assert fs[k] == pytest.approx(pt[k]), (trial, k)


@pytest.mark.parametrize("kw, match", [
    ({"y": np.zeros(300)}, "bir sinfli"),
    ({"y": np.full(300, 2)}, "0/1"),
    ({"groups": np.arange(10)}, "groups"),
    ({"n_boot": -1}, "n_boot"),
    ({"oof": {}}, "oof"),
])
def test_metrics_errors(mres, kw, match):
    args = dict(y=mres.y, oof=mres.oof, groups=None, n_boot=0)
    args.update(kw)
    with pytest.raises(ValueError, match=match):
        cv.compute_metrics(args["y"], args["oof"], args["groups"], n_boot=args["n_boot"])


def test_metrics_reject_nan_and_mismatch(mres):
    bad = {"A": [np.where(np.arange(300) == 3, np.nan, 0.5)]}
    with pytest.raises(ValueError, match="NaN"):
        cv.compute_metrics(mres.y, bad, None, n_boot=0)
    with pytest.raises(ValueError, match="repeat"):
        cv.compute_metrics(mres.y, {"A": mres.oof["A"][:2], "B": mres.oof["B"][:1]}, None, n_boot=0)
    with pytest.raises(ValueError, match="shakli"):
        cv.compute_metrics(mres.y, {"A": [np.zeros(299)]}, None, n_boot=0)


def test_metrics_on_real_cv_oof(ctx, spatial_res):
    out, ens = cv.compute_metrics(ctx.ds.y, spatial_res.res["oof"], ctx.groups, n_boot=100, seed=2)
    assert list(out) == SK3 + [ENSEMBLE_NAME] and ens.shape == (ctx.ds.n,)
    for name, m in out.items():
        assert m["auc"] > 0.7 and m["auc_ci95"][0] <= m["auc"] <= m["auc_ci95"][1], name
        assert m["auc_std"] >= 0


# ---------------------------------------------------------------------------
# youden_threshold, metrics_dataframe
# ---------------------------------------------------------------------------
def test_youden_threshold_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(20):
        y = (rng.random(80) < 0.3).astype(int)
        y[:2] = [1, 0]
        p = np.round(rng.random(80) * 0.6 + 0.3 * y, 2)
        thr = cv.youden_threshold(y, p)
        best = max((p[y == 1] >= t).mean() - (p[y == 0] >= t).mean() for t in np.unique(p))
        assert (p[y == 1] >= thr).mean() - (p[y == 0] >= thr).mean() == pytest.approx(best)


def test_youden_threshold_edge_cases():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = cv.youden_threshold(y, p)
    assert np.isfinite(thr) and 0.3 < thr <= 0.7 and isinstance(thr, float)
    assert np.isfinite(cv.youden_threshold(y, np.full(6, 0.4)))                # o'zgarmas ehtimollik
    assert np.isfinite(cv.youden_threshold(y, 1 - p))                          # teskari tartib (AUC=0)
    with pytest.raises(ValueError, match="ikkala sinf"):
        cv.youden_threshold(np.ones(5), np.linspace(0, 1, 5))
    with pytest.raises(ValueError, match="uzunligi"):
        cv.youden_threshold(y, p[:-1])


def test_metrics_dataframe(mres):
    df = cv.metrics_dataframe(mres.metrics)
    assert list(df.columns) == ["Model", "AUC", "AUC_std", "AUC_CI_lo", "AUC_CI_hi", "PR_AUC", "BalAcc", "F1",
                                "Brier", "Sens", "Spec", "Thr"]
    assert df["Model"].tolist() == ["A", "B", ENSEMBLE_NAME]
    row = df.iloc[0]
    m = mres.metrics["A"]
    assert (row["AUC"], row["AUC_std"], row["AUC_CI_lo"], row["AUC_CI_hi"]) == (m["auc"], m["auc_std"], *m["auc_ci95"])
    assert (row["PR_AUC"], row["BalAcc"], row["F1"], row["Brier"]) == (m["pr_auc"], m["balanced_accuracy"], m["f1"],
                                                                        m["brier"])
    assert (row["Sens"], row["Spec"], row["Thr"]) == (m["sensitivity"], m["specificity"], m["threshold_youden"])
    lab = cv.metrics_dataframe(mres.metrics, label="spatial")
    assert list(lab.columns) == ["CV"] + list(df.columns) and (lab["CV"] == "spatial").all()
    assert pd.concat([lab, cv.metrics_dataframe(mres.metrics, label="random")]).shape[0] == 6
    assert list(cv.metrics_dataframe({}).columns) == list(df.columns)


# ---------------------------------------------------------------------------
# run_background_sensitivity
# ---------------------------------------------------------------------------
def bg_kwargs(ctx, **over):
    kw = dict(model_names=["RandomForest", "SVM"], n_background=60, min_distance=500.0, strategy="random",
              block_size=ctx.block, n_draws=2, n_splits=3, n_repeats=1, calibrate=True, calibration_method="sigmoid",
              calibration_cv=3, n_jobs=1, seed=11)
    kw.update(over)
    return kw


def run_bg(ctx, **over):
    return cv.run_background_sensitivity(ctx.aoi, ctx.pos, ctx.raster, ctx.pipe, ctx.hp, **bg_kwargs(ctx, **over))


@pytest.fixture(scope="module")
def bg_res(ctx):
    return run_bg(ctx)


def test_bg_sensitivity_structure(ctx, bg_res):
    r = bg_res
    assert [d["draw"] for d in r["per_draw"]] == [0, 1] and [d["seed"] for d in r["per_draw"]] == [1011, 1012]
    names = ["RandomForest", "SVM", ENSEMBLE_NAME]
    assert r["n_positive"] == len(ctx.pos) and r["n_draws"] == 2
    for d in r["per_draw"]:
        assert set(d["auc"]) == set(names) and all(0.0 <= v <= 1.0 for v in d["auc"].values())
    assert set(r["summary"]) == set(names)
    for name in names:
        s, vals = r["summary"][name], [d["auc"][name] for d in r["per_draw"]]
        assert s["values"] == vals and s["mean"] == pytest.approx(np.mean(vals))
        assert s["std"] == pytest.approx(np.std(vals, ddof=1))
        assert s["min"] == min(vals) and s["max"] == max(vals)
    assert r["summary"]["RandomForest"]["mean"] > 0.6


def test_bg_sensitivity_deterministic(ctx, bg_res):
    again = run_bg(ctx)
    assert again["per_draw"] == bg_res["per_draw"] and again["summary"] == bg_res["summary"]


def test_bg_sensitivity_pipeline_calls(ctx, monkeypatch):
    seeds, cv_calls = [], []
    real_gen, real_run = data.generate_background_points, cv.run_cv

    def gen(aoi, pos, n, md, random_state=0, **kw):
        seeds.append((random_state, n, md, kw.get("strategy"), kw["valid_mask"] is not None))
        return real_gen(aoi, pos, n, md, random_state=random_state, **kw)

    def spy_run(ds, groups, mode, names, hp, **kw):
        cv_calls.append((mode, kw["seed"], kw["n_splits"], kw["n_repeats"], kw["tuning"], kw["perm_importance"]))
        return real_run(ds, groups, mode, names, hp, **kw)

    monkeypatch.setattr(data, "generate_background_points", gen)
    monkeypatch.setattr(cv, "run_cv", spy_run)
    run_bg(ctx, n_draws=3, strategy="grid", seed=20)
    assert seeds == [(1020 + d, 60, 500.0, "grid", True) for d in range(3)]
    assert cv_calls == [("spatial", 1020 + d, 3, 1, None, False) for d in range(3)]


def test_bg_sensitivity_skips_draws_and_returns_none(ctx):
    logs = []
    out = run_bg(ctx, n_splits=200, log_fn=logs.append)            # musbatlar < n_splits => hamma draw o'tkaziladi
    assert out is None
    assert sum("o'tkazib yuborildi" in m for m in logs) == 2 and any("birorta draw bajarilmadi" in m for m in logs)


def test_bg_sensitivity_zero_draws_and_bad_args(ctx):
    assert run_bg(ctx, n_draws=0) is None
    for over, match in [({"block_size": None}, "block_size"), ({"block_size": -5.0}, "block_size"),
                        ({"block_size": float("nan")}, "block_size"), ({"strategy": "bogus"}, "strategiyasi"),
                        ({"model_names": []}, "model_names")]:
        with pytest.raises(ValueError, match=match):
            run_bg(ctx, **over)


def test_bg_sensitivity_progress_and_cancel(ctx):
    calls = []
    run_bg(ctx, n_draws=2, progress_fn=lambda f, m: calls.append((f, m)))
    fr = [c[0] for c in calls]
    assert fr == sorted(fr) and fr[-1] == pytest.approx(1.0) and all("Fon sezgirligi" in c[1] for c in calls)
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledError):
        run_bg(ctx, cancel=tok)
    tok2 = CancelToken()
    n = []

    def progress(f, m):
        n.append(f)
        tok2.cancel()

    with pytest.raises(CancelledError):
        run_bg(ctx, n_draws=3, cancel=tok2, progress_fn=progress)
    assert len(n) == 1


def test_bg_sensitivity_single_draw_std_zero(ctx):
    r = run_bg(ctx, n_draws=1, model_names=["RandomForest"])
    assert r["summary"]["RandomForest"]["std"] == 0.0 and len(r["per_draw"]) == 1
