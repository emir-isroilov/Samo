# -*- coding: utf-8 -*-
"""mpm.pipeline testlari: run_training (kalitlar, shakllar, log fayli, cancel, tuning + K-fon + kategorik),
run_prediction, export_results, estimate_cost_text. To'liq pipeline va CNN testlari - slow."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio

from mpm import common, pipeline
from mpm.common import ENSEMBLE_NAME, CancelledError, CancelToken
from mpm.config import RunConfig, TuningConfig, default_hyperparams
from tests.synth import make_synthetic_project

SK3 = ["RandomForest", "SVM", "XGBoost"]
RESULT_KEYS = {
    "cfg", "band_names", "feature_names", "categorical_layers", "raster", "pipeline", "dataset", "groups",
    "block_size", "n_positive", "n_background", "spatial", "random", "bg_sensitivity", "final_models",
    "final_hyperparams", "importance", "shap", "data_dictionary", "metadata_csv_path", "diagnostics",
    "thresholds", "versions", "timings", "log_path", "warnings", "aoi_gdf", "positive_gdf", "background_gdf"}
CV_KEYS = {"mode", "oof", "metrics", "metrics_df", "ensemble_oof", "fold_map", "fold_table", "perm_importance",
           "tuned_params", "feature_names"}
PRED_KEYS = {"maps", "uncertainty", "valid_mask", "class_map", "class_breaks", "class_stats", "success_curve",
             "saved_paths"}


# ---------------------------------------------------------------------------
# Yordamchilar
# ---------------------------------------------------------------------------
def fast_hp():
    hp = default_hyperparams()
    hp["RandomForest"]["n_estimators"] = 15
    hp["XGBoost"]["n_estimators"] = 15
    return hp


def make_cfg(proj, out_dir, **kw):
    base = dict(tiff_folder=proj["tiff"], points_folder=proj["points"], aoi_folder=proj["aoi"],
                output_dir=str(out_dir), n_background=60, min_distance=300.0, n_splits=3, n_repeats=2,
                n_bootstrap=50, hyperparams=fast_hp(), perm_importance_repeats=2, n_jobs=2, seed=11,
                use_models={"RandomForest": True, "SVM": True, "XGBoost": True, "CNN": False})
    base.update(kw)
    return RunConfig(**base)


def run_until(cfg, frac):
    """frac progress'ga yetganda bekor qiladi; (loglar) qaytaradi."""
    logs, tok = [], CancelToken()

    def prog(f, m):
        if f >= frac:
            tok.cancel()

    with pytest.raises(CancelledError):
        pipeline.run_training(cfg, log_fn=logs.append, progress_fn=prog, cancel=tok)
    return logs


@pytest.fixture(scope="module")
def proj(tmp_path_factory):
    """100x100, 5 qatlam, AOI ichida 40 musbat nuqta."""
    return make_synthetic_project(str(tmp_path_factory.mktemp("pl_proj")), size=100, n_layers=5, n_pos=40)


@pytest.fixture(scope="module")
def trained(proj, tmp_path_factory):
    out = tmp_path_factory.mktemp("pl_out")
    logs, prog = [], []
    cfg = make_cfg(proj, out)
    res = pipeline.run_training(cfg, log_fn=logs.append, progress_fn=lambda f, m: prog.append((f, m)))
    return SimpleNamespace(res=res, logs=logs, prog=prog, out=out, cfg=cfg, proj=proj)


@pytest.fixture(scope="module")
def predicted(trained, tmp_path_factory):
    out = tmp_path_factory.mktemp("pl_pred")
    logs = []
    pred = pipeline.run_prediction(trained.res, out_dir=str(out), log_fn=logs.append)
    return SimpleNamespace(pred=pred, out=out, logs=logs)


# ---------------------------------------------------------------------------
# estimate_cost_text (tez)
# ---------------------------------------------------------------------------
def test_estimate_cost_basic():
    cfg = RunConfig(n_splits=5, n_repeats=10, use_models={"RandomForest": True, "SVM": True, "XGBoost": True,
                                                           "CNN": False})
    txt = pipeline.estimate_cost_text(cfg)
    assert "random + spatial" in txt and "100 fold" in txt          # 5*10*2
    assert "Jami taxminan" in txt
    assert "tuning" not in txt.lower() and "TensorFlow" not in txt
    assert "TensorFlow" in pipeline.estimate_cost_text(RunConfig(n_splits=5, n_repeats=10))     # CNN yoqilgan


def test_estimate_cost_counts_exact():
    cfg = RunConfig(n_splits=4, n_repeats=3, run_random_cv=False, calibrate=False, final_bg_draws=2,
                    use_models={"RandomForest": True, "SVM": False, "XGBoost": False, "CNN": False})
    txt = pipeline.estimate_cost_text(cfg)
    assert "spatial, 4-fold x 3 takror = 12 fold: 12 ta o'qitish" in txt
    assert "Yakuniy modellar (2 fon tanlovi): 2 ta o'qitish" in txt
    assert "Jami taxminan 14 ichki fit" in txt


def test_estimate_cost_tuning_and_cnn():
    t = TuningConfig(enabled=True, n_iter=10, inner_splits=3,
                     models={"RandomForest": True, "SVM": False, "XGBoost": False, "CNN": True})
    cfg = RunConfig(n_splits=5, n_repeats=2, tuning=t, bg_sensitivity_enabled=True,
                    use_models={"RandomForest": True, "SVM": False, "XGBoost": False, "CNN": True})
    txt = pipeline.estimate_cost_text(cfg)
    assert "Nested tuning" in txt and "Yakuniy tuning" in txt
    assert "tuning qimmat" in txt and "daraxt quriladi" in txt            # RF narxi
    assert "CNN qimmat" in txt and "TensorFlow xotirasi" in txt and "MB" in txt
    assert "CNN tuning yoqilgan" in txt
    assert "Fon sezgirligi" in txt
    # nested + final tuning: 5*2 fold * 2 model * 30 -> RF 300 + CNN 300 nested (+30+30 final)
    assert "n_iter x ichki fold = 30" in txt


def test_estimate_cost_accepts_dict_and_never_raises():
    assert "Taxminiy" in pipeline.estimate_cost_text(RunConfig().to_dict())
    bad = RunConfig()
    bad.tuning = None
    assert isinstance(pipeline.estimate_cost_text(bad), str)


def test_multiline_log_collects_each_warning_separately():
    """Regressiya: ko'p qatorli xabar (estimate_cost_text) bitta katta ogohlantirishga aylanib ketmasin."""
    t = TuningConfig(enabled=True, n_iter=2, inner_splits=2,
                     models={"RandomForest": True, "SVM": False, "XGBoost": False, "CNN": True})
    cfg = RunConfig(tuning=t, bg_sensitivity_enabled=True,
                    use_models={"RandomForest": True, "SVM": False, "XGBoost": False, "CNN": True})
    ctx = pipeline._Ctx(None, None, None)
    ctx.log(pipeline.estimate_cost_text(cfg))
    w = ctx.warnings
    assert any(x.startswith("tuning qimmat") for x in w) and any(x.startswith("CNN qimmat") for x in w)
    assert any(x.startswith("CNN tuning yoqilgan") for x in w)
    assert all("OGOHLANTIRISH" not in x.upper() for x in w)             # har biri alohida, ichma-ich emas
    # davom qatori (bo'sh joy bilan) shu ogohlantirishga qo'shiladi, keyingi oddiy qator esa yo'q
    ctx2 = pipeline._Ctx(None, None, None)
    ctx2.log("  OGOHLANTIRISH: birinchi\n  davomi\nalohida oddiy qator\n  Ogohlantirish: ikkinchi")
    assert ctx2.warnings == ["birinchi davomi", "ikkinchi"]


# ---------------------------------------------------------------------------
# run_training: argument xatolari (tez)
# ---------------------------------------------------------------------------
def test_validate_errors(proj, tmp_path):
    cfg = make_cfg(proj, tmp_path, use_models={m: False for m in ("RandomForest", "SVM", "XGBoost", "CNN")})
    with pytest.raises(ValueError, match="model"):
        pipeline.run_training(cfg)
    with pytest.raises(ValueError, match="K-fold"):
        pipeline.run_training(make_cfg(proj, tmp_path, n_splits=1))
    with pytest.raises(ValueError, match="TIFF"):
        pipeline.run_training(make_cfg({**proj, "tiff": str(tmp_path / "yo'q")}, tmp_path))
    with pytest.raises(ValueError, match="AOI"):
        pipeline.run_training(make_cfg({**proj, "aoi": str(tmp_path)}, tmp_path))


def test_too_few_positives_clear_error(tmp_path):
    p = make_synthetic_project(str(tmp_path / "p"), size=60, n_layers=3, n_pos=6)
    with pytest.raises(ValueError, match="juda kam"):
        pipeline.run_training(make_cfg(p, tmp_path / "o", n_background=30))


def test_unavailable_models_dropped_and_error(proj, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "get_xgboost", lambda: None)
    only_x = make_cfg(proj, tmp_path, use_models={"RandomForest": False, "SVM": False, "XGBoost": True,
                                                  "CNN": False})
    with pytest.raises(ValueError, match="ishlatib bo'lmaydi"):
        pipeline.run_training(only_x)
    logs = run_until(make_cfg(proj, tmp_path / "b"), 0.0)           # birinchi progress'dayoq bekor
    joined = "\n".join(logs)
    assert "XGBoost o'rnatilmagan" in joined
    assert "Modellar: RandomForest, SVM" in joined


def test_cancel_immediately(proj, tmp_path):
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledError):
        pipeline.run_training(make_cfg(proj, tmp_path), cancel=tok)


# ---------------------------------------------------------------------------
# run_training: blok o'lchami, diagnostika va metadata ogohlantirishlari (erta bekor qilish bilan, tez)
# ---------------------------------------------------------------------------
def test_block_size_manual_and_named_band(proj, tmp_path):
    logs = "\n".join(run_until(make_cfg(proj, tmp_path / "a", block_size=2500.0), 0.16))
    assert "(qo'lda): 2,500 m" in logs and "Ishlatiladigan blok o'lchami: 2,500 m" in logs
    logs = "\n".join(run_until(make_cfg(proj, tmp_path / "b", variogram_band="layer2"), 0.16))
    assert "'layer2' bandi" in logs and "Empirik semivariogram" in logs
    logs = "\n".join(run_until(make_cfg(proj, tmp_path / "c", variogram_band="yoq_band"), 0.16))
    assert "'yoq_band' bandi topilmadi" in logs and "barcha raqamli bandlar" in logs


def test_nested_note_only_for_nested_tuning_mode(proj, tmp_path):
    """Regressiya: tuning rejimi 'final' bo'lsa spatial CV tuning'siz - 'nested tuning bilan baholandi' deyilmasin."""
    def mk_t(mode):
        return TuningConfig(enabled=True, mode=mode, n_iter=2, inner_splits=2,
                            models={"RandomForest": True, "SVM": False, "XGBoost": False, "CNN": False},
                            spaces={"RandomForest": {"n_estimators": {"min": 10, "max": 20}}})
    kw = dict(n_repeats=1, shap_enabled=False, perm_importance=False,
              use_models={"RandomForest": True, "SVM": False, "XGBoost": False, "CNN": False})
    final = "\n".join(run_until(make_cfg(proj, tmp_path / "f", tuning=mk_t("final"), **kw), 0.80))   # CV metrikalari (bootstrap) endi Stop'ni hurmat qiladi: final fit'da bekor
    assert "Random vs Spatial" in final and "nested tuning bilan" not in final
    nested = "\n".join(run_until(make_cfg(proj, tmp_path / "n", tuning=mk_t("nested"), **kw), 0.80))
    assert "nested tuning bilan" in nested


def test_block_size_adapted_when_too_big(proj, tmp_path):
    logs = "\n".join(run_until(make_cfg(proj, tmp_path, block_size=50000.0), 0.16))      # butun maydon = 1 blok
    assert "kamaytirildi" in logs                                                    # spatial.adapt_block_size


def test_block_size_fallback_when_variogram_none(proj, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.spatial, "estimate_autocorrelation_range", lambda *a, **k: None)
    logs = "\n".join(run_until(make_cfg(proj, tmp_path), 0.16))
    assert "zaxira blok o'lchami 1,000 m" in logs


def test_patch_cnn_block_leakage_warning(proj, tmp_path):
    pytest.importorskip("tensorflow")
    hp = fast_hp()
    hp["CNN"].update(window=9)
    cfg = make_cfg(proj, tmp_path, block_size=500.0, hyperparams=hp,
                   use_models={"RandomForest": True, "SVM": False, "XGBoost": False, "CNN": True})
    logs = "\n".join(run_until(cfg, 0.16))
    assert "CNN oynasidan" in logs and "leakage" in logs


def test_metadata_semicolon_cp1251_autodetected_and_constant_layer(tmp_path):
    """Excel mintaqaviy sozlamasi (';' + cp1251) endi avtomatik taniladi: layer1 maydonlari o'qiladi, fayl o'zgarmaydi."""
    p = make_synthetic_project(str(tmp_path / "p"), size=60, n_layers=3, n_pos=30)
    meta = os.path.join(p["tiff"], "metadata.csv")
    with open(meta, "w", encoding="cp1251") as f:
        f.write("band_name;source_owner;survey_date\nlayer1;Институт;2020\n")
    before = open(meta, "rb").read()
    with rasterio.open(os.path.join(p["tiff"], "layer2.tif"), "r+") as ds:          # o'zgarmas qatlam
        ds.write(np.full((ds.height, ds.width), 3.0, dtype="float32"), 1)
    logs = "\n".join(run_until(make_cfg(p, tmp_path / "o", n_background=30), 0.14))
    assert "avtomatik aniqlandi: ajratgich ';', kodlash cp1251" in logs
    assert "2/3 qatlam uchun qo'lda metadata" in logs                                # layer1 to'ldirilgan deb o'qildi
    assert "UTF-8 emas" not in logs and "ustunlar tanilmaydi" not in logs
    assert open(meta, "rb").read() == before
    assert "'layer2' qatlami deyarli o'zgarmas" in logs and "konstant" in logs


# ---------------------------------------------------------------------------
# Hardening: tuning ogohlantirishlari, import xatolari, summary formati (tez)
# ---------------------------------------------------------------------------
RF_ONLY = {"RandomForest": True, "SVM": False, "XGBoost": False, "CNN": False}


def tuning_cfg(proj, tmp_path, mode):
    t = TuningConfig(enabled=True, mode=mode, n_iter=2, inner_splits=2, models=dict(RF_ONLY),
                     spaces={"RandomForest": {"n_estimators": {"min": 10, "max": 20}}})
    return make_cfg(proj, tmp_path, tuning=t, use_models=dict(RF_ONLY))


def prepare_warnings(cfg):
    ctx = pipeline._Ctx(None, None, None)
    try:
        pipeline._prepare(cfg, ctx)
    finally:
        ctx.close()
    return ctx.warnings


def test_non_nested_tuning_warns_cv_uses_base_hp(proj, tmp_path):
    """Regressiya: tuning.mode != 'nested' => CV bazaviy hp bilan, yakuniy model tuned hp bilan - aniq ogohlantirish."""
    w = [x for x in prepare_warnings(tuning_cfg(proj, tmp_path / "f", "final")) if "tuned hp" in x]
    assert len(w) == 1 and "CV metrikalari bazaviy giperparametrlar bilan" in w[0] and "'final'" in w[0]
    assert not [x for x in prepare_warnings(tuning_cfg(proj, tmp_path / "n", "nested")) if "tuned hp" in x]
    off = make_cfg(proj, tmp_path / "o", use_models=dict(RF_ONLY))                 # tuning o'chiq
    assert not [x for x in prepare_warnings(off) if "tuned hp" in x]


def test_broken_optional_import_reports_module_and_error(proj, tmp_path, monkeypatch):
    """Regressiya: modul bor, lekin import yiqilgan => "o'rnatilmagan" emas, modul nomi + haqiqiy xato ogohlantiriladi."""
    monkeypatch.setattr(pipeline, "get_xgboost", lambda: None)
    monkeypatch.setattr(pipeline, "tf_available", lambda: True)
    monkeypatch.setattr(pipeline, "get_tf", lambda: None)
    monkeypatch.setitem(common._OPTIONAL_ERRORS, "xgboost", "OSError: libxgboost.so buzilgan")
    monkeypatch.setitem(common._OPTIONAL_ERRORS, "tensorflow", "ImportError: DLL load failed")
    cfg = make_cfg(proj, tmp_path, use_models={"RandomForest": True, "SVM": False, "XGBoost": True, "CNN": True})
    w = prepare_warnings(cfg)
    xg = [x for x in w if "xgboost" in x]
    tf = [x for x in w if "tensorflow" in x]
    assert xg and "libxgboost.so buzilgan" in xg[0] and "o'rnatilgan, lekin import qilib bo'lmadi" in xg[0]
    assert tf and "DLL load failed" in tf[0] and "CNN o'tkazib yuboriladi" in tf[0]
    assert not any("o'rnatilmagan ('pip" in x for x in w)
    # xato matni yo'q (modul haqiqatan o'rnatilmagan) => eski xabar
    monkeypatch.delitem(common._OPTIONAL_ERRORS, "xgboost")
    assert any("XGBoost o'rnatilmagan ('pip install xgboost')" in x for x in prepare_warnings(cfg))


def test_shap_import_failure_is_reported(monkeypatch):
    monkeypatch.setitem(common._OPTIONAL_ERRORS, "shap", "ImportError: numba mos emas")
    monkeypatch.setattr(pipeline, "compute_shap_summary", lambda *a, **k: None)
    ctx = pipeline._Ctx(None, None, None)
    assert pipeline._shap(RunConfig(shap_enabled=True), ctx, {}, SimpleNamespace(X=None, feature_names=[])) is None
    assert any("shap" in w and "numba mos emas" in w for w in ctx.warnings)
    ctx2 = pipeline._Ctx(None, None, None)
    monkeypatch.delitem(common._OPTIONAL_ERRORS, "shap")
    pipeline._shap(RunConfig(shap_enabled=True), ctx2, {}, SimpleNamespace(X=None, feature_names=[]))
    assert ctx2.warnings == []


def fake_result(std=float("nan"), tuned=False, final_tuning=None, random=True, bg_values=(0.8,)):
    m = lambda auc: {"auc": auc, "auc_std": std, "auc_ci95": (auc - 0.05, auc + 0.05), "pr_auc": 0.7,
                     "sensitivity": 0.8, "specificity": 0.7, "brier": 0.2, "threshold_youden": 0.4}
    metrics = {"RandomForest": m(0.812), ENSEMBLE_NAME: m(0.83)}
    bg = {"n_draws": len(bg_values), "summary": {"RandomForest": {
        "mean": float(np.mean(bg_values)), "std": 0.0, "min": min(bg_values), "max": max(bg_values),
        "values": list(bg_values)}}}
    return {"cfg": {"seed": 1}, "spatial": {"metrics": metrics, "n_splits": 3, "n_repeats": 1,
                                            "tuned_params": {"RandomForest": [{"best_params": {}}]} if tuned else {}},
            "random": {"metrics": {"RandomForest": m(0.9), ENSEMBLE_NAME: m(0.91)}} if random else None,
            "n_positive": 30, "n_background": 60, "feature_names": ["a"], "band_names": ["a"],
            "categorical_layers": [], "block_size": 1000.0, "model_names": ["RandomForest"],
            "final_models": {"RandomForest": [object()]}, "bg_sensitivity": bg, "importance": {},
            "final_tuning": final_tuning or {}, "warnings": [], "timings": {"total": 1.0}}


def test_summary_nan_std_is_dash_not_zero_or_nan():
    txt = pipeline._summary_text(fake_result(), None)
    line = next(x for x in txt.splitlines() if x.strip().startswith("RandomForest: AUC=0.812"))
    assert "(std: -, 1 takror)" in line and "nan" not in line.lower() and "0.000" not in line
    line2 = next(x for x in pipeline._summary_text(fake_result(std=0.0123), None).splitlines()
                 if x.strip().startswith("RandomForest: AUC=0.812"))
    assert "+/- 0.012" in line2
    bg = next(x for x in txt.splitlines() if "AUC 0.800" in x)
    assert "(1 tanlov: std yo'q)" in bg and "0.000" not in bg                       # 1 draw: std 0.000 emas


def test_summary_random_vs_spatial_tuning_notes():
    nested = pipeline._summary_text(fake_result(tuned=True, final_tuning={"RandomForest": {}}), None)
    assert "nested tuning bilan baholandi" in nested and "qisman tuning farqi" in nested
    final = pipeline._summary_text(fake_result(final_tuning={"RandomForest": {}}), None)
    assert "ikkala CV ham bazaviy giperparametrlar bilan" in final and "nested tuning bilan baholandi" not in final
    plain = pipeline._summary_text(fake_result(), None)
    assert "Eslatma: random CV" not in plain and "ikkala CV ham bazaviy" not in plain
    assert "Random vs Spatial" not in pipeline._summary_text(fake_result(random=False, tuned=True), None)


def test_summary_final_tuning_floats_pretty():
    rec = {"scoring": "roc_auc", "best_score": 0.8123456789,
           "best_params": {"n_estimators": 123, "max_depth": 4.123456789, "lr": np.float64(0.0123456789), "w": "uniform"}}
    txt = pipeline._summary_text(fake_result(final_tuning={"RandomForest": rec}), None)
    line = next(x for x in txt.splitlines() if x.strip().startswith("RandomForest: roc_auc="))
    assert line.strip() == "RandomForest: roc_auc=0.8123 ; n_estimators=123, max_depth=4.123, lr=0.01235, w=uniform"
    nan = pipeline._summary_text(fake_result(final_tuning={"RandomForest": {"scoring": "roc_auc",
                                                                           "best_score": float("nan"),
                                                                           "best_params": {}}}), None)
    assert "roc_auc=- ;" in nan


def test_bg_sensitivity_receives_dataset_feature_stack(proj, tmp_path, monkeypatch):
    """Regressiya: pipeline dataset.feature_stack ni fon sezgirligiga uzatadi (CNN patch2d: qayta qurilmasin)."""
    monkeypatch.setattr(pipeline, "tf_available", lambda: True)
    monkeypatch.setattr(pipeline, "get_tf", lambda: object())                     # TensorFlow import qilinmaydi

    def fake_run_cv(ds, groups, mode, names, hp, **kw):
        rng = np.random.default_rng(0)
        return {"mode": mode, "oof": {m: [rng.random(ds.n) for _ in range(kw["n_repeats"])] for m in names},
                "fold_map": np.zeros(ds.n, dtype=int), "fold_table": pd.DataFrame(),
                "feature_names": list(ds.feature_names), "warnings": [], "hp": hp,
                "n_splits": kw["n_splits"], "n_repeats": kw["n_repeats"]}

    class Stop(Exception):
        pass

    got = {}

    def spy_bg(*a, **kw):
        got.update(kw)
        raise Stop

    monkeypatch.setattr(pipeline, "run_cv", fake_run_cv)
    monkeypatch.setattr(pipeline, "run_background_sensitivity", spy_bg)
    hp = fast_hp()
    hp["CNN"]["mode"] = "patch2d"
    cfg = make_cfg(proj, tmp_path, hyperparams=hp, bg_sensitivity_enabled=True, run_random_cv=False, n_repeats=1,
                   use_models={"RandomForest": True, "SVM": False, "XGBoost": False, "CNN": True})
    with pytest.raises(Stop):
        pipeline.run_training(cfg)
    assert "feature_stack" in got and got["feature_stack"] is not None and got["feature_stack"].ndim == 3


# ---------------------------------------------------------------------------
# run_training: to'liq natija (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_result_keys_and_shapes(trained):
    r = trained.res
    assert RESULT_KEYS <= set(r)
    ds = r["dataset"]
    assert ds.X.shape[1] == len(r["feature_names"]) == len(r["band_names"]) == 5
    assert r["groups"].shape == (ds.n,) and r["n_positive"] == ds.n_pos and r["n_background"] == ds.n_neg
    assert ds.n_pos >= 10 and r["block_size"] > 0
    for blk in (r["spatial"], r["random"]):
        assert CV_KEYS <= set(blk)
        assert set(blk["oof"]) == set(SK3) and all(len(v) == 2 for v in blk["oof"].values())
        assert blk["oof"]["SVM"][0].shape == (ds.n,) and np.isfinite(blk["ensemble_oof"]).all()
        assert list(blk["metrics_df"]["Model"]) == SK3 + [ENSEMBLE_NAME]
        assert blk["fold_map"].shape == (ds.n,) and len(blk["fold_table"]) == 6
        assert set(blk["metrics"]) == set(SK3) | {ENSEMBLE_NAME}
        assert all(len(c) == 2 for c in (blk["metrics"]["SVM"]["auc_ci95"],))
    assert r["spatial"]["mode"] == "spatial" and r["random"]["mode"] == "random"
    assert set(r["spatial"]["perm_importance"]) == set(SK3) and not r["random"]["perm_importance"]
    assert r["spatial"]["metrics"][ENSEMBLE_NAME]["auc"] > 0.7


@pytest.mark.slow
def test_models_thresholds_importance_shap(trained):
    r = trained.res
    assert all(len(v) == 1 and v[0].is_fitted for v in r["final_models"].values())
    assert set(r["final_models"]) == set(SK3)
    assert set(r["final_hyperparams"]) == set(SK3) and all(len(v) == 1 for v in r["final_hyperparams"].values())
    assert r["final_hyperparams"]["RandomForest"][0]["n_estimators"] == 15            # yagona hp manbai
    assert set(r["thresholds"]) == set(SK3) | {ENSEMBLE_NAME}
    assert all(0 <= v <= 1 for v in r["thresholds"].values())
    imp = r["importance"]
    assert "permutation" in imp["method"] and imp["feature_names"] == r["feature_names"]
    assert set(imp["models"]) == set(SK3)
    assert all(d["mean"].shape == (5,) and d["std"].shape == (5,) for d in imp["models"].values())
    assert imp["models"]["RandomForest"]["mean"].argmax() == r["feature_names"].index("layer2") or \
        imp["models"]["RandomForest"]["mean"].argmax() == r["feature_names"].index("layer1")
    sh = r["shap"]
    assert set(sh) == {"RandomForest", "XGBoost"} and sh["RandomForest"]["units"] == "ehtimollik"
    assert sh["XGBoost"]["units"] == "log-odds" and sh["RandomForest"]["shap_values"].ndim == 2
    assert r["bg_sensitivity"] is None and r["final_tuning"] == {}
    assert r["random"] is not None


@pytest.mark.slow
def test_metadata_timings_versions_warnings(trained):
    r = trained.res
    assert os.path.isfile(r["metadata_csv_path"])
    assert list(r["data_dictionary"]["band_name"]) == r["band_names"]
    assert {"data", "diagnostics", "blocks", "random_cv", "spatial_cv", "final_fit", "importance", "total"} \
        <= set(r["timings"])
    assert r["timings"]["total"] >= r["timings"]["spatial_cv"] > 0
    assert r["versions"]["python"] and "scikit-learn" in r["versions"]
    assert isinstance(r["warnings"], list) and all(isinstance(w, str) for w in r["warnings"])
    assert any("metadata" in w for w in r["warnings"])                                # bo'sh metadata ogohlantirishi
    assert "diagnostics" in r and "layer_stats" in r["diagnostics"]
    assert r["cfg"]["n_splits"] == 3 and r["cfg"]["seed"] == 11
    json.dumps(r["cfg"])                                                              # JSON'ga yaroqli


@pytest.mark.slow
def test_progress_monotonic_and_log(trained):
    fr = [f for f, _ in trained.prog]
    assert fr[0] == 0.0 and fr[-1] == 1.0 and all(0.0 <= f <= 1.0 for f in fr)
    assert all(b >= a - 1e-9 for a, b in zip(fr, fr[1:]))
    assert any(0.16 < f < 0.36 for f in fr) and any(0.36 < f < 0.70 for f in fr)        # random va spatial oraliqlari
    joined = "\n".join(trained.logs)
    assert "Random vs Spatial" in joined and "optimizm" in joined
    assert "kalibrlash=sigmoid" in joined and "Youden bo'sag'lari" in joined
    assert "prospektivlik indeksi" in joined
    assert "AUC=" in joined


@pytest.mark.slow
def test_log_file_utf8_and_sequential(trained, proj):
    path = trained.res["log_path"]
    assert path == os.path.join(str(trained.out), "mpm_run_1.log")
    text = open(path, encoding="utf-8").read()
    assert "±" in text and "Random vs Spatial" in text and "OGOHLANTIRISH" in text
    run_until(make_cfg(proj, trained.out), 0.0)                                       # ikkinchi ishga tushirish
    assert os.path.isfile(os.path.join(str(trained.out), "mpm_run_2.log"))


@pytest.mark.slow
def test_cancel_mid_run_closes_log(proj, tmp_path):
    logs = run_until(make_cfg(proj, tmp_path), 0.45)                                 # random CV oxiri / spatial boshi
    log_file = tmp_path / "mpm_run_1.log"
    assert log_file.exists()
    assert "to'xtatildi" in log_file.read_text(encoding="utf-8")
    log_file.unlink()                                                                # yopilgan fayl o'chadi
    assert any("RANDOM cross-validation" in m for m in logs)


@pytest.mark.slow
def test_reproducible_same_seed(proj, tmp_path):
    kw = dict(run_random_cv=False, n_repeats=1, shap_enabled=False, perm_importance=False, n_jobs=1,
              use_models={"RandomForest": True, "SVM": True, "XGBoost": False, "CNN": False})
    a = pipeline.run_training(make_cfg(proj, tmp_path / "a", **kw))
    b = pipeline.run_training(make_cfg(proj, tmp_path / "b", **kw))
    assert a["random"] is None and a["shap"] is None
    for m in ("RandomForest", "SVM"):
        np.testing.assert_allclose(a["spatial"]["oof"][m][0], b["spatial"]["oof"][m][0])
    np.testing.assert_array_equal(a["dataset"].coords, b["dataset"].coords)
    # perm importance o'chirilgan -> MDI zaxirasi (RF) va SVM uchun importance yo'q
    assert a["importance"]["models"]["RandomForest"]["source"] == "mdi" and "SVM" not in a["importance"]["models"]


# ---------------------------------------------------------------------------
# Bekor qilish: importance/SHAP atrofida va yakunlashda; 1 repeat: std NaN (slow)
# ---------------------------------------------------------------------------
def small_cfg(proj, out, **kw):
    base = dict(run_random_cv=False, n_repeats=1, n_bootstrap=0, shap_enabled=False, perm_importance=False, n_jobs=1,
                use_models={"RandomForest": True, "SVM": True, "XGBoost": False, "CNN": False})
    base.update(kw)
    return make_cfg(proj, out, **base)


def run_cancel_at(cfg, frac):
    """progress frac'ga yetganda cancel belgilaydi; natija yoki CancelledError ko'tariladi."""
    tok = CancelToken()
    seen = []

    def prog(f, m):
        seen.append(f)
        if f >= frac:
            tok.cancel()

    return pipeline.run_training(cfg, progress_fn=prog, cancel=tok), seen


@pytest.mark.slow
def test_cancel_between_importance_and_shap_raises(proj, tmp_path):
    """Regressiya: ~0.95 dan keyin (importance -> SHAP orasida) Stop e'tiborsiz qolmasin (avval natija qaytardi)."""
    with pytest.raises(CancelledError):
        run_cancel_at(small_cfg(proj, tmp_path), 0.95)


@pytest.mark.slow
def test_cancel_during_shap_stage_raises(proj, tmp_path):
    pytest.importorskip("shap")
    cfg = small_cfg(proj, tmp_path, shap_enabled=True,
                    use_models={"RandomForest": True, "SVM": False, "XGBoost": False, "CNN": False})
    with pytest.raises(CancelledError):
        run_cancel_at(cfg, 0.95)


@pytest.mark.slow
def test_cancel_after_shap_returns_finished_result(proj, tmp_path):
    """Qaror: SHAP tugagach (0.98: "Importance tayyor") yoki oxirida (1.0) Stop bosilsa tayyor natija qaytariladi."""
    for k, frac in enumerate((0.98, 1.0)):
        res, seen = run_cancel_at(small_cfg(proj, tmp_path / str(k)), frac)
        assert RESULT_KEYS <= set(res) and seen[-1] == 1.0 and res["thresholds"]
        assert res["timings"]["total"] > 0


@pytest.mark.slow
def test_single_repeat_run_std_nan_in_exports(proj, tmp_path):
    """1 repeat: auc_std NaN; CSV/XLSX'da bo'sh katak, summary.txt'da '-' (0.000 / nan EMAS)."""
    res = pipeline.run_training(small_cfg(proj, tmp_path / "o", run_random_cv=True))
    for blk in (res["spatial"], res["random"]):
        assert all(np.isnan(m["auc_std"]) for m in blk["metrics"].values())
        assert blk["metrics_df"]["AUC_std"].isna().all()
    d = tmp_path / "exp"
    pipeline.export_results(res, str(d))
    for f in ("metrics_spatial.csv", "metrics_random.csv"):
        df = pd.read_csv(d / f)
        assert df["AUC_std"].isna().all() and df["AUC"].notna().all()
        assert "nan" not in (d / f).read_text(encoding="utf-8").lower()
    x = pd.read_excel(d / "metrics_spatial.xlsx", sheet_name=None)
    assert all(df["AUC_std"].isna().all() for df in x.values() if "AUC_std" in df)
    summary = (d / "summary.txt").read_text(encoding="utf-8")
    line = next(x for x in summary.splitlines() if x.strip().startswith("RandomForest: AUC="))
    assert "(std: -, 1 takror)" in line and "0.000" not in line and "nan" not in line.lower()


# ---------------------------------------------------------------------------
# run_prediction (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_prediction_structure(trained, predicted):
    p, r = predicted.pred, trained.res
    assert PRED_KEYS <= set(p)
    H, W = r["raster"].height, r["raster"].width
    assert set(p["maps"]) == set(SK3) | {ENSEMBLE_NAME} and all(m.shape == (H, W) for m in p["maps"].values())
    assert p["uncertainty"].shape == (H, W) and p["valid_mask"].shape == (H, W)
    assert p["class_map"].shape == (H, W) and p["class_map"].dtype == np.int8
    assert len(p["class_breaks"]) == 4 and len(p["class_stats"]) == 5
    assert list(p["class_stats"].columns) == ["Sinf", "Nomi", "Piksel", "Maydon_km2", "Maydon_%", "Konlar_soni",
                                              "Konlar_%", "Boyitish"]
    assert p["class_stats"]["Konlar_soni"].sum() == r["n_positive"]
    sc = p["success_curve"]
    assert sc["n_pos"] == r["n_positive"] and 0.5 < sc["auc"] <= 1.0
    assert sc["area_frac"][0] == 0.0 and sc["capture_frac"][-1] == 1.0


@pytest.mark.slow
def test_prediction_saves_files(predicted):
    names = {os.path.basename(p) for p in predicted.pred["saved_paths"]}
    assert {"prognoz_RandomForest.tif", "prognoz_Ensemble_soft_voting.tif", "prognoz_classes.tif",
            "prognoz_uncertainty.tif", "predictor_data_dictionary.csv"} <= names
    for p in predicted.pred["saved_paths"]:
        assert os.path.isfile(p)
    with rasterio.open(os.path.join(str(predicted.out), "prognoz_Ensemble_soft_voting.tif")) as src:
        assert src.nodata == -9999.0 and src.dtypes[0] == "float32"
        assert "Prospektivlik indeksi" in (src.descriptions[0] or "")


@pytest.mark.slow
def test_prediction_class_overrides_and_no_out_dir(trained):
    p = pipeline.run_prediction(trained.res, class_method="equal_interval", n_classes=3)
    assert p["saved_paths"] == [] and len(p["class_stats"]) == 3 and p["class_breaks"] == pytest.approx([1 / 3, 2 / 3])
    p = pipeline.run_prediction(trained.res, class_method="fixed", class_breaks=[0.3, 0.6])
    assert len(p["class_stats"]) == 3 and p["class_breaks"] == [0.3, 0.6]
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledError):
        pipeline.run_prediction(trained.res, cancel=tok)


# ---------------------------------------------------------------------------
# export_results (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_export_files_readable(trained, predicted, tmp_path):
    logs = []
    paths = pipeline.export_results(trained.res, str(tmp_path / "exp"), predicted.pred, log_fn=logs.append)
    names = {os.path.basename(p) for p in paths}
    expected = {"metrics_spatial.csv", "metrics_spatial.xlsx", "metrics_random.csv", "hyperparameters_used.json",
                "run_config.json", "importance.csv", "fold_table.csv", "oof_predictions.csv",
                "predictor_data_dictionary.csv", "versions.json", "class_stats.csv", "summary.txt"}
    assert expected <= names and "bg_sensitivity.json" not in names and "tuned_params.csv" not in names
    assert all(os.path.isfile(p) for p in paths)
    d = str(tmp_path / "exp")
    m = pd.read_csv(os.path.join(d, "metrics_spatial.csv"))
    assert list(m["Model"]) == SK3 + [ENSEMBLE_NAME] and m["AUC"].between(0, 1).all()
    assert len(pd.read_csv(os.path.join(d, "metrics_random.csv"))) == 4
    assert set(pd.read_excel(os.path.join(d, "metrics_spatial.xlsx"), sheet_name=None)) == {
        "Spatial CV", "Random CV", "Sinflar"}
    oof = pd.read_csv(os.path.join(d, "oof_predictions.csv"))
    assert list(oof.columns[:4]) == ["x", "y", "y_true", "fold"] and len(oof) == trained.res["dataset"].n
    assert set(SK3) | {"Ensemble"} <= set(oof.columns) and oof["y_true"].sum() == trained.res["n_positive"]
    imp = pd.read_csv(os.path.join(d, "importance.csv"))
    assert set(imp["model"]) == set(SK3) and len(imp) == 15
    ft = pd.read_csv(os.path.join(d, "fold_table.csv"))
    assert set(ft["cv"]) == {"random", "spatial"} and len(ft) == 12
    hp = json.load(open(os.path.join(d, "hyperparameters_used.json"), encoding="utf-8"))
    assert hp["final"]["RandomForest"][0]["n_estimators"] == 15
    rc = json.load(open(os.path.join(d, "run_config.json"), encoding="utf-8"))
    assert rc["n_splits"] == 3 and rc["band_names"] == trained.res["band_names"]
    assert json.load(open(os.path.join(d, "versions.json"), encoding="utf-8"))["python"]
    cs = pd.read_csv(os.path.join(d, "class_stats.csv"))
    assert len(cs) == 5 and "Boyitish" in cs.columns
    dd = pd.read_csv(os.path.join(d, "predictor_data_dictionary.csv"))
    assert list(dd["band_name"]) == trained.res["band_names"]
    summary = open(os.path.join(d, "summary.txt"), encoding="utf-8").read()
    assert "prospektivlik indeksi" in summary and "ehtimollik EMAS" in summary
    assert "Random vs Spatial" in summary and "Ogohlantirishlar" in summary


@pytest.mark.slow
def test_export_without_prediction_and_without_openpyxl(trained, tmp_path, monkeypatch):
    real = pipeline.importlib.util.find_spec
    monkeypatch.setattr(pipeline.importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "openpyxl" else real(name, *a, **k))
    logs = []
    paths = pipeline.export_results(trained.res, str(tmp_path), log_fn=logs.append)
    names = {os.path.basename(p) for p in paths}
    assert "metrics_spatial.csv" in names and "metrics_spatial.xlsx" not in names and "class_stats.csv" not in names
    assert any("openpyxl" in m for m in logs)


# ---------------------------------------------------------------------------
# Tuning + K fon tanlovi + kategorik + fon sezgirligi + n_bootstrap=0 (slow)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tuned(synth_project_cat, tmp_path_factory):
    out = tmp_path_factory.mktemp("pl_tuned")
    t = TuningConfig(enabled=True, n_iter=2, inner_splits=2, scoring="roc_auc",
                     models={"RandomForest": True, "SVM": True, "XGBoost": False, "CNN": False},
                     spaces={"RandomForest": {"n_estimators": {"min": 10, "max": 20}}})       # tez qidiruv
    cfg = make_cfg(synth_project_cat, out, categorical_layers=["geology_cat"], tuning=t, final_bg_draws=2,
                   run_random_cv=False, n_bootstrap=0, n_repeats=1, bg_sensitivity_enabled=True,
                   bg_sensitivity_draws=2, bg_sensitivity_repeats=1, n_background=50, shap_enabled=False)
    logs = []
    res = pipeline.run_training(cfg, log_fn=logs.append)
    return SimpleNamespace(res=res, logs=logs, out=out)


@pytest.mark.slow
def test_tuned_categorical_final_draws(tuned):
    r = tuned.res
    assert r["categorical_layers"] == ["geology_cat"]
    assert any(f.startswith("geology_cat==") for f in r["feature_names"]) and len(r["feature_names"]) > 5
    assert r["random"] is None
    for m in SK3:
        assert len(r["final_models"][m]) == 2 and len(r["final_hyperparams"][m]) == 2
        assert r["final_models"][m][0] is not r["final_models"][m][1]
    ft = r["final_tuning"]
    assert set(ft) == {"RandomForest", "SVM"} and all(v["n_trials"] == 2 for v in ft.values())
    assert r["final_hyperparams"]["RandomForest"][0] == r["final_hyperparams"]["RandomForest"][1]
    assert r["final_hyperparams"]["RandomForest"][0] == ft["RandomForest"]["best_params"]
    assert r["final_hyperparams"]["XGBoost"][0]["n_estimators"] == 15                    # tuning'siz = bazaviy hp
    tp = r["spatial"]["tuned_params"]
    assert set(tp) == {"RandomForest", "SVM"} and len(tp["RandomForest"]) == 3
    assert all("best_params" in rec for recs in tp.values() for rec in recs)
    # n_bootstrap=0 => CI hisoblanmaydi (NaN), lekin natija to'liq
    lo, hi = r["spatial"]["metrics"]["RandomForest"]["auc_ci95"]
    assert np.isnan(lo) and np.isnan(hi) and 0 <= r["spatial"]["metrics"]["RandomForest"]["auc"] <= 1
    # feature_stack ulashilgan emas (CNN yo'q) va importance one-hot ustunlar bilan mos
    assert r["importance"]["models"]["SVM"]["mean"].shape == (len(r["feature_names"]),)
    # fon sezgirligi
    bg = r["bg_sensitivity"]
    assert bg is not None and set(bg["summary"]) == set(SK3) | {ENSEMBLE_NAME} and bg["n_draws"] >= 1
    joined = "\n".join(tuned.logs)
    assert "yakuniy tuning" in joined and "Tuning faqat 0-fon tanlovida" in joined
    assert "Fon tanloviga sezgirlik" in joined


@pytest.mark.slow
def test_tuned_prediction_and_export(tuned, tmp_path):
    pred = pipeline.run_prediction(tuned.res, out_dir=str(tmp_path / "pred"))
    assert pred["maps"][ENSEMBLE_NAME].shape == (tuned.res["raster"].height, tuned.res["raster"].width)
    paths = pipeline.export_results(tuned.res, str(tmp_path / "exp"), pred)
    names = {os.path.basename(p) for p in paths}
    assert {"tuned_params.csv", "tuning_trials.csv", "bg_sensitivity.json", "metrics_spatial.csv"} <= names
    assert "metrics_random.csv" not in names
    raw = (tmp_path / "exp" / "tuned_params.csv").read_text(encoding="utf-8").splitlines()
    assert raw[1].startswith("RandomForest,nested,0,0,")                # regressiya: repeat/fold '0.0' emas
    tp = pd.read_csv(tmp_path / "exp" / "tuned_params.csv")
    assert set(tp["scope"]) == {"nested", "final"} and set(tp["model"]) == {"RandomForest", "SVM"}
    assert json.loads(tp["best_params"].iloc[0])
    tt = pd.read_csv(tmp_path / "exp" / "tuning_trials.csv")
    assert len(tt) == len(tp) * 2 and {"score", "params", "trial"} <= set(tt.columns)
    bg = json.load(open(tmp_path / "exp" / "bg_sensitivity.json", encoding="utf-8"))
    assert "summary" in bg
    m = pd.read_csv(tmp_path / "exp" / "metrics_spatial.csv")
    assert m["AUC_CI_lo"].isna().all()                                                    # n_bootstrap=0


# ---------------------------------------------------------------------------
# CNN (patch2d) e2e (slow, TensorFlow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_cnn_end_to_end(proj, tmp_path):
    pytest.importorskip("tensorflow")
    hp = fast_hp()
    hp["CNN"].update(mode="patch2d", window=5, filters1=4, filters2=0, dense_units=8, epochs=2, patience=1,
                     batch_size=16, kernel_size=3)
    cfg = make_cfg(proj, tmp_path / "o", hyperparams=hp, n_repeats=1, run_random_cv=False, shap_enabled=False,
                   perm_importance_repeats=1, n_jobs=1,
                   use_models={"RandomForest": True, "SVM": False, "XGBoost": False, "CNN": True})
    logs = []
    res = pipeline.run_training(cfg, log_fn=logs.append)
    assert set(res["final_models"]) == {"RandomForest", "CNN"}
    cnn = res["final_models"]["CNN"][0]
    assert cnn.input_kind == "patch" and cnn.fit_info["epochs_run"] >= 1
    assert res["dataset"].feature_stack is not None
    assert set(res["spatial"]["oof"]) == {"RandomForest", "CNN"}
    assert "CNN" in res["spatial"]["perm_importance"] and "CNN" in res["importance"]["models"]
    assert any("CNN chiqishi kalibrlanmaydi" in w for w in res["warnings"])
    assert any("CNN (patch2d) fit" in m or "CNN 1/1-tanlov) fit" in m or "early_stopping" in m for m in logs)
    pred = pipeline.run_prediction(res, out_dir=str(tmp_path / "pred"), batch_size=2048)
    assert set(pred["maps"]) == {"RandomForest", "CNN", ENSEMBLE_NAME}
    assert np.isfinite(pred["maps"]["CNN"][pred["valid_mask"]]).all()
    assert pred["uncertainty"] is not None
    paths = pipeline.export_results(res, str(tmp_path / "exp"), pred)
    assert any(p.endswith("summary.txt") for p in paths)
