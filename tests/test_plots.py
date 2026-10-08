# -*- coding: utf-8 -*-
"""mpm.gui.plots testlari (Qt'siz, Agg): haqiqiy kichik CV natijasi (RF + SVM, tests/synth.py + mpm.data) va qo'lda
yasalgan minimal lug'atlar. Har bir draw_* uchun: oddiy holat, bo'sh holat, saqlash; BUG-01 (SHAP shakllari),
BUG-14 (kesilmaslik/legend), ENH-04/07 xususiyatlari."""
from __future__ import annotations

import os
import warnings
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.text import Text
from rasterio.transform import from_origin, xy as rio_xy
from shapely.geometry import MultiPolygon, Point, box

from mpm import cv, data, explain, predict, spatial
from mpm.common import ENSEMBLE_NAME, safe_name
from mpm.config import TuningConfig, default_hyperparams
from mpm.gui import plots
from mpm.models import make_model

ERR_TEXT = "Chizishda xato"
DRAWERS = ["draw_roc", "draw_pr", "draw_calibration", "draw_confusion", "draw_spatial_diagnostics",
           "draw_importance", "draw_shap_beeswarm", "draw_shap_dependence", "draw_corr_heatmap", "draw_map",
           "draw_success_rate", "draw_tuning_trials"]


# ---------------------------------------------------------------------------
# Yordamchilar
# ---------------------------------------------------------------------------
def new_fig(w=9, h=6):
    fig = Figure(figsize=(w, h))
    FigureCanvasAgg(fig)
    return fig


def texts(fig):
    return [t.get_text() for t in fig.findobj(Text) if t.get_text()]


def has_text(fig, sub):
    return any(sub in t for t in texts(fig))


def assert_ok(fig):
    """Xato fallback'i ishlamagan (haqiqiy chizilgan), "Ma'lumot yo'q" yo'q."""
    assert not has_text(fig, ERR_TEXT), [t for t in texts(fig) if ERR_TEXT in t]
    assert not has_text(fig, plots.NO_DATA)
    assert fig.axes


def assert_no_data(fig):
    assert has_text(fig, plots.NO_DATA)
    assert len(fig.axes) == 1
    assert not has_text(fig, ERR_TEXT), texts(fig)


def renders(fig, tmp_path, name="x.png"):
    """Savedir: PNG yoziladi, nolga teng emas."""
    path = os.path.join(str(tmp_path), name)
    fig.savefig(path, dpi=60)
    assert os.path.getsize(path) > 1000
    return path


def legends(fig):
    out = list(fig.legends)
    out += [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    return out


def assert_inside(fig):
    """BUG-14: legend/suptitle/axes figura chegarasidan chiqib kesilmaydi."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    tol = 2.0
    arts = list(legends(fig))
    if fig._suptitle is not None:
        arts.append(fig._suptitle)
    for a in arts:
        bb = a.get_window_extent(r)
        assert bb.x0 >= fb.x0 - tol and bb.y0 >= fb.y0 - tol and bb.x1 <= fb.x1 + tol and bb.y1 <= fb.y1 + tol, \
            (a, bb, fb)
    for ax in fig.axes:
        bb = ax.get_tightbbox(r)
        assert bb.x0 >= fb.x0 - tol and bb.x1 <= fb.x1 + tol and bb.y0 >= fb.y0 - tol and bb.y1 <= fb.y1 + tol, \
            (ax, bb, fb)


def assert_legend_outside_axes(fig):
    """Figura legend'lari axes maydoniga tushmaydi (ma'lumot ustida emas)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for lg in fig.legends:
        lb = lg.get_window_extent(r)
        for ax in fig.axes:
            ab = ax.get_window_extent(r)
            ov_x = min(lb.x1, ab.x1) - max(lb.x0, ab.x0)
            ov_y = min(lb.y1, ab.y1) - max(lb.y0, ab.y0)
            assert not (ov_x > 1 and ov_y > 1), (lg, ax)


# ---------------------------------------------------------------------------
# Haqiqiy kichik natija (bir marta)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ctx(synth_project_cat):
    p = synth_project_cat
    raster = data.load_and_align_rasters(data.find_tiff_files(p["tiff"]), categorical=["geology_cat"])
    pipe = data.FeaturePipeline(raster.band_names, raster.categorical).fit(raster)
    aoi = data.load_aoi(data.find_shapefile(p["aoi"]))
    valid = data.valid_pixel_mask(raster)
    inner = np.zeros_like(valid)
    inner[10:-10, 10:-10] = True
    score = raster.stack[0] + raster.stack[1]
    cand = valid & inner & (score > np.percentile(score[valid & inner], 85))
    rr, cc = np.nonzero(cand)
    sel = np.random.default_rng(0).choice(rr.size, size=30, replace=False)
    xs, ys = rio_xy(raster.transform, rr[sel], cc[sel])
    pos = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xs, ys)], crs="EPSG:28411")
    bg = data.generate_background_points(aoi, pos, 80, 500.0, random_state=1, valid_mask=valid,
                                         transform=raster.transform)
    ds = data.build_dataset(raster, pipe, pos, bg)
    block, groups = spatial.adapt_block_size(ds.coords, ds.y, 3, 2000.0)
    hp = default_hyperparams()
    hp["RandomForest"]["n_estimators"] = 15
    hp["XGBoost"]["n_estimators"] = 15
    names = ["RandomForest", "SVM"]
    sp = cv.run_cv(ds, groups, "spatial", names, hp, n_splits=3, n_repeats=2, perm_importance=True,
                   perm_repeats=2, seed=7)
    rn = cv.run_cv(ds, None, "random", names, hp, n_splits=3, n_repeats=2, seed=7)
    sm, _ = cv.compute_metrics(ds.y, sp["oof"], groups, n_boot=60, seed=1)
    rm, _ = cv.compute_metrics(ds.y, rn["oof"], None, n_boot=60, seed=1)
    final = {}
    for n in ("RandomForest", "XGBoost"):
        m = make_model(n, hp[n], calibrate=True, n_features=ds.X.shape[1])
        m.fit(ds.X, ds.y)
        final[n] = [m]
    shap = explain.compute_shap_summary(final, ds.X, ds.feature_names, max_background=40)
    maps = predict.predict_probability_maps(raster, pipe, final)
    ens = maps["maps"][ENSEMBLE_NAME]
    class_map, _ = predict.classify_map(ens)
    curve = predict.success_rate_curve(ens, ds.rows[ds.y == 1], ds.cols[ds.y == 1])
    result = {"spatial": {"mode": "spatial", "metrics": sm, "fold_map": sp["fold_map"],
                          "perm_importance": sp["perm_importance"]},
              "random": {"mode": "random", "metrics": rm}, "dataset": ds, "block_size": block,
              "n_positive": int(ds.y.sum()), "bg_sensitivity": None,
              "importance": {"method": "OOF permutation (AUC pasayishi)", "models": sp["perm_importance"],
                             "feature_names": list(ds.feature_names)},
              "shap": shap}
    return SimpleNamespace(raster=raster, ds=ds, aoi=aoi, pos=pos, bg=bg, sm=sm, rm=rm, sp=sp, groups=groups,
                           result=result, ens=ens, class_map=class_map, curve=curve, shap=shap, hp=hp,
                           corr=data.data_diagnostics(raster, ds)["corr"])


@pytest.fixture(scope="module")
def tuned(ctx):
    """Haqiqiy nested tuning (kichik qidiruv oralig'i) => run_cv['tuned_params']."""
    t = TuningConfig(enabled=True, n_iter=3, inner_splits=2,
                     models={"RandomForest": True, "SVM": True, "XGBoost": False, "CNN": False},
                     spaces={"RandomForest": {"n_estimators": {"min": 10, "max": 20}}})
    res = cv.run_cv(ctx.ds, ctx.groups, "spatial", ["RandomForest", "SVM"], ctx.hp, n_splits=3, n_repeats=1,
                    tuning=t, seed=3)
    return res["tuned_params"]


def manual_metrics(n_models=("A", "B"), cal_bins=(3, 10), with_ci=True, prev=0.3):
    """Qo'lda yasalgan minimal compute_metrics lug'ati."""
    out = {}
    for i, name in enumerate(list(n_models) + [ENSEMBLE_NAME]):
        k = cal_bins[i % len(cal_bins)]
        x = np.linspace(0, 1, 20)
        out[name] = {
            "auc": 0.8 + 0.02 * i, "auc_std": 0.01, "auc_ci95": (0.7, 0.9) if with_ci else (np.nan, np.nan),
            "pr_auc": 0.6, "pr_auc_std": 0.02, "pr_auc_ci95": (0.5, 0.7), "brier": 0.2,
            "fpr": x, "tpr": np.sqrt(x), "precision": 1 - 0.5 * x, "recall": x[::-1],
            "threshold_youden": 0.4, "sensitivity": 0.8, "specificity": 0.7,
            "confusion": [[70, 30], [6, 24]],
            "calibration": {"prob_pred": np.linspace(0.05, 0.9, k), "prob_true": np.linspace(0.0, 1.0, k),
                            "counts": np.full(k, 10)},
        }
    return out


def manual_shap(n=40, p=5, seed=0, shape="np", units="ehtimollik"):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    sv = np.column_stack([X[:, j] * (p - j) * 0.1 for j in range(p)]) + rng.normal(scale=0.01, size=(n, p))
    if shape == "npc":
        raw = np.stack([-sv, sv], axis=-1)
    elif shape == "list":
        raw = [-sv, sv]
    elif shape == "cnp":
        raw = np.stack([-sv, sv], axis=0)
    else:
        raw = sv
    return {"shap_values": raw, "X_background": X, "feature_names": [f"f{j}" for j in range(p)], "units": units,
            "expected_value": 0.4, "mean_abs_shap": np.abs(sv).mean(axis=0)}


# ---------------------------------------------------------------------------
# Umumiy: har bir draw_* tashlab yubormaydi
# ---------------------------------------------------------------------------
GARBAGE = [None, {}, [], 0, "x", {"a": 1}, float("nan"), np.empty((0, 0)), pd.DataFrame(), {"metrics": None}]


def call(fname, fig, g):
    """Har bir draw_* ni bitta (yaroqsiz) birinchi ma'lumot bilan chaqiradi."""
    extra = {"draw_pr": (None,), "draw_confusion": ("RandomForest",), "draw_shap_beeswarm": ("RandomForest",),
             "draw_shap_dependence": ("RandomForest", "layer1"), "draw_map": (None, "t")}.get(fname, ())
    return getattr(plots, fname)(fig, g, *extra)


@pytest.mark.parametrize("fname", DRAWERS)
@pytest.mark.parametrize("bad", range(len(GARBAGE)))
def test_garbage_never_raises(fname, bad, tmp_path):
    fig = new_fig()
    assert call(fname, fig, GARBAGE[bad]) is fig
    assert fig.axes
    assert not has_text(fig, ERR_TEXT), texts(fig)       # yaroqsiz kirish "ma'lumot yo'q", istisno emas
    renders(fig, tmp_path)


@pytest.mark.parametrize("fname", DRAWERS)
def test_previous_content_is_replaced(fname):
    """Oldingi chizma tozalanadi (fig.clear): bo'sh holatda faqat bitta o'q qoladi."""
    fig = new_fig()
    fig.add_subplot(121).plot([1, 2], [1, 2])
    fig.add_subplot(122).plot([1, 2], [1, 2])
    call(fname, fig, None)
    assert len(fig.axes) == 1


def test_wrapped_functions_exposed():
    for f in DRAWERS:
        assert hasattr(getattr(plots, f), "__wrapped__")
    assert plots.INDEX_LABEL == "Prospektivlik indeksi (0-1)"


def test_unexpected_exception_is_not_fatal(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise TypeError("sun'iy xato")
    monkeypatch.setattr(plots, "_models_dict", boom)
    fig = new_fig()
    plots.draw_roc(fig, manual_metrics())
    assert has_text(fig, plots.NO_DATA) and has_text(fig, "TypeError")
    renders(fig, tmp_path)


# ---------------------------------------------------------------------------
# draw_roc
# ---------------------------------------------------------------------------
def test_roc_real(ctx, tmp_path):
    fig = new_fig()
    assert plots.draw_roc(fig, ctx.sm, "spatial") is fig
    assert_ok(fig)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    for name in ctx.sm:
        row = next(l for l in labels if l.startswith(name))
        m = ctx.sm[name]
        assert f"{m['auc']:.3f}" in row and "±" in row and "95% CI" in row
        lo, hi = m["auc_ci95"]
        assert f"{lo:.3f}" in row and f"{hi:.3f}" in row
    assert "spatial" in ax.get_title()
    assert ax.get_xlabel() and ax.get_ylabel()
    lws = {l.get_label().split(":")[0]: l.get_linewidth() for l in ax.lines}
    ens_lw = lws[ENSEMBLE_NAME]
    assert all(ens_lw > lw for k, lw in lws.items() if k != ENSEMBLE_NAME and k.startswith(("Random", "SVM")))
    assert_inside(fig)
    renders(fig, tmp_path)


def test_roc_without_ci_and_variants():
    ms = manual_metrics(with_ci=False)
    fig = new_fig()
    plots.draw_roc(fig, ms)
    assert_ok(fig)
    rows = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert not any("95% CI" in r for r in rows if r.startswith("A:"))
    fig2 = new_fig()
    plots.draw_roc(fig2, (manual_metrics(), np.zeros(3)))          # compute_metrics kortej shakli
    assert_ok(fig2)
    fig3 = new_fig()
    plots.draw_roc(fig3, {"mode": "spatial", "oof": {}, "metrics": manual_metrics()})   # CVBlock
    assert_ok(fig3)


@pytest.mark.parametrize("bad", [None, {}, {"A": {}}, {"A": {"fpr": [0.0], "tpr": [0.0]}},
                                 {"A": {"fpr": [0, 1], "tpr": [0]}}])
def test_roc_empty(bad):
    fig = new_fig()
    plots.draw_roc(fig, bad)
    assert_no_data(fig)


# ---------------------------------------------------------------------------
# draw_pr
# ---------------------------------------------------------------------------
def test_pr_real_baseline(ctx, tmp_path):
    fig = new_fig()
    plots.draw_pr(fig, ctx.sm, ctx.ds.y)
    assert_ok(fig)
    ax = fig.axes[0]
    hl = [l for l in ax.lines if len(set(l.get_ydata())) == 1 and len(l.get_ydata()) == 2]
    assert len(hl) == 1
    assert hl[0].get_ydata()[0] == pytest.approx(ctx.ds.y.mean())
    assert any("PR-AUC" in t.get_text() for t in ax.get_legend().get_texts())
    assert any("Baseline" in t.get_text() for t in ax.get_legend().get_texts())
    assert ax.get_xlabel() == "Recall (sensitivlik)"
    renders(fig, tmp_path)


def test_pr_baseline_fallback_from_confusion_and_no_y():
    fig = new_fig()
    plots.draw_pr(fig, manual_metrics(), None)               # confusion: (6+24)/(130)
    assert_ok(fig)
    hl = [l for l in fig.axes[0].lines if len(l.get_ydata()) == 2]
    assert hl[0].get_ydata()[0] == pytest.approx(30 / 130)
    m = manual_metrics()
    for v in m.values():
        v.pop("confusion")
    fig2 = new_fig()
    plots.draw_pr(fig2, m, None)                             # baseline yo'q, lekin chiziladi
    assert_ok(fig2)
    assert not any("Baseline" in t.get_text() for t in fig2.axes[0].get_legend().get_texts())


@pytest.mark.parametrize("bad", [None, {}, {"A": {"recall": [0.5], "precision": [0.5]}}])
def test_pr_empty(bad):
    fig = new_fig()
    plots.draw_pr(fig, bad, [0, 1])
    assert_no_data(fig)


# ---------------------------------------------------------------------------
# draw_calibration
# ---------------------------------------------------------------------------
def test_calibration_real_variable_bins(ctx, tmp_path):
    fig = new_fig()
    plots.draw_calibration(fig, ctx.sm)
    assert_ok(fig)
    ax = fig.axes[0]
    diag = [l for l in ax.lines if list(l.get_xdata()) == [0, 1] and list(l.get_ydata()) == [0, 1]]
    assert len(diag) == 1
    for name, m in ctx.sm.items():
        line = next(l for l in ax.lines if l.get_label().startswith(name))
        np.testing.assert_allclose(line.get_xdata(), m["calibration"]["prob_pred"])
        np.testing.assert_allclose(line.get_ydata(), m["calibration"]["prob_true"])
    renders(fig, tmp_path)


def test_calibration_manual_lengths_differ():
    fig = new_fig()
    plots.draw_calibration(fig, manual_metrics(cal_bins=(3, 10)))
    assert_ok(fig)
    lens = sorted({len(l.get_xdata()) for l in fig.axes[0].lines if len(l.get_xdata()) > 2})
    assert lens == [3, 10]


@pytest.mark.parametrize("bad", [None, {}, {"A": {"calibration": None}},
                                 {"A": {"calibration": {"prob_pred": [0.1, 0.2], "prob_true": [0.1]}}}])
def test_calibration_empty(bad):
    fig = new_fig()
    plots.draw_calibration(fig, bad)
    assert_no_data(fig)


# ---------------------------------------------------------------------------
# draw_confusion
# ---------------------------------------------------------------------------
def test_confusion_real(ctx, tmp_path):
    fig = new_fig()
    plots.draw_confusion(fig, ctx.sm, ENSEMBLE_NAME)
    assert_ok(fig)
    m = ctx.sm[ENSEMBLE_NAME]
    (tn, fp), (fn, tp) = m["confusion"]
    tx = texts(fig)
    for v in (tn, fp, fn, tp):
        assert any(t.startswith(str(int(v)) + "\n") for t in tx), (v, tx)
    title = fig.axes[0].get_title()
    assert f"{m['sensitivity']:.3f}" in title and f"{m['specificity']:.3f}" in title
    assert f"{m['threshold_youden']:.3f}" in title
    assert ENSEMBLE_NAME in title
    renders(fig, tmp_path)


def test_confusion_default_model_and_errors():
    ms = manual_metrics()
    fig = new_fig()
    plots.draw_confusion(fig, ms)
    assert_ok(fig)
    assert ENSEMBLE_NAME in fig.axes[0].get_title()
    f2 = new_fig()
    plots.draw_confusion(f2, ms, "YO'Q")
    assert_no_data(f2)
    bad = {"A": {"confusion": [[1, 2, 3]], "sensitivity": 0.5}}
    f3 = new_fig()
    plots.draw_confusion(f3, bad, "A")
    assert_no_data(f3)
    f4 = new_fig()
    plots.draw_confusion(f4, {"A": {"confusion": [[0, 0], [0, 0]]}}, "A")      # nol qator: bo'linish xatosiz
    assert_ok(f4)
    f5 = new_fig()
    plots.draw_confusion(f5, None, "A")
    assert_no_data(f5)


# ---------------------------------------------------------------------------
# draw_spatial_diagnostics (BUG-14)
# ---------------------------------------------------------------------------
def test_spatial_real_legend_not_clipped(ctx, tmp_path):
    fig = new_fig(10, 5)
    plots.draw_spatial_diagnostics(fig, ctx.result)
    assert_ok(fig)
    assert fig._suptitle is not None and "Spatial" in fig._suptitle.get_text()
    lg = [l for l in fig.legends if l.get_title().get_text() == "Validation fold"]
    assert len(lg) == 1
    names = [t.get_text() for t in lg[0].get_texts()]
    assert names[:3] == ["Fold 1", "Fold 2", "Fold 3"] and any("konlar" in n for n in names)
    assert_inside(fig)
    assert_legend_outside_axes(fig)
    assert any(ax.get_xlabel().startswith("X") for ax in fig.axes) and any(ax.get_ylabel() for ax in fig.axes)
    # 2 ustunli bar (random va spatial) + fon nuqtalar uchun alohida o'qlar bor-yo'qligi
    bars = fig.axes[0].patches
    assert len(bars) >= 2 * len(ctx.sm)
    renders(fig, tmp_path)


def test_spatial_with_bg_sensitivity_and_narrow_figure(ctx):
    res = dict(ctx.result)
    res["bg_sensitivity"] = {"summary": {"RandomForest": {"mean": 0.8, "std": 0.05, "min": 0.7, "max": 0.9},
                                         ENSEMBLE_NAME: {"mean": 0.85, "std": 0.03, "min": 0.8, "max": 0.9}},
                             "n_draws": 3}
    fig = new_fig(8, 4.5)
    plots.draw_spatial_diagnostics(fig, res)
    assert_ok(fig)
    assert len(fig.axes) == 3
    assert_inside(fig)


def test_spatial_partial_inputs(ctx):
    only_rand = {"random": {"metrics": ctx.rm}}
    fig = new_fig()
    plots.draw_spatial_diagnostics(fig, only_rand)
    assert_ok(fig)
    only_map = {"spatial": {"fold_map": ctx.result["spatial"]["fold_map"]}, "dataset": ctx.ds}
    fig = new_fig()
    plots.draw_spatial_diagnostics(fig, only_map)
    assert_ok(fig)
    assert len(fig.axes) == 1
    legacy = {"roc_results": ctx.sm, "roc_results_random": ctx.rm}
    fig = new_fig()
    plots.draw_spatial_diagnostics(fig, legacy)
    assert_ok(fig)
    bad_len = {"spatial": {"fold_map": np.arange(5)}, "dataset": ctx.ds, "random": {"metrics": ctx.rm}}
    fig = new_fig()
    plots.draw_spatial_diagnostics(fig, bad_len)          # uzunlik mos emas: fold xaritasi o'rniga izoh
    assert not has_text(fig, ERR_TEXT)
    assert has_text(fig, "teng emas")


def test_spatial_many_folds_use_colorbar(ctx):
    res = dict(ctx.result)
    res["spatial"] = dict(ctx.result["spatial"], fold_map=np.arange(ctx.ds.n) % 25)
    fig = new_fig()
    plots.draw_spatial_diagnostics(fig, res)
    assert_ok(fig)
    assert not [l for l in fig.legends if l.get_title().get_text() == "Validation fold"]
    assert any(ax.get_label() == "<colorbar>" for ax in fig.axes)       # uzluksiz fold colorbar'i


@pytest.mark.parametrize("bad", [None, {}, {"spatial": None, "random": None}, {"dataset": None}])
def test_spatial_empty(bad):
    fig = new_fig()
    plots.draw_spatial_diagnostics(fig, bad)
    assert_no_data(fig)


# ---------------------------------------------------------------------------
# draw_importance (BUG-01)
# ---------------------------------------------------------------------------
def test_importance_real(ctx, tmp_path):
    fig = new_fig(10, 7)
    plots.draw_importance(fig, ctx.result)
    assert_ok(fig)
    n_perm = len(ctx.result["importance"]["models"])
    n_shap = len(ctx.shap) if ctx.shap else 0
    assert len(fig.axes) == n_perm + n_shap
    titles = [ax.get_title() for ax in fig.axes]
    assert sum("Permutation" in t for t in titles) == n_perm
    assert sum("SHAP" in t for t in titles) == n_shap
    perm_ax = fig.axes[0]
    assert any(getattr(c, "errorbar", None) is not None for c in perm_ax.containers)      # error-bar
    if ctx.shap:
        xl = [ax.get_xlabel() for ax in fig.axes if "SHAP" in ax.get_title()]
        assert any("ehtimollik" in x for x in xl) and any("log-odds" in x for x in xl)
    assert "OOF permutation" in fig._suptitle.get_text()
    assert_inside(fig)
    renders(fig, tmp_path)


@pytest.mark.parametrize("shape", ["np", "npc", "list", "cnp"])
def test_importance_shap_shapes(shape):
    sh = manual_shap(shape=shape)
    del sh["mean_abs_shap"]                                 # faqat xom shap_values - normalizatsiya kerak
    res = {"importance": {"method": "m", "models": {}, "feature_names": sh["feature_names"]},
           "shap": {"RandomForest": sh}}
    fig = new_fig()
    plots.draw_importance(fig, res)
    assert_ok(fig)
    assert len(fig.axes) == 1 and "SHAP" in fig.axes[0].get_title()


def test_importance_shap_invalid_and_empty_are_skipped():
    perm = {"RandomForest": {"mean": np.array([0.3, 0.1, -0.05]), "std": np.array([0.02, 0.02, 0.02]),
                             "n_folds": 3, "n_valid_folds": 3}}
    base = {"importance": {"method": "m", "models": perm, "feature_names": ["a", "b", "c"]}}
    for bad_shap in (None, {}, {"RandomForest": {"shap_values": [], "mean_abs_shap": []}},
                     {"RandomForest": {"shap_values": np.empty((0, 3, 2))}},
                     {"RandomForest": {"shap_values": np.zeros((4, 3, 5, 2))}},        # yaroqsiz 4D
                     {"RandomForest": {"mean_abs_shap": np.full(3, np.nan)}},
                     {"RandomForest": "yaroqsiz"}, {"RandomForest": None}):
        fig = new_fig()
        plots.draw_importance(dict(fig=fig)["fig"], dict(base, shap=bad_shap))
        assert_ok(fig)
        assert len(fig.axes) == 1                           # bo'sh SHAP qatori ochilmaydi
        assert "Permutation" in fig.axes[0].get_title()


def test_importance_no_perm_only_shap_and_nothing():
    fig = new_fig()
    plots.draw_importance(fig, {"shap": {"XGBoost": manual_shap(units="log-odds")}})
    assert_ok(fig)
    assert len(fig.axes) == 1
    for bad in (None, {}, {"importance": {"models": {}}, "shap": None},
                {"importance": {"models": {"RF": {"mean": np.full(3, np.nan), "std": np.zeros(3)}}}}):
        f = new_fig()
        plots.draw_importance(f, bad)
        assert_no_data(f)


def test_importance_top_n_for_many_features():
    p = 45
    rng = np.random.default_rng(0)
    perm = {"RandomForest": {"mean": rng.random(p), "std": rng.random(p) * 0.05},
            "XGBoost": {"mean": rng.random(p), "std": None}}
    names = [f"band_{i:02d}" for i in range(p)]
    res = {"importance": {"method": "m", "models": perm, "feature_names": names}}
    fig = new_fig()
    plots.draw_importance(fig, res)
    assert_ok(fig)
    for ax in fig.axes:
        assert len(ax.get_yticks()) == plots.TOP_N
        assert "20/45" in ax.get_title()
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    top = [names[i] for i in np.argsort(perm["RandomForest"]["mean"])[-20:]]
    assert labels == top                                    # eng muhimi tepada (oxirida)
    # 30 tagacha - hammasi ko'rinadi
    p = 28
    res["importance"]["models"] = {"RandomForest": {"mean": rng.random(p), "std": rng.random(p)}}
    res["importance"]["feature_names"] = [f"b{i}" for i in range(p)]
    fig = new_fig()
    plots.draw_importance(fig, res)
    assert len(fig.axes[0].get_yticks()) == 28


def test_importance_feature_name_mismatch_and_cvblock_fallback():
    perm = {"RandomForest": {"mean": np.array([0.3, 0.1]), "std": np.array([0.1, 0.1])}}
    fig = new_fig()
    plots.draw_importance(fig, {"importance": {"models": perm, "feature_names": ["faqat_bitta"]}})
    assert_ok(fig)
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()][-1] == "feature_1"
    fig = new_fig()
    plots.draw_importance(fig, {"spatial": {"perm_importance": perm}, "feature_names": ["a", "b"]})
    assert_ok(fig)


# ---------------------------------------------------------------------------
# SHAP beeswarm / dependence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", ["np", "npc", "list", "cnp"])
def test_beeswarm_shapes(shape, tmp_path):
    res = {"shap": {"RandomForest": manual_shap(shape=shape)}}
    fig = new_fig()
    plots.draw_shap_beeswarm(fig, res, "RandomForest")
    assert_ok(fig)
    ax = fig.axes[0]
    coll = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(coll) == 5                                   # har feature uchun bitta nuqtalar to'plami
    assert len(fig.axes) == 2                               # + colorbar
    assert has_text(fig, "Feature qiymati")
    assert "ehtimollik" in ax.get_xlabel()
    renders(fig, tmp_path)


def test_beeswarm_real(ctx, tmp_path):
    if not ctx.shap:
        pytest.skip("shap o'rnatilmagan")
    for name, unit in (("RandomForest", "ehtimollik"), ("XGBoost", "log-odds")):
        fig = new_fig()
        plots.draw_shap_beeswarm(fig, ctx.result, name)
        assert_ok(fig)
        assert unit in fig.axes[0].get_xlabel() and name in fig.axes[0].get_title()
        yl = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        ma = ctx.shap[name]["mean_abs_shap"]
        assert yl[-1] == ctx.shap[name]["feature_names"][int(np.argmax(ma))]       # eng muhimi tepada
        assert_inside(fig)
        renders(fig, tmp_path, f"b_{name}.png")


def test_beeswarm_variants():
    sh = manual_shap()
    # X_background yo'q -> rangsiz, colorbar yo'q
    nox = dict(sh, X_background=None)
    fig = new_fig()
    plots.draw_shap_beeswarm(fig, {"shap": {"M": nox}}, "M")
    assert_ok(fig)
    assert len(fig.axes) == 1 and has_text(fig, "rang berilmadi")
    # X shakli mos emas -> rangsiz
    fig = new_fig()
    plots.draw_shap_beeswarm(fig, {"shap": {"M": dict(sh, X_background=np.zeros((3, 5)))}}, "M")
    assert_ok(fig)
    # NaN'lar va o'zgarmas ustun
    sv = np.array(sh["shap_values"])
    sv[0, 0] = np.nan
    X = np.array(sh["X_background"])
    X[:, 1] = 1.0
    X[2, 2] = np.nan
    fig = new_fig()
    plots.draw_shap_beeswarm(fig, {"shap": {"M": dict(sh, shap_values=sv, X_background=X)}}, "M")
    assert_ok(fig)
    # entry o'zi, {model: entry} va model=None
    for res in (sh, {"M": sh}):
        fig = new_fig()
        plots.draw_shap_beeswarm(fig, res, None)
        assert_ok(fig)
    # kutilmagan model nomi, bo'sh, (n,p,2) bilan X mos kelmasa ham
    fig = new_fig()
    plots.draw_shap_beeswarm(fig, {"shap": {"M": sh}}, "YO'Q")
    assert_no_data(fig)
    for bad in (None, {}, {"shap": None}, {"shap": {}}, {"shap": {"M": {"shap_values": []}}},
                {"shap": {"M": {"shap_values": np.empty((0, 4))}}}, {"shap": {"M": {"shap_values": np.full((5, 3), np.nan)}}},
                {"shap": {"M": {"shap_values": np.zeros((3, 3, 3, 3))}}}):
        fig = new_fig()
        plots.draw_shap_beeswarm(fig, bad, "M")
        assert_no_data(fig)


def test_beeswarm_many_features_top_n_and_large_n():
    rng = np.random.default_rng(1)
    n, p = 2500, 40
    X = rng.normal(size=(n, p))
    sv = X * np.linspace(1, 0.01, p)
    ent = {"shap_values": np.stack([-sv, sv], -1), "X_background": X, "feature_names": [f"f{i}" for i in range(p)]}
    fig = new_fig(8, 9)
    plots.draw_shap_beeswarm(fig, {"shap": {"RandomForest": ent}}, "RandomForest")
    assert_ok(fig)
    assert len(fig.axes[0].get_yticks()) == plots.TOP_N
    assert has_text(fig, "20/40")


def test_dependence_real_and_manual(ctx, tmp_path):
    ent = manual_shap()
    res = {"shap": {"RandomForest": ent}}
    fig = new_fig()
    plots.draw_shap_dependence(fig, res, "RandomForest", "f1")
    assert_ok(fig)
    ax = fig.axes[0]
    assert "f1" in ax.get_xlabel() and "SHAP" in ax.get_ylabel() and "f1" in ax.get_title()
    pts = next(c for c in ax.collections if isinstance(c, PathCollection))
    assert len(pts.get_offsets()) == 40
    assert len(fig.axes) == 2 and has_text(fig, "Rang:")      # rang bo'yicha eng bog'liq feature
    # indeks bilan, katta-kichik harf farqsiz, rangsiz
    for feat in (2, "F2"):
        fig = new_fig()
        plots.draw_shap_dependence(fig, res, "RandomForest", feat, color_feature=None)
        assert_ok(fig)
        assert len(fig.axes) == 1
    fig = new_fig()
    plots.draw_shap_dependence(fig, res, "RandomForest", "f0", color_feature="f3")
    assert has_text(fig, "Rang: f3")
    fig = new_fig()
    plots.draw_shap_dependence(fig, res, "RandomForest")      # feature=None => eng muhimi
    assert_ok(fig)
    top = ent["feature_names"][int(np.argmax(ent["mean_abs_shap"]))]
    assert fig.axes[0].get_title().endswith(top)
    if ctx.shap:
        for name in ctx.shap:
            fig = new_fig()
            feat = ctx.shap[name]["feature_names"][0]
            plots.draw_shap_dependence(fig, ctx.result, name, feat)
            assert_ok(fig)
            assert_inside(fig)
        renders(fig, tmp_path)


def test_dependence_binary_feature_and_edge_cases():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.integers(0, 2, 30), np.ones(30), rng.normal(size=30)]).astype(float)
    sv = np.column_stack([X[:, 0] * 0.2, rng.normal(size=30) * 0.01, X[:, 2] * 0.1])
    ent = {"shap_values": sv, "X_background": X, "feature_names": ["cat==1", "const", "num"]}
    for feat in ("cat==1", "const", "num"):
        fig = new_fig()
        plots.draw_shap_dependence(fig, {"shap": {"M": ent}}, "M", feat)
        assert_ok(fig)
    # topilmaydi / indeks chegaradan tashqarida / X yo'q / model yo'q / bo'sh
    for feat in ("yo'q", 10, -1):
        fig = new_fig()
        plots.draw_shap_dependence(fig, {"shap": {"M": ent}}, "M", feat)
        assert_no_data(fig)
    fig = new_fig()
    plots.draw_shap_dependence(fig, {"shap": {"M": dict(ent, X_background=None)}}, "M", "num")
    assert_no_data(fig)
    fig = new_fig()
    plots.draw_shap_dependence(fig, {"shap": {"M": ent}}, "BOSHQA", "num")
    assert_no_data(fig)
    for bad in (None, {}, {"shap": None}, {"shap": {"M": {"shap_values": []}}}):
        fig = new_fig()
        plots.draw_shap_dependence(fig, bad, "M", "num")
        assert_no_data(fig)


def test_dependence_all_nan_feature():
    ent = manual_shap()
    X = np.array(ent["X_background"])
    X[:, 0] = np.nan
    fig = new_fig()
    plots.draw_shap_dependence(fig, {"shap": {"M": dict(ent, X_background=X)}}, "M", "f0")
    assert_no_data(fig)


# ---------------------------------------------------------------------------
# draw_corr_heatmap
# ---------------------------------------------------------------------------
def test_corr_real(ctx, tmp_path):
    fig = new_fig()
    plots.draw_corr_heatmap(fig, ctx.corr)
    assert_ok(fig)
    ax = fig.axes[0]
    assert len(ax.get_xticklabels()) == len(ctx.corr)
    renders(fig, tmp_path)


def test_corr_marks_high_pairs():
    names = list("abcd")
    c = np.eye(4)
    c[0, 1] = c[1, 0] = 0.95
    c[2, 3] = c[3, 2] = -0.92
    c[0, 2] = c[2, 0] = 0.5
    df = pd.DataFrame(c, index=names, columns=names)
    fig = new_fig()
    plots.draw_corr_heatmap(fig, df)
    assert_ok(fig)
    ax = fig.axes[0]
    assert len(ax.patches) == 4                             # 2 juft x 2 (simmetrik) katak
    assert "2 ta" in ax.get_title()
    assert fig.legends and fig.legends[0].get_texts()[0].get_text().startswith("|r|")
    assert has_text(fig, "0.95") and has_text(fig, "-0.92")
    # yuqori korrelyatsiya yo'q => ramka/legend yo'q
    fig = new_fig()
    plots.draw_corr_heatmap(fig, pd.DataFrame(np.eye(3) + 0.1 * (1 - np.eye(3)), columns=list("xyz")))
    assert_ok(fig)
    assert not fig.axes[0].patches and not fig.legends


def test_corr_nan_large_and_invalid():
    n = 60
    rng = np.random.default_rng(0)
    a = rng.normal(size=(n, n))
    m = np.corrcoef(a)
    m[3, 4] = np.nan
    fig = new_fig()
    plots.draw_corr_heatmap(fig, pd.DataFrame(m, columns=[f"c{i}" for i in range(n)]))
    assert_ok(fig)
    assert len(fig.axes[0].get_xticks()) <= 40
    fig = new_fig()
    plots.draw_corr_heatmap(fig, np.eye(2))                 # DataFrame bo'lmasa ham
    assert_ok(fig)
    for bad in (None, pd.DataFrame(), pd.DataFrame(np.zeros((2, 3))), pd.DataFrame([["a", "b"], ["c", "d"]]),
                pd.DataFrame(np.full((2, 2), np.nan))):
        fig = new_fig()
        plots.draw_corr_heatmap(fig, bad)
        assert_no_data(fig)


# ---------------------------------------------------------------------------
# draw_map
# ---------------------------------------------------------------------------
def test_map_real_overlays_and_extent(ctx, tmp_path):
    fig = new_fig()
    plots.draw_map(fig, ctx.ens, ctx.raster.transform, "Ansambl xaritasi", "RdYlGn_r", 0, 1, aoi_gdf=ctx.aoi,
                   positives=ctx.pos, background=ctx.bg, cbar_label=plots.INDEX_LABEL)
    assert_ok(fig)
    ax = fig.axes[0]
    t = ctx.raster.transform
    H, W = ctx.ens.shape
    l, r, b, tp = ax.images[0].get_extent()
    assert (l, tp) == (t.c, t.f) and r == pytest.approx(t.c + W * t.a) and b == pytest.approx(t.f + H * t.e)
    assert ax.get_xlim() == pytest.approx((l, r)) and sorted(ax.get_ylim()) == pytest.approx(sorted((b, tp)))
    assert "EPSG:28411" in ax.get_xlabel() and "EPSG:28411" in ax.get_ylabel()
    assert ax.get_title() == "Ansambl xaritasi"
    assert any(isinstance(c, LineCollection) for c in ax.collections)          # AOI kontur
    scat = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert sorted(len(c.get_offsets()) for c in scat) == sorted([len(ctx.pos), len(ctx.bg)])
    assert has_text(fig, plots.INDEX_LABEL)
    labs = [t.get_text() for lg in fig.legends for t in lg.get_texts()]
    assert any("AOI" in x for x in labs) and any("Fon nuqtalar" in x for x in labs) and any("konlar" in x for x in labs)
    assert_legend_outside_axes(fig)
    assert_inside(fig)
    # nodata (NaN) shaffof: maskalangan, "bad" rang alpha=0
    im = ax.images[0]
    arr = im.get_array()
    assert np.ma.is_masked(arr) and arr.mask.sum() == int((~np.isfinite(ctx.ens)).sum())
    assert im.get_cmap()(np.ma.masked)[3] == 0.0
    renders(fig, tmp_path)


def test_map_class_labels_discrete_colorbar(ctx, tmp_path):
    labels = predict.CLASS_LABELS(5)
    fig = new_fig()
    plots.draw_map(fig, ctx.class_map, ctx.raster.transform, "Sinflar", "RdYlGn_r", 0, 1, class_labels=labels,
                   cbar_label="Sinf")
    assert_ok(fig)
    cax = fig.axes[1]
    assert [t.get_text() for t in cax.get_yticklabels()] == labels
    arr = fig.axes[0].images[0].get_array()
    assert arr.mask[ctx.class_map == 0].all() and not arr.mask[ctx.class_map > 0].any()    # 0 = nodata shaffof
    from matplotlib.colors import BoundaryNorm
    assert isinstance(fig.axes[0].images[0].norm, BoundaryNorm)
    renders(fig, tmp_path)


def test_map_variants_and_errors():
    arr = np.random.default_rng(0).random((20, 30))
    arr[:3] = np.nan
    t = from_origin(11_500_000.0, 4_600_000.0, 100.0, 100.0)
    # transform yo'q => piksel koordinatalari
    fig = new_fig()
    plots.draw_map(fig, arr, None, "t")
    assert_ok(fig)
    assert fig.axes[0].get_xlabel().startswith("Ustun")
    # 6 elementli kortej, numpy int massiv, nuqtalar massiv/DataFrame/GeoSeries shaklida
    fig = new_fig()
    plots.draw_map(fig, arr, tuple(t)[:6], "t", positives=np.array([[11_501_000.0, 4_599_000.0]]),
                   background=pd.DataFrame({"x": [11_502_000.0], "y": [4_598_000.0]}),
                   aoi_gdf=gpd.GeoSeries([MultiPolygon([box(11_500_100, 4_598_100, 11_500_900, 4_598_900),
                                                       box(11_501_100, 4_598_100, 11_501_900, 4_598_900)])]))
    assert_ok(fig)
    assert len([c for c in fig.axes[0].collections if isinstance(c, PathCollection)]) == 2
    # bo'sh nuqtalar/AOI - xato emas, legend yo'q
    fig = new_fig()
    plots.draw_map(fig, arr, t, "t", aoi_gdf=gpd.GeoDataFrame(geometry=[], crs="EPSG:28411"),
                   positives=gpd.GeoDataFrame(geometry=[], crs="EPSG:28411"), background=[])
    assert_ok(fig)
    assert not fig.legends
    # vmin/vmax va rang xaritasi obyekti
    from matplotlib import colormaps
    fig = new_fig()
    plots.draw_map(fig, arr, t, "t", colormaps["viridis"], 0.2, 0.8, cbar_label="x")
    assert_ok(fig)
    assert fig.axes[0].images[0].get_clim() == (0.2, 0.8)
    # int sinflar, class_labels bo'lsa NaN ham 0 kabi
    cm = np.array([[0, 1, 2], [3, 3, 0]], dtype=np.int8)
    fig = new_fig()
    plots.draw_map(fig, cm, t, "t", class_labels=["a", "b", "c"])
    assert_ok(fig)
    for bad in (None, np.empty((0, 0)), np.full((4, 4), np.nan), np.zeros((4, 4)) - 1, np.zeros(5), np.zeros((2, 2, 2))):
        fig = new_fig()
        plots.draw_map(fig, bad, t, "t", class_labels=["a"] if (isinstance(bad, np.ndarray) and bad.size and bad.min() < 0) else None)
        assert_no_data(fig)


# ---------------------------------------------------------------------------
# draw_success_rate
# ---------------------------------------------------------------------------
def test_success_rate_real(ctx, tmp_path):
    fig = new_fig()
    plots.draw_success_rate(fig, ctx.curve)
    assert_ok(fig)
    ax = fig.axes[0]
    leg = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(f"{ctx.curve['auc']:.3f}" in l for l in leg) and any("Tasodifiy" in l for l in leg)
    line = ax.lines[0]
    np.testing.assert_allclose(line.get_xdata(), ctx.curve["area_frac"] * 100)
    np.testing.assert_allclose(line.get_ydata(), ctx.curve["capture_frac"] * 100)
    assert "prospektivlik" in ax.get_xlabel().lower() and "kon" in ax.get_ylabel().lower()
    renders(fig, tmp_path)


def test_success_rate_multi_and_empty():
    c = {"area_frac": np.array([0, 0.1, 1.0]), "capture_frac": np.array([0, 0.6, 1.0]), "auc": 0.8, "n_pos": 9}
    fig = new_fig()
    plots.draw_success_rate(fig, {"RandomForest": c, ENSEMBLE_NAME: c})
    assert_ok(fig)
    assert len([l for l in fig.axes[0].lines]) == 3         # 2 egri chiziq + diagonal
    empty = {"area_frac": np.empty(0), "capture_frac": np.empty(0), "auc": float("nan"), "n_pos": 0}
    for bad in (None, {}, empty, {"X": empty},
                {"area_frac": np.array([0, np.nan]), "capture_frac": np.array([0, 1]), "auc": 1.0}):
        fig = new_fig()
        plots.draw_success_rate(fig, bad)
        assert_no_data(fig)
    fig = new_fig()                                          # auc NaN, n_pos yo'q - baribir chiziladi
    plots.draw_success_rate(fig, {"area_frac": [0, 1], "capture_frac": [0, 1], "auc": float("nan")})
    assert_ok(fig)


# ---------------------------------------------------------------------------
# draw_tuning_trials
# ---------------------------------------------------------------------------
def test_tuning_real(tuned, tmp_path):
    fig = new_fig(10, 8)
    plots.draw_tuning_trials(fig, tuned)
    assert_ok(fig)
    titles = [ax.get_title() for ax in fig.axes]
    assert has_text(fig, "RandomForest") and has_text(fig, "SVM")          # model nomi - qator yorlig'i
    assert not any("RandomForest" in t or "SVM" in t for t in titles)      # panel sarlavhasida takrorlanmaydi
    assert any("roc_auc" in t for t in titles)                # har model uchun ball paneli
    assert any(t.endswith("C") for t in titles)               # SVM.C o'zgargan parametr
    assert "Giperparametr" in fig._suptitle.get_text()
    assert_inside(fig)
    renders(fig, tmp_path)


def _rec(fold, best, trials=True, **extra):
    r = {"repeat": 0, "fold": fold, "best_params": best, "best_score": 0.8, "base_score": 0.7,
         "trials": [{"params": dict(best, max_depth=d), "score": 0.7, "std": 0.01} for d in (3, None, 8)] if trials else [],
         "n_trials": 3 if trials else 0, "scoring": "roc_auc", "best_index": 1, "inner_splits_used": 2}
    r.update(extra)
    return r


def test_tuning_fallback_and_empty_trials():
    recs = [_rec(0, {"max_depth": 3, "criterion": "gini", "bootstrap": True}),
            _rec(1, {"max_depth": None, "criterion": "entropy", "bootstrap": False}),
            {"repeat": 0, "fold": 2, "best_params": {"max_depth": 5, "criterion": "gini", "bootstrap": True},
             "best_score": float("nan"), "trials": [], "n_trials": 0, "fallback": "ichki bo'linish imkonsiz"}]
    fig = new_fig()
    plots.draw_tuning_trials(fig, {"RandomForest": recs})
    assert_ok(fig)
    assert has_text(fig, "1/3 fold zaxira")
    assert any(ax.get_title().endswith("max_depth") for ax in fig.axes)
    assert any(ax.get_title().endswith("criterion") for ax in fig.axes)
    assert any(ax.get_title().endswith("bootstrap") for ax in fig.axes)
    # barcha yozuvlar fallback va trials=[] (o'zgarish yo'q) => paneli bor, xato yo'q
    allfb = [{"repeat": 0, "fold": i, "best_params": {"C": 1.0}, "best_score": float("nan"), "trials": [],
              "n_trials": 0, "skipped_reason": "kam musbat"} for i in range(3)]
    fig = new_fig()
    plots.draw_tuning_trials(fig, {"SVM": allfb})
    assert_ok(fig)
    assert has_text(fig, "o'zgarmadi")
    # fallback fold'lari bo'sh trials va farqli best_params
    fb2 = [dict(r, best_params={"C": float(10 ** i)}) for i, r in enumerate(allfb)]
    fig = new_fig()
    plots.draw_tuning_trials(fig, {"SVM": fb2})
    assert_ok(fig)
    assert any(ax.get_title().endswith("C") for ax in fig.axes)


def test_tuning_variants():
    r1 = _rec(0, {"max_depth": 3, "C": 0.01})
    r2 = _rec(1, {"max_depth": 8, "C": 100.0}, repeat=1)
    fig = new_fig()
    plots.draw_tuning_trials(fig, {"RandomForest": [r2, r1]})            # tartiblanmagan, ko'p repeat
    assert_ok(fig)
    assert has_text(fig, "Takror.fold")
    fig = new_fig()
    plots.draw_tuning_trials(fig, {"RandomForest": r1})                  # bitta yozuv (lug'at)
    assert_ok(fig)
    fig = new_fig()
    plots.draw_tuning_trials(fig, {"RandomForest": [r1], "CNN": []})     # bo'sh model e'tiborsiz
    assert_ok(fig)
    many = [_rec(i, {"max_depth": 2 + i % 5}) for i in range(20)]       # 20 fold: tick siyraklashadi
    fig = new_fig()
    plots.draw_tuning_trials(fig, {"RandomForest": many})
    assert_ok(fig)
    assert_inside(fig)
    for bad in (None, {}, {"RandomForest": []}, {"RandomForest": None}, {"RandomForest": [None, 3]},
                {"RandomForest": {"max_depth": 3}}):
        fig = new_fig()
        plots.draw_tuning_trials(fig, bad)
        if bad == {"RandomForest": {"max_depth": 3}}:
            assert not has_text(fig, ERR_TEXT)                           # tekis lug'at - yiqilmaydi
        else:
            assert_no_data(fig)


# ---------------------------------------------------------------------------
# save_all_figures
# ---------------------------------------------------------------------------
def test_save_all_figures(ctx, tmp_path):
    f1, f2 = new_fig(5, 4), new_fig(5, 4)
    plots.draw_roc(f1, ctx.sm)
    plots.draw_success_rate(f2, ctx.curve)
    out = tmp_path / "ichki" / "papka"                       # mavjud emas - yaratiladi
    paths = plots.save_all_figures({"ROC (spatial)": f1, "Ensemble (soft-voting) xarita": f2}, str(out), dpi=50)
    assert os.path.isdir(out)
    names = sorted(os.path.basename(p) for p in paths)
    s1, s2 = safe_name("ROC (spatial)"), safe_name("Ensemble (soft-voting) xarita")
    assert names == sorted([f"{s1}.png", f"{s1}.pdf", f"{s2}.png", f"{s2}.pdf"])
    assert all(os.path.getsize(p) > 500 for p in paths)
    with open([p for p in paths if p.endswith(".pdf")][0], "rb") as fh:
        assert fh.read(4) == b"%PDF"
    with open([p for p in paths if p.endswith(".png")][0], "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_save_all_figures_options_and_robustness(tmp_path):
    f = new_fig(5, 4)
    plots.draw_roc(f, manual_metrics())
    logs = []
    # nuqtali/katta formatlar, takror, bitta format matn sifatida; xavfli belgilar; takror nom (safe_name to'qnashuvi)
    paths = plots.save_all_figures({"a b": f, "a/b": f, "Bad": None, "str": "emas"}, str(tmp_path), formats=".PNG",
                                   dpi=40, log_fn=logs.append)
    assert sorted(os.path.basename(p) for p in paths) == ["a_b.png", "a_b_2.png"]
    assert len(logs) == 2 and all("o'tkazib" in l for l in logs)
    paths = plots.save_all_figures({"x": f}, str(tmp_path), formats=("png", "png", "svg"), dpi=40)
    assert sorted(os.path.basename(p) for p in paths) == ["x.png", "x.svg"]
    # noma'lum format - qolganlari saqlanadi, xato log orqali
    logs.clear()
    paths = plots.save_all_figures({"y": f}, str(tmp_path), formats=("png", "xyz"), dpi=40, log_fn=logs.append)
    assert [os.path.basename(p) for p in paths] == ["y.png"] and logs
    assert plots.save_all_figures({}, str(tmp_path / "bosh")) == []
    assert plots.save_all_figures(None, str(tmp_path / "bosh")) == []
    assert not (tmp_path / "bosh").exists()
    assert plots.save_all_figures({"": f}, str(tmp_path), formats=("png",), dpi=40)[0].endswith("figure.png")


# ---------------------------------------------------------------------------
# Reviewer regressiyalari (haqiqiy cv/pipeline natijalari shakliga mos)
# ---------------------------------------------------------------------------
def _wide_tuned(n_models=3, n_params=8, n_folds=3):
    """Haqiqiy (RF/SVM/XGBoost) nested tuning'ga o'xshash: har modelda ko'p o'zgargan parametr."""
    out = {}
    for m in range(n_models):
        recs = []
        for f in range(n_folds):
            best = {f"p{k}": float(f + k + 1) for k in range(n_params)}
            trials = [{"params": {kk: vv + d for kk, vv in best.items()}, "score": 0.7, "std": 0.01} for d in (1, 2)]
            recs.append({"repeat": 0, "fold": f, "best_params": best, "best_score": 0.8, "base_score": 0.75,
                         "trials": trials, "n_trials": 2, "scoring": "roc_auc", "best_index": 0,
                         "inner_splits_used": 2})
        out[f"Model{m}"] = recs
    return out


def test_tuning_many_panels_fit_small_canvas(tmp_path):
    """3 model x 8 parametr oddiy GUI o'lchamida (8x6) panellar yig'ilib qolmasin: kesilgan parametrlar izohda,
    constrained layout 'axes collapsed' ogohlantirishisiz; katta figurada hammasi ko'rsatiladi."""
    tp = _wide_tuned()
    fig = new_fig(8, 6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plots.draw_tuning_trials(fig, tp)
        fig.canvas.draw()
    assert_ok(fig)
    fig_h = fig.get_figheight()
    assert all(ax.get_position().height * fig_h >= 0.6 for ax in fig.axes), \
        [ax.get_position().height * fig_h for ax in fig.axes]
    assert has_text(fig, "ko'rsatilmadi")
    assert_inside(fig)
    big = new_fig(12, 12)
    plots.draw_tuning_trials(big, tp)
    assert_ok(big)
    assert len([ax for ax in big.axes if ax.get_title()]) == 3 * (1 + 8)        # + 3 ta model yorlig'i axes'i
    assert not has_text(big, "ko'rsatilmadi")


def test_titles_and_footnotes_fit_narrow_canvas():
    """MplCanvas boshlang'ich o'lchami 6x4.5: uzun importance usuli/sarlavha/izoh figura chegarasidan chiqmasin."""
    rng = np.random.default_rng(0)
    pm = lambda: {"mean": rng.normal(size=5), "std": np.ones(5) * 0.1, "n_folds": 3, "n_valid_folds": 3}  # noqa: E731
    res = {"importance": {"method": "out-of-fold permutation importance (spatial CV, AUC pasayishi)",
                          "models": {"RandomForest": pm(), "SVM": pm(), "XGBoost": pm()},
                          "feature_names": list("abcde")},
           "shap": {"RandomForest": {"mean_abs_shap": np.arange(5.0), "units": "ehtimollik"},
                    "XGBoost": {"mean_abs_shap": np.arange(5.0), "units": "log-odds"}}}
    fig = new_fig(6, 4.5)
    plots.draw_importance(fig, res)
    assert_ok(fig)
    fig.canvas.draw()
    r, fb = fig.canvas.get_renderer(), fig.bbox
    arts = list(fig.texts) + ([fig._suptitle] if fig._suptitle is not None else [])
    assert arts
    for a in arts:
        bb = a.get_window_extent(r)
        assert bb.x0 >= -2 and bb.x1 <= fb.x1 + 2, (a.get_text()[:40], bb, fb)
    for ax in fig.axes:
        bb = ax.get_tightbbox(r)
        assert bb.x0 >= -2 and bb.x1 <= fb.x1 + 2, (ax.get_title(), bb, fb)


def test_review_hardening_nan_std_trials_npos():
    rng = np.random.default_rng(1)
    sv = rng.normal(size=(30, 5))
    sv[:, 2] = np.nan                                         # butunlay NaN ustun
    entry = {"shap_values": sv, "X_background": rng.normal(size=(30, 5)), "feature_names": list("abcde")}
    with warnings.catch_warnings():
        warnings.simplefilter("error")                        # nanmean 'empty slice' ogohlantirishi bo'lmasin
        fig = new_fig()
        plots.draw_shap_dependence(fig, {"shap": {"RandomForest": entry}}, "RandomForest")
        assert_ok(fig)
        fig = new_fig()
        plots.draw_shap_dependence(fig, {"shap": {"RandomForest": dict(entry, shap_values=np.full((30, 5), np.nan))}})
        assert_no_data(fig)
    imp = {"importance": {"models": {"RF": {"mean": [1.0, 2.0, 3.0], "std": [-0.1, 0.2, 0.1], "n_folds": 3}}}}
    fig = new_fig()
    plots.draw_importance(fig, imp)                           # manfiy std (xato kirish) - barh yiqilmasin
    assert_ok(fig)
    for npos in (float("nan"), "abc", None):
        fig = new_fig()
        plots.draw_success_rate(fig, {"area_frac": [0, 1], "capture_frac": [0, 1], "auc": 0.5, "n_pos": npos})
        assert_ok(fig)
    rec = _rec(0, {"max_depth": 3}, trials=True)
    rec["trials"] = [None, "x", {"params": None}] + rec["trials"]   # buzilgan nomzodlar e'tiborsiz
    fig = new_fig()
    plots.draw_tuning_trials(fig, {"RandomForest": [rec, _rec(1, {"max_depth": 8})]})
    assert_ok(fig)


def test_no_forbidden_imports():
    import ast
    with open(plots.__file__, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(("." * node.level) + (node.module or ""))
            mods += [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print":
            pytest.fail("print ishlatilmasin")
    bad = [m for m in mods if m.split(".")[0] in ("PyQt5", "tensorflow", "shap", "xgboost") or m == "matplotlib.pyplot"]
    assert not bad, bad


# ---------------------------------------------------------------------------
# Yakuniy mustahkamlash regressiyalari (importance MDI, xarita, tuning sarlavhalari, std/CI legend)
# ---------------------------------------------------------------------------
def _imp_res(sources, names=None, p=6):
    rng = np.random.default_rng(0)
    models = {}
    for k, src in sources.items():
        models[k] = {"mean": rng.random(p), "std": np.zeros(p) if src == "mdi" else rng.random(p) * 0.05 + 0.01,
                     "n_folds": 3, "n_valid_folds": 3, "source": src}
    return {"importance": {"method": "m", "models": models,
                           "feature_names": names if names is not None else [f"b{i}" for i in range(p)]}}


def test_importance_mdi_source_title_label_no_errorbars():
    res = _imp_res({"RandomForest": "permutation", "XGBoost": "mdi"})
    fig = new_fig(10, 6)
    plots.draw_importance(fig, res)
    assert_ok(fig)
    perm_ax, mdi_ax = fig.axes[0], fig.axes[1]
    assert "Permutation importance" in perm_ax.get_title() and "ΔAUC" in perm_ax.get_xlabel()
    assert any(getattr(c, "errorbar", None) is not None for c in perm_ax.containers)
    assert "MDI/gain importance (zaxira)" in mdi_ax.get_title()
    assert "Permutation" not in mdi_ax.get_title()
    assert "ΔAUC" not in mdi_ax.get_xlabel() and "MDI" in mdi_ax.get_xlabel()
    assert not any(getattr(c, "errorbar", None) is not None for c in mdi_ax.containers)    # std nol - xato chizig'i yo'q


def test_importance_mdi_top_n_suffix_and_no_relabel_hack():
    """draw_importance 'source' ni o'zi hisobga oladi: result_tabs'dagi vaqtinchalik _relabel_mdi hack'i olib tashlandi."""
    res = _imp_res({"XGBoost": "mdi"}, p=45)
    fig = new_fig()
    plots.draw_importance(fig, res)
    assert "MDI/gain importance (zaxira)" in fig.axes[0].get_title() and "20/45" in fig.axes[0].get_title()
    assert "MDI" in fig.axes[0].get_xlabel()
    pytest.importorskip("PyQt5")
    from mpm.gui import result_tabs
    assert not hasattr(result_tabs, "_relabel_mdi")


def test_importance_numpy_feature_names():
    names = np.array([f"band_{i}" for i in range(6)])
    res = _imp_res({"RandomForest": "permutation"}, names=names)
    res["shap"] = {"RandomForest": dict(manual_shap(p=6), feature_names=np.array([f"s{i}" for i in range(6)]))}
    fig = new_fig()
    plots.draw_importance(fig, res)
    assert_ok(fig)
    assert len(fig.axes) == 2
    assert {t.get_text() for t in fig.axes[0].get_yticklabels()} == set(names)
    assert {t.get_text() for t in fig.axes[1].get_yticklabels()} == {f"s{i}" for i in range(6)}
    res2 = {"feature_names": names, "importance": {"models": res["importance"]["models"], "feature_names": np.array([])}}
    fig = new_fig()
    plots.draw_importance(fig, res2)
    assert_ok(fig)
    fig = new_fig()
    sh = dict(manual_shap(p=6), feature_names=np.array(list("abcdef")))
    plots.draw_shap_beeswarm(fig, {"shap": {"RandomForest": sh}}, "RandomForest")
    assert_ok(fig)


def _group_extent(fig):
    """map axes + colorbar guruhining (tick yorliqlarsiz, axes qutilari) gorizontal chegaralari, figura nisbatida."""
    fig.canvas.draw()
    boxes = [ax.get_position() for ax in fig.axes]
    return min(b.x0 for b in boxes), max(b.x1 for b in boxes)


@pytest.mark.parametrize("canvas", [(6, 6), (11.6, 6), (4, 7), (16, 4.5)])
@pytest.mark.parametrize("shape", [(100, 100), (60, 120), (140, 60)])
def test_map_group_centered_on_any_canvas(canvas, shape):
    arr = np.random.default_rng(0).random(shape)
    t = from_origin(11_500_000.0, 4_600_000.0 + shape[0] * 100, 100.0, 100.0)
    fig = new_fig(*canvas)
    plots.draw_map(fig, arr, t, "Xarita", cbar_label=plots.INDEX_LABEL)
    assert_ok(fig)
    ax, cax = fig.axes[0], fig.axes[1]
    fig.canvas.draw()
    a, c = ax.get_position(), cax.get_position()
    fw, fh = fig.get_size_inches()
    assert a.width * fw / (a.height * fh) == pytest.approx(shape[1] / shape[0], rel=0.02)      # equal-aspect saqlanadi
    assert c.x0 >= a.x1 - 1e-6 and (c.x0 - a.x1) * fw < 0.6                                  # colorbar axes'ga yopishgan
    assert c.height == pytest.approx(a.height, rel=0.05)                                     # va balandligi teng
    x0, x1 = _group_extent(fig)
    left, right = x0, 1.0 - x1                                                                 # chap/o'ng bo'sh joy
    assert abs(left - right) * fw < 0.9, (canvas, shape, left * fw, right * fw)               # markazlashgan (kvadrat tick yorliqlari farqi)
    assert_inside(fig)


def test_map_wide_canvas_no_big_left_gap():
    """Regressiya: 11.6x6 da xarita+colorbar o'ngga siljib, chapda katta bo'sh joy qolardi."""
    arr = np.random.default_rng(0).random((100, 100))
    t = from_origin(11_500_000.0, 4_610_000.0, 100.0, 100.0)
    fig = new_fig(11.6, 6)
    plots.draw_map(fig, arr, t, "Xarita", cbar_label="x")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tb = [ax.get_tightbbox(r) for ax in fig.axes]
    left = min(b.x0 for b in tb)
    right = fig.bbox.x1 - max(b.x1 for b in tb)
    assert abs(left - right) < 0.6 * fig.dpi, (left, right)
    assert left > 0.5 * fig.dpi                                  # yon bo'sh joylar bor (xarita cho'zilmaydi)


def test_map_huge_array_is_strided_and_fast():
    import time
    big = np.zeros((4000, 4000), dtype=np.float32)
    big[::3] = 0.5
    big[:100] = np.nan
    t = from_origin(11_500_000.0, 4_640_000.0, 10.0, 10.0)
    fig = new_fig()
    t0 = time.perf_counter()
    plots.draw_map(fig, big, t, "Katta", cbar_label="x")
    fig.canvas.draw()
    assert time.perf_counter() - t0 < 20
    assert_ok(fig)
    im = fig.axes[0].images[0]
    assert max(im.get_array().shape) <= plots.MAP_MAX_PX
    l, r, b, tp = im.get_extent()                               # extent to'liq raster bo'yicha qoladi
    assert (l, tp) == (t.c, t.f) and r == pytest.approx(t.c + 4000 * 10.0) and b == pytest.approx(t.f - 4000 * 10.0)
    # stride'da yo'qolib qoladigan siyrak yaroqli piksel: "Ma'lumot yo'q" EMAS
    sp = np.full((5000, 5000), np.nan)
    sp[1, 1] = 0.7
    fig = new_fig()
    plots.draw_map(fig, sp, None, "t")
    assert_ok(fig)


def test_tuning_titles_are_param_names_and_model_in_row_label():
    recs = []
    for f in range(3):
        best = {"n_estimators": [100, 300, 500][f], "min_samples_leaf": [1, 2, 4][f], "max_features": ["sqrt", "log2", "sqrt"][f]}
        recs.append({"repeat": 0, "fold": f, "best_params": best, "best_score": 0.8, "base_score": 0.7, "scoring": "roc_auc",
                     "n_trials": 2, "trials": [{"params": dict(best, n_estimators=100 + k), "score": 0.7} for k in range(2)]})
    fb = dict(recs[2], fallback="zaxira")
    tp = {"RandomForest": recs[:2] + [fb], "XGBoost": recs}
    for size in ((8, 6), (6, 8), (11, 7)):
        fig = new_fig(*size)
        plots.draw_tuning_trials(fig, tp)
        assert_ok(fig)
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert "n_estimators" in titles and "min_samples_leaf" in titles and "max_features" in titles
        assert not any("RandomForest" in t or "XGBoost" in t or "·" in t for t in titles)
        assert has_text(fig, "RandomForest") and has_text(fig, "XGBoost")
        assert has_text(fig, "1/3 fold zaxira")
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for ax in fig.axes:
            if not ax.get_title():
                continue
            tb = ax.title.get_window_extent(r)
            ab = ax.get_window_extent(r)
            assert tb.x0 >= ab.x0 - 0.6 * fig.dpi / 10 and tb.x1 <= ab.x1 + 0.6 * fig.dpi / 10, (size, ax.get_title())
        assert_inside(fig)


def test_legends_omit_undefined_std_and_ci():
    m = manual_metrics()
    for v in m.values():
        v["auc_std"] = float("nan")
        v["pr_auc_std"] = float("nan")
        v["pr_auc_ci95"] = (float("nan"), float("nan"))
    fig = new_fig()
    plots.draw_roc(fig, m)
    labs = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert all("±" not in x and "0.000" not in x for x in labs), labs
    assert any("95% CI 0.700–0.900" in x for x in labs)                    # CI bor - ko'rsatiladi
    assert "±" not in fig.axes[0].get_legend().get_title().get_text()
    fig = new_fig()
    plots.draw_pr(fig, m, None)
    labs = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert all("±" not in x and "CI" not in x for x in labs), labs
    # std va CI ikkalasi NaN: legend sarlavhasi ham yo'q
    m2 = manual_metrics(with_ci=False)
    for v in m2.values():
        v["auc_std"] = float("nan")
    fig = new_fig()
    plots.draw_roc(fig, m2)
    lg = fig.axes[0].get_legend()
    assert all("±" not in t.get_text() and "CI" not in t.get_text() for t in lg.get_texts())
    assert lg.get_title().get_text() == ""
    # std chekli bo'lsa avvalgidek
    fig = new_fig()
    plots.draw_roc(fig, manual_metrics())
    assert any("± 0.010" in t.get_text() for t in fig.axes[0].get_legend().get_texts())


def test_single_repeat_std_not_shown_even_if_zero():
    """cv bitta takrorda std=0.0 qaytarishi mumkin (auc_repeats uzunligi 1): "± 0.000" yozilmasin."""
    m = manual_metrics()
    for v in m.values():
        v["auc_std"] = 0.0
        v["auc_repeats"] = np.array([0.8])
    fig = new_fig()
    plots.draw_roc(fig, m)
    assert all("±" not in t.get_text() for t in fig.axes[0].get_legend().get_texts())
    for v in m.values():
        v["auc_repeats"] = np.array([0.8, 0.82])
        v["auc_std"] = 0.014
    fig = new_fig()
    plots.draw_roc(fig, m)
    assert any("± 0.014" in t.get_text() for t in fig.axes[0].get_legend().get_texts())


def test_spatial_bars_no_errorbar_when_std_and_ci_undefined():
    nan = float("nan")
    sm = {"A": {"auc": 0.8, "auc_std": nan, "auc_ci95": (nan, nan)}}
    res = {"spatial": {"metrics": sm}, "bg_sensitivity": {"n_draws": 1, "summary": {
        "A": {"mean": 0.8, "std": 0.0, "min": 0.78, "max": 0.82}}}}
    fig = new_fig(10, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plots.draw_spatial_diagnostics(fig, res)
        fig.canvas.draw()
    assert_ok(fig)
    bars_ax, bg_ax = fig.axes[0], fig.axes[1]
    assert not any(getattr(c, "errorbar", None) is not None for c in bars_ax.containers if hasattr(c, "errorbar")) \
        or all(c.errorbar is None for c in bars_ax.containers if hasattr(c, "errorbar"))
    assert not any(isinstance(c, type(bars_ax.containers[0])) and getattr(c, "errorbar", None) for c in bg_ax.containers)


# ---------------------------------------------------------------------------
# e2e tuzatishlari: korrelyatsiya heatmap markazlashishi, chalkashlik matritsasi yorliqlari
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("canvas", [(11.6, 6), (16, 4.5), (6, 6)])
def test_corr_heatmap_group_centered_on_wide_canvas(canvas):
    """Regressiya: keng canvasda kvadrat heatmap + colorbar o'ngga siljib, chapda katta bo'sh joy qolardi."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 7))
    corr = pd.DataFrame(np.corrcoef(X.T), columns=[f"layer{i}" for i in range(7)])
    fig = new_fig(*canvas)
    plots.draw_corr_heatmap(fig, corr)
    assert_ok(fig)
    fw, _fh = fig.get_size_inches()
    x0, x1 = _group_extent(fig)
    assert abs(x0 - (1.0 - x1)) * fw < 1.2, (canvas, x0 * fw, (1.0 - x1) * fw)       # tick yorliqlari farqi ichida


def test_confusion_tick_labels_are_two_lines_and_do_not_overlap():
    """Regressiya: tor canvasda "Bashorat: fon (0)" va "Bashorat: musbat (1)" yorliqlari bir-birining ustiga tushardi."""
    ms = {"RandomForest": {"confusion": [[71, 9], [3, 25]], "sensitivity": 0.89, "specificity": 0.89,
                           "threshold_youden": 0.3}}
    fig = new_fig(6, 4)
    plots.draw_confusion(fig, ms, "RandomForest")
    assert_ok(fig)
    ax = fig.axes[0]
    xt = [t.get_text() for t in ax.get_xticklabels()]
    assert all("\n" in t for t in xt) and all("\n" in t.get_text() for t in ax.get_yticklabels())
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    b0, b1 = [t.get_window_extent(r) for t in ax.get_xticklabels()]
    assert b0.x1 <= b1.x0 + 1.0                                                      # yonma-yon yorliqlar kesishmaydi
