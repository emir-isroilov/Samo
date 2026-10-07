# -*- coding: utf-8 -*-
"""mpm.workers testlari: run() sinxron chaqirilib signal'lar yig'iladi (TrainingWorker, PredictionWorker,
ApplyBundleWorker, ExportWorker), xato (traceback) va cancel."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QCoreApplication  # noqa: E402

from mpm import persist, workers  # noqa: E402
from mpm.common import ENSEMBLE_NAME  # noqa: E402
from mpm.config import RunConfig, default_hyperparams  # noqa: E402
from tests.synth import make_synthetic_project  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


def collect(worker):
    """Barcha signal'larni ro'yxatlarga yig'adi."""
    got = SimpleNamespace(log=[], progress=[], finished=[], error=[], cancelled=[])
    worker.log_signal.connect(got.log.append)
    worker.progress_signal.connect(lambda p, m: got.progress.append((p, m)))
    worker.finished_signal.connect(got.finished.append)
    worker.error_signal.connect(got.error.append)
    worker.cancelled_signal.connect(lambda: got.cancelled.append(True))
    return got


def small_cfg(proj, out_dir, **kw):
    hp = default_hyperparams()
    hp["RandomForest"]["n_estimators"] = 15
    base = dict(tiff_folder=proj["tiff"], points_folder=proj["points"], aoi_folder=proj["aoi"],
                output_dir=str(out_dir), n_background=60, min_distance=300.0, n_splits=3, n_repeats=1,
                n_bootstrap=20, hyperparams=hp, run_random_cv=False, shap_enabled=False, perm_importance=False,
                n_jobs=1, seed=5, use_models={"RandomForest": True, "SVM": True, "XGBoost": False, "CNN": False})
    base.update(kw)
    return RunConfig(**base)


@pytest.fixture(scope="module")
def proj(tmp_path_factory):
    return make_synthetic_project(str(tmp_path_factory.mktemp("wk_proj")), size=80, n_layers=4, n_pos=40)


@pytest.fixture(scope="module")
def trained(proj, tmp_path_factory):
    out = tmp_path_factory.mktemp("wk_out")
    w = workers.TrainingWorker(small_cfg(proj, out))
    got = collect(w)
    w.run()                                                     # sinxron
    return SimpleNamespace(worker=w, got=got, out=out, proj=proj)


# ---------------------------------------------------------------------------
# TrainingWorker
# ---------------------------------------------------------------------------
def test_training_worker_sync(trained):
    g = trained.got
    assert not g.error and not g.cancelled and len(g.finished) == 1
    res = g.finished[0]
    assert isinstance(res, dict) and {"spatial", "final_models", "dataset", "raster"} <= set(res)
    assert res["raster"].stack.shape[0] == 4                    # katta ob'ekt nusxalanmasdan uzatiladi
    assert g.log and all(isinstance(m, str) for m in g.log)
    assert any("Spatial" in m or "SPATIAL" in m for m in g.log)
    pct = [p for p, _ in g.progress]
    assert pct[0] == 0 and pct[-1] == 100 and all(isinstance(p, int) for p in pct)
    assert all(b >= a for a, b in zip(pct, pct[1:])) and all(isinstance(m, str) for _, m in g.progress)


def test_signal_signatures():
    for cls in (workers.TrainingWorker, workers.PredictionWorker, workers.ApplyBundleWorker, workers.ExportWorker):
        for name in ("log_signal", "progress_signal", "finished_signal", "error_signal", "cancelled_signal"):
            assert hasattr(cls, name), (cls, name)
        assert callable(cls.cancel)


def test_training_worker_error_has_traceback(proj, tmp_path):
    w = workers.TrainingWorker(small_cfg({**proj, "tiff": str(tmp_path / "yo'q")}, tmp_path))
    g = collect(w)
    w.run()
    assert len(g.error) == 1 and not g.finished and not g.cancelled
    assert "TIFF" in g.error[0] and "Traceback" in g.error[0] and "ValueError" in g.error[0]


def test_training_worker_invalid_config_error(proj, tmp_path):
    w = workers.TrainingWorker(small_cfg(proj, tmp_path, n_splits=1))
    g = collect(w)
    w.run()
    assert len(g.error) == 1 and "K-fold" in g.error[0] and not g.finished


def test_training_worker_cancel_before_run(proj, tmp_path):
    w = workers.TrainingWorker(small_cfg(proj, tmp_path))
    g = collect(w)
    w.cancel()
    assert w.is_cancelled
    w.run()
    assert g.cancelled == [True] and not g.error and not g.finished


def test_training_worker_cancel_via_progress(proj, tmp_path):
    w = workers.TrainingWorker(small_cfg(proj, tmp_path))
    g = collect(w)
    w.progress_signal.connect(lambda p, m: w.cancel() if p >= 20 else None)      # Stop tugmasi o'rnida
    w.run()
    assert g.cancelled == [True] and not g.error and not g.finished
    assert max(p for p, _ in g.progress) < 100


# ---------------------------------------------------------------------------
# PredictionWorker / ExportWorker / ApplyBundleWorker
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def predicted(trained, tmp_path_factory):
    out = tmp_path_factory.mktemp("wk_pred")
    res = trained.got.finished[0]
    w = workers.PredictionWorker(res, str(out), class_method="equal_interval", n_classes=4)
    g = collect(w)
    w.run()
    return SimpleNamespace(got=g, out=out, res=res)


def test_prediction_worker(predicted):
    g = predicted.got
    assert not g.error and len(g.finished) == 1
    p = g.finished[0]
    assert {"maps", "class_map", "class_stats", "success_curve", "saved_paths"} <= set(p)
    assert len(p["class_stats"]) == 4 and ENSEMBLE_NAME in p["maps"]
    assert any(os.path.basename(x) == "prognoz_classes.tif" for x in p["saved_paths"])
    assert [x for x, _ in g.progress][-1] == 100 and g.log


def test_prediction_worker_cancel_and_error(trained, tmp_path):
    res = trained.got.finished[0]
    w = workers.PredictionWorker(res, str(tmp_path))
    g = collect(w)
    w.cancel()
    w.run()
    assert g.cancelled == [True] and not g.finished
    w = workers.PredictionWorker({"cfg": {}}, str(tmp_path))          # yaroqsiz natija
    g = collect(w)
    w.run()
    assert len(g.error) == 1 and "Traceback" in g.error[0] and not g.cancelled


def test_export_worker(trained, predicted, tmp_path):
    w = workers.ExportWorker(trained.got.finished[0], str(tmp_path / "exp"), predicted.got.finished[0])
    g = collect(w)
    w.run()
    assert not g.error and len(g.finished) == 1
    out = g.finished[0]
    assert out["out_dir"] == str(tmp_path / "exp")
    names = {os.path.basename(p) for p in out["paths"]}
    assert {"metrics_spatial.csv", "summary.txt", "class_stats.csv"} <= names
    assert all(os.path.isfile(p) for p in out["paths"])
    assert len(pd.read_csv(tmp_path / "exp" / "metrics_spatial.csv")) == 3          # RF, SVM, ansambl
    assert [p for p, _ in g.progress][-1] == 100


def test_export_worker_error(tmp_path):
    w = workers.ExportWorker({}, str(tmp_path))
    g = collect(w)
    w.run()
    assert len(g.error) == 1 and "Traceback" in g.error[0] and not g.finished


def test_apply_bundle_worker(trained, tmp_path):
    res = trained.got.finished[0]
    bdir = str(tmp_path / "bundle")
    persist.save_bundle(bdir, final_models=res["final_models"], pipeline=res["pipeline"],
                        hyperparams_used=res["final_hyperparams"], cfg_dict=res["cfg"],
                        thresholds=res["thresholds"], block_size=res["block_size"])
    w = workers.ApplyBundleWorker(bdir, trained.proj["tiff"], str(tmp_path / "apply"), class_method="quantile",
                                  n_classes=3)
    g = collect(w)
    w.run()
    assert not g.error and len(g.finished) == 1
    out = g.finished[0]
    assert "raster" not in out                                             # katta stack uzatilmaydi
    assert {"transform", "profile", "pixel_size", "band_names", "manifest", "matched_files", "class_stats"} <= set(out)
    assert len(out["class_stats"]) == 3 and out["maps"][ENSEMBLE_NAME].ndim == 2
    assert out["manifest"]["model_names"] == ["RandomForest", "SVM"] and out["pixel_size"] == 100.0
    assert any(os.path.basename(x) == "prognoz_classes.tif" for x in out["saved_paths"])
    assert any("Faqat ishonchli" in m or "ishonchli" in m for m in g.log)       # xavfsizlik eslatmasi logda
    pct = [p for p, _ in g.progress]                                       # regressiya: 2% dan 0% ga qaytmasin
    assert all(b >= a for a, b in zip(pct, pct[1:])) and pct[-1] == 100


def test_apply_bundle_worker_errors(trained, tmp_path):
    w = workers.ApplyBundleWorker(str(tmp_path / "yoq"), trained.proj["tiff"], None)
    g = collect(w)
    w.run()
    assert len(g.error) == 1 and "manifest.json" in g.error[0] and not g.finished
    w = workers.ApplyBundleWorker(str(tmp_path / "yoq"), trained.proj["tiff"], None)
    g = collect(w)
    w.cancel()
    w.run()
    assert len(g.error) == 1                                                # bundle yo'q: bekor emas, xato
