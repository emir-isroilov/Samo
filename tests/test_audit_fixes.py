# -*- coding: utf-8 -*-
"""Yakuniy audit topilmalari uchun regressiya testlari (tuzatishsiz yiqiladi)."""
import json
import os
import time

import numpy as np
import pytest
from affine import Affine

from mpm import config, cv, data, persist, pipeline
from mpm.common import CancelToken, CancelledError
from tests.synth import make_synthetic_project


def _raster(transform, shape=(1, 6, 6)):
    return data.RasterStack(stack=np.ones(shape, dtype=np.float32), band_names=["a"], profile={},
                            transform=transform, crs_epsg=28411, categorical=[], tech_metadata=[])


def test_pixel_size_non_square_uses_pixel_area():
    r = _raster(Affine(100.0, 0, 0, 0, -50.0, 0))
    assert r.pixel_size == pytest.approx(np.sqrt(100.0 * 50.0))
    assert _raster(Affine(30.0, 0, 0, 0, -30.0, 0)).pixel_size == pytest.approx(30.0)


def test_sig_digits_keep_tiny_values():
    assert data._sig(1.234567891e-9) == pytest.approx(1.23457e-9)
    assert data._sig(-4.2887693729792e13) == pytest.approx(-4.28877e13)


def test_negative_seed_background_and_diagnostics(tmp_path):
    proj = make_synthetic_project(str(tmp_path), size=60, n_layers=3, n_pos=15)
    raster = data.load_and_align_rasters(data.find_tiff_files(proj["tiff"]))
    aoi = data.load_aoi(data.find_shapefile(proj["aoi"]))
    pos = data.load_positive_points(data.find_shapefile(proj["points"]), aoi)
    valid = data.valid_pixel_mask(raster)
    bg = data.generate_background_points(aoi, pos, 20, 300, random_state=-5, valid_mask=valid,
                                         transform=raster.transform)
    assert len(bg) == 20
    assert data.data_diagnostics(raster, None, seed=-5)["layer_stats"] is not None


def test_background_cancel_and_attempt_cap(tmp_path):
    proj = make_synthetic_project(str(tmp_path), size=50, n_layers=3, n_pos=15)
    raster = data.load_and_align_rasters(data.find_tiff_files(proj["tiff"]))
    aoi = data.load_aoi(data.find_shapefile(proj["aoi"]))
    pos = data.load_positive_points(data.find_shapefile(proj["points"]), aoi)
    valid = data.valid_pixel_mask(raster)
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledError):
        data.generate_background_points(aoi, pos, 50, 300, valid_mask=valid, transform=raster.transform,
                                        cancel=tok)
    for strategy in ("random", "grid"):
        t0 = time.time()
        bg = data.generate_background_points(aoi, pos, 5_000_000, 300, strategy=strategy, valid_mask=valid,
                                             transform=raster.transform)
        assert len(bg) <= int(valid.sum())
        assert time.time() - t0 < 30, "mavjud pikselldan ko'p so'ralganda behuda uzoq urinmasligi kerak"


def test_compute_metrics_bootstrap_cancel():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(20), np.zeros(60)].astype(int)
    oof = {"RandomForest": [rng.random(80)]}
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledError):
        cv.compute_metrics(y, oof, None, n_boot=200, cancel=tok)
    assert cv.compute_metrics(y, oof, None, n_boot=20)[0]["RandomForest"]["auc"] >= 0


def test_categorical_non_integer_values_warn():
    band = np.array([[1.0, 2.0], [2.4, 3.0]], dtype=np.float32)
    raster = data.RasterStack(stack=band[None], band_names=["geo"], profile={}, transform=Affine(1, 0, 0, 0, -1, 0),
                              crs_epsg=28411, categorical=["geo"], tech_metadata=[])
    fp = data.FeaturePipeline(["geo"], categorical=["geo"]).fit(np.array([[[1.0, 2.0], [2.0, 3.0]]], np.float32))
    logs = []
    persist._check_categorical_levels(raster, fp, logs.append)
    assert any("butun son emas" in m for m in logs)


def test_check_out_dir_rejects_unwritable(tmp_path):
    f = tmp_path / "afile"
    f.write_text("x")
    with pytest.raises(ValueError, match="yozib bo'lmaydi"):
        pipeline._check_out_dir(str(f / "sub"))
    pipeline._check_out_dir("")          # bo'sh => hech narsa
    pipeline._check_out_dir(str(tmp_path / "new" / "dir"))
    assert os.path.isdir(tmp_path / "new" / "dir")


def test_run_training_fails_fast_on_unwritable_output(tmp_path):
    f = tmp_path / "afile"
    f.write_text("x")
    cfg = config.RunConfig(tiff_folder="x", points_folder="y", aoi_folder="z", output_dir=str(f / "sub"))
    t0 = time.time()
    with pytest.raises(ValueError, match="yozib bo'lmaydi"):
        pipeline.run_training(cfg)
    assert time.time() - t0 < 5


@pytest.mark.slow
def test_export_run_config_is_reloadable(tmp_path):
    proj = make_synthetic_project(str(tmp_path / "p"), size=70, n_layers=3, n_pos=30, seed=2)
    hp = config.default_hyperparams()
    hp["RandomForest"]["n_estimators"] = 10
    cfg = config.RunConfig(tiff_folder=proj["tiff"], points_folder=proj["points"], aoi_folder=proj["aoi"],
                           n_background=40, min_distance=300.0, n_splits=3, n_repeats=1, n_bootstrap=0,
                           hyperparams=hp, run_random_cv=False, shap_enabled=False, perm_importance=False,
                           n_jobs=1, block_size=0.0,
                           use_models={"RandomForest": True, "SVM": False, "XGBoost": False, "CNN": False})
    res = pipeline.run_training(cfg)
    d = str(tmp_path / "exp")
    pipeline.export_results(res, d)
    payload = json.load(open(os.path.join(d, "run_config.json"), encoding="utf-8"))
    assert payload["kind"] == "run_config"
    assert payload["block_size"] == 0.0 and payload["block_size_used"] == pytest.approx(res["block_size"])
    loaded = config.load_preset(os.path.join(d, "run_config.json"))
    assert loaded["kind"] == "run_config" and loaded["cfg"].tiff_folder == proj["tiff"]
    assert loaded["cfg"].block_size == 0.0


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_result_tab_slot_error_cleared_after_success(qapp):
    from mpm.gui.result_tabs import ResultsTab
    tab = ResultsTab()
    tab._fail("disk to'la")
    assert tab.last_error and not tab.error_label.isHidden()
    tab._info("Jadval saqlandi")
    assert tab.last_error is None and tab.error_label.isHidden()
