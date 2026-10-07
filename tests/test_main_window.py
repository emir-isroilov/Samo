# -*- coding: utf-8 -*-
"""mpm.gui.main_window testlari (QT_QPA_PLATFORM=offscreen).

Dialoglar (QMessageBox/QFileDialog) oyna ob'ekti darajasida almashtiriladi (offscreen'da bloklanmasligi uchun).
Haqiqiy kichik sintetik loyihada TrainingWorker'ni ishga tushirish (RF+SVM, n_repeats=1, n_splits=3, n_bootstrap=50) -
`flow` fixture (modul darajasida bir marta): on_training_finished barcha tab'larni to'ldiradi, avto-eksport, prognoz,
qo'lda eksport, bundle saqlash/yuklash/qo'llash. Qolgan testlar: config round-trip, validatsiya, _set_busy, BUG-02 (xato/
bekor -> tugmalar tiklanadi), Stop, excepthook, closeEvent, qatlamlar avtotavsiyasi, cost hint, kutubxona yo'q holati."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QThread, Qt, QTimer  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from mpm import config  # noqa: E402
from mpm.common import ENSEMBLE_NAME, CancelledError  # noqa: E402
from mpm.config import RunConfig, TuningConfig, default_hyperparams  # noqa: E402
from mpm.gui import main_window as mw  # noqa: E402
from mpm.gui.main_window import (TAB_DIAG, TAB_MAP, TAB_RESULTS, TAB_TITLES, MainWindow,  # noqa: E402
                                 format_manifest, scalar_metrics)
from mpm.workers import _BaseWorker  # noqa: E402
from tests.synth import make_synthetic_project  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ===========================================================================
# yordamchilar
# ===========================================================================
class Dialogs:
    """Dialog javoblari va yozuvlari."""

    def __init__(self):
        self.warnings, self.infos, self.errors, self.questions = [], [], [], []
        self.trust = True
        self.close_ok = True
        self.dir = ""
        self.save_file = ""
        self.open_file = ""


def make_window(dlg):
    win = MainWindow()

    def warn(title, text):
        dlg.warnings.append((title, text))

    def info(title, text):
        dlg.infos.append((title, text))

    def err(title, text, detail=None):
        dlg.errors.append((title, text, detail))

    def trust():
        dlg.questions.append("trust")
        return dlg.trust

    def close_q():
        dlg.questions.append("close")
        return dlg.close_ok

    win._warn, win._info, win._show_error = warn, info, err
    win._confirm_bundle_trust, win._confirm_close = trust, close_q
    win._ask_directory = lambda title, start="": dlg.dir
    win._ask_save_file = lambda title, default, flt: dlg.save_file
    win._ask_open_file = lambda title, flt: dlg.open_file
    return win


def shutdown(win):
    win._shutdown_threads()
    win.close()
    win.deleteLater()
    QApplication.processEvents()


def small_cfg(proj, out_dir, **kw):
    hp = default_hyperparams()
    hp["RandomForest"]["n_estimators"] = 15
    base = dict(tiff_folder=proj["tiff"], points_folder=proj["points"], aoi_folder=proj["aoi"],
                output_dir=str(out_dir), n_background=60, min_distance=300.0, n_splits=3, n_repeats=1,
                n_bootstrap=50, hyperparams=hp, run_random_cv=False, shap_enabled=False, perm_importance=True,
                perm_importance_repeats=1, n_jobs=1, seed=5,
                use_models={"RandomForest": True, "SVM": True, "XGBoost": False, "CNN": False})
    base.update(kw)
    return RunConfig(**base)


class _Boom(_BaseWorker):
    def _work(self):
        raise RuntimeError("sinov xatosi")


class _Cancelled(_BaseWorker):
    def _work(self):
        raise CancelledError("to'xtatildi")


class _Slow(_BaseWorker):
    """Bekor qilishga javob beruvchi uzoq ish."""

    def _work(self):
        for _ in range(2000):
            self._cancel.raise_if_cancelled()
            time.sleep(0.01)
        return {}


class _Return(_BaseWorker):
    def __init__(self, result, delay=0.2):
        super().__init__()
        self.result, self.delay = result, delay

    def _work(self):
        time.sleep(self.delay)
        return self.result


@pytest.fixture(scope="module")
def proj(tmp_path_factory):
    return make_synthetic_project(str(tmp_path_factory.mktemp("mw_proj")), size=80, n_layers=4, n_pos=40)


@pytest.fixture(scope="module")
def proj_cat(tmp_path_factory):
    return make_synthetic_project(str(tmp_path_factory.mktemp("mw_cat")), size=60, n_layers=3, n_pos=30,
                                  categorical=True, seed=3)


@pytest.fixture
def dlg():
    return Dialogs()


@pytest.fixture
def win(dlg):
    w = make_window(dlg)
    yield w
    shutdown(w)


@pytest.fixture(scope="module")
def flow(proj, tmp_path_factory):
    """Haqiqiy o'qitish (TrainingWorker, QEventLoop orqali) - bir marta."""
    dlg = Dialogs()
    win = make_window(dlg)
    out = tmp_path_factory.mktemp("mw_out")
    cfg = small_cfg(proj, out, run_random_cv=True)
    win._apply_config(cfg)
    progress = []
    worker = win.start_training()
    assert worker is not None
    worker.progress_signal.connect(lambda p, m: progress.append(p))
    ok = win.wait_idle(300000)
    ns = SimpleNamespace(win=win, dlg=dlg, out=out, proj=proj, cfg=cfg, progress=progress, ok=ok,
                         tmp=tmp_path_factory.mktemp("mw_flow"))
    yield ns
    shutdown(win)


# ===========================================================================
# Qt'siz yordamchilar
# ===========================================================================
class TestHelpers:
    def test_heavy_libs_not_imported_at_module_level(self):
        """tensorflow/shap/xgboost/pipeline modul darajasida import qilinmaydi (GUI tez ochilishi uchun)."""
        import subprocess
        code = ("import sys; import mpm.gui.main_window as m; "
                "print([x for x in ('tensorflow', 'shap', 'xgboost', 'mpm.pipeline', 'mpm.cv', 'mpm.persist') "
                "if x in sys.modules])")
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        out = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, %r); " % os.getcwd() + code],
                             capture_output=True, text=True, env=env, timeout=120)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout

    def test_scalar_metrics(self):
        import numpy as np
        m = {"RF": {"auc": 0.9, "auc_ci95": (0.8, np.float32(0.95)), "fpr": np.arange(5), "confusion": [[1, 2], [3, 4]],
                    "brier": float("nan"), "n": np.int64(7), "ok": np.bool_(True),
                    "calibration": {"prob_pred": [0.1]}, "mean_proba": np.zeros(3)},
             "bad": None}
        out = scalar_metrics(m)
        assert set(out) == {"RF"}
        d = out["RF"]
        assert d["auc"] == 0.9 and d["auc_ci95_lo"] == 0.8 and d["auc_ci95_hi"] == pytest.approx(0.95)
        assert d["n"] == 7.0 and d["ok"] is True and d["brier"] != d["brier"]
        assert not ({"fpr", "confusion", "calibration", "mean_proba", "auc_ci95"} & set(d))
        assert scalar_metrics(None) == {}
        json.dumps({k: {a: (None if b != b else b) for a, b in v.items()} for k, v in out.items()})

    def test_format_manifest(self):
        man = {"bundle_version": 1, "created_utc": "2026-01-01T00:00:00+00:00", "mpm_version": "2.0.0",
               "band_names": ["a", "b"], "categorical": ["b"], "n_features": 3, "crs_epsg": 28411,
               "model_names": ["RandomForest"], "models": {"RandomForest": {"n_draws": 2, "class": "SklearnModel",
                                                                              "input_kind": "tabular", "loader": "joblib"}},
               "hyperparams_used": {"RandomForest": [{"n_estimators": 15}]},
               "metrics_summary": {"RandomForest": {"auc": 0.91, "auc_ci95_lo": 0.8, "auc_ci95_hi": 0.95,
                                                    "pr_auc": None}},
               "block_size": 1200.0, "notes": "izoh", "versions": {"python": "3.11", "numpy": "2.0"}}
        t = format_manifest(man)
        for s in ("Band nomlari (2): a, b", "Kategorik qatlamlar: b", "RandomForest: 2 ta fon tanlovi",
                  "n_estimators=15", "AUC=0.910", "95% CI 0.800-0.950", "1200 m", "izoh", "python 3.11",
                  "ishonchli manbadagi", "ehtimollik emas"):
            assert s in t, s
        assert "-" in t.split("PR-AUC=")[1].split(",")[0]            # None -> "-"
        assert "o'qib bo'lmadi" in format_manifest(None) and format_manifest({})

    def test_list_tiff_stems(self, proj, tmp_path):
        assert mw._list_tiff_stems(proj["tiff"])[:2] == ["layer1", "layer2"]
        assert mw._list_tiff_stems(str(tmp_path / "yoq")) == [] and mw._list_tiff_stems("") == []
        (tmp_path / ".hid.tif").write_bytes(b"")
        (tmp_path / "B.TIF").write_bytes(b"")
        (tmp_path / "a.tiff").write_bytes(b"")
        (tmp_path / "x.txt").write_bytes(b"")
        assert mw._list_tiff_stems(str(tmp_path)) == ["a", "B"]


# ===========================================================================
# oyna va konfiguratsiya
# ===========================================================================
class TestWindow:
    def test_eight_tabs(self, win):
        assert win.tabs.count() == 8
        assert [win.tabs.tabText(i) for i in range(8)] == list(TAB_TITLES)
        assert win.tabs.tabText(0) == "1. Ma'lumotlar va o'qitish" and win.tabs.tabText(7) == "8. Modellar"
        assert win.windowTitle().startswith("MPM ML GUI v2")
        names = [a.text() for a in win.menuBar().actions()]
        assert names == ["&Fayl", "&Yordam"]

    def test_initial_state_matches_runconfig_defaults(self, win):
        exp = RunConfig()
        for m in exp.use_models:
            if not mw._available(m):
                exp.use_models[m] = False
        assert win._collect_config().to_dict() == exp.to_dict()
        assert win.btn_train.isEnabled() and not win.btn_stop.isEnabled() and not win.is_busy()
        assert not win.btn_bundle_save.isEnabled() and not win.btn_apply.isEnabled()
        assert not win.map_tab.btn_predict.isEnabled()                 # natija yo'q => prognoz tugmasi o'chiq
        assert win.bundle_path_label.text() == "Bundle tanlanmagan."

    def test_config_roundtrip(self, win, proj):
        hp = default_hyperparams()
        hp["RandomForest"].update(n_estimators=77, max_depth=9, class_weight="balanced_subsample")
        hp["SVM"].update(C=2.5, gamma=0.05, kernel="linear")
        hp["XGBoost"].update(learning_rate=0.1, max_depth=5)
        hp["CNN"].update(mode="tabular1d", window=11, epochs=3)
        tun = TuningConfig(enabled=True, n_iter=7, inner_splits=4, scoring="average_precision",
                           models={"RandomForest": True, "SVM": False, "XGBoost": True, "CNN": False},
                           spaces={"RandomForest": {"n_estimators": {"min": 50, "max": 120}}})
        cfg = RunConfig.from_dict(dict(
            RunConfig(tiff_folder=proj["tiff"], points_folder=proj["points"], aoi_folder=proj["aoi"],
                      output_dir="/tmp/chiqish yo'li", categorical_layers=["layer2", "layer4"], assume_crs_if_missing=False,
                      n_background=123, min_distance=250.5, background_strategy="distance_weighted", final_bg_draws=3,
                      n_splits=4, n_repeats=6, block_size=1500.0, variogram_band="layer3", run_random_cv=False,
                      n_bootstrap=0, use_models={"RandomForest": True, "SVM": False, "XGBoost": True, "CNN": False},
                      hyperparams=hp, calibrate=False, calibration_method="isotonic", calibration_cv=4, tuning=tun,
                      bg_sensitivity_enabled=True, bg_sensitivity_draws=7, bg_sensitivity_repeats=2,
                      perm_importance=False, perm_importance_repeats=9, shap_enabled=False, shap_max_background=33,
                      seed=999, n_jobs=2, class_method="fixed", n_classes=3, class_breaks=[0.1, 0.5]).to_dict()))
        warns = win._apply_config(cfg)
        assert warns == [] or all("o'rnatilmagan" in w for w in warns)
        got = win._collect_config()
        assert got.to_dict() == cfg.to_dict()
        # vidjetlar haqiqatan yangilangan
        assert win.spin_background.value() == 123 and win.spin_bootstrap.value() == 0
        assert win.combo_variogram.currentData() == "layer3" and win._checked_layers() == ["layer2", "layer4"]
        assert win.hyper_panel.get_hyperparams()["SVM"]["C"] == 2.5
        assert win.hyper_panel.get_tuning().n_iter == 7
        assert win.map_tab.method_combo.currentData() == "fixed" and win.map_tab.breaks_edit.text() == "0.1, 0.5"
        assert not win.spin_cal_cv.isEnabled() and win.spin_bg_draws.isEnabled()

    def test_roundtrip_nonexistent_folder_keeps_categorical(self, win):
        cfg = RunConfig(tiff_folder="/yoq/papka", categorical_layers=["geo", "lit"], variogram_band="geo")
        win._apply_config(cfg)
        got = win._collect_config()
        assert got.categorical_layers == ["geo", "lit"] and got.variogram_band == "geo"
        assert got.tiff_folder == "/yoq/papka"

    def test_apply_config_clamps_with_warnings(self, win):
        cfg = RunConfig(n_splits=2)
        cfg.n_splits = 500
        cfg.class_method = "quantile"
        warns = win._apply_config(cfg)
        assert any("K-fold" in w and "diapazondan chiqdi" in w for w in warns)
        assert win.spin_kfold.value() == 20

    def test_config_file_roundtrip(self, win, dlg, proj, tmp_path):
        cfg = small_cfg(proj, tmp_path / "o", categorical_layers=["layer3"], n_background=99,
                        class_method="equal_interval", n_classes=4)
        cfg = RunConfig.from_dict(cfg.to_dict())
        win._apply_config(cfg)
        p = str(tmp_path / "cfg.json")
        dlg.save_file = p
        assert win.save_config_dialog() == p and os.path.isfile(p)
        with open(p, encoding="utf-8") as f:
            assert json.load(f)["kind"] == "run_config"
        win._apply_config(RunConfig())                                  # boshqa holat
        assert win._collect_config().n_background != 99
        dlg.open_file = p
        assert win.load_config_dialog() == p
        assert win._collect_config().to_dict() == cfg.to_dict()

    def test_load_hyperparams_only_preset(self, win, dlg, tmp_path):
        win.spin_background.setValue(321)
        hp = default_hyperparams()
        hp["SVM"]["C"] = 7.5
        p = str(tmp_path / "hp.json")
        config.save_preset(p, hyperparams=hp, tuning=TuningConfig(enabled=True, n_iter=3))
        warns = win.load_config_from(p)
        assert win.hyper_panel.get_hyperparams()["SVM"]["C"] == 7.5
        assert win.hyper_panel.get_tuning().enabled and win.hyper_panel.get_tuning().n_iter == 3
        assert win.spin_background.value() == 321                       # qolgan sozlamalar o'zgarmaydi
        assert any("faqat giperparametrlar" in w for w in warns)

    def test_load_config_errors_are_dialogs_not_exceptions(self, win, dlg, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{ bu json emas", encoding="utf-8")
        dlg.open_file = str(bad)
        assert win.load_config_dialog() is None
        assert dlg.errors and "Konfiguratsiya yuklanmadi" in dlg.errors[-1][0]
        assert not win.is_busy()

    def test_failed_config_load_leaves_widgets_unchanged(self, win, dlg, tmp_path):
        """Regressiya (reviewer): yaroqsiz tur (n_background='abc') yuklashda vidjetlar yarim yangilangan qolardi."""
        win.picker_points.setPath(str(tmp_path))
        win.spin_kfold.setValue(7)
        before = win._collect_config().to_dict()
        bad = tmp_path / "bad_types.json"
        bad.write_text(json.dumps({"kind": "run_config", "points_folder": "/boshqa/yol", "n_splits": 3,
                                   "n_background": "abc"}), encoding="utf-8")
        dlg.open_file = str(bad)
        assert win.load_config_dialog() is None
        assert dlg.errors and "Konfiguratsiya yuklanmadi" in dlg.errors[-1][0]
        assert win._collect_config().to_dict() == before
        assert win.picker_points.path() == str(tmp_path) and win.spin_kfold.value() == 7

    def test_tuning_final_mode_warns(self, win):
        cfg = RunConfig(tuning=TuningConfig(enabled=True, mode="final"))
        warns = win._apply_config(cfg)
        assert any("'final'" in w for w in warns)
        assert win._collect_config().tuning.mode == "final"             # preset qiymati saqlanadi

    def test_libraries_missing(self, monkeypatch, dlg):
        monkeypatch.setattr(mw, "_available", lambda m: m not in ("XGBoost", "CNN"))
        w = make_window(dlg)
        try:
            for m in ("XGBoost", "CNN"):
                cb = w.model_checks[m]
                assert not cb.isEnabled() and not cb.isChecked() and "o'rnatilmagan" in cb.text()
            assert w.model_checks["RandomForest"].isEnabled() and w.model_checks["SVM"].isEnabled()
            warns = w._apply_config(RunConfig())                        # hammasi True talab qilinadi
            assert any("XGBoost" in x for x in warns) and any("CNN" in x for x in warns)
            um = w._collect_config().use_models
            assert um == {"RandomForest": True, "SVM": True, "XGBoost": False, "CNN": False}
        finally:
            shutdown(w)

    def test_cnn_note_visibility(self, win):
        win.model_checks["CNN"].setEnabled(True)
        win.model_checks["CNN"].setChecked(True)
        win.model_checks["SVM"].setChecked(True)
        assert not win.cnn_note.isHidden()
        win.model_checks["SVM"].setChecked(False)
        win.model_checks["RandomForest"].setChecked(False)
        win.model_checks["XGBoost"].setChecked(False)
        assert win.cnn_note.isHidden()                                  # faqat CNN: shkala eslatmasi kerak emas

    def test_dependent_widgets(self, win):
        win.chk_calibrate.setChecked(False)
        assert not win.combo_cal_method.isEnabled() and not win.spin_cal_cv.isEnabled()
        win.chk_calibrate.setChecked(True)
        assert win.combo_cal_method.isEnabled()
        win.chk_shap.setChecked(False)
        assert not win.spin_shap_bg.isEnabled()
        win.chk_perm.setChecked(False)
        assert not win.spin_perm_repeats.isEnabled()

    def test_menu_help(self, win, dlg):
        win.act_index_help.trigger()
        assert "EHTIMOLLIK EMAS" in dlg.infos[-1][1] and "nisbiy" in dlg.infos[-1][1].lower()
        win.act_about.trigger()
        assert "MPM ML GUI v2" in dlg.infos[-1][1] and "prospektivlik indeksi" in dlg.infos[-1][1]


# ===========================================================================
# validatsiya
# ===========================================================================
class TestValidation:
    def test_no_folders(self, win, dlg):
        assert win.start_training() is None
        assert dlg.warnings and "TIFF qatlamlar papkasi tanlanmagan" in dlg.warnings[-1][1]
        assert "AOI" in dlg.warnings[-1][1] and not win.is_busy()
        assert win.btn_train.isEnabled()

    def test_missing_dirs_and_no_models(self, win, dlg, proj, tmp_path):
        cfg = small_cfg(proj, tmp_path, use_models={m: False for m in RunConfig().use_models})
        cfg.tiff_folder = str(tmp_path / "yoq")
        win._apply_config(cfg)
        assert win.start_training() is None
        msg = dlg.warnings[-1][1]
        assert "topilmadi" in msg and "Kamida bitta model" in msg and not win.is_busy()

    def test_tuning_without_models_message(self, win, dlg, proj, tmp_path):
        cfg = small_cfg(proj, tmp_path, tuning=TuningConfig(enabled=True, models={m: False for m in
                                                                                 RunConfig().use_models}))
        win._apply_config(cfg)
        assert win.start_training() is None and "Tuning yoqilgan" in dlg.warnings[-1][1]

    def test_output_path_is_file(self, win, dlg, proj, tmp_path):
        f = tmp_path / "fayl.txt"
        f.write_text("x")
        win._apply_config(small_cfg(proj, f))
        assert win.start_training() is None and "papka emas" in dlg.warnings[-1][1]

    def test_prediction_and_export_need_result(self, win, dlg):
        assert win.start_prediction({"out_dir": ""}) is None and "o'qiting" in dlg.warnings[-1][1]
        assert win.start_export() is None and "o'qiting" in dlg.warnings[-1][1]
        assert win.save_bundle_dialog() is None
        assert win.start_diagnostics() is None and "TIFF" in dlg.warnings[-1][1]
        assert win.start_apply() is None and "bundle" in dlg.warnings[-1][1].lower()


# ===========================================================================
# _set_busy, BUG-02, Stop
# ===========================================================================
class TestBusyAndErrors:
    def test_set_busy_states_in_one_place(self, win):
        win._job_kind = "train"
        win._set_busy(True)
        assert win.is_busy() and not win.btn_train.isEnabled() and win.btn_stop.isEnabled()
        assert not win.settings_widget.isEnabled() and not win.hyper_panel.isEnabled()
        assert not win.btn_load_cfg.isEnabled() and not win.act_load_cfg.isEnabled()
        assert all(t.is_busy() for t in win.result_tab_widgets)
        assert not win.diag_tab.btn_refresh.isEnabled()
        win._stopping = True
        win._update_controls()
        assert not win.btn_stop.isEnabled()                             # to'xtatilmoqda: qayta bosib bo'lmaydi
        win._set_busy(False)
        assert not win._stopping and win.btn_train.isEnabled() and not win.btn_stop.isEnabled()
        assert win.settings_widget.isEnabled() and win.hyper_panel.isEnabled() and win.btn_load_cfg.isEnabled()
        assert not any(t.is_busy() for t in win.result_tab_widgets)
        win._job_kind = "export"
        win._set_busy(True)
        assert not win.btn_stop.isEnabled()                             # eksportni to'xtatib bo'lmaydi
        win._job_kind = None
        win._set_busy(False)

    def test_error_signal_restores_buttons_no_result(self, win, dlg):
        win._begin_job("predict", _Boom())
        assert win.is_busy() and not win.btn_train.isEnabled()
        assert win.wait_idle(20000)
        assert not win.is_busy() and win.btn_train.isEnabled() and not win.btn_stop.isEnabled()
        assert win.settings_widget.isEnabled()
        assert dlg.errors and "sinov xatosi" in dlg.errors[-1][1] and "Prognozda xato" == dlg.errors[-1][0]
        assert "Traceback" in dlg.errors[-1][2]                          # to'liq matn batafsilda
        assert win.progress.stage_text().startswith("Xato") and "sinov xatosi" in win.last_job_error
        assert "XATO [predict]" in win.log_view.toPlainText()

    def test_cancel_signal_restores_buttons(self, win):
        win._begin_job("train", _Cancelled())
        assert win.wait_idle(20000)
        assert not win.is_busy() and win.btn_train.isEnabled() and not win.btn_stop.isEnabled()
        assert win.progress.stage_text() == "To'xtatildi"
        assert win.training_result is None

    def test_stop_state_then_cancelled(self, win):
        win._begin_job("predict", _Slow())
        assert win.btn_stop.isEnabled()
        assert win.stop_current() is True
        assert win.is_busy() and not win.btn_stop.isEnabled()           # to'xtatilmoqda...
        assert "To'xtatilmoqda" in win.progress.stage_text()
        assert win.stop_current() is False                              # ikkinchi marta e'tiborsiz
        assert win.wait_idle(20000)
        assert not win.is_busy() and win.btn_train.isEnabled() and win.progress.stage_text() == "To'xtatildi"
        assert "to'xtatildi" in win.log_view.toPlainText().lower()

    def test_stop_not_available_for_export(self, win):
        win._begin_job("autoexport", _Slow(), progress=False)
        assert not win.btn_stop.isEnabled() and win.stop_current() is False
        win._job_worker.cancel()
        assert win.wait_idle(20000)

    def test_autoexport_error_is_logged_not_dialog(self, win, dlg):
        win._begin_job("autoexport", _Boom(), progress=False)
        assert win.wait_idle(20000)
        assert not dlg.errors and "Avtomatik eksport o'tkazib yuborildi" in win.log_view.toPlainText()
        assert win.btn_train.isEnabled()

    def test_garbage_result_does_not_abort(self, win, dlg):
        win._job_kind = "train"
        win._set_busy(True)
        win._on_job_finished("train", "axlat")                           # istisno ushlanadi, busy tiklanadi
        assert not win.is_busy() and win.btn_train.isEnabled()
        assert "lug'at emas" in (win.last_job_error or "")

    def test_stop_after_point_of_no_return_still_shows_result(self, flow, dlg):
        """~95% dan keyin Stop: finished_signal keladi - natija ko'rsatiladi va log'da izoh bo'ladi."""
        res = dict(flow.win.training_result)
        res["cfg"] = {**res["cfg"], "output_dir": ""}
        w = make_window(dlg)
        try:
            w._begin_job("train", _Return(res, 0.3))
            assert w.stop_current()
            assert w.wait_idle(60000)
            assert w.training_result is res and w.results_tab.has_result() and not w.is_busy()
            assert "Stop bosilgan edi" in w.log_view.toPlainText()
            assert w.btn_train.isEnabled()
        finally:
            shutdown(w)

    def test_tab_failure_is_isolated(self, flow, dlg, monkeypatch):
        res = dict(flow.win.training_result)
        res["cfg"] = {**res["cfg"], "output_dir": ""}
        w = make_window(dlg)
        try:
            def boom(_r):
                raise RuntimeError("spatial tab yiqildi")
            monkeypatch.setattr(w.spatial_tab, "set_result", boom)
            w._job_kind = "train"
            w._set_busy(True)
            w._on_job_finished("train", res)
            assert not w.is_busy() and w.training_result is res
            assert not w.spatial_tab.has_result()
            assert w.diag_tab.has_result() and w.results_tab.has_result() and w.importance_tab.has_result()
            assert "XATO: 'Spatial CV diagnostika'" in w.log_view.toPlainText()
            assert "spatial tab yiqildi" in w.log_view.toPlainText()
            assert w.tabs.currentIndex() == TAB_RESULTS
            assert w.map_tab.btn_predict.isEnabled() and w.btn_bundle_save.isEnabled()
        finally:
            shutdown(w)


# ===========================================================================
# haqiqiy o'qitish -> tab'lar -> prognoz -> eksport -> bundle
# ===========================================================================
class TestFullFlow:
    def test_training_fills_all_tabs(self, flow):
        win = flow.win
        assert flow.ok, "o'qitish/eksport tugamadi (timeout)"
        assert not flow.dlg.errors, flow.dlg.errors
        res = win.training_result
        assert isinstance(res, dict) and {"spatial", "random", "final_models"} <= set(res)
        assert win.diag_tab.has_result() and win.results_tab.has_result()
        assert win.spatial_tab.has_result() and win.importance_tab.has_result()
        for t in win.result_tab_widgets:
            assert t.last_error is None, (type(t).__name__, t.last_error)
        assert not win.map_tab.has_result()                              # prognoz hali yo'q
        assert win.map_tab.btn_predict.isEnabled() and win.map_tab.btn_export.isEnabled()
        assert win.tabs.currentIndex() == TAB_RESULTS
        assert win.results_tab.metrics_table.rowCount() == 3             # RF, SVM, ansambl
        assert win.results_tab.mode_combo.model().item(1).isEnabled()    # random CV bor
        assert win.progress.value() == 100
        pct = flow.progress
        assert pct and all(b >= a for a, b in zip(pct, pct[1:]))
        text = win.log_view.toPlainText()
        assert "O'qitish muvaffaqiyatli tugadi" in text and "spatial CV AUC (ansambl)" in text
        assert win.btn_train.isEnabled() and not win.btn_stop.isEnabled() and not win.is_busy()
        assert win.btn_bundle_save.isEnabled()

    def test_auto_export_ran(self, flow):
        out = flow.out
        for name in ("metrics_spatial.csv", "run_config.json", "summary.txt", "importance.csv"):
            assert (out / name).is_file(), name
        assert any(p.name.startswith("mpm_run_") for p in out.iterdir())   # log fayli
        assert "Avtomatik eksport" in flow.win.log_view.toPlainText()
        assert "Eksport tugadi" in flow.win.log_view.toPlainText()

    def test_prediction_via_map_tab(self, flow):
        win = flow.win
        assert win.map_tab.out_picker.path() == str(flow.out)             # cfg.output_dir dan to'ldirilgan
        win.map_tab.method_combo.setCurrentIndex(win.map_tab.method_combo.findData("equal_interval"))
        win.map_tab.n_classes_spin.setValue(4)
        win.map_tab.btn_predict.click()
        assert win.is_busy() and not win.map_tab.btn_predict.isEnabled()
        assert win.wait_idle(120000)
        assert not flow.dlg.errors, flow.dlg.errors
        assert win.map_tab.has_result() and win.tabs.currentIndex() == TAB_MAP
        pred = win._last_prediction
        assert pred is not None and ENSEMBLE_NAME in pred["maps"] and len(pred["class_stats"]) == 4
        assert (flow.out / "prognoz_classes.tif").is_file()
        assert win.map_tab.btn_predict.isEnabled() and win.btn_train.isEnabled()

    def test_prediction_error_restores_prediction_button(self, flow):
        """BUG-02: prognoz xatosidan keyin prognoz tugmasi qulflanib qolmaydi."""
        win = flow.win
        out_file = flow.tmp / "fayl.txt"
        out_file.write_text("x")
        w = win.start_prediction({"out_dir": str(out_file), "class_method": "quantile", "n_classes": 5,
                                  "class_breaks": None})
        assert w is not None
        assert win.wait_idle(120000)
        assert flow.dlg.errors and flow.dlg.errors[-1][0] == "Prognozda xato"
        assert not win.is_busy() and win.map_tab.btn_predict.isEnabled() and win.btn_train.isEnabled()
        assert win.map_tab.btn_export.isEnabled() and win.settings_widget.isEnabled()
        assert win.progress.stage_text().startswith("Xato")
        flow.dlg.errors.clear()

    def test_manual_export(self, flow):
        win = flow.win
        exp = flow.tmp / "manual_export"
        flow.dlg.dir = str(exp)
        win.results_tab.btn_export.click()
        assert win.wait_idle(60000)
        assert not flow.dlg.errors
        names = {p.name for p in exp.iterdir()}
        assert {"metrics_spatial.csv", "metrics_random.csv", "class_stats.csv", "summary.txt"} <= names
        flow.dlg.dir = ""
        assert win.start_export() is None                                 # katalog tanlanmadi: hech narsa bo'lmaydi
        assert not win.is_busy()

    def test_bundle_save_load_apply(self, flow):
        win, dlg = flow.win, flow.dlg
        bdir = flow.tmp / "bundle"
        dlg.dir = str(bdir)
        win.btn_bundle_save.click()
        assert win.is_busy() and not win.btn_bundle_save.isEnabled()
        assert win.wait_idle(120000)
        assert not dlg.errors, dlg.errors
        man = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
        assert man["model_names"] == ["RandomForest", "SVM"] and man["band_names"][0] == "layer1"
        for name, m in man["metrics_summary"].items():                    # faqat skalyar metrikalar
            assert all(not isinstance(v, (list, dict)) for v in m.values()), name
        assert "auc" in man["metrics_summary"]["RandomForest"] and "auc_ci95_lo" in man["metrics_summary"]["RandomForest"]
        assert man["crs_epsg"] == 28411 and "ehtimollik emas" in man["notes"]
        assert win._bundle_dir == str(bdir) and win.btn_apply.isEnabled()
        info = win.bundle_info.toPlainText()
        assert "layer1" in info and "RandomForest" in info and "AUC=" in info
        # ---- boshqa oynada yuklash: avval xavfsizlik tasdig'i
        dlg2 = Dialogs()
        w2 = make_window(dlg2)
        try:
            dlg2.trust = False
            assert w2.load_bundle_dialog(str(bdir)) is None
            assert dlg2.questions == ["trust"] and w2._bundle_dir is None and not w2.btn_apply.isEnabled()
            assert "ishonchli manba tasdiqlanmadi" in w2.log_view.toPlainText()
            dlg2.trust = True
            assert w2.load_bundle_dialog(str(bdir)) == str(bdir)
            assert w2._bundle_dir == str(bdir) and w2.btn_apply.isEnabled()
            assert "layer1" in w2.bundle_info.toPlainText() and "ishonchli manbadagi" in w2.bundle_info.toPlainText()
            # papka bundle emas
            dlg2.dir = str(flow.tmp)
            w2._bundle_dir = None
            assert w2.load_bundle_dialog() is None and "manifest.json" in dlg2.errors[-1][1]
        finally:
            shutdown(w2)
        # ---- yangi maydonga qo'llash (ApplyBundleWorker) -> 7-tab
        before = win._last_prediction
        applied = flow.tmp / "applied"
        win.apply_tiff_picker.setPath(flow.proj["tiff"])
        win.apply_out_picker.setPath(str(applied))
        win.map_tab.method_combo.setCurrentIndex(win.map_tab.method_combo.findData("quantile"))
        win.map_tab.n_classes_spin.setValue(3)
        win.btn_apply.click()
        assert win.is_busy()
        assert win.wait_idle(120000)
        assert not dlg.errors, dlg.errors
        assert win.map_tab.has_result() and win.tabs.currentIndex() == TAB_MAP
        assert (applied / "prognoz_classes.tif").is_file()
        assert win._last_prediction is before                              # yangi maydon natijasi eksportga aralashmaydi
        assert len(win.map_tab.stats_table.dataframe()) == 3
        assert "yangi maydonga qo'llandi" in win.log_view.toPlainText()
        assert win.btn_train.isEnabled() and win.map_tab.btn_predict.isEnabled()

    def test_apply_validation(self, flow):
        win, dlg = flow.win, flow.dlg
        win.apply_tiff_picker.setPath(str(flow.tmp / "yoq"))
        assert win.start_apply() is None and "TIFF papkani" in dlg.warnings[-1][1]
        win.apply_tiff_picker.setPath(flow.proj["tiff"])
        f = flow.tmp / "chiqish.txt"
        f.write_text("x")
        win.apply_out_picker.setPath(str(f))
        assert win.start_apply() is None and "papka emas" in dlg.warnings[-1][1]
        win.apply_out_picker.setPath("")
        win.map_tab.method_combo.setCurrentIndex(win.map_tab.method_combo.findData("fixed"))
        win.map_tab.breaks_edit.setText("0.9, 0.1")
        assert win.start_apply() is None and not win.is_busy()
        win.map_tab.breaks_edit.setText("0.2, 0.4")
        win.map_tab.method_combo.setCurrentIndex(win.map_tab.method_combo.findData("quantile"))

    def test_save_figures_includes_all_tabs(self, flow):
        out = flow.tmp / "figs"
        paths = flow.win.results_tab.save_figures(str(out), formats=("png",), dpi=60)
        names = {os.path.basename(p) for p in paths}
        assert "roc_spatial.png" in names and "spatial_cv_diagnostics.png" in names and "importance.png" in names
        assert any(n.startswith("map_") for n in names) and "corr_heatmap.png" in names


# ===========================================================================
# haqiqiy Stop (o'qitish paytida)
# ===========================================================================
def test_real_training_stop(proj, tmp_path, dlg):
    win = make_window(dlg)
    try:
        win._apply_config(small_cfg(proj, tmp_path / "stop_out", run_random_cv=True, n_repeats=2))
        w = win.start_training()
        assert w is not None
        seen = {"stopped": False}

        def maybe_stop(p, m):
            if p >= 20 and not seen["stopped"]:
                seen["stopped"] = win.stop_current()
        w.progress_signal.connect(maybe_stop)
        assert win.wait_idle(120000)
        assert seen["stopped"]
        assert win.training_result is None and not win.is_busy() and win.btn_train.isEnabled()
        assert win.progress.stage_text() == "To'xtatildi" and not dlg.errors
        assert not win.results_tab.has_result()
        assert "to'xtatildi" in win.log_view.toPlainText().lower()
        assert not win.map_tab.btn_predict.isEnabled() and not win.btn_bundle_save.isEnabled()
    finally:
        shutdown(win)


# ===========================================================================
# qatlamlar ro'yxati, diagnostika, cost hint
# ===========================================================================
class TestBundleSelectionAtomic:
    def test_garbage_manifest_does_not_select_bundle(self, win, dlg, tmp_path):
        """Regressiya (reviewer): format_manifest xatosidan oldin _bundle_dir o'rnatilib, "Qo'llash" tugmasi
        rad etilgan bundle uchun yoqilib qolardi."""
        d = tmp_path / "bad_bundle"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"bundle_version": 1, "models": [1, 2], "band_names": 5}),
                                         encoding="utf-8")
        assert win.load_bundle_dialog(str(d)) is None
        assert dlg.errors and dlg.errors[-1][0] == "Bundle yuklanmadi"
        assert win._bundle_dir is None and not win.btn_apply.isEnabled()
        assert win.bundle_path_label.text() == "Bundle tanlanmagan."


class TestLayersDiagnosticsCost:
    def test_layer_scan_suggests_categorical(self, win, proj_cat):
        win.picker_tiff.setPath(proj_cat["tiff"])
        assert win._layer_scan_timer.isActive()                           # debounce
        win._layer_scan_timer.stop()
        win._scan_layers()
        assert win.layer_list.count() == 4 and win._checked_layers() == []      # tavsiya hali kelmagan
        assert win.wait_idle(60000)
        assert win._checked_layers() == ["geology_cat"]
        assert "1 tasi kategorik" in win.layer_summary.text()
        assert [win.combo_variogram.itemData(i) for i in range(win.combo_variogram.count())][0] == "auto"
        assert win.combo_variogram.findData("layer1") >= 0
        assert win._collect_config().categorical_layers == ["geology_cat"]
        assert "avtomatik tavsiya" in win.log_view.toPlainText()

    def test_layer_scan_respects_manual_changes_and_stale_ids(self, win, proj_cat):
        win.picker_tiff.blockSignals(True)
        win.picker_tiff.setPath(proj_cat["tiff"])
        win.picker_tiff.blockSignals(False)
        win._scan_layers()
        assert win.wait_idle(60000)
        assert win._checked_layers() == ["geology_cat"] and win.layer_list.item(1).text() == "layer1"
        win.layer_list.item(1).setCheckState(Qt.Checked)                  # qo'lda: layer1 ham kategorik
        assert win._layers_touched
        win._on_scan_finished({"id": win._scan_id, "categorical": ["layer2"]})
        assert win._checked_layers() == ["geology_cat", "layer1"]
        assert "qo'lda o'zgartirilgani" in win.log_view.toPlainText()
        win._on_scan_finished({"id": win._scan_id - 5, "categorical": ["layer3"]})   # eskirgan natija
        assert "layer3" not in win._checked_layers()
        win._on_scan_finished("axlat")                                    # xato yutiladi
        win._on_scan_error("Traceback...")

    def test_empty_folder_message(self, win, tmp_path):
        win.picker_tiff.setPath(str(tmp_path))
        win._layer_scan_timer.stop()
        win._scan_layers()
        assert win.layer_list.count() == 0 and "topilmadi" in win.layer_summary.text()
        assert win.wait_idle(20000)

    def test_diagnostics_refresh_without_side_effects(self, win, dlg, tmp_path):
        proj = make_synthetic_project(str(tmp_path / "dproj"), size=60, n_layers=3, n_pos=25)
        win.picker_tiff.blockSignals(True)
        win.picker_tiff.setPath(proj["tiff"])
        win.picker_tiff.blockSignals(False)
        win._populate_layers(proj["tiff"], [])
        win.diag_tab.btn_refresh.click()
        assert win.is_busy() and not win.diag_tab.btn_refresh.isEnabled()
        assert win.wait_idle(120000)
        assert not dlg.errors, dlg.errors
        assert win.diag_tab.has_result() and win.diag_tab.last_error is None
        assert win.diag_tab.stats_table.rowCount() == 3 and win.diag_tab.vif_table.rowCount() == 3
        assert win.diag_tab.dd_table.rowCount() == 3
        assert win.tabs.currentIndex() == TAB_DIAG
        assert not os.path.exists(os.path.join(proj["tiff"], "metadata.csv"))   # yon ta'sirsiz
        assert win.training_result is None and win.btn_train.isEnabled()

    def test_diagnostics_error_dialog(self, win, dlg, tmp_path):
        folder = tmp_path / "bad"
        folder.mkdir()
        (folder / "a.tif").write_bytes(b"bu tiff emas")
        win.picker_tiff.setPath(str(folder))
        win._layer_scan_timer.stop()
        win.start_diagnostics()
        assert win.wait_idle(60000)
        assert dlg.errors and dlg.errors[-1][0] == "Ma'lumotlar tahlilida xato"
        assert not win.is_busy() and win.diag_tab.btn_refresh.isEnabled()

    def test_cost_hint_updates(self, win):
        text = win.update_cost_hint()
        assert "Taxminiy hisob-kitob" in text and win.cost_view.toPlainText() == text
        assert win.hyper_panel.cost_hint() == text
        win.spin_repeats.setValue(3)
        assert win._cost_timer.isActive()
        QTest.qWait(450)                                                   # debounce taymeri
        assert "3 takror" in win.cost_view.toPlainText()
        win.hyper_panel.set_tuning(TuningConfig(enabled=True, n_iter=5))
        QTest.qWait(450)
        assert "tuning" in win.cost_view.toPlainText().lower()
        win.model_checks["CNN"].setChecked(win.model_checks["CNN"].isEnabled())
        QTest.qWait(450)
        if win.model_checks["CNN"].isChecked():
            assert "CNN qimmat" in win.cost_view.toPlainText()

    def test_cost_hint_never_raises(self, win, monkeypatch):
        monkeypatch.setattr(win, "_collect_config", lambda: (_ for _ in ()).throw(RuntimeError("kutilmagan")))
        assert "baholab bo'lmadi" in win.update_cost_hint()


# ===========================================================================
# excepthook, closeEvent, main()
# ===========================================================================
class TestProtection:
    @pytest.fixture
    def hooks(self):
        prev, prev_thread = sys.excepthook, threading.excepthook
        state = dict(mw._HOOK_STATE)
        yield
        sys.excepthook, threading.excepthook = prev, prev_thread
        mw._HOOK_STATE.update(state)

    def test_install_excepthook(self, win, dlg, hooks):
        mw.install_excepthook(win)
        assert sys.excepthook is mw._excepthook and sys.excepthook is not sys.__excepthook__
        sys.excepthook(ValueError, ValueError("portlash"), None)
        QApplication.processEvents()
        QTest.qWait(30)
        assert "portlash" in win.last_job_error
        assert "KUTILMAGAN XATO" in win.log_view.toPlainText()
        assert len(dlg.errors) == 1 and dlg.errors[0][0] == "Kutilmagan xato"
        sys.excepthook(KeyError, KeyError("ikkinchi"), None)               # qayta ishlash ham yiqilmaydi
        QTest.qWait(30)
        assert len(dlg.errors) == 2

    def test_exception_in_python_thread_is_marshalled_to_gui_thread(self, win, dlg, hooks):
        """Regressiya (reviewer): threading.excepthook vidjetlarga boshqa oqimdan tegardi ('Cannot queue QTextCursor',
        dialog hech qachon chiqmasdi). Endi report_unhandled GUI oqimida ishlaydi."""
        seen = []
        orig = win.report_unhandled

        def spy(text):
            seen.append(QThread.currentThread() is QApplication.instance().thread())
            orig(text)
        win.report_unhandled = spy
        mw.install_excepthook(win)

        def boom():
            raise RuntimeError("oqim portlashi")
        t = threading.Thread(target=boom)
        t.start()
        t.join()
        deadline = time.time() + 3
        while (not seen or not dlg.errors) and time.time() < deadline:
            QTest.qWait(20)
        assert seen == [True]
        assert dlg.errors and dlg.errors[0][0] == "Kutilmagan xato"
        assert "oqim portlashi" in win.log_view.toPlainText()

    def test_slot_exception_does_not_abort(self, win, hooks):
        """PyQt5 standart hook'da slot istisnosi qFatal bilan abort qiladi; bizning hook'da - yo'q (BUG-01)."""
        mw.install_excepthook(win)
        QTimer.singleShot(0, lambda: 1 / 0)
        QTest.qWait(60)
        assert "ZeroDivisionError" in win.log_view.toPlainText()

    def test_guarded_slot_catches(self, win, monkeypatch):
        monkeypatch.setattr(win, "_collect_config", lambda: (_ for _ in ()).throw(RuntimeError("yig'ib bo'lmadi")))
        assert win.start_training() is None                                 # _guard: tashqariga chiqmaydi
        assert "yig'ib bo'lmadi" in win.last_job_error and not win.is_busy()

    def test_close_event_confirms_when_running(self, win, dlg):
        win._begin_job("predict", _Slow())
        worker = win._job_worker
        dlg.close_ok = False
        assert win.close() is False
        assert dlg.questions[-1] == "close" and not worker.is_cancelled and worker.isRunning()
        dlg.close_ok = True
        assert win.close() is True
        assert worker.is_cancelled and not worker.isRunning()

    def test_close_event_idle_no_question(self, win, dlg):
        assert win.close() is True and dlg.questions == []

    def test_main_creates_window_and_installs_hook(self, qapp, monkeypatch):
        prev, prev_thread = sys.excepthook, threading.excepthook
        state = dict(mw._HOOK_STATE)
        monkeypatch.setattr(mw, "_warmup_imports", lambda: None)
        QTimer.singleShot(150, qapp.quit)
        try:
            rc = mw.main(["mpm_ml_gui.py"])
            assert rc == 0
            win = mw.LAST_WINDOW
            assert isinstance(win, MainWindow) and win.tabs.count() == 8
            assert sys.excepthook is mw._excepthook and mw._HOOK_STATE["window"] is win
            shutdown(win)
        finally:
            sys.excepthook, threading.excepthook = prev, prev_thread
            mw._HOOK_STATE.update(state)
