# -*- coding: utf-8 -*-
"""mpm.gui.result_tabs testlari (QT_QPA_PLATFORM=offscreen).

Haqiqiy kichik TrainingResult (pipeline.run_training, sintetik loyiha, RF+SVM, n_repeats=1, n_bootstrap=50) va
PredictionOutput (pipeline.run_prediction) - session fixture, bir marta. Tekshiriladi: har tab set_result -> xatosiz,
figures() bo'sh emas; bo'sh holat ("Hali natija yo'q"); MapTab validatsiya va predict_requested; saqlash; BUG-01
regressiyasi (shap (n,p,2) bilan ImportanceTab); MDI zaxirasi yorlig'i; NaN CI; katta xarita stride'i."""
from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from mpm import pipeline  # noqa: E402
from mpm.common import ENSEMBLE_NAME  # noqa: E402
from mpm.config import RunConfig, default_hyperparams  # noqa: E402
from mpm.gui import plots  # noqa: E402
from mpm.gui import result_tabs as rt  # noqa: E402
from mpm.gui.result_tabs import (EMPTY_TEXT, DiagnosticsTab, ImportanceTab, MapTab, ResultsTab,  # noqa: E402
                                 SpatialTab)
from tests.synth import make_synthetic_project  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="session")
def result(tmp_path_factory):
    proj = make_synthetic_project(str(tmp_path_factory.mktemp("rt_proj")), size=80, n_layers=4, n_pos=40, seed=1)
    out = tmp_path_factory.mktemp("rt_out")
    hp = default_hyperparams()
    hp["RandomForest"]["n_estimators"] = 15
    cfg = RunConfig(tiff_folder=proj["tiff"], points_folder=proj["points"], aoi_folder=proj["aoi"],
                    output_dir=str(out), n_background=60, min_distance=300.0, n_splits=3, n_repeats=1,
                    n_bootstrap=50, hyperparams=hp, run_random_cv=True, shap_enabled=True, perm_importance=True,
                    perm_importance_repeats=2, n_jobs=1, seed=5,
                    use_models={"RandomForest": True, "SVM": True, "XGBoost": False, "CNN": False})
    return pipeline.run_training(cfg)


@pytest.fixture(scope="session")
def prediction(result):
    return pipeline.run_prediction(result)


def _ok(tab):
    assert tab.last_error is None, tab.last_error


def _png_or_pdf(paths):
    return sorted(os.path.splitext(p)[1] for p in paths)


# ===========================================================================
# yordamchi (Qt'siz) funksiyalar
# ===========================================================================
class TestHelpers:
    def test_parse_breaks_ok(self):
        assert rt.parse_breaks("0.2, 0.4; 0.6 0.8") == [0.2, 0.4, 0.6, 0.8]
        assert rt.parse_breaks(" 0.5 ") == [0.5]

    @pytest.mark.parametrize("bad", ["", "   ", "a, b", "0.2, 0.2", "0.5, 0.3", "0, 0.5", "0.5, 1", "1.2",
                                     "-0.1, 0.5", "nan", "0,5"])
    def test_parse_breaks_bad(self, bad):
        with pytest.raises(ValueError):
            rt.parse_breaks(bad)

    def test_format_breaks(self):
        assert rt.format_breaks([0.2, 0.4]) == "0.2, 0.4"
        assert rt.format_breaks(None) == ""

    def test_downsample_map(self):
        from affine import Affine
        a = np.arange(4000 * 3000, dtype=np.float32).reshape(4000, 3000)
        tr = Affine(30.0, 0, 1000.0, 0, -30.0, 9000.0)
        b, t2 = rt.downsample_map(a, tr, 1000)
        assert max(b.shape) <= 1000 and b.shape == (1000, 750)
        assert t2.a == 120.0 and t2.e == -120.0 and t2.c == 1000.0 and t2.f == 9000.0
        b2, t3 = rt.downsample_map(a[:100, :100], tr, 1000)
        assert b2.shape == (100, 100) and t3 is tr                    # kichik xarita o'zgarmaydi
        b4, t4 = rt.downsample_map(a, None, 1000)
        assert t4 is None and b4.shape == (1000, 750)

    def test_tuning_dataframe(self):
        rec = {"repeat": 0, "fold": 1, "best_score": 0.8, "base_score": 0.7, "scoring": "roc_auc", "n_trials": 5,
               "best_params": {"n_estimators": 100, "max_depth": None, "x": 0.12345678}}
        res = {"spatial": {"tuned_params": {"RandomForest": [rec]}},
               "final_tuning": {"RandomForest": {"best_score": 0.81, "best_params": {"a": 1}, "fallback": "xato"}}}
        df = rt.tuning_dataframe(res)
        assert len(df) == 2 and list(df["Doira"])[0].startswith("nested")
        assert df.loc[0, "Fold"] == 2 and df.loc[0, "Takror"] == 1
        assert "n_estimators=100" in df.loc[0, "Tanlangan giperparametrlar"]
        assert df.loc[1, "Zaxira"] == "xato" and df.loc[1, "Fold"] is None
        assert rt.tuning_dataframe({}).empty and rt.tuning_dataframe(None).empty

    def test_importance_dataframe_mdi_has_no_std(self):
        res = {"feature_names": ["a", "b", "c"],
               "importance": {"feature_names": ["a", "b", "c"], "models": {
                   "RandomForest": {"mean": [0.1, 0.5, 0.2], "std": [0.0, 0.0, 0.0], "source": "mdi"},
                   "SVM": {"mean": [0.3, 0.1, 0.0], "std": [0.05, 0.02, 0.01], "source": "permutation"}}}}
        df = rt.importance_dataframe(res)
        assert list(df["Feature"]) == ["b", "c", "a"]                 # birinchi model o'rtachasi bo'yicha kamayish
        assert df["RandomForest std"].isna().all()                    # MDI nol std'si ko'rsatilmaydi
        assert df["SVM std"].notna().all()
        assert "RandomForest [mdi] o'rtacha" in df.columns
        assert rt.importance_dataframe({}).columns.tolist() == ["Feature"]

    def test_bg_sensitivity_frames(self):
        bg = {"summary": {"RF": {"mean": 0.8, "std": 0.02, "min": 0.78, "max": 0.83, "values": [0.8, 0.78, 0.83]}},
              "per_draw": [{"draw": 0, "seed": 1, "auc": {"RF": 0.8}, "n_background": 50, "block_size": 900.0}]}
        s, d = rt.bg_sensitivity_frames(bg)
        assert len(s) == 1 and s.loc[0, "Tanlovlar soni"] == 3
        assert d.loc[0, "AUC: RF"] == 0.8 and d.loc[0, "Tanlov"] == 1
        assert rt.bg_sensitivity_frames(None) == (None, None)
        assert rt.bg_sensitivity_frames({"summary": {}}) == (None, None)


# ===========================================================================
# DiagnosticsTab
# ===========================================================================
class TestDiagnosticsTab:
    def test_empty_state(self):
        t = DiagnosticsTab()
        assert not t.has_result() and t.empty_label.text() == EMPTY_TEXT
        assert t.stack.currentIndex() == 0 and t.figures() == {}
        t.set_diagnostics(None)
        t.set_diagnostics({})
        assert not t.has_result()
        _ok(t)

    def test_set_diagnostics(self, result):
        t = DiagnosticsTab()
        t.set_diagnostics(result["diagnostics"], result["data_dictionary"])
        _ok(t)
        assert t.has_result() and t.stack.currentIndex() == 1
        n_bands = len(result["band_names"])
        assert t.stats_table.rowCount() == n_bands
        assert t.vif_table.rowCount() == n_bands
        assert t.dd_table.rowCount() == len(result["data_dictionary"])
        assert t.tabs.isTabEnabled(5) and t.effects_table.rowCount() > 0
        assert "Dataset:" in t.summary_label.text()
        figs = t.figures()
        assert "corr_heatmap" in figs and figs["corr_heatmap"].axes
        t.clear()
        assert not t.has_result() and t.stats_table.rowCount() == 0 and t.figures() == {}

    def test_partial_and_garbage_input(self, result):
        t = DiagnosticsTab()
        diag = {"layer_stats": result["diagnostics"]["layer_stats"], "corr": None, "vif": "yaroqsiz",
                "high_corr_pairs": None}
        t.set_diagnostics(diag)
        _ok(t)
        assert t.has_result() and t.vif_table.rowCount() == 0 and not t.tabs.isTabEnabled(5)
        t.set_diagnostics(None, result["data_dictionary"])             # faqat data dictionary
        assert t.has_result() and t.dd_table.rowCount() > 0
        t.set_diagnostics("matn")                                      # yaroqsiz tur => bo'sh holat
        assert not t.has_result()

    def test_vif_flag_colored(self, result):
        t = DiagnosticsTab()
        diag = dict(result["diagnostics"])
        diag["vif"] = pd.DataFrame({"band": ["a", "b", "c"], "vif": [1.0, 55.0, 12.0],
                                    "flag": ["past", "yuqori", "o'rta"]})
        t.set_diagnostics(diag)
        col = [c for c in range(t.vif_table.columnCount()) if t.vif_table.horizontalHeaderItem(c).text() == "flag"][0]
        colored = {t.vif_table.item(r, col).text(): t.vif_table.item(r, col).background().color().red()
                   for r in range(3)}
        assert colored["yuqori"] == 255 and colored["o'rta"] == 255

    def test_signal_and_busy(self):
        t = DiagnosticsTab()
        got = []
        t.refresh_requested.connect(lambda: got.append(1))
        assert t.btn_refresh.text() == "Tahlilni hisoblash (TIFF'lardan)"
        t.btn_refresh.click()
        assert got == [1]
        t.set_busy(True)
        assert not t.btn_refresh.isEnabled()
        t.btn_refresh.click()
        assert got == [1]
        t.set_busy(False)
        assert t.btn_refresh.isEnabled()


# ===========================================================================
# ResultsTab
# ===========================================================================
class TestResultsTab:
    def test_empty_state(self):
        t = ResultsTab()
        assert not t.has_result() and t.empty_label.text() == EMPTY_TEXT and t.figures() == {}
        for b in (t.btn_table, t.btn_figs, t.btn_export):
            assert not b.isEnabled()
        t.set_result(None)
        t.set_result({})
        t.set_result({"spatial": None})
        assert not t.has_result()
        _ok(t)
        assert t.save_table() is None and t.save_figures() == []

    def test_set_result(self, result):
        t = ResultsTab()
        t.set_result(result)
        _ok(t)
        assert t.has_result() and t.current_mode() == "spatial"
        names = set(result["spatial"]["metrics"])
        assert ENSEMBLE_NAME in names and len(names) == 3             # RF, SVM, ansambl
        assert t.metrics_table.rowCount() == 3
        assert t.model_combo.count() == 3 and t.model_combo.currentData() == ENSEMBLE_NAME
        assert "blok-bootstrap" in t.note_label.text() and "Spatial" in t.note_label.text()
        figs = t.figures()
        assert {"roc_spatial", "pr_spatial", "calibration_spatial", "roc_random", "pr_random"} <= set(figs)
        assert any(k.startswith("confusion_") for k in figs)
        assert all(f.axes for f in figs.values())
        for b in (t.btn_table, t.btn_figs, t.btn_export):
            assert b.isEnabled()
        assert t.roc_canvas.last_error is None and t.conf_canvas.last_error is None

    def test_mode_switch_and_model_combo(self, result):
        t = ResultsTab()
        t.set_result(result)
        t.mode_combo.setCurrentIndex(1)
        _ok(t)
        assert t.current_mode() == "random" and "benchmark" in t.note_label.text()
        rnd_df = result["random"]["metrics_df"]
        assert t.metrics_table.rowCount() == len(rnd_df)
        t.model_combo.setCurrentIndex(0)                              # confusion qayta chiziladi
        _ok(t)
        assert t.conf_canvas.fig.axes
        t.mode_combo.setCurrentIndex(0)
        assert t.current_mode() == "spatial"

    def test_random_disabled_when_missing(self, result):
        t = ResultsTab()
        res = dict(result)
        res["random"] = None
        t.set_result(res)
        _ok(t)
        assert not t.mode_combo.model().item(1).isEnabled()
        assert "roc_random" not in t.figures()
        t.set_result(result)
        assert t.mode_combo.model().item(1).isEnabled()
        t.mode_combo.setCurrentIndex(1)
        t.set_result(res)                                             # random yo'qoldi => spatial'ga qaytadi
        assert t.current_mode() == "spatial"
        _ok(t)

    def test_nan_ci_and_no_bootstrap(self, result):
        t = ResultsTab()
        res = copy.copy(result)
        sp = dict(result["spatial"])
        sp["metrics"] = {k: {**v, "auc_ci95": (float("nan"), float("nan"))} for k, v in sp["metrics"].items()}
        df = sp["metrics_df"].copy()
        df["AUC_CI_lo"] = np.nan
        df["AUC_CI_hi"] = np.nan
        sp["metrics_df"] = df
        res["spatial"] = sp
        t.set_result(res)
        _ok(t)
        assert "CI hisoblanmagan" in t.note_label.text()
        assert t.metrics_table.rowCount() == 3
        txt = t.metrics_table.item(0, list(df.columns).index("AUC_CI_lo")).text()
        assert txt == ""                                               # NaN bo'sh katak

    def test_missing_metrics_df_is_rebuilt(self, result):
        t = ResultsTab()
        res = copy.copy(result)
        sp = {k: v for k, v in result["spatial"].items() if k != "metrics_df"}
        res["spatial"] = sp
        t.set_result(res)
        _ok(t)
        assert t.metrics_table.rowCount() == 3

    def test_note_mentions_cnn_and_tuning(self, result):
        t = ResultsTab()
        res = copy.copy(result)
        sp = dict(result["spatial"])
        sp["metrics"] = {**sp["metrics"], "CNN": dict(next(iter(sp["metrics"].values())))}
        res["spatial"] = sp
        cfg = copy.deepcopy(result["cfg"])
        cfg["tuning"]["enabled"] = True
        cfg["tuning"]["mode"] = "nested"
        res["cfg"] = cfg
        t.set_result(res)
        _ok(t)
        txt = t.note_label.text()
        assert "nested tuning" in txt and "CNN chiqishi kalibrlanmaydi" in txt

    def test_save_table_csv_xlsx(self, result, tmp_path):
        t = ResultsTab()
        msgs = []
        t.message.connect(msgs.append)
        t.set_result(result)
        p = t.save_table(str(tmp_path / "m.csv"))
        assert p and pd.read_csv(p, encoding="utf-8-sig").shape[0] == 3
        px = t.save_table(str(tmp_path / "m.xlsx"))
        sheets = pd.read_excel(px, sheet_name=None)
        assert set(sheets) == {"Spatial block CV", "Random CV"}
        assert any("Jadval saqlandi" in m for m in msgs)

    def test_save_figures_includes_registered_sources(self, result, prediction, tmp_path):
        t = ResultsTab()
        sp = SpatialTab()
        sp.set_result(result)
        t.register_figure_source(sp)
        t.register_figure_source(sp)                                   # takror qo'shilmaydi
        t.register_figure_source(t)                                    # o'zi qo'shilmaydi
        assert len(t._sources) == 1
        t.set_result(result)
        paths = t.save_figures(str(tmp_path / "figs"))
        exts = _png_or_pdf(paths)
        assert exts.count(".png") == exts.count(".pdf") > 5
        names = {os.path.basename(p) for p in paths}
        assert "roc_spatial.png" in names and "spatial_cv_diagnostics.pdf" in names
        for p in paths:
            assert os.path.getsize(p) > 0

    def test_export_signal(self, result):
        t = ResultsTab()
        got = []
        t.export_requested.connect(lambda: got.append(1))
        t.set_result(result)
        t.btn_export.click()
        assert got == [1]
        t.set_busy(True)
        assert not t.btn_export.isEnabled() and not t.mode_combo.isEnabled()
        t.set_busy(False)
        assert t.btn_export.isEnabled()
        t.clear()
        assert not t.has_result() and t.metrics_table.rowCount() == 0 and not t.btn_export.isEnabled()

    def test_garbage_does_not_raise(self, result):
        t = ResultsTab()
        res = copy.copy(result)
        res["spatial"] = {"metrics": "yaroqsiz", "metrics_df": 5}
        t.set_result(res)                                              # istisno tashqariga chiqmaydi
        assert t.has_result() or t.last_error is not None


# ===========================================================================
# SpatialTab
# ===========================================================================
class TestSpatialTab:
    def test_empty_state(self):
        t = SpatialTab()
        assert not t.has_result() and t.empty_label.text() == EMPTY_TEXT and t.figures() == {}
        t.set_result(None)
        t.set_result({"foo": 1})
        assert not t.has_result()
        _ok(t)

    def test_set_result(self, result):
        t = SpatialTab()
        t.set_result(result)
        _ok(t)
        assert t.has_result()
        figs = t.figures()
        assert "spatial_cv_diagnostics" in figs and figs["spatial_cv_diagnostics"].axes
        assert "tuning_trials" not in figs                              # tuning o'chirilgan
        n_folds = len(result["spatial"]["fold_table"]) + len(result["random"]["fold_table"])
        assert t.fold_table.rowCount() == n_folds
        assert t.fold_table.horizontalHeaderItem(0).text() == "CV"
        assert "hisoblanmagan" in t.bg_label.text() and t.bg_table.rowCount() == 0
        assert t.tuning_table.rowCount() == 0 and "bajarilmagan" in t.tuning_note.text()
        n_w = len(result["warnings"])
        assert t.warn_list.count() == n_w and f"({n_w})" in t.tabs.tabText(t.WARN_TAB)
        assert t.diag_canvas.last_error is None

    def test_with_bg_and_tuning(self, result):
        res = copy.copy(result)
        res["bg_sensitivity"] = {"n_draws": 2, "summary": {"RandomForest": {"mean": 0.8, "std": 0.02, "min": 0.78,
                                                                           "max": 0.82, "values": [0.78, 0.82]}},
                                 "per_draw": [{"draw": 0, "seed": 1, "auc": {"RandomForest": 0.78}},
                                              {"draw": 1, "seed": 2, "auc": {"RandomForest": 0.82}}]}
        trials = [{"params": {"n_estimators": 50 + 10 * i, "max_depth": [None, 4, 8][i % 3]}, "score": 0.7 + 0.01 * i,
                   "std": 0.01} for i in range(4)]
        recs = [{"repeat": 0, "fold": f, "best_params": trials[f]["params"], "best_score": 0.75, "base_score": 0.7,
                 "n_trials": 4, "trials": trials, "scoring": "roc_auc"} for f in range(3)]
        sp = dict(result["spatial"])
        sp["tuned_params"] = {"RandomForest": recs}
        res["spatial"] = sp
        res["final_tuning"] = {"RandomForest": {"best_params": {"n_estimators": 60}, "best_score": 0.76,
                                                "base_score": 0.7, "n_trials": 4, "scoring": "roc_auc", "trials": []}}
        t = SpatialTab()
        t.set_result(res)
        _ok(t)
        assert t.bg_table.rowCount() == 1 and t.bg_draw_table.rowCount() == 2
        assert t.tuning_table.rowCount() == 4                           # 3 nested + 1 yakuniy
        assert "Nested tuning" in t.tuning_note.text() and "HAR DOIM" in t.tuning_note.text()
        figs = t.figures()
        assert "tuning_trials" in figs and len(figs["tuning_trials"].axes) >= 2
        assert t.tuning_canvas.last_error is None
        t.clear()
        assert not t.has_result() and t.tuning_table.rowCount() == 0 and t.warn_list.count() == 0

    def test_empty_fold_warning_note(self, result):
        res = copy.copy(result)
        sp = dict(result["spatial"])
        ft = sp["fold_table"].copy()
        ft.loc[0, "pos_val"] = 0
        sp["fold_table"] = ft
        res["spatial"] = sp
        t = SpatialTab()
        t.set_result(res)
        assert "musbat nuqtasiz" in t.fold_note.text()

    def test_garbage_does_not_raise(self, result):
        t = SpatialTab()
        res = copy.copy(result)
        res["warnings"] = None
        res["bg_sensitivity"] = {"summary": "yaroqsiz"}
        res["spatial"] = {**result["spatial"], "tuned_params": {"RandomForest": "yaroqsiz"}}
        t.set_result(res)
        assert t.has_result()


# ===========================================================================
# ImportanceTab
# ===========================================================================
class TestImportanceTab:
    def test_empty_state(self):
        t = ImportanceTab()
        assert not t.has_result() and t.empty_label.text() == EMPTY_TEXT and t.figures() == {}
        t.set_result(None)
        t.set_result({"foo": 1})
        assert not t.has_result()
        _ok(t)

    def test_set_result(self, result):
        t = ImportanceTab()
        t.set_result(result)
        _ok(t)
        assert t.has_result()
        assert "permutation" in t.method_label.text().lower()
        shap_models = [k for k, v in (result.get("shap") or {}).items()]
        assert t.bee_combo.count() == len(shap_models) == t.dep_model_combo.count()
        figs = t.figures()
        assert "importance" in figs and figs["importance"].axes
        if shap_models:
            assert f"shap_beeswarm_{shap_models[0]}" in figs and f"shap_dependence_{shap_models[0]}" in figs
            assert "SHAP birliklari modelga bog'liq" in t.units_label.text()
            assert t.dep_feature_combo.count() == len(result["feature_names"]) + 1
            assert t.dep_feature_combo.itemData(0) is None
        assert t.imp_table.rowCount() == len(result["feature_names"])
        assert "OOF" in t.units_label.text()

    def test_feature_and_model_selection_redraw(self, result):
        t = ImportanceTab()
        t.set_result(result)
        if not t.bee_combo.count():
            pytest.skip("shap o'rnatilmagan")
        t.dep_feature_combo.setCurrentIndex(2)
        t.dep_model_combo.setCurrentIndex(0)
        t.bee_combo.setCurrentIndex(0)
        _ok(t)
        assert "SHAP dependence" in t.dep_canvas.fig.axes[0].get_title()
        feat = t.dep_feature_combo.currentData()
        assert feat in result["feature_names"] or feat is None

    def test_bug01_shap_n_p_2(self, result):
        """BUG-01: yangi shap RF uchun (n, p, 2) qaytaradi - tab abort qilmasligi va chizishi kerak."""
        if not result.get("shap"):
            pytest.skip("shap o'rnatilmagan")
        res = copy.copy(result)
        shap3 = {}
        for name, e in result["shap"].items():
            sv = np.asarray(e["shap_values"])
            shap3[name] = {**e, "shap_values": np.stack([-sv, sv], axis=-1)}   # (n, p, 2)
        res["shap"] = shap3
        t = ImportanceTab()
        t.set_result(res)
        _ok(t)
        name = next(iter(shap3))
        assert t.bee_canvas.fig.axes and t.dep_canvas.fig.axes
        assert "Ma'lumot yo'q" not in " ".join(tx.get_text() for ax in t.bee_canvas.fig.axes for tx in ax.texts)
        figs = t.figures()
        assert f"shap_beeswarm_{name}" in figs and figs[f"shap_beeswarm_{name}"].axes
        # list ko'rinishi ([neg, pos]) ham
        res2 = copy.copy(result)
        res2["shap"] = {name: {**e, "shap_values": [-np.asarray(e["shap_values"]), np.asarray(e["shap_values"])]}
                        for name, e in result["shap"].items()}
        t.set_result(res2)
        _ok(t)

    def test_mdi_fallback_labelled(self, result):
        """MDI zaxirasi: sarlavha 'MDI/gain', xato chiziqlari yo'q, jadvalda std bo'sh."""
        res = copy.copy(result)
        imp = copy.deepcopy(result["importance"])
        rf = imp["models"]["RandomForest"]
        p = len(result["feature_names"])
        imp["models"]["RandomForest"] = {"mean": np.abs(np.asarray(rf["mean"], dtype=float)), "std": np.zeros(p),
                                         "source": "mdi"}
        res["importance"] = imp
        t = ImportanceTab()
        t.set_result(res)
        _ok(t)
        axes = t.imp_canvas.fig.axes
        titles = [ax.get_title() for ax in axes]
        assert any("MDI/gain importance" in s and "RandomForest" in s for s in titles)
        assert any("Permutation importance" in s for s in titles)       # SVM hali permutation
        mdi_ax = next(ax for ax in axes if "MDI/gain importance" in ax.get_title())
        assert "MDI/gain" in mdi_ax.get_xlabel()
        assert all(getattr(c, "errorbar", None) is None or not any(
            (a is not None) and a.axes is not None for part in c.errorbar.lines
            for a in (part if isinstance(part, (tuple, list)) else [part])) for c in mdi_ax.containers)
        assert "MDI/gain zaxirasi" in t.units_label.text() and "RandomForest" in t.units_label.text()
        col = "RandomForest std"
        assert t.imp_table.dataframe()[col].isna().all()
        assert "importance" in t.figures()

    def test_no_importance_no_shap(self, result):
        """Faqat-SVM / perm o'chiq: importance['models'] bo'sh, shap None - xato emas."""
        res = copy.copy(result)
        res["importance"] = {"method": "hisoblanmadi", "models": {}, "feature_names": list(result["feature_names"])}
        res["shap"] = None
        spatial = dict(result["spatial"])
        spatial["perm_importance"] = {}
        res["spatial"] = spatial
        t = ImportanceTab()
        t.set_result(res)
        _ok(t)
        assert t.has_result() and "hisoblanmadi" in t.method_label.text()
        assert not t.bee_combo.isEnabled() and t.bee_combo.count() == 0
        assert "SHAP hisoblanmagan" in t.units_label.text()
        assert t.imp_table.rowCount() == 0
        assert t.figures() == {}                  # importance ham, SHAP ham yo'q: bo'sh "Ma'lumot yo'q" rasm saqlanmasin

    def test_clear(self, result):
        t = ImportanceTab()
        t.set_result(result)
        t.clear()
        assert not t.has_result() and t.bee_combo.count() == 0 and t.imp_table.rowCount() == 0
        assert t.figures() == {}


# ===========================================================================
# MapTab
# ===========================================================================
class TestMapTab:
    def test_empty_state_and_buttons(self):
        t = MapTab()
        assert not t.has_result() and t.empty_label.text() == EMPTY_TEXT and t.figures() == {}
        assert not t.btn_predict.isEnabled() and not t.btn_export.isEnabled()
        assert "prospektivlik indeksi" in t.index_note.text() and "ehtimollik emas" in t.index_note.text()
        assert t.btn_predict.text() == "Prognoz xarita yaratish"
        assert t.btn_export.text() == "Natijalarni eksport (CSV/JSON/XLSX)"
        t.set_prediction(None)
        t.set_prediction({})
        t.set_prediction({"maps": {}})
        t.set_result(None)
        assert not t.has_result()
        _ok(t)

    def test_set_result_defaults_and_overlays(self, result):
        t = MapTab()
        t.set_result(result)
        _ok(t)
        assert t.btn_predict.isEnabled() and t.btn_export.isEnabled()
        assert t.method_combo.currentData() == result["cfg"]["class_method"]
        assert t.n_classes_spin.value() == result["cfg"]["n_classes"]
        assert t.out_picker.path() == result["cfg"]["output_dir"]
        assert t.cb_aoi.isEnabled() and t.cb_pos.isEnabled() and t.cb_bg.isEnabled()
        assert not t.has_result()                                       # prognoz hali yo'q
        assert "Prognoz xarita yaratish" in t.hint_label.text()
        assert t.cnn_note.isHidden()

    def test_validation_and_predict_signal(self, result, tmp_path):
        t = MapTab()
        t.set_result(result)
        got = []
        t.predict_requested.connect(got.append)
        t.out_picker.setPath(str(tmp_path / "pred"))
        # quantile
        t.method_combo.setCurrentIndex(t.method_combo.findData("quantile"))
        t.n_classes_spin.setValue(4)
        t.btn_predict.click()
        assert got == [{"out_dir": str(tmp_path / "pred"), "class_method": "quantile", "n_classes": 4,
                        "class_breaks": None}]
        assert t.validation_label.isHidden()
        # fixed + noto'g'ri chegaralar => signal yo'q, xato ko'rinadi
        got.clear()
        t.method_combo.setCurrentIndex(t.method_combo.findData("fixed"))
        assert t.breaks_edit.isEnabled() and not t.n_classes_spin.isEnabled()
        for bad in ("", "abc", "0.5, 0.3", "0.2, 1.5"):
            t.breaks_edit.setText(bad)
            t.btn_predict.click()
            assert got == [] and not t.validation_label.isHidden() and t.validation_label.text()
        # to'g'ri fixed
        t.breaks_edit.setText("0.3, 0.6, 0.9")
        assert t.validation_label.isHidden()                            # matn o'zgarganda xato yashiriladi
        t.btn_predict.click()
        assert got == [{"out_dir": str(tmp_path / "pred"), "class_method": "fixed", "n_classes": 4,
                        "class_breaks": [0.3, 0.6, 0.9]}]
        # equal_interval
        got.clear()
        t.method_combo.setCurrentIndex(t.method_combo.findData("equal_interval"))
        assert t.n_classes_spin.isEnabled() and not t.breaks_edit.isEnabled()
        t.btn_predict.click()
        assert got[0]["class_method"] == "equal_interval" and got[0]["class_breaks"] is None

    def test_out_dir_is_file_rejected(self, result, tmp_path):
        t = MapTab()
        t.set_result(result)
        f = tmp_path / "fayl.txt"
        f.write_text("x")
        t.out_picker.setPath(str(f))
        s, err = t.collect_settings()
        assert s is None and "papka emas" in err
        got = []
        t.predict_requested.connect(got.append)
        t.btn_predict.click()
        assert got == [] and not t.validation_label.isHidden()

    def test_set_prediction(self, result, prediction):
        t = MapTab()
        t.set_result(result)
        t.set_prediction(prediction)
        _ok(t)
        assert t.has_result() and t.stack.currentIndex() == 1
        keys = [t.layer_combo.itemData(i) for i in range(t.layer_combo.count())]
        assert keys[0] == f"map:{ENSEMBLE_NAME}"
        assert {"map:RandomForest", "map:SVM", "uncertainty", "classes"} <= set(keys)
        assert t.layer_combo.currentData() == f"map:{ENSEMBLE_NAME}"
        # barcha qatlamlar xatosiz chiziladi (overlay yoqilgan)
        for cb in (t.cb_aoi, t.cb_pos, t.cb_bg):
            cb.setChecked(True)
        for i in range(t.layer_combo.count()):
            t.layer_combo.setCurrentIndex(i)
            _ok(t)
            assert t.map_canvas.last_error is None
            texts = " ".join(tx.get_text() for ax in t.map_canvas.fig.axes for tx in ax.texts)
            assert "Ma'lumot yo'q" not in texts
        assert t.map_canvas.fig.axes
        assert t.stats_table.rowCount() == len(prediction["class_stats"])
        assert "prospektivlik indeksi" in t.stats_note.text()
        figs = t.figures()
        assert {"map_RandomForest", "map_uncertainty", "map_classes", "success_rate"} <= set(figs)
        assert all(f.axes for f in figs.values())
        t.clear()
        assert not t.has_result() and not t.btn_predict.isEnabled() and t.figures() == {}
        assert t.stats_table.rowCount() == 0

    def test_set_result_clears_old_prediction(self, result, prediction):
        t = MapTab()
        t.set_result(result)
        t.set_prediction(prediction)
        assert t.has_result()
        t.set_result(result)
        assert not t.has_result() and t.layer_combo.count() == 0

    def test_class_stats_nan_blank(self, result, prediction):
        t = MapTab()
        t.set_result(result)
        pred = dict(prediction)
        stats = prediction["class_stats"].copy()
        stats["Konlar_soni"] = np.nan
        stats["Konlar_%"] = np.nan
        stats["Boyitish"] = np.nan
        pred["class_stats"] = stats
        t.set_prediction(pred)
        _ok(t)
        cols = list(stats.columns)
        for name in ("Konlar_soni", "Konlar_%", "Boyitish"):
            c = cols.index(name)
            assert all(t.stats_table.item(r, c).text() == "" for r in range(t.stats_table.rowCount()))
        c = cols.index("Maydon_km2")
        assert t.stats_table.item(0, c).text() != ""

    def test_prediction_without_optional_parts(self, result, prediction):
        t = MapTab()
        t.set_result(result)
        pred = {"maps": prediction["maps"], "uncertainty": None, "class_map": None, "class_stats": None,
                "success_curve": None}
        t.set_prediction(pred)
        _ok(t)
        keys = [t.layer_combo.itemData(i) for i in range(t.layer_combo.count())]
        assert "uncertainty" not in keys and "classes" not in keys
        assert t.stats_table.rowCount() == 0

    def test_prediction_without_training_result_uses_pred_transform(self, result, prediction):
        """ApplyBundle natijasi: 'transform' pred ichida, overlay (AOI/nuqta) yo'q."""
        t = MapTab()
        pred = dict(prediction)
        pred["transform"] = result["raster"].transform
        t.set_prediction(pred)
        _ok(t)
        assert t.has_result() and not t.cb_aoi.isEnabled()
        assert t.map_canvas.fig.axes
        assert t.figures()

    def test_busy(self, result):
        t = MapTab()
        t.set_result(result)
        t.set_busy(True)
        assert t.is_busy()
        assert not (t.btn_predict.isEnabled() or t.btn_export.isEnabled() or t.out_picker.isEnabled())
        assert not t.method_combo.isEnabled() and not t.n_classes_spin.isEnabled()
        t.set_busy(False)
        assert t.btn_predict.isEnabled() and t.btn_export.isEnabled() and t.out_picker.isEnabled()
        assert t.n_classes_spin.isEnabled()

    def test_export_signal(self, result):
        t = MapTab()
        got = []
        t.export_requested.connect(lambda: got.append(1))
        t.btn_export.click()
        assert got == []                                                # natija yo'q: o'chiq
        t.set_result(result)
        t.btn_export.click()
        assert got == [1]

    def test_big_map_is_downsampled(self, result, prediction, monkeypatch):
        """Katta xarita draw_map ga stride bilan kamaytirilgan holda beriladi (<= MAP_MAX_PX_SHOW)."""
        seen = []
        orig = plots.draw_map

        def spy(fig, array, *a, **k):
            seen.append(np.asarray(array).shape)
            return orig(fig, array, *a, **k)

        monkeypatch.setattr(plots, "draw_map", spy)
        monkeypatch.setattr(rt, "MAP_MAX_PX_SHOW", 40)
        t = MapTab()
        t.set_result(result)
        t.set_prediction(prediction)
        _ok(t)
        assert seen and max(max(s) for s in seen) <= 40
        h, w = next(iter(prediction["maps"].values())).shape
        assert max(h, w) > 40

    def test_cnn_note_visible(self, result):
        res = copy.copy(result)
        res["final_models"] = {**result["final_models"], "CNN": []}
        t = MapTab()
        t.set_result(res)
        assert not t.cnn_note.isHidden()

    def test_figures_class_map_all_nan_without_breaks(self, result, prediction):
        """Regressiya: class_breaks yo'q va class_map butunlay NaN => figures() ValueError/RuntimeWarning bermasin."""
        import warnings
        t = MapTab()
        t.set_result(result)
        pred = dict(prediction)
        pred["class_breaks"] = None
        pred["class_stats"] = None
        pred["class_map"] = np.full_like(np.asarray(prediction["class_map"], dtype=float), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            t.set_prediction(pred)
            figs = t.figures()
        _ok(t)
        assert "map_classes" in figs and rt._n_classes_of(pred["class_map"], None) == 0
        assert rt._n_classes_of(np.array([[1.0, 3.0, np.nan]]), None) == 3
        assert rt._n_classes_of("x", None) == 0 and rt._n_classes_of(None, [0.5]) == 2

    def test_overlay_only_for_same_grid(self, result, prediction):
        """Regressiya: bundle boshqa to'rga qo'llanganda o'qitishning AOI/konlari ustiga chizilmasin."""
        from affine import Affine
        tr = result["raster"].transform
        same = dict(prediction, transform=tr)
        shifted = dict(prediction, transform=Affine(tr.a, tr.b, tr.c + 5000.0, tr.d, tr.e, tr.f))
        t = MapTab()
        t.set_result(result)
        t.set_prediction(same)
        assert t.cb_aoi.isEnabled() and t._overlay_kwargs(tr)["aoi_gdf"] is not None
        t.set_prediction(shifted)
        _ok(t)
        assert not t.cb_aoi.isEnabled() and not t.cb_pos.isEnabled()
        assert t._overlay_kwargs(tr) == {} and t._overlay_kwargs(tr, use_checks=False) == {}
        assert t.figures()                                              # overlaysiz ham saqlanadi
        t.set_prediction(prediction)                                    # o'zining transform'i yo'q => o'qitish to'ri
        assert t.cb_aoi.isEnabled()


# ===========================================================================
# Umumiy chidamlilik
# ===========================================================================
class TestRobustness:
    def test_draw_helper_survives_exception(self):
        from mpm.gui.widgets import MplCanvas
        c = MplCanvas()

        def boom(fig, *a, **k):
            raise TypeError("sinov xatosi")

        rt._draw(c, boom)                                               # istisno tashqariga chiqmaydi
        texts = " ".join(tx.get_text() for ax in c.fig.axes for tx in ax.texts)
        assert "sinov xatosi" in texts

    def test_slot_guard_records_error(self, result, monkeypatch):
        t = ResultsTab()

        def boom(*a, **k):
            raise RuntimeError("jadval buzildi")

        monkeypatch.setattr(t.metrics_table, "set_dataframe", boom)
        msgs = []
        t.message.connect(msgs.append)
        t.set_result(result)                                            # xato ushlanadi, qolgan qismlar chiziladi
        assert t.last_error and "jadval buzildi" in t.last_error
        assert not t.error_label.isHidden() and msgs
        assert t.roc_canvas.fig.axes

    def test_all_tabs_accept_all_inputs_without_raising(self, result, prediction):
        tabs = [DiagnosticsTab(), ResultsTab(), SpatialTab(), ImportanceTab(), MapTab()]
        junk = [None, {}, [], "x", 5, {"spatial": []}, {"importance": 3, "shap": "x"}]
        for t in tabs:
            for j in junk:
                if isinstance(t, DiagnosticsTab):
                    t.set_diagnostics(j)
                else:
                    t.set_result(j)
            t.set_busy(True)
            t.set_busy(False)
            t.clear()
            assert t.figures() == {}
        tabs[-1].set_prediction("x")
        tabs[-1].set_prediction({"maps": {"a": "yaroqsiz"}})            # yaroqsiz massiv: xato emas

    def test_figures_are_independent_of_ui(self, result, prediction):
        """figures() har chaqiruvda yangi Figure qaytaradi (ekran canvas'i emas) va saqlanadi."""
        t = SpatialTab()
        t.set_result(result)
        f1, f2 = t.figures(), t.figures()
        assert f1["spatial_cv_diagnostics"] is not f2["spatial_cv_diagnostics"]
        assert f1["spatial_cv_diagnostics"] is not t.diag_canvas.fig

    def test_all_tabs_full_flow(self, result, prediction, tmp_path):
        tabs = {"diag": DiagnosticsTab(), "res": ResultsTab(), "sp": SpatialTab(), "imp": ImportanceTab(),
                "map": MapTab()}
        tabs["diag"].set_diagnostics(result["diagnostics"], result["data_dictionary"])
        for k in ("res", "sp", "imp", "map"):
            tabs[k].set_result(result)
        tabs["map"].set_prediction(prediction)
        for t in tabs.values():
            _ok(t)
            assert t.figures()
        for k in ("diag", "sp", "imp", "map"):
            tabs["res"].register_figure_source(tabs[k])
        paths = tabs["res"].save_figures(str(tmp_path / "all"), formats=("png",), dpi=60)
        names = {os.path.basename(p) for p in paths}
        assert {"corr_heatmap.png", "importance.png", "spatial_cv_diagnostics.png", "success_rate.png",
                "roc_spatial.png"} <= names


# ===========================================================================
# GUI qotmasligi: tembel (lazy) chizish; ixcham layout; std izohi (e2e tuzatishlari)
# ===========================================================================
def _spin(ms=400):
    from PyQt5.QtTest import QTest
    QTest.qWait(ms)


def _wait_until(pred, timeout_ms=15000):
    from PyQt5.QtTest import QTest
    left = timeout_ms
    while not pred() and left > 0:
        QTest.qWait(20)
        left -= 20
    return pred()


class TestLazyFill:
    def test_default_is_eager(self, result):
        """lazy_fill=False (standart): hammasi darhol chiziladi, kutayotgan canvas yo'q."""
        for t in (ResultsTab(), SpatialTab(), ImportanceTab()):
            t.set_result(result)
            assert t.pending_count() == 0 and t.lazy_fill is False
        t = SpatialTab()
        t.set_result(result)
        assert t.diag_canvas.fig.axes

    def test_lazy_defers_heavy_drawing_until_flush(self, result):
        """Tuzatishsiz (lazy_fill yo'q) set_result canvas'larni darhol chizadi => bu test yiqiladi."""
        t = SpatialTab()
        t.lazy_fill = True
        t.set_result(result)
        t.lazy_fill = False
        _ok(t)
        assert t.has_result() and t.fold_table.rowCount() > 0           # yengil qismlar (jadval) darhol
        assert t.pending_count() >= 1 and not t.diag_canvas.fig.axes     # og'ir grafik hali chizilmagan
        assert t.flush() >= 1 and t.pending_count() == 0
        assert t.diag_canvas.fig.axes and t.diag_canvas.last_error is None

    @pytest.mark.parametrize("make", [lambda r: (ImportanceTab(), "imp_canvas"), lambda r: (ResultsTab(), "roc_canvas")])
    def test_lazy_then_flush_equals_eager(self, result, make):
        lazy, name = make(result)
        eager, _ = make(result)
        lazy.lazy_fill = True
        lazy.set_result(result)
        lazy.lazy_fill = False
        eager.set_result(result)
        assert lazy.pending_count() >= 1
        lazy.flush()
        la, ea = getattr(lazy, name).fig.axes, getattr(eager, name).fig.axes
        assert len(la) == len(ea) > 0
        assert [a.get_title() for a in la] == [a.get_title() for a in ea]

    def test_flush_one_draws_one_canvas(self, result):
        t = ImportanceTab()
        t.lazy_fill = True
        t.set_result(result)
        t.lazy_fill = False
        n = t.pending_count()
        assert n >= 2                                                   # importance + beeswarm + dependence
        assert t.flush_one() == n - 1

    def test_new_result_or_clear_drops_stale_pending(self, result):
        t = ImportanceTab()
        t.lazy_fill = True
        t.set_result(result)
        t.lazy_fill = False
        assert t.pending_count() >= 1
        t.clear()
        assert t.pending_count() == 0 and t.flush() == 0

    def test_visible_canvas_is_drawn_without_flush(self, result):
        """Canvas ko'rsatilganda (tab ochilganda) kechiktirilgan chizish o'zi bajariladi (event loop orqali)."""
        t = SpatialTab()
        t.resize(1000, 600)
        t.lazy_fill = True
        t.set_result(result)
        t.lazy_fill = False
        assert not t.diag_canvas.fig.axes
        t.show()
        try:
            assert _wait_until(lambda: t.diag_canvas.fig.axes and t.visible_pending_count() == 0)
            assert t.pending_count() >= 1                              # ko'rinmagan sahifadagi tuning canvas kutadi
            t.tabs.setCurrentIndex(3)                                   # "Tuning (nested)" sahifasi ochildi
            assert _wait_until(lambda: t.pending_count() == 0 and t.visible_pending_count() == 0)
            _ok(t)
        finally:
            t.close()

    def test_map_prediction_lazy(self, result, prediction):
        t = MapTab()
        t.set_result(result)
        t.lazy_fill = True
        t.set_prediction(prediction)
        t.lazy_fill = False
        _ok(t)
        assert t.has_result() and t.pending_count() >= 1 and t.stats_table.rowCount() > 0
        t.flush()
        texts = " ".join(tx.get_text() for ax in t.map_canvas.fig.axes for tx in ax.texts)
        assert t.map_canvas.fig.axes and "Ma'lumot yo'q" not in texts

    def test_diagnostics_lazy(self, result):
        t = DiagnosticsTab()
        t.lazy_fill = True
        t.set_diagnostics(result["diagnostics"], result["data_dictionary"])
        t.lazy_fill = False
        assert t.has_result() and t.pending_count() == 1 and not t.corr_canvas.fig.axes
        t.flush()
        assert t.corr_canvas.fig.axes

    def test_deferred_error_is_contained(self, result, monkeypatch):
        t = ImportanceTab()

        def boom(fig, *a, **k):
            raise RuntimeError("chizish yiqildi")

        monkeypatch.setattr(plots, "draw_importance", boom)
        t.lazy_fill = True
        t.set_result(result)
        t.lazy_fill = False
        t.flush()                                                       # istisno tashqariga chiqmaydi
        texts = " ".join(tx.get_text() for ax in t.imp_canvas.fig.axes for tx in ax.texts)
        assert "chizish yiqildi" in texts


class TestCompactLayout:
    def _shown(self, tab, w=1024, h=600):
        tab.resize(w, h)
        tab.show()
        _spin(120)

    def test_no_nested_scroll_and_canvas_fits_1024x700(self, result, prediction):
        """Natija tab'lari 1024x700 oyna ichida (viewport ~ 600px) yagona scroll'siz sig'adi: tana minimal balandligi
        viewport'dan kichik (canvas pastki qismi va x yorliqlari ko'rinadi)."""
        tabs = {"res": ResultsTab(), "sp": SpatialTab(), "imp": ImportanceTab(), "map": MapTab(), "diag": DiagnosticsTab()}
        tabs["diag"].set_diagnostics(result["diagnostics"], result["data_dictionary"])
        for k in ("res", "sp", "imp", "map"):
            tabs[k].set_result(result)
        tabs["map"].set_prediction(prediction)
        for name, t in tabs.items():
            self._shown(t, 1000, 612)
            sub = getattr(t, "plot_tabs", None) or getattr(t, "tabs", None)
            for j in range(sub.count()):
                sub.setCurrentIndex(j)
                _spin(40)
                assert t.scroll.verticalScrollBar().maximum() == 0, (name, j, t.scroll.verticalScrollBar().maximum())
            t.close()

    def test_inner_tabs_do_not_propagate_height_for_width(self):
        from PyQt5.QtWidgets import QLabel
        tw = rt._Tabs()
        lab = QLabel("so'z " * 80)
        lab.setWordWrap(True)
        tw.addTab(lab, "a")
        assert tw.hasHeightForWidth() is False and tw.heightForWidth(300) == -1

    def test_long_notes_are_collapsed_but_keep_text(self, result):
        t = ResultsTab()
        t.set_result(result)
        assert "blok-bootstrap" in t.note_label.text()                 # matn saqlanadi (testlar/ma'lumot)
        assert t.note_box.button.isChecked() is False and t.note_label.isHidden()
        t.note_box.button.setChecked(True)
        assert not t.note_label.isHidden()

    def test_spatial_warning_tab_index(self, result):
        t = SpatialTab()
        t.set_result(result)
        assert t.tabs.tabText(0) == "Diagnostika grafigi"
        assert t.tabs.tabText(t.WARN_TAB).startswith("Ogohlantirishlar")

    def test_tables_have_small_minimum_height(self):
        assert rt._table(220).minimumHeight() <= 100 and rt._table().minimumHeight() <= 100

    def test_canvas_minimum_is_small(self):
        c = rt._canvas()
        assert c.canvas.minimumHeight() <= 260 and c.canvas.minimumWidth() <= 520


class TestStdNote:
    def _res(self, result, std):
        res = copy.copy(result)
        sp = dict(result["spatial"])
        sp["metrics"] = {k: {**v, "auc_std": std} for k, v in sp["metrics"].items()}
        df = sp["metrics_df"].copy()
        df["AUC_std"] = std
        sp["metrics_df"] = df
        res["spatial"] = sp
        return res, list(df.columns).index("AUC_std")

    def test_single_repeat_std_is_blank_with_note(self, result):
        t = ResultsTab()
        res, col = self._res(result, float("nan"))
        t.set_result(res)
        _ok(t)
        assert t.metrics_table.item(0, col).text() == ""                 # NaN bo'sh katak ("0.000" emas)
        assert not t.std_note.isHidden() and "1 takror: std aniqlanmagan" in t.std_note.text()

    def test_defined_std_shows_no_note(self, result):
        t = ResultsTab()
        res, col = self._res(result, 0.0123)
        t.set_result(res)
        assert t.metrics_table.item(0, col).text() == "0.012"
        assert t.std_note.isHidden() and t.std_note.text() == ""
        t.set_result(self._res(result, float("nan"))[0])                 # qayta NaN => izoh qaytadi
        assert not t.std_note.isHidden()
        t.clear()
        assert t.std_note.isHidden()


class TestImportanceNoRelabelHack:
    def test_relabel_hack_removed_and_plots_handle_mdi(self):
        assert not hasattr(rt, "_relabel_mdi") and not hasattr(rt, "_draw_importance")
        res = {"feature_names": ["a", "b", "c"],
               "importance": {"method": "mdi", "feature_names": ["a", "b", "c"], "models": {
                   "RandomForest": {"mean": [0.1, 0.5, 0.2], "std": [0.0, 0.0, 0.0], "source": "mdi"}}}}
        t = ImportanceTab()
        t.set_result(res)
        _ok(t)
        titles = " ".join(a.get_title() for a in t.imp_canvas.fig.axes)
        assert "MDI/gain importance (zaxira)" in titles                  # plots.draw_importance o'zi to'g'ri yozadi
        assert "MDI" in t.imp_canvas.fig.axes[0].get_xlabel()
        assert "MDI/gain importance (zaxira)" in " ".join(a.get_title() for a in t.figures()["importance"].axes)
