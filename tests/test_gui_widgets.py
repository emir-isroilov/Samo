# -*- coding: utf-8 -*-
"""mpm.gui.widgets va mpm.gui.param_panel testlari (QT_QPA_PLATFORM=offscreen).

widgets: FolderPicker, MplCanvas, DataFrameTable, ProgressPanel, LogView.
param_panel: har bir PARAM_SPECS parametri uchun vidjet (komplektlik), get/set roundtrip, reset, preset (fayl),
tuning get/set, SearchSpaceDialog (tahrirlash va noto'g'ri oraliq), CNN mode bog'liqligi, signal'lar."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,  # noqa: E402
                             QSpinBox)

from mpm import config  # noqa: E402
from mpm.config import (PARAM_SPECS, ParamSpec, TuningConfig, default_hyperparams,  # noqa: E402
                        default_search_space)
from mpm.gui import param_panel as pp  # noqa: E402
from mpm.gui.param_panel import HyperParamPanel, SearchSpaceDialog, TuningGroup  # noqa: E402
from mpm.gui.widgets import (DataFrameTable, FolderPicker, LogView, MplCanvas, ProgressPanel,  # noqa: E402
                             format_duration)

_APP = None


@pytest.fixture(scope="module", autouse=True)
def qapp():
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    elif not isinstance(app, QApplication):
        pytest.skip("QCoreApplication allaqachon yaratilgan (QApplication kerak)")
    _APP = app
    return app


# ===========================================================================
# widgets
# ===========================================================================
class TestFolderPicker:
    def test_path_roundtrip_and_signal(self):
        fp = FolderPicker("Papka:")
        got = []
        fp.changed.connect(got.append)
        assert fp.path() == ""
        fp.setPath("/tmp/abc")
        assert fp.path() == "/tmp/abc"
        assert got == ["/tmp/abc"]
        fp.setPath("  /x y  ")
        assert fp.path() == "/x y"                      # bo'sh joylar qirqiladi
        fp.setPath(None)
        assert fp.path() == ""
        assert fp.label.text() == "Papka:"

    def test_browse_sets_path(self, monkeypatch):
        from PyQt5.QtWidgets import QFileDialog
        fp = FolderPicker("P")
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: "/data/new"))
        fp.browse()
        assert fp.path() == "/data/new"
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: ""))
        fp.browse()                                     # bekor qilindi => o'zgarmaydi
        assert fp.path() == "/data/new"


class TestMplCanvas:
    def test_draw_clear_save(self, tmp_path):
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        mc = MplCanvas()
        assert isinstance(mc.toolbar, NavigationToolbar2QT)
        ax = mc.fig.add_subplot(111)
        ax.plot([0, 1], [1, 0])
        mc.redraw()
        assert mc.last_error is None
        assert mc.save_figure(str(tmp_path / "f.png"), dpi=50).endswith("f.png")
        assert (tmp_path / "f.png").stat().st_size > 0
        mc.clear()
        assert len(mc.fig.axes) == 0

    def test_redraw_error_does_not_raise(self):
        mc = MplCanvas(with_toolbar=False)
        assert mc.toolbar is None
        ax = mc.fig.add_subplot(111)
        ax.add_artist(_bad_artist())
        mc.redraw()                                     # istisno ichkarida yutiladi
        assert mc.last_error is not None


def _bad_artist():
    """draw paytida xato beradigan artist."""
    from matplotlib.artist import Artist

    class Bad(Artist):
        def draw(self, renderer):
            raise RuntimeError("buzuq artist")
    return Bad()


class TestDataFrameTable:
    def _df(self):
        return pd.DataFrame({"Model": ["RF", "SVM", "XGB"], "AUC": [0.91234, 0.8, float("nan")],
                             "n": [10, 2, 33], "ok": [True, False, True]})

    def test_set_dataframe_contents(self):
        t = DataFrameTable()
        t.set_dataframe(self._df())
        assert t.rowCount() == 3 and t.columnCount() == 4
        assert [t.horizontalHeaderItem(i).text() for i in range(4)] == ["Model", "AUC", "n", "ok"]
        assert t.item(0, 1).text() == "0.912"
        assert t.item(2, 1).text() == ""                # NaN => bo'sh
        assert t.item(0, 2).text() == "10"              # int formatlanmaydi
        assert t.item(1, 3).text() == "Yo'q"
        t.set_dataframe(self._df(), float_fmt="{:.1f}")
        assert t.item(0, 1).text() == "0.9"
        t.set_dataframe(self._df(), float_fmt={"AUC": "{:.2f}"})
        assert t.item(0, 1).text() == "0.91"

    def test_empty_and_none(self):
        t = DataFrameTable()
        t.set_dataframe(None)
        assert t.rowCount() == 0 and t.columnCount() == 0
        t.set_dataframe(pd.DataFrame())
        assert t.rowCount() == 0
        t.set_dataframe(pd.DataFrame({"a": []}))
        assert t.rowCount() == 0 and t.columnCount() == 1

    def test_numeric_sort(self):
        t = DataFrameTable()
        t.set_dataframe(pd.DataFrame({"n": [10, 2, 33, 4]}))
        t.sortItems(0, Qt.AscendingOrder)
        assert [t.item(i, 0).text() for i in range(4)] == ["2", "4", "10", "33"]    # matn tartibi EMAS
        t.sortItems(0, Qt.DescendingOrder)
        assert [t.item(i, 0).text() for i in range(4)] == ["33", "10", "4", "2"]

    def test_sort_with_nan_and_text(self):
        t = DataFrameTable()
        t.set_dataframe(pd.DataFrame({"x": [0.5, float("nan"), 0.1]}))
        t.sortItems(0, Qt.AscendingOrder)
        assert [t.item(i, 0).text() for i in range(3)] == ["", "0.100", "0.500"]
        t.set_dataframe(pd.DataFrame({"s": ["b", "A", "c"]}))
        t.sortItems(0, Qt.AscendingOrder)
        assert [t.item(i, 0).text() for i in range(3)] == ["A", "b", "c"]

    def test_index_shown_when_not_default(self):
        corr = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"])
        corr.index.name = "band"
        t = DataFrameTable()
        t.set_dataframe(corr)
        assert t.columnCount() == 3
        assert t.horizontalHeaderItem(0).text() == "band"
        assert t.item(1, 0).text() == "b"

    def test_copy_selection(self):
        t = DataFrameTable()
        t.set_dataframe(self._df())
        t.setRangeSelected(__import__("PyQt5.QtWidgets", fromlist=["QTableWidgetSelectionRange"])
                           .QTableWidgetSelectionRange(0, 0, 1, 1), True)
        text = t.copy_selection()
        assert text == "RF\t0.912\nSVM\t0.800"
        assert QApplication.clipboard().text() == text
        withh = t.copy_selection(with_header=True)
        assert withh.splitlines()[0] == "Model\tAUC"
        allt = t.copy_all()
        assert allt.splitlines()[0] == "Model\tAUC\tn\tok" and len(allt.splitlines()) == 4

    def test_save_csv(self, tmp_path):
        df = pd.DataFrame({"Nomi": ["O'rta", "Yuqori ±"], "v": [1.23456789, 2.0]})
        t = DataFrameTable()
        assert t.save_csv(str(tmp_path / "none.csv")) is None            # ma'lumot yo'q
        t.set_dataframe(df)
        p = t.save_csv(str(tmp_path / "t.csv"))
        back = pd.read_csv(p, encoding="utf-8-sig")
        assert list(back["Nomi"]) == ["O'rta", "Yuqori ±"]
        assert back["v"][0] == pytest.approx(1.23456789)                   # to'liq aniqlik (ko'rsatish emas)
        assert t.dataframe() is df

    def test_truncation(self):
        t = DataFrameTable(max_rows=5)
        t.set_dataframe(pd.DataFrame({"a": range(12)}))
        assert t.rowCount() == 5 and t.truncated


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class TestProgressPanel:
    def test_int_percent_and_float_fraction(self):
        pnl = ProgressPanel()
        pnl.update_progress(40, "Fold 2/5")
        assert pnl.value() == 40 and pnl.stage_text() == "Fold 2/5"
        pnl.update_progress(0.5, "yarmi")
        assert pnl.value() == 50
        pnl.update_progress(75.0, "")
        assert pnl.value() == 75 and pnl.stage_text() == "yarmi"           # bo'sh msg matnni o'zgartirmaydi
        pnl.update_progress(1, "x")                                          # int 1 => 1%
        assert pnl.value() == 1
        pnl.reset()
        pnl.update_progress(1.0, "tugadi")                                   # float 1.0 => 100%
        assert pnl.value() == 100 and not pnl.is_active()
        pnl.update_progress(-5)
        assert pnl.value() == 0
        pnl.update_progress(float("nan"))
        assert pnl.value() == 0

    def test_reset(self):
        pnl = ProgressPanel()
        pnl.update_progress(30, "ish")
        pnl.reset()
        assert pnl.value() == 0 and pnl.stage_text() == "Tayyor" and pnl.time_label.text() == ""
        assert not pnl.is_active()

    def test_eta_with_fake_clock(self):
        clk = FakeClock()
        pnl = ProgressPanel(clock=clk)
        pnl.update_progress(0, "boshlandi")
        assert pnl.eta_seconds() is None
        clk.t += 20
        pnl.update_progress(0.25, "")
        assert pnl.elapsed() == pytest.approx(20.0)
        assert pnl.eta_seconds() == pytest.approx(60.0)                      # 20 s => 25%; qolgan 75% => 60 s
        assert "ETA: 01:00" in pnl.time_label.text() and "O'tgan: 00:20" in pnl.time_label.text()
        clk.t += 10
        pnl.update_progress(1.0, "")
        assert pnl.eta_seconds() is None and "ETA" not in pnl.time_label.text()

    def test_restart_when_progress_goes_back(self):
        clk = FakeClock()
        pnl = ProgressPanel(clock=clk)
        pnl.update_progress(80, "a")
        clk.t += 50
        pnl.update_progress(10, "b")                                         # yangi ish: taymer qayta boshlanadi
        assert pnl.elapsed() == pytest.approx(0.0)

    def test_stopped_and_format(self):
        pnl = ProgressPanel()
        pnl.update_progress(20, "ish")
        pnl.stopped("To'xtatildi")
        assert pnl.stage_text() == "To'xtatildi" and pnl.value() == 20
        assert format_duration(75) == "01:15" and format_duration(3725) == "1:02:05"
        assert format_duration(None) == "--:--" and format_duration(float("inf")) == "--:--"


class TestLogView:
    def test_append_and_clear(self):
        lv = LogView()
        assert lv.isReadOnly()
        lv.append_line("birinchi")
        lv.append_line("ikkinchi\n")
        lv.append_line("uch\nto'rt")
        assert lv.lines() == ["birinchi", "ikkinchi", "uch", "to'rt"]
        assert lv.total_lines == 4
        lv.clear_log()
        assert lv.lines() == [] and lv.total_lines == 0

    def test_max_lines_limit(self):
        lv = LogView(max_lines=50)
        for i in range(200):
            lv.append_line(f"qator {i}")
        ls = lv.lines()
        assert len(ls) == 50
        assert ls[-1] == "qator 199" and ls[0] == "qator 150"
        assert lv.n_dropped() == 150
        lv.set_max_lines(10)
        lv.append_line("yana")
        assert len(lv.lines()) == 10

    def test_save_utf8(self, tmp_path):
        lv = LogView()
        lv.append_line("AUC = 0.91 ± 0.03")
        lv.append_line("o'rta, g'arb")
        p = lv.save_to_file(str(tmp_path / "log.txt"))
        with open(p, encoding="utf-8") as f:
            txt = f.read()
        assert "±" in txt and "o'rta, g'arb" in txt and txt.endswith("\n")

    def test_none_text(self):
        lv = LogView()
        lv.append_line(None)                            # xato bermaydi
        lv.append_line("x")
        assert lv.lines()[-1] == "x"


# ===========================================================================
# param_panel
# ===========================================================================
@pytest.fixture()
def panel():
    p = HyperParamPanel()
    yield p
    p.deleteLater()


def _expected_widget(spec):
    return {"int": QSpinBox, "float": QDoubleSpinBox, "bool": QCheckBox, "choice": QComboBox,
            "optint": QSpinBox, "optfloat": QDoubleSpinBox}[spec.kind]


class TestPanelCompleteness:
    def test_widget_for_every_spec(self, panel):
        n = 0
        for model, specs in PARAM_SPECS.items():
            for spec in specs:
                w = panel.widget_for(model, spec.name)
                assert isinstance(w, _expected_widget(spec)), (model, spec.name)
                n += 1
        assert n == len(panel.fields) == sum(len(s) for s in PARAM_SPECS.values())

    def test_optional_widgets_have_none_checkbox(self, panel):
        for model, specs in PARAM_SPECS.items():
            for spec in specs:
                f = panel.field(model, spec.name)
                if spec.kind in ("optint", "optfloat"):
                    assert f.none_check is not None and f.none_check.text() == spec.none_label
                    assert f.none_check.isChecked() == (spec.default is None)
                else:
                    assert f.none_check is None

    def test_tooltips_and_ranges(self, panel):
        for model, specs in PARAM_SPECS.items():
            for spec in specs:
                f = panel.field(model, spec.name)
                tip = f.editor.toolTip()
                assert "Standart:" in tip, (model, spec.name)
                if spec.tooltip:
                    assert spec.tooltip.split("\n")[0] in tip
                if spec.kind in ("int", "optint"):
                    assert f.editor.minimum() == int(spec.min) and f.editor.maximum() == int(spec.max)
                    assert "Diapazon" in tip
                if spec.kind == "choice":
                    assert [f.editor.itemText(i) for i in range(f.editor.count())] == list(spec.choices)

    def test_tabs_per_model_and_groups(self, panel):
        assert panel.models() == list(PARAM_SPECS)
        assert [panel.tabs.tabText(i) for i in range(panel.tabs.count())] == ["Random Forest", "SVM", "XGBoost", "CNN"]
        from PyQt5.QtWidgets import QGroupBox, QScrollArea
        page = panel.tabs.widget(0)
        assert isinstance(page, QScrollArea)
        titles = [g.title() for g in page.widget().findChildren(QGroupBox)]
        assert titles == list(dict.fromkeys(s.group for s in PARAM_SPECS["RandomForest"]))

    def test_new_spec_appears_automatically(self, monkeypatch):
        new = ParamSpec("shrinking_x", "Yangi parametr", "float", 0.5, min=0.0, max=1.0, step=0.1, decimals=2,
                        group="Yadro")
        monkeypatch.setitem(PARAM_SPECS, "SVM", PARAM_SPECS["SVM"] + [new])
        p = HyperParamPanel()
        w = p.widget_for("SVM", "shrinking_x")
        assert isinstance(w, QDoubleSpinBox)
        assert p.get_hyperparams()["SVM"]["shrinking_x"] == 0.5
        p.set_hyperparams({"SVM": {"shrinking_x": 0.8}})
        assert p.get_hyperparams()["SVM"]["shrinking_x"] == 0.8


class TestPanelRoundtrip:
    def test_default_roundtrip(self, panel):
        assert panel.get_hyperparams() == default_hyperparams()

    def test_changed_values_roundtrip(self, panel):
        hp = default_hyperparams()
        hp["RandomForest"].update(n_estimators=250, criterion="entropy", max_depth=12, min_samples_leaf=4,
                                  max_features="all", bootstrap=True, max_samples=0.6, class_weight="none",
                                  ccp_alpha=0.01)
        hp["SVM"].update(kernel="poly", C=12.5, gamma=0.02, degree=4, class_weight="none")
        hp["XGBoost"].update(n_estimators=120, learning_rate=0.1, max_depth=6, subsample=0.65,
                             scale_pos_weight=3.5, tree_method="exact", reg_alpha=0.3)
        hp["CNN"].update(mode="tabular1d", window=11, filters1=8, filters2=0, kernel_size=5, dropout=0.45,
                         learning_rate=0.0007, batch_size=16, epochs=30, patience=5, augment=False, optimizer="sgd")
        panel.set_hyperparams(hp)
        assert panel.get_hyperparams() == config.validate_hyperparams(hp)[0]
        got = panel.get_hyperparams()
        assert got["RandomForest"]["max_depth"] == 12 and got["RandomForest"]["max_samples"] == 0.6
        assert got["SVM"]["gamma"] == 0.02 and got["XGBoost"]["scale_pos_weight"] == 3.5
        assert got["CNN"]["window"] == 11 and got["CNN"]["augment"] is False

    def test_none_values(self, panel):
        hp = default_hyperparams()
        hp["RandomForest"]["max_depth"] = 9
        hp["SVM"]["gamma"] = 0.5
        panel.set_hyperparams(hp)
        assert panel.get_hyperparams()["RandomForest"]["max_depth"] == 9
        assert not panel.field("RandomForest", "max_depth").none_check.isChecked()
        assert panel.widget_for("RandomForest", "max_depth").isEnabled()
        panel.set_hyperparams({"RandomForest": {"max_depth": None}, "SVM": {"gamma": None}})
        got = panel.get_hyperparams()
        assert got["RandomForest"]["max_depth"] is None and got["SVM"]["gamma"] is None
        assert panel.field("RandomForest", "max_depth").none_check.isChecked()
        assert not panel.widget_for("RandomForest", "max_depth").isEnabled()      # None => spin o'chiq

    def test_none_checkbox_user_toggle(self, panel):
        f = panel.field("XGBoost", "scale_pos_weight")
        assert panel.get_hyperparams()["XGBoost"]["scale_pos_weight"] is None
        f.none_check.setChecked(False)
        v = panel.get_hyperparams()["XGBoost"]["scale_pos_weight"]
        assert v is not None and 0.01 <= v <= 1000.0
        f.editor.setValue(4.0)
        assert panel.get_hyperparams()["XGBoost"]["scale_pos_weight"] == 4.0
        f.none_check.setChecked(True)
        assert panel.get_hyperparams()["XGBoost"]["scale_pos_weight"] is None

    def test_float_precision_preserved(self, panel):
        panel.set_hyperparams({"XGBoost": {"learning_rate": 0.0123456}, "CNN": {"learning_rate": 0.000123}})
        got = panel.get_hyperparams()
        assert got["XGBoost"]["learning_rate"] == pytest.approx(0.0123456, abs=1e-9)
        assert got["CNN"]["learning_rate"] == pytest.approx(0.000123, abs=1e-9)
        panel.reset_defaults("XGBoost")
        assert panel.field("XGBoost", "learning_rate").editor.decimals() == 4     # standart aniqlikka qaytadi

    def test_partial_set_keeps_other_values(self, panel):
        panel.set_hyperparams({"SVM": {"C": 3.0}})
        panel.set_hyperparams({"XGBoost": {"max_depth": 7}})
        got = panel.get_hyperparams()
        assert got["SVM"]["C"] == 3.0 and got["XGBoost"]["max_depth"] == 7
        assert got["RandomForest"] == default_hyperparams()["RandomForest"]

    def test_invalid_values_clamped_with_warnings(self, panel):
        warns = panel.set_hyperparams({"RandomForest": {"n_estimators": 1, "criterion": "xyz"}, "Nomalum": {"a": 1}})
        got = panel.get_hyperparams()["RandomForest"]
        assert got["n_estimators"] == 10 and got["criterion"] == "gini"
        assert any("Nomalum" in w for w in warns) and any("n_estimators" in w for w in warns)

    def test_get_hyperparams_is_validated_and_complete(self, panel):
        hp = panel.get_hyperparams()
        assert set(hp) == set(PARAM_SPECS)
        for m, specs in PARAM_SPECS.items():
            assert set(hp[m]) == {s.name for s in specs}
        assert config.validate_hyperparams(hp) == (hp, [])


class TestPanelReset:
    def _dirty(self, panel):
        panel.set_hyperparams({"RandomForest": {"n_estimators": 77, "max_depth": 5},
                               "SVM": {"C": 9.0}, "CNN": {"epochs": 12, "mode": "tabular1d"}})

    def test_reset_single_model(self, panel):
        self._dirty(panel)
        panel.reset_defaults("RandomForest")
        got = panel.get_hyperparams()
        d = default_hyperparams()
        assert got["RandomForest"] == d["RandomForest"]
        assert got["SVM"]["C"] == 9.0 and got["CNN"]["epochs"] == 12

    def test_reset_all(self, panel):
        self._dirty(panel)
        panel.reset_defaults()
        assert panel.get_hyperparams() == default_hyperparams()

    def test_reset_unknown_model(self, panel):
        with pytest.raises(KeyError):
            panel.reset_defaults("Nomalum")

    def test_reset_does_not_touch_tuning(self, panel):
        t = TuningConfig(enabled=True, n_iter=33)
        panel.set_tuning(t)
        panel.reset_defaults()
        assert panel.get_tuning().enabled and panel.get_tuning().n_iter == 33

    def test_buttons_reset(self, panel):
        self._dirty(panel)
        panel.set_current_model("SVM")
        panel.btn_reset_model.click()
        got = panel.get_hyperparams()
        assert got["SVM"]["C"] == 1.0 and got["RandomForest"]["n_estimators"] == 77
        panel.btn_reset_all.click()
        assert panel.get_hyperparams() == default_hyperparams()


class TestPanelSignals:
    def test_user_edit_emits_once(self, panel):
        n = []
        panel.changed.connect(lambda: n.append(1))
        panel.widget_for("SVM", "C").setValue(3.5)
        assert len(n) == 1
        panel.widget_for("RandomForest", "criterion").setCurrentIndex(1)
        assert len(n) == 2
        panel.widget_for("RandomForest", "bootstrap").setChecked(False)
        assert len(n) == 3

    def test_programmatic_set_emits_once(self, panel):
        n = []
        panel.changed.connect(lambda: n.append(1))
        panel.set_hyperparams({"RandomForest": {"n_estimators": 100, "max_depth": 4, "criterion": "entropy"},
                               "CNN": {"mode": "tabular1d"}})
        assert len(n) == 1
        panel.reset_defaults()
        assert len(n) == 2
        panel.set_tuning(TuningConfig(enabled=True))
        assert len(n) == 3

    def test_tuning_edit_emits(self, panel):
        n = []
        panel.changed.connect(lambda: n.append(1))
        panel.tuning_group.n_iter.setValue(7)
        assert len(n) == 1 and panel.get_tuning().n_iter == 7


class TestDependencies:
    def test_cnn_mode_toggles_window_and_augment(self, panel):
        win = panel.field("CNN", "window")
        aug = panel.field("CNN", "augment")
        assert win.widget.isEnabled() and aug.widget.isEnabled()
        panel.widget_for("CNN", "mode").setCurrentText("tabular1d")
        assert not win.widget.isEnabled() and not aug.widget.isEnabled()
        assert not win.label.isEnabled()
        assert panel.get_hyperparams()["CNN"]["window"] == 9                  # qiymat saqlanadi
        panel.widget_for("CNN", "mode").setCurrentText("patch2d")
        assert win.widget.isEnabled() and aug.widget.isEnabled()

    def test_set_hyperparams_applies_dependencies(self, panel):
        panel.set_hyperparams({"CNN": {"mode": "tabular1d"}})
        assert not panel.field("CNN", "window").widget.isEnabled()
        panel.reset_defaults("CNN")
        assert panel.field("CNN", "window").widget.isEnabled()

    def test_rf_max_samples_needs_bootstrap(self, panel):
        ms = panel.field("RandomForest", "max_samples")
        assert ms.widget.isEnabled()
        panel.widget_for("RandomForest", "bootstrap").setChecked(False)
        assert not ms.widget.isEnabled()
        panel.widget_for("RandomForest", "bootstrap").setChecked(True)
        assert ms.widget.isEnabled()

    def test_disabled_optional_keeps_none_state(self, panel):
        ms = panel.field("RandomForest", "max_samples")
        panel.widget_for("RandomForest", "bootstrap").setChecked(False)
        panel.widget_for("RandomForest", "bootstrap").setChecked(True)
        assert ms.none_check.isChecked() and not ms.editor.isEnabled()        # None => spin hamon o'chiq

    def test_window_always_odd(self, panel):
        sb = panel.widget_for("CNN", "window")
        assert sb.singleStep() == 2
        sb.setValue(10)
        assert sb.value() % 2 == 1 and sb.value() == 11
        sb.setValue(config.get_spec("CNN", "window").max)
        assert sb.value() % 2 == 1
        sb.setValue(4)
        assert sb.value() == 5
        k = panel.widget_for("CNN", "kernel_size")
        k.setValue(4)
        assert k.value() % 2 == 1
        assert panel.get_hyperparams()["CNN"]["window"] % 2 == 1

    def test_even_window_via_set_hyperparams(self, panel):
        warns = panel.set_hyperparams({"CNN": {"window": 8}})
        assert panel.get_hyperparams()["CNN"]["window"] == 9
        assert any("toq" in w for w in warns)


class TestPreset:
    def test_save_load_file(self, panel, tmp_path):
        panel.set_hyperparams({"RandomForest": {"n_estimators": 321, "max_depth": 8},
                               "XGBoost": {"scale_pos_weight": 2.5}, "CNN": {"window": 13, "mode": "tabular1d"}})
        panel.set_tuning(TuningConfig(enabled=True, n_iter=9, inner_splits=4, scoring="average_precision",
                                      models={"RandomForest": False, "SVM": True, "XGBoost": True, "CNN": True},
                                      spaces={"SVM": {"C": {"min": 0.1, "max": 10.0, "log": True}}}))
        before_hp, before_t = panel.get_hyperparams(), panel.get_tuning()
        path = str(tmp_path / "preset.json")
        assert panel.save_preset_to(path) == path
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["kind"] == "hyperparams" and payload["hyperparams"]["RandomForest"]["n_estimators"] == 321
        assert payload["tuning"]["n_iter"] == 9

        other = HyperParamPanel()
        warns = other.load_preset_from(path)
        assert warns == []
        assert other.get_hyperparams() == before_hp
        t = other.get_tuning()
        assert t.enabled and t.n_iter == 9 and t.inner_splits == 4 and t.scoring == "average_precision"
        assert t.models == before_t.models and t.spaces == before_t.spaces
        assert not other.field("CNN", "window").widget.isEnabled()
        other.deleteLater()

    def test_load_replaces_everything(self, panel, tmp_path):
        path = str(tmp_path / "p.json")
        config.save_preset(path, hyperparams={"SVM": {"C": 5.0}})                   # faqat SVM.C (qolgani standart)
        panel.set_hyperparams({"RandomForest": {"n_estimators": 99}, "SVM": {"C": 1.5}})
        panel.load_preset_from(path)
        got = panel.get_hyperparams()
        assert got["SVM"]["C"] == 5.0 and got["RandomForest"]["n_estimators"] == 500    # preset to'liq almashtiradi

    def test_load_warnings_returned(self, panel, tmp_path):
        path = str(tmp_path / "w.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": "hyperparams", "hyperparams": {"RandomForest": {"n_estimators": 1},
                                                              "Nomalum": {"a": 1}}}, f)
        warns = panel.load_preset_from(path)
        assert any("Nomalum" in w for w in warns) and any("n_estimators" in w for w in warns)
        assert panel.get_hyperparams()["RandomForest"]["n_estimators"] == 10

    def test_load_run_config_file(self, panel, tmp_path):
        cfg = config.RunConfig(tuning=TuningConfig(enabled=True, n_iter=5))
        cfg.hyperparams["SVM"]["C"] = 7.0
        path = str(tmp_path / "cfg.json")
        config.save_preset(path, cfg=cfg)
        panel.load_preset_from(path)
        assert panel.get_hyperparams()["SVM"]["C"] == 7.0
        assert panel.get_tuning().enabled and panel.get_tuning().n_iter == 5

    def test_load_bad_file_raises(self, panel, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{buzuq", encoding="utf-8")
        with pytest.raises(Exception):
            panel.load_preset_from(str(p))

    def test_dialogs(self, panel, tmp_path, monkeypatch):
        from PyQt5.QtWidgets import QFileDialog
        msgs = []
        monkeypatch.setattr(panel, "_show_message", lambda kind, title, text: msgs.append((kind, text)))
        path = str(tmp_path / "d.json")
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (path, "")))
        panel.set_hyperparams({"SVM": {"C": 4.0}})
        assert panel.save_preset_dialog() == path
        assert "saqlandi" in panel.status_label.text()
        panel.reset_defaults()
        monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (path, "")))
        assert panel.load_preset_dialog() == path
        assert panel.get_hyperparams()["SVM"]["C"] == 4.0 and "yuklandi" in panel.status_label.text()
        assert msgs == []
        # bekor qilish
        monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))
        assert panel.load_preset_dialog() is None
        # xato fayl => critical xabar, panel o'zgarmaydi
        bad = tmp_path / "bad.json"
        bad.write_text("nope", encoding="utf-8")
        monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(bad), "")))
        assert panel.load_preset_dialog() is None
        assert msgs and msgs[-1][0] == "critical"
        assert panel.get_hyperparams()["SVM"]["C"] == 4.0
        # ogohlantirishli preset => warning xabar
        warn_path = tmp_path / "warn.json"
        warn_path.write_text(json.dumps({"hyperparams": {"Nomalum": {}}}), encoding="utf-8")
        monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(warn_path), "")))
        panel.load_preset_dialog()
        assert msgs[-1][0] == "warning" and "Nomalum" in msgs[-1][1]


class TestCostHint:
    def test_set_cost_hint(self, panel):
        assert panel.cost_hint() == "" and not panel.cost_label.isVisibleTo(panel)
        panel.set_cost_hint("Taxminan 120 ta fit")
        assert panel.cost_hint() == "Taxminan 120 ta fit"
        assert panel.cost_label.isVisibleTo(panel)
        panel.set_cost_hint("")
        assert not panel.cost_label.isVisibleTo(panel)
        panel.set_cost_hint(None)
        assert panel.cost_hint() == ""


class TestTuningGroup:
    def test_defaults_match_tuningconfig(self):
        g = TuningGroup()
        t = g.get_tuning()
        d = TuningConfig()
        assert t.enabled is False and t.n_iter == d.n_iter and t.inner_splits == d.inner_splits
        assert t.scoring == d.scoring and t.models == d.models and t.spaces == {} and t.mode == d.mode

    def test_get_set_roundtrip(self):
        g = TuningGroup()
        t = TuningConfig(enabled=True, mode="nested", n_iter=12, inner_splits=5, scoring="average_precision",
                         models={"RandomForest": False, "SVM": False, "XGBoost": True, "CNN": True},
                         spaces={"XGBoost": {"max_depth": {"min": 3, "max": 6, "log": False}}})
        g.set_tuning(t)
        assert g.get_tuning() == t
        assert g.isChecked() and g.model_checks["CNN"].isChecked() and not g.model_checks["RandomForest"].isChecked()
        assert "1 ta" in g.spaces_label.text()
        g.set_tuning(TuningConfig())
        assert g.get_tuning() == TuningConfig()
        assert g.spaces_label.text() == "Oraliqlar: standart"

    def test_set_from_dict_and_none(self):
        g = TuningGroup()
        g.set_tuning({"enabled": True, "n_iter": 3, "scoring": "zzz", "inner_splits": 1})
        t = g.get_tuning()
        assert t.enabled and t.n_iter == 3 and t.scoring == "roc_auc" and t.inner_splits == 2
        g.set_tuning(None)
        assert g.get_tuning() == TuningConfig()

    def test_user_edits_reflected(self):
        g = TuningGroup()
        g.setChecked(True)
        g.n_iter.setValue(40)
        g.inner_splits.setValue(4)
        g.scoring.setCurrentIndex(g.scoring.findData("average_precision"))
        g.model_checks["CNN"].setChecked(True)
        g.model_checks["SVM"].setChecked(False)
        t = g.get_tuning()
        assert (t.enabled, t.n_iter, t.inner_splits, t.scoring) == (True, 40, 4, "average_precision")
        assert t.models["CNN"] is True and t.models["SVM"] is False

    def test_cnn_checkbox_warns_expensive(self):
        g = TuningGroup()
        cb = g.model_checks["CNN"]
        assert "qimmat" in cb.text().lower() and "qimmat" in cb.toolTip().lower()
        assert not cb.isChecked()                                  # standart: CNN tuning o'chiq

    def test_only_tunable_models_listed(self):
        g = TuningGroup()
        assert set(g.model_checks) == {m for m, s in PARAM_SPECS.items() if any(x.tunable for x in s)}

    def test_get_tuning_returns_independent_copy(self):
        g = TuningGroup()
        g.set_tuning(TuningConfig(spaces={"SVM": {"C": {"min": 0.1, "max": 1.0, "log": True}}}))
        t = g.get_tuning()
        t.spaces["SVM"]["C"]["min"] = 99
        assert g.get_tuning().spaces["SVM"]["C"]["min"] == 0.1

    def test_unknown_model_keys_preserved(self):
        g = TuningGroup()
        g.set_tuning(TuningConfig(models={"RandomForest": True, "Boshqa": True}))
        assert g.get_tuning().models["Boshqa"] is True

    def test_edit_spaces_button(self, monkeypatch):
        g = TuningGroup()
        n = []
        g.changed.connect(lambda: n.append(1))

        def fake_exec(self):
            self.set_row("SVM", "C", min=0.5, max=5)
            self.accept()
            return self.result()
        monkeypatch.setattr(SearchSpaceDialog, "exec_", fake_exec)
        assert g.edit_spaces() is True
        assert g.get_tuning().spaces == {"SVM": {"C": {"min": 0.5, "max": 5.0, "log": True}}}
        assert len(n) == 1 and "1 ta" in g.spaces_label.text()
        monkeypatch.setattr(SearchSpaceDialog, "exec_", lambda self: QDialog.Rejected)
        assert g.edit_spaces() is False
        assert len(n) == 1

    def test_panel_delegates(self, panel):
        panel.set_tuning(TuningConfig(enabled=True, n_iter=6))
        assert panel.tuning_group.isChecked() and panel.get_tuning().n_iter == 6


class TestSearchSpaceDialog:
    def test_rows_for_all_tunable_params(self):
        dlg = SearchSpaceDialog(TuningConfig())
        expected = [(m, s.name) for m, specs in PARAM_SPECS.items() for s in specs if s.tunable]
        assert [(r["model"], r["name"]) for r in dlg._rows] == expected
        assert dlg.table.rowCount() == len(expected)
        # boshlang'ich qiymatlar default_search_space bilan mos
        space = default_search_space("SVM")
        assert float(dlg.cell("SVM", "C", pp.COL_MIN).text()) == space["C"]["min"]
        assert float(dlg.cell("SVM", "C", pp.COL_MAX).text()) == space["C"]["max"]
        assert dlg.cell("SVM", "C", pp.COL_LOG).checkState() == Qt.Checked
        assert dlg.cell("SVM", "gamma", pp.COL_LOG).checkState() == Qt.Checked
        assert dlg.cell("RandomForest", "n_estimators", pp.COL_LOG).checkState() == Qt.Unchecked
        assert dlg.cell("RandomForest", "max_features", pp.COL_CHOICES).text() == "sqrt, log2, all"
        assert dlg.cell("CNN", "window", pp.COL_CHOICES).text() == "5, 7, 9, 11, 15"

    def test_numeric_and_choice_cells_editable_state(self):
        dlg = SearchSpaceDialog(TuningConfig())
        num_min = dlg.cell("XGBoost", "max_depth", pp.COL_MIN)
        assert num_min.flags() & Qt.ItemIsEditable
        assert not (dlg.cell("XGBoost", "max_depth", pp.COL_CHOICES).flags() & Qt.ItemIsEnabled)
        ch = dlg.cell("RandomForest", "max_features", pp.COL_CHOICES)
        assert ch.flags() & Qt.ItemIsEditable
        assert not (dlg.cell("RandomForest", "max_features", pp.COL_MIN).flags() & Qt.ItemIsEnabled)
        assert not (dlg.cell("RandomForest", "max_features", pp.COL_PARAM).flags() & Qt.ItemIsEditable)

    def test_unchanged_ok_keeps_spaces_empty(self):
        t = TuningConfig(spaces={"SVM": {"C": {"min": 0.5, "max": 5.0}}})
        dlg = SearchSpaceDialog(t)
        assert float(dlg.cell("SVM", "C", pp.COL_MIN).text()) == 0.5               # mavjud o'zgartirish ko'rinadi
        dlg.reset_to_defaults()
        dlg.accept()
        assert dlg.result() == QDialog.Accepted and t.spaces == {}

    def test_edit_and_accept_updates_spaces(self):
        t = TuningConfig()
        dlg = SearchSpaceDialog(t)
        dlg.set_row("RandomForest", "n_estimators", min=200, max=400)
        dlg.set_row("SVM", "C", min=0.1, max=50, log=True)
        dlg.set_row("XGBoost", "learning_rate", min="0,02", max="0.2")             # vergulli kasr
        dlg.set_row("RandomForest", "max_features", choices="sqrt, all")
        dlg.set_row("CNN", "window", choices=[5, 9])
        dlg.set_row("XGBoost", "subsample", log=False)                             # o'zgarmagan => saqlanmaydi
        dlg.accept()
        assert dlg.result() == QDialog.Accepted
        sp = t.spaces
        assert sp["RandomForest"]["n_estimators"] == {"min": 200, "max": 400, "log": False}
        assert isinstance(sp["RandomForest"]["n_estimators"]["min"], int)
        assert sp["SVM"]["C"] == {"min": 0.1, "max": 50.0, "log": True}
        assert sp["XGBoost"]["learning_rate"]["min"] == 0.02
        assert sp["RandomForest"]["max_features"] == {"choices": ["sqrt", "all"]}
        assert sp["CNN"]["window"] == {"choices": [5, 9]} and all(isinstance(c, int) for c in sp["CNN"]["window"]["choices"])
        assert "subsample" not in sp.get("XGBoost", {})
        # resolved_space shu qiymatlarni qaytaradi
        rs = t.resolved_space("RandomForest")
        assert rs["n_estimators"]["min"] == 200 and rs["max_features"]["choices"] == ["sqrt", "all"]

    def test_log_toggle_stored(self):
        t = TuningConfig()
        dlg = SearchSpaceDialog(t)
        dlg.set_row("XGBoost", "max_depth", log=True)
        dlg.accept()
        assert t.spaces["XGBoost"]["max_depth"]["log"] is True

    @pytest.mark.parametrize("kw,needle", [
        (dict(min=10, max=5), "min < max"),
        (dict(min=5, max=5), "min < max"),
        (dict(min="abc", max=5), "son bo'lishi kerak"),
        (dict(min=2.5, max=8), "butun son"),
        (dict(min=0, max=8, log=True), "min > 0"),
        (dict(min=-3, max=8), "ruxsat etilgan"),
        (dict(min=2, max=500), "ruxsat etilgan"),
    ])
    def test_invalid_numeric_ranges(self, kw, needle):
        t = TuningConfig()
        dlg = SearchSpaceDialog(t)
        dlg.set_row("RandomForest", "max_depth", **kw)
        dlg.accept()
        assert dlg.result() != QDialog.Accepted                                   # dialog yopilmadi
        assert t.spaces == {}                                                      # o'zgarmadi
        assert not dlg.error_label.isHidden() and needle in dlg.error_label.text()
        assert "max_depth" in dlg.error_label.text()
        bad = dlg.cell("RandomForest", "max_depth", pp.COL_MIN).background().color()
        assert bad.red() == 255 and bad.green() < 255 or dlg.cell("RandomForest", "max_depth",
                                                                    pp.COL_MAX).background().color().green() < 255

    def test_error_then_fix_then_accept(self):
        t = TuningConfig()
        dlg = SearchSpaceDialog(t)
        dlg.set_row("SVM", "C", min=10, max=1)
        dlg.accept()
        assert dlg.result() != QDialog.Accepted and not dlg.error_label.isHidden()
        dlg.set_row("SVM", "C", min=1, max=10)
        dlg.accept()
        assert dlg.result() == QDialog.Accepted and dlg.error_label.isHidden()
        assert t.spaces["SVM"]["C"] == {"min": 1.0, "max": 10.0, "log": True}

    @pytest.mark.parametrize("model,name,text,needle", [
        ("RandomForest", "max_features", "", "kamida bitta"),
        ("RandomForest", "max_features", "sqrt, banana", "ruxsat etilmagan"),
        ("CNN", "window", "5, x", "son emas"),
        ("CNN", "window", "5, 6", "toq"),
        ("CNN", "window", "5, 7.5", "butun"),
        ("CNN", "window", "1, 9", "ruxsat etilgan"),
        ("CNN", "batch_size", "4, 9999", "ruxsat etilgan"),
    ])
    def test_invalid_choices(self, model, name, text, needle):
        t = TuningConfig()
        dlg = SearchSpaceDialog(t)
        dlg.set_row(model, name, choices=text)
        dlg.accept()
        assert dlg.result() != QDialog.Accepted and t.spaces == {}
        assert needle in dlg.error_label.text()

    def test_choices_dedupe_and_semicolon(self):
        t = TuningConfig()
        dlg = SearchSpaceDialog(t)
        dlg.set_row("CNN", "filters1", choices="32; 16, 32 ,8, 64,")
        dlg.accept()
        assert dlg.result() == QDialog.Accepted
        assert t.spaces["CNN"]["filters1"] == {"choices": [32, 16, 8, 64]}

    def test_multiple_errors_reported_together(self):
        t = TuningConfig()
        dlg = SearchSpaceDialog(t)
        dlg.set_row("SVM", "C", min=5, max=1)
        dlg.set_row("CNN", "window", choices="4")
        dlg.accept()
        txt = dlg.error_label.text()
        assert "SVM.C" in txt and "CNN.window" in txt

    def test_cancel_does_not_modify(self):
        t = TuningConfig()
        dlg = SearchSpaceDialog(t)
        dlg.set_row("SVM", "C", min=0.2, max=2)
        dlg.reject()
        assert t.spaces == {} and dlg.result() == QDialog.Rejected

    def test_reset_to_defaults_button_state(self):
        t = TuningConfig()
        dlg = SearchSpaceDialog(t)
        dlg.set_row("SVM", "C", min=0.2, max=2)
        dlg.set_row("CNN", "window", choices="5")
        dlg.reset_to_defaults()
        assert float(dlg.cell("SVM", "C", pp.COL_MIN).text()) == default_search_space("SVM")["C"]["min"]
        assert dlg.cell("CNN", "window", pp.COL_CHOICES).text() == "5, 7, 9, 11, 15"

    def test_resolved_space_used_by_sampling(self):
        """Dialog'da yozilgan oraliq tuning.sample_params orqali haqiqatan ishlatiladi."""
        from mpm.tuning import sample_params
        t = TuningConfig()
        dlg = SearchSpaceDialog(t)
        dlg.set_row("RandomForest", "n_estimators", min=111, max=113)
        dlg.set_row("RandomForest", "max_features", choices="log2")
        dlg.accept()
        rng = np.random.default_rng(0)
        base = default_hyperparams()["RandomForest"]
        for _ in range(10):
            p = sample_params("RandomForest", base, t.resolved_space("RandomForest"), rng)
            assert 111 <= p["n_estimators"] <= 113 and p["max_features"] == "log2"


class TestIntegration:
    def test_panel_hp_feeds_runconfig(self, panel):
        panel.set_hyperparams({"RandomForest": {"n_estimators": 40}})
        panel.set_tuning(TuningConfig(enabled=True, n_iter=4))
        cfg = config.RunConfig(hyperparams=panel.get_hyperparams(), tuning=panel.get_tuning())
        assert cfg.hyperparams["RandomForest"]["n_estimators"] == 40 and cfg.tuning.n_iter == 4
        assert config.RunConfig.from_dict(cfg.to_dict()).hyperparams == cfg.hyperparams
        assert copy.deepcopy(cfg.hyperparams) == panel.get_hyperparams()


# ===========================================================================
# Mustaqil review regressiyalari
# ===========================================================================
class TestReviewRegressions:
    def test_sort_mixed_type_column_does_not_raise(self):
        """Aralash (son + matn + bo'sh) ustunni saralash TypeError berib dasturni abort qilmasligi kerak."""
        t = DataFrameTable()
        t.set_dataframe(pd.DataFrame({"a": [1.5, "abc", 2, None, "Zed", float("nan")], "b": range(6)}))
        for order in (Qt.AscendingOrder, Qt.DescendingOrder):
            t.sortItems(0, order)
        t.sortItems(0, Qt.AscendingOrder)
        texts = [t.item(i, 0).text() for i in range(t.rowCount())]
        assert texts == ["", "", "1.500", "2", "abc", "Zed"]       # bo'sh < sonlar < matnlar

    def test_pd_na_shown_empty(self):
        t = DataFrameTable()
        t.set_dataframe(pd.DataFrame({"x": pd.array([1, None], dtype="Int64"),
                                      "s": pd.array(["a", None], dtype="string")}))
        assert t.item(1, 0).text() == "" and t.item(1, 1).text() == ""

    def test_progress_restart_after_early_stop(self):
        clk = FakeClock()
        pr = ProgressPanel(clock=clk)
        clk.t = 100.0
        pr.update_progress(3, "a")
        pr.stopped("x")
        clk.t = 1000.0
        pr.update_progress(0, "yangi")                 # yangi ish: eski t0 ishlatilmasin
        clk.t = 1001.0
        assert pr.elapsed() == pytest.approx(1.0)

    def test_progress_elapsed_frozen_after_finish(self):
        clk = FakeClock()
        pr = ProgressPanel(clock=clk)
        clk.t = 0.0
        pr.update_progress(50, "x")
        clk.t = 10.0
        pr.update_progress(100, "x")
        clk.t = 500.0
        pr.update_progress(100, "takror")
        assert pr.elapsed() == pytest.approx(10.0)

    def test_nonfinite_values_do_not_crash_or_become_max(self, panel):
        before = panel.get_hyperparams()
        w = panel.set_hyperparams({"XGBoost": {"learning_rate": float("nan"), "gamma": float("inf")},
                                   "RandomForest": {"n_estimators": float("inf")},
                                   "SVM": {"gamma": float("nan")}})
        assert len(w) == 4
        assert panel.get_hyperparams() == before

    def test_set_value_nonfinite_falls_back_to_default(self, panel):
        f = panel.field("XGBoost", "learning_rate")
        f.set_value(float("nan"))
        assert f.value() == pytest.approx(f.spec.default)

    def test_dialog_survives_broken_spaces(self):
        t = TuningConfig(spaces={"RandomForest": {"max_depth": {"min": "a", "max": 5},
                                                  "max_features": {"choices": 5}},
                                 "SVM": 3, "Foo": 1})
        dlg = SearchSpaceDialog(t)
        spaces, errors = dlg.collect()
        assert errors == [] and spaces == {}               # hammasi standart oraliqqa qaytdi
