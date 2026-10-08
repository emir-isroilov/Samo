# -*- coding: utf-8 -*-
"""
Giperparametr paneli (PyQt5): `config.PARAM_SPECS` dan AVTOMATIK quriladi (ENH-01).

  HyperParamPanel   - har model uchun sub-tab (RF, SVM, XGBoost, CNN), har parametr uchun mos vidjet:
                      int -> QSpinBox, float -> QDoubleSpinBox, bool -> QCheckBox, choice -> QComboBox,
                      optint/optfloat -> "avtomatik/None" checkbox + spin; group bo'yicha QGroupBox, QScrollArea.
                      .get_hyperparams() / .set_hyperparams(hp) / .reset_defaults(model=None) / .get_tuning() /
                      .set_tuning(t) / .set_cost_hint(text), signal changed(); preset (JSON) saqlash/yuklash.
  TuningGroup       - giperparametr qidirish (ENH-02) sozlamalari: yoqish, modellar, n_iter, ichki fold, scoring,
                      qidiruv oraliqlarini tahrirlash. .get_tuning() / .set_tuning(t).
  SearchSpaceDialog - tunable parametrlar oraliqlarini jadvalda tahrirlash (OK bosilganda TuningConfig.spaces yangilanadi).

PARAM_SPECS ga parametr qo'shilsa, panel (qayta yaratilganda) uni o'zi ko'rsatadi: hech narsa qo'lda yozilmaydi.
"""
from __future__ import annotations

import copy
import math
import numbers
import os

from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

from .. import config
from ..config import TuningConfig

__all__ = ["HyperParamPanel", "TuningGroup", "SearchSpaceDialog", "ParamField", "MODEL_TITLES"]

MODEL_TITLES = {"RandomForest": "Random Forest", "SVM": "SVM", "XGBoost": "XGBoost", "CNN": "CNN"}
COST_MAX_HEIGHT = 72                                  # xarajat izohi paneli maks. balandligi (px); oshsa - o'z scroll'i

# Bir parametr boshqasining qiymatiga bog'liq bo'lib, ma'nosiz bo'lganda o'chiriladi (qiymat saqlanadi):
# (boshqaruvchi model, parametr, bog'liq parametrlar, o'chirish sharti(qiymat) -> bool)
_DEPENDENCIES = (
    ("CNN", "mode", ("window", "augment"), lambda v: v == "tabular1d"),
    ("RandomForest", "bootstrap", ("max_samples",), lambda v: not v),
)

_BAD_BG = QColor(255, 190, 190)
_INT_LIMIT = 2_000_000_000


def _is_int_kind(spec):
    return spec.kind in ("int", "optint")


def _is_odd_spec(spec):
    """Faqat toq qiymatli int parametr (CNN window, kernel_size): step=2 va min toq."""
    return (_is_int_kind(spec) and spec.step == 2 and spec.min is not None and int(spec.min) % 2 == 1)


def _fmt_num(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    return format(float(v), ".10g")


def _fmt_value(v, none_label="None"):
    if v is None:
        return none_label
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return _fmt_num(v)
    return str(v)


def _spec_tooltip(spec):
    """spec.tooltip + diapazon/variantlar + standart qiymat."""
    parts = []
    if spec.tooltip:
        parts.append(spec.tooltip)
    if spec.kind in ("int", "float", "optint", "optfloat") and (spec.min is not None or spec.max is not None):
        parts.append(f"Diapazon: {_fmt_num(spec.min)} .. {_fmt_num(spec.max)}")
    if spec.kind == "choice":
        parts.append("Variantlar: " + ", ".join(map(str, spec.choices)))
    if spec.kind in ("optint", "optfloat"):
        parts.append(f"Belgilansa: {spec.none_label}")
    parts.append("Standart: " + _fmt_value(spec.default, spec.none_label))
    if spec.tunable:
        parts.append("Tuning'da qidirilishi mumkin.")
    return "\n".join(parts)


def _opt_initial(spec):
    """optint/optfloat uchun None bo'lmaganda ko'rinadigan boshlang'ich qiymat (ma'noli o'rta nuqta)."""
    lo = spec.search_min if spec.search_min is not None else spec.min
    hi = spec.search_max if spec.search_max is not None else spec.max
    if lo is None or hi is None:
        v = spec.default if spec.default is not None else (lo if lo is not None else 1)
    elif lo > 0 and hi / lo >= 100.0:
        v = math.sqrt(lo * hi)
    else:
        v = 0.5 * (lo + hi)
    if _is_int_kind(spec):
        v = int(round(v))
        if _is_odd_spec(spec) and v % 2 == 0:
            v += 1
        return v
    return float(round(v, spec.decimals))


def _nonfinite(v):
    """Chekli bo'lmagan (NaN/inf) son qiymatmi (bool va matnlar emas)."""
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and not math.isfinite(float(v))


def _safe_space_cfg(cfg, default):
    """Qidiruv oralig'i yozuvi yaroqsiz bo'lsa (buzuq preset) standart yozuvga qaytaradi: dialog qulamasligi uchun."""
    if not isinstance(cfg, dict):
        return default
    if default.get("type") == "choice":
        ch = cfg.get("choices")
        return cfg if isinstance(ch, (list, tuple)) and len(ch) > 0 else default
    try:
        lo, hi = float(cfg.get("min")), float(cfg.get("max"))
    except (TypeError, ValueError):
        return default
    return cfg if (math.isfinite(lo) and math.isfinite(hi)) else default


def _needed_decimals(v, base, limit=10):
    """v ni yo'qotishsiz ko'rsatish uchun kerakli kasr xonalari (kamida base, ko'pi bilan limit)."""
    d = int(base)
    while d < limit and abs(round(v, d) - v) > 1e-12 * max(1.0, abs(v)):
        d += 1
    return d


# ---------------------------------------------------------------------------
# G'ildirak bilan tasodifiy o'zgarishdan himoyalangan vidjetlar (scroll area ichida)
# ---------------------------------------------------------------------------
class _NoWheel:
    """Fokus bo'lmasa g'ildirak hodisasini ota vidjetga (scroll area) uzatadi: aylantirishda qiymat o'zgarmaydi."""

    def wheelEvent(self, event):                 # noqa: N802
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _SpinBox(_NoWheel, QSpinBox):
    pass


class _DoubleSpinBox(_NoWheel, QDoubleSpinBox):
    pass


class _ComboBox(_NoWheel, QComboBox):
    pass


# ---------------------------------------------------------------------------
# ParamField: bitta ParamSpec <-> vidjet
# ---------------------------------------------------------------------------
class ParamField(QObject):
    """Bitta parametr uchun vidjet(lar). `.editor` - asosiy kiritish vidjeti (spin/combo/checkbox),
    `.widget` - formaga qo'yiladigan konteyner (optional turlarda checkbox + spin), `.label` - yorliq."""

    changed = pyqtSignal()

    def __init__(self, model, spec, parent=None):
        super().__init__(parent)
        self.model = model
        self.spec = spec
        self.none_check = None
        self.label = QLabel(spec.label)
        tip = _spec_tooltip(spec)
        self.label.setToolTip(tip)
        kind = spec.kind
        if kind in ("int", "optint"):
            self.editor = self._make_int(spec)
        elif kind in ("float", "optfloat"):
            self.editor = self._make_float(spec)
        elif kind == "bool":
            self.editor = QCheckBox("yoqilgan")
            self.editor.toggled.connect(lambda _v: self.changed.emit())
        elif kind == "choice":
            self.editor = _ComboBox()
            self.editor.addItems([str(c) for c in spec.choices])
            self.editor.setFocusPolicy(Qt.StrongFocus)
            self.editor.currentIndexChanged.connect(lambda _i: self.changed.emit())
        else:
            raise ValueError(f"{model}.{spec.name}: noma'lum parametr turi '{kind}'")
        if kind in ("optint", "optfloat"):
            self.widget = QWidget()
            lay = QHBoxLayout(self.widget)
            lay.setContentsMargins(0, 0, 0, 0)
            self.none_check = QCheckBox(spec.none_label)
            self.none_check.setToolTip(tip)
            lay.addWidget(self.none_check)
            lay.addWidget(self.editor)
            lay.addStretch(1)
            self.editor.setValue(_opt_initial(spec))
            self.none_check.toggled.connect(self._on_none_toggled)
            self.none_check.setChecked(True)
            self._on_none_toggled(True)
        else:
            self.widget = self.editor
        self.editor.setToolTip(tip)
        self.reset()

    # ---- vidjet yaratish
    def _make_int(self, spec):
        sb = _SpinBox()
        sb.setFocusPolicy(Qt.StrongFocus)
        sb.setRange(int(spec.min) if spec.min is not None else -_INT_LIMIT,
                    int(spec.max) if spec.max is not None else _INT_LIMIT)
        sb.setSingleStep(int(spec.step or 1))
        sb.setMinimumWidth(90)
        if _is_odd_spec(spec):
            sb.setKeyboardTracking(False)        # yozayotganda emas, Enter/fokus yo'qolganda tuzatiladi
            sb.valueChanged.connect(self._fix_odd)
        else:
            sb.valueChanged.connect(lambda _v: self.changed.emit())
        return sb

    def _make_float(self, spec):
        sb = _DoubleSpinBox()
        sb.setFocusPolicy(Qt.StrongFocus)
        sb.setDecimals(int(spec.decimals))
        sb.setRange(float(spec.min) if spec.min is not None else -1e9,
                    float(spec.max) if spec.max is not None else 1e9)
        sb.setSingleStep(float(spec.step) if spec.step else 10.0 ** (-int(spec.decimals)))
        sb.setMinimumWidth(110)
        sb.valueChanged.connect(lambda _v: self.changed.emit())
        return sb

    # ---- ichki qayta chaqiruvlar
    def _fix_odd(self, v):
        """Toq bo'lishi shart parametr: juft kiritilsa +1 (yuqoriga sig'masa -1) qilinadi."""
        if v % 2 == 0:
            nv = v + 1 if v + 1 <= self.editor.maximum() else v - 1
            self.editor.setValue(nv)             # qayta chaqiriladi va changed chiqaradi
            return
        self.changed.emit()

    def _on_none_toggled(self, checked):
        self.editor.setEnabled(not checked)
        self.changed.emit()

    # ---- qiymat
    def value(self):
        """Joriy qiymat (Python turida; optional + belgilangan => None)."""
        kind = self.spec.kind
        if self.none_check is not None and self.none_check.isChecked():
            return None
        if kind in ("int", "optint"):
            return int(self.editor.value())
        if kind in ("float", "optfloat"):
            return float(self.editor.value())
        if kind == "bool":
            return bool(self.editor.isChecked())
        return self.editor.currentText()

    def set_value(self, v):
        """Qiymatni vidjetga qo'yadi (signal'lar chiqadi: to'plab chiqarish panelning vazifasi)."""
        spec = self.spec
        kind = spec.kind
        if _nonfinite(v):                        # NaN/inf (buzuq preset) => standart qiymat
            v = spec.default
        if kind in ("optint", "optfloat"):
            if v is None:
                self.none_check.setChecked(True)
                return
            self.none_check.setChecked(False)
        if kind in ("int", "optint"):
            self.editor.setValue(int(round(float(v))))
        elif kind in ("float", "optfloat"):
            f = float(v)
            self.editor.setDecimals(_needed_decimals(f, spec.decimals))
            self.editor.setValue(f)
        elif kind == "bool":
            self.editor.setChecked(bool(v))
        else:
            i = self.editor.findText(str(v))
            self.editor.setCurrentIndex(i if i >= 0 else max(0, self.editor.findText(str(spec.default))))

    def reset(self):
        self.set_value(self.spec.default)

    def set_enabled(self, enabled):
        """Parametrni (yorlig'i bilan) faollashtiradi/o'chiradi; qiymat saqlanadi."""
        self.widget.setEnabled(bool(enabled))
        self.label.setEnabled(bool(enabled))


# ---------------------------------------------------------------------------
# SearchSpaceDialog
# ---------------------------------------------------------------------------
COL_MODEL, COL_PARAM, COL_TYPE, COL_MIN, COL_MAX, COL_LOG, COL_CHOICES, COL_DEFAULT = range(8)


def _parse_float(text):
    """Matn -> chekli float; vergulli kasr ('0,01') ham qabul qilinadi. Xato bo'lsa ValueError."""
    t = str(text).strip().replace(",", ".")
    v = float(t)
    if not math.isfinite(v):
        raise ValueError("chekli son kerak")
    return v


def _parse_choices(spec, text):
    """'a, b, c' -> ([qiymatlar], xato|None) - spec turiga mos tur va diapazon tekshiruvi bilan."""
    toks = [t.strip() for t in str(text).replace(";", ",").split(",") if t.strip()]
    if not toks:
        return [], "kamida bitta variant kerak"
    out = []
    for t in toks:
        try:
            if _is_int_kind(spec):
                f = _parse_float(t)
                if abs(f - round(f)) > 1e-9:
                    return [], f"'{t}' butun son emas"
                v = int(round(f))
            elif spec.kind in ("float", "optfloat"):
                v = _parse_float(t)
            elif spec.kind == "bool":
                low = t.lower()
                if low in ("true", "1", "ha", "yes"):
                    v = True
                elif low in ("false", "0", "yo'q", "yoq", "no"):
                    v = False
                else:
                    return [], f"'{t}' mantiqiy qiymat emas"
            else:
                if t not in spec.choices:
                    return [], f"'{t}' ruxsat etilmagan (mumkin: {', '.join(map(str, spec.choices))})"
                v = t
        except ValueError:
            return [], f"'{t}' son emas"
        if spec.kind != "bool" and spec.kind != "choice":
            if spec.min is not None and v < spec.min:
                return [], f"{_fmt_num(v)} < {_fmt_num(spec.min)} (ruxsat etilgan eng kichik qiymat)"
            if spec.max is not None and v > spec.max:
                return [], f"{_fmt_num(v)} > {_fmt_num(spec.max)} (ruxsat etilgan eng katta qiymat)"
            if _is_odd_spec(spec) and v % 2 == 0:
                return [], f"{v} toq bo'lishi kerak"
        if v not in out:
            out.append(v)
    return out, None


class SearchSpaceDialog(QDialog):
    """Tunable parametrlarning qidiruv oraliqlarini jadvalda tahrirlash.

    Ustunlar: Model, Parametr, Tur, Min, Max, Log, Variantlar (vergul bilan), Standart. Son parametrlarda Min/Max/Log,
    variantli (choice) parametrlarda faqat Variantlar tahrirlanadi. OK bosilganda tekshiriladi (min < max, log uchun
    min > 0, parametrning ruxsat etilgan diapazonida, variantlar to'g'ri turda); xato bo'lsa dialog yopilmaydi va xatolar
    dialog ichida ko'rsatiladi. Muvaffaqiyatda `tuning.spaces` yangilanadi (faqat standartdan farq qiladiganlar)."""

    def __init__(self, tuning, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Qidiruv oraliqlarini tahrirlash")
        self.tuning = tuning
        self._rows = []                          # {"model","name","spec","type","default","row"}
        lay = QVBoxLayout(self)
        info = QLabel("Tuning har bir nomzod uchun parametrlarni shu oraliqda tasodifiy tanlaydi. "
                      "Son parametrlar: min < max (log yoqilsa min > 0); variantli parametrlar: vergul bilan "
                      "ajratilgan qiymatlar. Oraliq parametrning ruxsat etilgan chegarasidan chiqmasligi kerak.")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Model", "Parametr", "Tur", "Min", "Max", "Log",
                                              "Variantlar (vergul bilan)", "Standart oraliq"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(False)
        lay.addWidget(self.table, 1)
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #c62828;")
        self.error_label.setVisible(False)
        lay.addWidget(self.error_label)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        bb.button(QDialogButtonBox.Ok).setText("OK")
        bb.button(QDialogButtonBox.Cancel).setText("Bekor qilish")
        bb.button(QDialogButtonBox.RestoreDefaults).setText("Standartga qaytarish")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.reset_to_defaults)
        lay.addWidget(bb)
        self._build_rows()
        self.resize(900, 520)

    # ---- jadvalni qurish
    def _build_rows(self):
        self.table.setRowCount(0)
        self._rows = []
        for model, specs in config.PARAM_SPECS.items():
            defaults = config.default_search_space(model)
            try:
                current = self.tuning.resolved_space(model)
            except Exception:                    # noqa: BLE001  (buzuq spaces => standart oraliqlar)
                current = defaults
            for spec in specs:
                if spec.name not in defaults:
                    continue
                self._add_row(model, spec, defaults[spec.name],
                              _safe_space_cfg(current.get(spec.name), defaults[spec.name]))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    @staticmethod
    def _ro_item(text, tip=""):
        it = QTableWidgetItem(text)
        it.setFlags(Qt.ItemIsEnabled)
        if tip:
            it.setToolTip(tip)
        return it

    def _add_row(self, model, spec, default_cfg, cfg):
        r = self.table.rowCount()
        self.table.insertRow(r)
        ptype = default_cfg.get("type", "choice" if default_cfg.get("choices") else "float")
        tip = _spec_tooltip(spec)
        self.table.setItem(r, COL_MODEL, self._ro_item(MODEL_TITLES.get(model, model)))
        self.table.setItem(r, COL_PARAM, self._ro_item(spec.name, f"{spec.label}\n{tip}"))
        self.table.setItem(r, COL_TYPE, self._ro_item(ptype))
        if ptype == "choice":
            dtext = ", ".join(_fmt_num(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else str(c)
                              for c in default_cfg["choices"])
        else:
            dtext = f"{_fmt_num(default_cfg.get('min'))} .. {_fmt_num(default_cfg.get('max'))}" + \
                    (" (log)" if default_cfg.get("log") else "")
        self.table.setItem(r, COL_DEFAULT, self._ro_item(dtext))
        for c in (COL_MIN, COL_MAX, COL_LOG, COL_CHOICES):
            self.table.setItem(r, c, QTableWidgetItem(""))
        self._rows.append({"model": model, "name": spec.name, "spec": spec, "type": ptype,
                           "default": default_cfg, "row": r})
        self._fill_row(self._rows[-1], cfg)

    def _fill_row(self, row, cfg):
        r, t = row["row"], self.table
        editable = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
        if row["type"] == "choice":
            ch = cfg.get("choices") or []
            t.item(r, COL_CHOICES).setText(", ".join(_fmt_num(c) if isinstance(c, (int, float))
                                                     and not isinstance(c, bool) else str(c) for c in ch))
            t.item(r, COL_CHOICES).setFlags(editable)
            for c in (COL_MIN, COL_MAX, COL_LOG):
                t.item(r, c).setText("")
                t.item(r, c).setFlags(Qt.NoItemFlags)
        else:
            t.item(r, COL_MIN).setText(_fmt_num(cfg.get("min")))
            t.item(r, COL_MAX).setText(_fmt_num(cfg.get("max")))
            t.item(r, COL_MIN).setFlags(editable)
            t.item(r, COL_MAX).setFlags(editable)
            log_item = t.item(r, COL_LOG)
            log_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            log_item.setCheckState(Qt.Checked if cfg.get("log") else Qt.Unchecked)
            t.item(r, COL_CHOICES).setText("")
            t.item(r, COL_CHOICES).setFlags(Qt.NoItemFlags)
        for c in (COL_MIN, COL_MAX, COL_LOG, COL_CHOICES):
            t.item(r, c).setBackground(QBrush())

    # ---- dasturiy kirish (testlar va tashqi kod uchun)
    def row_of(self, model, name):
        for row in self._rows:
            if row["model"] == model and row["name"] == name:
                return row["row"]
        raise KeyError(f"{model}.{name}")

    def cell(self, model, name, col):
        return self.table.item(self.row_of(model, name), col)

    def set_row(self, model, name, *, min=None, max=None, log=None, choices=None):     # noqa: A002
        """Qatorni to'ldiradi (matn/son/ro'yxat qabul qiladi); xatolik tekshiruvi OK bosilganda."""
        r = self.row_of(model, name)
        if min is not None:
            self.table.item(r, COL_MIN).setText(_fmt_num(min) if not isinstance(min, str) else min)
        if max is not None:
            self.table.item(r, COL_MAX).setText(_fmt_num(max) if not isinstance(max, str) else max)
        if log is not None:
            self.table.item(r, COL_LOG).setCheckState(Qt.Checked if log else Qt.Unchecked)
        if choices is not None:
            txt = choices if isinstance(choices, str) else ", ".join(_fmt_num(c) if isinstance(c, (int, float))
                                                                    and not isinstance(c, bool) else str(c)
                                                                    for c in choices)
            self.table.item(r, COL_CHOICES).setText(txt)

    def reset_to_defaults(self):
        """Barcha qatorlarni standart oraliqlarga qaytaradi."""
        for row in self._rows:
            self._fill_row(row, row["default"])
        self._show_errors([])

    # ---- tekshiruv va yig'ish
    def collect(self):
        """(spaces, errors): spaces = {model: {param: {...}}} (faqat standartdan farq qiladiganlar)."""
        spaces, errors = {}, []
        t = self.table
        for row in self._rows:
            r, spec, model = row["row"], row["spec"], row["model"]
            where = f"{MODEL_TITLES.get(model, model)}.{spec.name}"
            bad_cols = []
            new = None
            if row["type"] == "choice":
                choices, err = _parse_choices(spec, t.item(r, COL_CHOICES).text())
                if err:
                    errors.append(f"{where}: {err}")
                    bad_cols.append(COL_CHOICES)
                else:
                    new = {"choices": choices}
            else:
                vals = {}
                for col, key in ((COL_MIN, "min"), (COL_MAX, "max")):
                    try:
                        vals[key] = _parse_float(t.item(r, col).text())
                    except ValueError:
                        errors.append(f"{where}: {key} son bo'lishi kerak")
                        bad_cols.append(col)
                if len(vals) == 2:
                    lo, hi = vals["min"], vals["max"]
                    log = t.item(r, COL_LOG).checkState() == Qt.Checked
                    if row["type"] == "int":
                        for col, key in ((COL_MIN, "min"), (COL_MAX, "max")):
                            if abs(vals[key] - round(vals[key])) > 1e-9:
                                errors.append(f"{where}: {key} butun son bo'lishi kerak")
                                bad_cols.append(col)
                        lo, hi = int(round(lo)), int(round(hi))
                    if not bad_cols:
                        if not lo < hi:
                            errors.append(f"{where}: min < max bo'lishi kerak ({_fmt_num(lo)} >= {_fmt_num(hi)})")
                            bad_cols += [COL_MIN, COL_MAX]
                        if log and lo <= 0:
                            errors.append(f"{where}: log shkala uchun min > 0 bo'lishi kerak")
                            bad_cols += [COL_MIN, COL_LOG]
                        if spec.min is not None and lo < spec.min:
                            errors.append(f"{where}: min {_fmt_num(lo)} < ruxsat etilgan {_fmt_num(spec.min)}")
                            bad_cols.append(COL_MIN)
                        if spec.max is not None and hi > spec.max:
                            errors.append(f"{where}: max {_fmt_num(hi)} > ruxsat etilgan {_fmt_num(spec.max)}")
                            bad_cols.append(COL_MAX)
                    if not bad_cols:
                        new = {"min": lo, "max": hi, "log": bool(log)}
            for c in (COL_MIN, COL_MAX, COL_LOG, COL_CHOICES):
                it = t.item(r, c)
                if it is not None:
                    it.setBackground(QBrush(_BAD_BG) if c in bad_cols else QBrush())
            if new is not None and not self._same_as_default(new, row["default"]):
                spaces.setdefault(model, {})[spec.name] = new
        return spaces, errors

    @staticmethod
    def _same_as_default(new, default):
        if "choices" in new:
            return list(new["choices"]) == list(default.get("choices") or [])
        return (float(new["min"]) == float(default.get("min")) and float(new["max"]) == float(default.get("max"))
                and bool(new["log"]) == bool(default.get("log")))

    def _show_errors(self, errors):
        self.error_label.setText("\n".join(errors))
        self.error_label.setVisible(bool(errors))

    def accept(self):
        spaces, errors = self.collect()
        self._show_errors(errors)
        if errors:
            return                               # dialog yopilmaydi
        self.tuning.spaces = spaces
        super().accept()


# ---------------------------------------------------------------------------
# TuningGroup
# ---------------------------------------------------------------------------
class TuningGroup(QGroupBox):
    """Giperparametr qidirish (tuning) sozlamalari. `changed` signali; `.get_tuning()` -> TuningConfig."""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Giperparametrlarni avtomatik qidirish (tuning, nested spatial CV)", parent)
        self.setCheckable(True)
        self.setChecked(False)
        self._mode = "nested"
        self._spaces = {}
        self._extra_models = {}                  # tunable bo'lmagan/noma'lum model kalitlari (saqlanadi)
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Qidiriladigan modellar:"))
        self.model_checks = {}
        defaults = TuningConfig().models
        for model, specs in config.PARAM_SPECS.items():
            if not any(s.tunable for s in specs):
                continue
            title = MODEL_TITLES.get(model, model)
            if model == "CNN":
                cb = QCheckBox("CNN (qimmat!)")
                cb.setToolTip("Qimmat: har nomzod uchun neyron tarmoq o'qitiladi (RF/SVM/XGBoost'dan ancha sekin), "
                              "nested CV da TensorFlow xotirasi ham o'sadi (~8 MB/fit). Odatda tavsiya etilmaydi.")
            else:
                cb = QCheckBox(title)
                cb.setToolTip(f"{title} tunable parametrlari qidiriladi.")
            cb.setChecked(bool(defaults.get(model, False)))
            cb.toggled.connect(lambda _v: self.changed.emit())
            self.model_checks[model] = cb
            row.addWidget(cb)
        row.addStretch(1)
        lay.addLayout(row)

        form = QHBoxLayout()                      # 3 ta parametr bitta qatorda (vertikal joy tejaladi)
        self.n_iter = _SpinBox()
        self.n_iter.setRange(1, 2000)
        self.n_iter.setValue(TuningConfig().n_iter)
        self.n_iter.setFocusPolicy(Qt.StrongFocus)
        self.n_iter.setToolTip("Har bir tashqi fold uchun tasodifiy nomzodlar soni (+ boshlang'ich giperparametrlar).")
        self.inner_splits = _SpinBox()
        self.inner_splits.setRange(2, 10)
        self.inner_splits.setValue(TuningConfig().inner_splits)
        self.inner_splits.setFocusPolicy(Qt.StrongFocus)
        self.inner_splits.setToolTip("Nomzodlarni baholash uchun ichki spatial CV fold'lari soni.")
        self.scoring = _ComboBox()
        self.scoring.setFocusPolicy(Qt.StrongFocus)
        self.scoring.addItem("ROC AUC", "roc_auc")
        self.scoring.addItem("PR AUC (average precision)", "average_precision")
        self.scoring.setToolTip("Nomzodlarni solishtirish mezoni.")
        for text, w in (("Nomzodlar soni (n_iter):", self.n_iter), ("Ichki fold'lar:", self.inner_splits),
                        ("Baholash mezoni:", self.scoring)):
            form.addWidget(QLabel(text))
            form.addWidget(w)
            form.addSpacing(12)
        form.addStretch(1)
        lay.addLayout(form)

        srow = QHBoxLayout()
        self.btn_spaces = QPushButton("Qidiruv oraliqlarini tahrirlash...")
        self.btn_spaces.clicked.connect(self.edit_spaces)
        self.spaces_label = QLabel("")
        srow.addWidget(self.btn_spaces)
        srow.addWidget(self.spaces_label, 1)
        lay.addLayout(srow)

        note = QLabel("Eslatma: tuning hisoblash vaqtini bir necha baravar oshiradi (nomzodlar x ichki fold'lar x "
                      "tashqi fold'lar). CNN tuning va nested CV da TensorFlow xotirasi o'sadi.")
        note.setWordWrap(True)
        lay.addWidget(note)

        self.toggled.connect(lambda _v: self.changed.emit())
        self.n_iter.valueChanged.connect(lambda _v: self.changed.emit())
        self.inner_splits.valueChanged.connect(lambda _v: self.changed.emit())
        self.scoring.currentIndexChanged.connect(lambda _i: self.changed.emit())
        self._update_spaces_label()

    # ---- holat
    def _update_spaces_label(self):
        n = sum(len(v) for v in self._spaces.values() if isinstance(v, dict))
        self.spaces_label.setText("Oraliqlar: standart" if n == 0 else f"{n} ta parametr oralig'i o'zgartirilgan")

    def get_tuning(self):
        """Joriy sozlamalar -> yangi TuningConfig."""
        models = copy.deepcopy(self._extra_models)
        for m, cb in self.model_checks.items():
            models[m] = bool(cb.isChecked())
        return TuningConfig(enabled=bool(self.isChecked()), mode=self._mode, models=models,
                            n_iter=int(self.n_iter.value()), inner_splits=int(self.inner_splits.value()),
                            scoring=str(self.scoring.currentData()), spaces=copy.deepcopy(self._spaces))

    def set_tuning(self, t):
        """TuningConfig (yoki lug'at) ni vidjetlarga qo'yadi; `changed` bir marta chiqadi."""
        if t is None:
            t = TuningConfig()
        t = TuningConfig.from_dict(t.to_dict() if isinstance(t, TuningConfig) else dict(t))
        widgets = [self, self.n_iter, self.inner_splits, self.scoring, *self.model_checks.values()]
        for w in widgets:
            w.blockSignals(True)
        try:
            self.setChecked(bool(t.enabled))
            self._mode = t.mode
            self._extra_models = {m: bool(v) for m, v in (t.models or {}).items() if m not in self.model_checks}
            for m, cb in self.model_checks.items():
                cb.setChecked(bool((t.models or {}).get(m, False)))
            self.n_iter.setValue(int(t.n_iter))
            self.inner_splits.setValue(int(t.inner_splits))
            i = self.scoring.findData(t.scoring)
            self.scoring.setCurrentIndex(i if i >= 0 else 0)
            self._spaces = copy.deepcopy(t.spaces or {})
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._update_spaces_label()
        self.changed.emit()

    def edit_spaces(self):
        """Oraliqlar dialogini ochadi; OK bosilsa saqlaydi va True qaytaradi."""
        t = self.get_tuning()
        dlg = SearchSpaceDialog(t, self)
        if dlg.exec_() == QDialog.Accepted:
            self._spaces = copy.deepcopy(t.spaces)
            self._update_spaces_label()
            self.changed.emit()
            return True
        return False


# ---------------------------------------------------------------------------
# HyperParamPanel
# ---------------------------------------------------------------------------
class HyperParamPanel(QWidget):
    """Barcha modellar giperparametrlari paneli (PARAM_SPECS dan avtomatik) + tuning + preset tugmalari."""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fields = {}                        # (model, name) -> ParamField
        self._loading = False
        self._model_page = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # --- tugmalar
        btns = QHBoxLayout()
        self.btn_reset_model = QPushButton("Standartga qaytarish (joriy model)")
        self.btn_reset_all = QPushButton("Hammasini standartga qaytarish")
        self.btn_save = QPushButton("Preset saqlash...")
        self.btn_load = QPushButton("Preset yuklash...")
        self.btn_reset_model.clicked.connect(lambda: self.reset_defaults(self.current_model()))
        self.btn_reset_all.clicked.connect(lambda: self.reset_defaults(None))
        self.btn_save.clicked.connect(self.save_preset_dialog)
        self.btn_load.clicked.connect(self.load_preset_dialog)
        self.btn_reset_model.setToolTip("Joriy model (tab) giperparametrlarini standart qiymatlarga qaytarish.")
        self.btn_save.setToolTip("Giperparametrlar va tuning sozlamalarini JSON presetga saqlash.")
        self.btn_load.setToolTip("Giperparametrlar va tuning sozlamalarini JSON presetdan yuklash.")
        for b in (self.btn_reset_model, self.btn_reset_all, self.btn_save, self.btn_load):
            btns.addWidget(b)
        btns.addStretch(1)
        root.addLayout(btns)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)                   # bo'sh bo'lganda joy egallamasin
        root.addWidget(self.status_label)

        # --- umumiy xarajat izohi
        # (uzun matn joyni egallamasin: balandligi cheklangan, o'z scroll'i bor)
        self.cost_label = QLabel("")
        self.cost_label.setWordWrap(True)
        self.cost_label.setFrameShape(QFrame.StyledPanel)
        self.cost_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.cost_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.cost_scroll = QScrollArea()
        self.cost_scroll.setWidgetResizable(True)
        self.cost_scroll.setFrameShape(QFrame.NoFrame)
        self.cost_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.cost_scroll.setWidget(self.cost_label)
        self.cost_scroll.setVisible(False)
        root.addWidget(self.cost_scroll)

        # --- model tablari
        self.tabs = QTabWidget()
        for model, specs in config.PARAM_SPECS.items():
            self.tabs.addTab(self._build_model_page(model, specs), MODEL_TITLES.get(model, model))
            self._model_page[model] = self.tabs.count() - 1
        root.addWidget(self.tabs, 1)

        # --- tuning
        self.tuning_group = TuningGroup()
        root.addWidget(self.tuning_group)
        self.tuning_group.changed.connect(self._emit_changed)

        self._apply_dependencies()

    # ---- qurish
    def _build_model_page(self, model, specs):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        vlay = QVBoxLayout(inner)
        groups = {}
        order = []
        for spec in specs:
            g = spec.group or "Boshqa"
            if g not in groups:
                box = QGroupBox(g)
                form = QFormLayout(box)
                form.setRowWrapPolicy(QFormLayout.WrapLongRows)
                form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
                groups[g] = form
                order.append((g, box))
            field = ParamField(model, spec, parent=self)
            groups[g].addRow(field.label, field.widget)
            self._fields[(model, spec.name)] = field
            field.changed.connect(self._on_field_changed)
        for _g, box in order:
            vlay.addWidget(box)
        vlay.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    # ---- signal'lar va bog'liqliklar
    def _emit_changed(self):
        if not self._loading:
            self.changed.emit()

    def _on_field_changed(self):
        if self._loading:
            return
        self._apply_dependencies()
        self.changed.emit()

    def _apply_dependencies(self):
        for model, ctrl, dependents, disable_if in _DEPENDENCIES:
            c = self._fields.get((model, ctrl))
            if c is None:
                continue
            off = bool(disable_if(c.value()))
            for name in dependents:
                f = self._fields.get((model, name))
                if f is not None:
                    f.set_enabled(not off)

    # ---- kirish (introspeksiya)
    @property
    def fields(self):
        """{(model, parametr): ParamField}"""
        return self._fields

    def field(self, model, name):
        return self._fields[(model, name)]

    def widget_for(self, model, name):
        """Parametrning asosiy kiritish vidjeti (spin/combo/checkbox)."""
        return self._fields[(model, name)].editor

    def models(self):
        return list(self._model_page)

    def current_model(self):
        return list(self._model_page)[self.tabs.currentIndex()]

    def set_current_model(self, model):
        self.tabs.setCurrentIndex(self._model_page[model])

    # ---- giperparametrlar
    def get_hyperparams(self):
        """To'liq giperparametr lug'ati (validate_hyperparams dan o'tgan): {model: {param: qiymat}}."""
        hp = {m: {} for m in self._model_page}
        for (m, n), f in self._fields.items():
            hp[m][n] = f.value()
        return config.validate_hyperparams(hp)[0]

    def _load_values(self, clean):
        self._loading = True
        try:
            for (m, n), f in self._fields.items():
                if m in clean and n in clean[m]:
                    f.set_value(clean[m][n])
        finally:
            self._loading = False
        self._apply_dependencies()

    def set_hyperparams(self, hp):
        """Berilgan qiymatlarni paneldagilar ustiga qo'yadi (berilmaganlari o'zgarmaydi), validatsiyadan o'tkazadi.
        Ogohlantirishlar ro'yxatini qaytaradi; `changed` bir marta chiqadi."""
        merged = self.get_hyperparams()
        warns = []
        for m, p in (hp or {}).items():
            if m in merged and isinstance(p, dict):
                for k, v in p.items():
                    if _nonfinite(v):            # NaN/inf: jimgina maksimumga aylanmasligi uchun e'tiborsiz
                        warns.append(f"{m}.{k}: chekli bo'lmagan qiymat e'tiborsiz qoldirildi")
                    else:
                        merged[m][k] = v
            else:
                warns.append(f"Noma'lum model e'tiborsiz qoldirildi: {m}")
        clean, w2 = config.validate_hyperparams(merged)
        self._load_values(clean)
        self.changed.emit()
        return warns + w2

    def reset_defaults(self, model=None):
        """Standart qiymatlarga qaytaradi: model=None => hamma modellar, aks holda faqat shu model."""
        if model is not None and model not in self._model_page:
            raise KeyError(f"Noma'lum model: {model}")
        defaults = config.default_hyperparams()
        if model is not None:
            defaults = {model: defaults[model]}
        self._load_values(defaults)
        self.changed.emit()

    # ---- tuning
    def get_tuning(self):
        return self.tuning_group.get_tuning()

    def set_tuning(self, t):
        self.tuning_group.set_tuning(t)

    # ---- xarajat izohi
    def set_cost_hint(self, text):
        """estimate_cost_text natijasini ko'rsatadi (bo'sh matn => yashiriladi)."""
        text = "" if text is None else str(text)
        self.cost_label.setText(text)
        self.cost_scroll.setVisible(bool(text.strip()))
        self._fit_cost_height()

    def _set_status(self, text):
        self.status_label.setText(text)
        self.status_label.setVisible(bool(text))

    def _fit_cost_height(self):
        """Xarajat izohi balandligi: matnga mos, lekin COST_MAX_HEIGHT dan oshmaydi (oshsa - ichki scroll)."""
        try:
            w = max(200, self.cost_scroll.viewport().width() - 4)
            h = self.cost_label.heightForWidth(w) if self.cost_label.hasHeightForWidth() else self.cost_label.sizeHint().height()
            self.cost_scroll.setFixedHeight(int(min(COST_MAX_HEIGHT, max(24, h + 6))))
        except RuntimeError:
            pass

    def resizeEvent(self, event):                # noqa: N802
        super().resizeEvent(event)
        if self.cost_scroll.isVisibleTo(self):
            self._fit_cost_height()

    def cost_hint(self):
        return self.cost_label.text()

    # ---- preset
    def save_preset_to(self, path):
        """Joriy giperparametr + tuning ni JSON presetga yozadi. Yo'lni qaytaradi."""
        config.save_preset(path, hyperparams=self.get_hyperparams(), tuning=self.get_tuning())
        return path

    def load_preset_from(self, path):
        """Presetni o'qib panelga to'liq qo'llaydi (preset'dagi hamma giperparametr almashadi). To'liq RunConfig
        fayli bo'lsa ham faqat giperparametr va tuning olinadi. Ogohlantirishlar ro'yxatini qaytaradi;
        fayl/JSON xatosi istisno sifatida ko'tariladi."""
        res = config.load_preset(path)
        self._load_values(res["hyperparams"])
        if res.get("tuning") is not None:
            self.tuning_group.set_tuning(res["tuning"])
        self.changed.emit()
        return list(res.get("warnings") or [])

    def _show_message(self, kind, title, text):
        """Xabar oynasi (testlarda almashtirish mumkin)."""
        fn = {"warning": QMessageBox.warning, "critical": QMessageBox.critical}.get(kind, QMessageBox.information)
        fn(self, title, text)

    def save_preset_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Preset saqlash", "mpm_preset.json", "JSON (*.json)")
        if not path:
            return None
        try:
            self.save_preset_to(path)
        except Exception as exc:                 # noqa: BLE001
            self._show_message("critical", "Preset saqlanmadi", f"{type(exc).__name__}: {exc}")
            return None
        self._set_status(f"Preset saqlandi: {path}")
        return path

    def load_preset_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Preset yuklash", "", "JSON (*.json)")
        if not path:
            return None
        try:
            warns = self.load_preset_from(path)
        except Exception as exc:                 # noqa: BLE001
            self._show_message("critical", "Preset yuklanmadi", f"{type(exc).__name__}: {exc}")
            return None
        self._set_status(f"Preset yuklandi: {os.path.basename(path)}")
        if warns:
            self._show_message("warning", "Preset ogohlantirishlari", "\n".join(warns))
        return path
