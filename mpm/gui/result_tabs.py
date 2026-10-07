# -*- coding: utf-8 -*-
"""
Natija ko'rsatuvchi tab-vidjetlar (PyQt5). Har biri `QWidget`; ichida `widgets.MplCanvas` / `DataFrameTable`,
chizish `plots.draw_*` orqali. Hammasi xatoga chidamli: bo'sh/None natijada "Hali natija yo'q" yoziladi,
chizish/jadval xatosi slotni qulatmaydi (BUG-01) - xato `last_error` va tab ichidagi qizil yozuvda ko'rinadi.

  DiagnosticsTab  (3-tab "Ma'lumotlar tahlili")    set_diagnostics(diag, data_dictionary=None) clear()  refresh_requested()
  ResultsTab      (4-tab "Natijalar")              set_result(result) clear() figures()                 export_requested()
  SpatialTab      (5-tab "Spatial CV diagnostika") set_result(result) clear() figures()
  ImportanceTab   (6-tab "Feature importance")     set_result(result) clear() figures()
  MapTab          (7-tab "Prognoz xarita")         set_result(result) set_prediction(pred) set_busy(b) clear() figures()
                                                   predict_requested(dict) export_requested()

Barcha tab'larda: `set_busy(bool)` (tugmalarni o'chirish), `has_result()`, `last_error`, signal `message(str)` (log uchun:
fayl saqlandi/xato). Tab'lar QScrollArea ichida: canvas kamida ~6x4.5 dyuym (kichik oynada constrained_layout
ogohlantirishi bo'lmasligi uchun). Chiqish - "prospektivlik indeksi" (ehtimollik emas).
"""
from __future__ import annotations

import functools
import logging
import math
import os
import re

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFrame, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QListWidget, QPushButton, QScrollArea, QSpinBox,
                             QStackedWidget, QTabWidget, QVBoxLayout, QWidget)

from ..common import ENSEMBLE_NAME
from . import plots
from .widgets import DataFrameTable, FolderPicker, MplCanvas

__all__ = ["DiagnosticsTab", "ResultsTab", "SpatialTab", "ImportanceTab", "MapTab", "EMPTY_TEXT", "INDEX_NOTE",
           "parse_breaks", "format_breaks", "downsample_map", "tuning_dataframe", "importance_dataframe",
           "bg_sensitivity_frames", "fold_table_frame"]

_log = logging.getLogger("mpm.gui.result_tabs")

EMPTY_TEXT = "Hali natija yo'q"
INDEX_NOTE = "Chiqish - prospektivlik indeksi (0-1), haqiqiy ehtimollik emas: fon nisbati sun'iy"
CNN_NOTE = ("CNN chiqishi kalibrlanmaydi va neg/pos namuna og'irligi bilan o'qitiladi, ansambl esa oddiy o'rtacha: "
            "CNN qiymatlari boshqa modellar bilan bir shkalada bo'lmasligi mumkin.")
CI_NOTE = ("AUC - CV takrorlari o'rtachasi; 95% CI - o'rtacha OOF bashorat ustida blok-bootstrap, shuning uchun nuqtaviy "
           "AUC CI chetida yoki undan tashqarida chiqishi mumkin (xato emas). Sens/Spec - Youden bo'sag'ida.")
MAP_MAX_PX_SHOW = 2500                  # ekranda ko'rsatish uchun xarita tomoni (stride bilan kamaytiriladi)
MAP_MAX_PX_SAVE = 4000                  # rasmga saqlash uchun
_MAX_CLASSES_GUI = 12
_GRAY = "color: gray;"
_ERR = "color: #B00020;"
_VIF_COLORS = {"yuqori": QColor(255, 205, 205), "o'rta": QColor(255, 236, 190), "o'zgarmas": QColor(214, 214, 214)}


# ---------------------------------------------------------------------------
# Qt'siz yordamchilar (testlanadi)
# ---------------------------------------------------------------------------
def parse_breaks(text):
    """'0.2, 0.4; 0.6 0.8' -> [0.2, 0.4, 0.6, 0.8]. Chegaralar chekli, (0, 1) ichida va qat'iy o'suvchi bo'lishi shart;
    aks holda ValueError (o'zbekcha xabar bilan)."""
    parts = [p for p in re.split(r"[,;\s]+", str(text or "").strip()) if p]
    if not parts:
        raise ValueError("Qat'iy chegaralar kiritilmagan (masalan: 0.2, 0.4, 0.6, 0.8).")
    vals = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            raise ValueError(f"Chegara raqam emas: '{p}'. Vergul bilan ajrating (kasr uchun nuqta: 0.25).") from None
        if not math.isfinite(v):
            raise ValueError(f"Chegara chekli son bo'lishi kerak: '{p}'.")
        vals.append(v)
    if any(v <= 0.0 or v >= 1.0 for v in vals):
        raise ValueError("Chegaralar prospektivlik indeksi oralig'ida (0 < chegara < 1) bo'lishi kerak.")
    if any(b <= a for a, b in zip(vals, vals[1:])):
        raise ValueError("Chegaralar qat'iy o'suvchi bo'lishi kerak (masalan: 0.2, 0.4, 0.6, 0.8).")
    if len(vals) + 1 > _MAX_CLASSES_GUI:
        raise ValueError(f"Sinflar soni ko'pi bilan {_MAX_CLASSES_GUI} bo'lishi mumkin ({len(vals)} chegara berilgan).")
    return vals


def format_breaks(breaks):
    """[0.2, 0.4] -> '0.2, 0.4' (bo'sh/None => '')."""
    if breaks is None:
        return ""
    try:
        return ", ".join(f"{float(b):.6g}" for b in breaks)
    except (TypeError, ValueError):
        return ""


def downsample_map(arr, transform=None, max_px=MAP_MAX_PX_SHOW):
    """Katta xaritani stride bilan max_px (tomon bo'yicha) ga kamaytiradi va transform'ni mos masshtablaydi
    (draw_map 16 mln pikselda sekin). Qaytaradi: (massiv, transform). Kichik xarita o'zgarmaydi."""
    a = np.asarray(arr)
    if a.ndim != 2 or a.size == 0:
        return a, transform
    step = int(math.ceil(max(a.shape) / float(max(1, int(max_px)))))
    if step <= 1:
        return a, transform
    tr = transform
    if transform is not None:
        try:
            from affine import Affine
            tr = transform @ Affine.scale(step)
        except Exception:                                    # noqa: BLE001 - Affine bo'lmasa 6 elementli kortej
            try:
                t = [float(v) for v in transform]
                t[0] *= step
                t[4] *= step
                tr = tuple(t)
            except Exception:                                # noqa: BLE001
                tr = None
    return a[::step, ::step], tr


def _hp_text(params):
    if not isinstance(params, dict):
        return "" if params is None else str(params)
    return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())


def _num(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def tuning_dataframe(result):
    """Nested (tashqi fold) va yakuniy tuning yozuvlari jadvali (bo'sh bo'lsa bo'sh DataFrame)."""
    cols = ["Model", "Doira", "Takror", "Fold", "Ball", "Bazaviy ball", "Ball turi", "Nomzodlar",
            "Tanlangan giperparametrlar", "Zaxira"]
    rows = []

    def add(model, scope, rec):
        if not isinstance(rec, dict):
            return
        rep, fold = rec.get("repeat"), rec.get("fold")
        rows.append({
            "Model": str(model), "Doira": scope,
            "Takror": None if rep is None else int(rep) + 1, "Fold": None if fold is None else int(fold) + 1,
            "Ball": _num(rec.get("best_score")), "Bazaviy ball": _num(rec.get("base_score")),
            "Ball turi": rec.get("scoring") or "", "Nomzodlar": rec.get("n_trials"),
            "Tanlangan giperparametrlar": _hp_text(rec.get("best_params")),
            "Zaxira": str(rec.get("fallback") or rec.get("skipped_reason") or "")})

    if isinstance(result, dict):
        sp = result.get("spatial") if isinstance(result.get("spatial"), dict) else {}
        tp = sp.get("tuned_params") if isinstance(sp.get("tuned_params"), dict) else {}
        for m, recs in tp.items():
            for rec in (recs if isinstance(recs, (list, tuple)) else [recs]):
                add(m, "nested (tashqi fold)", rec)
        ft = result.get("final_tuning") if isinstance(result.get("final_tuning"), dict) else {}
        for m, rec in ft.items():
            add(m, "yakuniy (butun ma'lumot)", rec)
    return pd.DataFrame(rows, columns=cols, dtype=object)


def importance_dataframe(result):
    """Feature importance jadvali: har model uchun o'rtacha (va permutation bo'lsa std). MDI zaxirasida std NaN
    (nollar xato emas), ustun sarlavhasida manba ('permutation'/'mdi') ko'rsatiladi. Birinchi modelning
    o'rtachasi bo'yicha kamayish tartibida."""
    imp = result.get("importance") if isinstance(result, dict) and isinstance(result.get("importance"), dict) else {}
    models = imp.get("models") if isinstance(imp.get("models"), dict) else {}
    names = list(imp.get("feature_names") or (result.get("feature_names") if isinstance(result, dict) else None) or [])
    data = {}
    first = None
    for k, v in models.items():
        if not isinstance(v, dict):
            continue
        try:
            mean = np.asarray(v.get("mean"), dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if mean.size == 0:
            continue
        if not names:
            names = [f"feature_{i + 1}" for i in range(mean.size)]
        if mean.size != len(names):
            continue
        src = str(v.get("source") or "?")
        std = np.full(mean.size, np.nan)
        if src == "permutation":
            try:
                s = np.asarray(v.get("std"), dtype=float).ravel()
                if s.size == mean.size:
                    std = s
            except (TypeError, ValueError):
                pass
        data[f"{k} [{src}] o'rtacha"] = mean
        data[f"{k} std"] = std
        if first is None:
            first = f"{k} [{src}] o'rtacha"
    if not data:
        return pd.DataFrame(columns=["Feature"])
    df = pd.DataFrame({"Feature": names, **data})
    return df.sort_values(first, ascending=False, na_position="last", kind="stable").reset_index(drop=True)


def bg_sensitivity_frames(bg):
    """(xulosa DataFrame, har-draw DataFrame); bg None/bo'sh bo'lsa (None, None)."""
    if not isinstance(bg, dict) or not isinstance(bg.get("summary"), dict) or not bg["summary"]:
        return None, None
    rows = []
    for name, s in bg["summary"].items():
        if not isinstance(s, dict):
            continue
        vals = s.get("values")
        rows.append({"Model": name, "AUC o'rtacha": _num(s.get("mean")), "std": _num(s.get("std")),
                     "min": _num(s.get("min")), "max": _num(s.get("max")),
                     "Tanlovlar soni": len(vals) if hasattr(vals, "__len__") else None})
    summ = pd.DataFrame(rows, columns=["Model", "AUC o'rtacha", "std", "min", "max", "Tanlovlar soni"], dtype=object)
    drows = []
    for rec in bg.get("per_draw") or []:
        if not isinstance(rec, dict):
            continue
        d = {"Tanlov": None if rec.get("draw") is None else int(rec["draw"]) + 1, "Seed": rec.get("seed"),
             "Fon nuqtalar": rec.get("n_background"), "Blok (m)": _num(rec.get("block_size"))}
        for k, v in (rec.get("auc") or {}).items():
            d[f"AUC: {k}"] = _num(v)
        drows.append(d)
    return summ, (pd.DataFrame(drows, dtype=object) if drows else None)


def fold_table_frame(result):
    """Random va spatial fold jadvallari bitta DataFrame'da (birinchi ustun 'CV'); yo'q bo'lsa None."""
    parts = []
    if isinstance(result, dict):
        for key in ("random", "spatial"):
            blk = result.get(key)
            ft = blk.get("fold_table") if isinstance(blk, dict) else None
            if isinstance(ft, pd.DataFrame) and len(ft):
                t = ft.copy()
                t.insert(0, "CV", blk.get("mode") or key)
                parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else None


def _block(result, key):
    b = result.get(key) if isinstance(result, dict) else None
    return b if isinstance(b, dict) else None


def _cfg(result, *keys, default=None):
    cur = result.get("cfg") if isinstance(result, dict) else None
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def _class_labels(n):
    try:
        from ..predict import CLASS_LABELS
        return list(CLASS_LABELS(n))
    except Exception:                                        # noqa: BLE001
        return [f"{i}-sinf" for i in range(1, int(n) + 1)]


def _n_classes_of(class_map, breaks):
    """Sinflar soni: chegaralardan (len+1), bo'lmasa class_map ning eng katta qiymatidan; hammasi NaN/yaroqsiz => 0
    (np.nanmax ogohlantirishi/istisnosiz)."""
    if breaks is not None and len(breaks):
        return len(breaks) + 1
    try:
        a = np.asarray(class_map, dtype=float)
        a = a[np.isfinite(a)]
        return int(a.max()) if a.size else 0
    except (TypeError, ValueError):
        return 0


def _disp_name(name):
    return "Ansambl" if str(name) == ENSEMBLE_NAME else str(name)


def _new_fig(figsize=(8.0, 6.0)):
    return Figure(figsize=figsize, layout="constrained")


# ---------------------------------------------------------------------------
# Vidjet yordamchilari
# ---------------------------------------------------------------------------
def _canvas(min_w=640, min_h=460, figsize=(6.4, 4.8)):
    c = MplCanvas(figsize=figsize)
    c.canvas.setMinimumSize(int(min_w), int(min_h))
    c.setMinimumSize(int(min_w), int(min_h) + 36)
    return c


def _blank(c, msg=EMPTY_TEXT, detail=None):
    try:
        plots.no_data(c.fig, msg, detail)
    except Exception:                                        # noqa: BLE001
        pass
    c.redraw()


def _draw(c, fn, *args, **kw):
    """fn(c.fig, *args, **kw) + redraw; istisno bo'lsa figuraga yoziladi (slot qulamaydi)."""
    try:
        fn(c.fig, *args, **kw)
    except Exception as exc:                                 # noqa: BLE001
        _log.warning("%s xato: %s: %s", getattr(fn, "__name__", fn), type(exc).__name__, exc, exc_info=True)
        try:
            plots.no_data(c.fig, detail=f"Chizishda xato: {type(exc).__name__}: {exc}")
        except Exception:                                    # noqa: BLE001
            pass
    c.redraw()


def _table(min_h=170):
    t = DataFrameTable()
    t.setMinimumHeight(int(min_h))
    return t


def _note(text="", wrap=True):
    lb = QLabel(text)
    lb.setWordWrap(wrap)
    lb.setStyleSheet(_GRAY)
    lb.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lb


def _title(text):
    lb = QLabel(text)
    f = lb.font()
    f.setBold(True)
    lb.setFont(f)
    return lb


def _page(*widgets):
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(4, 4, 4, 4)
    for x in widgets:
        lay.addWidget(x, 1 if isinstance(x, (DataFrameTable, MplCanvas, QListWidget)) else 0)
    return w


_KEEP = object()


def _fill_combo(combo, items, current=_KEEP):
    """items: [(matn, userData)]; signal'lar bloklanadi; avvalgi tanlov (userData) saqlanadi (yoki `current`
    berilsa shu), topilmasa birinchisi."""
    prev = combo.currentData() if current is _KEEP else current
    combo.blockSignals(True)
    try:
        combo.clear()
        for text, data in items:
            combo.addItem(text, data)
        idx = 0
        for i, (_t, d) in enumerate(items):
            if d == prev:
                idx = i
                break
        if items:
            combo.setCurrentIndex(idx)
    finally:
        combo.blockSignals(False)
    combo.setEnabled(bool(items))


def _slot(fn):
    """Slot/ochiq metod himoyasi: istisno PyQt5 abort'iga olib bormaydi (BUG-01), xato tab'da ko'rsatiladi."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as exc:                             # noqa: BLE001
            _log.exception("%s.%s xatosi", type(self).__name__, fn.__name__)
            try:
                self._fail(f"{fn.__name__}: {type(exc).__name__}: {exc}")
            except Exception:                                # noqa: BLE001
                pass
            return None
    return wrapper


class _TabBase(QWidget):
    """Umumiy asos: yuqorida doimiy boshqaruv (header), pastda stack: [Hali natija yo'q | QScrollArea(body)]."""

    message = pyqtSignal(str)

    def __init__(self, parent=None, hint=""):
        super().__init__(parent)
        self._busy = False
        self._has_result = False
        self._result = None
        self._errors = []
        self.last_error = None
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        self.header = QVBoxLayout()
        root.addLayout(self.header)
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(_ERR)
        self.error_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.error_label.hide()
        root.addWidget(self.error_label)
        # bo'sh holat
        self.empty_label = QLabel(EMPTY_TEXT)
        f = self.empty_label.font()
        f.setPointSize(f.pointSize() + 3)
        self.empty_label.setFont(f)
        self.empty_label.setStyleSheet(_GRAY)
        self.hint_label = QLabel(hint)
        self.hint_label.setWordWrap(True)
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setStyleSheet(_GRAY)
        ph = QWidget()
        pl = QVBoxLayout(ph)
        pl.addStretch(1)
        pl.addWidget(self.empty_label, 0, Qt.AlignHCenter)
        pl.addWidget(self.hint_label, 0, Qt.AlignHCenter)
        pl.addStretch(2)
        # natija (scroll ichida)
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setWidget(self.body)
        self.stack = QStackedWidget()
        self.stack.addWidget(ph)
        self.stack.addWidget(self.scroll)
        root.addWidget(self.stack, 1)

    # ---- holat
    def has_result(self):
        return bool(self._has_result)

    def is_busy(self):
        return bool(self._busy)

    def _set_has_result(self, flag):
        self._has_result = bool(flag)
        self.stack.setCurrentIndex(1 if flag else 0)
        self._refresh_buttons()

    def set_busy(self, busy):
        """Hisoblash ketayotganda tugmalar o'chadi (tiklash: set_busy(False))."""
        self._busy = bool(busy)
        self._refresh_buttons()

    def _refresh_buttons(self):
        """Voris sinflar tugmalar holatini shu yerda yangilaydi."""

    # ---- xatolar
    def _begin(self):
        self._errors = []
        self.last_error = None
        self.error_label.setText("")
        self.error_label.hide()

    def _step(self, name, fn, *args, **kwargs):
        """Bitta qismni xavfsiz bajaradi: istisno yig'iladi, qolgan qismlar davom etadi."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:                             # noqa: BLE001
            _log.exception("%s: '%s' qismida xato", type(self).__name__, name)
            self._errors.append(f"{name}: {type(exc).__name__}: {exc}")
            return None

    def _end(self):
        if self._errors:
            self._fail("; ".join(self._errors), prefix="Natijani ko'rsatishda xato: ")
            self._errors = []

    def _fail(self, text, prefix="Xato: "):
        self.last_error = str(text)
        self.error_label.setText(prefix + str(text))
        self.error_label.show()
        self.message.emit(prefix + str(text))

    def _info(self, text):
        self.message.emit(str(text))

    def figures(self):
        return {}


# ---------------------------------------------------------------------------
# 3-tab: Ma'lumotlar tahlili
# ---------------------------------------------------------------------------
class DiagnosticsTab(_TabBase):
    """layer_stats, korrelyatsiya heatmap, VIF, yuqori korrelyatsiyali juftliklar, data dictionary (+ musbat/fon farqi)."""

    refresh_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent, hint="O'qitish tugagach avtomatik to'ldiriladi; yoki «Tahlilni hisoblash» tugmasini "
                                      "bosing (TIFF qatlamlari qayta o'qiladi).")
        row = QHBoxLayout()
        self.btn_refresh = QPushButton("Tahlilni hisoblash (TIFF'lardan)")
        self.btn_refresh.setToolTip("TIFF papkasidagi qatlamlar bo'yicha statistika, korrelyatsiya va VIF ni "
                                    "o'qitishsiz qayta hisoblaydi.")
        self.btn_refresh.clicked.connect(lambda _=False: self._on_refresh())
        self.info_label = _note("")
        row.addWidget(self.btn_refresh)
        row.addWidget(self.info_label, 1)
        self.header.addLayout(row)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.hide()
        self.body_layout.addWidget(self.summary_label)
        self.tabs = QTabWidget()
        self.body_layout.addWidget(self.tabs, 1)

        self.stats_table = _table(220)
        self.tabs.addTab(_page(_note("valid_pct - qatlam bo'yicha yaroqli piksellar ulushi (barcha qatlamlar "
                                     "kesishmasi shundan ham kichik bo'lishi mumkin). kind: raqamli/kategorik."),
                               self.stats_table), "Qatlam statistikasi")
        self.corr_canvas = _canvas(620, 540, (6.4, 5.4))
        self.corr_note = _note("")
        self.tabs.addTab(_page(self.corr_note, self.corr_canvas), "Korrelyatsiya")
        self.vif_table = _table(220)
        self.tabs.addTab(_page(_note("VIF >= 10 - multikollinearlik (yuqori). Korrelyatsiyalangan prediktorlarda "
                                     "permutation importance ishonchsiz bo'ladi; o'zgarmas qatlamlar alohida belgilanadi."),
                               self.vif_table), "VIF")
        self.pairs_label = _note("")
        self.pairs_table = _table(180)
        self.tabs.addTab(_page(self.pairs_label, self.pairs_table), "Yuqori korrelyatsiya")
        self.dd_table = _table(220)
        self.tabs.addTab(_page(_note("Data dictionary: texnik metadata avtomatik, manba/sana/transformatsiya - "
                                     "metadata.csv dan. Diqqat: o'qitishda metadata.csv shabloni TIFF papkasiga YOZILADI "
                                     "(yon ta'sir); uni to'ldirib, keyingi ishga tushirishda ishlating - maqolaga "
                                     "chiqarishdan oldin to'liq bo'lishi kerak."), self.dd_table), "Data dictionary")
        self.effects_table = _table(200)
        self.tabs.addTab(_page(_note("Musbat va fon nuqtalar farqi: o'rtachalar, Cohen d va bir o'zgaruvchili AUC "
                                     "(Mann-Whitney). Tavsifiy; model baholash emas."), self.effects_table),
                         "Musbat/fon farqi")
        self._refresh_buttons()

    def _refresh_buttons(self):
        self.btn_refresh.setEnabled(not self._busy)

    @_slot
    def _on_refresh(self):
        self.refresh_requested.emit()

    @staticmethod
    def _color_vif(table):
        col = next((c for c in range(table.columnCount())
                    if table.horizontalHeaderItem(c) is not None and table.horizontalHeaderItem(c).text() == "flag"),
                   None)
        if col is None:
            return
        for r in range(table.rowCount()):
            it = table.item(r, col)
            if it is not None and it.text() in _VIF_COLORS:
                it.setBackground(_VIF_COLORS[it.text()])

    @_slot
    def set_diagnostics(self, diag, data_dictionary=None):
        """diag - data.data_diagnostics natijasi (layer_stats, corr, vif, high_corr_pairs, n_sample, ixtiyoriy
        dataset_summary/feature_effects); data_dictionary - DataFrame. Bo'sh/None => clear()."""
        diag = diag if isinstance(diag, dict) else {}
        dd = data_dictionary if isinstance(data_dictionary, pd.DataFrame) and len(data_dictionary) else None
        if not diag and dd is None:
            self.clear()
            return
        self._begin()
        self._result = diag

        def df(key):
            v = diag.get(key)
            return v if isinstance(v, pd.DataFrame) else None

        self._step("qatlam statistikasi", self.stats_table.set_dataframe, df("layer_stats"))
        self._step("VIF", self.vif_table.set_dataframe, df("vif"), "{:.2f}")
        self._step("VIF rangi", self._color_vif, self.vif_table)
        pairs = df("high_corr_pairs")
        self._step("yuqori korrelyatsiya", self.pairs_table.set_dataframe, pairs, "{:.3f}")
        self.pairs_label.setText(
            "Yuqori korrelyatsiyali (|r| >= 0.9) juftliklar: " + (str(len(pairs)) if pairs is not None else "yo'q")
            + (" - bittasini olib tashlashni o'ylab ko'ring." if pairs is not None and len(pairs) else "."))
        self._step("data dictionary", self.dd_table.set_dataframe, dd)
        fe = df("feature_effects")
        self._step("musbat/fon farqi", self.effects_table.set_dataframe, fe)
        self.tabs.setTabEnabled(5, fe is not None and len(fe) > 0)
        corr = diag.get("corr")
        n_s = diag.get("n_sample")
        self.corr_note.setText("Pearson korrelyatsiya (faqat raqamli qatlamlar)"
                               + (f", {int(n_s):,} piksel namunasi" if n_s else "")
                               + ". Qora ramka: |r| >= 0.9.")
        if corr is None:
            _blank(self.corr_canvas, detail="Korrelyatsiya matritsasi hisoblanmagan.")
        else:
            self._step("korrelyatsiya", _draw, self.corr_canvas, plots.draw_corr_heatmap, corr)
        summ = diag.get("dataset_summary")
        if isinstance(summ, dict):
            txt = (f"Dataset: {summ.get('n', '?')} nuqta ({summ.get('n_pos', '?')} musbat + {summ.get('n_neg', '?')} fon), "
                   f"{summ.get('n_features', '?')} feature; EPV = {_num(summ.get('epv')):.1f}.")
            for w in summ.get("warnings") or []:
                txt += "\nOgohlantirish: " + str(w)
            self.summary_label.setText(txt)
            self.summary_label.show()
        else:
            self.summary_label.hide()
        self.info_label.setText(f"Tahlil: {len(df('layer_stats'))} qatlam" if df("layer_stats") is not None else "")
        self._set_has_result(True)
        self._end()

    @_slot
    def clear(self):
        self._begin()
        self._result = None
        for t in (self.stats_table, self.vif_table, self.pairs_table, self.dd_table, self.effects_table):
            t.set_dataframe(None)
        _blank(self.corr_canvas)
        self.summary_label.hide()
        self.info_label.setText("")
        self._set_has_result(False)

    def figures(self):
        if not self._has_result or not isinstance(self._result, dict) or self._result.get("corr") is None:
            return {}
        return {"corr_heatmap": plots.draw_corr_heatmap(_new_fig((8.0, 7.0)), self._result["corr"])}


# ---------------------------------------------------------------------------
# 4-tab: Natijalar
# ---------------------------------------------------------------------------
_MODES = (("spatial", "Spatial block CV (asosiy)"), ("random", "Random CV (benchmark)"))


class ResultsTab(_TabBase):
    """Metrikalar jadvali + ROC / PR / Kalibrlash / Chalkashlik matritsasi (rejim: spatial yoki random)."""

    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent, hint="Modelni o'qitgandan so'ng metrikalar va grafiklar shu yerda ko'rinadi.")
        self._sources = []                                   # figures() beruvchi boshqa tab'lar (saqlash uchun)
        row = QHBoxLayout()
        row.addWidget(QLabel("CV rejimi:"))
        self.mode_combo = QComboBox()
        for key, text in _MODES:
            self.mode_combo.addItem(text, key)
        self.mode_combo.currentIndexChanged.connect(lambda _i=0: self._on_mode_changed())
        row.addWidget(self.mode_combo)
        row.addStretch(1)
        self.header.addLayout(row)
        row2 = QHBoxLayout()
        self.btn_table = QPushButton("Jadvalni CSV/XLSX ga saqlash")
        self.btn_figs = QPushButton("Barcha grafiklarni saqlash (PNG+PDF)")
        self.btn_export = QPushButton("Natijalarni eksport (CSV/JSON/XLSX)")
        self.btn_table.clicked.connect(lambda _=False: self._on_save_table())
        self.btn_figs.clicked.connect(lambda _=False: self._on_save_figures())
        self.btn_export.clicked.connect(lambda _=False: self._on_export())
        for b in (self.btn_table, self.btn_figs, self.btn_export):
            row2.addWidget(b)
        row2.addStretch(1)
        self.header.addLayout(row2)

        self.note_label = _note("")
        self.body_layout.addWidget(self.note_label)
        self.metrics_table = _table(70)
        self.body_layout.addWidget(self.metrics_table)
        self.plot_tabs = QTabWidget()
        self.body_layout.addWidget(self.plot_tabs, 1)
        self.roc_canvas = _canvas()
        self.pr_canvas = _canvas()
        self.cal_canvas = _canvas()
        self.conf_canvas = _canvas()
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(lambda _i=0: self._on_model_changed())
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Model:"))
        crow.addWidget(self.model_combo)
        crow.addStretch(1)
        conf_page = QWidget()
        cl = QVBoxLayout(conf_page)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.addLayout(crow)
        cl.addWidget(self.conf_canvas, 1)
        self.plot_tabs.addTab(_page(self.roc_canvas), "ROC")
        self.plot_tabs.addTab(_page(self.pr_canvas), "PR")
        self.plot_tabs.addTab(_page(self.cal_canvas), "Kalibrlash")
        self.plot_tabs.addTab(conf_page, "Chalkashlik matritsasi")
        self._refresh_buttons()

    # ---- ochiq yordamchilar
    def register_figure_source(self, source):
        """'Barcha grafiklarni saqlash' ga boshqa tab'larning figures() natijasini ham qo'shadi."""
        if source is not None and source is not self and source not in self._sources:
            self._sources.append(source)

    def current_mode(self):
        return self.mode_combo.currentData() or "spatial"

    def _metrics(self, mode=None):
        blk = _block(self._result, mode or self.current_mode())
        m = blk.get("metrics") if blk else None
        return m if isinstance(m, dict) else {}

    def _refresh_buttons(self):
        on = self._has_result and not self._busy
        self.btn_table.setEnabled(on)
        self.btn_figs.setEnabled(on)
        self.btn_export.setEnabled(on)
        self.mode_combo.setEnabled(self._has_result and not self._busy)

    def _y(self):
        ds = self._result.get("dataset") if isinstance(self._result, dict) else None
        return getattr(ds, "y", None)

    # ---- natija
    @_slot
    def set_result(self, result):
        if not isinstance(result, dict) or not isinstance(result.get("spatial"), dict):
            self.clear()
            return
        self._begin()
        self._result = result
        has_random = isinstance(result.get("random"), dict)
        item = self.mode_combo.model().item(1)
        if item is not None:
            item.setEnabled(has_random)
        if not has_random and self.current_mode() == "random":
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(0)
            self.mode_combo.blockSignals(False)
        self.mode_combo.setToolTip("" if has_random else "Random CV o'chirilgan (sozlamalarda 'Random CV' yoqilmagan).")
        self._set_has_result(True)
        self._refresh_mode()
        self._end()

    @_slot
    def _on_mode_changed(self):
        if not self._has_result:
            return
        self._begin()
        self._refresh_mode()
        self._end()

    def _refresh_mode(self):
        mode = self.current_mode()
        metrics = self._metrics(mode)
        blk = _block(self._result, mode) or {}
        self._step("jadval", self._fill_table, blk, metrics)
        self.note_label.setText(self._step("izoh", self._note_text, mode, metrics) or "")
        label = dict(_MODES)[mode]
        names = list(metrics)
        prev = self.model_combo.currentData()
        cur = prev if prev in names else (ENSEMBLE_NAME if ENSEMBLE_NAME in names else (names[0] if names else None))
        _fill_combo(self.model_combo, [(_disp_name(n), n) for n in names], current=cur)
        self._step("ROC", _draw, self.roc_canvas, plots.draw_roc, metrics, label)
        self._step("PR", _draw, self.pr_canvas, plots.draw_pr, metrics, self._y())
        self._step("kalibrlash", _draw, self.cal_canvas, plots.draw_calibration, metrics)
        self._step("chalkashlik matritsasi", self._draw_confusion)

    def _fill_table(self, blk, metrics):
        df = blk.get("metrics_df")
        if not isinstance(df, pd.DataFrame) or df.empty:
            if metrics:
                from ..cv import metrics_dataframe
                df = metrics_dataframe(metrics)
            else:
                df = None
        self.metrics_table.set_dataframe(df, "{:.3f}")
        n = self.metrics_table.rowCount()                    # jadval balandligi qatorlarga mos (bo'sh joy qolmasin)
        h = self.metrics_table.horizontalHeader().height() + sum(self.metrics_table.rowHeight(i) for i in range(n)) + 8
        self.metrics_table.setFixedHeight(int(min(260, max(70, h))))

    def _note_text(self, mode, metrics):
        parts = [CI_NOTE]
        ci_ok = any(np.all(np.isfinite(np.asarray(m.get("auc_ci95", (np.nan, np.nan)), dtype=float)))
                    for m in metrics.values() if isinstance(m, dict))
        if metrics and not ci_ok:
            parts.append("95% CI hisoblanmagan (n_bootstrap = 0 yoki bootstrap bajarilmadi) - ustunlar bo'sh.")
        if mode == "spatial":
            tun = _cfg(self._result, "tuning", default={}) or {}
            if tun.get("enabled"):
                if tun.get("mode") == "nested":
                    parts.append("Spatial CV nested tuning bilan baholangan (har tashqi fold ichida giperparametr "
                                 "qidirilgan); yakuniy model giperparametrlari esa butun ma'lumotda qidirilgan.")
                else:
                    parts.append("Tuning rejimi 'final': spatial CV bazaviy giperparametrlar bilan baholangan "
                                 "(tuning faqat yakuniy modelda). Nested rejim tavsiya etiladi.")
            parts.append("Spatial blok CV - asosiy, fazoviy avtokorrelyatsiyani hisobga oluvchi baho.")
        else:
            parts.append("Random CV - faqat benchmark: fazoviy avtokorrelyatsiya tufayli optimistik; bazaviy "
                         "giperparametrlar bilan, tuning va importance yo'q.")
        if "CNN" in metrics:
            parts.append(CNN_NOTE)
        return " ".join(parts)

    @_slot
    def _on_model_changed(self):
        if self._has_result:
            self._step("chalkashlik matritsasi", self._draw_confusion)
            self._end()

    def _draw_confusion(self):
        metrics = self._metrics()
        _draw(self.conf_canvas, plots.draw_confusion, metrics, self.model_combo.currentData())

    @_slot
    def clear(self):
        self._begin()
        self._result = None
        self.metrics_table.set_dataframe(None)
        self.note_label.setText("")
        _fill_combo(self.model_combo, [])
        for c in (self.roc_canvas, self.pr_canvas, self.cal_canvas, self.conf_canvas):
            _blank(c)
        self._set_has_result(False)

    # ---- grafiklar / saqlash
    def figures(self):
        """{nom: Figure}: har rejim uchun ROC/PR/Kalibrlash, spatial uchun har model chalkashlik matritsasi
        (random uchun faqat ROC/PR/Kalibrlash). Yangi (ekrandan mustaqil) Figure'lar."""
        if not self._has_result:
            return {}
        out = {}
        y = self._y()
        for mode, label in _MODES:
            metrics = self._metrics(mode)
            if not metrics:
                continue
            out[f"roc_{mode}"] = plots.draw_roc(_new_fig((8.0, 6.0)), metrics, label)
            out[f"pr_{mode}"] = plots.draw_pr(_new_fig((8.0, 6.0)), metrics, y)
            out[f"calibration_{mode}"] = plots.draw_calibration(_new_fig((8.0, 6.0)), metrics)
            if mode == "spatial":
                for name in metrics:
                    out[f"confusion_{name}_{mode}"] = plots.draw_confusion(_new_fig((6.0, 5.5)), metrics, name)
        return out

    def _all_figures(self):
        figs = dict(self.figures())
        for src in self._sources:
            try:
                for k, v in (src.figures() or {}).items():
                    figs.setdefault(k, v)
            except Exception as exc:                         # noqa: BLE001
                _log.warning("figures() xato (%s): %s", type(src).__name__, exc, exc_info=True)
        return figs

    def save_table(self, path=None):
        """Joriy jadvalni CSV yoki XLSX ga saqlaydi (path=None => dialog). .xlsx da mavjud rejimlar varaqlarga
        yoziladi. Yozilgan yo'l yoki None qaytaradi."""
        df = self.metrics_table.dataframe()
        if df is None:
            return None
        if path is None:
            path, flt = QFileDialog.getSaveFileName(self, "Metrikalar jadvalini saqlash", "metrics.csv",
                                                    "CSV (*.csv);;Excel (*.xlsx)")
            if not path:
                return None
            if not os.path.splitext(path)[1]:
                path += ".xlsx" if "xlsx" in (flt or "") else ".csv"
        ext = os.path.splitext(str(path))[1].lower()
        if ext == ".xlsx":
            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                wrote = False
                for mode, label in _MODES:
                    blk = _block(self._result, mode)
                    d = blk.get("metrics_df") if blk else None
                    if isinstance(d, pd.DataFrame) and len(d):
                        d.to_excel(xw, sheet_name=label.split(" (")[0][:31], index=False)
                        wrote = True
                if not wrote:
                    df.to_excel(xw, sheet_name="Metrikalar", index=False)
        else:
            df.to_csv(path, index=False, encoding="utf-8-sig")
        self._info(f"Jadval saqlandi: {path}")
        return path

    def save_figures(self, out_dir=None, formats=("png", "pdf"), dpi=300):
        """Barcha grafiklarni (shu tab + ro'yxatdan o'tgan tab'lar) papkaga saqlaydi. Yo'llar ro'yxati."""
        if not self._has_result:
            return []
        if out_dir is None:
            out_dir = QFileDialog.getExistingDirectory(self, "Grafiklar saqlanadigan papka", "")
            if not out_dir:
                return []
        paths = plots.save_all_figures(self._all_figures(), out_dir, formats=formats, dpi=dpi, log_fn=self._info)
        self._info(f"{len(paths)} ta grafik fayli saqlandi: {out_dir}")
        return paths

    @_slot
    def _on_save_table(self):
        self.save_table()

    @_slot
    def _on_save_figures(self):
        self.save_figures()

    @_slot
    def _on_export(self):
        self.export_requested.emit()


# ---------------------------------------------------------------------------
# 5-tab: Spatial CV diagnostika
# ---------------------------------------------------------------------------
class SpatialTab(_TabBase):
    """Random vs spatial AUC + fold xaritasi, fold jadvali, fon sezgirligi, nested tuning, ogohlantirishlar."""

    def __init__(self, parent=None):
        super().__init__(parent, hint="Modelni o'qitgandan so'ng spatial CV diagnostikasi shu yerda ko'rinadi.")
        self.note_label = _note("")
        self.body_layout.addWidget(self.note_label)
        self.diag_canvas = _canvas(900, 460, (10.0, 4.8))
        self.body_layout.addWidget(self.diag_canvas)
        self.tabs = QTabWidget()
        self.body_layout.addWidget(self.tabs, 1)

        self.fold_note = _note("")
        self.fold_table = _table(200)
        self.tabs.addTab(_page(self.fold_note, self.fold_table), "Fold jadvali")

        self.bg_label = _note("")
        self.bg_table = _table(130)
        self.bg_draw_table = _table(130)
        self.tabs.addTab(_page(self.bg_label, self.bg_table, self.bg_draw_table), "Fon sezgirligi")

        self.tuning_note = _note("")
        self.tuning_table = _table(160)
        self.tuning_canvas = _canvas(860, 520, (9.0, 5.4))
        self.tabs.addTab(_page(self.tuning_note, self.tuning_table, self.tuning_canvas), "Tuning (nested)")

        self.warn_label = _note("")
        self.warn_list = QListWidget()
        self.warn_list.setWordWrap(True)
        self.warn_list.setTextElideMode(Qt.ElideNone)
        self.warn_list.setMinimumHeight(180)
        self.tabs.addTab(_page(self.warn_label, self.warn_list), "Ogohlantirishlar")

    @_slot
    def set_result(self, result):
        if not isinstance(result, dict) or not isinstance(result.get("spatial"), dict):
            self.clear()
            return
        self._begin()
        self._result = result
        sp = result["spatial"]
        bs = _num(result.get("block_size"))
        self.note_label.setText(
            "Spatial blok CV: bitta blok HECH QACHON train va validation orasida bo'linmaydi"
            + (f" (blok o'lchami {bs:,.0f} m)" if math.isfinite(bs) and bs > 0 else "")
            + ". Random - spatial AUC farqi avtokorrelyatsiya tufayli optimizm (leakage) taxminini beradi.")
        self._step("diagnostika grafigi", _draw, self.diag_canvas, plots.draw_spatial_diagnostics, result)
        # fold jadvali
        ft = fold_table_frame(result)
        self._step("fold jadvali", self.fold_table.set_dataframe, ft)
        note = "n_blocks_val - validation fold'dagi bloklar soni; pos_val - validation'dagi musbat nuqtalar."
        if ft is not None and "pos_val" in ft.columns and bool((pd.to_numeric(ft["pos_val"], errors="coerce") == 0).any()):
            note += " DIQQAT: musbat nuqtasiz validation fold bor (AUC ishonchsiz)."
        self.fold_note.setText(note)
        # fon sezgirligi
        summ, per_draw = bg_sensitivity_frames(result.get("bg_sensitivity"))
        self._step("fon sezgirligi", self.bg_table.set_dataframe, summ)
        self._step("fon sezgirligi (har tanlov)", self.bg_draw_table.set_dataframe, per_draw)
        bg = result.get("bg_sensitivity")
        if summ is None:
            self.bg_label.setText("Fon sezgirligi hisoblanmagan (sozlamalarda o'chirilgan yoki yetarli tanlov yo'q).")
        else:
            self.bg_label.setText(f"Fon nuqtalar {bg.get('n_draws', len(bg.get('per_draw') or []))} marta mustaqil "
                                  "qayta tanlandi (bazaviy giperparametrlar, spatial CV). Kichik std - natija fon "
                                  "tanloviga barqaror.")
        # tuning
        tp = sp.get("tuned_params") if isinstance(sp.get("tuned_params"), dict) else {}
        tdf = tuning_dataframe(result)
        self._step("tuning jadvali", self.tuning_table.set_dataframe, tdf if len(tdf) else None)
        self._step("tuning grafigi", _draw, self.tuning_canvas, plots.draw_tuning_trials, tp)
        self.tuning_note.setText(
            ("Nested tuning: har tashqi fold'ning train qismida giperparametr qidirilgan, baho shu fold validation'ida. "
             if tp else "Nested tuning bajarilmagan (o'chirilgan). ")
            + "Yakuniy tuning HAR DOIM butun ma'lumotda bajariladi; 'final' rejimida spatial CV bazaviy "
              "giperparametrlar bilan baholanadi (tavsiya: nested).")
        # ogohlantirishlar
        warns = [str(w) for w in (result.get("warnings") or [])]
        self.warn_list.clear()
        for w in warns:
            self.warn_list.addItem(w)
        self.warn_list.setVisible(bool(warns))
        self.warn_label.setText(f"{len(warns)} ta ogohlantirish." if warns else "Ogohlantirish yo'q.")
        self.tabs.setTabText(3, f"Ogohlantirishlar ({len(warns)})")
        self._set_has_result(True)
        self._end()

    @_slot
    def clear(self):
        self._begin()
        self._result = None
        for t in (self.fold_table, self.bg_table, self.bg_draw_table, self.tuning_table):
            t.set_dataframe(None)
        self.warn_list.clear()
        self.tabs.setTabText(3, "Ogohlantirishlar")
        _blank(self.diag_canvas)
        _blank(self.tuning_canvas)
        self._set_has_result(False)

    def figures(self):
        if not self._has_result or not isinstance(self._result, dict):
            return {}
        out = {"spatial_cv_diagnostics": plots.draw_spatial_diagnostics(_new_fig((12.0, 5.5)), self._result)}
        tp = (_block(self._result, "spatial") or {}).get("tuned_params")
        if isinstance(tp, dict) and tp:
            out["tuning_trials"] = plots.draw_tuning_trials(_new_fig((11.0, 7.0)), tp)
        return out


# ---------------------------------------------------------------------------
# 6-tab: Feature importance
# ---------------------------------------------------------------------------
def _importance_models(result):
    """[(model, source)] - draw_importance birinchi qatorida chiziladigan (chekli o'rtachali) modellar, shu tartibda."""
    imp = result.get("importance") if isinstance(result, dict) and isinstance(result.get("importance"), dict) else {}
    models = imp.get("models") if isinstance(imp.get("models"), dict) else None
    if models is None:
        sp = _block(result, "spatial") or {}
        models = sp.get("perm_importance") if isinstance(sp.get("perm_importance"), dict) else {}
    out = []
    for k, v in models.items():
        if not isinstance(v, dict):
            continue
        try:
            mean = np.asarray(v.get("mean"), dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if mean.size == 0 or not np.isfinite(mean).any():
            continue
        out.append((str(k), v.get("source") or "permutation"))
    return out


def _relabel_mdi(fig, result):
    """draw_importance barcha model panellarini 'Permutation importance' deb yozadi. MDI zaxirasi (source='mdi')
    bo'lgan modellarda sarlavha/o'q yorlig'i to'g'rilanadi va (ma'nosiz nol) xato chiziqlari olib tashlanadi."""
    keys = _importance_models(result)
    axes = list(fig.axes)
    if len(axes) < len(keys):
        return
    for i, (name, src) in enumerate(keys):
        if src != "mdi":
            continue
        ax = axes[i]
        old = ax.get_title()
        pos = old.find("(eng muhim")
        suffix = ("\n" + old[pos:]) if pos >= 0 else ""
        ax.set_title(f"{name}\nMDI/gain importance (zaxira){suffix}", fontsize=9)
        ax.set_xlabel("MDI/gain (model ichki o'lchovi; xato chizig'i yo'q)", fontsize=8)
        for cont in list(ax.containers):
            eb = getattr(cont, "errorbar", None)
            if eb is None:
                continue
            for part in eb.lines:
                for art in (part if isinstance(part, (tuple, list)) else [part]):
                    try:
                        if art is not None:
                            art.remove()
                    except Exception:                        # noqa: BLE001
                        pass


def _draw_importance(fig, result):
    plots.draw_importance(fig, result)
    try:
        _relabel_mdi(fig, result)
    except Exception:                                        # noqa: BLE001 - chiroyli yorliq xatosi grafikni buzmasin
        _log.warning("MDI yorlig'ini almashtirib bo'lmadi", exc_info=True)
    return fig


class ImportanceTab(_TabBase):
    """Permutation + SHAP bar, SHAP beeswarm (model), SHAP dependence (model + feature), importance jadvali."""

    def __init__(self, parent=None):
        super().__init__(parent, hint="Modelni o'qitgandan so'ng feature importance shu yerda ko'rinadi.")
        self.method_label = _title("")
        self.method_label.setWordWrap(True)
        self.units_label = _note("")
        self.body_layout.addWidget(self.method_label)
        self.body_layout.addWidget(self.units_label)
        self.tabs = QTabWidget()
        self.body_layout.addWidget(self.tabs, 1)
        self.imp_canvas = _canvas(900, 620, (9.0, 6.4))
        self.tabs.addTab(_page(self.imp_canvas), "Importance (perm + SHAP)")
        # beeswarm
        self.bee_combo = QComboBox()
        self.bee_combo.currentIndexChanged.connect(lambda _i=0: self._on_bee_changed())
        self.bee_canvas = _canvas(700, 520, (7.0, 5.2))
        brow = QHBoxLayout()
        brow.addWidget(QLabel("Model:"))
        brow.addWidget(self.bee_combo)
        brow.addStretch(1)
        bee_page = QWidget()
        bl = QVBoxLayout(bee_page)
        bl.setContentsMargins(4, 4, 4, 4)
        bl.addLayout(brow)
        bl.addWidget(self.bee_canvas, 1)
        self.tabs.addTab(bee_page, "SHAP beeswarm")
        # dependence
        self.dep_model_combo = QComboBox()
        self.dep_feature_combo = QComboBox()
        self.dep_model_combo.currentIndexChanged.connect(lambda _i=0: self._on_dep_model_changed())
        self.dep_feature_combo.currentIndexChanged.connect(lambda _i=0: self._on_dep_feature_changed())
        self.dep_canvas = _canvas(700, 520, (7.0, 5.2))
        drow = QHBoxLayout()
        drow.addWidget(QLabel("Model:"))
        drow.addWidget(self.dep_model_combo)
        drow.addWidget(QLabel("Feature:"))
        drow.addWidget(self.dep_feature_combo, 1)
        dep_page = QWidget()
        dl = QVBoxLayout(dep_page)
        dl.setContentsMargins(4, 4, 4, 4)
        dl.addLayout(drow)
        dl.addWidget(self.dep_canvas, 1)
        self.tabs.addTab(dep_page, "SHAP dependence")
        # jadval
        self.table_note = _note("")
        self.imp_table = _table(260)
        self.tabs.addTab(_page(self.table_note, self.imp_table), "Jadval")

    # ---- yordamchilar
    def _shap_models(self):
        sh = self._result.get("shap") if isinstance(self._result, dict) else None
        return [str(k) for k, v in sh.items() if isinstance(v, dict)] if isinstance(sh, dict) else []

    def _feature_names(self, model):
        r = self._result if isinstance(self._result, dict) else {}
        sh = r.get("shap") if isinstance(r.get("shap"), dict) else {}
        entry = sh.get(model) if isinstance(sh.get(model), dict) else {}
        names = entry.get("feature_names")
        if not names:
            imp = r.get("importance") if isinstance(r.get("importance"), dict) else {}
            names = imp.get("feature_names") or r.get("feature_names")
        return [str(n) for n in (names or [])]

    def _units_text(self):
        parts = []
        mdi = [n for n, s in _importance_models(self._result) if s == "mdi"]
        if mdi:
            parts.append(f"{', '.join(mdi)} uchun permutation importance mavjud emas: MDI/gain zaxirasi ko'rsatilmoqda "
                         "(korrelyatsiyalangan prediktorlarda noto'g'ri bo'lishi mumkin; xato chiziqlari yo'q).")
        if any(s == "permutation" for _n, s in _importance_models(self._result)):
            parts.append("Permutation importance - OOF AUC pasayishi (spatial CV, 1-takror); one-hot kategorik "
                         "ustunlar alohida aralashtiriladi.")
        if self._shap_models():
            parts.append("SHAP birliklari modelga bog'liq: RandomForest - ehtimollik, XGBoost - log-odds; "
                         "modellararo magnitudani to'g'ridan-to'g'ri solishtirmang. SHAP fon nuqtalar bo'yicha hisoblangan.")
        else:
            parts.append("SHAP hisoblanmagan (shap o'rnatilmagan, o'chirilgan yoki faqat RF/XGBoost uchun mavjud).")
        if not _importance_models(self._result):
            parts.append("Importance qiymatlari yo'q (permutation importance o'chirilgan yoki faqat SVM tanlangan).")
        return " ".join(parts)

    # ---- natija
    @_slot
    def set_result(self, result):
        if not isinstance(result, dict) or (not isinstance(result.get("importance"), dict)
                                            and not isinstance(result.get("shap"), dict)):
            self.clear()
            return
        self._begin()
        self._result = result
        imp = result.get("importance") if isinstance(result.get("importance"), dict) else {}
        self.method_label.setText(f"Importance usuli: {imp.get('method') or 'hisoblanmadi'}")
        self.units_label.setText(self._units_text())
        self._step("importance grafigi", _draw, self.imp_canvas, _draw_importance, result)
        shap_models = self._shap_models()
        items = [(_disp_name(m), m) for m in shap_models]
        _fill_combo(self.bee_combo, items)
        _fill_combo(self.dep_model_combo, items)
        self._step("SHAP beeswarm", self._draw_bee)
        self._fill_features()
        self._step("SHAP dependence", self._draw_dep)
        df = importance_dataframe(result)
        self._step("jadval", self.imp_table.set_dataframe, df if len(df) else None, "{:.4f}")
        self.table_note.setText("Ustun sarlavhasidagi [manba]: permutation (OOF, std - fold'lar bo'yicha) yoki mdi "
                                "(zaxira; std yo'q). 'AUC' kamayishi katta bo'lsa feature muhimroq.")
        self._set_has_result(True)
        self._end()

    def _fill_features(self):
        model = self.dep_model_combo.currentData()
        items = [("(eng muhimi - avtomatik)", None)] + [(n, n) for n in self._feature_names(model)] if model else []
        prev = self.dep_feature_combo.currentData()
        _fill_combo(self.dep_feature_combo, items, current=prev)
        self.dep_feature_combo.setEnabled(bool(model))

    def _draw_bee(self):
        _draw(self.bee_canvas, plots.draw_shap_beeswarm, self._result, self.bee_combo.currentData())

    def _draw_dep(self):
        _draw(self.dep_canvas, plots.draw_shap_dependence, self._result, self.dep_model_combo.currentData(),
              self.dep_feature_combo.currentData())

    @_slot
    def _on_bee_changed(self):
        if self._has_result:
            self._draw_bee()

    @_slot
    def _on_dep_model_changed(self):
        if self._has_result:
            self._fill_features()
            self._draw_dep()

    @_slot
    def _on_dep_feature_changed(self):
        if self._has_result:
            self._draw_dep()

    @_slot
    def clear(self):
        self._begin()
        self._result = None
        self.method_label.setText("")
        self.units_label.setText("")
        for c in (self.bee_combo, self.dep_model_combo, self.dep_feature_combo):
            _fill_combo(c, [])
        self.imp_table.set_dataframe(None)
        for c in (self.imp_canvas, self.bee_canvas, self.dep_canvas):
            _blank(c)
        self._set_has_result(False)

    def figures(self):
        """importance; har SHAP modeli uchun beeswarm va dependence (eng muhim feature) - yangi Figure'lar."""
        if not self._has_result or not isinstance(self._result, dict):
            return {}
        out = {}
        if _importance_models(self._result) or self._shap_models():      # bo'sh "Ma'lumot yo'q" rasm saqlanmasin
            out["importance"] = _draw_importance(_new_fig((11.0, 7.5)), self._result)
        for m in self._shap_models():
            out[f"shap_beeswarm_{m}"] = plots.draw_shap_beeswarm(_new_fig((8.0, 6.5)), self._result, m)
            out[f"shap_dependence_{m}"] = plots.draw_shap_dependence(_new_fig((8.0, 6.0)), self._result, m, None)
        return out


# ---------------------------------------------------------------------------
# 7-tab: Prognoz xarita
# ---------------------------------------------------------------------------
_CLASS_METHOD_TEXT = (("quantile", "Kvantil (teng maydonli sinflar)"), ("equal_interval", "Teng intervalli (0-1)"),
                      ("fixed", "Qat'iy chegaralar"))


class MapTab(_TabBase):
    """Prognoz xaritasi: chiqish papkasi, sinflash sozlamalari, xarita (overlay), success-rate, sinf statistikasi."""

    predict_requested = pyqtSignal(dict)
    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent, hint="Avval modelni o'qiting.")
        self._pred = None
        self._train_aoi = self._train_pos = self._train_bg = None
        self._train_transform = None
        # --- boshqaruv
        self.out_picker = FolderPicker("Chiqish papkasi (GeoTIFF, jadvallar):", label_width=210)
        self.header.addWidget(self.out_picker)
        box = QGroupBox("Sinflash sozlamalari")
        form = QHBoxLayout(box)
        self.method_combo = QComboBox()
        for key, text in _CLASS_METHOD_TEXT:
            self.method_combo.addItem(text, key)
        self.n_classes_spin = QSpinBox()
        self.n_classes_spin.setRange(2, _MAX_CLASSES_GUI)
        self.n_classes_spin.setValue(5)
        self.breaks_edit = QLineEdit("0.2, 0.4, 0.6, 0.8")
        self.breaks_edit.setPlaceholderText("0.2, 0.4, 0.6, 0.8  (vergul bilan, o'suvchi, 0..1 ichida)")
        form.addWidget(QLabel("Usul:"))
        form.addWidget(self.method_combo)
        form.addWidget(QLabel("Sinflar soni:"))
        form.addWidget(self.n_classes_spin)
        form.addWidget(QLabel("Qat'iy chegaralar:"))
        form.addWidget(self.breaks_edit, 1)
        self.header.addWidget(box)
        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet(_ERR)
        self.validation_label.hide()
        self.header.addWidget(self.validation_label)
        brow = QHBoxLayout()
        self.btn_predict = QPushButton("Prognoz xarita yaratish")
        self.btn_export = QPushButton("Natijalarni eksport (CSV/JSON/XLSX)")
        brow.addWidget(self.btn_predict)
        brow.addWidget(self.btn_export)
        brow.addStretch(1)
        self.header.addLayout(brow)
        self.index_note = _note(INDEX_NOTE + ".")
        self.header.addWidget(self.index_note)
        self.cnn_note = _note(CNN_NOTE)
        self.cnn_note.hide()
        self.header.addWidget(self.cnn_note)
        self.method_combo.currentIndexChanged.connect(lambda _i=0: self._update_method_widgets())
        self.out_picker.changed.connect(lambda _t="": self._hide_validation())
        self.breaks_edit.textChanged.connect(lambda _t="": self._hide_validation())
        self.btn_predict.clicked.connect(lambda _=False: self._on_predict())
        self.btn_export.clicked.connect(lambda _=False: self._on_export())
        # --- natija
        self.tabs = QTabWidget()
        self.body_layout.addWidget(self.tabs, 1)
        self.layer_combo = QComboBox()
        self.layer_combo.currentIndexChanged.connect(lambda _i=0: self._on_layer_changed())
        self.cb_aoi = QCheckBox("AOI chegarasi")
        self.cb_pos = QCheckBox("Musbat nuqtalar (konlar)")
        self.cb_bg = QCheckBox("Fon nuqtalar")
        for cb in (self.cb_aoi, self.cb_pos, self.cb_bg):
            cb.setChecked(cb is self.cb_aoi)
            cb.toggled.connect(lambda _c=False: self._on_layer_changed())
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Qatlam:"))
        mrow.addWidget(self.layer_combo, 1)
        for cb in (self.cb_aoi, self.cb_pos, self.cb_bg):
            mrow.addWidget(cb)
        self.map_canvas = _canvas(760, 560, (7.6, 5.6))
        map_page = QWidget()
        ml = QVBoxLayout(map_page)
        ml.setContentsMargins(4, 4, 4, 4)
        ml.addLayout(mrow)
        ml.addWidget(self.map_canvas, 1)
        self.tabs.addTab(map_page, "Xarita")
        self.success_canvas = _canvas()
        self.success_note = _note("Success-rate o'qitish (musbat) nuqtalarida hisoblangan - optimistik, mustaqil "
                                  "validatsiya emas (mustaqil baho: spatial CV).")
        self.tabs.addTab(_page(self.success_note, self.success_canvas), "Success-rate")
        self.stats_note = _note("")
        self.stats_table = _table(200)
        self.tabs.addTab(_page(self.stats_note, self.stats_table), "Sinf statistikasi")
        self._update_method_widgets()
        self._refresh_buttons()

    # ---- sozlamalar
    def _update_method_widgets(self):
        fixed = self.method_combo.currentData() == "fixed"
        self.n_classes_spin.setEnabled(not fixed and not self._busy)
        self.breaks_edit.setEnabled(fixed and not self._busy)

    def _hide_validation(self):
        self.validation_label.hide()

    def collect_settings(self):
        """(sozlamalar lug'ati, None) yoki (None, xato matni). Lug'at: out_dir, class_method, n_classes, class_breaks."""
        method = self.method_combo.currentData() or "quantile"
        out_dir = self.out_picker.path()
        if out_dir and os.path.exists(out_dir) and not os.path.isdir(out_dir):
            return None, f"Chiqish yo'li papka emas: {out_dir}"
        if method == "fixed":
            try:
                breaks = parse_breaks(self.breaks_edit.text())
            except ValueError as e:
                return None, str(e)
            return {"out_dir": out_dir, "class_method": method, "n_classes": len(breaks) + 1,
                    "class_breaks": breaks}, None
        return {"out_dir": out_dir, "class_method": method, "n_classes": int(self.n_classes_spin.value()),
                "class_breaks": None}, None

    @_slot
    def _on_predict(self):
        settings, err = self.collect_settings()
        if err:
            self.validation_label.setText(err)
            self.validation_label.show()
            return
        self.validation_label.hide()
        self.predict_requested.emit(settings)

    @_slot
    def _on_export(self):
        self.export_requested.emit()

    def _refresh_buttons(self):
        has_train = self._result is not None
        self.btn_predict.setEnabled(has_train and not self._busy)
        self.btn_export.setEnabled(has_train and not self._busy)
        self.out_picker.setEnabled(not self._busy)
        self.method_combo.setEnabled(not self._busy)
        self._update_method_widgets()

    # ---- o'qitish natijasi (overlay, defaultlar)
    @_slot
    def set_result(self, result):
        """O'qitish natijasi: sinflash defaultlari (cfg), overlay (AOI/nuqtalar/transform). Oldingi prognoz tozalanadi."""
        if not isinstance(result, dict) or not result:
            self.clear()
            return
        self._begin()
        self._clear_prediction()
        self._result = result
        raster = result.get("raster")
        self._train_transform = getattr(raster, "transform", None) or result.get("transform")
        self._train_aoi, self._train_pos, self._train_bg = (result.get("aoi_gdf"), result.get("positive_gdf"),
                                                            result.get("background_gdf"))
        cfg = result.get("cfg") if isinstance(result.get("cfg"), dict) else {}
        m = cfg.get("class_method")
        i = self.method_combo.findData(m)
        if i >= 0:
            self.method_combo.setCurrentIndex(i)
        try:
            self.n_classes_spin.setValue(int(cfg.get("n_classes", 5)))
        except (TypeError, ValueError):
            pass
        if cfg.get("class_breaks"):
            self.breaks_edit.setText(format_breaks(cfg["class_breaks"]))
        if not self.out_picker.path() and cfg.get("output_dir"):
            self.out_picker.setPath(cfg["output_dir"])
        fm = result.get("final_models")
        self.cnn_note.setVisible(isinstance(fm, dict) and "CNN" in fm and len(fm) > 1)
        self._update_overlay_widgets()
        self.hint_label.setText("Model o'qitilgan. Sinflash sozlamalarini tanlab «Prognoz xarita yaratish» tugmasini "
                                "bosing.")
        self._set_has_result(False)
        self._refresh_buttons()
        self._end()

    def _overlay_ok(self):
        """O'qitish AOI/nuqtalari faqat o'sha to'r (transform) dagi xaritaga mos: bundle boshqa maydonga qo'llanganda
        (prognozning o'z 'transform'i o'qitishnikidan farq qilsa) eski AOI/konlar noto'g'ri ustiga chizilmasin."""
        pt = self._pred.get("transform") if isinstance(self._pred, dict) else None
        if pt is None or self._train_transform is None:
            return True
        try:
            return tuple(round(float(v), 6) for v in tuple(pt)[:6]) == \
                tuple(round(float(v), 6) for v in tuple(self._train_transform)[:6])
        except (TypeError, ValueError):
            return False

    def _update_overlay_widgets(self):
        tr = self._transform()
        ok = self._overlay_ok()
        for cb, g in ((self.cb_aoi, self._train_aoi), (self.cb_pos, self._train_pos), (self.cb_bg, self._train_bg)):
            cb.setEnabled(g is not None and tr is not None and ok)
            cb.setToolTip("" if ok else "Prognoz boshqa to'rda (bundle yangi maydonga qo'llangan): o'qitish "
                                         "nuqtalari/AOI ko'rsatilmaydi.")

    def _transform(self):
        if isinstance(self._pred, dict) and self._pred.get("transform") is not None:
            return self._pred["transform"]
        return self._train_transform

    # ---- prognoz
    @_slot
    def set_prediction(self, pred):
        """PredictionOutput (pipeline.run_prediction) yoki ApplyBundleWorker natijasi ('transform' kaliti bilan)."""
        if not isinstance(pred, dict) or not isinstance(pred.get("maps"), dict) or not pred["maps"]:
            self._clear_prediction()
            self._set_has_result(False)
            return
        self._begin()
        self._pred = pred
        items = []
        names = sorted(pred["maps"], key=lambda n: 0 if n == ENSEMBLE_NAME else 1)
        for n in names:
            items.append((f"{_disp_name(n)} - prospektivlik indeksi", f"map:{n}"))
        if pred.get("uncertainty") is not None:
            items.append(("Noaniqlik (modellar o'rtasida std)", "uncertainty"))
        if pred.get("class_map") is not None:
            items.append(("Sinflangan xarita", "classes"))
        _fill_combo(self.layer_combo, items, current=f"map:{ENSEMBLE_NAME}")
        self._update_overlay_widgets()
        self._set_has_result(True)
        self._step("xarita", self._draw_layer)
        curve = pred.get("success_curve")
        if isinstance(curve, dict) and curve:
            self._step("success-rate", _draw, self.success_canvas, plots.draw_success_rate, curve)
        else:
            _blank(self.success_canvas, detail="Success-rate hisoblanmagan.")
        stats = pred.get("class_stats")
        self._step("sinf statistikasi", self.stats_table.set_dataframe,
                   stats if isinstance(stats, pd.DataFrame) else None,
                   {"Maydon_km2": "{:.2f}", "Maydon_%": "{:.2f}", "Konlar_%": "{:.1f}", "Boyitish": "{:.2f}"})
        br = pred.get("class_breaks")
        self.stats_note.setText(
            f"Sinflash: {pred.get('class_method', '?')}, {pred.get('n_classes', '?')} sinf"
            + (f"; chegaralar (indeks): {format_breaks(br)}" if br else "")
            + ". Konlar_* ustunlari - o'qitish nuqtalari bo'yicha (bo'sh = noma'lum). Boyitish = Konlar_% / Maydon_% "
              "(1 dan katta - tasodifiydan yaxshi). " + INDEX_NOTE + ".")
        self._refresh_buttons()
        self._end()

    def _clear_prediction(self):
        self._pred = None
        _fill_combo(self.layer_combo, [])
        self.stats_table.set_dataframe(None)
        self.stats_note.setText("")
        _blank(self.map_canvas)
        _blank(self.success_canvas)

    @_slot
    def _on_layer_changed(self):
        if self._has_result and self._pred is not None:
            self._draw_layer()

    def _layer_spec(self, key, max_px):
        """Tanlangan qatlam uchun draw_map argumentlari (stride bilan kamaytirilgan)."""
        pred = self._pred
        tr = self._transform()
        kw = dict(cmap="RdYlGn_r", vmin=0.0, vmax=1.0, cbar_label=plots.INDEX_LABEL)
        if key == "uncertainty":
            arr = pred.get("uncertainty")
            kw.update(cmap="viridis", cbar_label="Noaniqlik: modellar o'rtasidagi std (indeks birligida)")
            title = "Noaniqlik xaritasi (modellar bashorati std)"
        elif key == "classes":
            arr = pred.get("class_map")
            stats = pred.get("class_stats")
            br = pred.get("class_breaks")
            n = _n_classes_of(arr, br)
            labels = None
            if isinstance(stats, pd.DataFrame) and "Nomi" in stats.columns and len(stats) == n:
                labels = [str(x) for x in stats["Nomi"]]
            kw.update(class_labels=labels or _class_labels(max(n, 1)), cbar_label="Prospektivlik sinfi")
            title = f"Sinflangan prospektivlik indeksi ({pred.get('class_method', '?')}, {n} sinf)"
        else:
            name = str(key).split(":", 1)[1] if ":" in str(key) else str(key)
            arr = pred["maps"].get(name)
            title = f"Prospektivlik indeksi - {_disp_name(name)}"
        if arr is None:
            return None
        a, tr2 = downsample_map(arr, tr, max_px)
        if key == "uncertainty":
            fin = a[np.isfinite(a)] if a.size else a
            kw["vmax"] = float(np.nanmax(fin)) if fin.size and float(np.nanmax(fin)) > 0 else 1.0
        return {"array": a, "transform": tr2, "title": title, **kw}

    def _overlay_kwargs(self, tr, use_checks=True):
        if tr is None or not self._overlay_ok():
            return {}
        want = (lambda cb: cb.isChecked()) if use_checks else (lambda cb: cb is self.cb_aoi)
        return {"aoi_gdf": self._train_aoi if want(self.cb_aoi) else None,
                "positives": self._train_pos if want(self.cb_pos) else None,
                "background": self._train_bg if want(self.cb_bg) else None}

    def _draw_layer(self):
        key = self.layer_combo.currentData()
        spec = self._layer_spec(key, MAP_MAX_PX_SHOW) if key else None
        if spec is None:
            _blank(self.map_canvas, detail="Tanlangan qatlam yo'q.")
            return
        arr, tr = spec.pop("array"), spec.pop("transform")
        _draw(self.map_canvas, plots.draw_map, arr, tr, **spec, **self._overlay_kwargs(tr))

    @_slot
    def clear(self):
        self._begin()
        self._result = None
        self._train_aoi = self._train_pos = self._train_bg = self._train_transform = None
        self._clear_prediction()
        self.cnn_note.hide()
        self.validation_label.hide()
        self.hint_label.setText("Avval modelni o'qiting.")
        self._update_overlay_widgets()
        self._set_has_result(False)

    def figures(self):
        """Har qatlam xaritasi (yuqori ruxsatda, AOI bilan) va success-rate - yangi Figure'lar."""
        if not self._has_result or not isinstance(self._pred, dict):
            return {}
        out = {}
        for i in range(self.layer_combo.count()):
            key = self.layer_combo.itemData(i)
            spec = self._layer_spec(key, MAP_MAX_PX_SAVE)
            if spec is None:
                continue
            arr, tr = spec.pop("array"), spec.pop("transform")
            fig = plots.draw_map(_new_fig((9.0, 7.5)), arr, tr, **spec, **self._overlay_kwargs(tr, use_checks=False))
            out["map_" + str(key).replace(":", "_")] = fig
        curve = self._pred.get("success_curve")
        if isinstance(curve, dict) and curve:
            out["success_rate"] = plots.draw_success_rate(_new_fig((7.5, 5.5)), curve)
        return out

    @_slot
    def set_busy(self, busy):
        self._busy = bool(busy)
        self._refresh_buttons()
