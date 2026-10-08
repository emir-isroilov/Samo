# -*- coding: utf-8 -*-
"""
Umumiy GUI vidjetlari (PyQt5): papka tanlash, matplotlib canvas, jadval (DataFrame), progress paneli, log oynasi.

  FolderPicker(label_text)   .path() .setPath(p)  signal changed(str)
  MplCanvas                  .fig .redraw() .clear()   (Figure + FigureCanvasQTAgg + NavigationToolbar2QT)
  DataFrameTable             .set_dataframe(df, float_fmt) .save_csv(path) .copy_selection()  (saralash, Ctrl+C)
  ProgressPanel              .update_progress(frac|0..100, msg) .reset()   (bar + bosqich matni + ETA)
  LogView                    .append_line(text) .save_to_file(path) .clear_log()   (qatorlar chegarasi bilan)
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QGuiApplication, QKeySequence
from PyQt5.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMenu,
                             QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

__all__ = ["FolderPicker", "MplCanvas", "DataFrameTable", "ProgressPanel", "LogView", "format_duration", "elide_text"]

_log = logging.getLogger(__name__)

PHASE_PAUSE_MS = 4                   # og'ir chizish bosqichlari orasidagi pauza: event loop (sichqoncha, paint, timer) nafas oladi


def format_duration(seconds):
    """Soniyalarni 'mm:ss' (yoki 'h:mm:ss') ko'rinishiga o'tkazadi; noma'lum qiymat => '--:--'."""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "--:--"
    if not math.isfinite(s) or s < 0:
        return "--:--"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def elide_text(text, limit=150):
    """Matnni `limit` belgigacha qisqartiradi: oxiriga '...' qo'shiladi (jami uzunlik limit'dan oshmaydi).
    Qisqartirilmagan matn o'zgarmaydi; limit < 4 bo'lsa ham '...' sig'adigan eng kichik uzunlik olinadi."""
    text = "" if text is None else str(text)
    limit = max(4, int(limit))
    if len(text) <= limit:
        return text
    return text[:limit - 3].rstrip() + "..."


# ---------------------------------------------------------------------------
# FolderPicker
# ---------------------------------------------------------------------------
class FolderPicker(QWidget):
    """Label + matn maydoni + 'Tanlash...' tugmasi. Yo'l o'zgarganda `changed(str)` chiqadi."""

    changed = pyqtSignal(str)

    def __init__(self, label_text="", parent=None, label_width=220):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(label_text)
        if label_width:
            self.label.setMinimumWidth(int(label_width))
        self.line_edit = QLineEdit()
        self.btn = QPushButton("Tanlash...")
        self.btn.clicked.connect(self.browse)
        lay.addWidget(self.label)
        lay.addWidget(self.line_edit, 1)
        lay.addWidget(self.btn)
        self.line_edit.textChanged.connect(lambda _t: self.changed.emit(self.path()))

    def browse(self):
        """Papka tanlash dialogi (boshlang'ich papka = joriy yo'l)."""
        folder = QFileDialog.getExistingDirectory(self, "Papkani tanlang", self.path())
        if folder:
            self.setPath(folder)

    def path(self):
        return self.line_edit.text().strip()

    def setPath(self, p):                        # noqa: N802 (Qt uslubi)
        self.line_edit.setText("" if p is None else str(p))


# ---------------------------------------------------------------------------
# MplCanvas
# ---------------------------------------------------------------------------
class _PhasedCanvas(FigureCanvasQTAgg):
    """`draw_idle()` (oyna/tab o'lchami o'zgarganda, zoom/pan, kechiktirilgan chizish) ni ikki hodisa-tsikl bosqichiga
    bo'ladi: 1) layout (constrained) hisoblash, 2) rasterlash. Bitta og'ir grafik ~0.6 s qotish o'rniga ~0.2-0.3 s
    bo'laklar beradi. Bosqichlar orasida yana `draw_idle()` kelsa (grafik yoki o'lcham o'zgardi), eskirgan rasterlash
    o'tkazib yuboriladi va layout qayta hisoblanadi (ortiqcha ish yo'q). Layout dvigateli bosqichlar orasida vaqtincha
    o'chiriladi va HAR DOIM tiklanadi; biror xato bo'lsa matplotlib'ning standart (bir bosqichli) yo'liga qaytiladi."""

    def __init__(self, fig):
        super().__init__(fig)
        self._pp_busy = False                    # bosqichli chizish kutilmoqda/bajarilmoqda
        self._pp_again = False                   # shu paytda yana so'rov kelgan
        self._pp_eng = None
        self._pp_hold = None                     # layout vaqtida o'rnatilgan "none" dvigatel (tiklashda solishtiriladi)
        self._pp_size = None

    def draw_idle(self):                         # noqa: D401
        try:
            if self._pp_busy:
                self._pp_again = True
                return
            self._pp_busy = True
            QTimer.singleShot(PHASE_PAUSE_MS, self._pp_layout)
        except Exception:                        # noqa: BLE001
            self._pp_busy = False
            super().draw_idle()

    def _pp_restore(self):
        """Layout dvigatelini tiklaydi (grafik shu orada qayta chizilmagan bo'lsa). O'lcham o'zgargan bo'lsa True."""
        changed = False
        try:
            if self._pp_eng is not None:
                if self.figure.get_layout_engine() is self._pp_hold:
                    self.figure.set_layout_engine(self._pp_eng)
                changed = tuple(self.figure.get_size_inches()) != self._pp_size
        except Exception:                        # noqa: BLE001
            pass
        self._pp_eng = self._pp_hold = None
        return changed

    def _pp_layout(self):
        try:
            self._pp_again = False
            fig = self.figure
            self._pp_eng = None
            if self.width() > 0 and self.height() > 0 and fig is not None and fig.axes:
                eng = fig.get_layout_engine()
                if eng is not None:
                    size = tuple(fig.get_size_inches())
                    eng.execute(fig)
                    fig.set_layout_engine("none")        # rasterlashda qayta hisoblanmasin
                    self._pp_eng, self._pp_size = eng, size
                    self._pp_hold = fig.get_layout_engine()
        except RuntimeError:                     # vidjet o'chirilgan
            self._pp_busy = False
            return
        except Exception:                        # noqa: BLE001
            self._pp_eng = None
        QTimer.singleShot(PHASE_PAUSE_MS, self._pp_render)

    def _pp_render(self):
        try:
            if self._pp_again:                   # layout'dan keyin grafik/o'lcham o'zgardi: eskirgan rasterlash kerak emas
                self._pp_restore()
                QTimer.singleShot(PHASE_PAUSE_MS, self._pp_layout)
                return
            try:
                if self.width() > 0 and self.height() > 0:
                    self.draw()
            except Exception:                    # noqa: BLE001
                _log.exception("canvas chizish xatosi")
            again = self._pp_restore()
            self._pp_busy = False
            self._pp_again = False
            if again:
                self.draw_idle()
        except RuntimeError:                     # vidjet o'chirilgan
            self._pp_busy = False


class MplCanvas(QWidget):
    """matplotlib Figure + Qt canvas + navigatsiya paneli (zoom/pan/saqlash).

    `.fig` - oddiy `matplotlib.figure.Figure` (plots.draw_* funksiyalari shuni oladi);
    chizgandan so'ng `.redraw()` chaqiriladi."""

    def __init__(self, parent=None, figsize=(6.0, 4.5), dpi=100, with_toolbar=True):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, dpi=dpi, layout="constrained")
        self.canvas = _PhasedCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(200, 150)
        self.toolbar = NavigationToolbar2QT(self.canvas, self) if with_toolbar else None
        self.last_error = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        if self.toolbar is not None:
            lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

    def redraw(self):
        """Canvas'ni qayta chizadi. Chizish xatosi slotni qulatmaydi (xato `last_error` da)."""
        try:
            self.canvas.draw()
            self.last_error = None
        except Exception as exc:                 # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            _log.exception("MplCanvas.redraw xatosi")

    def clear(self):
        """Figurani tozalaydi va qayta chizadi."""
        self.fig.clear()
        self.redraw()

    def save_figure(self, path, dpi=300):
        """Figurani faylga saqlaydi (kengaytma bo'yicha: png/pdf/...). Yo'lni qaytaradi."""
        self.fig.savefig(path, dpi=dpi)
        return path


# ---------------------------------------------------------------------------
# DataFrameTable
# ---------------------------------------------------------------------------
def _sort_key(v):
    """Aralash turdagi ustunlar uchun tartiblash kaliti: bo'sh/NaN < sonlar < matnlar."""
    if v is None:
        return (0, 0.0, "")
    if isinstance(v, (bool, int, float, np.integer, np.floating, np.bool_)):
        f = float(v)
        return (0, 0.0, "") if math.isnan(f) else (1, f, "")
    return (2, 0.0, str(v).lower())


class _SortItem(QTableWidgetItem):
    """Son ustunlar son sifatida, matnlar matn sifatida saralanadigan jadval katakchasi.
    Kalit har doim (daraja, son, matn) uchligi: aralash turli ustunda ham TypeError chiqmaydi
    (PyQt5 da __lt__ ichidagi istisno dasturni abort qiladi)."""

    def __init__(self, text, key):
        super().__init__(text)
        self.sort_key = _sort_key(key)

    def __lt__(self, other):
        ka = getattr(self, "sort_key", None)
        kb = getattr(other, "sort_key", None)
        if ka is None or kb is None:
            return super().__lt__(other)
        return ka < kb


class DataFrameTable(QTableWidget):
    """pandas.DataFrame ni ko'rsatuvchi jadval: son/matn bo'yicha saralash, Ctrl+C (TSV), CSV ga saqlash.

    Float'lar `float_fmt` bilan (str yoki {ustun: fmt} lug'ati), NaN/None bo'sh katak. Juda katta jadvalda
    faqat dastlabki `max_rows` qator ko'rsatiladi (CSV'ga esa hammasi saqlanadi)."""

    def __init__(self, parent=None, max_rows=20000):
        super().__init__(parent)
        self.max_rows = int(max_rows)
        self._df = None
        self._show_index = False
        self.truncated = False
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.horizontalHeader().setStretchLastSection(False)
        self.setContextMenuPolicy(Qt.DefaultContextMenu)

    # ---- ma'lumot
    @staticmethod
    def _fmt_cell(v, fmt):
        """(matn, saralash kaliti)."""
        if v is None:
            return "", None
        if isinstance(v, (bool, np.bool_)):
            return ("Ha" if bool(v) else "Yo'q"), float(bool(v))
        if isinstance(v, (int, np.integer)):
            return str(int(v)), float(v)
        if isinstance(v, (float, np.floating)):
            f = float(v)
            if math.isnan(f):
                return "", None
            if math.isinf(f):
                return ("inf" if f > 0 else "-inf"), f
            try:
                return fmt.format(f), f
            except (ValueError, KeyError, IndexError):
                return str(f), f
        if v is pd.NaT or v is pd.NA:
            return "", None
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v)
            if arr.size > 8:
                return f"[{arr.size} element]", None
            return str(arr.tolist()), None
        return str(v), str(v)

    def set_dataframe(self, df, float_fmt="{:.3f}"):
        """Jadvalni df bilan to'ldiradi (None/bo'sh => tozalaydi). Indeks standart RangeIndex bo'lmasa,
        birinchi ustun sifatida ko'rsatiladi."""
        sorting = self.isSortingEnabled()
        self.setSortingEnabled(False)
        self.setUpdatesEnabled(False)
        try:
            self.clear()
            self.truncated = False
            if df is None or not isinstance(df, pd.DataFrame):
                self._df = None
                self._show_index = False
                self.setRowCount(0)
                self.setColumnCount(0)
                return
            self._df = df
            idx = df.index
            default_idx = isinstance(idx, pd.RangeIndex) and idx.start == 0 and idx.step == 1
            self._show_index = not default_idx
            cols = ["" if idx.name is None else str(idx.name)] if self._show_index else []
            for c in df.columns:
                cols.append(" / ".join(map(str, c)) if isinstance(c, tuple) else str(c))
            view = df.iloc[: self.max_rows]
            self.truncated = len(df) > len(view)
            n_rows, n_cols = len(view), len(cols)
            self.setRowCount(n_rows)
            self.setColumnCount(n_cols)
            self.setHorizontalHeaderLabels(cols)
            fmts = list(df.columns)
            col_fmt = ({c: (float_fmt.get(c, "{:.3f}")) for c in fmts} if isinstance(float_fmt, dict)
                       else {c: float_fmt for c in fmts})
            off = 1 if self._show_index else 0
            values = view.to_numpy(dtype=object) if n_rows and len(df.columns) else None
            for i in range(n_rows):
                if self._show_index:
                    txt, key = self._fmt_cell(idx[i], "{:.3f}")
                    self.setItem(i, 0, _SortItem(txt, key if key is not None else str(idx[i])))
                if values is None:
                    continue
                row = values[i]
                for j, c in enumerate(fmts):
                    txt, key = self._fmt_cell(row[j], col_fmt[c])
                    self.setItem(i, j + off, _SortItem(txt, key))
            if self.truncated:
                self.setVerticalHeaderLabels([str(k) for k in range(1, n_rows + 1)])
            self.resizeColumnsToContents()
        finally:
            self.setUpdatesEnabled(True)
            self.setSortingEnabled(sorting)

    def dataframe(self):
        """Hozir ko'rsatilayotgan (asl) DataFrame yoki None."""
        return self._df

    def clear_table(self):
        self.set_dataframe(None)

    # ---- nusxalash
    def copy_selection(self, with_header=False):
        """Tanlangan katakchalarni TSV matn sifatida buferga nusxalaydi va matnni qaytaradi."""
        rng = self.selectedIndexes()
        if not rng:
            return ""
        rows = sorted({i.row() for i in rng})
        cols = sorted({i.column() for i in rng})
        chosen = {(i.row(), i.column()) for i in rng}
        lines = []
        if with_header:
            lines.append("\t".join(self.horizontalHeaderItem(c).text() if self.horizontalHeaderItem(c) else ""
                                   for c in cols))
        for r in rows:
            cells = []
            for c in cols:
                it = self.item(r, c)
                cells.append(it.text() if (it is not None and (r, c) in chosen) else "")
            lines.append("\t".join(cells))
        text = "\n".join(lines)
        cb = QGuiApplication.clipboard()
        if cb is not None:
            cb.setText(text)
        return text

    def copy_all(self, with_header=True):
        self.selectAll()
        return self.copy_selection(with_header=with_header)

    def keyPressEvent(self, event):              # noqa: N802
        if event.matches(QKeySequence.Copy):
            self.copy_selection()
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):           # noqa: N802
        menu = QMenu(self)
        menu.addAction("Nusxalash (Ctrl+C)", self.copy_selection)
        menu.addAction("Sarlavha bilan nusxalash", lambda: self.copy_selection(with_header=True))
        menu.addAction("Hammasini nusxalash", self.copy_all)
        menu.addSeparator()
        menu.addAction("CSV ga saqlash...", self.save_csv)
        menu.exec_(event.globalPos())

    # ---- CSV
    def save_csv(self, path=None):
        """To'liq DataFrame'ni CSV ga yozadi (utf-8-sig: Excel o'zbekcha belgilarni to'g'ri ochadi).
        path=None bo'lsa fayl dialogi ochiladi. Yozilgan yo'lni (yoki bekor qilinsa None) qaytaradi."""
        if self._df is None:
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "CSV ga saqlash", "jadval.csv", "CSV (*.csv)")
            if not path:
                return None
        self._df.to_csv(path, index=self._show_index, encoding="utf-8-sig")
        return path


# ---------------------------------------------------------------------------
# ProgressPanel
# ---------------------------------------------------------------------------
class ProgressPanel(QWidget):
    """Progress bar + bosqich matni + o'tgan vaqt va taxminiy qolgan vaqt (ETA).

    update_progress(value, msg): value - float bo'lsa 0..1 ulush (agar > 1 bo'lsa foiz), int bo'lsa 0..100 foiz
    (ishchilarning `progress_signal(int, str)` i shu ko'rinishda). ETA butun jarayon bo'yicha chiziqli
    ekstrapolyatsiya (bosqichlar og'irligi teng emas, shuning uchun taxminiy)."""

    def __init__(self, parent=None, clock=None):
        super().__init__(parent)
        self._clock = clock or time.monotonic
        self._t0 = None
        self._t_end = None                        # tugash/to'xtash vaqti (o'tgan vaqt qotib qoladi)
        self._f0 = 0.0
        self._frac = 0.0
        self._active = False
        self._done = False
        lay = QHBoxLayout(self)                   # bitta ixcham qator: [bar] [bosqich matni] [vaqt]
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setMinimumWidth(180)
        self.bar.setMaximumHeight(20)
        self.stage_label = QLabel("Tayyor")
        self.stage_label.setWordWrap(True)
        self.time_label = QLabel("")
        lay.addWidget(self.bar, 2)
        lay.addWidget(self.stage_label, 3)
        lay.addWidget(self.time_label)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._refresh_time)

    # ---- yordamchilar
    @staticmethod
    def _to_fraction(value):
        if value is None:
            return 0.0
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            f = float(value) / 100.0
        else:
            f = float(value)
            if f > 1.0:
                f /= 100.0
        if math.isnan(f):
            return 0.0
        return min(1.0, max(0.0, f))

    def elapsed(self):
        if self._t0 is None:
            return 0.0
        end = self._clock() if self._t_end is None else self._t_end
        return max(0.0, end - self._t0)

    def _mark_end(self):
        if self._t0 is not None and self._t_end is None:
            self._t_end = self._clock()

    def eta_seconds(self):
        """Taxminiy qolgan vaqt (soniya) yoki None (hali baholab bo'lmaydi)."""
        if self._t0 is None or self._done:
            return None
        el = self.elapsed()
        gain = self._frac - self._f0
        if gain <= 0.0 or self._frac < 0.02 or el < 0.5:
            return None
        return el * (1.0 - self._frac) / gain

    def _refresh_time(self):
        if self._t0 is None:
            self.time_label.setText("")
            return
        txt = f"O'tgan: {format_duration(self.elapsed())}"
        if not self._done:
            eta = self.eta_seconds()
            txt += f"  |  ETA: {format_duration(eta) if eta is not None else '--:--'}"
        self.time_label.setText(txt)

    # ---- ochiq API
    def update_progress(self, value, msg=""):
        """Progressni yangilaydi (value: ulush 0..1 yoki foiz 0..100) va bosqich matnini qo'yadi."""
        frac = self._to_fraction(value)
        # yangi ish: birinchi chaqiruv, progress orqaga qaytdi yoki oldingi ish tugagan/to'xtatilgan edi
        if self._t0 is None or frac < self._frac - 0.05 or (self._done and frac < 1.0):
            self._t0 = self._clock()
            self._t_end = None
            self._f0 = frac
        self._frac = frac
        self._done = False
        self._active = True
        self.bar.setValue(int(round(frac * 100)))
        if msg:
            self.stage_label.setText(str(msg))
            self.stage_label.setToolTip("")
        if not self._timer.isActive():
            self._timer.start()
        self._refresh_time()
        if frac >= 1.0:
            self.finish()

    def finish(self, msg="Tugadi"):
        """100% va yakuniy matn; taymer to'xtaydi (o'tgan vaqt ko'rinib turadi)."""
        self._frac = 1.0
        self._done = True
        self._active = False
        self._mark_end()
        self._timer.stop()
        self.bar.setValue(100)
        if msg:
            self.stage_label.setText(str(msg))
            self.stage_label.setToolTip("")
        self._refresh_time()

    def stopped(self, msg="To'xtatildi", tooltip=None):
        """Bekor qilinganda/xatoda: bar joyida qoladi, taymer to'xtaydi. `tooltip` (ixtiyoriy) - qisqartirilgan
        `msg` ning to'liq matni (sichqoncha ustiga olib borilganda ko'rinadi)."""
        self._done = True
        self._active = False
        self._mark_end()
        self._timer.stop()
        if msg:
            self.stage_label.setText(str(msg))
            self.stage_label.setToolTip("" if tooltip is None else str(tooltip))
        self._refresh_time()

    def reset(self):
        """Boshlang'ich holat."""
        self._timer.stop()
        self._t0 = None
        self._t_end = None
        self._f0 = 0.0
        self._frac = 0.0
        self._active = False
        self._done = False
        self.bar.setValue(0)
        self.stage_label.setText("Tayyor")
        self.stage_label.setToolTip("")
        self.time_label.setText("")

    def value(self):
        """Joriy foiz (0..100)."""
        return int(self.bar.value())

    def stage_text(self):
        return self.stage_label.text()

    def is_active(self):
        return self._active


# ---------------------------------------------------------------------------
# LogView
# ---------------------------------------------------------------------------
class LogView(QPlainTextEdit):
    """Faqat o'qiladigan log oynasi. `max_lines` dan oshsa eng eski qatorlar o'chadi (xotira chegarasi).
    Pastga avtomatik aylantiradi (foydalanuvchi yuqoriga o'tgan bo'lsa - aralashmaydi)."""

    def __init__(self, parent=None, max_lines=20000):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        font = QFont("Monospace")
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)
        self.setMaximumBlockCount(int(max_lines))
        self.total_lines = 0                      # jami qo'shilgan qatorlar (o'chirilganlari ham)

    def max_lines(self):
        return self.maximumBlockCount()

    def set_max_lines(self, n):
        self.setMaximumBlockCount(max(1, int(n)))

    def append_line(self, text):
        """Bitta (yoki ko'p qatorli) matnni oxiriga qo'shadi."""
        text = "" if text is None else str(text).rstrip("\r\n")
        bar = self.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 2
        self.appendPlainText(text)
        self.total_lines += text.count("\n") + 1
        if at_bottom:
            bar.setValue(bar.maximum())

    def lines(self):
        """Ko'rinib turgan qatorlar ro'yxati."""
        txt = self.toPlainText()
        return txt.split("\n") if txt else []

    def n_dropped(self):
        """Chegaradan oshgani uchun o'chirilgan qatorlar soni."""
        return max(0, self.total_lines - len(self.lines()))

    def save_to_file(self, path=None):
        """Ko'rinib turgan matnni utf-8 faylga yozadi (path=None => dialog). Yo'lni yoki None qaytaradi."""
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Logni saqlash", "mpm_log.txt", "Matn (*.txt *.log)")
            if not path:
                return None
        with open(path, "w", encoding="utf-8") as f:
            txt = self.toPlainText()
            f.write(txt + ("\n" if txt and not txt.endswith("\n") else ""))
        return path

    def clear_log(self):
        self.clear()
        self.total_lines = 0
