# -*- coding: utf-8 -*-
"""
Asosiy oyna (PyQt5): `MainWindow` va `main(argv=None)`.

Tab'lar:
  1. Ma'lumotlar va o'qitish   (kirish papkalari, qatlamlar, fon, CV, modellar, kalibrlash, talqin, umumiy; O'qitish/Stop,
                                progress, log, cost hint)
  2. Giperparametrlar           (HyperParamPanel: PARAM_SPECS dan avtomatik + tuning + cost hint)
  3. Ma'lumotlar tahlili        (DiagnosticsTab)       4. Natijalar (ResultsTab)      5. Spatial CV diagnostika (SpatialTab)
  6. Feature importance         (ImportanceTab)        7. Prognoz xarita (MapTab)     8. Modellar (bundle saqlash/yuklash/qo'llash)

Qoidalar:
  * barcha uzoq ishlar `workers.*` (QThread) orqali - GUI qotmaydi; bir vaqtda bitta asosiy ish (o'qitish/prognoz/
    bundle'ni qo'llash/eksport/diagnostika/bundle saqlash);
  * `_set_busy(bool)` -> `_update_controls()`: BARCHA tugmalar holati bitta joyda hisoblanadi (BUG-02: xato, to'xtatish
    yoki tugashdan keyin hammasi tiklanadi; prognoz tugmasi faqat o'qitish natijasi bo'lsa yoqiladi);
  * slotlar `_guard` bilan o'ralgan: istisno PyQt5 abort'iga olib bormaydi (BUG-01); `main()` global `sys.excepthook` o'rnatadi;
  * dialoglar (`_warn/_info/_show_error/_confirm_*/_ask_directory`) alohida metodlarda - testlarda almashtiriladi;
  * `tensorflow`/`shap` modul darajasida import qilinmaydi; `pipeline` (cost hint uchun) kechiktirib import qilinadi.

Chiqish - "prospektivlik indeksi" (0-1), haqiqiy ehtimollik emas.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import sys
import threading
import traceback
from functools import partial

import numpy as np
from PyQt5.QtCore import QEventLoop, QThread, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QAction, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QSplitter,
                             QTabWidget, QVBoxLayout, QWidget)

from .. import __version__ as MPM_VERSION
from .. import config
from ..common import ENSEMBLE_NAME, RANDOM_STATE, TARGET_EPSG, tf_available, xgboost_available
from ..config import BACKGROUND_STRATEGIES, MODEL_NAMES, RunConfig
from ..workers import ApplyBundleWorker, ExportWorker, PredictionWorker, TrainingWorker, _BaseWorker
from .param_panel import MODEL_TITLES, HyperParamPanel
from .result_tabs import (INDEX_NOTE, DiagnosticsTab, ImportanceTab, MapTab, ResultsTab, SpatialTab, format_breaks,
                          parse_breaks)
from .widgets import FolderPicker, LogView, ProgressPanel

__all__ = ["MainWindow", "main", "install_excepthook", "format_manifest", "scalar_metrics", "WINDOW_TITLE",
           "TAB_TITLES", "TAB_DATA", "TAB_HYPER", "TAB_DIAG", "TAB_RESULTS", "TAB_SPATIAL", "TAB_IMPORTANCE",
           "TAB_MAP", "TAB_MODELS"]

_log = logging.getLogger("mpm.gui.main_window")

WINDOW_TITLE = "MPM ML GUI v2 - Oltin ma'danlashuvi prospektivligini bashoratlash (RF / SVM / XGBoost / CNN / ansambl)"
TAB_DATA, TAB_HYPER, TAB_DIAG, TAB_RESULTS, TAB_SPATIAL, TAB_IMPORTANCE, TAB_MAP, TAB_MODELS = range(8)
TAB_TITLES = ("1. Ma'lumotlar va o'qitish", "2. Giperparametrlar", "3. Ma'lumotlar tahlili", "4. Natijalar",
              "5. Spatial CV diagnostika", "6. Feature importance", "7. Prognoz xarita", "8. Modellar")

_STRATEGY_TEXT = {"random": "Tasodifiy (random)", "grid": "To'r (grid, jitterli)",
                  "distance_weighted": "Konlardan masofaga proporsional (distance_weighted)"}
_CAL_TEXT = {"sigmoid": "Sigmoid (Platt)", "isotonic": "Isotonik (isotonic)"}

# ish turi -> (xato dialogi sarlavhasi, to'xtatish mumkinmi)
_JOBS = {
    "train": ("O'qitishda xato", True),
    "predict": ("Prognozda xato", True),
    "apply": ("Bundle'ni yangi maydonga qo'llashda xato", True),
    "diag": ("Ma'lumotlar tahlilida xato", True),
    "export": ("Eksportda xato", False),
    "autoexport": ("Avtomatik eksportda xato", False),
    "bundle_save": ("Modelni saqlashda xato", False),
}

INDEX_HELP = (
    "Prospektivlik indeksi (0-1) - bu HAQIQIY EHTIMOLLIK EMAS.\n\n"
    "Modellar ma'lum konlar (musbat nuqtalar) bilan tasodifiy tanlangan fon nuqtalar o'rtasidagi farqni o'rganadi. "
    "Musbat:fon nisbati sun'iy (fon nuqtalar sonini foydalanuvchi tanlaydi), shuning uchun chiqish qiymati "
    "\"bu piksel qatorida kon topilish ehtimoli\"ni bildirmaydi - u faqat maydonlarni o'zaro NISBIY tartiblash "
    "(qaysi joy konlarga o'xshashroq) uchun xizmat qiladi.\n\n"
    "Muhim eslatmalar:\n"
    "- Natijani baholash uchun spatial block CV ishlatiladi (random CV faqat benchmark, optimistik).\n"
    "- Success-rate egri chizig'i o'qitish nuqtalarida hisoblanadi - optimistik, mustaqil validatsiya emas.\n"
    "- Noaniqlik xaritasi - modellar kelishmovchiligi (std), statistik ishonch oralig'i emas.\n"
    "- CNN chiqishi kalibrlanmaydi va neg/pos og'irlik bilan o'qitiladi; ansambl oddiy o'rtacha bo'lgani uchun "
    "shkalalar farq qilishi mumkin.\n"
    "- Fayl nomlari `prognoz_*.tif` (tarixiy nom), lekin ularning mazmuni - prospektivlik indeksi."
)

BUNDLE_SECURITY_TEXT = (
    "Diqqat: xavfsizlik!\n\n"
    "Saqlangan modellar (RF, SVM, XGBoost) joblib/pickle formatida. Bunday fayl yuklanganda u ichidagi "
    "o'zboshimchalik bilan kod bajarilishi mumkin (kompyuteringiz xavf ostida qoladi).\n\n"
    "Faqat o'zingiz yaratgan yoki to'liq ishonchli manbadan olingan bundle'ni yuklang.\n\n"
    "Bu bundle ishonchli manbadanmi?"
)

METADATA_NOTE = ("Eslatma: o'qitishda TIFF papkasiga metadata.csv shabloni YOZILADI (yo'q bo'lsa). Uni to'ldirib, "
                 "keyingi ishga tushirishda ishlating (data dictionary uchun).")

_ERR_STYLE = "color: #B00020;"
_NOTE_STYLE = "color: gray;"


# ---------------------------------------------------------------------------
# Qt'siz yordamchilar (testlanadi)
# ---------------------------------------------------------------------------
def _is_scalar(v):
    return isinstance(v, (bool, int, float, np.integer, np.floating, np.bool_))


def scalar_metrics(metrics):
    """Bundle manifest'iga yoziladigan ixcham metrikalar: {model: {nom: skalyar}} (massivlar tushiriladi;
    `*_ci95` juftligi `*_ci95_lo/_hi` ga ajratiladi). NaN saqlanadi (persist uni null qiladi)."""
    out = {}
    for name, m in (metrics or {}).items():
        if not isinstance(m, dict):
            continue
        d = {}
        for k, v in m.items():
            if _is_scalar(v):
                d[str(k)] = float(v) if not isinstance(v, (bool, np.bool_)) else bool(v)
            elif str(k).endswith("_ci95"):
                try:
                    lo, hi = v
                    d[f"{k}_lo"], d[f"{k}_hi"] = float(lo), float(hi)
                except (TypeError, ValueError):
                    pass
        out[str(name)] = d
    return out


def _fmt(v, nd=3):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    return "-" if f != f else f"{f:.{nd}f}"


def format_manifest(manifest):
    """Bundle manifest.json -> foydalanuvchiga ko'rsatiladigan o'zbekcha matn (xatoga chidamli)."""
    if not isinstance(manifest, dict):
        return "Manifest o'qib bo'lmadi."
    L = []
    L.append(f"Yaratilgan (UTC): {manifest.get('created_utc', '?')}   |   mpm versiyasi: {manifest.get('mpm_version', '?')}"
             f"   |   bundle_version: {manifest.get('bundle_version', '?')}")
    bands = list(manifest.get("band_names") or [])
    cats = list(manifest.get("categorical") or [])
    L.append(f"Band nomlari ({len(bands)}): {', '.join(bands) if bands else '-'}")
    L.append(f"Kategorik qatlamlar: {', '.join(cats) if cats else 'yo`q'}   |   feature'lar soni: "
             f"{manifest.get('n_features', '?')}   |   CRS: EPSG:{manifest.get('crs_epsg', '?')}")
    models = manifest.get("models") or {}
    L.append("Modellar:")
    for name in (manifest.get("model_names") or list(models)):
        info = models.get(name) or {}
        L.append(f"  - {name}: {info.get('n_draws', '?')} ta fon tanlovi, {info.get('class', '?')} "
                 f"({info.get('input_kind', '?')}, {info.get('loader', '?')})")
    hpu = manifest.get("hyperparams_used") or {}
    if isinstance(hpu, dict) and hpu:
        L.append("Giperparametrlar (1-fon tanlovi):")
        for name, val in hpu.items():
            p = val[0] if isinstance(val, list) and val else val
            if isinstance(p, dict):
                L.append(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in p.items()))
    ms = manifest.get("metrics_summary") or {}
    if isinstance(ms, dict) and ms:
        L.append("Metrikalar (o'qitishdagi spatial CV):")
        for name, m in ms.items():
            if isinstance(m, dict):
                ci = ""
                if m.get("auc_ci95_lo") is not None and m.get("auc_ci95_hi") is not None:
                    ci = f" (95% CI {_fmt(m.get('auc_ci95_lo'))}-{_fmt(m.get('auc_ci95_hi'))})"
                L.append(f"  {name}: AUC={_fmt(m.get('auc'))}{ci}, PR-AUC={_fmt(m.get('pr_auc'))}, "
                         f"BalAcc={_fmt(m.get('balanced_accuracy'))}")
    if manifest.get("block_size") is not None:
        L.append(f"Spatial blok o'lchami: {_fmt(manifest.get('block_size'), 0)} m")
    if manifest.get("notes"):
        L.append(f"Izoh: {manifest['notes']}")
    versions = manifest.get("versions") or {}
    if isinstance(versions, dict) and versions:
        keys = ("python", "scikit-learn", "xgboost", "tensorflow", "numpy")
        L.append("Kutubxona versiyalari: " + ", ".join(f"{k} {versions.get(k)}" for k in keys if versions.get(k)))
    L.append("")
    L.append("Eslatma: manifest faqat ma'lumot uchun o'qildi (modellar hali yuklanmagan). Modellar (joblib/pickle) "
             "'Yangi maydonga qo'llash' paytida yuklanadi - faqat ishonchli manbadagi bundle'ni qo'llang. "
             "Chiqish - prospektivlik indeksi, ehtimollik emas.")
    return "\n".join(L)


def _list_tiff_stems(folder):
    """Papkadagi .tif/.tiff fayl nomlari (kengaytmasiz = band nomi), alifbo tartibida (data.find_tiff_files kabi)."""
    if not folder or not os.path.isdir(folder):
        return []
    try:
        names = [n for n in os.listdir(folder)
                 if not n.startswith(".") and n.lower().endswith((".tif", ".tiff"))
                 and os.path.isfile(os.path.join(folder, n))]
    except OSError:
        return []
    return [os.path.splitext(n)[0] for n in sorted(names, key=lambda s: (s.lower(), s))]


def _available(model):
    """Model uchun kutubxona mavjudmi (import qilmasdan; haqiqiy tekshiruvni pipeline bajaradi)."""
    if model == "XGBoost":
        return xgboost_available()
    if model == "CNN":
        return tf_available()
    return True


# ---------------------------------------------------------------------------
# Ishchi oqimlar xavfsizligi
# ---------------------------------------------------------------------------
_THREAD_KEEPER = set()            # ishlayotgan QThread'larga kuchli havola: oyna o'chsa ham "Destroyed while running" bo'lmaydi
_KEEPER_LOCK = threading.Lock()


def _forget_thread(worker):
    with _KEEPER_LOCK:
        _THREAD_KEEPER.discard(worker)
    try:
        worker.deleteLater()
    except RuntimeError:
        pass


class _CallWorker(_BaseWorker):
    """Ixtiyoriy `fn(worker) -> dict` ni ishchi oqimda bajaradi (qatlamlarni skanerlash, tez diagnostika, bundle saqlash)."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def _work(self):
        return self._fn(self)


def _guard(fn):
    """Slot himoyasi (BUG-01): istisno tashqariga chiqmaydi (PyQt5 abort qilmaydi), log/xabarga yoziladi."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as exc:                                  # noqa: BLE001
            try:
                self._report_exception(fn.__name__, exc)
            except Exception:                                     # noqa: BLE001
                _log.exception("_guard: xatoni ko'rsatib bo'lmadi")
            return None
    return wrapper


class _Spin(QSpinBox):
    """Sichqoncha g'ildiragi faqat fokusda ishlaydi (scroll paytida tasodifan qiymat o'zgarmasin)."""

    def __init__(self, lo, hi, val, step=1, parent=None):
        super().__init__(parent)
        self.setRange(int(lo), int(hi))
        self.setValue(int(val))
        self.setSingleStep(int(step))
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):                                  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _DSpin(QDoubleSpinBox):
    def __init__(self, lo, hi, val, step=1.0, decimals=1, parent=None):
        super().__init__(parent)
        self.setDecimals(int(decimals))
        self.setRange(float(lo), float(hi))
        self.setValue(float(val))
        self.setSingleStep(float(step))
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):                                  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _Combo(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):                                  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


def _note_label(text, style=_NOTE_STYLE):
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(style)
    return lab


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    """MPM ML GUI v2 asosiy oynasi (8 tab)."""

    # ixtiyoriy oqimdan (threading.excepthook) GUI oqimiga xavfsiz o'tkazish (vidjetlarga faqat GUI oqimidan tegiladi)
    _unhandled_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1400, 900)
        # --- holat
        self.training_result = None
        self._last_prediction = None            # faqat o'qitish natijasi bo'yicha prognoz (eksportga qo'shiladi)
        self._busy = False
        self._stopping = False
        self._closing = False
        self._applying = False
        self._job_worker = None
        self._job_kind = None
        self._scan_id = 0
        self._layers_touched = False
        self._bundle_dir = None
        self._bundle_manifest = None
        self._class_breaks_fallback = list(RunConfig().class_breaks)
        self._error_box_open = False
        self.last_message = None                # (tur, sarlavha, matn) - oxirgi dialog (testlar uchun)
        self.last_job_error = None
        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._unhandled_signal.connect(self._on_unhandled_signal)
        self._apply_config(RunConfig())
        self._update_controls()
        self._cost_timer.start()

    # ======================================================================
    # UI qurish
    # ======================================================================
    def _build_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_data_tab(), TAB_TITLES[TAB_DATA])
        self.tabs.addTab(self._build_hyper_tab(), TAB_TITLES[TAB_HYPER])
        self.diag_tab = DiagnosticsTab()
        self.results_tab = ResultsTab()
        self.spatial_tab = SpatialTab()
        self.importance_tab = ImportanceTab()
        self.map_tab = MapTab()
        self.tabs.addTab(self.diag_tab, TAB_TITLES[TAB_DIAG])
        self.tabs.addTab(self.results_tab, TAB_TITLES[TAB_RESULTS])
        self.tabs.addTab(self.spatial_tab, TAB_TITLES[TAB_SPATIAL])
        self.tabs.addTab(self.importance_tab, TAB_TITLES[TAB_IMPORTANCE])
        self.tabs.addTab(self.map_tab, TAB_TITLES[TAB_MAP])
        self.tabs.addTab(self._build_models_tab(), TAB_TITLES[TAB_MODELS])
        self.result_tab_widgets = (self.diag_tab, self.results_tab, self.spatial_tab, self.importance_tab, self.map_tab)
        for t in (self.diag_tab, self.spatial_tab, self.importance_tab, self.map_tab):
            self.results_tab.register_figure_source(t)
        self.statusBar().showMessage("Tayyor")
        self._cost_timer = QTimer(self)
        self._cost_timer.setSingleShot(True)
        self._cost_timer.setInterval(300)
        self._layer_scan_timer = QTimer(self)
        self._layer_scan_timer.setSingleShot(True)
        self._layer_scan_timer.setInterval(500)

    # ---------------------------------------------------------------- tab 1
    def _build_data_tab(self):
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        # ---- yuqori: sozlamalar (scroll ichida)
        self.settings_widget = QWidget()
        grid = QGridLayout(self.settings_widget)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.addWidget(self._group_inputs(), 0, 0, 1, 2)
        grid.addWidget(self._group_background(), 1, 0)
        grid.addWidget(self._group_cv(), 1, 1)
        grid.addWidget(self._group_models(), 2, 0)
        grid.addWidget(self._group_bg_sensitivity(), 2, 1)
        grid.addWidget(self._group_interpretation(), 3, 0)
        grid.addWidget(self._group_general(), 3, 1)
        grid.setRowStretch(4, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.settings_widget)
        splitter.addWidget(scroll)

        # ---- pastki: boshqaruv, cost hint, progress, log
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.btn_train = QPushButton("O'qitish (Train + Cross-Validation)")
        self.btn_train.setStyleSheet("font-weight: bold; padding: 6px 14px;")
        self.btn_stop = QPushButton("To'xtatish (Stop)")
        self.btn_stop.setToolTip("Hisoblashni hamkorlikda to'xtatadi (joriy qadam tugagach). Taxminan 95% dan keyin "
                                 "o'qitishni to'xtatib bo'lmaydi - u deyarli tugagan bo'ladi.")
        self.btn_save_cfg = QPushButton("Konfiguratsiyani saqlash (JSON)...")
        self.btn_load_cfg = QPushButton("Konfiguratsiyani yuklash (JSON)...")
        for b in (self.btn_train, self.btn_stop, self.btn_save_cfg, self.btn_load_cfg):
            row.addWidget(b)
        row.addStretch(1)
        bl.addLayout(row)
        self.cost_view = QPlainTextEdit()
        self.cost_view.setReadOnly(True)
        self.cost_view.setMaximumHeight(96)
        self.cost_view.setPlaceholderText("Taxminiy hisob-kitob (sozlamalar o'zgarganda yangilanadi)")
        bl.addWidget(self.cost_view)
        self.progress = ProgressPanel()
        bl.addWidget(self.progress)
        lrow = QHBoxLayout()
        lrow.addWidget(QLabel("Log:"))
        lrow.addStretch(1)
        self.btn_save_log = QPushButton("Logni saqlash...")
        self.btn_clear_log = QPushButton("Logni tozalash")
        lrow.addWidget(self.btn_save_log)
        lrow.addWidget(self.btn_clear_log)
        bl.addLayout(lrow)
        self.log_view = LogView()
        self.log_view.setMinimumHeight(110)
        bl.addWidget(self.log_view, 1)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([560, 340])
        return tab

    def _group_inputs(self):
        box = QGroupBox("Kirish ma'lumotlari (barchasi EPSG:28411 / GK 1942 zone 11 ga avtomatik moslashtiriladi)")
        lay = QVBoxLayout(box)
        self.picker_tiff = FolderPicker("TIFF qatlamlar papkasi:", label_width=250)
        self.picker_points = FolderPicker("Musbat nuqtalar (konlar) papkasi (.shp):", label_width=250)
        self.picker_aoi = FolderPicker("Maydon konturi (AOI) papkasi (.shp):", label_width=250)
        self.picker_output = FolderPicker("Chiqish papkasi (log, eksport, GeoTIFF):", label_width=250)
        self.picker_output.setToolTip("Bo'sh bo'lmasa: log fayli (mpm_run_N.log) yoziladi va o'qitish tugagach natijalar "
                                      "avtomatik eksport qilinadi.")
        for p in (self.picker_tiff, self.picker_points, self.picker_aoi, self.picker_output):
            lay.addWidget(p)
        lay.addWidget(QLabel("Qatlamlar (belgi = kategorik: nearest resample + one-hot; tavsiya avtomatik qo'yiladi):"))
        self.layer_list = QListWidget()
        self.layer_list.setMaximumHeight(96)
        self.layer_list.setToolTip("Belgilanmagan = raqamli qatlam. Kategorik qatlamlar (litologiya, razlom zonasi kodi) "
                                   "bilinear emas, nearest bilan moslanadi va one-hot kodlanadi.")
        lay.addWidget(self.layer_list)
        self.layer_summary = _note_label("")
        lay.addWidget(self.layer_summary)
        self.chk_assume_crs = QCheckBox("CRS yozilmagan TIFF/SHP fayllar EPSG:28411 (GK 1942 zone 11) deb qabul qilinsin")
        self.chk_assume_crs.setToolTip("Yoqilgan: CRS metadata yo'q fayllar qayta proyeksiya qilinmaydi, koordinatalari "
                                       "allaqachon EPSG:28411 da deb hisoblanadi.\nO'chirilgan: CRS yo'q fayl uchrasa, xato.")
        lay.addWidget(self.chk_assume_crs)
        lay.addWidget(_note_label(METADATA_NOTE))
        return box

    def _group_background(self):
        box = QGroupBox("Fon (pseudo-absence) nuqtalar")
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.spin_background = _Spin(5, 500000, 80, 10)
        self.spin_background.setToolTip("Fon nuqtalar soni (musbat:fon nisbati SUN'IY - chiqish ehtimollik emas).")
        self.spin_min_dist = _DSpin(0, 1_000_000, 500, 50, 1)
        self.spin_min_dist.setSuffix(" m")
        self.spin_min_dist.setToolTip("Fon nuqta musbat nuqtadan kamida shuncha masofada bo'lishi kerak.")
        self.combo_strategy = _Combo()
        for key in BACKGROUND_STRATEGIES:
            self.combo_strategy.addItem(_STRATEGY_TEXT.get(key, key), key)
        self.combo_strategy.setToolTip("random - AOI ichida tekis; grid - jitterli to'r; distance_weighted - konlardan "
                                       "uzoqroq joylar ko'proq tanlanadi.")
        self.spin_final_bg = _Spin(1, 50, 1, 1)
        self.spin_final_bg.setToolTip("Yakuniy modelni shuncha turli fon tanlovida o'qitib, xaritalarni o'rtachalaydi "
                                      "(fon tanloviga sezgirlikni kamaytiradi; vaqt shunga proporsional).")
        form.addRow("Fon nuqtalar soni:", self.spin_background)
        form.addRow("Musbat nuqtadan min. masofa:", self.spin_min_dist)
        form.addRow("Fon strategiyasi:", self.combo_strategy)
        form.addRow("Yakuniy fon ansambli (K):", self.spin_final_bg)
        return box

    def _group_cv(self):
        box = QGroupBox("Cross-validation (spatial block CV - asosiy)")
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.spin_kfold = _Spin(2, 20, 5, 1)
        self.spin_kfold.setToolTip("K-fold soni (har validation fold'da kamida bitta musbat nuqta bo'lishi kerak).")
        self.spin_repeats = _Spin(1, 100, 10, 1)
        self.spin_repeats.setToolTip("CV takrorlari (har safar yangi tasodifiy bo'linish): AUC o'rtacha va std. "
                                     "Hisoblash vaqti shunga proporsional (CNN yoqilgan bo'lsa ayniqsa).")
        self.spin_block = _DSpin(0, 1_000_000, 0, 100, 1)
        self.spin_block.setSpecialValueText("0 (avtomatik: variogram)")
        self.spin_block.setSuffix(" m")
        self.spin_block.setToolTip("Blok o'lchami (m). 0 = avtomatik: variogram range'i. Bir blokdagi nuqtalar train va "
                                   "validation orasida hech qachon bo'linmaydi.")
        self.combo_variogram = _Combo()
        self.combo_variogram.setToolTip("Variogram qaysi qatlam bo'yicha baholanadi: auto = barcha raqamli qatlamlar medianasi.")
        self.chk_random_cv = QCheckBox("Random CV benchmark ham bajarilsin")
        self.chk_random_cv.setToolTip("Faqat solishtirish uchun: fazoviy avtokorrelyatsiya tufayli optimistik.")
        self.spin_bootstrap = _Spin(0, 20000, 1000, 100)
        self.spin_bootstrap.setSpecialValueText("0 (CI hisoblanmaydi)")
        self.spin_bootstrap.setToolTip("Blok-bootstrap resample soni (AUC 95% CI uchun). 0 = o'chirilgan (tez).")
        form.addRow("K-fold soni:", self.spin_kfold)
        form.addRow("CV takrorlash soni:", self.spin_repeats)
        form.addRow("Blok o'lchami:", self.spin_block)
        form.addRow("Variogram qatlami:", self.combo_variogram)
        form.addRow("", self.chk_random_cv)
        form.addRow("Bootstrap soni (95% CI):", self.spin_bootstrap)
        return box

    def _group_models(self):
        box = QGroupBox("Modellar va kalibrlash")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        self.model_checks = {}
        for m in MODEL_NAMES:
            title = MODEL_TITLES.get(m, m)
            ok = _available(m)
            cb = QCheckBox(title if ok else f"{title} (o'rnatilmagan!)")
            cb.setEnabled(ok)
            if not ok:
                cb.setChecked(False)
                cb.setToolTip("Kutubxona o'rnatilmagan: " + ("'pip install xgboost'" if m == "XGBoost"
                                                              else "'pip install tensorflow'"))
            elif m == "CNN":
                cb.setToolTip("Qimmat model (patch2d CNN). CNN chiqishi kalibrlanmaydi va neg/pos og'irlik bilan o'qitiladi.")
            self.model_checks[m] = cb
            row.addWidget(cb)
        row.addStretch(1)
        lay.addLayout(row)
        self.cnn_note = _note_label("CNN chiqishi kalibrlanmaydi va neg/pos namuna og'irligi bilan o'qitiladi; ansambl "
                                    "oddiy o'rtacha bo'lgani uchun CNN qiymatlari boshqa modellar bilan bir shkalada "
                                    "bo'lmasligi mumkin.")
        lay.addWidget(self.cnn_note)
        self.chk_calibrate = QCheckBox("Ehtimolliklarni kalibrlash (CalibratedClassifierCV)")
        self.chk_calibrate.setToolTip("RF/SVM/XGBoost chiqishlarini solishtiriladigan shkalaga keltiradi. SVM har doim "
                                      "Platt kalibrlash bilan ishlaydi.")
        lay.addWidget(self.chk_calibrate)
        form = QFormLayout()
        self.combo_cal_method = _Combo()
        for key, text in _CAL_TEXT.items():
            self.combo_cal_method.addItem(text, key)
        self.spin_cal_cv = _Spin(2, 10, 3, 1)
        form.addRow("Kalibrlash usuli:", self.combo_cal_method)
        form.addRow("Kalibrlash CV (calibration_cv):", self.spin_cal_cv)
        lay.addLayout(form)
        return box

    def _group_bg_sensitivity(self):
        box = QGroupBox("Fon sezgirligi tahlili")
        form = QFormLayout(box)
        self.chk_bg_sens = QCheckBox("Fon sezgirligi tahlili bajarilsin")
        self.chk_bg_sens.setToolTip("Spatial CV bir necha marta, har safar boshqa tasodifiy fon tanlovi bilan takrorlanadi: "
                                    "AUC fon tanloviga qanchalik bog'liqligini ko'rsatadi. Vaqtni oshiradi (tuning yo'q).")
        self.spin_bg_draws = _Spin(2, 50, 5, 1)
        self.spin_bg_repeats = _Spin(1, 20, 3, 1)
        form.addRow(self.chk_bg_sens)
        form.addRow("Fon tanlovlari soni (draw):", self.spin_bg_draws)
        form.addRow("Har tanlovda CV takrorlari:", self.spin_bg_repeats)
        return box

    def _group_interpretation(self):
        box = QGroupBox("Talqin (feature importance)")
        form = QFormLayout(box)
        self.chk_perm = QCheckBox("Permutation importance (spatial CV, OOF)")
        self.spin_perm_repeats = _Spin(1, 50, 5, 1)
        self.chk_shap = QCheckBox("SHAP (RF/XGBoost)")
        self.spin_shap_bg = _Spin(5, 5000, 50, 10)
        self.spin_shap_bg.setToolTip("SHAP hisoblanadigan fon qatorlar soni (ko'p bo'lsa sekin).")
        form.addRow(self.chk_perm)
        form.addRow("Permutation takrorlari:", self.spin_perm_repeats)
        form.addRow(self.chk_shap)
        form.addRow("SHAP qatorlari soni:", self.spin_shap_bg)
        return box

    def _group_general(self):
        box = QGroupBox("Umumiy")
        form = QFormLayout(box)
        self.spin_seed = _Spin(0, 2_147_483_647, RANDOM_STATE, 1)
        self.spin_seed.setToolTip("Bir xil ma'lumot + bir xil seed = bir xil natija.")
        self.spin_jobs = _Spin(1, 64, 1, 1)
        self.spin_jobs.setToolTip("Parallel ishchilar soni (RF/XGBoost/permutation).")
        form.addRow("Seed:", self.spin_seed)
        form.addRow("n_jobs:", self.spin_jobs)
        return box

    # ---------------------------------------------------------------- tab 2
    def _build_hyper_tab(self):
        self.hyper_panel = HyperParamPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.hyper_panel)
        return scroll

    # ---------------------------------------------------------------- tab 8
    def _build_models_tab(self):
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(tab)
        lay = QVBoxLayout(tab)

        g1 = QGroupBox("1. Modelni saqlash (bundle)")
        l1 = QVBoxLayout(g1)
        l1.addWidget(_note_label("O'qitilgan yakuniy modellar, FeaturePipeline, giperparametrlar va skalyar metrikalar bitta "
                                 "papkaga (manifest.json + models/) saqlanadi. Yangi yoki bo'sh papka tanlang."))
        self.btn_bundle_save = QPushButton("Modelni saqlash (bundle)...")
        l1.addWidget(self.btn_bundle_save, 0, Qt.AlignLeft)
        lay.addWidget(g1)

        g2 = QGroupBox("2. Saqlangan modelni yuklash")
        l2 = QVBoxLayout(g2)
        l2.addWidget(_note_label("Xavfsizlik: RF/SVM/XGBoost modellari joblib/pickle formatida - faqat ISHONCHLI manbadagi "
                                 "bundle'ni yuklang. Yuklashdan oldin tasdiq so'raladi.", _ERR_STYLE))
        self.btn_bundle_load = QPushButton("Modelni yuklash (bundle)...")
        l2.addWidget(self.btn_bundle_load, 0, Qt.AlignLeft)
        self.bundle_path_label = QLabel("Bundle tanlanmagan.")
        self.bundle_path_label.setWordWrap(True)
        l2.addWidget(self.bundle_path_label)
        self.bundle_info = QPlainTextEdit()
        self.bundle_info.setReadOnly(True)
        self.bundle_info.setMinimumHeight(190)
        self.bundle_info.setPlaceholderText("Yuklangan bundle haqida ma'lumot (manifest) shu yerda ko'rinadi.")
        l2.addWidget(self.bundle_info)
        lay.addWidget(g2)

        g3 = QGroupBox("3. Yangi maydonga qo'llash")
        l3 = QVBoxLayout(g3)
        l3.addWidget(_note_label("Yangi TIFF papkadagi fayl nomlari (kengaytmasiz, katta-kichik harf farqsiz) bundle band "
                                 "nomlariga mos bo'lishi shart; ortiqcha fayllar e'tiborsiz. Sinflash sozlamalari 7-tabdan "
                                 "olinadi. Natija 7-tabda ko'rsatiladi (prospektivlik indeksi, ehtimollik emas)."))
        self.apply_tiff_picker = FolderPicker("Yangi TIFF papka:", label_width=170)
        self.apply_out_picker = FolderPicker("Chiqish papkasi (ixtiyoriy):", label_width=170)
        self.apply_out_picker.setToolTip("Bo'sh bo'lsa GeoTIFF'lar saqlanmaydi (natija faqat ekranda).")
        l3.addWidget(self.apply_tiff_picker)
        l3.addWidget(self.apply_out_picker)
        self.btn_apply = QPushButton("Yangi maydonga qo'llash")
        self.btn_apply.setStyleSheet("font-weight: bold; padding: 6px 14px;")
        l3.addWidget(self.btn_apply, 0, Qt.AlignLeft)
        lay.addWidget(g3)
        lay.addStretch(1)
        return scroll

    # ---------------------------------------------------------------- menyu
    def _build_menu(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&Fayl")
        self.act_save_cfg = QAction("Konfiguratsiyani saqlash...", self)
        self.act_load_cfg = QAction("Konfiguratsiyani yuklash...", self)
        self.act_exit = QAction("Chiqish", self)
        self.act_exit.setShortcut("Ctrl+Q")
        m_file.addAction(self.act_save_cfg)
        m_file.addAction(self.act_load_cfg)
        m_file.addSeparator()
        m_file.addAction(self.act_exit)
        m_help = mb.addMenu("&Yordam")
        self.act_about = QAction("Dastur haqida", self)
        self.act_index_help = QAction("Prospektivlik indeksi haqida", self)
        m_help.addAction(self.act_index_help)
        m_help.addAction(self.act_about)

    # ---------------------------------------------------------------- signal'lar
    def _watch_settings(self):
        """Sozlama vidjetlari o'zgarganda cost hint yangilanadi."""
        for w in self.settings_widget.findChildren(QWidget):
            if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.valueChanged.connect(self._schedule_cost_update)
            elif isinstance(w, QCheckBox):
                w.toggled.connect(self._schedule_cost_update)
            elif isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self._schedule_cost_update)
        self.hyper_panel.changed.connect(self._schedule_cost_update)
        self.map_tab.method_combo.currentIndexChanged.connect(self._schedule_cost_update)

    def _connect_signals(self):
        self._watch_settings()
        self._cost_timer.timeout.connect(self.update_cost_hint)
        self._layer_scan_timer.timeout.connect(self._scan_layers)
        self.picker_tiff.changed.connect(self._on_tiff_changed)
        self.layer_list.itemChanged.connect(self._on_layer_item_changed)
        for cb in list(self.model_checks.values()) + [self.chk_calibrate, self.chk_bg_sens, self.chk_perm, self.chk_shap]:
            cb.toggled.connect(self._update_dependent_widgets)
        # tugmalar (clicked(bool) argumenti slotga o'tmasligi uchun lambda)
        self.btn_train.clicked.connect(lambda _=False: self.start_training())
        self.btn_stop.clicked.connect(lambda _=False: self.stop_current())
        self.btn_save_cfg.clicked.connect(lambda _=False: self.save_config_dialog())
        self.btn_load_cfg.clicked.connect(lambda _=False: self.load_config_dialog())
        self.btn_save_log.clicked.connect(lambda _=False: self._save_log())
        self.btn_clear_log.clicked.connect(lambda _=False: self.log_view.clear_log())
        self.btn_bundle_save.clicked.connect(lambda _=False: self.save_bundle_dialog())
        self.btn_bundle_load.clicked.connect(lambda _=False: self.load_bundle_dialog())
        self.btn_apply.clicked.connect(lambda _=False: self.start_apply())
        # menyu
        self.act_save_cfg.triggered.connect(lambda _=False: self.save_config_dialog())
        self.act_load_cfg.triggered.connect(lambda _=False: self.load_config_dialog())
        self.act_exit.triggered.connect(lambda _=False: self.close())
        self.act_about.triggered.connect(lambda _=False: self.show_about())
        self.act_index_help.triggered.connect(lambda _=False: self.show_index_help())
        # natija tab'lari
        self.map_tab.predict_requested.connect(self.start_prediction)
        self.map_tab.export_requested.connect(lambda: self.start_export())
        self.results_tab.export_requested.connect(lambda: self.start_export())
        self.diag_tab.refresh_requested.connect(self.start_diagnostics)
        for t in self.result_tab_widgets:
            t.message.connect(self.log)

    # ======================================================================
    # Dialoglar (testlarda almashtiriladi)
    # ======================================================================
    def _warn(self, title, text):
        self.last_message = ("warning", title, text)
        QMessageBox.warning(self, title, text)

    def _info(self, title, text):
        self.last_message = ("info", title, text)
        QMessageBox.information(self, title, text)

    def _show_error(self, title, text, detail=None):
        self.last_message = ("critical", title, text)
        box = QMessageBox(QMessageBox.Critical, title, text, QMessageBox.Ok, self)
        if detail:
            box.setDetailedText(detail)
        box.exec_()

    def _confirm_bundle_trust(self):
        """Joblib/pickle xavfsizligi: ishonchli manbadan ekanini tasdiqlash (standart javob - Yo'q)."""
        self.last_message = ("question", "Bundle xavfsizligi", BUNDLE_SECURITY_TEXT)
        r = QMessageBox.question(self, "Bundle xavfsizligi", BUNDLE_SECURITY_TEXT,
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return r == QMessageBox.Yes

    def _confirm_close(self):
        r = QMessageBox.question(self, "Chiqish",
                                 "Hisoblash ketmoqda. Chiqsangiz u bekor qilinadi va natijalar yo'qoladi.\n\nChiqilsinmi?",
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return r == QMessageBox.Yes

    def _ask_directory(self, title, start=""):
        return QFileDialog.getExistingDirectory(self, title, start or "") or ""

    def _ask_save_file(self, title, default, flt):
        return QFileDialog.getSaveFileName(self, title, default, flt)[0] or ""

    def _ask_open_file(self, title, flt):
        return QFileDialog.getOpenFileName(self, title, "", flt)[0] or ""

    def show_about(self):
        self._info("Dastur haqida",
                   f"MPM ML GUI v2 (mpm {MPM_VERSION})\n\nOltin ma'danlashuvi prospektivligini bashoratlash: "
                   f"Random Forest, SVM, XGBoost, CNN va ansambl; spatial block CV, giperparametr qidirish, SHAP.\n\n"
                   f"{INDEX_NOTE}.")

    def show_index_help(self):
        self._info("Prospektivlik indeksi haqida", INDEX_HELP)

    # ======================================================================
    # Log / holat
    # ======================================================================
    def log(self, msg=""):
        """Log oynasiga qator qo'shadi (slotlarda xavfsiz)."""
        try:
            self.log_view.append_line(str(msg))
        except Exception:                                         # noqa: BLE001
            _log.exception("log xatosi")

    def _status(self, text):
        try:
            self.statusBar().showMessage(str(text))
        except Exception:                                         # noqa: BLE001
            pass

    def _report_exception(self, where, exc):
        """_guard: kutilmagan istisno (log + holat qatori; ish ketayotgan bo'lsa busy tiklanmaydi - handlerlar o'zi tiklaydi)."""
        text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        _log.error("MainWindow.%s xatosi:\n%s", where, text)
        self.last_job_error = f"{where}: {type(exc).__name__}: {exc}"
        self.log(f"XATO ({where}): {type(exc).__name__}: {exc}")
        self.log(text)
        self._status(f"Xato: {type(exc).__name__}: {exc}")

    def _on_unhandled_signal(self, text):
        try:
            self.report_unhandled(text)
        except Exception:                                         # noqa: BLE001
            _log.exception("report_unhandled xatosi")

    def report_unhandled(self, text):
        """sys.excepthook: kutilmagan xato logga yoziladi va BIR MARTA dialog ko'rsatiladi (jarayon abort qilinmaydi)."""
        self.last_job_error = str(text).strip().splitlines()[-1] if str(text).strip() else "noma'lum xato"
        self.log("KUTILMAGAN XATO:\n" + str(text))
        self._status("Kutilmagan xato (log'ga qarang)")
        if self._error_box_open:
            return
        self._error_box_open = True

        def _show():
            try:
                self._show_error("Kutilmagan xato", "Dasturda kutilmagan xato yuz berdi (dastur ishlashda davom etadi). "
                                 "Batafsil ma'lumot - log oynasida.", detail=str(text))
            finally:
                self._error_box_open = False
        QTimer.singleShot(0, _show)

    # ======================================================================
    # Konfiguratsiya: _collect_config / _apply_config
    # ======================================================================
    def _class_settings(self):
        """Sinflash sozlamalari (7-tab vidjetlaridan; yaroqsiz chegaralarda oxirgi yaroqli qiymat)."""
        mt = self.map_tab
        method = mt.method_combo.currentData() or "quantile"
        try:
            breaks = parse_breaks(mt.breaks_edit.text())
            self._class_breaks_fallback = list(breaks)
        except ValueError:
            breaks = list(self._class_breaks_fallback)
        n = len(breaks) + 1 if method == "fixed" else int(mt.n_classes_spin.value())
        return method, n, breaks

    def _checked_layers(self):
        return [self.layer_list.item(i).text() for i in range(self.layer_list.count())
                if self.layer_list.item(i).checkState() == Qt.Checked]

    def _collect_config(self):
        """Barcha vidjetlar -> RunConfig (validatsiyasiz; muammolarni `_validate_config` ko'rsatadi)."""
        method, n_classes, breaks = self._class_settings()
        return RunConfig(
            tiff_folder=self.picker_tiff.path(), points_folder=self.picker_points.path(),
            aoi_folder=self.picker_aoi.path(), output_dir=self.picker_output.path(),
            categorical_layers=self._checked_layers(), assume_crs_if_missing=self.chk_assume_crs.isChecked(),
            n_background=int(self.spin_background.value()), min_distance=float(self.spin_min_dist.value()),
            background_strategy=str(self.combo_strategy.currentData()), final_bg_draws=int(self.spin_final_bg.value()),
            n_splits=int(self.spin_kfold.value()), n_repeats=int(self.spin_repeats.value()),
            block_size=float(self.spin_block.value()), variogram_band=str(self.combo_variogram.currentData() or "auto"),
            run_random_cv=self.chk_random_cv.isChecked(), n_bootstrap=int(self.spin_bootstrap.value()),
            use_models={m: bool(cb.isChecked()) for m, cb in self.model_checks.items()},
            hyperparams=self.hyper_panel.get_hyperparams(), calibrate=self.chk_calibrate.isChecked(),
            calibration_method=str(self.combo_cal_method.currentData()), calibration_cv=int(self.spin_cal_cv.value()),
            tuning=self.hyper_panel.get_tuning(),
            bg_sensitivity_enabled=self.chk_bg_sens.isChecked(), bg_sensitivity_draws=int(self.spin_bg_draws.value()),
            bg_sensitivity_repeats=int(self.spin_bg_repeats.value()),
            perm_importance=self.chk_perm.isChecked(), perm_importance_repeats=int(self.spin_perm_repeats.value()),
            shap_enabled=self.chk_shap.isChecked(), shap_max_background=int(self.spin_shap_bg.value()),
            seed=int(self.spin_seed.value()), n_jobs=int(self.spin_jobs.value()),
            class_method=method, n_classes=n_classes, class_breaks=breaks)

    @staticmethod
    def _put(widget, value, label, warns):
        """Spin/double-spin ga qiymat qo'yadi; diapazondan chiqsa ogohlantirish yig'adi."""
        try:
            widget.setValue(value)
        except (TypeError, ValueError, OverflowError):
            warns.append(f"{label}: '{value}' noto'g'ri qiymat, o'zgartirilmadi")
            return
        if abs(float(widget.value()) - float(value)) > 1e-9:
            warns.append(f"{label}: {value} diapazondan chiqdi, {widget.value()} ga o'zgartirildi")

    @staticmethod
    def _put_combo(combo, value, label, warns):
        i = combo.findData(value)
        if i < 0:
            warns.append(f"{label}: '{value}' noma'lum, standart qoldirildi")
            return
        combo.setCurrentIndex(i)

    def _apply_config(self, cfg):
        """RunConfig (yoki lug'at) -> barcha vidjetlar (yuklash/round-trip). Ogohlantirishlar ro'yxatini qaytaradi."""
        if isinstance(cfg, dict):
            cfg = RunConfig.from_dict(cfg)
        warns = []
        self._applying = True
        try:
            for picker, val in ((self.picker_tiff, cfg.tiff_folder), (self.picker_points, cfg.points_folder),
                                (self.picker_aoi, cfg.aoi_folder), (self.picker_output, cfg.output_dir)):
                picker.blockSignals(True)
                picker.setPath(val)
                picker.blockSignals(False)
            self._layer_scan_timer.stop()
            self._scan_id += 1                                    # kutilayotgan skanerlash natijalari e'tiborsiz
            self._populate_layers(cfg.tiff_folder, cfg.categorical_layers)
            self._set_variogram(cfg.variogram_band)
            self.chk_assume_crs.setChecked(bool(cfg.assume_crs_if_missing))
            self._put(self.spin_background, int(cfg.n_background), "Fon nuqtalar soni", warns)
            self._put(self.spin_min_dist, float(cfg.min_distance), "Min. masofa", warns)
            self._put_combo(self.combo_strategy, cfg.background_strategy, "Fon strategiyasi", warns)
            self._put(self.spin_final_bg, int(cfg.final_bg_draws), "Yakuniy fon ansambli", warns)
            self._put(self.spin_kfold, int(cfg.n_splits), "K-fold", warns)
            self._put(self.spin_repeats, int(cfg.n_repeats), "CV takrorlari", warns)
            self._put(self.spin_block, float(cfg.block_size), "Blok o'lchami", warns)
            self.chk_random_cv.setChecked(bool(cfg.run_random_cv))
            self._put(self.spin_bootstrap, int(cfg.n_bootstrap), "Bootstrap soni", warns)
            um = cfg.use_models if isinstance(cfg.use_models, dict) else {}
            for m, cb in self.model_checks.items():
                want = bool(um.get(m, False))
                if want and not cb.isEnabled():
                    warns.append(f"{m}: kutubxona o'rnatilmagan, model o'chirildi")
                    want = False
                cb.setChecked(want)
            self.chk_calibrate.setChecked(bool(cfg.calibrate))
            self._put_combo(self.combo_cal_method, cfg.calibration_method, "Kalibrlash usuli", warns)
            self._put(self.spin_cal_cv, int(cfg.calibration_cv), "Kalibrlash CV", warns)
            self.chk_bg_sens.setChecked(bool(cfg.bg_sensitivity_enabled))
            self._put(self.spin_bg_draws, int(cfg.bg_sensitivity_draws), "Fon sezgirligi: tanlovlar", warns)
            self._put(self.spin_bg_repeats, int(cfg.bg_sensitivity_repeats), "Fon sezgirligi: takrorlar", warns)
            self.chk_perm.setChecked(bool(cfg.perm_importance))
            self._put(self.spin_perm_repeats, int(cfg.perm_importance_repeats), "Permutation takrorlari", warns)
            self.chk_shap.setChecked(bool(cfg.shap_enabled))
            self._put(self.spin_shap_bg, int(cfg.shap_max_background), "SHAP qatorlari", warns)
            self._put(self.spin_seed, int(cfg.seed), "Seed", warns)
            self._put(self.spin_jobs, int(cfg.n_jobs), "n_jobs", warns)
            # giperparametrlar va tuning (panel MERGE qiladi: avval standartga qaytaramiz)
            self.hyper_panel.reset_defaults()
            warns.extend(self.hyper_panel.set_hyperparams(cfg.hyperparams))
            self.hyper_panel.set_tuning(cfg.tuning)
            if cfg.tuning.enabled and cfg.tuning.mode != "nested":
                warns.append("Tuning rejimi 'final': spatial CV bazaviy giperparametrlar bilan baholanadi (nested tuning "
                             "yo'q). Tavsiya: 'nested' rejim.")
            # sinflash (7-tab)
            mt = self.map_tab
            self._put_combo(mt.method_combo, cfg.class_method, "Sinflash usuli", warns)
            self._put(mt.n_classes_spin, int(cfg.n_classes), "Sinflar soni", warns)
            if cfg.class_breaks:
                mt.breaks_edit.setText(format_breaks(cfg.class_breaks))
                self._class_breaks_fallback = [float(b) for b in cfg.class_breaks]
        finally:
            self._applying = False
        self._update_dependent_widgets()
        self._schedule_cost_update()
        return warns

    def _update_dependent_widgets(self, *_):
        """Bog'liq vidjetlarni yoqish/o'chirish (qiymatlar saqlanadi)."""
        cal = self.chk_calibrate.isChecked()
        self.combo_cal_method.setEnabled(cal)
        self.spin_cal_cv.setEnabled(cal)
        on = self.chk_bg_sens.isChecked()
        self.spin_bg_draws.setEnabled(on)
        self.spin_bg_repeats.setEnabled(on)
        self.spin_perm_repeats.setEnabled(self.chk_perm.isChecked())
        self.spin_shap_bg.setEnabled(self.chk_shap.isChecked())
        n_on = sum(1 for cb in self.model_checks.values() if cb.isChecked())
        self.cnn_note.setVisible(self.model_checks["CNN"].isChecked() and n_on > 1)

    def _validate_config(self, cfg):
        """cfg.validate() muammolari + papkalar tekshiruvi. Muammo bo'lsa QMessageBox va False."""
        problems = []
        for label, path in (("TIFF qatlamlar papkasi", cfg.tiff_folder), ("Musbat nuqtalar papkasi", cfg.points_folder),
                            ("Maydon konturi (AOI) papkasi", cfg.aoi_folder)):
            if not path:
                problems.append(f"{label} tanlanmagan.")
            elif not os.path.isdir(path):
                problems.append(f"{label} topilmadi: {path}")
        if cfg.output_dir and os.path.exists(cfg.output_dir) and not os.path.isdir(cfg.output_dir):
            problems.append(f"Chiqish yo'li papka emas: {cfg.output_dir}")
        problems.extend(cfg.validate())
        if problems:
            self._warn("Sozlamalarda xatolar bor", "\n".join(f"- {p}" for p in problems))
            return False
        return True

    # ---------------------------------------------------------------- cost hint
    def _schedule_cost_update(self, *_):
        if not self._applying and hasattr(self, "_cost_timer"):
            self._cost_timer.start()

    def update_cost_hint(self):
        """pipeline.estimate_cost_text(cfg) -> tab 1 va tab 2 dagi taxminiy hisob-kitob matni."""
        try:
            from ..pipeline import estimate_cost_text                   # kechiktirib (og'ir importlar)
            text = estimate_cost_text(self._collect_config())
        except Exception as exc:                                  # noqa: BLE001
            text = f"Narxni baholab bo'lmadi ({type(exc).__name__}: {exc})."
        self.cost_view.setPlainText(text)
        self.hyper_panel.set_cost_hint(text)
        return text

    # ---------------------------------------------------------------- konfiguratsiya fayli
    def save_config_to(self, path):
        config.save_preset(path, cfg=self._collect_config())
        self.log(f"Konfiguratsiya saqlandi: {path}")
        return path

    def load_config_from(self, path):
        """JSON'dan yuklaydi: to'liq RunConfig bo'lsa hamma vidjetlar, faqat giperparametr preset bo'lsa 2-tab. Ogohlantirishlar."""
        res = config.load_preset(path)
        warns = list(res.get("warnings") or [])
        snapshot = self._collect_config()                         # xato bo'lsa vidjetlar yarim yangilangan holda qolmasin
        try:
            if res.get("cfg") is not None:
                warns += self._apply_config(res["cfg"])
            else:
                self.hyper_panel.reset_defaults()
                warns += self.hyper_panel.set_hyperparams(res["hyperparams"])
                if res.get("tuning") is not None:
                    self.hyper_panel.set_tuning(res["tuning"])
                warns.append("Bu fayl faqat giperparametrlar presetidir: 2-tab yangilandi, qolgan sozlamalar o'zgarmadi.")
        except Exception:
            try:
                self._apply_config(snapshot)
            except Exception:                                     # noqa: BLE001
                _log.exception("konfiguratsiyani tiklab bo'lmadi")
            raise
        self.log(f"Konfiguratsiya yuklandi: {path}")
        for w in warns:
            self.log("  Ogohlantirish: " + str(w))
        return warns

    @_guard
    def save_config_dialog(self):
        path = self._ask_save_file("Konfiguratsiyani saqlash", "mpm_config.json", "JSON (*.json)")
        if not path:
            return None
        try:
            return self.save_config_to(path)
        except Exception as exc:                                  # noqa: BLE001
            self._show_error("Konfiguratsiya saqlanmadi", f"{type(exc).__name__}: {exc}")
            return None

    @_guard
    def load_config_dialog(self):
        if self._busy:
            return None
        path = self._ask_open_file("Konfiguratsiyani yuklash", "JSON (*.json)")
        if not path:
            return None
        try:
            warns = self.load_config_from(path)
        except Exception as exc:                                  # noqa: BLE001
            self._show_error("Konfiguratsiya yuklanmadi", f"{type(exc).__name__}: {exc}")
            return None
        if warns:
            self._warn("Konfiguratsiya ogohlantirishlari", "\n".join(str(w) for w in warns))
        return path

    def _save_log(self):
        try:
            p = self.log_view.save_to_file()
            if p:
                self.log(f"Log saqlandi: {p}")
        except Exception as exc:                                  # noqa: BLE001
            self._show_error("Log saqlanmadi", f"{type(exc).__name__}: {exc}")

    # ======================================================================
    # Qatlamlar ro'yxati (kategorik tavsiya)
    # ======================================================================
    def _on_tiff_changed(self, _path=""):
        if not self._applying:
            self._layer_scan_timer.start()

    def _on_layer_item_changed(self, _item):
        self._layers_touched = True
        self._update_layer_summary()

    def _update_layer_summary(self):
        n = self.layer_list.count()
        c = len(self._checked_layers())
        self.layer_summary.setText(f"{n} ta qatlam, shundan {c} tasi kategorik." if n else
                                   "Qatlamlar yo'q (TIFF papkani tanlang).")

    def _populate_layers(self, folder, categorical=()):
        """Ro'yxatni papkadagi fayllar (+ cfg'dagi kategorik nomlar) bilan to'ldiradi; vidjet signallari jim."""
        names = _list_tiff_stems(folder)
        cat = [str(c) for c in (categorical or [])]
        items = names + [c for c in cat if c not in names]
        prev_var = self.combo_variogram.currentData() if hasattr(self, "combo_variogram") else "auto"
        self.layer_list.blockSignals(True)
        try:
            self.layer_list.clear()
            for n in items:
                it = QListWidgetItem(n)
                it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it.setCheckState(Qt.Checked if n in cat else Qt.Unchecked)
                if n not in names:
                    it.setToolTip("Bu nom papkada topilmadi (konfiguratsiyadan olingan).")
                self.layer_list.addItem(it)
        finally:
            self.layer_list.blockSignals(False)
        self._layers_touched = False
        self._refresh_variogram_items(items, prev_var)
        self._update_layer_summary()

    def _refresh_variogram_items(self, band_names, keep="auto"):
        self.combo_variogram.blockSignals(True)
        try:
            self.combo_variogram.clear()
            self.combo_variogram.addItem("auto (barcha raqamli qatlamlar medianasi)", "auto")
            for b in band_names:
                self.combo_variogram.addItem(str(b), str(b))
            i = self.combo_variogram.findData(keep)
            self.combo_variogram.setCurrentIndex(i if i >= 0 else 0)
        finally:
            self.combo_variogram.blockSignals(False)

    def _set_variogram(self, value):
        value = str(value or "auto")
        i = self.combo_variogram.findData(value)
        if i < 0:
            self.combo_variogram.addItem(value, value)
            i = self.combo_variogram.findData(value)
        self.combo_variogram.setCurrentIndex(i)

    @_guard
    def _scan_layers(self):
        """TIFF papka o'zgargach: ro'yxatni yangilaydi va kategorik tavsiyani ishchi oqimda hisoblaydi."""
        if self._closing:
            return
        folder = self.picker_tiff.path()
        self._scan_id += 1
        sid = self._scan_id
        self._populate_layers(folder, [])
        if not _list_tiff_stems(folder):
            if folder:
                self.layer_summary.setText("Bu papkada .tif/.tiff fayllar topilmadi.")
            return

        def fn(w):
            from ..data import find_tiff_files, suggest_categorical_layers
            return {"id": sid, "folder": folder, "categorical": list(suggest_categorical_layers(find_tiff_files(folder)))}

        worker = _CallWorker(fn)
        worker.log_signal.connect(self.log)
        worker.finished_signal.connect(self._on_scan_finished)
        worker.error_signal.connect(self._on_scan_error)
        self._start_thread(worker)

    @_guard
    def _on_scan_finished(self, out):
        if self._closing or not isinstance(out, dict) or out.get("id") != self._scan_id:
            return
        suggested = [c for c in out.get("categorical", [])]
        if not suggested:
            self.log("Kategorik qatlam tavsiyasi: yo'q (barcha qatlamlar raqamli deb olindi).")
            return
        if self._layers_touched:
            self.log("Kategorik qatlam tavsiyasi (qo'lda o'zgartirilgani uchun qo'llanmadi): " + ", ".join(suggested))
            return
        self.layer_list.blockSignals(True)
        try:
            for i in range(self.layer_list.count()):
                it = self.layer_list.item(i)
                if it.text() in suggested:
                    it.setCheckState(Qt.Checked)
        finally:
            self.layer_list.blockSignals(False)
        self._update_layer_summary()
        self.log("Kategorik qatlamlar avtomatik tavsiya qilindi: " + ", ".join(suggested)
                 + " (kerak bo'lmasa belgini olib tashlang).")

    @_guard
    def _on_scan_error(self, text):
        self.log("Qatlamlarni tahlil qilib bo'lmadi (kategorik tavsiya yo'q): " + str(text).split("\n\n")[0])

    # ======================================================================
    # Ishchi oqimlarni boshqarish
    # ======================================================================
    def _start_thread(self, worker):
        with _KEEPER_LOCK:
            _THREAD_KEEPER.add(worker)
        worker.finished.connect(partial(_forget_thread, worker))
        worker.start()

    def is_busy(self):
        return bool(self._busy)

    def _begin_job(self, kind, worker, *, progress=True):
        """Asosiy ishni ishga tushiradi: busy, signal'lar, progress (reset)."""
        self._job_kind = kind
        self._job_worker = worker
        self._stopping = False
        if progress:
            self.progress.reset()
        self._set_busy(True)
        worker.log_signal.connect(self.log)
        if kind in ("export", "autoexport", "bundle_save"):
            worker.progress_signal.connect(self._on_stage_only)
        else:
            worker.progress_signal.connect(self._on_progress)
        worker.finished_signal.connect(partial(self._on_job_finished, kind))
        worker.error_signal.connect(partial(self._on_job_error, kind))
        worker.cancelled_signal.connect(partial(self._on_job_cancelled, kind))
        self._start_thread(worker)

    def _set_busy(self, busy):
        """Bitta joyda: ish ketayotganini belgilaydi va BARCHA tugmalar holatini qayta hisoblaydi (BUG-02)."""
        self._busy = bool(busy)
        if not busy:
            self._stopping = False
        self._update_controls()

    def _end_job(self):
        self._job_worker = None
        self._job_kind = None
        self._set_busy(False)

    def _update_controls(self):
        """Barcha tugma/vidjetlar holati (yagona manba): busy, stopping, natija bor-yo'qligi, tanlangan bundle."""
        busy = self._busy
        has_result = self.training_result is not None
        cancellable = bool(self._job_kind and _JOBS.get(self._job_kind, ("", False))[1])
        self.btn_train.setEnabled(not busy)
        self.btn_stop.setEnabled(busy and cancellable and not self._stopping)
        self.btn_load_cfg.setEnabled(not busy)
        self.act_load_cfg.setEnabled(not busy)
        self.settings_widget.setEnabled(not busy)
        self.hyper_panel.setEnabled(not busy)
        self.btn_bundle_save.setEnabled(has_result and not busy)
        self.btn_bundle_load.setEnabled(not busy)
        self.btn_apply.setEnabled(self._bundle_dir is not None and not busy)
        self.apply_tiff_picker.setEnabled(not busy)
        self.apply_out_picker.setEnabled(not busy)
        for t in self.result_tab_widgets:
            try:
                t.set_busy(busy)
            except Exception:                                     # noqa: BLE001
                _log.exception("tab.set_busy xatosi")

    @_guard
    def _on_progress(self, pct, msg):
        if self._closing:
            return
        self.progress.update_progress(int(pct), str(msg))
        self._status(f"{int(pct)}%  {msg}")

    @_guard
    def _on_stage_only(self, pct, msg):
        if not self._closing and msg:
            self.progress.stage_label.setText(str(msg))
            self._status(str(msg))

    def stop_current(self):
        """Stop: ishchiga cancel() yuboradi va 'to'xtatilmoqda...' holatiga o'tadi (tiklanish cancelled/finished/error'da)."""
        w = self._job_worker
        if w is None or not self._busy or self._stopping:
            return False
        if not _JOBS.get(self._job_kind, ("", False))[1]:
            return False
        self._stopping = True
        w.cancel()
        self.progress.stage_label.setText("To'xtatilmoqda... (joriy qadam tugashini kuting)")
        self._status("To'xtatilmoqda...")
        self.log("To'xtatish so'raldi: hisoblash joriy qadam tugagach to'xtaydi.")
        self._update_controls()
        return True

    # ---- umumiy signal handlerlari (BUG-02)
    def _on_job_finished(self, kind, result):
        if self._closing:
            return
        handler = {"train": self.on_training_finished, "predict": self.on_prediction_finished,
                   "apply": self.on_apply_finished, "export": self.on_export_finished,
                   "autoexport": self.on_export_finished, "diag": self.on_diagnostics_finished,
                   "bundle_save": self.on_bundle_saved}.get(kind)
        try:
            if handler is not None:
                handler(result)
        except Exception as exc:                                  # noqa: BLE001
            self._report_exception(f"on_job_finished[{kind}]", exc)
        finally:
            if self._job_kind == kind:
                self._end_job()

    def _on_job_error(self, kind, text):
        if self._closing:
            return
        try:
            self.on_job_error(text, kind)
        except Exception as exc:                                  # noqa: BLE001
            self._report_exception("on_job_error", exc)
        finally:
            if self._job_kind in (kind, None):
                self._end_job()

    def _on_job_cancelled(self, kind):
        if self._closing:
            return
        try:
            self.on_job_cancelled(kind)
        except Exception as exc:                                  # noqa: BLE001
            self._report_exception("on_job_cancelled", exc)
        finally:
            if self._job_kind in (kind, None):
                self._end_job()

    def on_job_error(self, text, kind=None):
        """Ishchi xatosi: log, progress 'Xato', dialog. (Tugmalarni `_end_job`/`_set_busy(False)` tiklaydi.)"""
        kind = kind or self._job_kind
        text = str(text)
        head = text.split("\n\nTraceback")[0].strip() or "noma'lum xato"
        self.last_job_error = head
        self.log(f"XATO [{kind}]: {head}")
        self.log(text)
        self.progress.stopped("Xato: " + head.splitlines()[0][:150])
        self._status("Xato: " + head.splitlines()[0][:150])
        if kind == "autoexport":
            self.log("Avtomatik eksport o'tkazib yuborildi (xato yuqorida). 'Natijalarni eksport' tugmasi bilan qayta urining.")
            return
        title = _JOBS.get(kind, ("Xato", False))[0]
        self._show_error(title, head, detail=text)

    def on_job_cancelled(self, kind=None):
        self.log("Hisoblash foydalanuvchi tomonidan to'xtatildi.")
        self.progress.stopped("To'xtatildi")
        self._status("To'xtatildi")

    # ======================================================================
    # O'qitish
    # ======================================================================
    @_guard
    def start_training(self):
        """"O'qitish": sozlamalarni yig'adi, tekshiradi va TrainingWorker'ni ishga tushiradi."""
        if self._busy:
            return None
        cfg = self._collect_config()
        if not self._validate_config(cfg):
            return None
        self.log_view.clear_log()
        self.log("O'qitish boshlandi. Chiqish - prospektivlik indeksi (0-1), ehtimollik emas.")
        self.log(METADATA_NOTE)
        worker = TrainingWorker(cfg)
        self._begin_job("train", worker)
        self.tabs.setCurrentIndex(TAB_DATA)
        return worker

    def _step(self, name, fn, *args):
        """Bitta tab'ni to'ldirish: istisno qolgan tab'larni to'xtatmaydi (logga yoziladi)."""
        try:
            fn(*args)
            return True
        except Exception as exc:                                  # noqa: BLE001
            _log.exception("%s to'ldirishda xato", name)
            self.log(f"XATO: '{name}' tab'ini to'ldirib bo'lmadi: {type(exc).__name__}: {exc}")
            return False

    def _fill_tabs(self, result):
        self._step("Ma'lumotlar tahlili", self.diag_tab.set_diagnostics, result.get("diagnostics"),
                   result.get("data_dictionary"))
        self._step("Natijalar", self.results_tab.set_result, result)
        self._step("Spatial CV diagnostika", self.spatial_tab.set_result, result)
        self._step("Feature importance", self.importance_tab.set_result, result)
        self._step("Prognoz xarita", self.map_tab.set_result, result)

    @staticmethod
    def _result_summary(result):
        parts = []
        try:
            parts.append(f"{result.get('n_positive', '?')} musbat + {result.get('n_background', '?')} fon nuqta")
            names = result.get("model_names") or []
            if names:
                parts.append("modellar: " + ", ".join(names))
            m = ((result.get("spatial") or {}).get("metrics") or {}).get(ENSEMBLE_NAME)
            if isinstance(m, dict) and m.get("auc") is not None:
                parts.append(f"spatial CV AUC (ansambl) = {float(m['auc']):.3f}")
            if result.get("block_size"):
                parts.append(f"blok = {float(result['block_size']):,.0f} m")
            nw = len(result.get("warnings") or [])
            if nw:
                parts.append(f"{nw} ta ogohlantirish (5-tabda)")
        except Exception:                                         # noqa: BLE001
            pass
        return "; ".join(parts)

    def on_training_finished(self, result):
        """O'qitish tugadi: barcha tab'larga set_result (har biri alohida try), Natijalar tab'iga o'tish, avto-eksport."""
        if not isinstance(result, dict):
            raise TypeError("O'qitish natijasi lug'at emas.")
        was_stopping = self._stopping
        self.training_result = result
        self._last_prediction = None
        self._fill_tabs(result)
        self.progress.finish("Tayyor")
        self._status("O'qitish tugadi")
        self.log("O'qitish muvaffaqiyatli tugadi: " + self._result_summary(result))
        if was_stopping:
            self.log("Eslatma: Stop bosilgan edi, lekin hisoblash (~95% dan keyin to'xtatib bo'lmaydi) yakunlandi - "
                     "natijalar to'liq va ko'rsatildi.")
        self._set_busy(False)
        self.tabs.setCurrentIndex(TAB_RESULTS)
        out_dir = str((result.get("cfg") or {}).get("output_dir") or "").strip()
        if out_dir:
            self._end_job_keep_for_export()
            self.start_export(auto=True)

    def _end_job_keep_for_export(self):
        """Avto-eksport oldidan joriy ish holatini tozalaydi (yangi ish boshlanishi uchun)."""
        self._job_worker = None
        self._job_kind = None

    # ======================================================================
    # Prognoz
    # ======================================================================
    @_guard
    def start_prediction(self, settings=None):
        """MapTab.predict_requested(dict) yoki qo'lda: PredictionWorker. out_dir bo'sh bo'lsa None beriladi."""
        if self._busy:
            return None
        if self.training_result is None:
            self._warn("Prognoz", "Avval modellarni o'qiting.")
            return None
        if settings is None:
            settings, err = self.map_tab.collect_settings()
            if err:
                self._warn("Sinflash sozlamalari", err)
                return None
        out_dir = (settings.get("out_dir") or "").strip() or None
        worker = PredictionWorker(self.training_result, out_dir, class_method=settings.get("class_method"),
                                  n_classes=settings.get("n_classes"), class_breaks=settings.get("class_breaks"))
        self.log("Prognoz xarita hisoblanmoqda (prospektivlik indeksi, ehtimollik emas)...")
        self._begin_job("predict", worker)
        return worker

    def on_prediction_finished(self, pred):
        self._last_prediction = pred if isinstance(pred, dict) else None
        self.progress.finish("Prognoz tayyor")
        self._status("Prognoz tayyor")
        if isinstance(pred, dict):
            for p in pred.get("saved_paths") or []:
                self.log(f"  Saqlandi: {p}")
        self._step("Prognoz xarita", self.map_tab.set_prediction, pred)
        self.tabs.setCurrentIndex(TAB_MAP)

    # ======================================================================
    # Eksport
    # ======================================================================
    @_guard
    def start_export(self, auto=False, out_dir=None):
        """Natijalarni CSV/JSON/XLSX ga eksport (ExportWorker). auto=True: cfg.output_dir ga, dialogsiz."""
        if self.training_result is None:
            if not auto:
                self._warn("Eksport", "Avval modellarni o'qiting.")
            return None
        if self._busy:
            return None
        cfg_out = str((self.training_result.get("cfg") or {}).get("output_dir") or "").strip()
        if out_dir is None:
            if auto:
                out_dir = cfg_out
            else:
                out_dir = self._ask_directory("Natijalar eksport qilinadigan papka", cfg_out or self.picker_output.path())
        if not out_dir:
            return None
        worker = ExportWorker(self.training_result, out_dir, prediction=self._last_prediction)
        self.log(("Avtomatik eksport: " if auto else "Eksport: ") + str(out_dir))
        self._begin_job("autoexport" if auto else "export", worker, progress=False)
        return worker

    def on_export_finished(self, out):
        paths = list((out or {}).get("paths") or []) if isinstance(out, dict) else []
        d = (out or {}).get("out_dir", "") if isinstance(out, dict) else ""
        self.progress.finish("Eksport tayyor")
        self.log(f"Eksport tugadi: {len(paths)} ta fayl -> {d}")
        self._status(f"Eksport tugadi: {len(paths)} ta fayl")

    # ======================================================================
    # Tez diagnostika (3-tab)
    # ======================================================================
    @_guard
    def start_diagnostics(self):
        """DiagnosticsTab.refresh_requested: TIFF'lardan statistika/korrelyatsiya/VIF (o'qitishsiz)."""
        if self._busy:
            return None
        folder = self.picker_tiff.path()
        if not folder or not os.path.isdir(folder):
            self._warn("Ma'lumotlar tahlili", "Avval 1-tabda TIFF qatlamlar papkasini tanlang.")
            return None
        if not _list_tiff_stems(folder):
            self._warn("Ma'lumotlar tahlili", f"Papkada .tif/.tiff fayllar topilmadi: {folder}")
            return None
        categorical = self._checked_layers()
        assume = self.chk_assume_crs.isChecked()
        seed = int(self.spin_seed.value())

        def fn(w):
            from ..data import (build_data_dictionary, data_diagnostics, find_tiff_files, load_and_align_rasters)
            w._progress(0.02, "Rasterlar o'qilmoqda")
            raster = load_and_align_rasters(find_tiff_files(folder), categorical=categorical, log_fn=w._log,
                                            assume_crs_if_missing=assume, cancel=w.cancel_token)
            w._progress(0.7, "Diagnostika hisoblanmoqda")
            w.cancel_token.raise_if_cancelled()
            diag = data_diagnostics(raster, None, seed=seed)
            # metadata.csv shabloni YOZILMAYDI (yon ta'sirsiz): faqat texnik metadata
            dd = build_data_dictionary(raster.band_names, raster.tech_metadata, {})
            w._progress(1.0, "Tahlil tayyor")
            return {"diagnostics": diag, "data_dictionary": dd}

        self.log("Ma'lumotlar tahlili: TIFF qatlamlar o'qilmoqda (o'qitishsiz)...")
        worker = _CallWorker(fn)
        self._begin_job("diag", worker)
        return worker

    def on_diagnostics_finished(self, out):
        self.progress.finish("Tahlil tayyor")
        self._step("Ma'lumotlar tahlili", self.diag_tab.set_diagnostics, (out or {}).get("diagnostics"),
                   (out or {}).get("data_dictionary"))
        self.log("Ma'lumotlar tahlili tayyor (qo'lda metadata to'ldirilmagan: o'qitishda metadata.csv ishlatiladi).")
        self.tabs.setCurrentIndex(TAB_DIAG)

    # ======================================================================
    # 8-tab: bundle
    # ======================================================================
    @_guard
    def save_bundle_dialog(self):
        if self._busy or self.training_result is None:
            if self.training_result is None:
                self._warn("Modelni saqlash", "Avval modellarni o'qiting.")
            return None
        cfg_out = str((self.training_result.get("cfg") or {}).get("output_dir") or "")
        path = self._ask_directory("Bundle saqlanadigan papka (yangi yoki bo'sh)", cfg_out)
        if not path:
            return None
        return self.save_bundle_to(path)

    def save_bundle_to(self, path):
        """persist.save_bundle ishchi oqimda (faqat skalyar metrikalar)."""
        result = self.training_result
        if result is None or self._busy:
            return None
        path = str(path)

        def fn(w):
            from .. import persist
            raster = result.get("raster")
            persist.save_bundle(
                path, final_models=result["final_models"], pipeline=result["pipeline"],
                hyperparams_used=result.get("final_hyperparams") or {}, cfg_dict=result.get("cfg") or {},
                metrics_summary=scalar_metrics((result.get("spatial") or {}).get("metrics")),
                thresholds=result.get("thresholds"), block_size=result.get("block_size"),
                crs_epsg=int(getattr(raster, "crs_epsg", TARGET_EPSG)),
                notes="Chiqish - prospektivlik indeksi (0-1), ehtimollik emas.", log_fn=w._log)
            return {"directory": path}

        self.log(f"Model saqlanmoqda: {path}")
        worker = _CallWorker(fn)
        self._begin_job("bundle_save", worker, progress=False)
        return worker

    def on_bundle_saved(self, out):
        d = (out or {}).get("directory")
        self.progress.finish("Model saqlandi")
        self.log(f"Model (bundle) saqlandi: {d}")
        self._status(f"Model saqlandi: {d}")
        if d:
            # o'zimiz yaratgan bundle: ishonchli, qo'llash uchun tanlanadi
            self._select_bundle(d)

    def _read_manifest(self, path):
        mp = os.path.join(str(path), "manifest.json")
        if not os.path.isfile(mp):
            raise FileNotFoundError(f"'{mp}' topilmadi: bu MPM bundle papkasi emas yoki saqlash yakunlanmagan.")
        with open(mp, "r", encoding="utf-8") as f:
            m = json.load(f)
        if not isinstance(m, dict) or "bundle_version" not in m:
            raise ValueError("manifest.json MPM bundle manifest'i emas (bundle_version yo'q).")
        return m

    def _select_bundle(self, path):
        manifest = self._read_manifest(path)
        info = format_manifest(manifest)                          # avval hammasini tayyorlaymiz: xato bo'lsa holat o'zgarmaydi
        self._bundle_dir = str(path)
        self._bundle_manifest = manifest
        self.bundle_path_label.setText(f"Tanlangan bundle: {path}")
        self.bundle_info.setPlainText(info)
        self._update_controls()
        return manifest

    @_guard
    def load_bundle_dialog(self, path=None):
        """Modelni yuklash: AVVAL xavfsizlik tasdig'i, keyin papka tanlash va manifest ko'rsatish (pickle hali yuklanmaydi)."""
        if self._busy:
            return None
        if not self._confirm_bundle_trust():
            self.log("Bundle yuklash bekor qilindi (ishonchli manba tasdiqlanmadi).")
            return None
        if path is None:
            path = self._ask_directory("Bundle papkasini tanlang", self._bundle_dir or "")
        if not path:
            return None
        try:
            manifest = self._select_bundle(path)
        except Exception as exc:                                  # noqa: BLE001
            self._show_error("Bundle yuklanmadi", f"{type(exc).__name__}: {exc}")
            return None
        self.log(f"Bundle tanlandi: {path} ({len(manifest.get('model_names') or [])} model). Modellar (joblib/pickle) "
                 f"qo'llash paytida yuklanadi - faqat ishonchli manbadagi bundle'ni qo'llang.")
        return path

    @_guard
    def start_apply(self):
        """Tanlangan bundle'ni yangi TIFF papkaga qo'llash (ApplyBundleWorker); natija 7-tabda."""
        if self._busy:
            return None
        if not self._bundle_dir:
            self._warn("Yangi maydonga qo'llash", "Avval bundle'ni yuklang (8-tab, 2-bo'lim).")
            return None
        tiff = self.apply_tiff_picker.path()
        if not tiff or not os.path.isdir(tiff):
            self._warn("Yangi maydonga qo'llash", "Yangi TIFF papkani tanlang (mavjud papka bo'lishi kerak).")
            return None
        out = self.apply_out_picker.path()
        if out and os.path.exists(out) and not os.path.isdir(out):
            self._warn("Yangi maydonga qo'llash", f"Chiqish yo'li papka emas: {out}")
            return None
        settings, err = self.map_tab.collect_settings()
        if err:
            self._warn("Sinflash sozlamalari", err)
            return None
        worker = ApplyBundleWorker(self._bundle_dir, tiff, out or None, class_method=settings["class_method"],
                                   n_classes=settings["n_classes"], class_breaks=settings["class_breaks"],
                                   assume_crs_if_missing=self.chk_assume_crs.isChecked())
        self.log(f"Bundle yangi maydonga qo'llanmoqda: {tiff}")
        self._begin_job("apply", worker)
        return worker

    def on_apply_finished(self, out):
        self.progress.finish("Qo'llash tayyor")
        self._status("Bundle qo'llandi")
        if isinstance(out, dict):
            for p in out.get("saved_paths") or []:
                self.log(f"  Saqlandi: {p}")
            ign = out.get("ignored_files") or []
            if ign:
                self.log(f"  Ortiqcha TIFF'lar e'tiborsiz qoldirildi: {len(ign)} ta")
        self._step("Prognoz xarita", self.map_tab.set_prediction, out)
        self.log("Bundle yangi maydonga qo'llandi (7-tabda). Success-rate/Konlar_* ustunlari yo'q: yangi maydonda konlar noma'lum.")
        self.tabs.setCurrentIndex(TAB_MAP)

    # ======================================================================
    # Tozalash / yopish
    # ======================================================================
    def clear_results(self):
        """Natijalarni tozalaydi (tab'lar bo'sh holatga)."""
        self.training_result = None
        self._last_prediction = None
        for t in self.result_tab_widgets:
            self._step("tozalash", t.clear)
        self._update_controls()

    def wait_idle(self, timeout_ms=120000):
        """Barcha ishchi oqimlar tugaguncha hodisalarni qayta ishlaydi (testlar/skriptlar uchun). True = tugadi."""
        loop = QEventLoop()
        timer = QTimer()
        timer.setInterval(15)
        state = {"left": int(timeout_ms)}

        def tick():
            state["left"] -= 15
            if (not self._busy and not _THREAD_KEEPER) or state["left"] <= 0:
                loop.quit()
        timer.timeout.connect(tick)
        timer.start()
        if self._busy or _THREAD_KEEPER:
            loop.exec_()
        timer.stop()
        QApplication.processEvents()
        return not self._busy and not _THREAD_KEEPER

    def _shutdown_threads(self, timeout_ms=30000):
        """Barcha ishchilarni bekor qiladi va tugashini kutadi. True = hammasi to'xtadi."""
        with _KEEPER_LOCK:
            workers = list(_THREAD_KEEPER)
        for w in workers:
            try:
                w.cancel()
            except Exception:                                     # noqa: BLE001
                pass
        ok = True
        for w in workers:
            try:
                if not w.wait(int(timeout_ms)):
                    ok = False
            except RuntimeError:
                pass
        return ok

    def closeEvent(self, event):                                  # noqa: N802
        job_running = self._job_worker is not None and self._busy
        try:
            job_running = job_running and self._job_worker.isRunning()
        except RuntimeError:
            job_running = False
        if job_running and not self._confirm_close():
            event.ignore()
            return
        self._closing = True
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ok = self._shutdown_threads()
        finally:
            QApplication.restoreOverrideCursor()
        if not ok:
            self._closing = False
            self._warn("Chiqish", "Hisoblash hali to'xtamadi (joriy qadam tugashini kuting) - oyna yopilmadi. "
                                  "Birozdan so'ng qayta urining.")
            event.ignore()
            return
        event.accept()


# ---------------------------------------------------------------------------
# Global excepthook va main()
# ---------------------------------------------------------------------------
_HOOK_STATE = {"window": None, "prev": None}
LAST_WINDOW = None


def _excepthook(exc_type, exc, tb):
    """Global istisno ushlagich: logga/dialogga chiqaradi, jarayonni abort QILMAYDI (PyQt5 standart hook'da qFatal)."""
    if issubclass(exc_type, KeyboardInterrupt):
        prev = _HOOK_STATE.get("prev") or sys.__excepthook__
        prev(exc_type, exc, tb)
        return
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    try:
        _log.error("Ushlanmagan istisno:\n%s", text)
        sys.stderr.write(text)
    except Exception:                                             # noqa: BLE001
        pass
    win = _HOOK_STATE.get("window")
    if win is not None:
        try:
            app = QApplication.instance()
            if app is None or QThread.currentThread() is app.thread():
                win.report_unhandled(text)
            else:                                                 # boshqa oqim: vidjetlarga tegmaymiz, GUI oqimiga yuboramiz
                win._unhandled_signal.emit(text)
        except Exception:                                         # noqa: BLE001
            pass


def install_excepthook(window=None):
    """sys.excepthook (va threading.excepthook) ni o'rnatadi; oldingisini qaytaradi (tiklash uchun)."""
    prev = sys.excepthook
    if prev is not _excepthook:
        _HOOK_STATE["prev"] = prev
    _HOOK_STATE["window"] = window
    sys.excepthook = _excepthook

    def _thread_hook(args):
        if args.exc_type is SystemExit:
            return
        _excepthook(args.exc_type, args.exc_value, args.exc_traceback)
    threading.excepthook = _thread_hook
    return prev


def _warmup_imports():
    """Og'ir modullarni (pipeline: sklearn/rasterio/geopandas) fon oqimida oldindan yuklaydi: birinchi cost hint GUI'ni qotirmasin."""
    if "mpm.pipeline" in sys.modules:
        return

    def _run():
        try:
            import importlib
            importlib.import_module("mpm.pipeline")
        except Exception:                                         # noqa: BLE001
            pass
    threading.Thread(target=_run, name="mpm-warmup", daemon=True).start()


def main(argv=None):
    """GUI'ni ishga tushiradi. QApplication shu yerda yaratiladi (mavjud bo'lsa qayta ishlatiladi). Chiqish kodini qaytaradi."""
    global LAST_WINDOW
    argv = list(sys.argv if argv is None else argv)
    app = QApplication.instance()
    if app is None:
        try:
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        except Exception:                                         # noqa: BLE001
            pass
        app = QApplication(argv)
    app.setApplicationName("MPM ML GUI")
    _warmup_imports()
    win = MainWindow()
    LAST_WINDOW = win
    install_excepthook(win)
    win.show()
    return int(app.exec_())


if __name__ == "__main__":                                        # pragma: no cover
    sys.exit(main())
