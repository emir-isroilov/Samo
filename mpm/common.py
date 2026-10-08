# -*- coding: utf-8 -*-
"""
Umumiy yordamchilar: konstantalar, bekor qilish (cancel) tokeni, progress yordamchisi,
ixtiyoriy kutubxonalarni (xgboost, tensorflow, shap) kechiktirib yuklash, global seed.

Bu modul boshqa mpm modullariga bog'liq emas (faqat numpy + standart kutubxona).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import re
import sys
import threading

import numpy as np

# ---------------------------------------------------------------------------
# Konstantalar
# ---------------------------------------------------------------------------
TARGET_EPSG = 28411          # Pulkovo 1942 / Gauss-Kruger zone 11
RANDOM_STATE = 42
NODATA_OUT = -9999.0         # chiqish GeoTIFF'lardagi nodata qiymati
ENSEMBLE_NAME = "Ensemble (soft-voting)"
MODEL_ORDER = ("RandomForest", "SVM", "XGBoost", "CNN")


def noop_log(msg=""):
    """log_fn=None bo'lganda ishlatiladigan bo'sh log."""
    return None


def default_n_jobs():
    """Standart parallel ishchilar soni: CPU soni, lekin ko'pi bilan 4."""
    try:
        return max(1, min(4, os.cpu_count() or 1))
    except Exception:
        return 1


def safe_name(name):
    """Fayl nomi uchun xavfsiz ko'rinish: 'Ensemble (soft-voting)' -> 'Ensemble_soft_voting'."""
    s = str(name).replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    return re.sub(r"[^0-9A-Za-z_.]+", "_", s)


# ---------------------------------------------------------------------------
# Bekor qilish (Stop tugmasi) - hamkorlikdagi (cooperative) mexanizm
# ---------------------------------------------------------------------------
class CancelledError(Exception):
    """Foydalanuvchi hisoblashni to'xtatganda ko'tariladi."""


class CancelToken:
    """threading.Event asosidagi bekor qilish tokeni. Uzoq tsikllar
    `token.raise_if_cancelled()` ni muntazam chaqirishi kerak."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    @property
    def is_cancelled(self):
        return self._event.is_set()

    def raise_if_cancelled(self):
        if self._event.is_set():
            raise CancelledError("Hisoblash foydalanuvchi tomonidan to'xtatildi.")


def check_cancel(cancel):
    """cancel None bo'lishi mumkin."""
    if cancel is not None:
        cancel.raise_if_cancelled()


def sub_progress(progress_fn, lo, hi):
    """progress_fn(frac, msg) ni [lo, hi] oralig'iga proporsional siquvchi yangi callable qaytaradi.
    progress_fn None bo'lsa, bo'sh funksiya qaytaradi."""
    if progress_fn is None:
        return lambda frac=0.0, msg="": None

    def _fn(frac=0.0, msg=""):
        frac = 0.0 if frac is None else float(min(1.0, max(0.0, frac)))
        progress_fn(lo + (hi - lo) * frac, msg)

    return _fn


# ---------------------------------------------------------------------------
# Global seed
# ---------------------------------------------------------------------------
def set_global_seed(seed=RANDOM_STATE):
    """numpy, python random va (agar yuklangan bo'lsa) TensorFlow seedlarini bir qiymatga o'rnatadi.
    TensorFlow'ni o'zi import QILMAYDI (faqat allaqachon yuklangan bo'lsa)."""
    import random as _random
    seed = int(seed)
    _random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    tf = sys.modules.get("tensorflow")
    if tf is not None:
        try:
            tf.random.set_seed(seed)
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Ixtiyoriy kutubxonalar - kechiktirib (lazy) yuklanadi, GUI tez ochilishi uchun
# ---------------------------------------------------------------------------
_OPTIONAL_CACHE = {}
_OPTIONAL_ERRORS = {}        # modname -> "XatoTuri: matn" (modul topilgan, lekin import YIQILGAN bo'lsa)
_ERR_MAX = 300


def _has_spec(modname):
    try:
        return importlib.util.find_spec(modname) is not None
    except Exception:
        return False


def xgboost_available():
    return _has_spec("xgboost")


def tf_available():
    return _has_spec("tensorflow")


def shap_available():
    return _has_spec("shap")


def import_error(modname):
    """Modul topilgan (find_spec), lekin import YIQILGAN bo'lsa xato matni ("XatoTuri: xabar"), aks holda None
    (modul o'rnatilmagan yoki hali import qilinmagan yoki import muvaffaqiyatli). Foydalanuvchi "o'rnatilmagan" emas,
    haqiqiy sababni (masalan buzilgan TensorFlow/shap) ko'rishi uchun."""
    return _OPTIONAL_ERRORS.get(modname)


def _lazy_import(modname):
    """Modulni bir marta import qiladi; muvaffaqiyatsiz bo'lsa None (keshlanadi), xato matni esa
    import_error(modname) orqali olinadi."""
    if modname in _OPTIONAL_CACHE:
        return _OPTIONAL_CACHE[modname]
    mod = None
    _OPTIONAL_ERRORS.pop(modname, None)
    if _has_spec(modname):
        try:
            if modname == "tensorflow":
                os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import warnings
            with warnings.catch_warnings():
                # uchinchi tomon import-vaqti ogohlantirishlari (-W error muhitida ham) importni yiqitmasin
                warnings.simplefilter("ignore")
                mod = importlib.import_module(modname)
        except Exception as e:
            mod = None
            msg = " ".join(f"{type(e).__name__}: {e}".split())
            _OPTIONAL_ERRORS[modname] = msg if len(msg) <= _ERR_MAX else msg[:_ERR_MAX - 3] + "..."
    _OPTIONAL_CACHE[modname] = mod
    return mod


def get_xgboost():
    """xgboost moduli yoki None."""
    return _lazy_import("xgboost")


def get_tf():
    """tensorflow moduli yoki None (birinchi chaqiruvda import qilinadi - bir necha soniya)."""
    tf = _lazy_import("tensorflow")
    if tf is not None:
        try:
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass
    return tf


def get_shap():
    """shap moduli yoki None."""
    return _lazy_import("shap")


def collect_versions():
    """Reproduktivlik uchun asosiy kutubxonalar versiyalari (import qilmasdan)."""
    from importlib import metadata
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "tensorflow",
                "shap", "rasterio", "geopandas", "shapely", "matplotlib", "PyQt5",
                "joblib", "openpyxl"):
        try:
            out[pkg] = metadata.version(pkg)
        except Exception:
            out[pkg] = None
    return out
