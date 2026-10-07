# -*- coding: utf-8 -*-
"""
QThread ishchilar (PyQt5): o'qitish, prognoz, bundle'ni qo'llash va eksport.

Hammasida: cancel() (CancelToken), signal'lar:
  log_signal(str), progress_signal(int 0-100, str), finished_signal(object -> dict), error_signal(str),
  cancelled_signal().
run() sinxron chaqirilganda ham ishlaydi (testlar uchun): signal'lar bevosita chaqiriladi.
Xato matni traceback bilan beriladi; CancelledError => cancelled_signal.

Eslatma: finished_signal `pyqtSignal(object)` (dict uzatiladi): `pyqtSignal(dict)` lug'atni QVariantMap'ga
aylantirib nusxalaydi, natijada katta massivlar/RasterStack identifikatsiyasi yo'qoladi va xotira ikki baravar bo'ladi.
"""
from __future__ import annotations

import traceback

from PyQt5.QtCore import QThread, pyqtSignal

from .common import CancelledError, CancelToken, sub_progress


class _BaseWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    cancelled_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._cancel = CancelToken()

    # ---- bekor qilish
    def cancel(self):
        """Stop tugmasi: hamkorlikdagi bekor qilish (hisoblash keyingi tekshiruv nuqtasida to'xtaydi)."""
        self._cancel.cancel()

    @property
    def cancel_token(self):
        return self._cancel

    @property
    def is_cancelled(self):
        return self._cancel.is_cancelled

    # ---- pipeline uchun callback'lar
    def _log(self, msg=""):
        self.log_signal.emit(str(msg))

    def _progress(self, frac=0.0, msg=""):
        try:
            pct = int(round(100.0 * min(1.0, max(0.0, float(frac)))))
        except (TypeError, ValueError):
            pct = 0
        self.progress_signal.emit(pct, str(msg))

    # ---- ish
    def _work(self):
        raise NotImplementedError

    def run(self):
        try:
            result = self._work()
        except CancelledError:
            self.cancelled_signal.emit()
        except Exception as e:                      # GUI oqimini yiqitmaslik uchun hamma xato signalga
            self.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")
        else:
            self.finished_signal.emit(result)


class TrainingWorker(_BaseWorker):
    """run_training(cfg) -> TrainingResult."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def _work(self):
        from .pipeline import run_training
        return run_training(self.cfg, log_fn=self._log, progress_fn=self._progress, cancel=self._cancel)


class PredictionWorker(_BaseWorker):
    """run_prediction(result, ...) -> PredictionOutput. class_* None bo'lsa result["cfg"] dan olinadi."""

    def __init__(self, result, out_dir, class_method=None, n_classes=None, class_breaks=None, batch_size=8192):
        super().__init__()
        self.result = result
        self.out_dir = out_dir
        self.class_method = class_method
        self.n_classes = n_classes
        self.class_breaks = class_breaks
        self.batch_size = batch_size

    def _work(self):
        from .pipeline import run_prediction
        return run_prediction(self.result, out_dir=self.out_dir or None, class_method=self.class_method,
                              n_classes=self.n_classes, class_breaks=self.class_breaks,
                              batch_size=self.batch_size, log_fn=self._log, progress_fn=self._progress,
                              cancel=self._cancel)


class ApplyBundleWorker(_BaseWorker):
    """Saqlangan bundle'ni yangi maydonga qo'llaydi (persist.load_bundle + apply_bundle).

    Natija: apply_bundle lug'ati, lekin katta "raster" (RasterStack) OLIB TASHLANADI (Qt orqali uzatilganda xotira
    ikki baravar bo'lmasligi uchun); o'rniga yengil "transform", "profile", "pixel_size", "crs_epsg", "band_names"
    va "manifest", "bundle_dir" qo'shiladi."""

    def __init__(self, bundle_dir, tiff_folder, out_dir=None, class_method="quantile", n_classes=5,
                 class_breaks=None, batch_size=8192, assume_crs_if_missing=True):
        super().__init__()
        self.bundle_dir = bundle_dir
        self.tiff_folder = tiff_folder
        self.out_dir = out_dir
        self.class_method = class_method
        self.n_classes = n_classes
        self.class_breaks = class_breaks
        self.batch_size = batch_size
        self.assume_crs_if_missing = assume_crs_if_missing

    def _work(self):
        from .persist import apply_bundle, load_bundle
        bundle = load_bundle(self.bundle_dir, log_fn=self._log)
        self._progress(0.02, "Bundle yuklandi")
        # apply_bundle o'z progress'ini 0 dan boshlaydi: [0.02, 1] oralig'iga siqamiz (progress orqaga qaytmasin)
        out = apply_bundle(bundle, self.tiff_folder, out_dir=self.out_dir or None, batch_size=self.batch_size,
                           assume_crs_if_missing=self.assume_crs_if_missing, class_method=self.class_method,
                           n_classes=self.n_classes, class_breaks=self.class_breaks, log_fn=self._log,
                           progress_fn=sub_progress(self._progress, 0.02, 1.0), cancel=self._cancel)
        raster = out.pop("raster", None)
        if raster is not None:
            out.update({"transform": raster.transform, "profile": raster.profile, "pixel_size": raster.pixel_size,
                        "crs_epsg": raster.crs_epsg, "band_names": list(raster.band_names)})
        out["manifest"] = bundle["manifest"]
        out["bundle_dir"] = str(self.bundle_dir)
        return out


class ExportWorker(_BaseWorker):
    """export_results(result, out_dir, prediction) -> {"paths": [...], "out_dir": out_dir}."""

    def __init__(self, result, out_dir, prediction=None):
        super().__init__()
        self.result = result
        self.out_dir = out_dir
        self.prediction = prediction

    def _work(self):
        from .pipeline import export_results
        self._progress(0.0, "Eksport boshlandi")
        paths = export_results(self.result, self.out_dir, prediction=self.prediction, log_fn=self._log)
        self._progress(1.0, "Eksport tayyor")
        return {"paths": list(paths), "out_dir": self.out_dir}
