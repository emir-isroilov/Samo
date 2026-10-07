# -*- coding: utf-8 -*-
"""
Barcha modellar uchun yagona interfeys (ModelWrapper).

MUHIM shartnoma:
  * Wrapper ICHIDA kerakli preprocessing (StandardScaler va h.k.) bajariladi, shuning uchun
    chaqiruvchi har doim XOM (scaling'siz) feature'larni beradi.
  * `X` shakli (n, n_features), `patches` shakli (n, w, w, n_features) (faqat patch-CNN uchun).
  * predict_proba_pos(...) musbat sinf (1) ehtimolligi (n,) float64 qaytaradi.
  * fit(...) o'zini (self) qaytaradi.
"""
from __future__ import annotations

import abc
import copy
import os

import numpy as np

from .common import RANDOM_STATE


class ModelWrapper(abc.ABC):
    name = ""                 # "RandomForest" | "SVM" | "XGBoost" | "CNN"
    input_kind = "tabular"    # "tabular" | "patch"

    def __init__(self, name, params, seed=RANDOM_STATE, n_jobs=1):
        self.name = name
        self.params = copy.deepcopy(dict(params))
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.classes_ = np.array([0, 1])
        self.is_fitted = False
        self.fit_info = {}    # ixtiyoriy diagnostika (masalan CNN: epochs_run, val_loss)

    # ---- majburiy
    @abc.abstractmethod
    def fit(self, X, y, patches=None, cancel=None, log_fn=None):
        """O'qitadi, self qaytaradi."""

    @abc.abstractmethod
    def predict_proba_pos(self, X=None, patches=None, batch_size=8192):
        """Musbat sinf ehtimolligi, shakli (n,)."""

    # ---- umumiy
    def predict_proba(self, X=None, patches=None, batch_size=8192):
        p = np.asarray(self.predict_proba_pos(X, patches=patches, batch_size=batch_size), dtype=float)
        return np.column_stack([1.0 - p, p])

    def tree_model(self):
        """SHAP TreeExplainer uchun O'QITILGAN daraxt modeli (RF/XGB) yoki None."""
        return None

    def importance_mdi(self):
        """Daraxt modellari uchun MDI/gain importance (p,) yoki None."""
        return None

    def get_params(self):
        return copy.deepcopy(self.params)

    # ---- saqlash/yuklash (CNN override qiladi)
    def save(self, directory):
        import joblib
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "model.joblib")
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, directory):
        import joblib
        return joblib.load(os.path.join(directory, "model.joblib"))
