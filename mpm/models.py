# -*- coding: utf-8 -*-
"""
Modellar: make_model() fabrikasi va SklearnModel (RandomForest / SVM / XGBoost) wrapper'i.

Asosiy qoidalar:
  * giperparametrlar YAGONA manbadan (config.PARAM_SPECS) olinadi va HAR BIRI modelga uzatiladi
    (jim e'tiborsiz qoldirilmaydi; noma'lum parametr - ValueError) - BUG-15;
  * SVM har doim CalibratedClassifierCV(Pipeline(StandardScaler, SVC), ensemble=False) ichida
    (SVC ning eskirgan ehtimollik parametri HECH QACHON berilmaydi) - BUG-06;
  * kalibrlash ensemble=False: bazaviy model TO'LIQ train ma'lumotida o'qitiladi, SHAP/MDI uchun
    tree_model() shu o'qitilgan modelni qaytaradi - BUG-03, BUG-12;
  * eff_cv = min(calibration_cv, eng kichik sinf soni); < 2 bo'lsa kalibrlashsiz (crash yo'q) - BUG-13.
"""
from __future__ import annotations

import inspect

import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from .base import ModelWrapper
from .common import RANDOM_STATE, check_cancel, get_xgboost, noop_log, tf_available
from .config import MODEL_NAMES, PARAM_SPECS

SKLEARN_MODELS = ("RandomForest", "SVM", "XGBoost")
CALIBRATION_METHODS = ("sigmoid", "isotonic")
_SCAN_ROWS = 1 << 16        # NaN/inf tekshiruvi bo'lagi (qatorlar)


# ---------------------------------------------------------------------------
# Parametrlarni tekshirish va xaritalash (PARAM_SPECS -> sklearn/xgboost kwargs)
# ---------------------------------------------------------------------------
def _is_none_token(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in ("", "none"))


def _coerce_value(model, spec, value):
    """Qiymatni spec turiga keltiradi; noto'g'ri bo'lsa aniq ValueError."""
    where = f"{model}.{spec.name}"
    bad = ValueError(f"{where}: noto'g'ri qiymat {value!r} (tur: {spec.kind})")
    if spec.kind in ("optint", "optfloat") and _is_none_token(value):
        return None
    if spec.kind == "bool":
        if isinstance(value, (bool, np.bool_, int, np.integer)):
            return bool(value)
        raise bad
    if spec.kind == "choice":
        v = str(value)
        if v not in spec.choices:
            raise ValueError(f"{where}: '{v}' ruxsat etilmagan; mumkin: {list(spec.choices)}")
        return v
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise bad from None
    if not np.isfinite(f):
        raise bad
    return int(round(f)) if spec.kind in ("int", "optint") else f


def normalize_params(model, params):
    """To'liq (PARAM_SPECS dagi barcha kalitlar) va tur jihatidan tozalangan lug'at.
    Yetishmaganlari standart bilan to'ldiriladi; noma'lum kalit - ValueError."""
    specs = {s.name: s for s in PARAM_SPECS[model]}
    params = dict(params or {})
    unknown = sorted(set(params) - set(specs))
    if unknown:
        raise ValueError(f"{model}: noma'lum parametr(lar) {unknown}; mumkin: {sorted(specs)}")
    return {n: _coerce_value(model, s, params[n] if n in params else s.default) for n, s in specs.items()}


def _none_if(v, token="none"):
    return None if v == token else v


def _ensure_consumed(model, left):
    if left:
        raise RuntimeError(f"{model}: PARAM_SPECS parametrlari modelga xaritalanmagan: {sorted(left)}")


_TREE_CHUNK = 25


def _sum_trees(trees, X, n_classes):
    acc = np.zeros((X.shape[0], n_classes), dtype=np.float64)
    for t in trees:
        acc += t.predict_proba(X, check_input=False)
    return acc


class _DeterministicRF(RandomForestClassifier):
    """RandomForestClassifier: predict_proba yig'indisi daraxtlarning qat'iy tartibida (25 talik bo'laklar)
    hisoblanadi, shuning uchun natija n_jobs va thread tartibiga bog'liq emas (bit-darajasida reproduktiv)."""

    def predict_proba(self, X):
        check_is_fitted(self)
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X shakli (n, {self.n_features_in_}) bo'lishi kerak, berilgan {X.shape}")
        X = np.ascontiguousarray(X, dtype=np.float32)
        trees = self.estimators_
        chunks = [trees[i:i + _TREE_CHUNK] for i in range(0, len(trees), _TREE_CHUNK)]
        parts = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_sum_trees)(c, X, self.n_classes_) for c in chunks)
        acc = parts[0]
        for part in parts[1:]:
            acc = acc + part
        return acc / len(trees)


def _make_rf(p, seed, n_jobs):
    q = dict(p)
    bootstrap = q.pop("bootstrap")
    max_samples = q.pop("max_samples")
    est = _DeterministicRF(
        n_estimators=q.pop("n_estimators"), criterion=q.pop("criterion"), max_depth=q.pop("max_depth"),
        min_samples_split=q.pop("min_samples_split"), min_samples_leaf=q.pop("min_samples_leaf"),
        max_features=_none_if(q.pop("max_features"), "all"), bootstrap=bootstrap,
        max_samples=max_samples if bootstrap else None,          # max_samples faqat bootstrap=True bilan
        class_weight=_none_if(q.pop("class_weight")), ccp_alpha=q.pop("ccp_alpha"),
        random_state=seed, n_jobs=n_jobs)
    _ensure_consumed("RandomForest", q)
    return est


def _make_svm(p):
    q = dict(p)
    gamma = q.pop("gamma")
    svc = SVC(kernel=q.pop("kernel"), C=q.pop("C"), gamma="scale" if gamma is None else gamma,
              degree=q.pop("degree"), class_weight=_none_if(q.pop("class_weight")))
    _ensure_consumed("SVM", q)
    return Pipeline([("scaler", StandardScaler()), ("svc", svc)])


def _make_xgb(p, seed, n_jobs, scale_pos_weight):
    xgb = get_xgboost()
    q = dict(p)
    q.pop("scale_pos_weight")        # fit() paytida (neg/pos yoki foydalanuvchi qiymati) beriladi
    est = xgb.XGBClassifier(
        n_estimators=q.pop("n_estimators"), learning_rate=q.pop("learning_rate"), max_depth=q.pop("max_depth"),
        min_child_weight=q.pop("min_child_weight"), subsample=q.pop("subsample"),
        colsample_bytree=q.pop("colsample_bytree"), gamma=q.pop("gamma"), reg_alpha=q.pop("reg_alpha"),
        reg_lambda=q.pop("reg_lambda"), tree_method=q.pop("tree_method"), scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=seed, n_jobs=n_jobs, verbosity=0)
    _ensure_consumed("XGBoost", q)
    return est


# ---------------------------------------------------------------------------
# SVM zaxirasi: kalibrlash imkonsiz bo'lganda
# ---------------------------------------------------------------------------
class _SigmoidMinMaxSVM:
    """Kalibrlash imkonsiz (juda kam musbat nuqta) bo'lganda SVM uchun zaxira.
    decision_function train oralig'i (max-min) ga normallanib sigmoid'dan o'tkaziladi: chegara (d=0) -> 0.5,
    tartib saqlanadi. Bu kalibrlangan ehtimollik EMAS, faqat 0-1 oralig'idagi monoton indeks."""
    K = 8.0

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.span_ = 1.0

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        d = np.asarray(self.pipeline.decision_function(X), dtype=np.float64)
        span = float(d.max() - d.min())
        self.span_ = span if span > 1e-12 else 1.0
        return self

    def predict_proba(self, X):
        d = np.asarray(self.pipeline.decision_function(X), dtype=np.float64)
        p = expit(self.K * d / self.span_)
        return np.column_stack([1.0 - p, p])


# ---------------------------------------------------------------------------
# SklearnModel
# ---------------------------------------------------------------------------
class SklearnModel(ModelWrapper):
    """RF / SVM / XGBoost uchun yagona wrapper (XOM feature'lar beriladi; SVM scaling ichida)."""
    input_kind = "tabular"

    def __init__(self, name, params, *, calibrate=True, calibration_method="sigmoid", calibration_cv=3,
                 seed=RANDOM_STATE, n_jobs=1, n_features=None):
        if name not in SKLEARN_MODELS:
            raise ValueError(f"SklearnModel faqat {SKLEARN_MODELS} uchun; berilgan: {name!r}")
        if calibration_method not in CALIBRATION_METHODS:
            raise ValueError(f"Kalibrlash usuli noto'g'ri: {calibration_method!r}; mumkin: {CALIBRATION_METHODS}")
        if name == "XGBoost" and get_xgboost() is None:
            raise RuntimeError("XGBoost o'rnatilmagan: 'pip install xgboost' bilan o'rnating yoki modelni o'chiring.")
        super().__init__(name, normalize_params(name, params), seed=seed, n_jobs=n_jobs)
        self.calibrate = bool(calibrate)
        self.calibration_method = calibration_method
        self.calibration_cv = int(calibration_cv)
        self.n_features = None if n_features is None else int(n_features)
        self.n_features_ = None
        self.model_ = None

    @property
    def _rs(self):
        """sklearn/xgboost random_state uchun 0..2**32-2 oralig'idagi seed (CNN bilan bir xil qoida):
        katta yoki manfiy seed RF'ni yiqitmasin."""
        return self.seed % (2 ** 32 - 1)

    # ---- kirishni tekshirish
    def _check_X(self, X, fitting=False):
        """X ni (n, p) raqamli massivga keltiradi (dtype saqlanadi, nusxa yo'q) va NaN/inf ni tekshiradi.
        fitting=True: faqat e'lon qilingan n_features bilan solishtiriladi (qayta fit boshqa p bilan mumkin)."""
        if X is None:
            raise ValueError(f"{self.name}: X berilmagan (tabular model X (n, n_features) talab qiladi).")
        try:
            X = np.asarray(X)
            if X.dtype.kind not in "biuf":
                X = X.astype(np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{self.name}: X raqamli bo'lishi kerak ({e})") from None
        if X.ndim != 2:
            raise ValueError(f"{self.name}: X 2 o'lchamli (n, n_features) bo'lishi kerak, berilgan shakl {X.shape}")
        expected = self.n_features if fitting or self.n_features_ is None else self.n_features_
        if expected is not None and X.shape[1] != expected:
            raise ValueError(f"{self.name}: feature'lar soni {X.shape[1]}, kutilgan {expected}")
        bad = 0
        if X.dtype.kind == "f":          # bo'laklab: katta X uchun to'liq o'lchamli vaqtinchalik massiv yo'q
            for i in range(0, len(X), _SCAN_ROWS):
                c = X[i:i + _SCAN_ROWS]
                bad += int(c.size - np.count_nonzero(np.isfinite(c)))
        if bad:
            raise ValueError(f"{self.name}: X da {bad} ta NaN/inf qiymat bor; faqat chekli qiymatlar qabul qilinadi.")
        return X

    def _check_y(self, y, n):
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != n:
            raise ValueError(f"{self.name}: y shakli (n,) bo'lishi kerak (n={n}), berilgan {y.shape}")
        try:
            yf = y.astype(np.float64)
        except (TypeError, ValueError):
            raise ValueError(f"{self.name}: y faqat 0/1 qiymatlardan iborat bo'lishi kerak") from None
        if not np.all(np.isin(yf, (0.0, 1.0))):
            raise ValueError(f"{self.name}: y faqat 0/1 qiymatlardan iborat bo'lishi kerak")
        return yf.astype(np.int64)

    # ---- o'qitish
    def _build_base(self, n_pos, n_neg, log):
        if self.name == "RandomForest":
            if not self.params["bootstrap"] and self.params["max_samples"] is not None:
                log("  Ogohlantirish: RandomForest max_samples faqat bootstrap yoqilganda ishlaydi - e'tiborsiz qoldirildi.")
            return _make_rf(self.params, self._rs, self.n_jobs), None
        if self.name == "SVM":
            return _make_svm(self.params), None
        spw = self.params["scale_pos_weight"]
        spw = float(n_neg) / float(n_pos) if spw is None else float(spw)
        return _make_xgb(self.params, self._rs, self.n_jobs, spw), spw

    def _calibrator(self, est, eff_cv, method):
        # shuffle=True: ichki fold'lar kirish tartibiga bog'liq bo'lmasin (seed bilan reproduktiv)
        cv = StratifiedKFold(n_splits=eff_cv, shuffle=True, random_state=self._rs)
        return CalibratedClassifierCV(estimator=est, method=method, cv=cv, ensemble=False)

    def fit(self, X, y, patches=None, cancel=None, log_fn=None):
        log = log_fn or noop_log
        X = np.asarray(self._check_X(X, fitting=True), dtype=np.float64)
        y = self._check_y(y, len(X))
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos < 1 or n_neg < 1:
            raise ValueError(f"{self.name}: o'qitish uchun ikkala sinf (0 va 1) kerak: musbat={n_pos}, fon={n_neg}")
        check_cancel(cancel)

        self.is_fitted = False
        self.n_features_ = X.shape[1]
        est, spw = self._build_base(n_pos, n_neg, log)
        info = {"n_train": int(len(y)), "n_pos": n_pos, "n_neg": n_neg, "calibration": "none",
                "calibration_cv": None, "calibration_reason": "kalibrlash o'chirilgan"}
        if spw is not None:
            info["scale_pos_weight"] = spw

        is_svm = self.name == "SVM"
        model = None
        if self.calibrate or is_svm:
            method = self.calibration_method if self.calibrate else "sigmoid"
            eff_cv = min(self.calibration_cv, n_pos, n_neg)
            if eff_cv < 2:
                reason = (f"kalibrlash uchun har sinfdan kamida 2 ta namuna va calibration_cv>=2 kerak "
                          f"(musbat={n_pos}, fon={n_neg}, calibration_cv={self.calibration_cv})")
            else:
                reason = None
                try:
                    model = self._calibrator(est, eff_cv, method).fit(X, y)
                except Exception as e:      # kalibrlash muvaffaqiyatsiz - crash o'rniga zaxira yo'li
                    model = None
                    reason = f"kalibrlash xatosi ({type(e).__name__}: {e})"
            if model is not None:
                info["calibration"] = method if self.calibrate else "svm_platt_only"
                info["calibration_cv"] = int(eff_cv)
                info["calibration_reason"] = ("" if self.calibrate
                                              else "SVM ehtimollik chiqishi uchun Platt kalibrlash majburiy")
            else:
                info["calibration"] = "skipped"
                info["calibration_reason"] = reason
                log(f"  Ogohlantirish: {self.name} kalibrlanmadi - {reason}.")
                if is_svm:
                    info["svm_fallback"] = "sigmoid_minmax"
                    log("  SVM: decision_function -> sigmoid(min-max) zaxira indeksi ishlatiladi "
                        "(kalibrlangan ehtimollik emas).")
        check_cancel(cancel)
        if model is None:
            model = _SigmoidMinMaxSVM(est).fit(X, y) if is_svm else est.fit(X, y)

        self.model_ = model
        self.fit_info = info
        self.is_fitted = True
        return self

    # ---- bashorat
    def predict_proba_pos(self, X=None, patches=None, batch_size=8192):
        if not self.is_fitted or self.model_ is None:
            raise RuntimeError(f"{self.name}: model hali o'qitilmagan - avval fit() chaqiring.")
        X = self._check_X(X)
        bs = int(batch_size)
        if bs < 1:
            raise ValueError(f"batch_size >= 1 bo'lishi kerak, berilgan {batch_size}")
        out = np.empty(len(X), dtype=np.float64)
        for i in range(0, len(X), bs):      # float64 ga o'tkazish bo'lak-bo'lak (katta X uchun xotira tejaladi)
            out[i:i + bs] = self.model_.predict_proba(np.asarray(X[i:i + bs], dtype=np.float64))[:, 1]
        return np.clip(out, 0.0, 1.0)

    # ---- talqin
    def tree_model(self):
        """O'qitilgan RF/XGB (kalibrlangan bo'lsa ichidagi bazaviy model); SVM => None."""
        if not self.is_fitted or self.name == "SVM":
            return None
        if isinstance(self.model_, CalibratedClassifierCV):
            return self.model_.calibrated_classifiers_[0].estimator
        return self.model_

    def importance_mdi(self):
        tm = self.tree_model()
        imp = getattr(tm, "feature_importances_", None) if tm is not None else None
        if imp is None:
            return None
        return np.nan_to_num(np.asarray(imp, dtype=np.float64))


# ---------------------------------------------------------------------------
# Fabrika
# ---------------------------------------------------------------------------
def _make_cnn(params, seed, n_jobs, n_features):
    """CNN lazy import: TensorFlow/mpm.cnn yo'q bo'lsa RuntimeError (aniq xabar)."""
    if not tf_available():
        raise RuntimeError("TensorFlow o'rnatilmagan: CNN modeli uchun 'pip install tensorflow' kerak.")
    try:
        from .cnn import CNNModel
    except ImportError as e:
        raise RuntimeError(f"CNN modulini (mpm.cnn) yuklab bo'lmadi: {e}") from e
    sig = inspect.signature(CNNModel.__init__).parameters
    has_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.values())
    kwargs = {"seed": seed, "n_jobs": n_jobs, "n_features": n_features}
    kwargs = {k: v for k, v in kwargs.items() if has_kw or k in sig}
    args = ("CNN", params) if "name" in sig else (params,)
    return CNNModel(*args, **kwargs)


def make_model(name, params, *, calibrate=True, calibration_method="sigmoid", calibration_cv=3,
               n_jobs=1, seed=RANDOM_STATE, n_features=None):
    """name in MODEL_NAMES -> ModelWrapper. "CNN" => cnn.CNNModel (lazy). Mavjud bo'lmagan kutubxona => RuntimeError."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Noma'lum model: {name!r}; mumkin: {list(MODEL_NAMES)}")
    if name == "CNN":
        return _make_cnn(params, seed, n_jobs, n_features)
    return SklearnModel(name, params, calibrate=calibrate, calibration_method=calibration_method,
                        calibration_cv=calibration_cv, seed=seed, n_jobs=n_jobs, n_features=n_features)
