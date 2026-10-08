# -*- coding: utf-8 -*-
"""
Talqin (interpretatsiya): out-of-fold permutation importance (AUC pasayishi), SHAP va MDI/gain.

Legacy kodga nisbatan tuzatishlar:
  * BUG-01: yangi `shap` RF uchun (n, p, 2) qaytaradi (eskisi - ro'yxat, XGBoost - (n, p)); barchasi
    `normalize_shap_values` orqali (n, p) ga keltiriladi, 3D massiv hech qachon "qolib ketmaydi";
  * BUG-03: MDI/SHAP o'qitilgan bazaviy daraxt modelidan (`ModelWrapper.tree_model()` / `importance_mdi()`),
    kalibrlash ichidagi o'qitilmagan `estimator`dan EMAS;
  * BUG-12: SHAP yakuniy (to'liq ma'lumotda o'qitilgan) bazaviy modelga nisbatan hisoblanadi;
  * sklearn `permutation_importance` ishlatilmaydi (eskirgan/cheklangan): o'zimiznikida patch-model
    (kanalni butun patchlar bo'yicha aralashtirish) ham qo'llab-quvvatlanadi va takrorlar bitta
    bashoratga jamlanadi (tez).

`shap` modul darajasida import QILINMAYDI (common.get_shap() funksiya ichida).
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

from .common import RANDOM_STATE, CancelledError, check_cancel, get_shap, import_error, noop_log

__all__ = [
    "permutation_importance_auc", "summarize_perm_importance", "normalize_shap_values",
    "compute_shap_summary", "mdi_importance",
]

SHAP_MODELS = ("RandomForest", "XGBoost")          # TreeExplainer faqat daraxt modellari uchun
_SHAP_UNITS = {"RandomForest": "ehtimollik", "XGBoost": "log-odds"}
_PERM_MAX_ELEMS = 1 << 24                          # bitta bashoratdagi taxminiy element soni (~64 MB float32)


# ---------------------------------------------------------------------------
# SHAP: shakllarni normalizatsiya qilish (BUG-01)
# ---------------------------------------------------------------------------
def _as_float_array(a, what="SHAP qiymatlari"):
    try:
        return np.asarray(a, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{what} raqamli massiv bo'lishi kerak ({e})") from None


def _pick_class(n_classes, class_index):
    """Sinf o'qi uzunligi n_classes bo'lganda tanlanadigan indeks. Bitta chiqish - doim 0."""
    if n_classes == 1:
        return 0
    if not -n_classes <= class_index < n_classes:
        raise ValueError(f"class_index={class_index} sinflar soniga ({n_classes}) mos emas.")
    return class_index


def normalize_shap_values(sv, class_index=1, *, n_samples=None, n_features=None):
    """SHAP chiqishini musbat sinf uchun (n, p) float64 massivga keltiradi.

    Qamrab oladi: ro'yxat/kortej `[sinf0 (n,p), sinf1 (n,p)]` (eski shap), (n, p) (XGBoost, bitta chiqish),
    (n, p, 2) (yangi shap RF), (2, n, p), (n, p, 1), (1, n, p), `shap.Explanation` (.values).
    Yaroqsiz shakl (0/1/4+ o'lcham, sinf o'qi 1-2 emas, bo'sh ro'yxat, raqamli bo'lmagan) => ValueError.

    n_samples/n_features berilsa (compute_shap_summary beradi): (n, p, C) va (C, n, p) orasidagi
    noaniqlik (masalan p=2) aniq hal qilinadi va natija shakli shu qiymatlarga tengligi tekshiriladi."""
    vals = getattr(sv, "values", None)
    if vals is not None and not callable(vals) and not isinstance(sv, np.ndarray):    # shap.Explanation
        sv = vals
    if isinstance(sv, (list, tuple)):
        if len(sv) == 0:
            raise ValueError("SHAP qiymatlari ro'yxati bo'sh.")
        mats = [_as_float_array(s) for s in sv]
        if all(m.ndim == 2 for m in mats):                                # sinflar ro'yxati
            if len({m.shape for m in mats}) != 1:
                raise ValueError(f"SHAP sinf massivlari shakli bir xil emas: {[m.shape for m in mats]}")
            out = mats[_pick_class(len(mats), class_index)]
        elif all(m.ndim == 1 for m in mats):                              # oddiy ichma-ich ro'yxat (n, p)
            out = _as_float_array(sv)
        else:
            raise ValueError("SHAP ro'yxati elementlari (n, p) massivlar bo'lishi kerak.")
    else:
        a = _as_float_array(sv)
        if a.ndim == 2:
            out = a
        elif a.ndim == 3:
            out = _squeeze_class_axis(a, class_index, n_samples, n_features)
        else:
            raise ValueError(f"SHAP qiymatlari 2D (n, p) yoki 3D (n, p, sinf) bo'lishi kerak, shakl: {a.shape}")
    if n_samples is not None and n_features is not None and out.shape != (int(n_samples), int(n_features)):
        raise ValueError(f"SHAP natijasi shakli {out.shape}, kutilgan ({int(n_samples)}, {int(n_features)}).")
    return np.ascontiguousarray(out, dtype=np.float64)


def _squeeze_class_axis(a, class_index, n_samples, n_features):
    s0, s1, s2 = a.shape
    layout = None
    if n_samples is not None and n_features is not None:
        n, p = int(n_samples), int(n_features)
        if (s0, s1) == (n, p):
            layout = "npc"
        elif (s1, s2) == (n, p):
            layout = "cnp"
    elif s2 in (1, 2):
        layout = "npc"                      # yangi shap konvensiyasi: sinf oxirgi o'qda
    elif s0 in (1, 2):
        layout = "cnp"
    if layout is None:
        raise ValueError(f"SHAP 3D massivi shakli {a.shape} tanilmadi: (n, p, 2), (n, p, 1), (2, n, p) yoki "
                         "(1, n, p) kutilgan.")
    if layout == "npc":
        return a[:, :, _pick_class(s2, class_index)]
    return a[_pick_class(s0, class_index)]


def _expected_value(ev, class_index=1):
    """explainer.expected_value (skalyar yoki sinflar bo'yicha) -> musbat sinf uchun float | None."""
    try:
        arr = np.asarray(ev, dtype=np.float64).ravel()
        if arr.size == 0:
            return None
        return float(arr[_pick_class(arr.size, class_index)])
    except (TypeError, ValueError):
        return None


def _mean_abs(sv):
    """Ustunlar bo'yicha |SHAP| o'rtachasi (p,); butunlay NaN ustun => NaN, RuntimeWarning'siz (BUG-14)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # "Mean of empty slice"
        return np.nanmean(np.abs(sv), axis=0)


def _make_explainer(shap, tree_model):
    """TreeExplainer (standart tree_path_dependent, xom chiqish); eski/yangi shap versiyalariga chidamli."""
    try:
        return shap.TreeExplainer(tree_model, model_output="raw")
    except Exception:
        return shap.TreeExplainer(tree_model)


def _shap_values(explainer, Xb, log, name):
    """shap_values; additivity tekshiruvi float yaxlitlashi tufayli yiqilsa, tekshiruvsiz qayta uriniladi."""
    try:
        return explainer.shap_values(Xb)
    except Exception as e:
        log(f"  {name}: SHAP additivity tekshiruvi/chaqiruvi xato berdi ({type(e).__name__}: {e}); "
            "tekshiruvsiz qayta uriniladi.")
        return explainer.shap_values(Xb, check_additivity=False)


def compute_shap_summary(final_models, X, feature_names, max_background=50, seed=RANDOM_STATE, log_fn=None,
                         cancel=None):
    """RF/XGBoost yakuniy modellari uchun SHAP (TreeExplainer, musbat sinf, tree_path_dependent/raw).

    final_models: {name: [ModelWrapper, ...]}; har model uchun birinchi fon-draw modelining `tree_model()`i
    (kalibrlangan bo'lsa ham to'liq ma'lumotda o'qitilgan aniq bazaviy daraxt). X: (n, p) xom feature'lar;
    fon sifatida `seed` bo'yicha tasodifiy `max_background` ta qator olinadi (hamma model uchun bir xil).
    Qaytaradi: {model: {"mean_abs_shap": (p,), "shap_values": (k, p), "X_background": (k, p),
    "feature_names": [...], "expected_value": float|None, "units": "ehtimollik"|"log-odds"}} yoki None
    (shap yo'q / mos model yo'q / hisoblanmadi). RF qiymatlari ehtimollik, XGBoost - log-odds birligida
    (modellararo magnituda to'g'ridan-to'g'ri solishtirilmaydi)."""
    log = log_fn or noop_log
    shap = get_shap()
    if shap is None:
        err = import_error("shap")           # modul bor, lekin import yiqilgan bo'lsa haqiqiy sabab ko'rsatiladi
        if err:
            log(f"  SHAP kutubxonasi import qilib bo'lmadi ({err}) - SHAP tahlili o'tkazib yuborildi.")
        else:
            log("  SHAP kutubxonasi o'rnatilmagan ('pip install shap') - SHAP tahlili o'tkazib yuborildi.")
        return None
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X 2 o'lchamli (n, p) bo'lishi kerak, berilgan shakl: {X.shape}")
    feature_names = [str(f) for f in feature_names]
    if len(feature_names) != X.shape[1]:
        raise ValueError(f"feature_names soni ({len(feature_names)}) X ustunlari soniga ({X.shape[1]}) teng emas.")
    names = [m for m in SHAP_MODELS if final_models and final_models.get(m)]
    if not names:
        log("  SHAP: RandomForest/XGBoost yakuniy modeli yo'q - o'tkazib yuborildi.")
        return None
    n = X.shape[0]
    if n == 0:
        log("  SHAP: X bo'sh - o'tkazib yuborildi.")
        return None
    k = int(min(max(1, int(max_background)), n))
    idx = np.sort(np.random.default_rng(int(seed) % (2 ** 32 - 1)).choice(n, size=k, replace=False))
    X_bg = X[idx]
    X_in = np.asarray(X_bg, dtype=np.float32)       # daraxt modellari float32 bilan o'qitilgan/ishlaydi

    out = {}
    for name in names:
        check_cancel(cancel)
        wrappers = final_models[name]
        wrapper = wrappers[0] if isinstance(wrappers, (list, tuple)) else wrappers
        try:
            tree = wrapper.tree_model()
            if tree is None:
                log(f"  OGOHLANTIRISH: {name} uchun o'qitilgan daraxt modeli topilmadi, SHAP o'tkazib yuborildi.")
                continue
            explainer = _make_explainer(shap, tree)
            raw = _shap_values(explainer, X_in, log, name)
            sv = normalize_shap_values(raw, 1, n_samples=k, n_features=X.shape[1])
            if not np.isfinite(sv).all():
                log(f"  OGOHLANTIRISH: {name} SHAP qiymatlarida NaN/inf bor.")
            out[name] = {
                "mean_abs_shap": _mean_abs(sv), "shap_values": sv, "X_background": X_bg,
                "feature_names": list(feature_names),
                "expected_value": _expected_value(getattr(explainer, "expected_value", None)),
                "units": _SHAP_UNITS[name],
            }
            log(f"  SHAP hisoblandi: {name} ({k} ta fon nuqta; birinchi fon-ansambl modeli, "
                f"birlik: {_SHAP_UNITS[name]}).")
        except CancelledError:
            raise
        except Exception as e:
            log(f"  OGOHLANTIRISH: {name} uchun SHAP hisoblanmadi ({type(e).__name__}: {e}).")
    return out or None


# ---------------------------------------------------------------------------
# MDI (BUG-03)
# ---------------------------------------------------------------------------
def mdi_importance(model):
    """O'qitilgan bazaviy daraxt modelidan MDI/gain importance (p,) yoki None (SVM/CNN/model yo'q)."""
    fn = getattr(model, "importance_mdi", None)
    if fn is None:
        return None
    imp = fn()
    return None if imp is None else np.asarray(imp, dtype=np.float64)


# ---------------------------------------------------------------------------
# Permutation importance (AUC pasayishi)
# ---------------------------------------------------------------------------
def _check_binary(y):
    y = np.asarray(y).ravel()
    if y.size and not np.isin(y, (0, 1)).all():
        raise ValueError("y faqat 0/1 qiymatlardan iborat bo'lishi kerak.")
    return y.astype(np.int64)


def _safe_auc(y, p):
    """AUC; bashorat chekli bo'lmasa yoki bir sinfli bo'lsa NaN."""
    if not np.isfinite(p).all():
        return np.nan
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return np.nan


def permutation_importance_auc(model, X, y, *, patches=None, n_repeats=5, seed=RANDOM_STATE,
                               batch_size=8192, cancel=None, progress_fn=None):
    """Har feature (patch-modelda - kanal) j uchun AUC pasayishining o'rtachasi: (p,) float64.

    Tabular model: X ning j-ustuni namunalar bo'yicha aralashtiriladi. `model.input_kind == "patch"`:
    patches (n, w, w, p) ning j-kanali BUTUN patchlar bilan (barcha namunalarda bir xil permutatsiya,
    patch ichidagi fazoviy tuzilma saqlanadi) aralashtiriladi, X ishlatilmaydi. y bir sinfli bo'lsa
    (AUC aniqlanmagan) NaN massiv. Kirish massivlari o'zgartirilmaydi. Tezlik: n_repeats ta takror
    bitta (katta) predict_proba_pos(batch_size=...) chaqiruviga jamlanadi."""
    y = _check_binary(y)
    n = y.size
    n_repeats = int(n_repeats)
    if n_repeats < 1:
        raise ValueError(f"n_repeats kamida 1 bo'lishi kerak, berilgan: {n_repeats}")
    bs = int(batch_size)
    if bs < 1:
        raise ValueError(f"batch_size kamida 1 bo'lishi kerak, berilgan: {batch_size}")
    is_patch = getattr(model, "input_kind", "tabular") == "patch"
    if is_patch:
        if patches is None:
            raise ValueError("Patch modeli uchun patches (n, w, w, p) kerak.")
        arr = np.asarray(patches, dtype=np.float32)
        if arr.ndim != 4:
            raise ValueError(f"patches 4 o'lchamli (n, w, w, p) bo'lishi kerak, berilgan shakl: {arr.shape}")
    else:
        if X is None:
            raise ValueError("Tabular model uchun X (n, p) kerak.")
        arr = np.asarray(X)
        if arr.dtype.kind != "f":
            arr = arr.astype(np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X 2 o'lchamli (n, p) bo'lishi kerak, berilgan shakl: {arr.shape}")
    if arr.shape[0] != n:
        raise ValueError(f"Kirish qatorlari ({arr.shape[0]}) y uzunligiga ({n}) teng emas.")
    p = arr.shape[-1]
    if n < 2 or len(np.unique(y)) < 2:
        return np.full(p, np.nan)

    def predict(a):
        if is_patch:
            return np.asarray(model.predict_proba_pos(None, patches=a, batch_size=bs), dtype=np.float64)
        return np.asarray(model.predict_proba_pos(a, batch_size=bs), dtype=np.float64)

    base = _safe_auc(y, predict(arr))
    if not np.isfinite(base):
        return np.full(p, np.nan)
    g = int(max(1, min(n_repeats, _PERM_MAX_ELEMS // max(arr.size, 1))))      # bitta chaqiruvdagi takrorlar
    buf = np.concatenate([arr] * g, axis=0)
    rng = np.random.default_rng(int(seed) % (2 ** 32 - 1))
    out = np.full(p, np.nan)
    for j in range(p):
        check_cancel(cancel)
        perms = [rng.permutation(n) for _ in range(n_repeats)]
        col = arr[..., j]
        aucs = []
        for r0 in range(0, n_repeats, g):
            grp = perms[r0:r0 + g]
            for t, perm in enumerate(grp):                       # j-kanal: butun patch/qiymat bo'yicha aralashtirish
                buf[t * n:(t + 1) * n, ..., j] = arr[perm, ..., j]
            pred = predict(buf[:len(grp) * n])
            for t in range(len(grp)):
                aucs.append(_safe_auc(y, pred[t * n:(t + 1) * n]))
            for t in range(len(grp)):
                buf[t * n:(t + 1) * n, ..., j] = col
        good = [a for a in aucs if np.isfinite(a)]
        if good:
            out[j] = float(np.mean([base - a for a in good]))      # AUC pasayishlari o'rtachasi
        if progress_fn is not None:
            progress_fn((j + 1) / p, f"Permutation importance: {j + 1}/{p}")
    return out


def summarize_perm_importance(per_fold):
    """Fold'lar bo'yicha importance massivlarini jamlaydi: {"mean" (p,), "std" (p,), "per_fold" (nf, p),
    "n_folds"} (nanmean/nanstd, ddof=0; butunlay NaN ustun => NaN, ogohlantirishsiz). None elementlar
    tashlanadi; "n_valid_folds" - kamida bitta chekli qiymatli fold'lar soni."""
    rows = [np.asarray(a, dtype=np.float64).ravel() for a in (per_fold or []) if a is not None]
    if not rows:
        return {"mean": np.zeros(0), "std": np.zeros(0), "per_fold": np.zeros((0, 0)), "n_folds": 0,
                "n_valid_folds": 0}
    if len({r.size for r in rows}) != 1:
        raise ValueError(f"Fold importance massivlari uzunligi bir xil emas: {sorted({r.size for r in rows})}")
    arr = np.vstack(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # "Mean of empty slice" / "Degrees of freedom"
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
    return {"mean": mean, "std": std, "per_fold": arr, "n_folds": int(arr.shape[0]),
            "n_valid_folds": int(np.isfinite(arr).any(axis=1).sum())}
