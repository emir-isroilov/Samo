# -*- coding: utf-8 -*-
"""
Cross-validation: run_cv (spatial / random, nested tuning, OOF permutation importance), compute_metrics
(blok-bootstrap CI), metrics_dataframe, youden_threshold va run_background_sensitivity.

Legacy kodga nisbatan tuzatishlar:
  BUG-05  spatial fold'larda har validation fold'ida musbat nuqta (spatial.stratified_group_splits);
  BUG-08  AUC CI - blok-bootstrap (mean OOF ehtimollik ustida), repeat-std alohida kalit;
  BUG-15  har model uchun BIR XIL hp lug'ati (CV va final bir manbadan; nested tuning bo'lsa fold'ning best_params'i);
  ENH-02  nested tuning (tashqi fold'ning train qismida tune_model), ENH-06 progress + ETA + cancel.

OOF kafolati: har nuqta har repeat'da aynan bir marta validation'ga tushadi, OOF massivlari NaN-siz.
Modellardagi xato jim yutilmaydi: aniq log + RuntimeError.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)

from . import spatial
from .common import (ENSEMBLE_NAME, RANDOM_STATE, CancelledError, check_cancel, noop_log, set_global_seed)
from .config import BACKGROUND_STRATEGIES, MODEL_NAMES, PARAM_SPECS, TuningConfig, validate_hyperparams
from .models import make_model

__all__ = ["run_cv", "compute_metrics", "metrics_dataframe", "youden_threshold", "run_background_sensitivity"]

_CV_MODES = ("spatial", "random")
_PATCH_CACHE_MAX = 3          # window bo'yicha patch keshi: eng ko'pi bilan shuncha oyna o'lchami
_TUNE_PROGRESS_SHARE = 0.95   # fold oralig'ining tuning'ga ajratilgan ulushi (qolgani fold yakuniga qoladi)
_STAT_KEYS = ("auc", "pr_auc", "balanced_accuracy", "f1", "brier", "balanced_accuracy_youden", "f1_youden")
_MIN_BOOT_WARN = 30           # CI uchun yaroqli bootstrap resample soni shundan kam bo'lsa ogohlantiriladi


# ---------------------------------------------------------------------------
# Umumiy yordamchilar
# ---------------------------------------------------------------------------
def _fmt_eta(sec):
    sec = int(round(max(0.0, float(sec))))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _check_model_names(model_names):
    names = [str(m) for m in (model_names or [])]
    if not names:
        raise ValueError("model_names bo'sh: kamida bitta model kerak.")
    unknown = [m for m in names if m not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"Noma'lum model(lar): {unknown}; mumkin: {list(MODEL_NAMES)}")
    if len(set(names)) != len(names):
        raise ValueError(f"model_names takrorlanmasligi kerak: {names}")
    return names


def _prefixed_log(log, prefix):
    """Model/tuning xabarlariga fold kontekstini qo'shadi (GUI logida qaysi fold ekani ko'rinsin)."""
    def _fn(msg=""):
        log(f"    [{prefix}] {str(msg).strip()}")
    return _fn


def _warn_only_log(log, prefix):
    """Faqat ogohlantirish/xato xabarlarini o'tkazadi (ichki CV'larning har fold logi shovqin bo'lmasin)."""
    def _fn(msg=""):
        s = str(msg)
        low = s.lower()
        if "ogohlantirish" in low or "xato" in low:
            log(f"  [{prefix}] {s.strip()}")
    return _fn


def _scaled_progress(progress_fn, lo, hi, prefix):
    """progress_fn(frac, msg) ni [lo, hi] oralig'iga siqadi va xabarga prefiks qo'shadi. None => None."""
    if progress_fn is None:
        return None

    def _fn(frac=0.0, msg=""):
        try:
            f = min(1.0, max(0.0, float(frac)))
        except (TypeError, ValueError):
            f = 0.0
        progress_fn(lo + (hi - lo) * f, f"{prefix} {msg}".strip())

    return _fn


class _PatchCache:
    """dataset.get_patches(window) natijasini oyna o'lchami bo'yicha keshlaydi (tuning window'ni o'zgartirishi mumkin)."""

    def __init__(self, dataset):
        self.dataset = dataset
        self._cache = {}

    def get(self, window):
        window = int(window)
        if window not in self._cache:
            while len(self._cache) >= _PATCH_CACHE_MAX:
                self._cache.pop(next(iter(self._cache)))
            self._cache[window] = self.dataset.get_patches(window)
        return self._cache[window]


def _has_tunable(model):
    return any(s.tunable for s in PARAM_SPECS[model])


# ---------------------------------------------------------------------------
# Fold'lar
# ---------------------------------------------------------------------------
def _make_splits(y, groups, cv_mode, n_splits, n_repeats, seed, log):
    """Fold'lar ro'yxati [(repeat, fold, train_idx, val_idx)]; argument xatolari shu yerda ko'tariladi."""
    if cv_mode == "spatial":
        gen = spatial.stratified_group_splits(y, groups, n_splits, n_repeats, random_state=seed, log_fn=log)
    else:
        gen = spatial.random_stratified_splits(y, n_splits, n_repeats, random_state=seed)
    return [(int(r), int(f), np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
            for r, f, tr, va in gen]


def _check_splits(splits, y, n_splits, n_repeats):
    """OOF kafolati: har repeat'da har nuqta aynan bir marta val'da; train/val kesishmaydi; train'da ikkala sinf bor."""
    n = len(y)
    if len(splits) != n_splits * n_repeats:
        raise ValueError(f"Fold'lar soni ({len(splits)}) kutilgan {n_splits * n_repeats} ga teng emas.")
    seen = {}
    for r, f, tr, va in splits:
        if va.size == 0 or tr.size == 0:
            raise ValueError(f"{r + 1}-takror, {f + 1}-fold: train yoki validation bo'sh.")
        in_val = np.zeros(n, dtype=bool)
        in_val[va] = True
        if in_val[tr].any() or tr.size + va.size != n:
            raise ValueError(f"{r + 1}-takror, {f + 1}-fold: train va validation kesishadi yoki barcha nuqtalarni "
                             f"qamramaydi (train {tr.size}, val {va.size}, jami {n}).")
        ytr = y[tr]
        if ytr.min() == ytr.max():
            raise ValueError(f"{r + 1}-takror, {f + 1}-fold: train qismida bitta sinf qoldi - modelni o'qitib bo'lmaydi.")
        seen.setdefault(r, []).append(va)
    for r, vals in seen.items():
        cnt = np.bincount(np.concatenate(vals), minlength=n)
        if cnt.min() != 1 or cnt.max() != 1:
            raise ValueError(f"OOF to'liq emas: {r + 1}-takrorda {int((cnt == 0).sum())} nuqta hech bir validation "
                             f"fold'ida emas, {int((cnt > 1).sum())} nuqta bir necha marta validation'da.")


# ---------------------------------------------------------------------------
# Tuning (nested)
# ---------------------------------------------------------------------------
def _tune_fold(tune_model, name, base_hp, train_ds, g_train, tuning, *, calibration_method,
               calibration_cv, n_jobs, seed, log, cancel, progress_fn):
    """Bitta tashqi fold'ning train qismida tune_model. Qaytaradi: (hp, yozuv, o'tkazib yuborish sababi|None).
    Ichki split ValueError'i (musbat bloklar kam va h.k.) yoki tune_model "skipped_reason" qaytarsa standart hp
    ishlatiladi (ogohlantirish bilan) - bu baholash emas, optimallashtirish bosqichi.
    Qidiruv ichida kalibrlash O'CHIRILADI (calibrate=False): ball (roc_auc / average_precision) faqat tartibga
    bog'liq, kalibrlash esa monoton (bazaviy model baribir to'liq train'da o'qitiladi) - nomzod ballari bir xil,
    lekin ichki o'qitishlar RF uchun ~4x, XGBoost uchun ~5x tez (SVM har doim Platt'li). Tashqi fold'ning
    o'zi (va yakuniy model) foydalanuvchi sozlamasi (calibrate) bilan o'qitiladi."""
    try:
        res = tune_model(name, dict(base_hp), train_ds, g_train, tuning=tuning, calibrate=False,
                         calibration_method=calibration_method, calibration_cv=calibration_cv, n_jobs=n_jobs,
                         seed=seed, log_fn=_prefixed_log(log, f"{name} tuning"), cancel=cancel,
                         progress_fn=progress_fn)
    except ValueError as e:
        log(f"  Ogohlantirish: {name} tuning bajarilmadi ({e}); standart giperparametrlar ishlatildi.")
        return dict(base_hp), {"best_params": dict(base_hp), "best_score": float("nan"), "trials": [],
                               "n_trials": 0, "fallback": str(e)}, str(e)
    best = validate_hyperparams({name: res["best_params"]})[0][name]
    rec = dict(res)
    rec["best_params"] = dict(best)
    skipped = res.get("skipped_reason")       # tune_model ichki bo'linish imkonsiz bo'lsa base_params qaytaradi
    return best, rec, (str(skipped) if skipped else None)


# ---------------------------------------------------------------------------
# run_cv
# ---------------------------------------------------------------------------
def run_cv(dataset, groups, cv_mode, model_names, hp, *, n_splits, n_repeats, calibrate=True,
           calibration_method="sigmoid", calibration_cv=3, tuning=None, perm_importance=False,
           perm_repeats=5, n_jobs=1, seed=RANDOM_STATE, log_fn=None, progress_fn=None, cancel=None):
    """Takroriy k-fold CV: har model uchun out-of-fold (OOF) bashoratlar.

    cv_mode: "spatial" (spatial.stratified_group_splits, groups majburiy) | "random" (random_stratified_splits;
    groups faqat nested tuning ichki fold'lari uchun ishlatiladi, None bo'lsa har nuqta alohida guruh).
    hp - config hyperparams lug'ati (validate_hyperparams orqali tozalanadi). Har model uchun CV va (pipeline'dagi)
    final bir xil hp; nested tuning bo'lsa fold'ning best_params'i. Model seed = seed + split_num, har fold
    (va model) oldidan set_global_seed. Tuning: tuning.enabled, tuning.mode == "nested" va tuning.models[name]
    bo'lsa har tashqi fold'ning train qismida tune_model (mode != "nested" bo'lsa ichki tuning qilinmaydi;
    qidiruv kalibrlashsiz ishlaydi - ball tartibga bog'liq, natija bir xil, ~4-5x tez; fold modeli esa calibrate bilan).
    perm_importance: faqat spatial, faqat 1-repeat, har fold val'ida, barcha modellar uchun (patch modellarga patches).

    Qaytaradi: {"mode", "oof": {name: [(n,) float64 NaN-siz, har repeat uchun]}, "fold_map": (n,) int (1-repeat, fold
    indeksi 0 dan), "fold_table": DataFrame (repeat, fold, n_train, n_val, pos_train, pos_val, n_blocks_val -
    spatial.fold_report bilan bir xil - va seconds_<model>), "perm_importance": {name: summarize_perm_importance(...)},
    "tuned_params": {name: [{"repeat","fold","best_params","best_score","trials","n_trials"}, ...]} (faqat tuning
    qilingan modellar), "feature_names", "model_names", "hp": {name: bazaviy hp}, "n_splits", "n_repeats",
    "warnings": [str]}. Model xatosi: aniq log + RuntimeError (jim NaN yo'q). Bekor qilinsa CancelledError.
    """
    log = log_fn or noop_log
    names = _check_model_names(model_names)
    if cv_mode not in _CV_MODES:
        raise ValueError(f"cv_mode noto'g'ri: {cv_mode!r}; mumkin: {list(_CV_MODES)}")
    n_splits, n_repeats = int(n_splits), int(n_repeats)
    if n_splits < 2:
        raise ValueError(f"k-fold soni (n_splits) kamida 2 bo'lishi kerak, berildi: {n_splits}.")
    if n_repeats < 1:
        raise ValueError(f"Takrorlash soni (n_repeats) kamida 1 bo'lishi kerak, berildi: {n_repeats}.")
    seed = int(seed)

    X = np.asarray(dataset.X)
    y = np.asarray(dataset.y).reshape(-1).astype(np.int64)
    n = int(y.size)
    if X.ndim != 2 or X.shape[0] != n:
        raise ValueError(f"dataset.X shakli (n, n_features) bo'lishi kerak va y ({n}) bilan mos: {X.shape}.")
    if groups is not None:
        groups = np.asarray(groups).reshape(-1)
        if groups.size != n:
            raise ValueError(f"groups uzunligi ({groups.size}) nuqtalar soniga ({n}) teng emas.")
    if cv_mode == "spatial" and groups is None:
        raise ValueError("spatial CV uchun groups (spatial bloklar) majburiy.")

    hp_clean, hp_warn = validate_hyperparams(hp)
    for w in hp_warn:
        log(f"  Giperparametr: {w}")

    # tuning
    tune_names = set()
    tune_model = None
    if isinstance(tuning, dict):
        tuning = TuningConfig.from_dict(tuning)
    if tuning is not None and tuning.enabled:
        if getattr(tuning, "mode", "nested") != "nested":
            log(f"  Tuning rejimi '{tuning.mode}': nested ichki tuning o'tkazib yuborildi "
                f"(run_cv faqat 'nested' rejimda fold ichida tuning qiladi).")
        else:
            tune_names = {m for m in names if tuning.models.get(m) and _has_tunable(m)}
            if tune_names:
                from .tuning import tune_model       # lazy: boshqa agent yozadigan modul
                log(f"  Nested tuning: {', '.join(m for m in names if m in tune_names)} "
                    f"(n_iter={tuning.n_iter}, ichki fold={tuning.inner_splits}, ball={tuning.scoring}).")

    # modellarni oldindan yaratib ko'rish: kutubxona yo'q / parametr xato bo'lsa CV boshlanmasdan aniq xato
    patch_models = set()
    for name in names:
        probe = make_model(name, hp_clean[name], calibrate=calibrate, calibration_method=calibration_method,
                           calibration_cv=calibration_cv, n_jobs=n_jobs, seed=seed, n_features=X.shape[1])
        if getattr(probe, "input_kind", "tabular") == "patch":
            patch_models.add(name)
    if patch_models and getattr(dataset, "feature_stack", None) is None:
        raise ValueError(f"{', '.join(sorted(patch_models))} (patch2d) uchun dataset.feature_stack kerak: "
                         "build_dataset(..., need_feature_stack=True) bilan yarating.")

    splits = _make_splits(y, groups, cv_mode, n_splits, n_repeats, seed, log)
    _check_splits(splits, y, n_splits, n_repeats)
    total = len(splits)

    do_perm = bool(perm_importance) and cv_mode == "spatial"
    if perm_importance and not do_perm:
        log("  Permutation importance faqat spatial CV'da hisoblanadi; random CV'da o'tkazib yuborildi.")
    if do_perm:
        from .explain import permutation_importance_auc, summarize_perm_importance   # lazy
        perm_repeats = max(1, int(perm_repeats))

    oof = {m: [np.full(n, np.nan, dtype=np.float64) for _ in range(n_repeats)] for m in names}
    fold_map = np.full(n, -1, dtype=np.int64)
    perm_folds = {m: [] for m in names}
    tuned = {m: [] for m in names if m in tune_names}
    table_rows = []
    notes = []
    seen_notes = set()
    cache = _PatchCache(dataset)
    t0 = time.perf_counter()

    def note(msg, key=None):
        """Muhim voqealarni natijadagi "warnings" ga bir marta yozadi (key bo'yicha takrorlanmaydi)."""
        key = msg if key is None else key
        if key not in seen_notes:
            seen_notes.add(key)
            notes.append(msg)

    for split_num, (repeat, fold, tr, va) in enumerate(splits):
        check_cancel(cancel)
        fold_seed = seed + split_num
        set_global_seed(fold_seed)
        ctx = f"{repeat + 1}-takror, {fold + 1}/{n_splits}-fold"
        train_ds = g_train = None
        secs = {}
        for j, name in enumerate(names):
            check_cancel(cancel)
            tm = time.perf_counter()
            try:
                fold_hp = hp_clean[name]
                if name in tune_names:
                    if train_ds is None:
                        train_ds = dataset.subset(tr)
                        g_train = groups[tr] if groups is not None else np.arange(tr.size)
                    lo = (split_num + _TUNE_PROGRESS_SHARE * j / len(names)) / total
                    hi = (split_num + _TUNE_PROGRESS_SHARE * (j + 1) / len(names)) / total
                    fold_hp, rec, fallback = _tune_fold(
                        tune_model, name, fold_hp, train_ds, g_train, tuning,
                        calibration_method=calibration_method, calibration_cv=calibration_cv, n_jobs=n_jobs,
                        seed=fold_seed, log=log, cancel=cancel,
                        progress_fn=_scaled_progress(progress_fn, lo, hi,
                                                     f"Fold {split_num + 1}/{total} {name} tuning:"))
                    rec["repeat"], rec["fold"] = repeat, fold
                    tuned[name].append(rec)
                    if fallback:
                        note(f"{name}: tuning bajarilmadi ({fallback}); standart giperparametrlar ishlatildi.",
                             key=f"tune:{name}")
                check_cancel(cancel)
                set_global_seed(fold_seed)
                model = make_model(name, fold_hp, calibrate=calibrate, calibration_method=calibration_method,
                                   calibration_cv=calibration_cv, n_jobs=n_jobs, seed=fold_seed,
                                   n_features=X.shape[1])
                is_patch = model.input_kind == "patch"
                P = cache.get(fold_hp["window"]) if is_patch else None
                model.fit(X[tr], y[tr], patches=P[tr] if is_patch else None, cancel=cancel,
                          log_fn=_prefixed_log(log, f"{name} {repeat + 1}.{fold + 1}"))
                check_cancel(cancel)
                p = np.asarray(model.predict_proba_pos(X[va], patches=P[va] if is_patch else None),
                               dtype=np.float64).reshape(-1)
                if p.shape != (va.size,) or not np.all(np.isfinite(p)):
                    raise RuntimeError(f"bashorat yaroqsiz: shakl {p.shape}, chekli emas: "
                                       f"{int(p.size - np.count_nonzero(np.isfinite(p)))} ta qiymat")
            except CancelledError:
                raise
            except Exception as e:
                msg = f"{name} modeli ({ctx}) xato berdi: {type(e).__name__}: {e}"
                log(f"  XATO: {msg}")
                raise RuntimeError(msg) from e
            secs[name] = time.perf_counter() - tm
            oof[name][repeat][va] = p
            info = getattr(model, "fit_info", None) or {}
            if info.get("calibration") == "skipped":
                note(f"{name}: kalibrlash o'tkazib yuborildi ({info.get('calibration_reason', '')}).",
                     key=f"cal:{name}")
            if info.get("svm_fallback"):
                note(f"{name}: kalibrlanmagan zaxira indeks ishlatildi ({info['svm_fallback']}).",
                     key=f"svmfb:{name}")

            if do_perm and repeat == 0:
                try:
                    imp = permutation_importance_auc(model, X[va], y[va], patches=P[va] if is_patch else None,
                                                     n_repeats=perm_repeats, seed=fold_seed, cancel=cancel)
                    perm_folds[name].append(np.asarray(imp, dtype=np.float64))
                except CancelledError:
                    raise
                except Exception as e:
                    log(f"  Ogohlantirish: {name} permutation importance ({ctx}) hisoblanmadi: "
                        f"{type(e).__name__}: {e}")
                    note(f"{name}: permutation importance hisoblanmadi ({type(e).__name__}: {e}).",
                         key=f"perm:{name}")

        if repeat == 0:
            fold_map[va] = fold
        row = {"repeat": repeat, "fold": fold, "n_train": int(tr.size), "n_val": int(va.size),
               "pos_train": int(y[tr].sum()), "pos_val": int(y[va].sum()),
               # spatial.fold_report bilan bir xil: groups None bo'lsa har nuqta alohida blok
               "n_blocks_val": int(np.unique(groups[va]).size) if groups is not None else int(va.size)}
        row.update({f"seconds_{m}": float(secs[m]) for m in names})
        table_rows.append(row)

        done = split_num + 1
        elapsed = time.perf_counter() - t0
        eta = elapsed / done * (total - done)
        log(f"  [{cv_mode}] {ctx}: " + ", ".join(f"{m} {secs[m]:.1f}s" for m in names)
            + f" (val: {int(va.size)} nuqta, {row['pos_val']} musbat)")
        if progress_fn is not None:
            progress_fn(done / total, f"Fold {done}/{total} ({cv_mode}, {ctx}) ... ETA {_fmt_eta(eta)}")

    # OOF to'liqligi (qo'shimcha himoya: bo'linishlar oldindan tekshirilgan, lekin NaN jim qolmasin)
    for name in names:
        for r, arr in enumerate(oof[name]):
            if not np.all(np.isfinite(arr)):
                raise RuntimeError(f"OOF to'liq emas: {name}, {r + 1}-takror: {int((~np.isfinite(arr)).sum())} "
                                   f"nuqta uchun bashorat yo'q (ichki xato).")
    if (fold_map < 0).any():
        raise RuntimeError("fold_map to'liq emas (ichki xato).")

    perm_out = {}
    if do_perm:
        for name in names:
            if perm_folds[name]:
                perm_out[name] = summarize_perm_importance(perm_folds[name])
            else:
                log(f"  Ogohlantirish: {name} uchun permutation importance hisoblanmadi.")
    columns = ["repeat", "fold", "n_train", "n_val", "pos_train", "pos_val", "n_blocks_val"] \
        + [f"seconds_{m}" for m in names]
    return {"mode": cv_mode, "oof": oof, "fold_map": fold_map,
            "fold_table": pd.DataFrame(table_rows, columns=columns), "perm_importance": perm_out,
            "tuned_params": tuned, "feature_names": list(getattr(dataset, "feature_names", [])),
            "model_names": list(names), "hp": {m: dict(hp_clean[m]) for m in names},
            "n_splits": n_splits, "n_repeats": n_repeats, "warnings": notes}


# ---------------------------------------------------------------------------
# Metrikalar
# ---------------------------------------------------------------------------
def youden_threshold(y, p):
    """Youden J = TPR - FPR maksimal bo'ladigan bo'sag' (p >= bo'sag' => musbat). y bir sinfli bo'lsa ValueError."""
    y = np.asarray(y).reshape(-1).astype(np.int64)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if y.size != p.size:
        raise ValueError(f"y ({y.size}) va p ({p.size}) uzunligi teng emas.")
    if y.size == 0 or y.min() == y.max():
        raise ValueError("Youden bo'sag'i uchun ikkala sinf (0 va 1) kerak.")
    fpr, tpr, thr = roc_curve(y, p)
    j = np.where(np.isfinite(thr), tpr - fpr, -np.inf)
    if not np.isfinite(j).any():
        return 0.5
    return float(thr[int(np.argmax(j))])


def _point_stats(y, p, thr):
    """Bitta ehtimollik massivi uchun nuqtaviy metrikalar (sklearn bilan): bo'sag' 0.5 va Youden."""
    pred5 = (p >= 0.5).astype(np.int64)
    predy = (p >= thr).astype(np.int64)
    return {"auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred5)),
            "f1": float(f1_score(y, pred5, zero_division=0)), "brier": float(brier_score_loss(y, p)),
            "balanced_accuracy_youden": float(balanced_accuracy_score(y, predy)),
            "f1_youden": float(f1_score(y, predy, zero_division=0))}


def _auc_fast(yb, p):
    n1 = int(yb.sum())
    n0 = yb.size - n1
    return float((rankdata(p)[yb].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _ap_fast(yb, p):
    """average_precision_score bilan bir xil (bog'langan qiymatlar bo'yicha guruhlanadi), lekin ancha tez."""
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], yb[order]
    distinct = np.r_[np.flatnonzero(ps[1:] != ps[:-1]), ps.size - 1]
    tps = np.cumsum(ys)[distinct]
    precision = tps / (distinct + 1.0)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _thr_stats_fast(yb, p, thr):
    """(balanced_accuracy, f1) bo'sag'ida (p >= thr)."""
    n1 = int(yb.sum())
    n0 = yb.size - n1
    pred = p >= thr
    tp = int(np.count_nonzero(pred & yb))
    fp = int(np.count_nonzero(pred & ~yb))
    fn, tn = n1 - tp, n0 - fp
    den = 2 * tp + fp + fn
    return 0.5 * (tp / n1 + tn / n0), (2.0 * tp / den if den > 0 else 0.0)


def _boot_stats(yb, p, thr):
    """_point_stats bilan bir xil natija (sklearn'siz) - bootstrap tsiklida tezlik uchun."""
    ba5, f15 = _thr_stats_fast(yb, p, 0.5)
    bay, f1y = _thr_stats_fast(yb, p, thr)
    return {"auc": _auc_fast(yb, p), "pr_auc": _ap_fast(yb, p), "balanced_accuracy": ba5, "f1": f15,
            "brier": float(np.mean((p - yb) ** 2)), "balanced_accuracy_youden": bay, "f1_youden": f1y}


def _bootstrap_ci(yb, series, groups, n_boot, seed, log):
    """Blok-bootstrap (groups None => nuqta bootstrap): {name: {stat: (lo, hi)}} - 2.5/97.5 persentillar.
    series: {name: (mean_proba, youden_threshold)}. Bir sinfli resample'lar tashlanadi; resample'lar barcha
    seriyalar uchun umumiy (juftlashgan). n_boot=0 => (nan, nan)."""
    nan_ci = (float("nan"), float("nan"))
    out = {name: {k: nan_ci for k in _STAT_KEYS} for name in series}
    if n_boot < 1:
        return out
    n = yb.size
    g = np.arange(n) if groups is None else groups
    vals = {name: {k: [] for k in _STAT_KEYS} for name in series}
    used = 0
    for idx in spatial.block_bootstrap_indices(g, n_boot, random_state=seed):
        yi = yb[idx]
        n1 = int(yi.sum())
        if n1 == 0 or n1 == idx.size:
            continue
        used += 1
        for name, (p, thr) in series.items():
            st = _boot_stats(yi, p[idx], thr)
            for k in _STAT_KEYS:
                vals[name][k].append(st[k])
    if used < n_boot:
        log(f"  Bootstrap: {n_boot - used}/{n_boot} resample bir sinfli bo'lgani uchun tashlandi.")
    if used < _MIN_BOOT_WARN:
        log(f"  Ogohlantirish: yaroqli bootstrap resample'lar kam ({used}) - CI ishonchsiz"
            f"{' (hisoblanmadi)' if used < 2 else ''}.")
    if used < 2:
        return out
    for name in series:
        for k in _STAT_KEYS:
            lo, hi = np.percentile(vals[name][k], [2.5, 97.5])
            out[name][k] = (float(lo), float(hi))
    return out


def _calibration(y, p):
    """Kvantil kalibrlash egri chizig'i; bin soni musbatlarga moslab 3..10 (har binda ~5 musbat)."""
    n_pos = int(y.sum())
    n_bins = int(min(10, max(3, n_pos // 5)))
    prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    bins = np.percentile(p, np.linspace(0.0, 1.0, n_bins + 1) * 100)      # calibration_curve bilan bir xil bin'lash
    total = np.bincount(np.searchsorted(bins[1:-1], p), minlength=n_bins)
    counts = total[total != 0]
    if counts.size != prob_true.size:                                    # himoya: bin'lash mos kelmasa
        counts = np.full(prob_true.size, -1, dtype=np.int64)
    return {"prob_pred": np.asarray(prob_pred, dtype=np.float64), "prob_true": np.asarray(prob_true, dtype=np.float64),
            "counts": counts.astype(np.int64)}


def _series_result(y, reps, ci, thr):
    """Bitta model (yoki ansambl) uchun barcha metrikalar. reps: (n_repeats, n) OOF; ci: _bootstrap_ci natijasi;
    thr: mean OOF ehtimollikdan topilgan Youden bo'sag'i."""
    n_rep = reps.shape[0]
    mean_p = reps.mean(axis=0)
    per_rep = [_point_stats(y, r, thr) for r in reps]
    res = {}
    for k in _STAT_KEYS:
        v = np.array([d[k] for d in per_rep], dtype=np.float64)
        res[k] = float(v.mean())
        res[k + "_std"] = float(v.std(ddof=1)) if n_rep > 1 else 0.0
        res[k + "_ci95"] = ci[k]
    res["auc_single"] = float(roc_auc_score(y, mean_p))
    res["auc_repeats"] = np.array([d["auc"] for d in per_rep], dtype=np.float64)
    pred = mean_p >= thr
    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    n1 = int(y.sum())
    fn, tn = n1 - tp, int(y.size - n1) - fp
    res["threshold_youden"] = float(thr)
    res["sensitivity"] = float(tp / n1)
    res["specificity"] = float(tn / (tn + fp))
    res["confusion"] = [[tn, fp], [fn, tp]]
    fpr, tpr, _ = roc_curve(y, mean_p)
    precision, recall, _ = precision_recall_curve(y, mean_p)
    res["fpr"], res["tpr"] = fpr, tpr
    res["precision"], res["recall"] = precision, recall
    res["calibration"] = _calibration(y, mean_p)
    res["mean_proba"] = mean_p
    return res


def compute_metrics(y, oof, groups=None, n_boot=1000, seed=RANDOM_STATE, log_fn=None):
    """OOF bashoratlardan metrikalar: har model va ENSEMBLE_NAME uchun. Qaytaradi: (natijalar, ansambl mean_proba).

    oof: {model: [(n,) massiv har repeat uchun]} (yoki (n_repeats, n) massiv / bitta (n,) massiv). Barcha modellarda
    repeat soni bir xil. Ansambl: har repeat ichida modellar o'rtachasi, so'ng repeat'lar bo'yicha statistikalar.
    Nuqtaviy metrikalar (auc, pr_auc, balanced_accuracy, f1, brier, *_youden) - repeat'lar bo'yicha O'RTACHA, "_std" -
    repeat'lar orasidagi tanlanma std (ddof=1; 1 repeat => 0), "_ci95" - blok-bootstrap (groups None => nuqta
    bootstrap) persentil oralig'i, mean OOF ehtimollik ustida (Youden bo'sag'i to'liq tanlanmadan olinib bootstrap'da
    qotirilgan); bir sinfli resample'lar tashlanadi, n_boot=0 => (nan, nan). "balanced_accuracy"/"f1" bo'sag'i 0.5,
    "*_youden" - mean OOF'dan topilgan Youden bo'sag'i. threshold_youden, sensitivity, specificity, confusion
    ([[tn, fp], [fn, tp]]), fpr/tpr/precision/recall va calibration (kvantil, 3..10 bin) - mean OOF ehtimollik ustida.
    """
    log = log_fn or noop_log
    y = np.asarray(y).reshape(-1)
    if y.size == 0 or not np.all((y == 0) | (y == 1)):
        raise ValueError("y faqat 0/1 qiymatlardan iborat va bo'sh bo'lmasligi kerak.")
    y = y.astype(np.int64)
    if y.min() == y.max():
        raise ValueError("y bir sinfli: metrikalar (AUC) uchun ham musbat, ham fon nuqtalar kerak.")
    n = int(y.size)
    if not isinstance(oof, dict) or not oof:
        raise ValueError("oof bo'sh: kamida bitta model bashorati kerak.")
    n_boot = int(n_boot)
    if n_boot < 0:
        raise ValueError(f"n_boot manfiy bo'lmasligi kerak: {n_boot}.")
    if groups is not None:
        groups = np.asarray(groups).reshape(-1)
        if groups.size != n:
            raise ValueError(f"groups uzunligi ({groups.size}) y uzunligiga ({n}) teng emas.")

    names = list(oof.keys())
    reps = {}
    for name in names:
        arr = np.atleast_2d(np.asarray(oof[name], dtype=np.float64))
        if arr.ndim != 2 or arr.shape[1] != n or arr.shape[0] < 1:
            raise ValueError(f"oof['{name}'] shakli (n_repeats, {n}) bo'lishi kerak, berilgan: {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"oof['{name}'] da {int((~np.isfinite(arr)).sum())} ta NaN/inf bor - OOF to'liq bo'lishi kerak.")
        reps[name] = np.clip(arr, 0.0, 1.0)
    n_rep = {reps[m].shape[0] for m in names}
    if len(n_rep) != 1:
        raise ValueError(f"Modellar repeat soni har xil: { {m: reps[m].shape[0] for m in names} }.")
    reps[ENSEMBLE_NAME] = np.mean([reps[m] for m in names], axis=0)

    yb = y == 1
    mean_ps = {name: r.mean(axis=0) for name, r in reps.items()}
    series = {name: (mean_ps[name], youden_threshold(y, mean_ps[name])) for name in reps}
    ci = _bootstrap_ci(yb, series, groups, n_boot, int(seed), log)

    results = {}
    for name in reps:
        results[name] = _series_result(y, reps[name], ci[name], series[name][1])
        r = results[name]
        lo, hi = r["auc_ci95"]
        ci_txt = f"95%CI {lo:.3f}-{hi:.3f}" if np.isfinite(lo) and np.isfinite(hi) else "CI hisoblanmadi"
        log(f"  {name}: AUC={r['auc']:.3f}±{r['auc_std']:.3f} ({ci_txt}) "
            f"| PR-AUC={r['pr_auc']:.3f} | BalAcc={r['balanced_accuracy']:.3f} | F1={r['f1']:.3f} "
            f"| Brier={r['brier']:.3f} | Sens/Spec@Youden={r['sensitivity']:.2f}/{r['specificity']:.2f}")
    return results, results[ENSEMBLE_NAME]["mean_proba"]


_DF_COLUMNS = ["Model", "AUC", "AUC_std", "AUC_CI_lo", "AUC_CI_hi", "PR_AUC", "BalAcc", "F1", "Brier",
               "Sens", "Spec", "Thr"]


def metrics_dataframe(results, label=""):
    """compute_metrics natijasidan jadval: bir qator har model (ansambl oxirida). label berilsa birinchi
    "CV" ustuni (masalan "spatial" / "random") qo'shiladi - ikki jadvalni birlashtirish uchun."""
    nan = float("nan")
    rows = []
    for name, m in results.items():
        lo, hi = m.get("auc_ci95", (nan, nan))
        rows.append({"Model": name, "AUC": m.get("auc", nan), "AUC_std": m.get("auc_std", nan),
                     "AUC_CI_lo": lo, "AUC_CI_hi": hi, "PR_AUC": m.get("pr_auc", nan),
                     "BalAcc": m.get("balanced_accuracy", nan), "F1": m.get("f1", nan),
                     "Brier": m.get("brier", nan), "Sens": m.get("sensitivity", nan),
                     "Spec": m.get("specificity", nan), "Thr": m.get("threshold_youden", nan)})
    cols = _DF_COLUMNS
    if label:
        cols = ["CV"] + cols
        for r in rows:
            r["CV"] = label
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Fon nuqtalar sezgirligi
# ---------------------------------------------------------------------------
def run_background_sensitivity(aoi_gdf, positive_gdf, raster, pipeline, hp, *, model_names, n_background, min_distance,
                               strategy, block_size, n_draws, n_splits, n_repeats, calibrate, calibration_method,
                               calibration_cv, n_jobs, seed, log_fn=None, progress_fn=None, cancel=None):
    """Fon nuqtalar tanlovi natijaga ta'sirini baholaydi: har draw (seed + 1000 + draw, draw 0 dan) uchun yangi fon
    nuqtalar -> dataset -> spatial bloklar (block_size; yetmasa draw o'tkazib yuboriladi) -> spatial CV ->
    compute_metrics(n_boot=0). Qaytaradi: {"per_draw": [{"draw","seed","auc":{name:auc},"n_background","block_size"}],
    "summary": {name: {"mean","std"(ddof=1),"min","max","values"}} (ansambl ham), "n_positive", "n_draws",
    "n_draws_requested"} yoki hech draw bajarilmasa None. Model xatosi - RuntimeError (run_cv kabi)."""
    from . import data      # lazy: og'ir (geopandas/rasterio) import faqat shu yerda kerak

    log = log_fn or noop_log
    names = _check_model_names(model_names)
    n_draws = int(n_draws)
    if n_draws < 1:
        log("  Fon sezgirligi: draw soni 0 - o'tkazib yuborildi.")
        return None
    if strategy not in BACKGROUND_STRATEGIES:
        raise ValueError(f"Fon strategiyasi noto'g'ri: {strategy!r}; mumkin: {list(BACKGROUND_STRATEGIES)}")
    try:
        bs = float(block_size)
    except (TypeError, ValueError):
        bs = float("nan")
    if not np.isfinite(bs) or bs <= 0:
        raise ValueError(f"block_size musbat son bo'lishi kerak, berildi: {block_size!r}.")
    seed = int(seed)
    hp_clean, hp_warn = validate_hyperparams(hp)
    for w in hp_warn:
        log(f"  Giperparametr: {w}")
    need_fs = "CNN" in names and hp_clean["CNN"]["mode"] == "patch2d"
    valid = data.valid_pixel_mask(raster)
    log(f"Fon sezgirligi: {n_draws} ta draw, har birida {n_background} fon nuqta ({strategy}), "
        f"blok {bs:,.0f} m, {n_splits}-fold x {n_repeats} takror.")

    per_draw = []
    n_pos = None
    for d in range(n_draws):
        check_cancel(cancel)
        sd = seed + 1000 + d
        tag = f"Fon sezgirligi {d + 1}/{n_draws}"
        try:
            bg = data.generate_background_points(aoi_gdf, positive_gdf, n_background, min_distance, random_state=sd,
                                                 strategy=strategy, valid_mask=valid, transform=raster.transform,
                                                 log_fn=log)
            ds = data.build_dataset(raster, pipeline, positive_gdf, bg, log_fn=log, need_feature_stack=need_fs)
            bs_d, groups = spatial.adapt_block_size(ds.coords, ds.y, n_splits, bs, log_fn=log)
        except ValueError as e:
            log(f"  Ogohlantirish: {tag} o'tkazib yuborildi: {e}")
            if progress_fn is not None:
                progress_fn((d + 1) / n_draws, f"{tag}: o'tkazib yuborildi")
            continue
        res = run_cv(ds, groups, "spatial", names, hp_clean, n_splits=n_splits, n_repeats=n_repeats,
                     calibrate=calibrate, calibration_method=calibration_method, calibration_cv=calibration_cv,
                     tuning=None, perm_importance=False, n_jobs=n_jobs, seed=sd,
                     log_fn=_warn_only_log(log, tag),
                     progress_fn=_scaled_progress(progress_fn, d / n_draws, (d + 1) / n_draws, f"{tag}:"),
                     cancel=cancel)
        metrics, _ = compute_metrics(ds.y, res["oof"], groups, n_boot=0, seed=sd, log_fn=noop_log)
        auc = {name: float(m["auc"]) for name, m in metrics.items()}
        per_draw.append({"draw": d, "seed": sd, "auc": auc, "n_background": int(ds.n_neg),
                         "block_size": float(bs_d)})
        n_pos = int(ds.n_pos) if n_pos is None else n_pos
        log(f"  {tag}: " + ", ".join(f"{k} AUC={v:.3f}" for k, v in auc.items()))
        if progress_fn is not None:
            progress_fn((d + 1) / n_draws, f"{tag}: tugadi")

    if not per_draw:
        log("  Ogohlantirish: fon sezgirligi bo'yicha birorta draw bajarilmadi.")
        return None
    summary = {}
    for name in names + [ENSEMBLE_NAME]:
        vals = [float(pd_["auc"][name]) for pd_ in per_draw]
        summary[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                         "min": float(np.min(vals)), "max": float(np.max(vals)), "values": vals}
        log(f"  Fon sezgirligi {name}: AUC {summary[name]['mean']:.3f}±{summary[name]['std']:.3f} "
            f"[{summary[name]['min']:.3f}-{summary[name]['max']:.3f}] ({len(vals)} draw)")
    if len(per_draw) < n_draws:
        log(f"  Ogohlantirish: {n_draws - len(per_draw)}/{n_draws} draw o'tkazib yuborildi.")
    return {"per_draw": per_draw, "summary": summary, "n_positive": n_pos, "n_draws": len(per_draw),
            "n_draws_requested": n_draws}
