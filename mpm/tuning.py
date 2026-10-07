# -*- coding: utf-8 -*-
"""
Giperparametr qidiruvi (ENH-02): ichki spatial CV bilan tasodifiy qidiruv.

  * sample_params - bitta tasodifiy nomzod (tur/diapazon/log hurmat qilinadi, natija config.validate_hyperparams
    dan o'tkaziladi);
  * tune_model - nomzodlar: 1) base_params o'zi, 2..n_iter) tasodifiy. Har nomzod ichki
    `stratified_group_splits(n_splits=tuning.inner_splits, n_repeats=1)` bo'yicha baholanadi
    (bir xil fold'lar va fold-seed'lar - nomzodlar juftlab solishtiriladi). Eng yaxshi ball g'olib; teng bo'lsa
    base_params afzal. Ichki bo'linish imkonsiz bo'lsa fold soni kamaytiriladi, u ham bo'lmasa base_params
    ogohlantirish bilan qaytariladi (hech qachon yiqilmaydi).

CNN (patch2d): `window` nomzod ichida o'zgarishi mumkin - patchlar `dataset.get_patches(window)` bilan
olinadi va nomzodlar window bo'yicha guruhlanib baholanadi (har window uchun bir marta, xotira tejaladi).
CNN tuning qimmat (yuzlab o'qitish; TensorFlow xotirasi har fit'da ~8 MB o'sadi) - standart bo'yicha o'chirilgan.
"""
from __future__ import annotations

import copy
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import CancelledError, check_cancel, noop_log
from .config import PARAM_SPECS, TuningConfig, validate_hyperparams
from .models import make_model
from .spatial import stratified_group_splits

__all__ = ["sample_params", "tune_model"]

_MAX_DUP_TRIES = 30        # ketma-ket shuncha takroriy nomzod chiqsa, qidiruv fazosi tugagan hisoblanadi
_TIE_EPS = 1e-12           # ballar shu aniqlikda teng bo'lsa base afzal
_PRED_BATCH = 8192
_MAX_FIT_WARNINGS = 5      # fit() ogohlantirishlaridan logga uzatiladigan eng ko'pi


# ---------------------------------------------------------------------------
# sample_params
# ---------------------------------------------------------------------------
def _spec_kinds(model):
    if model not in PARAM_SPECS:
        raise ValueError(f"Noma'lum model: {model!r}; mumkin: {list(PARAM_SPECS)}")
    return {s.name: s.kind for s in PARAM_SPECS[model]}


def _bounds(where, cfg, need_positive):
    try:
        lo, hi = float(cfg.get("min")), float(cfg.get("max"))
    except (TypeError, ValueError):
        raise ValueError(f"{where}: qidiruv oralig'i uchun son 'min' va 'max' kerak, berilgan: "
                         f"{cfg.get('min')!r}, {cfg.get('max')!r}") from None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"{where}: min/max chekli son bo'lishi kerak ({lo}, {hi})")
    if lo > hi:
        raise ValueError(f"{where}: min ({lo:g}) max ({hi:g}) dan katta bo'lmasligi kerak")
    if need_positive and lo <= 0:
        raise ValueError(f"{where}: logarifmik shkala uchun min > 0 bo'lishi kerak (berilgan {lo:g})")
    return lo, hi


def _draw_one(model, pname, kind, cfg, rng):
    """Bitta parametr qiymati (Python turida): int -> int, float -> float, choice -> asl qiymat."""
    where = f"{model}.{pname}"
    if not isinstance(cfg, dict):
        raise ValueError(f"{where}: qidiruv oralig'i lug'at bo'lishi kerak, berilgan: {cfg!r}")
    ptype = cfg.get("type")
    if ptype is None:
        ptype = "choice" if cfg.get("choices") else ("int" if kind in ("int", "optint") else "float")
    if ptype not in ("int", "float", "choice"):
        raise ValueError(f"{where}: noma'lum qidiruv turi {ptype!r} (int | float | choice)")
    if ptype == "choice":
        choices = list(cfg.get("choices") or [])
        if not choices:
            raise ValueError(f"{where}: 'choice' uchun bo'sh bo'lmagan 'choices' kerak")
        v = choices[int(rng.integers(len(choices)))]          # indeks bo'yicha: asl tur saqlanadi
        return v.item() if isinstance(v, np.generic) else v
    log = bool(cfg.get("log", False))
    lo, hi = _bounds(where, cfg, need_positive=log)
    if ptype == "float":
        if lo == hi:
            return float(lo)
        v = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if log else float(rng.uniform(lo, hi))
        return float(min(max(v, lo), hi))
    ilo, ihi = int(round(lo)), int(round(hi))
    if log:
        v = int(round(float(np.exp(rng.uniform(np.log(lo), np.log(hi))))))
        return int(min(max(v, ilo), ihi))
    return int(rng.integers(ilo, ihi + 1))


def sample_params(model, base_params, space, rng):
    """Bitta tasodifiy nomzod: base_params nusxasida `space` dagi parametrlar almashtiriladi.

    space = TuningConfig.resolved_space(model): {param: {"type": "int"|"float"|"choice", "min", "max",
    "log", "choices"}}. int - tekis butun (yoki log + round), float - tekis yoki log, choice - qiymat asl
    turda. Boshqa kalitlar saqlanadi. Natija `config.validate_hyperparams` (shu model qismi) dan o'tadi:
    to'liq, tur/diapazon jihatidan yaroqli lug'at. Noma'lum parametr yoki yaroqsiz oraliq - ValueError.
    rng - numpy Generator (yoki default_rng qabul qiladigan seed)."""
    kinds = _spec_kinds(model)
    space = dict(space or {})
    unknown = sorted(set(space) - set(kinds))
    if unknown:
        raise ValueError(f"{model}: qidiruv oralig'ida noma'lum parametr(lar) {unknown}; mumkin: {sorted(kinds)}")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    params = dict(base_params or {})
    for pname in kinds:                                   # PARAM_SPECS tartibi: space lug'ati tartibiga bog'liq emas
        if pname in space:
            params[pname] = _draw_one(model, pname, kinds[pname], space[pname], rng)
    clean, _ = validate_hyperparams({model: params})
    return clean[model]


# ---------------------------------------------------------------------------
# tune_model yordamchilari
# ---------------------------------------------------------------------------
def _score(scoring, y, p):
    """Ball (roc_auc | average_precision); bashorat chekli bo'lmasa yoki y bir sinfli bo'lsa NaN."""
    p = np.asarray(p, dtype=np.float64)
    if not np.isfinite(p).all() or y.min() == y.max():
        return float("nan")
    fn = roc_auc_score if scoring == "roc_auc" else average_precision_score
    return float(fn(y, p))


def _key(params):
    return json.dumps(params, sort_keys=True, default=str)


def _fallback(base, scoring, reason, name, log):
    log(f"  OGOHLANTIRISH: {name} tuning o'tkazilmadi ({reason}); berilgan giperparametrlar ishlatiladi.")
    return {"best_params": copy.deepcopy(base), "best_score": float("nan"), "trials": [], "n_trials": 0,
            "scoring": scoring, "base_score": float("nan"), "best_index": 0, "inner_splits_used": 0,
            "skipped_reason": reason}              # kalitlar to'plami oddiy natija bilan bir xil (+ skipped_reason)


def _inner_splits(y, groups, n_req, seed, log):
    """Ichki spatial fold'lar; imkonsiz bo'lsa fold soni 2 gacha kamaytiriladi. (splits, n_used, xato)."""
    err = None
    for k in range(max(2, int(n_req)), 1, -1):
        try:
            splits = [(tr, va) for _, _, tr, va in stratified_group_splits(y, groups, k, 1, seed, log_fn=log)]
        except ValueError as e:
            err = e
            continue
        if k < n_req:
            log(f"  Ichki fold soni {n_req} dan {k} gacha kamaytirildi (musbat nuqtali bloklar yetarli emas).")
        return splits, k, None
    return None, 0, err


def _generate_candidates(name, base, space, n_iter, rng):
    """[base, tasodifiy...] (takrorlarsiz). Qidiruv fazosi tugasa n_iter dan kam bo'lishi mumkin."""
    cands, seen, dup = [copy.deepcopy(base)], {_key(base)}, 0
    while len(cands) < n_iter:
        c = sample_params(name, base, space, rng)
        k = _key(c)
        if k in seen:
            dup += 1
            if dup >= _MAX_DUP_TRIES:
                break
            continue
        dup = 0
        seen.add(k)
        cands.append(c)
    return cands


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def _diff_text(params, base):
    d = [f"{k}={_fmt(v)}" for k, v in params.items() if base.get(k) != v]
    return ", ".join(d) if d else "base"


# ---------------------------------------------------------------------------
# tune_model
# ---------------------------------------------------------------------------
def tune_model(name, base_params, dataset, groups, *, tuning, calibrate, calibration_method, calibration_cv,
               n_jobs, seed, log_fn=None, cancel=None, progress_fn=None):
    """Bitta model uchun tasodifiy giperparametr qidiruvi (ichki spatial CV).

    dataset/groups - TRAIN to'plami (tashqi fold'ning train qismi). Qaytaradi:
    {"best_params": TO'LIQ param lug'ati, "best_score": float, "trials": [{"params", "score", "std",
    "n_folds"}] (nomzodlar tartibida; trials[0] = base_params), "n_trials": int, "scoring", "base_score",
    "best_index", "inner_splits_used"}. Ichki bo'linish/ma'lumot yetarli bo'lmasa:
    best_params=base_params, best_score=NaN, trials=[] va "skipped_reason" (ogohlantirish logda).

    Narx: n_iter x inner_splits ta o'qitish; calibrate=True bo'lsa har o'qitish (calibration_cv+1) marta qimmatroq.
    roc_auc/average_precision tartibga asoslangan (kalibrlash monoton) - tezlik kerak bo'lsa chaqiruvchi
    calibrate=False berishi mumkin (qidiruv natijasi amalda o'zgarmaydi)."""
    log = log_fn or noop_log
    progress = progress_fn or (lambda frac=0.0, msg="": None)
    _spec_kinds(name)                                     # noma'lum model => ValueError
    if isinstance(tuning, dict):
        tuning = TuningConfig.from_dict(tuning)
    clean, warns = validate_hyperparams({name: base_params})
    base = clean[name]
    for w in warns:
        log(f"  Tuning: {w}")
    scoring = tuning.scoring if tuning.scoring in ("roc_auc", "average_precision") else "roc_auc"
    if scoring != tuning.scoring:
        log(f"  OGOHLANTIRISH: noma'lum scoring {tuning.scoring!r}, 'roc_auc' ishlatildi.")
    n_iter = max(1, int(tuning.n_iter))
    y = np.asarray(dataset.y).ravel().astype(np.int64)
    groups = np.asarray(groups).ravel()
    if groups.size != y.size:
        raise ValueError(f"groups uzunligi ({groups.size}) dataset.y uzunligiga ({y.size}) teng emas.")
    seed = int(seed)
    check_cancel(cancel)

    is_patch_mode = name == "CNN" and base["mode"] == "patch2d"
    if is_patch_mode and getattr(dataset, "feature_stack", None) is None:
        return _fallback(base, scoring, "patch2d uchun dataset.feature_stack yo'q", name, log)
    space = dict(tuning.resolved_space(name))
    if name == "CNN" and not is_patch_mode:
        space.pop("window", None)                         # tabular1d da oyna ma'nosiz
    rng = np.random.default_rng(seed % (2 ** 32 - 1))
    try:
        sample_params(name, base, space, np.random.default_rng(0))    # oraliqlar yaroqliligini oldindan tekshirish
    except ValueError as e:
        return _fallback(base, scoring, f"qidiruv oralig'i yaroqsiz: {e}", name, log)

    splits, n_used, err = _inner_splits(y, groups, tuning.inner_splits, seed, log)
    if not splits:
        return _fallback(base, scoring, f"ichki spatial bo'linish imkonsiz: {err}", name, log)

    cands = _generate_candidates(name, base, space, n_iter, rng)
    n_cand = len(cands)
    if n_cand < n_iter:
        log(f"  Tuning {name}: qidiruv fazosi kichik - {n_cand} ta noyob nomzod (so'ralgan {n_iter}).")
    log(f"  Tuning {name}: {n_cand} nomzod x {n_used} ichki fold, ball: {scoring}.")
    if name == "CNN":
        log(f"  OGOHLANTIRISH: CNN tuning qimmat: {n_cand * n_used} ta o'qitish; TensorFlow xotirasi har "
            f"o'qitishda ~8 MB o'sadi - nomzodlar/fold sonini kamaytirish mumkin.")

    X = np.asarray(dataset.X)
    n_features = X.shape[1]
    warn_seen = set()

    def fit_log(msg):                                     # faqat (takrorlanmaydigan, cheklangan) ogohlantirishlar
        m = str(msg).strip()
        if "ohlantirish" in m.lower() and m not in warn_seen and len(warn_seen) < _MAX_FIT_WARNINGS:
            warn_seen.add(m)
            log(f"    [tuning {name}] {m}")

    patch_cache = {}                                      # window -> (n, w, w, p); faqat joriy window saqlanadi

    def patches_for(window):
        if window not in patch_cache:
            patch_cache.clear()
            patch_cache[window] = dataset.get_patches(window)
        return patch_cache[window]

    def evaluate(params):
        scores = []
        for fi, (tr, va) in enumerate(splits):
            check_cancel(cancel)
            if y[va].min() == y[va].max() or y[tr].min() == y[tr].max():
                continue                                  # bir sinfli fold - tashlanadi
            model = make_model(name, params, calibrate=calibrate, calibration_method=calibration_method,
                               calibration_cv=calibration_cv, n_jobs=n_jobs, seed=seed + fi,
                               n_features=n_features)
            if model.input_kind == "patch":
                pt = patches_for(int(params["window"]))
                model.fit(None, y[tr], patches=pt[tr], cancel=cancel, log_fn=fit_log)
                pred = model.predict_proba_pos(None, patches=pt[va], batch_size=_PRED_BATCH)
            else:
                model.fit(X[tr], y[tr], cancel=cancel, log_fn=fit_log)
                pred = model.predict_proba_pos(X[va], batch_size=_PRED_BATCH)
            s = _score(scoring, y[va], pred)
            if np.isfinite(s):
                scores.append(s)
        if not scores:
            return float("nan"), float("nan"), 0
        return float(np.mean(scores)), float(np.std(scores)), len(scores)

    # CNN patch2d: window bo'yicha guruhlab baholash (har window patchlari bir marta olinadi)
    order = list(range(n_cand))
    if is_patch_mode:
        first = {}
        for i, c in enumerate(cands):
            first.setdefault(int(c["window"]), i)
        order.sort(key=lambda i: (first[int(cands[i]["window"])], i))

    trials = [None] * n_cand
    err_seen = set()
    for done, i in enumerate(order, 1):
        check_cancel(cancel)
        try:
            mean, std, nf = evaluate(cands[i])
        except CancelledError:
            raise
        except Exception as e:                            # bitta nomzod yiqilsa ham qidiruv davom etadi
            mean, std, nf = float("nan"), float("nan"), 0
            msg = f"{type(e).__name__}: {e}"
            if msg not in err_seen and len(err_seen) < _MAX_FIT_WARNINGS:
                err_seen.add(msg)
                log(f"  OGOHLANTIRISH: {name} nomzodi ({_diff_text(cands[i], base)}) baholanmadi - {msg}")
        trials[i] = {"params": copy.deepcopy(cands[i]), "score": mean, "std": std, "n_folds": nf}
        progress(done / n_cand, f"Tuning {name}: nomzod {done}/{n_cand}")
        check_cancel(cancel)

    # tanlash: nomzodlar tartibida (base = 0), faqat qat'iy yaxshiroq almashtiradi => teng bo'lsa base afzal
    best_i = None
    for i, t in enumerate(trials):
        s = t["score"]
        if np.isfinite(s) and (best_i is None or s > trials[best_i]["score"] + _TIE_EPS):
            best_i = i
    base_score = trials[0]["score"]
    if best_i is None:
        log(f"  OGOHLANTIRISH: {name} uchun birorta nomzod baholanmadi - berilgan giperparametrlar ishlatiladi.")
        return {"best_params": copy.deepcopy(base), "best_score": float("nan"), "trials": trials,
                "n_trials": n_cand, "scoring": scoring, "base_score": base_score, "best_index": 0,
                "inner_splits_used": n_used}
    best = trials[best_i]
    log(f"  Tuning {name}: eng yaxshi {scoring}={best['score']:.4f}±{best['std']:.4f} "
        f"(base {base_score:.4f}); {_diff_text(best['params'], base)}")
    return {"best_params": copy.deepcopy(best["params"]), "best_score": float(best["score"]), "trials": trials,
            "n_trials": n_cand, "scoring": scoring, "base_score": base_score, "best_index": int(best_i),
            "inner_splits_used": n_used}
