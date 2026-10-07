# -*- coding: utf-8 -*-
"""
Fazoviy (spatial) blok CV yordamchilari: variogram bo'yicha blok o'lchamini baholash, nuqtalarni
bloklarga ajratish, musbat nuqtalarni saqlovchi guruhli (blok) fold'lar, blok-bootstrap indekslari.

Asosiy g'oya: bir-biriga yaqin nuqtalar (spatial autokorrelyatsiya tufayli deyarli bir xil qiymatli)
train va validation orasida bo'linib qolmasligi kerak - bloklar BUTUN holda bitta fold'ga tushadi.
Legacy kodga nisbatan tuzatishlar: BUG-05 (musbatsiz validation fold), BUG-09 (variogram faqat 1-band).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.spatial.distance import pdist
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

from .common import RANDOM_STATE, noop_log

__all__ = [
    "estimate_autocorrelation_range", "assign_spatial_blocks", "adapt_block_size",
    "stratified_group_splits", "random_stratified_splits", "block_bootstrap_indices", "fold_report",
]

_MIN_PAIRS = 30            # lag oralig'ida kamida shuncha juftlik bo'lishi kerak
_MIN_LAGS_USED = 3         # variogram uchun kamida shuncha yaroqli lag
_MAX_POINTS = 4000         # pdist xotirasini cheklash (4000 nuqta ~ 8 mln juftlik)
_MAX_LAG_FRAC = 1.0 / 3.0  # maksimal lag = hudud diagonalining 1/3
_STRUCT_MIN = 0.10         # nugget'dan tashqari struktura ulushi shundan kam bo'lsa - band tuzilmasiz
_N_RANGE_GRID = 80         # model range'ini qidirish to'ri
_MIN_RANGE_LAGS = 2.0      # range kamida shuncha lag kengligi: qisqaroq struktura nugget'dan ajralmaydi
_SEED_MOD = 2 ** 32        # sklearn/numpy seed chegarasi (manfiy yoki katta seed'lar ham ishlashi uchun)
_MAX_HALVINGS = 30         # adapt_block_size: blokni ko'pi bilan shuncha marta yarmiga kamaytirish
_MAX_ATTEMPTS = 50         # stratified_group_splits: har repeat uchun qayta urinishlar soni
_REPORT_COLUMNS = ["repeat", "fold", "n_train", "n_val", "pos_train", "pos_val", "n_blocks_val"]


# ---------------------------------------------------------------------------
# Ichki yordamchilar
# ---------------------------------------------------------------------------
def _as_coords(coords):
    """(n, 2) chekli float64 massivi; aks holda ValueError."""
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"coords shakli (n, 2) bo'lishi kerak, berildi: {c.shape}.")
    if not np.all(np.isfinite(c)):
        raise ValueError("coords ichida NaN/inf bor - avval yaroqsiz nuqtalarni olib tashlang.")
    return c


def _as_binary_labels(y, n=None):
    """y -> int64 (n,), faqat 0/1 qiymatlar."""
    y = np.asarray(y).reshape(-1)
    if n is not None and y.size != n:
        raise ValueError(f"y uzunligi ({y.size}) nuqtalar soniga ({n}) teng emas.")
    if y.size and not np.all((y == 0) | (y == 1)):
        raise ValueError("y faqat 0 (fon) va 1 (musbat) qiymatlardan iborat bo'lishi kerak.")
    return y.astype(np.int64)


def _require_background(y):
    if not np.any(y == 0):
        raise ValueError("Fon (0) nuqtalar yo'q: faqat musbat nuqtalar bilan validatsiya (AUC) imkonsiz. "
                         "Fon nuqtalar sonini oshiring.")


def _check_splits_args(n_splits, n_repeats):
    n_splits, n_repeats = int(n_splits), int(n_repeats)
    if n_splits < 2:
        raise ValueError(f"k-fold soni (n_splits) kamida 2 bo'lishi kerak, berildi: {n_splits}.")
    if n_repeats < 1:
        raise ValueError(f"Takrorlash soni (n_repeats) kamida 1 bo'lishi kerak, berildi: {n_repeats}.")
    return n_splits, n_repeats


# ---------------------------------------------------------------------------
# Variogram: avtokorrelyatsiya masofasi (range)
# ---------------------------------------------------------------------------
def _pixel_xy(transform, rows, cols):
    """Piksel markazlari koordinatalari (affin transformatsiya bo'yicha, vektorlashtirilgan)."""
    c = np.asarray(cols, dtype=np.float64) + 0.5
    r = np.asarray(rows, dtype=np.float64) + 0.5
    x = transform.a * c + transform.b * r + transform.c
    y = transform.d * c + transform.e * r + transform.f
    return x, y


def _region_diagonal(transform, valid):
    """Valid piksellar chegaraviy to'rtburchagi (bbox) diagonali, metr."""
    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    ec = float(cols[-1] - cols[0] + 1)
    er = float(rows[-1] - rows[0] + 1)
    dx = transform.a * ec + transform.b * er
    dy = transform.d * ec + transform.e * er
    return float(np.hypot(dx, dy))


def _empirical_variogram(coords, z, max_lag, n_lags):
    """Klassik (Matheron) empirik semivariogram. Masofa bo'yicha n_lags ta teng lag oralig'i;
    juftligi _MIN_PAIRS dan kam oraliqlar tashlanadi. Qaytaradi: (h, gamma, juftliklar soni)."""
    dist = pdist(coords)
    dz2 = pdist(z.reshape(-1, 1), "sqeuclidean")
    idx = np.floor(dist / (max_lag / n_lags)).astype(np.int64)
    ok = (idx < n_lags) & (dist > 0)
    idx, dist, dz2 = idx[ok], dist[ok], dz2[ok]
    cnt = np.bincount(idx, minlength=n_lags)
    sum_h = np.bincount(idx, weights=dist, minlength=n_lags)
    sum_g = np.bincount(idx, weights=dz2, minlength=n_lags)
    use = cnt >= _MIN_PAIRS
    h = sum_h[use] / cnt[use]           # lag markazi o'rniga haqiqiy o'rtacha masofa
    g = 0.5 * sum_g[use] / cnt[use]
    n = cnt[use]
    fin = np.isfinite(h) & np.isfinite(g)
    return h[fin], g[fin], n[fin]


def _spherical(h, a):
    r = np.minimum(h / a, 1.0)
    return 1.5 * r - 0.5 * r ** 3


def _fit_spherical(h, g, n, max_lag, min_range=0.0):
    """Sferik modelni (nugget + sill*sph(h/a)) Cressie og'irliklari (n / gamma^2) bilan eng kichik
    kvadratlar usulida moslaydi. Berilgan a uchun model chiziqli (nugget, sill >= 0 -> nnls), a esa
    to'r bo'yicha qidiriladi: konvergensiya muammosi yo'q. a to'ri max(h[0], min_range) dan boshlanadi:
    birinchi lagdan qisqa range'da model ustunlari bir xil (nugget va sill ajralmaydi) va sof shovqin
    ham 'struktura' bo'lib chiqardi. Qaytaradi: (a, nugget, sill_qismi) yoki None."""
    floor = max(0.02 * float(g.max()), 1e-12)
    sw = np.sqrt(n / np.maximum(g, floor) ** 2)
    rhs = g * sw
    best = None
    a_lo = min(max(float(h[0]), float(min_range)), max_lag)
    for a in np.linspace(a_lo, max_lag, _N_RANGE_GRID):
        design = np.column_stack([np.ones_like(h), _spherical(h, a)]) * sw[:, None]
        try:
            coef, resid = nnls(design, rhs)
        except Exception:
            continue
        if best is None or resid < best[0]:
            best = (resid, float(a), float(coef[0]), float(coef[1]))
    if best is None:
        return None
    return best[1:]


def _band_range(band, transform, n_lags, n_points, rng):
    """Bitta band uchun avtokorrelyatsiya masofasi. Qaytaradi: (range_m|None, izoh)."""
    valid = np.isfinite(band)
    n_valid = int(valid.sum())
    if n_valid < 50:
        return None, "yetarli valid piksel yo'q"
    diag = _region_diagonal(transform, valid)
    max_lag = diag * _MAX_LAG_FRAC
    pix = float(np.hypot(transform.a, transform.d))
    if not np.isfinite(max_lag) or max_lag < 3 * pix:
        return None, "hudud juda kichik"

    flat = np.flatnonzero(valid)
    if flat.size > n_points:
        flat = flat[rng.choice(flat.size, size=n_points, replace=False)]
    rows, cols = np.divmod(flat, band.shape[1])
    z = band[rows, cols].astype(np.float64)
    lo, hi = np.percentile(z, [0.5, 99.5])       # kam sonli keskin chetlanishlardan himoya
    zc = np.clip(z, lo, hi)
    if zc.std() > 0:
        z = zc
    std = z.std()
    if not np.isfinite(std) or std <= 0:
        return None, "band deyarli o'zgarmas"
    z = (z - z.mean()) / std
    x, y = _pixel_xy(transform, rows, cols)

    h, g, n = _empirical_variogram(np.column_stack([x, y]), z, max_lag, n_lags)
    if h.size < _MIN_LAGS_USED:
        return None, "yetarli lag hosil bo'lmadi"
    fit = _fit_spherical(h, g, n, max_lag, min_range=_MIN_RANGE_LAGS * max_lag / n_lags)
    if fit is None:
        return None, "variogram modeli moslanmadi"
    a, nugget, psill = fit
    sill = nugget + psill
    if not np.isfinite(sill) or sill <= 0 or psill / sill < _STRUCT_MIN:
        return None, "fazoviy struktura yo'q (sof nugget)"
    return float(a), ""


def estimate_autocorrelation_range(stack, transform, band_indices=None, n_lags=15, n_points=1500,
                                   random_state=RANDOM_STATE, log_fn=None):
    """Predictor bandlarining fazoviy avtokorrelyatsiya masofasini (range, metr) baholaydi.

    Har bir band (band_indices None bo'lsa - barchasi) uchun: n_points ta tasodifiy valid piksel,
    barcha juftliklar bo'yicha empirik semivariogram (maksimal lag = hudud diagonalining 1/3),
    unga sferik model moslanadi va uning range'i olinadi. Natija - bandlar range'larining MEDIANASI.
    Fazoviy strukturasiz (sof shovqin, o'zgarmas) bandlar e'tiborga olinmaydi. Hech biri
    baholanmasa None qaytaradi. Natija blok o'lchami uchun boshlang'ich taklif; tasdiqlang.
    """
    log_fn = log_fn or noop_log
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"stack shakli (n_bands, H, W) bo'lishi kerak, berildi: {stack.shape}.")
    n_bands = stack.shape[0]
    n_lags = int(n_lags)
    if n_lags < 4:
        raise ValueError(f"n_lags kamida 4 bo'lishi kerak, berildi: {n_lags}.")
    n_points = int(min(max(int(n_points), 50), _MAX_POINTS))

    if band_indices is None:
        bands = list(range(n_bands))
    else:
        bands = [int(b) for b in np.atleast_1d(band_indices)]
        bad = [b for b in bands if not 0 <= b < n_bands]
        if bad:
            raise ValueError(f"band_indices noto'g'ri: {bad} (bandlar soni {n_bands}).")
    if not bands:
        log_fn("  Variogram: tanlangan band yo'q, standart blok o'lchami ishlatiladi.")
        return None

    seed = int(random_state) & 0xFFFFFFFF
    ranges, skipped = [], []
    for b in bands:
        rng = np.random.default_rng([seed, b])
        r, why = _band_range(stack[b], transform, n_lags, n_points, rng)
        if r is None:
            skipped.append(f"{b}: {why}")
        else:
            ranges.append(r)

    if skipped:
        log_fn(f"  Variogram: {len(skipped)} band e'tiborga olinmadi ({'; '.join(skipped[:5])}"
               f"{'...' if len(skipped) > 5 else ''}).")
    if not ranges:
        log_fn("  Variogram: birorta band baholanmadi, standart blok o'lchami ishlatiladi.")
        return None

    est = float(np.median(ranges))
    log_fn(f"  Empirik semivariogram: avtokorrelyatsiya range ≈ {est:,.0f} m "
           f"(mediana, {len(ranges)}/{len(bands)} band; min {min(ranges):,.0f}, max {max(ranges):,.0f} m). "
           f"Spatial block CV uchun boshlang'ich blok o'lchami sifatida taklif qilinadi - "
           f"yakuniy qiymatni tekshirib tasdiqlang.")
    return est


# ---------------------------------------------------------------------------
# Bloklar
# ---------------------------------------------------------------------------
def assign_spatial_blocks(coords, block_size):
    """Har nuqtani kvadrat blokka biriktiradi: blok = (floor(x/block_size), floor(y/block_size)).
    Qaytaradi: (n,) int64, bloklar 0..n_blocks-1 ga zich raqamlangan. Bir blokdagi nuqtalar CV'da
    har doim bitta fold'da qoladi (train/val orasida fazoviy oqish yo'q)."""
    coords = _as_coords(coords)
    block_size = float(block_size)
    if not np.isfinite(block_size) or block_size <= 0:
        raise ValueError(f"Blok o'lchami musbat son bo'lishi kerak, berildi: {block_size}.")
    if coords.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    scaled = coords / block_size
    if np.abs(scaled).max() >= 2.0 ** 62:
        raise ValueError(f"Blok o'lchami ({block_size:g} m) koordinatalarga nisbatan juda kichik: "
                         "blok raqamlari butun songa sig'maydi.")
    cell = np.floor(scaled).astype(np.int64)
    _, groups = np.unique(cell, axis=0, return_inverse=True)
    return np.asarray(groups, dtype=np.int64).reshape(-1)


def adapt_block_size(coords, y, n_splits, block_size, min_size=10.0, log_fn=None):
    """Blok o'lchamini moslashtiradi: bloklar soni >= n_splits VA musbat nuqtali bloklar >= n_splits
    bo'lguncha o'lchamni yarmiga kamaytiradi (ko'pi bilan 30 marta, min_size dan kichik bo'lmaydi).
    Qaytaradi: (block_size, groups). Imkonsiz bo'lsa - aniq xabarli ValueError."""
    log_fn = log_fn or noop_log
    coords = _as_coords(coords)
    y = _as_binary_labels(y, len(coords))
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError(f"k-fold soni (n_splits) kamida 2 bo'lishi kerak, berildi: {n_splits}.")
    bs = float(block_size)
    if not np.isfinite(bs) or bs <= 0:
        raise ValueError(f"Blok o'lchami musbat son bo'lishi kerak, berildi: {bs}.")

    pos = y == 1
    n_pos = int(pos.sum())
    if n_pos < n_splits:
        raise ValueError(f"Musbat nuqtalar soni ({n_pos}) k-fold sonidan ({n_splits}) kam: har validation "
                         f"fold'ida musbat nuqta bo'lishi uchun kamida {n_splits} ta musbat nuqta kerak. "
                         "k-fold sonini kamaytiring yoki musbat nuqtalar qo'shing.")
    _require_background(y)
    n_pos_xy = len(np.unique(coords[pos], axis=0))
    if n_pos_xy < n_splits:
        raise ValueError(f"Musbat nuqtalarning turli joylari soni ({n_pos_xy}) k-fold sonidan ({n_splits}) kam "
                         "(koordinatalari bir xil nuqtalar bitta blokka tushadi). k-fold sonini kamaytiring.")

    start = bs
    n_blocks = n_pos_blocks = 0
    for step in range(_MAX_HALVINGS + 1):
        groups = assign_spatial_blocks(coords, bs)
        n_blocks = int(groups.max()) + 1 if groups.size else 0
        n_pos_blocks = int(np.unique(groups[pos]).size)
        if n_blocks >= n_splits and n_pos_blocks >= n_splits:
            if step:
                log_fn(f"  Blok o'lchami {start:,.0f} m dan {bs:,.0f} m gacha kamaytirildi ({step} marta "
                       f"yarmiga): bloklar {n_blocks}, musbat nuqtali bloklar {n_pos_blocks} "
                       f"(k-fold = {n_splits}).")
            return bs, groups
        if step == _MAX_HALVINGS or bs / 2.0 < float(min_size):
            break
        bs /= 2.0
    raise ValueError(f"Blok o'lchamini {start:,.0f} m dan {bs:,.0f} m gacha kamaytirib ham yetarli bloklar "
                     f"hosil bo'lmadi: bloklar {n_blocks}, musbat nuqtali bloklar {n_pos_blocks}, "
                     f"k-fold = {n_splits}. k-fold sonini kamaytiring, musbat nuqtalar qo'shing yoki "
                     "min. blok o'lchamini (min_size) kichraytiring.")


# ---------------------------------------------------------------------------
# Fold'lar
# ---------------------------------------------------------------------------
def stratified_group_splits(y, groups, n_splits, n_repeats, random_state=RANDOM_STATE,
                            log_fn=None, max_attempts=_MAX_ATTEMPTS):
    """Blok (guruh) bo'yicha stratifikatsiyalangan k-fold, n_repeats marta takrorlanadi.

    Generator: (repeat, fold, train_idx, val_idx). Har repeat'da yangi seed
    (random_state + repeat*1000 + urinish) bilan StratifiedGroupKFold(shuffle=True). Har validation
    fold'da kamida 1 musbat bo'lishi kerak: bo'lmasa boshqa seed bilan qayta uriniladi (max_attempts
    marta); oxirida ham bo'lmasa musbat bloki >= 2 ta fold'dan butun blok musbatsiz fold'ga ko'chiriladi
    (musbat bloklar >= n_splits bo'lgani uchun odatda mumkin), buning ham iloji bo'lmasa log_fn orqali
    ogohlantirib, eng yaxshi bo'linish ishlatiladi. Bitta blok HECH QACHON train va val orasida
    bo'linmaydi. Argumentlar chaqiruv paytidayoq tekshiriladi: musbat nuqtali bloklar < n_splits yoki
    fon nuqtalar yo'q bo'lsa aniq ValueError.
    """
    log_fn = log_fn or noop_log
    n_splits, n_repeats = _check_splits_args(n_splits, n_repeats)
    groups = np.asarray(groups).reshape(-1)
    y = _as_binary_labels(y, groups.size)
    max_attempts = max(1, int(max_attempts))
    n_blocks = int(np.unique(groups).size)
    n_pos_blocks = int(np.unique(groups[y == 1]).size)
    if n_blocks < n_splits:
        raise ValueError(f"Spatial bloklar soni ({n_blocks}) k-fold sonidan ({n_splits}) kam. "
                         "Blok o'lchamini kichraytiring yoki k-fold sonini kamaytiring.")
    if n_pos_blocks < n_splits:
        raise ValueError(f"Musbat nuqtali bloklar soni ({n_pos_blocks}) k-fold sonidan ({n_splits}) kam: "
                         "har validation fold'ida musbat nuqta bo'lishi mumkin emas. Blok o'lchamini "
                         "kichraytiring, k-fold sonini kamaytiring yoki musbat nuqtalar qo'shing.")
    _require_background(y)
    return _iter_group_splits(y, groups, n_splits, n_repeats, int(random_state), log_fn, max_attempts)


def _count_no_positive(folds, y):
    return sum(1 for _, val in folds if not np.any(y[val] == 1))


def _repair_positive_folds(folds, y, groups):
    """Musbatsiz validation fold'larni tuzatadi: musbat bloki >= 2 ta bo'lgan fold'dan eng kichik musbat
    blokni BUTUNLAY musbatsiz fold'ga ko'chiradi (blok baribir bo'linmaydi). Musbat bloklar >= n_splits
    bo'lsa har doim muvaffaqiyatli; fold'lar blok bo'yicha izchil bo'lmasa o'zgartirilmaydi."""
    n_splits = len(folds)
    _, gi = np.unique(groups, return_inverse=True)
    gi = np.asarray(gi).reshape(-1)
    label = np.full(len(y), -1, dtype=np.int64)
    for k, (_, val) in enumerate(folds):
        label[val] = k
    fold_of = np.full(int(gi.max()) + 1, -1, dtype=np.int64)       # blok -> validation fold
    fold_of[gi] = label
    if np.any(label < 0) or not np.array_equal(fold_of[gi], label):
        return folds
    pos_cnt = np.bincount(gi[y == 1], minlength=fold_of.size)       # blokdagi musbatlar soni
    is_pos = pos_cnt > 0
    while True:
        pos_blocks = np.bincount(fold_of[is_pos], minlength=n_splits)
        empty = np.flatnonzero(np.bincount(fold_of, weights=pos_cnt, minlength=n_splits) == 0)
        donors = np.flatnonzero(pos_blocks >= 2)
        if empty.size == 0 or donors.size == 0:
            break
        donor = donors[np.argmax(pos_blocks[donors])]
        cand = np.flatnonzero((fold_of == donor) & is_pos)
        fold_of[cand[np.argmin(pos_cnt[cand])]] = empty[0]
    label = fold_of[gi]
    idx = np.arange(len(y))
    return [(idx[label != k], idx[label == k]) for k in range(n_splits)]


def _iter_group_splits(y, groups, n_splits, n_repeats, seed, log_fn, max_attempts):
    dummy = np.zeros(len(y))
    for repeat in range(n_repeats):
        best, best_bad = None, None
        for attempt in range(max_attempts):
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=(seed + repeat * 1000 + attempt) % _SEED_MOD)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)   # "least populated class" ogohlantirishi
                folds = list(sgkf.split(dummy, y, groups))
            bad = _count_no_positive(folds, y)
            if best is None or bad < best_bad:
                best, best_bad = folds, bad
            if bad == 0:
                break
        if best_bad:
            fixed = _repair_positive_folds(best, y, groups)
            fixed_bad = _count_no_positive(fixed, y)
            if fixed_bad < best_bad:
                log_fn(f"  {repeat + 1}-takrorda {best_bad} ta validation fold'ida musbat nuqta yo'q edi "
                       f"({max_attempts} urinishdan keyin): musbat bloklar fold'lar orasida qayta taqsimlandi.")
                best, best_bad = fixed, fixed_bad
        if best_bad:
            log_fn(f"  Ogohlantirish: {repeat + 1}-takrorda {best_bad} ta validation fold'ida musbat nuqta "
                   f"yo'q ({max_attempts} urinishdan keyin); eng yaxshi bo'linish ishlatildi.")
        for fold, (train_idx, val_idx) in enumerate(best):
            yield repeat, fold, train_idx, val_idx


def random_stratified_splits(y, n_splits, n_repeats, random_state=RANDOM_STATE):
    """Oddiy (fazoviy bo'lmagan) takroriy stratifikatsiyalangan k-fold - benchmark uchun.
    stratified_group_splits bilan bir xil format: generator (repeat, fold, train_idx, val_idx);
    deterministik (RepeatedStratifiedKFold). Musbatlar soni < n_splits bo'lsa ValueError."""
    n_splits, n_repeats = _check_splits_args(n_splits, n_repeats)
    y = _as_binary_labels(y)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if min(n_pos, n_neg) < n_splits:
        raise ValueError(f"Kichik sinf nuqtalari soni ({min(n_pos, n_neg)}: musbat {n_pos}, fon {n_neg}) "
                         f"k-fold sonidan ({n_splits}) kam. k-fold sonini kamaytiring.")
    return _iter_random_splits(y, n_splits, n_repeats, int(random_state))


def _iter_random_splits(y, n_splits, n_repeats, seed):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed % _SEED_MOD)
    for i, (train_idx, val_idx) in enumerate(rskf.split(np.zeros(len(y)), y)):
        yield i // n_splits, i % n_splits, train_idx, val_idx


# ---------------------------------------------------------------------------
# Blok-bootstrap
# ---------------------------------------------------------------------------
def block_bootstrap_indices(groups, n_boot, random_state=RANDOM_STATE):
    """Blok-bootstrap: generator, har bootstrap uchun nuqta indekslari. Har safar bloklar soniga teng
    miqdorda blok ALMASHTIRISH bilan tanlanadi va ularning barcha nuqtalari olinadi (takroriy
    bloklar takroran). Deterministik (random_state)."""
    groups = np.asarray(groups).reshape(-1)
    n_boot = int(n_boot)
    if groups.size == 0:
        raise ValueError("groups bo'sh: bootstrap uchun nuqtalar yo'q.")
    if n_boot < 1:
        raise ValueError(f"Bootstrap soni (n_boot) kamida 1 bo'lishi kerak, berildi: {n_boot}.")
    return _iter_bootstrap(groups, n_boot, int(random_state))


def _iter_bootstrap(groups, n_boot, seed):
    order = np.argsort(groups, kind="stable")
    _, starts, counts = np.unique(groups[order], return_index=True, return_counts=True)
    n_blocks = len(starts)
    rng = np.random.default_rng(seed % _SEED_MOD)
    for _ in range(n_boot):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        lens = counts[chosen]
        cum = np.cumsum(lens)
        base = np.repeat(starts[chosen] - (cum - lens), lens)
        yield order[base + np.arange(cum[-1])]


# ---------------------------------------------------------------------------
# Diagnostika
# ---------------------------------------------------------------------------
def fold_report(y, coords, groups, splits):
    """Fold'lar jadvali: DataFrame(repeat, fold, n_train, n_val, pos_train, pos_val, n_blocks_val).
    splits - (repeat, fold, train_idx, val_idx) iteratori yoki ro'yxati; groups None bo'lsa har nuqta
    alohida blok hisoblanadi."""
    y = np.asarray(y).reshape(-1)
    if coords is not None and len(coords) != y.size:
        raise ValueError(f"coords uzunligi ({len(coords)}) y uzunligiga ({y.size}) teng emas.")
    if groups is not None:
        groups = np.asarray(groups).reshape(-1)
        if groups.size != y.size:
            raise ValueError(f"groups uzunligi ({groups.size}) y uzunligiga ({y.size}) teng emas.")
    rows = []
    for repeat, fold, train_idx, val_idx in splits:
        train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
        rows.append({
            "repeat": int(repeat), "fold": int(fold),
            "n_train": int(train_idx.size), "n_val": int(val_idx.size),
            "pos_train": int(np.sum(y[train_idx] == 1)), "pos_val": int(np.sum(y[val_idx] == 1)),
            "n_blocks_val": int(np.unique(groups[val_idx]).size) if groups is not None else int(val_idx.size),
        })
    return pd.DataFrame(rows, columns=_REPORT_COLUMNS).astype("int64")
