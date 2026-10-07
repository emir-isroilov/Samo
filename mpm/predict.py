# -*- coding: utf-8 -*-
"""
Prognoz xaritalari: butun maydon bo'yicha bashorat, sinflash, sinf maydon statistikasi, success-rate egri
chizig'i va GeoTIFF saqlash.

Chiqish "ehtimollik" emas, "prospektivlik indeksi (0-1)" (musbat:fon nisbati sun'iy).

Asl koddagi tuzatilgan xatolar:
  BUG-07  bashorat batch_size=8192 bilan (CNN predict standart batch=32 edi); butun (n_pixels, n_features)
          massiv bir yo'la yaratilmaydi - faqat valid indekslar bo'yicha batch;
  BUG-14  ansambl nanmean ogohlantirishsiz (butunlay NaN piksel NaN qoladi), toza GeoTIFF profili
          (tiled/blocksize ko'chirilmaydi), nodata moslashtirilgan (float32 -9999, int8 sinf xaritasi 0).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import rasterio

from .base import ModelWrapper
from .common import ENSEMBLE_NAME, NODATA_OUT, check_cancel, noop_log, safe_name
from .config import CLASS_METHODS
from .data import extract_patches, valid_pixel_mask

DEFAULT_BATCH = 8192
_PATCH_BATCH_ELEMS = 1 << 24      # patch batch'idagi maks. element soni (~64 MB float32): katta oyna uchun batch kichrayadi
_MAX_CLASSES = 127                # int8 sinf xaritasi
_CLASS_LABELS_5 = ("Juda past", "Past", "O'rta", "Yuqori", "Juda yuqori")
_STATS_COLUMNS = ["Sinf", "Nomi", "Piksel", "Maydon_km2", "Maydon_%", "Konlar_soni", "Konlar_%", "Boyitish"]


# ---------------------------------------------------------------------------
# Xarita bashorati
# ---------------------------------------------------------------------------
def _as_stack(raster):
    arr = np.asarray(getattr(raster, "stack", raster))
    if arr.ndim != 3:
        raise ValueError(f"raster stack 3 o'lchamli (n_bands, H, W) bo'lishi kerak, berilgan: {arr.shape}")
    return arr


def _check_batch(batch_size):
    try:
        bs = int(batch_size)
    except (TypeError, ValueError):
        raise ValueError(f"batch_size butun son bo'lishi kerak, berilgan {batch_size!r}") from None
    if bs < 1:
        raise ValueError(f"batch_size >= 1 bo'lishi kerak, berilgan {batch_size}")
    return bs


def _normalize_models(final_models):
    """{name: [wrapper, ...]} - bo'sh emasligi, o'qitilganligi va ansambl nomi band emasligini tekshiradi."""
    if not isinstance(final_models, dict) or not final_models:
        raise ValueError("final_models bo'sh: kamida bitta o'qitilgan model kerak ({nom: [model, ...]}).")
    out = {}
    for name, draws in final_models.items():
        if name == ENSEMBLE_NAME:
            raise ValueError(f"'{ENSEMBLE_NAME}' model nomi sifatida ishlatilishi mumkin emas (ansambl uchun band).")
        if isinstance(draws, ModelWrapper):
            draws = [draws]
        draws = list(draws)
        if not draws:
            raise ValueError(f"{name}: modellar ro'yxati bo'sh.")
        for m in draws:
            if not getattr(m, "is_fitted", False):
                raise ValueError(f"{name}: model o'qitilmagan (is_fitted=False).")
        out[name] = draws
    return out


def _plan_jobs(models):
    """Modellarni (nom, kind, window) bo'yicha guruhlaydi: bir guruh bir xil kirish (X yoki bir oyna patchlari)
    ishlatadi. Qaytaradi: tabular guruhlar ro'yxati, {window: guruhlar} (patch), nom -> jami draw soni."""
    tab, patch = [], {}
    for name, draws in models.items():
        groups = {}
        for m in draws:
            kind = getattr(m, "input_kind", "tabular")
            if kind == "patch":
                window = (getattr(m, "params", None) or {}).get("window")
                if window is None:
                    raise ValueError(f"{name}: patch modelida params['window'] yo'q.")
                key = ("patch", int(window))
            elif kind == "tabular":
                key = ("tabular", 0)
            else:
                raise ValueError(f"{name}: noma'lum input_kind {kind!r}.")
            groups.setdefault(key, []).append(m)
        total = len(draws)
        for (kind, window), ms in groups.items():
            job = {"name": name, "models": ms, "weight": len(ms) / total}
            if kind == "tabular":
                tab.append(job)
            else:
                patch.setdefault(window, []).append(job)
    return tab, patch


def _accumulate(flat, idx, job, preds_fn):
    """Guruh draw'lari o'rtachasini (vazn bilan) xarita massiviga qo'shadi."""
    acc = None
    for m in job["models"]:
        p = np.asarray(preds_fn(m), dtype=np.float64).reshape(-1)
        acc = p if acc is None else acc + p
    flat[idx] += (acc * (job["weight"] / len(job["models"]))).astype(np.float32)


def _nanmean_std(arrs, shape):
    """Xaritalar bo'yicha nanmean va (>=2 model bo'lsa) std; ogohlantirishsiz, xotira tejamkor.
    Butunlay NaN piksel NaN qoladi; std ikkitadan kam chekli modelli pikselda NaN."""
    total = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int16)
    for a in arrs:
        ok = np.isfinite(a)
        total[ok] += a[ok]
        count += ok
    has = count > 0
    mean = np.full(shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=mean, where=has)
    std = None
    if len(arrs) >= 2:
        sq = np.zeros(shape, dtype=np.float64)
        for a in arrs:
            ok = np.isfinite(a)
            sq[ok] += (a[ok].astype(np.float64) - mean[ok]) ** 2
        std = np.full(shape, np.nan, dtype=np.float64)
        np.divide(sq, count, out=std, where=count >= 2)
        np.sqrt(std, out=std, where=count >= 2)
        std = std.astype(np.float32)
    return mean.astype(np.float32), std


def predict_probability_maps(raster, pipeline, final_models, *, batch_size=DEFAULT_BATCH, log_fn=None,
                             progress_fn=None, cancel=None):
    """
    Butun maydon bo'yicha prospektivlik indeksi xaritalari.

    final_models: {nom: [ModelWrapper, ...]} - har model xaritasi bg-draw modellari o'rtachasi.
    Faqat valid (barcha bandlar chekli) piksellar bashorat qilinadi, batch_size bo'yicha bo'laklab.
    Tabular modellar: pipeline.transform_pixels(batch); patch modellar (CNN patch2d): feature stack
    (transform_stack) BIR marta, so'ng extract_patches(batch).

    Qaytaradi: {"maps": {nom: (H,W) float32 NaN-nodata, ENSEMBLE_NAME ham (nanmean)},
                "uncertainty": (H,W) float32 (alohida modellar xaritalari std'i; <2 model bo'lsa None),
                "valid_mask": (H,W) bool}
    """
    log = log_fn or noop_log
    report = progress_fn or (lambda frac=0.0, msg="": None)
    bs = _check_batch(batch_size)
    stack = _as_stack(raster)
    n_bands, height, width = stack.shape
    if list(pipeline.band_names) != list(getattr(raster, "band_names", pipeline.band_names)):
        raise ValueError("FeaturePipeline band_names raster band_names bilan mos emas.")
    if len(pipeline.band_names) != n_bands:
        raise ValueError(f"pipeline bandlari ({len(pipeline.band_names)}) raster bandlariga ({n_bands}) mos emas.")
    models = _normalize_models(final_models)
    check_cancel(cancel)

    valid = valid_pixel_mask(stack)
    idx_all = np.flatnonzero(valid.ravel())
    n_valid = int(idx_all.size)
    if n_valid == 0:
        raise ValueError("Bashorat uchun yaroqli piksel yo'q: barcha qatlamlar bir vaqtda chekli bo'lgan piksel "
                         "topilmadi (qatlamlar kesishmasligini yoki nodata'ni tekshiring).")
    desc = ", ".join(f"{n} ({len(d)} ta)" for n, d in models.items())
    log(f"Bashorat: {n_valid:,} piksel (jami {height * width:,} dan), batch={bs}; modellar: {desc}")

    tab_jobs, patch_jobs = _plan_jobs(models)
    flats = {}
    for name in models:
        f = np.full(height * width, np.nan, dtype=np.float32)
        f[idx_all] = 0.0
        flats[name] = f

    # --- ish rejasi (progress uchun): har (batch, guruh) bitta birlik
    n_batches = int(np.ceil(n_valid / bs))
    patch_bs = {}
    if patch_jobs:
        n_feat = pipeline.n_features
        for window in patch_jobs:
            patch_bs[window] = max(1, min(bs, _PATCH_BATCH_ELEMS // (window * window * n_feat)))
    total = n_batches * len(tab_jobs) + sum(
        int(np.ceil(n_valid / patch_bs[w])) * len(js) for w, js in patch_jobs.items())
    done = 0

    def step(label):
        nonlocal done
        done += 1
        report(done / total, f"Bashorat: {label} ({done}/{total})")

    # --- tabular modellar: har batch uchun X bir marta hisoblanadi, barcha tabular modellarga beriladi
    if tab_jobs:
        for b, s in enumerate(range(0, n_valid, bs)):
            check_cancel(cancel)
            idx = idx_all[s:s + bs]
            rr, cc = np.divmod(idx, width)
            pix = np.ascontiguousarray(stack[:, rr, cc].T)
            X = pipeline.transform_pixels(pix)
            del pix
            for job in tab_jobs:
                check_cancel(cancel)
                _accumulate(flats[job["name"]], idx, job,
                            lambda m: m.predict_proba_pos(X, batch_size=bs))
                step(f"{job['name']}, batch {b + 1}/{n_batches}")
            del X

    # --- patch modellar: feature stack bir marta; har oyna uchun o'z batch o'lchami
    if patch_jobs:
        check_cancel(cancel)
        fstack = pipeline.transform_stack(stack)
        for window, jobs in patch_jobs.items():
            pbs = patch_bs[window]
            n_pb = int(np.ceil(n_valid / pbs))
            for b, s in enumerate(range(0, n_valid, pbs)):
                check_cancel(cancel)
                idx = idx_all[s:s + pbs]
                rr, cc = np.divmod(idx, width)
                patches = extract_patches(fstack, rr, cc, window)
                for job in jobs:
                    check_cancel(cancel)
                    _accumulate(flats[job["name"]], idx, job,
                                lambda m: m.predict_proba_pos(None, patches=patches, batch_size=pbs))
                    step(f"{job['name']}, oyna {window}, batch {b + 1}/{n_pb}")
                del patches
        del fstack

    maps = {name: flats[name].reshape(height, width) for name in models}
    del flats
    arrs = [maps[n] for n in models]
    ens, unc = _nanmean_std(arrs, (height, width))
    maps[ENSEMBLE_NAME] = ens
    n_bad = int((valid & ~np.isfinite(ens)).sum())
    if n_bad:
        log(f"  OGOHLANTIRISH: {n_bad:,} valid pikselda ansambl bashorati chekli emas (NaN qoldirildi).")
    log("Bashorat tayyor." + ("" if unc is not None else " (noaniqlik xaritasi uchun kamida 2 model kerak)"))
    return {"maps": maps, "uncertainty": unc, "valid_mask": valid}


# ---------------------------------------------------------------------------
# Sinflash
# ---------------------------------------------------------------------------
def CLASS_LABELS(n):
    """Sinf nomlari: n=5 => ["Juda past", "Past", "O'rta", "Yuqori", "Juda yuqori"], aks holda "1-sinf", "2-sinf", ..."""
    n = int(n)
    if n < 1:
        raise ValueError(f"Sinflar soni kamida 1 bo'lishi kerak, berilgan {n}")
    if n == 5:
        return list(_CLASS_LABELS_5)
    return [f"{i}-sinf" for i in range(1, n + 1)]


def classify_map(prob_map, method="quantile", n_classes=5, breaks=None):
    """
    Prospektivlik indeksi xaritasini sinflarga bo'ladi. Qaytaradi: (class_map int8: 0 = nodata, 1..n; breaks_used).
      "quantile"       - teng maydonli sinflar (chegaralar valid piksellar kvantillari; qiymatlar takrorlansa
                         ba'zi sinflar bo'sh qolishi mumkin);
      "equal_interval" - [0, 1] ni n ta teng bo'lakka;
      "fixed"          - breaks: qat'iy o'suvchi chegaralar, sinflar soni len(breaks)+1 (n_classes e'tiborsiz).
    Qiymat chegaraga teng bo'lsa yuqori sinfga kiradi. NaN/cheksiz piksellar 0 (nodata).
    """
    if method not in CLASS_METHODS:
        raise ValueError(f"Sinflash usuli noto'g'ri: {method!r}; mumkin: {list(CLASS_METHODS)}")
    arr = np.asarray(prob_map)
    if arr.dtype.kind not in "fiub":
        raise ValueError(f"prob_map raqamli bo'lishi kerak, berilgan dtype: {arr.dtype}")
    ok = np.isfinite(arr)
    v = arr[ok]
    if method == "fixed":
        if breaks is None:
            raise ValueError("'fixed' sinflash uchun breaks (o'suvchi chegaralar) kerak.")
        b = np.asarray(breaks, dtype=np.float64).ravel()
        if b.size < 1:
            raise ValueError("'fixed' sinflash uchun kamida bitta chegara kerak.")
        if not np.all(np.isfinite(b)) or np.any(np.diff(b) <= 0):
            raise ValueError(f"breaks chekli va qat'iy o'suvchi bo'lishi kerak, berilgan: {list(breaks)}")
        n = b.size + 1
    else:
        try:
            n = int(n_classes)
        except (TypeError, ValueError):
            raise ValueError(f"n_classes butun son bo'lishi kerak, berilgan {n_classes!r}") from None
        if n < 2:
            raise ValueError(f"Sinflar soni kamida 2 bo'lishi kerak, berilgan {n}")
        if n > _MAX_CLASSES:
            raise ValueError(f"Sinflar soni ko'pi bilan {_MAX_CLASSES} bo'lishi mumkin, berilgan {n}")
        if method == "quantile":
            if v.size == 0:
                raise ValueError("Kvantil sinflash uchun yaroqli (chekli) piksel yo'q.")
            b = np.quantile(v, np.linspace(0.0, 1.0, n + 1)[1:-1]).astype(np.float64)
        else:
            b = np.linspace(0.0, 1.0, n + 1)[1:-1]
    if n > _MAX_CLASSES:
        raise ValueError(f"Sinflar soni ko'pi bilan {_MAX_CLASSES} bo'lishi mumkin, berilgan {n}")
    class_map = np.zeros(arr.shape, dtype=np.int8)
    if v.size:
        class_map[ok] = (np.digitize(v, b) + 1).astype(np.int8)
    return class_map, [float(x) for x in b]


def class_area_stats(class_map, pixel_size, rows=None, cols=None, n_classes=None):
    """
    Sinflar bo'yicha statistika: [Sinf, Nomi, Piksel, Maydon_km2, Maydon_%, Konlar_soni, Konlar_%, Boyitish].
    pixel_size - piksel tomoni (m). rows/cols (konlar piksel indekslari) berilsa har sinfdagi konlar soni
    hisoblanadi (raster tashqarisi va nodata (0) sinfdagilar hisobga olinmaydi); berilmasa Konlar_* va
    Boyitish ustunlari NaN (noma'lum). Boyitish = Konlar_% / Maydon_% (sinf konlarni tasodifiydan necha
    marta ko'p ushlaydi). n_classes berilmasa sinflar soni = max(class_map).
    """
    cm = np.asarray(class_map)
    if cm.ndim != 2:
        raise ValueError(f"class_map 2 o'lchamli (H, W) bo'lishi kerak, berilgan: {cm.shape}")
    if cm.dtype.kind not in "iu":
        raise ValueError(f"class_map butun sonli bo'lishi kerak, berilgan dtype: {cm.dtype}")
    ps = float(pixel_size)
    if not np.isfinite(ps) or ps <= 0:
        raise ValueError(f"pixel_size musbat bo'lishi kerak, berilgan {pixel_size!r}")
    if (rows is None) != (cols is None):
        raise ValueError("rows va cols birga berilishi kerak.")
    if cm.size and int(cm.min()) < 0:
        raise ValueError("class_map manfiy qiymatlar mavjud (0 = nodata, sinflar 1..n).")
    top = int(cm.max()) if cm.size else 0
    n = max(top, int(n_classes) if n_classes else 0)
    if n < 1:
        return pd.DataFrame(columns=_STATS_COLUMNS)
    counts = np.bincount(cm.ravel(), minlength=n + 1)[1:n + 1].astype(np.int64)
    total_px = int(counts.sum())

    n_dep = np.full(n, np.nan)
    if rows is not None:
        r, okr = _as_index(rows)
        c, okc = _as_index(cols)
        if r.shape != c.shape:
            raise ValueError("rows va cols uzunligi teng bo'lishi kerak.")
        inb = okr & okc & (r >= 0) & (r < cm.shape[0]) & (c >= 0) & (c < cm.shape[1])
        cls = np.zeros(r.shape, dtype=np.int64)
        cls[inb] = cm[r[inb], c[inb]]
        n_dep = np.bincount(cls[cls > 0], minlength=n + 1)[1:n + 1].astype(np.float64)
    dep_total = float(np.nansum(n_dep)) if rows is not None else np.nan

    area_pct = 100.0 * counts / total_px if total_px else np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        dep_pct = 100.0 * n_dep / dep_total if dep_total and dep_total > 0 else np.full(n, np.nan)
        enrich = np.where(area_pct > 0, dep_pct / area_pct, np.nan)
    df = pd.DataFrame({
        "Sinf": np.arange(1, n + 1), "Nomi": CLASS_LABELS(n), "Piksel": counts,
        "Maydon_km2": counts * ps * ps / 1e6, "Maydon_%": area_pct,
        "Konlar_soni": n_dep if rows is None else n_dep.astype(np.int64), "Konlar_%": dep_pct, "Boyitish": enrich,
    }, columns=_STATS_COLUMNS)
    return df


# ---------------------------------------------------------------------------
# Success-rate egri chizig'i
# ---------------------------------------------------------------------------
def _as_index(a):
    """Piksel indekslari -> (int64 massiv, yaroqlilik maskasi). Chekli bo'lmagan qiymat yaroqsiz."""
    a = np.asarray(a).ravel()
    if a.dtype.kind in "iu":
        return a.astype(np.int64), np.ones(a.shape, dtype=bool)
    f = a.astype(np.float64)
    ok = np.isfinite(f) & (np.abs(f) < 1e15)
    out = np.full(f.shape, -1, dtype=np.int64)
    out[ok] = np.floor(f[ok]).astype(np.int64)
    return out, ok


def success_rate_curve(prob_map, rows, cols):
    """
    Success-rate (capture) egri chizig'i: piksellar indeks bo'yicha kamayish tartibida; x = eng yuqori
    piksellar maydon ulushi, y = shu maydon ushlagan konlar ulushi. AUC trapetsiya usuli bilan (tasodifiy ~0.5,
    mukammal ~1). Teng qiymatli piksellar guruhi to'g'ri chiziq bilan (tasodifiy tartib kutilmasi) bog'lanadi.
    Nodata (NaN) piksellar maydondan chetlatiladi; qiymati NaN bo'lgan yoki raster tashqarisidagi konlar
    tashlanadi, n_pos - haqiqatda ishlatilgan konlar soni.
    Qaytaradi: {"area_frac", "capture_frac", "auc", "n_pos"}; kon yoki valid piksel bo'lmasa bo'sh massivlar va
    auc=NaN (xato emas).
    """
    pm = np.asarray(prob_map)
    if pm.ndim != 2:
        raise ValueError(f"prob_map 2 o'lchamli (H, W) bo'lishi kerak, berilgan: {pm.shape}")
    r, okr = _as_index(rows)
    c, okc = _as_index(cols)
    if r.shape != c.shape:
        raise ValueError("rows va cols uzunligi teng bo'lishi kerak.")
    height, width = pm.shape
    inb = okr & okc & (r >= 0) & (r < height) & (c >= 0) & (c < width)
    pv = np.full(r.shape, np.nan, dtype=pm.dtype if pm.dtype.kind == "f" else np.float64)   # searchsorted'da upcast yo'q
    pv[inb] = pm[r[inb], c[inb]]
    pos = pv[np.isfinite(pv)]
    n_pos = int(pos.size)
    valid_vals = pm[np.isfinite(pm)]
    n_all = int(valid_vals.size)
    if n_pos == 0 or n_all == 0:
        return {"area_frac": np.empty(0), "capture_frac": np.empty(0), "auc": float("nan"), "n_pos": n_pos}

    vs = np.sort(valid_vals)
    ps = np.sort(pos)
    u = np.unique(pos)[::-1]                                  # konlar qiymatlari, kamayish tartibida
    px_ge = n_all - np.searchsorted(vs, u, side="left")       # qiymati >= u bo'lgan piksellar
    px_gt = n_all - np.searchsorted(vs, u, side="right")      # qiymati > u
    dp_ge = n_pos - np.searchsorted(ps, u, side="left")
    dp_gt = n_pos - np.searchsorted(ps, u, side="right")
    x = np.concatenate([[0.0], np.column_stack([px_gt, px_ge]).ravel(), [float(n_all)]]) / n_all
    y = np.concatenate([[0.0], np.column_stack([dp_gt, dp_ge]).ravel(), [float(n_pos)]]) / n_pos
    keep = np.ones(x.size, dtype=bool)
    keep[1:] = (np.diff(x) != 0) | (np.diff(y) != 0)          # ketma-ket takror nuqtalar olib tashlanadi
    x, y = x[keep], y[keep]
    auc = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
    return {"area_frac": x, "capture_frac": y, "auc": auc, "n_pos": n_pos}


# ---------------------------------------------------------------------------
# GeoTIFF saqlash
# ---------------------------------------------------------------------------
def _clean_profile(profile, shape, dtype, nodata, log):
    """Manba profilidan faqat kerakli maydonlar: tiled/blockxsize/blockysize/interleave kabilar ko'chirilmaydi."""
    if profile is None or profile.get("transform") is None:
        raise ValueError("profile'da 'transform' yo'q (RasterStack.profile kerak).")
    height, width = int(shape[0]), int(shape[1])
    for key, size in (("height", height), ("width", width)):
        if profile.get(key) is not None and int(profile[key]) != size:
            raise ValueError(f"Xarita o'lchami ({height}x{width}) profile {key}={profile[key]} ga mos emas.")
    out = {"driver": "GTiff", "height": height, "width": width, "count": 1, "dtype": dtype, "nodata": nodata,
           "transform": profile["transform"], "compress": profile.get("compress") or "lzw"}
    if profile.get("crs") is not None:
        out["crs"] = profile["crs"]
    else:
        log("  OGOHLANTIRISH: profile'da CRS yo'q; GeoTIFF CRS'siz yoziladi.")
    return out


def _write_band(path, arr, prof, description):
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, description)


def save_rasters(out_dir, maps, profile, class_map=None, uncertainty=None, log_fn=None):
    """
    GeoTIFF'larni saqlaydi (toza profil, LZW): prognoz_{safe_name}.tif (float32, nodata -9999),
    prognoz_classes.tif (int8, nodata 0), prognoz_uncertainty.tif (float32, nodata -9999).
    NaN/cheksiz -> nodata. Qaytaradi: yozilgan yo'llar ro'yxati (xaritalar, sinflar, noaniqlik tartibida).
    """
    log = log_fn or noop_log
    out_dir = os.fspath(out_dir)
    maps = dict(maps or {})
    shape = None
    for name, arr in maps.items():
        a = np.asarray(arr)
        if a.ndim != 2:
            raise ValueError(f"'{name}' xaritasi 2 o'lchamli bo'lishi kerak, berilgan {a.shape}")
        if shape is not None and a.shape != shape:
            raise ValueError(f"Xaritalar o'lchami bir xil emas: '{name}' {a.shape}, kutilgan {shape}")
        shape = a.shape
    extra = [("class_map", class_map), ("uncertainty", uncertainty)]
    for label, arr in extra:
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.ndim != 2 or (shape is not None and a.shape != shape):
            raise ValueError(f"{label} shakli {a.shape} xaritalar shakliga {shape} mos emas.")
        shape = a.shape
    if shape is None:
        log("Saqlanadigan xarita yo'q.")
        return []
    os.makedirs(out_dir, exist_ok=True)

    saved, used = [], set()
    for name, arr in maps.items():
        base = safe_name(name)
        stem, k = base, 1
        while stem.lower() in used:           # turli nomlar bir xil fayl nomiga tushsa, ustiga yozilmasin
            k += 1
            stem = f"{base}_{k}"
        used.add(stem.lower())
        path = os.path.join(out_dir, f"prognoz_{stem}.tif")
        a = np.asarray(arr, dtype=np.float32)
        out = np.where(np.isfinite(a), a, np.float32(NODATA_OUT)).astype(np.float32)
        prof = _clean_profile(profile, shape, "float32", NODATA_OUT, log)
        _write_band(path, out, prof, f"Prospektivlik indeksi (0-1): {name}")
        saved.append(path)
        log(f"Saqlandi: {path}")
    if class_map is not None:
        cm = np.asarray(class_map)
        if cm.dtype.kind not in "iu" or (cm.size and (int(cm.min()) < 0 or int(cm.max()) > _MAX_CLASSES)):
            raise ValueError(f"class_map butun sonli va 0..{_MAX_CLASSES} oralig'ida bo'lishi kerak.")
        path = os.path.join(out_dir, "prognoz_classes.tif")
        desc = "Prospektivlik sinfi (0 = nodata, 1 = eng past)"
        try:
            _write_band(path, cm.astype(np.int8), _clean_profile(profile, shape, "int8", 0, log), desc)
        except (TypeError, ValueError, rasterio.errors.RasterioError):    # int8 qo'llab-quvvatlanmaydigan eski GDAL
            log("  OGOHLANTIRISH: int8 GeoTIFF yozib bo'lmadi, uint8 ishlatildi.")
            _write_band(path, cm.astype(np.uint8), _clean_profile(profile, shape, "uint8", 0, log), desc)
        saved.append(path)
        log(f"Saqlandi: {path}")
    if uncertainty is not None:
        a = np.asarray(uncertainty, dtype=np.float32)
        out = np.where(np.isfinite(a), a, np.float32(NODATA_OUT)).astype(np.float32)
        path = os.path.join(out_dir, "prognoz_uncertainty.tif")
        _write_band(path, out, _clean_profile(profile, shape, "float32", NODATA_OUT, log),
                    "Noaniqlik: modellar prospektivlik indeksi xaritalari std'i")
        saved.append(path)
        log(f"Saqlandi: {path}")
    return saved
