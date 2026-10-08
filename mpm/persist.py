# -*- coding: utf-8 -*-
"""
Model to'plamini (bundle) saqlash / yuklash va yangi maydonga qo'llash (ENH-05).

Bundle tuzilmasi:
  manifest.json            versiyalar, vaqt (UTC), band/feature nomlari, modellar, giperparametrlar, sozlamalar,
                           metrikalar, bo'sag'alar, blok o'lchami, CRS, izohlar, xavfsizlik ogohlantirishi
  pipeline.json            FeaturePipeline (band nomlari, kategorik qatlamlar va darajalari)
  models/<model>/<draw>/   wrapper.save(): RF/SVM/XGBoost - model.joblib, CNN - model.keras + meta.json
  README.txt               qisqa izoh (o'zbekcha)

XAVFSIZLIK: joblib/pickle yuklash o'zboshimchalik bilan kod bajarishi mumkin - bundle'ni FAQAT ishonchli
manbadan yuklang.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import __version__ as MPM_VERSION
from .base import ModelWrapper
from .common import (ENSEMBLE_NAME, TARGET_EPSG, check_cancel, collect_versions, noop_log, safe_name,
                     sub_progress)
from .data import _INT_TOL, FeaturePipeline, RasterStack, find_tiff_files, load_and_align_rasters
from .predict import class_area_stats, classify_map, predict_probability_maps, save_rasters

BUNDLE_VERSION = 1
MANIFEST_FILE = "manifest.json"
PIPELINE_FILE = "pipeline.json"
README_FILE = "README.txt"
MODELS_DIR = "models"
INCOMPLETE_MARKER = ".mpm_saving"           # saqlash boshlanganda yoziladi, manifest'dan keyin o'chiriladi
SECURITY_WARNING = ("Model fayllari joblib/pickle (RF/SVM/XGBoost) formatida: yuklash o'zboshimchalik bilan kod "
                    "bajarishi mumkin. Bundle'ni FAQAT ishonchli manbadan yuklang.")
_CHECK_LIBS = ("scikit-learn", "numpy", "xgboost", "tensorflow", "joblib")   # versiya farqi ogohlantiriladi


# ---------------------------------------------------------------------------
# Yordamchilar
# ---------------------------------------------------------------------------
def _jsonable(o):
    """Ixtiyoriy (numpy / DataFrame / dataclass ichida) ob'ektni qat'iy JSON'ga yaroqli ko'rinishga o'tkazadi
    (NaN/cheksiz -> None)."""
    if o is None or isinstance(o, (str, bool)):
        return o
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict("records"))
    if isinstance(o, pd.Series):
        return _jsonable(o.to_dict())
    if isinstance(o, os.PathLike):
        return os.fspath(o)
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _jsonable(dataclasses.asdict(o))
    return str(o)


def _is_cnn(model):
    return getattr(model, "name", "") == "CNN" or type(model).__name__ == "CNNModel"


def _write_json(path, obj):
    """Qat'iy JSON'ni avval vaqtinchalik faylga yozib, keyin almashtiradi (yarim yozilgan fayl qolmaydi)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_mpm_manifest(path):
    """path MPM bundle manifest.json'i mi (dict + butun bundle_version + modellar tavsifi)."""
    try:
        m = _read_json(path)
    except (OSError, ValueError):
        return False
    ver = m.get("bundle_version") if isinstance(m, dict) else None
    return isinstance(ver, int) and not isinstance(ver, bool) and ("models" in m or "model_names" in m)


def _model_dir(directory, name, draw):
    return os.path.join(directory, MODELS_DIR, safe_name(name), str(int(draw)))


def _check_final_models(final_models, pipeline):
    if not isinstance(final_models, dict) or not final_models:
        raise ValueError("final_models bo'sh: kamida bitta o'qitilgan model kerak ({nom: [model, ...]}).")
    out = {}
    names_seen = {}
    for name, draws in final_models.items():
        if isinstance(draws, ModelWrapper):
            draws = [draws]
        draws = list(draws)
        if not draws:
            raise ValueError(f"{name}: modellar ro'yxati bo'sh.")
        key = safe_name(name)
        if key in names_seen:
            raise ValueError(f"Model nomlari fayl nomiga bir xil tushadi: '{names_seen[key]}' va '{name}'.")
        names_seen[key] = name
        for i, m in enumerate(draws):
            if not getattr(m, "is_fitted", False):
                raise ValueError(f"{name}[{i}]: model o'qitilmagan (is_fitted=False).")
            nf = getattr(m, "n_features_", None)
            if nf is not None and int(nf) != pipeline.n_features:
                raise ValueError(f"{name}[{i}]: modelning feature'lari soni ({nf}) pipeline.n_features "
                                 f"({pipeline.n_features}) ga teng emas.")
        out[name] = draws
    return out


def _check_bundle_target(directory):
    """Saqlash joyi mavjud FAYL bo'lsa, "[Errno 17] File exists" o'rniga aniq xato ko'taradi."""
    if os.path.exists(directory) and not os.path.isdir(directory):
        raise ValueError(f"'{directory}' papka emas (bu mavjud fayl): bundle'ni saqlash uchun papka yo'lini "
                         f"kiriting yoki boshqa (yangi) papka tanlang.")


def _makedirs_checked(directory):
    """os.makedirs; yo'l ichidagi biror qism fayl bo'lsa (Errno 17/20) aniq o'zbekcha ValueError."""
    try:
        os.makedirs(directory, exist_ok=True)
    except (FileExistsError, NotADirectoryError) as e:
        raise ValueError(f"'{directory}' papkasini yaratib bo'lmadi: yo'ldagi biror qism papka emas (fayl) "
                         f"({type(e).__name__}). Boshqa (yangi) papka tanlang.") from e


def _readme_text(manifest):
    bands = ", ".join(manifest["band_names"])
    cats = ", ".join(manifest["categorical"]) or "yo'q"
    models = "\n".join(f"  - {n}: {info['n_draws']} ta model (fon tanlovi)" for n, info in manifest["models"].items())
    return f"""MPM ML - saqlangan modellar to'plami (bundle)
===============================================
Yaratilgan (UTC): {manifest['created_utc']}   |   mpm versiyasi: {manifest['mpm_version']}

NIMA SAQLANGAN
  manifest.json            versiyalar, band/feature nomlari, modellar, giperparametrlar, sozlamalar, metrikalar
  pipeline.json            FeaturePipeline (band nomlari, kategorik qatlamlar va darajalari)
  models/<model>/<draw>/   o'qitilgan modellar (RF/SVM/XGBoost: model.joblib; CNN: model.keras + meta.json)
Modellar:
{models}

QANDAY YUKLASH VA QO'LLASH
  from mpm.persist import load_bundle, apply_bundle
  bundle = load_bundle(r"<shu papka>")
  natija = apply_bundle(bundle, r"<yangi TIFF papka>", out_dir=r"<natija papka>")
  (GUI'da ham "Modellar" bo'limidan yuklash va yangi maydonga qo'llash mumkin.)

YANGI MAYDONGA QO'YILADIGAN TALABLAR
  - Yangi papkadagi TIFF fayl nomlari (kengaytmasiz, katta-kichik harf farqsiz) shu bandlarga mos bo'lishi kerak:
    {bands}
  - Kategorik qatlamlar: {cats} (nearest bilan moslashtiriladi). Ortiqcha TIFF'lar e'tiborsiz qoldiriladi.
  - Referens grid - papkadagi (alifbo bo'yicha) birinchi kerakli fayl; hamma qatlam shunga moslanadi (EPSG:{manifest['crs_epsg']}).
  - Qatlamlar o'qitishdagi bilan bir xil birlik va ma'noda bo'lishi shart.

NATIJA TALQINI
  Chiqish - "prospektivlik indeksi (0-1)": kalibrlangan ehtimollik EMAS (musbat:fon nisbati sun'iy tanlangan).
  Indeks maydonlarni o'zaro solishtirish va saralash uchun; "ma'dan topilish ehtimoli" sifatida o'qilmasin.

XAVFSIZLIK
  RF/SVM/XGBoost modellari joblib/pickle formatida saqlanadi: bunday fayl yuklanganda o'zboshimchalik bilan kod
  bajarilishi mumkin. Bundle'ni FAQAT ishonchli manbadan yuklang va begona kishidan olingan bundle'ni ochmang.
  Bir xil natija uchun o'qitishdagi kutubxona versiyalari tavsiya etiladi (manifest.json -> versions).
"""


# ---------------------------------------------------------------------------
# Saqlash
# ---------------------------------------------------------------------------
def save_bundle(directory, *, final_models, pipeline, hyperparams_used, cfg_dict, metrics_summary=None,
                thresholds=None, block_size=None, crs_epsg=TARGET_EPSG, notes="", log_fn=None):
    """
    Modellar (har model uchun bg-draw ro'yxati), FeaturePipeline va metama'lumotlarni papkaga yozadi.
    Mavjud bundle ustiga yozilsa, eski models/ o'chiriladi (aralash holat bo'lmasligi uchun); manifest.json
    OXIRIDA yoziladi (yarim saqlangan bundle yuklanmaydi). Qaytaradi: directory.
    """
    log = log_fn or noop_log
    directory = os.fspath(directory)
    _check_bundle_target(directory)
    models = _check_final_models(final_models, pipeline)
    band_names = list(pipeline.band_names)
    feature_names = list(pipeline.feature_names)

    manifest_path = os.path.join(directory, MANIFEST_FILE)
    models_root = os.path.join(directory, MODELS_DIR)
    marker = os.path.join(directory, INCOMPLETE_MARKER)
    if os.path.isfile(manifest_path):
        if not _is_mpm_manifest(manifest_path):
            raise ValueError(f"'{manifest_path}' mavjud, lekin MPM bundle manifest'i emas: boshqa fayllarni "
                             f"buzmaslik uchun saqlanmadi. Bo'sh yoki yangi papka tanlang.")
        os.remove(manifest_path)                         # eski bundle: manifest yo'qolsa bundle yaroqsiz hisoblanadi
        shutil.rmtree(models_root, ignore_errors=True)
    elif os.path.exists(models_root):
        if not os.path.isfile(marker):                   # marker bor => oldingi (yarim qolgan) MPM saqlash
            raise ValueError(f"'{models_root}' mavjud, lekin papka MPM bundle emas (manifest.json yo'q): "
                             f"boshqa fayllarni buzmaslik uchun saqlanmadi. Bo'sh yoki yangi papka tanlang.")
        shutil.rmtree(models_root, ignore_errors=True)
    _makedirs_checked(directory)
    with open(marker, "w", encoding="utf-8") as f:       # saqlash yakunlanmasa, qayta saqlashga ruxsat beradi
        f.write("MPM bundle saqlanmoqda; yakunlanmagan bo'lsa bu papkadagi models/ xavfsiz o'chiriladi.\n")

    info = {}
    for name, draws in models.items():
        for i, m in enumerate(draws):
            m.save(_model_dir(directory, name, i))
        info[name] = {"n_draws": len(draws), "dir": f"{MODELS_DIR}/{safe_name(name)}",
                      "class": type(draws[0]).__name__, "input_kind": getattr(draws[0], "input_kind", "tabular"),
                      "loader": "keras" if _is_cnn(draws[0]) else "joblib",
                      "params": [_jsonable(m.get_params()) for m in draws]}
        log(f"  Saqlandi: {name} ({len(draws)} ta draw)")
    _write_json(os.path.join(directory, PIPELINE_FILE), _jsonable(pipeline.to_dict()))

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mpm_version": MPM_VERSION,
        "versions": _jsonable(collect_versions()),
        "band_names": band_names,
        "categorical": list(pipeline.categorical),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "model_names": list(models),
        "models": info,
        "hyperparams_used": _jsonable(hyperparams_used),
        "cfg": _jsonable(cfg_dict),
        "metrics_summary": _jsonable(metrics_summary),
        "thresholds": _jsonable(thresholds),
        "block_size": _jsonable(block_size),
        "crs_epsg": int(crs_epsg),
        "ensemble_name": ENSEMBLE_NAME,
        "notes": str(notes or ""),
        "security_warning": SECURITY_WARNING,
    }
    with open(os.path.join(directory, README_FILE), "w", encoding="utf-8") as f:
        f.write(_readme_text(manifest))
    _write_json(manifest_path, manifest)
    try:
        os.remove(marker)
    except OSError:
        pass
    log(f"Bundle saqlandi: {directory}")
    return directory


# ---------------------------------------------------------------------------
# Yuklash
# ---------------------------------------------------------------------------
def _load_model(directory, name, draw, loader):
    path = _model_dir(directory, name, draw)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Model papkasi topilmadi: {path}")
    try:
        if loader == "keras":
            from .cnn import CNNModel           # TensorFlow faqat CNN bo'lsa yuklanadi
            model = CNNModel.load(path)
        else:
            from .models import SklearnModel
            model = SklearnModel.load(path)
    except (FileNotFoundError, RuntimeError):
        raise
    except Exception as e:                      # buzilgan fayl: "unpack requires a buffer..." kabi tushunarsiz xabar o'rniga
        raise ValueError(f"Model fayli o'qib bo'lmadi: {path} ({type(e).__name__}: {e}). Fayl buzilgan yoki "
                         f"kutubxona versiyalari mos emas.") from e
    if not isinstance(model, ModelWrapper):
        raise ValueError(f"{path}: yuklangan ob'ekt ModelWrapper emas ({type(model).__name__}).")
    return model


def _warn_version_diff(manifest, log):
    saved = manifest.get("versions") or {}
    now = collect_versions()
    diff = [f"{lib}: saqlangan {saved.get(lib)} -> hozirgi {now.get(lib)}"
            for lib in _CHECK_LIBS if saved.get(lib) and saved.get(lib) != now.get(lib)]
    if diff:
        log("  OGOHLANTIRISH: kutubxona versiyalari o'qitishdagidan farq qiladi, natija boshqacha bo'lishi yoki "
            "yuklash xato berishi mumkin: " + "; ".join(diff))


def load_bundle(directory, log_fn=None):
    """
    Bundle'ni yuklaydi. Qaytaradi: {"final_models", "pipeline" (FeaturePipeline), "manifest", "cfg" (dict),
    "hyperparams_used", "metrics_summary", "thresholds", "block_size", "crs_epsg", "notes", "directory"}.
    CNN => cnn.CNNModel.load, boshqalar => SklearnModel.load (joblib). Manifest versiyasi tekshiriladi.
    XAVFSIZLIK: faqat ishonchli manbadagi bundle'ni yuklang.
    """
    log = log_fn or noop_log
    directory = os.fspath(directory)
    manifest_path = os.path.join(directory, MANIFEST_FILE)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"'{manifest_path}' topilmadi: bu MPM bundle papkasi emas yoki saqlash yakunlanmagan.")
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError) as e:
        raise ValueError(f"manifest.json o'qib bo'lmadi: {e}") from e
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json noto'g'ri tuzilgan (JSON ob'ekt emas): MPM bundle emas.")
    ver = manifest.get("bundle_version")
    if not isinstance(ver, int) or isinstance(ver, bool):
        raise ValueError("manifest.json'da 'bundle_version' yo'q yoki noto'g'ri: MPM bundle emas.")
    if ver > BUNDLE_VERSION:
        raise ValueError(f"Bundle versiyasi ({ver}) bu dastur qo'llaydigan versiyadan ({BUNDLE_VERSION}) yangi: "
                         f"mpm dasturini yangilang.")
    if ver < 1:
        raise ValueError(f"Bundle versiyasi qo'llab-quvvatlanmaydi: {ver}")
    log(f"Bundle yuklanmoqda: {directory} (versiya {ver}, yaratilgan {manifest.get('created_utc')}). "
        f"Faqat ishonchli manbadagi bundle'ni yuklang.")
    _warn_version_diff(manifest, log)

    pipe_path = os.path.join(directory, PIPELINE_FILE)
    if not os.path.isfile(pipe_path):
        raise FileNotFoundError(f"'{pipe_path}' topilmadi: bundle buzilgan.")
    pipeline = FeaturePipeline.from_dict(_read_json(pipe_path))
    if list(pipeline.band_names) != list(manifest.get("band_names", [])):
        raise ValueError("pipeline.json band nomlari manifest.json bilan mos emas: bundle buzilgan.")
    if list(pipeline.feature_names) != list(manifest.get("feature_names", [])):
        raise ValueError("pipeline.json feature nomlari manifest.json bilan mos emas: bundle buzilgan.")

    model_info = manifest.get("models") or {}
    names = list(manifest.get("model_names") or model_info)
    if not names:
        raise ValueError("manifest.json'da modellar ro'yxati bo'sh.")
    final_models = {}
    for name in names:
        mi = model_info.get(name)
        if not mi or int(mi.get("n_draws", 0)) < 1:
            raise ValueError(f"manifest.json'da '{name}' modeli tavsifi yo'q yoki draw soni 0.")
        loader = mi.get("loader") or ("keras" if name == "CNN" else "joblib")
        final_models[name] = [_load_model(directory, name, i, loader) for i in range(int(mi["n_draws"]))]
        log(f"  Yuklandi: {name} ({len(final_models[name])} ta draw)")

    return {"final_models": final_models, "pipeline": pipeline, "manifest": manifest,
            "cfg": manifest.get("cfg") or {}, "hyperparams_used": manifest.get("hyperparams_used"),
            "metrics_summary": manifest.get("metrics_summary"), "thresholds": manifest.get("thresholds"),
            "block_size": manifest.get("block_size"), "crs_epsg": manifest.get("crs_epsg", TARGET_EPSG),
            "notes": manifest.get("notes", ""), "directory": directory}


# ---------------------------------------------------------------------------
# Yangi maydonga qo'llash
# ---------------------------------------------------------------------------
def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _match_bands(band_names, paths):
    """band nomi -> fayl yo'li (avval aniq, keyin katta-kichik harf farqsiz moslik). Yetishmasa ValueError."""
    stems = {p: _stem(p) for p in paths}
    matched, used, missing = {}, set(), []
    for band in band_names:
        exact = [p for p in paths if stems[p] == band and p not in used]
        cand = exact or [p for p in paths if stems[p].lower() == band.lower() and p not in used]
        if len(cand) > 1:
            raise ValueError(f"'{band}' bandiga bir nechta fayl mos keladi: {[os.path.basename(p) for p in cand]}. "
                             f"Papkada nomi bir xil yoki faqat harf registri/kengaytmasi bilan farq qiladigan "
                             f"fayllar bo'lmasligi kerak.")
        if not cand:
            missing.append(band)
            continue
        matched[band] = cand[0]
        used.add(cand[0])
    if missing:
        avail = sorted(stems.values(), key=str.lower)
        raise ValueError(f"Yangi papkada bundle uchun kerakli bandlar topilmadi. Yetishmaydigan bandlar: {missing}. "
                         f"Papkadagi mavjud nomlar: {avail}. Fayl nomlari (kengaytmasiz, katta-kichik harf farqsiz) "
                         f"o'qitishdagi band nomlariga mos bo'lishi kerak.")
    return matched


def _reorder_raster(raster, band_names, categorical, matched):
    """load_and_align_rasters (papka tartibida) natijasini bundle band tartibiga o'tkazadi va nomlarni bundle
    nomlariga qaytaradi."""
    pos = {s: i for i, s in enumerate(raster.band_names)}
    order = [pos[_stem(matched[b])] for b in band_names]
    stack = raster.stack if order == list(range(len(order))) else raster.stack[order]
    tech = [dict(raster.tech_metadata[i], band_name=b) for i, b in zip(order, band_names)] \
        if len(raster.tech_metadata) == len(raster.band_names) else []
    return RasterStack(stack=stack, band_names=list(band_names), profile=raster.profile, transform=raster.transform,
                       crs_epsg=raster.crs_epsg, categorical=list(categorical), tech_metadata=tech)


def _check_categorical_levels(raster, pipeline, log):
    for b in pipeline.categorical:
        band = raster.stack[raster.band_names.index(b)]
        vals = np.unique(np.rint(band[np.isfinite(band)]))
        known = set(pipeline.levels.get(b, []))
        vals_f = band[np.isfinite(band)]
        n_frac = int(np.count_nonzero(np.abs(vals_f - np.rint(vals_f)) > _INT_TOL)) if vals_f.size else 0
        if n_frac:
            log(f"  OGOHLANTIRISH: '{b}' kategorik qatlamida {n_frac:,} ta piksel butun son emas: ularning "
                f"one-hot ustunlari 0 bo'ladi (noma'lum daraja kabi).")
        unknown = [int(v) for v in vals if int(v) not in known]
        if unknown:
            log(f"  OGOHLANTIRISH: '{b}' kategorik qatlamida o'qitishda bo'lmagan daraja(lar) bor: "
                f"{unknown[:15]}{' ...' if len(unknown) > 15 else ''}; ularning one-hot ustunlari 0 bo'ladi.")


def apply_bundle(bundle, tiff_folder, *, out_dir=None, batch_size=8192, assume_crs_if_missing=True,
                 class_method="quantile", n_classes=5, class_breaks=None, log_fn=None, progress_fn=None,
                 cancel=None):
    """
    Bundle'ni yangi maydonga qo'llaydi. TIFF'lar band nomi bo'yicha (katta-kichik harf farqsiz) moslanadi;
    yetishmaydigan band bo'lsa ValueError (yetishmaydigan va mavjud nomlar bilan); ortiqcha TIFF'lar
    e'tiborsiz (logda). Referens grid - papkadagi birinchi tanlangan fayl; band tartibi bundle'niki.
    bundle: load_bundle natijasi (yoki bundle papkasi yo'li).
    Qaytaradi: predict_probability_maps natijasi ("maps", "uncertainty", "valid_mask") +
      "class_map", "class_breaks", "class_stats" (konlarsiz), "raster" (RasterStack), "saved_paths",
      "matched_files" ({band: fayl}), "ignored_files".
    """
    log = log_fn or noop_log
    if out_dir:
        from .pipeline import _check_out_dir
        _check_out_dir(out_dir)
    if isinstance(bundle, (str, os.PathLike)):
        bundle = load_bundle(bundle, log_fn=log)
    pipeline = bundle["pipeline"]
    final_models = bundle["final_models"]
    band_names = list(pipeline.band_names)
    # sinflash parametrlarini hisoblashdan OLDIN tekshiramiz (uzoq bashoratdan keyin xato chiqmasin)
    classify_map(np.zeros((1, 1), dtype=np.float32), class_method, n_classes, class_breaks)

    report = progress_fn
    if report is not None:
        report(0.0, "Qatlamlar qidirilmoqda")
    paths = find_tiff_files(tiff_folder)
    if not paths:
        raise ValueError(f"'{tiff_folder}' papkasida .tif/.tiff fayl topilmadi.")
    matched = _match_bands(band_names, paths)
    chosen = set(matched.values())
    selected = [p for p in paths if p in chosen]                 # papka tartibi: birinchisi - referens grid
    ignored = [p for p in paths if p not in chosen]
    if ignored:
        log(f"  Ortiqcha TIFF'lar e'tiborsiz qoldirildi ({len(ignored)} ta): "
            f"{[os.path.basename(p) for p in ignored]}")
    log(f"Yangi maydon: {len(selected)} ta qatlam moslandi; referens grid: {os.path.basename(selected[0])}")
    check_cancel(cancel)

    cat_stems = [_stem(matched[b]) for b in pipeline.categorical]
    loaded = load_and_align_rasters(selected, categorical=cat_stems, log_fn=log,
                                    assume_crs_if_missing=assume_crs_if_missing, cancel=cancel)
    raster = _reorder_raster(loaded, band_names, pipeline.categorical, matched)
    del loaded
    _check_categorical_levels(raster, pipeline, log)
    if report is not None:
        report(0.10, "Qatlamlar yuklandi")

    pred = predict_probability_maps(raster, pipeline, final_models, batch_size=batch_size, log_fn=log,
                                    progress_fn=sub_progress(progress_fn, 0.10, 0.88), cancel=cancel)
    check_cancel(cancel)
    class_map, breaks = classify_map(pred["maps"][ENSEMBLE_NAME], class_method, n_classes, class_breaks)
    stats = class_area_stats(class_map, raster.pixel_size, n_classes=len(breaks) + 1)
    saved = []
    if out_dir:
        saved = save_rasters(out_dir, pred["maps"], raster.profile, class_map=class_map,
                             uncertainty=pred["uncertainty"], log_fn=log)
    if report is not None:
        report(1.0, "Tayyor")
    out = dict(pred)
    out.update({"class_map": class_map, "class_breaks": breaks, "class_stats": stats, "raster": raster,
                "saved_paths": saved, "matched_files": dict(matched),
                "ignored_files": [os.path.basename(p) for p in ignored]})
    return out
