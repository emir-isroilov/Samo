# -*- coding: utf-8 -*-
"""
Pipeline: butun ish oqimining orkestratori (run_training / run_prediction / export_results / estimate_cost_text).

Asl koddagi xatolar bu yerda birlashadi: yagona hp lug'ati (BUG-15), blok o'lchami (BUG-09), fon valid-mask (BUG-11),
spatial CV (BUG-05), CNN val_loss (BUG-04), MDI zaxirasi (BUG-03), SHAP (BUG-01), blok-bootstrap CI (BUG-08).

Chiqish "ehtimollik" emas, "prospektivlik indeksi (0-1)": musbat:fon nisbati sun'iy tanlangan.
Log fayli (cfg.output_dir berilsa): <output_dir>/mpm_run_<N>.log, UTF-8.
"""
from __future__ import annotations

import contextlib
import copy
import datetime
import importlib.util
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd

from . import __version__ as MPM_VERSION
from . import spatial
from .common import (ENSEMBLE_NAME, CancelledError, check_cancel, collect_versions, get_xgboost, noop_log,
                     set_global_seed, sub_progress, tf_available)
from .config import PARAM_SPECS, RunConfig, hp_summary_text, validate_hyperparams
from .cv import compute_metrics, metrics_dataframe, run_background_sensitivity, run_cv
from .data import (FeaturePipeline, build_data_dictionary, build_dataset, data_diagnostics, dedupe_points_by_pixel,
                   find_shapefile, find_tiff_files, generate_background_points, load_aoi,
                   load_and_align_rasters, load_or_create_manual_metadata, load_positive_points,
                   save_data_dictionary, valid_pixel_mask)
from .explain import compute_shap_summary, mdi_importance
from .models import make_model
from .predict import (class_area_stats, classify_map, predict_probability_maps, save_rasters,
                      success_rate_curve)

__all__ = ["run_training", "run_prediction", "export_results", "estimate_cost_text"]

MIN_POSITIVE = 10                 # modelni o'qitish uchun eng kam musbat nuqta
FALLBACK_BLOCK = 1000.0           # variogram baholanmasa zaxira blok o'lchami (m)
_MAX_WARNINGS = 300
_CONST_STD = 1e-6                 # layer std shundan kam => o'zgarmas deb ogohlantiriladi
_TF_MB_PER_FIT = 8                # CNN: TensorFlow xotirasi har o'qitishda taxminan shuncha MB o'sadi
_WARN_RE = re.compile(r"ogohlantirish\s*:\s*(.*)", re.IGNORECASE)
_LOG_RE = re.compile(r"^mpm_run_(\d+)\.log$")
_NOTE_CNN = ("CNN chiqishi kalibrlanmaydi va neg/pos namuna og'irligi bilan o'qitiladi, ansambl esa oddiy o'rtacha: "
             "CNN ehtimolliklari boshqa modellar bilan bir shkalada bo'lmasligi mumkin.")
_NOTE_INDEX = ("Chiqish - prospektivlik indeksi (0-1), ehtimollik EMAS: musbat:fon nisbati sun'iy tanlangan, "
               "shuning uchun qiymatlar haqiqiy kon topish ehtimolini bildirmaydi, faqat nisbiy tartib beradi.")


# ---------------------------------------------------------------------------
# Kontekst: log (+fayl), ogohlantirishlar, vaqtlar, progress
# ---------------------------------------------------------------------------
class _Ctx:
    """run_training holati: log (foydalanuvchi + fayl), yig'ilgan ogohlantirishlar, bosqich vaqtlari, progress."""

    def __init__(self, log_fn, progress_fn, cancel):
        self.user_log = log_fn or noop_log
        self.progress_fn = progress_fn
        self.cancel = cancel
        self.warnings = []
        self._seen = set()
        self.timings = {}
        self.log_path = None
        self._fh = None

    # ---- log fayli
    def open_log(self, output_dir):
        """output_dir mavjud/yaratilsa mpm_run_<N>.log (N ketma-ket) ochiladi (UTF-8). Muvaffaqiyatsiz => faylsiz."""
        if not output_dir:
            return
        try:
            os.makedirs(output_dir, exist_ok=True)
            used = [int(m.group(1)) for n in os.listdir(output_dir) if (m := _LOG_RE.match(n))]
            n = (max(used) + 1) if used else 1
            while True:
                path = os.path.join(output_dir, f"mpm_run_{n}.log")
                try:
                    self._fh = open(path, "x", encoding="utf-8", buffering=1)
                    break
                except FileExistsError:
                    n += 1
            self.log_path = path
        except OSError as e:
            self._fh = None
            self.log_path = None
            self.user_log(f"  OGOHLANTIRISH: log faylini ochib bo'lmadi ({e}); log faqat oynada ko'rsatiladi.")

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    # ---- log / ogohlantirish
    def log(self, msg=""):
        msg = str(msg)
        if self._fh is not None:
            try:
                self._fh.write(f"[{datetime.datetime.now():%H:%M:%S}] {msg}\n")
            except (OSError, ValueError):
                self._fh = None
        self._collect_warnings(msg)
        self.user_log(msg)

    def _collect_warnings(self, msg):
        """Ko'p qatorli xabarda HAR bir 'Ogohlantirish:' qatori alohida yoziladi; uning ostidagi bo'sh joy bilan
        boshlangan davom qatorlari shu ogohlantirishga qo'shiladi (boshqasi bilan aralashmaydi)."""
        cur = None
        found = []
        for line in msg.splitlines():
            m = _WARN_RE.search(line)
            if m:
                cur = [m.group(1)]
                found.append(cur)
            elif cur is not None and line[:1].isspace():
                cur.append(line)
            else:
                cur = None
        for parts in found:
            text = " ".join(" ".join(parts).split())
            if text and text not in self._seen and len(self.warnings) < _MAX_WARNINGS:
                self._seen.add(text)
                self.warnings.append(text)

    def warn(self, msg):
        self.log(f"  OGOHLANTIRISH: {msg}")

    # ---- progress / bekor qilish / vaqt
    def report(self, frac, msg=""):
        if self.progress_fn is not None:
            self.progress_fn(float(min(1.0, max(0.0, frac))), msg)

    def prog(self, lo, hi):
        return sub_progress(self.progress_fn, lo, hi)

    def check(self):
        check_cancel(self.cancel)

    @contextlib.contextmanager
    def stage(self, key):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[key] = self.timings.get(key, 0.0) + time.perf_counter() - t0


def _prefixed(log, prefix):
    return lambda msg="": log(f"    [{prefix}] {str(msg).strip()}")


# ---------------------------------------------------------------------------
# Sozlamalarni tayyorlash
# ---------------------------------------------------------------------------
def _has_tunable(model):
    return any(s.tunable for s in PARAM_SPECS[model])


def _tune_names(cfg, names):
    """Tuning qilinadigan (yoqilgan, belgilangan va tunable parametrli) modellar."""
    if not cfg.tuning.enabled:
        return []
    return [m for m in names if cfg.tuning.models.get(m) and _has_tunable(m)]


def _prepare(cfg, ctx):
    """cfg tekshiruvi, hp validatsiyasi, mavjud bo'lmagan modellarni chiqarish. (cfg nusxa, hp, model nomlari)."""
    if isinstance(cfg, dict):
        cfg = RunConfig.from_dict(cfg)
    problems = cfg.validate()
    if problems:
        raise ValueError("Sozlamalarda xatolar bor:\n- " + "\n- ".join(problems))
    cfg = copy.deepcopy(cfg)
    cfg.n_jobs = max(1, int(cfg.n_jobs or 1))
    cfg.final_bg_draws = max(1, int(cfg.final_bg_draws or 1))
    cfg.n_bootstrap = max(0, int(cfg.n_bootstrap or 0))
    ctx.open_log(cfg.output_dir)
    ctx.log(f"MPM ML v{MPM_VERSION}: o'qitish boshlandi. Chiqish prospektivlik indeksi (0-1) - ehtimollik emas.")
    if ctx.log_path:
        ctx.log(f"Log fayli: {ctx.log_path}")

    hp, hp_warn = validate_hyperparams(cfg.hyperparams)
    for w in hp_warn:
        ctx.warn(f"giperparametr: {w}")
    cfg.hyperparams = hp

    names = []
    for m in cfg.enabled_models():
        if m == "XGBoost" and get_xgboost() is None:
            ctx.warn("XGBoost o'rnatilmagan ('pip install xgboost'), XGBoost o'tkazib yuboriladi.")
            cfg.use_models["XGBoost"] = False
        elif m == "CNN" and not tf_available():
            ctx.warn("TensorFlow o'rnatilmagan ('pip install tensorflow'), CNN o'tkazib yuboriladi.")
            cfg.use_models["CNN"] = False
        else:
            names.append(m)
    if not names:
        raise ValueError("Yoqilgan modellardan birortasi ishlatib bo'lmaydi (kutubxonalar yo'q). "
                         "Kamida bitta mavjud modelni (RandomForest/SVM) yoqing.")
    if cfg.tuning.enabled and not _tune_names(cfg, names):
        ctx.warn("tuning yoqilgan, lekin mavjud modellar orasida tuning uchun belgilangani yo'q: tuning bajarilmaydi.")
    ctx.log("Modellar: " + ", ".join(names))
    ctx.log("Giperparametrlar (CV va yakuniy model uchun yagona manba):\n" + hp_summary_text(hp, names))
    if "CNN" in names and len(names) > 1:
        ctx.warn(_NOTE_CNN)
    ctx.log(estimate_cost_text(cfg))
    return cfg, hp, names


# ---------------------------------------------------------------------------
# Ma'lumotlar bosqichi
# ---------------------------------------------------------------------------
def _check_metadata_csv(folder, ctx):
    """metadata.csv ajratgichi (';') yoki kodlash (cp1251) muammosini oldindan aniqlaydi va aniq maslahat beradi."""
    path = os.path.join(folder, "metadata.csv")
    if not os.path.isfile(path):
        return
    try:
        with open(path, "rb") as f:
            raw = f.read(65536)
    except OSError:
        return
    hint = ("metadata.csv vergul (,) bilan ajratilgan va UTF-8 kodlashda bo'lishi kerak. Excel ba'zan ';' ajratgich "
            "va cp1251 kodlash bilan saqlaydi: 'CSV UTF-8 (vergul bilan ajratilgan)' formatida qayta saqlang.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        ctx.warn("metadata.csv UTF-8 emas (cp1251 bo'lishi mumkin) - o'qib bo'lmaydi. " + hint)
        return
    header = text.splitlines()[0] if text.strip() else ""
    if ";" in header and "," not in header:
        ctx.warn("metadata.csv ';' ajratgich bilan yozilgan - ustunlar tanilmaydi. " + hint)


def _load_data(cfg, ctx, hp, names):
    """Rasterlar -> metadata -> FeaturePipeline -> nuqtalar -> fon -> Dataset -> diagnostika."""
    tiff_paths = find_tiff_files(cfg.tiff_folder)
    if not tiff_paths:
        raise ValueError(f"TIFF papkasida .tif/.tiff fayllar topilmadi: '{cfg.tiff_folder}'.")
    points_shp = find_shapefile(cfg.points_folder)
    if not points_shp:
        raise ValueError(f"Musbat nuqtalar papkasida .shp fayl topilmadi: '{cfg.points_folder}'.")
    aoi_shp = find_shapefile(cfg.aoi_folder)
    if not aoi_shp:
        raise ValueError(f"AOI (maydon konturi) papkasida .shp fayl topilmadi: '{cfg.aoi_folder}'.")
    ctx.log(f"{len(tiff_paths)} ta TIFF qatlam topildi.")

    # ---- rasterlar + metadata (0-8%)
    with ctx.stage("data"):
        ctx.report(0.0, "Rasterlar yuklanmoqda")
        raster = load_and_align_rasters(tiff_paths, categorical=cfg.categorical_layers, log_fn=ctx.log,
                                        assume_crs_if_missing=cfg.assume_crs_if_missing, cancel=ctx.cancel)
        ctx.report(0.06, "Rasterlar yuklandi")
        ctx.check()
        _check_metadata_csv(cfg.tiff_folder, ctx)
        manual, meta_path = load_or_create_manual_metadata(cfg.tiff_folder, raster.band_names, log_fn=ctx.log)
        data_dictionary = build_data_dictionary(raster.band_names, raster.tech_metadata, manual)
        n_inc = sum(1 for b in raster.band_names if not str((manual.get(b) or {}).get("source_owner", "")).strip())
        if n_inc:
            ctx.warn(f"{n_inc}/{raster.n_bands} qatlam uchun qo'lda metadata (manba, sana, transformatsiya) "
                     f"to'ldirilmagan. '{meta_path}' faylini to'ldiring: maqolaga chiqarishdan oldin data "
                     f"dictionary to'liq bo'lishi kerak.")
        fpipe = FeaturePipeline(raster.band_names, categorical=raster.categorical).fit(raster.stack)
        if raster.categorical:
            ctx.log("  Kategorik qatlamlar (one-hot): " + ", ".join(
                f"{b} ({len(fpipe.levels[b])} daraja)" for b in raster.categorical))
        ctx.log(f"Feature'lar: {fpipe.n_features} ta ({raster.n_bands} band).")
        ctx.report(0.08, "Metadata tayyor")

        # ---- nuqtalar / fon / dataset (8-12%)
        ctx.check()
        ctx.log(f"AOI yuklanmoqda: {aoi_shp}")
        aoi_gdf = load_aoi(aoi_shp, assume_crs_if_missing=cfg.assume_crs_if_missing, log_fn=ctx.log)
        ctx.log(f"Musbat nuqtalar yuklanmoqda: {points_shp}")
        positive_gdf = load_positive_points(points_shp, aoi_gdf, assume_crs_if_missing=cfg.assume_crs_if_missing,
                                            log_fn=ctx.log)
        positive_gdf = dedupe_points_by_pixel(positive_gdf, raster.transform, raster.stack.shape, log_fn=ctx.log)
        ctx.log(f"  {len(positive_gdf)} ta musbat nuqta (AOI ichida, piksel dublikatsiz).")
        valid = valid_pixel_mask(raster)
        if not valid.any():
            raise ValueError("Barcha qatlamlar bir vaqtda chekli bo'lgan piksel yo'q: qatlamlar kesishmasligini "
                             "yoki nodata'ni tekshiring.")
        ctx.log(f"Fon (pseudo-absence) nuqtalar: {cfg.n_background} ta, strategiya={cfg.background_strategy}, "
                f"min. masofa={cfg.min_distance:g} m.")
        background_gdf = generate_background_points(
            aoi_gdf, positive_gdf, cfg.n_background, cfg.min_distance, random_state=cfg.seed,
            strategy=cfg.background_strategy, valid_mask=valid, transform=raster.transform, log_fn=ctx.log)
        need_fs = "CNN" in names and hp["CNN"]["mode"] == "patch2d"
        dataset = build_dataset(raster, fpipe, positive_gdf, background_gdf, log_fn=ctx.log, need_feature_stack=need_fs)
        if dataset.n_pos < MIN_POSITIVE:
            raise ValueError(f"Musbat nuqtalar soni juda kam ({dataset.n_pos} < {MIN_POSITIVE}). Modelni ishonchli "
                             f"o'qitish uchun kamida ~{MIN_POSITIVE}-15 musbat nuqta kerak (AOI ichida, yaroqli pikselda).")
        ctx.log(f"Yakuniy dataset: {dataset.n_pos} musbat + {dataset.n_neg} fon = {dataset.n} nuqta, "
                f"{len(dataset.feature_names)} feature.")
        ctx.report(0.12, "Dataset tayyor")

    # ---- diagnostika (12-14%)
    with ctx.stage("diagnostics"):
        ctx.check()
        diag = data_diagnostics(raster, dataset, seed=cfg.seed)
        for w in diag.get("dataset_summary", {}).get("warnings", []):
            ctx.warn(w)
        ls = diag["layer_stats"]
        for _, r in ls.iterrows():
            if np.isfinite(r["std"]) and r["std"] < _CONST_STD:
                ctx.warn(f"'{r['band']}' qatlami deyarli o'zgarmas (std={r['std']:.2g}): RF bunday feature'ni "
                         f"konstant deb hisoblaydi va u hech qanday ma'lumot bermaydi - qatlamni olib tashlang.")
        low = ls[ls["valid_pct"] < 50]
        for _, r in low.iterrows():
            ctx.warn(f"'{r['band']}' qatlami faqat {r['valid_pct']:.0f}% maydonni qoplaydi: valid piksellar "
                     f"(barcha qatlamlar kesishmasi) kamayadi.")
        hc = diag.get("high_corr_pairs")
        if hc is not None and len(hc):
            ctx.log(f"  Yuqori korrelyatsiyali (|r|>=0.9) juftliklar: {len(hc)} ta (eng yuqorisi: "
                    f"{hc.iloc[0]['band_a']} - {hc.iloc[0]['band_b']}, r={hc.iloc[0]['corr']:.2f}).")
        vif = diag.get("vif")
        if vif is not None and len(vif):
            n_hi = int((vif["flag"] == "yuqori").sum())
            if n_hi:
                ctx.log(f"  VIF yuqori (>=10) qatlamlar: {n_hi} ta (multikollinearlik).")
        ctx.report(0.14, "Diagnostika tayyor")
    return {"raster": raster, "fpipe": fpipe, "aoi_gdf": aoi_gdf, "positive_gdf": positive_gdf,
            "background_gdf": background_gdf, "dataset": dataset, "valid": valid,
            "data_dictionary": data_dictionary, "metadata_csv_path": meta_path, "diagnostics": diag}


# ---------------------------------------------------------------------------
# Blok o'lchami
# ---------------------------------------------------------------------------
def _block_size(cfg, ctx, raster, dataset, hp, names):
    with ctx.stage("blocks"):
        ctx.check()
        if cfg.block_size and cfg.block_size > 0:
            bs = float(cfg.block_size)
            ctx.log(f"Spatial blok o'lchami (qo'lda): {bs:,.0f} m")
        else:
            band = str(cfg.variogram_band or "auto")
            idx = [i for i, b in enumerate(raster.band_names) if b not in raster.categorical]
            if band != "auto":
                if band in raster.band_names:
                    idx = [raster.band_names.index(band)]
                else:
                    ctx.warn(f"variogram uchun '{band}' bandi topilmadi, barcha raqamli bandlar ishlatiladi.")
            what = "barcha raqamli bandlar" if band == "auto" or len(idx) != 1 else f"'{raster.band_names[idx[0]]}' bandi"
            ctx.log(f"Spatial blok o'lchami avtomatik baholanmoqda (empirik semivariogram, {what})...")
            est = spatial.estimate_autocorrelation_range(raster.stack, raster.transform, band_indices=idx,
                                                         random_state=cfg.seed, log_fn=ctx.log)
            if est:
                bs = float(est)
            else:
                bs = FALLBACK_BLOCK
                ctx.log(f"  Variogram baholanmadi: zaxira blok o'lchami {bs:,.0f} m ishlatiladi.")
        bs, groups = spatial.adapt_block_size(dataset.coords, dataset.y, cfg.n_splits, bs, log_fn=ctx.log)
        n_blocks = int(np.unique(groups).size)
        n_pos_blocks = int(np.unique(groups[dataset.y == 1]).size)
        ctx.log(f"Ishlatiladigan blok o'lchami: {bs:,.0f} m; bloklar: {n_blocks} (musbat nuqtali: {n_pos_blocks}).")

        if "CNN" in names and hp["CNN"]["mode"] == "patch2d":
            windows = [int(hp["CNN"]["window"])]
            if "CNN" in _tune_names(cfg, names):
                ch = cfg.tuning.resolved_space("CNN").get("window", {}).get("choices") or []
                windows += [int(w) for w in ch]
            w = max(windows)
            if bs < w * raster.pixel_size:
                ctx.warn(f"blok o'lchami ({bs:,.0f} m) CNN oynasidan ({w} x {raster.pixel_size:g} m = "
                         f"{w * raster.pixel_size:,.0f} m) kichik: train va validation patchlari ustma-ust tushib, "
                         f"CNN natijasi optimistik bo'ladi (leakage). Blok o'lchamini kattalashtiring yoki "
                         f"oynani kichraytiring.")
        ctx.report(0.16, "Bloklar tayyor")
    return bs, groups


# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------
def _cv_block(cv_res, dataset, groups, cfg, ctx, label):
    metrics, ens = compute_metrics(dataset.y, cv_res["oof"], groups, n_boot=cfg.n_bootstrap, seed=cfg.seed,
                                   log_fn=ctx.log)
    for w in cv_res.get("warnings", []):
        ctx.warn(f"{label} CV: {w}")
    return {"mode": cv_res["mode"], "oof": cv_res["oof"], "metrics": metrics,
            "metrics_df": metrics_dataframe(metrics), "ensemble_oof": ens, "fold_map": cv_res["fold_map"],
            "fold_table": cv_res["fold_table"], "perm_importance": cv_res.get("perm_importance", {}),
            "tuned_params": cv_res.get("tuned_params", {}), "feature_names": list(cv_res["feature_names"]),
            "hp": cv_res.get("hp"), "n_splits": cv_res.get("n_splits"), "n_repeats": cv_res.get("n_repeats")}


def _run_cvs(cfg, ctx, names, hp, dataset, groups):
    common = dict(n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, calibrate=cfg.calibrate,
                  calibration_method=cfg.calibration_method, calibration_cv=cfg.calibration_cv,
                  n_jobs=cfg.n_jobs, seed=cfg.seed, log_fn=ctx.log, cancel=ctx.cancel)
    random_block = None
    if cfg.run_random_cv:
        with ctx.stage("random_cv"):
            ctx.check()
            ctx.log(f"\n=== RANDOM cross-validation (benchmark, {cfg.n_splits}-fold x {cfg.n_repeats} takror): "
                    f"faqat solishtirish uchun, tuning va importance yo'q ===")
            res = run_cv(dataset, groups, "random", names, hp, tuning=None, perm_importance=False,
                         progress_fn=ctx.prog(0.16, 0.36), **common)
            random_block = _cv_block(res, dataset, groups, cfg, ctx, "Random")
            ctx.report(0.36, "Random CV tugadi")
    with ctx.stage("spatial_cv"):
        ctx.check()
        tuning = cfg.tuning if _tune_names(cfg, names) else None
        ctx.log(f"\n=== SPATIAL BLOCK cross-validation (asosiy natija, {cfg.n_splits}-fold x {cfg.n_repeats} "
                f"takror) ===")
        res = run_cv(dataset, groups, "spatial", names, hp, tuning=tuning, perm_importance=cfg.perm_importance,
                     perm_repeats=cfg.perm_importance_repeats, progress_fn=ctx.prog(0.36, 0.70), **common)
        spatial_block = _cv_block(res, dataset, groups, cfg, ctx, "Spatial")
        ctx.report(0.70, "Spatial CV tugadi")
    if random_block is not None:
        ctx.log("\n--- Random vs Spatial-block AUC solishtiruvi (optimizm = random - spatial) ---")
        for name in spatial_block["metrics"]:
            r = random_block["metrics"].get(name, {}).get("auc", float("nan"))
            s = spatial_block["metrics"][name]["auc"]
            ctx.log(f"  {name}: random AUC={r:.3f} | spatial AUC={s:.3f} | optimizm (leakage taxmini) = {r - s:+.3f}")
        if _tuning_on(cfg, names) and cfg.tuning.mode == "nested":
            ctx.log("  Eslatma: random CV bazaviy giperparametrlar bilan, spatial CV esa nested tuning bilan "
                    "baholandi - farq qisman shundan ham bo'lishi mumkin.")
    return random_block, spatial_block


def _tuning_on(cfg, names):
    return bool(_tune_names(cfg, names))


# ---------------------------------------------------------------------------
# Yakuniy modellar
# ---------------------------------------------------------------------------
def _log_fit_info(ctx, name, tag, model):
    info = getattr(model, "fit_info", None) or {}
    parts = []
    cal = info.get("calibration")
    if cal:
        s = f"kalibrlash={cal}"
        if info.get("calibration_cv"):
            s += f" (cv={info['calibration_cv']})"
        parts.append(s)
        if cal == "skipped":
            ctx.warn(f"{name} ({tag}): kalibrlash o'tkazib yuborildi ({info.get('calibration_reason', '')}).")
    if info.get("svm_fallback"):
        parts.append(f"zaxira={info['svm_fallback']}")
        ctx.warn(f"{name} ({tag}): SVM kalibrlanmagan zaxira indeks (sigmoid min-max) ishlatildi.")
    if "epochs_run" in info:
        parts.append(f"epoch={info['epochs_run']}, eng yaxshi={info.get('best_epoch')}, "
                     f"early_stopping={info.get('early_stopping')}, val_loss={info.get('val_loss')}, "
                     f"n_train={info.get('n_train')}, n_val={info.get('n_val')}")
        if info.get("early_stopping") == "train_loss":
            ctx.warn(f"{name} ({tag}): validatsiya ajratib bo'lmadi, early stopping train loss bo'yicha ishladi.")
    if parts:
        ctx.log(f"    {name} ({tag}) fit: " + "; ".join(parts))


def _diff_text(params, base):
    d = [f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items() if base.get(k) != v]
    return ", ".join(d) if d else "bazaviy bilan bir xil"


def _tune_final(name, base_hp, dataset, groups, cfg, ctx, progress):
    """Butun ma'lumotda tuning (nested emas: shu protsedura tashqi CV'da baholangan). Kalibrlashsiz (ball tartibga
    bog'liq, 4-5x tez). Xato/ imkonsizlik => bazaviy hp (ogohlantirish bilan)."""
    from .tuning import tune_model
    try:
        res = tune_model(name, dict(base_hp), dataset, groups, tuning=cfg.tuning, calibrate=False,
                         calibration_method=cfg.calibration_method, calibration_cv=cfg.calibration_cv,
                         n_jobs=cfg.n_jobs, seed=cfg.seed, log_fn=_prefixed(ctx.log, f"{name} final tuning"),
                         cancel=ctx.cancel, progress_fn=progress)
    except ValueError as e:
        ctx.warn(f"{name} final tuning bajarilmadi ({e}); bazaviy giperparametrlar ishlatildi.")
        return dict(base_hp), {"best_params": dict(base_hp), "best_score": float("nan"), "trials": [],
                               "n_trials": 0, "fallback": str(e)}
    best = validate_hyperparams({name: res["best_params"]})[0][name]
    rec = dict(res)
    rec["best_params"] = dict(best)
    if res.get("skipped_reason"):
        ctx.warn(f"{name} final tuning o'tkazib yuborildi ({res['skipped_reason']}); bazaviy hp ishlatildi.")
    else:
        ctx.log(f"  {name} yakuniy tuning: {res.get('scoring')}={res.get('best_score', float('nan')):.4f} "
                f"(bazaviy {res.get('base_score', float('nan')):.4f}); {_diff_text(best, base_hp)}")
    return best, rec


def _draw_dataset(cfg, ctx, k, data_, block_size):
    """k-chi fon tanlovi (seed = cfg.seed + 2000 + k): (Dataset, groups) yoki (None, None). feature_stack ulashiladi."""
    seed = cfg.seed + 2000 + k
    try:
        bg = generate_background_points(
            data_["aoi_gdf"], data_["positive_gdf"], cfg.n_background, cfg.min_distance, random_state=seed,
            strategy=cfg.background_strategy, valid_mask=data_["valid"], transform=data_["raster"].transform,
            log_fn=ctx.log)
        ds = build_dataset(data_["raster"], data_["fpipe"], data_["positive_gdf"], bg, log_fn=ctx.log,
                           need_feature_stack=False)
        ds.feature_stack = data_["dataset"].feature_stack
        groups = spatial.assign_spatial_blocks(ds.coords, block_size)
    except ValueError as e:
        ctx.warn(f"{k}-fon tanlovi (seed={seed}) tayyorlanmadi, o'tkazib yuborildi: {e}")
        return None, None
    return ds, groups


def _final_fit(cfg, ctx, names, hp, data_, groups, block_size):
    dataset = data_["dataset"]
    K = cfg.final_bg_draws
    tune = _tune_names(cfg, names)
    state = {"done": 0}
    n_units = len(tune) + K * len(names)
    prog = ctx.prog(0.80, 0.92)

    def bump(msg, inner=0.0):
        prog((state["done"] + inner) / n_units, msg)

    ctx.log(f"\n=== Yakuniy modellar: {K} ta fon tanlovi x {len(names)} model"
            f"{' (tuning: ' + ', '.join(tune) + ')' if tune else ''} ===")
    if tune and K > 1:
        ctx.log("  Tuning faqat 0-fon tanlovida (CV dataset'ida) bajariladi, qolgan tanlovlar shu giperparametrlarni "
                "ishlatadi (narx K marta oshmasligi uchun).")
    final_models = {m: [] for m in names}
    final_hp = {m: [] for m in names}
    final_tuning = {}
    use_hp = {m: dict(hp[m]) for m in names}
    n_drawn = 0
    for k in range(K):
        ctx.check()
        if k == 0:
            ds, g = dataset, groups
        else:
            ds, g = _draw_dataset(cfg, ctx, k, data_, block_size)
            if ds is None:
                state["done"] += len(names)
                continue
        n_drawn += 1
        for m in names:
            ctx.check()
            if k == 0 and m in tune:
                bump(f"Yakuniy tuning: {m}")
                use_hp[m], final_tuning[m] = _tune_final(
                    m, use_hp[m], ds, g, cfg, ctx,
                    lambda frac=0.0, msg="", _m=m: bump(msg or f"Yakuniy tuning: {_m}", frac))
                state["done"] += 1
            seed = cfg.seed + k
            set_global_seed(seed)
            model = make_model(m, use_hp[m], calibrate=cfg.calibrate, calibration_method=cfg.calibration_method,
                               calibration_cv=cfg.calibration_cv, n_jobs=cfg.n_jobs, seed=seed,
                               n_features=ds.X.shape[1])
            patches = ds.get_patches(use_hp[m]["window"]) if model.input_kind == "patch" else None
            bump(f"Yakuniy model: {m} ({k + 1}/{K})")
            tag = f"{k + 1}/{K}-tanlov"
            try:
                model.fit(ds.X, ds.y, patches=patches, cancel=ctx.cancel, log_fn=_prefixed(ctx.log, f"{m} {tag}"))
            except CancelledError:
                raise
            except Exception as e:
                ctx.log(f"  XATO: {m} yakuniy modeli ({tag}) o'qitilmadi: {type(e).__name__}: {e}")
                raise RuntimeError(f"{m} yakuniy modeli ({tag}) o'qitilmadi: {type(e).__name__}: {e}") from e
            del patches
            _log_fit_info(ctx, m, tag, model)
            final_models[m].append(model)
            final_hp[m].append(dict(use_hp[m]))
            state["done"] += 1
    ctx.log(f"Yakuniy modellar tayyor: {n_drawn}/{K} fon tanlovi ishlatildi.")
    ctx.report(0.92, "Yakuniy modellar tayyor")
    return final_models, final_hp, final_tuning


# ---------------------------------------------------------------------------
# Importance / SHAP / bo'sag'lar
# ---------------------------------------------------------------------------
def _importance(cfg, ctx, names, spatial_block, final_models, feature_names):
    perm = spatial_block.get("perm_importance") or {}
    p = len(feature_names)
    models, via_perm, via_mdi = {}, [], []
    for m in names:
        pi = perm.get(m)
        if pi is not None and np.size(pi.get("mean")) == p and np.isfinite(pi["mean"]).any():
            models[m] = {"mean": np.asarray(pi["mean"], dtype=float), "std": np.asarray(pi["std"], dtype=float),
                         "per_fold": pi.get("per_fold"), "n_folds": pi.get("n_folds"),
                         "n_valid_folds": pi.get("n_valid_folds"), "source": "permutation"}
            via_perm.append(m)
            continue
        imp = mdi_importance(final_models[m][0])
        if imp is not None and imp.size == p:
            models[m] = {"mean": imp, "std": np.zeros(p), "source": "mdi"}
            via_mdi.append(m)
            msg = (f"{m}: permutation importance mavjud emas, zaxira sifatida MDI/gain ishlatildi "
                   f"(korrelyatsiyalangan prediktorlarda noto'g'ri bo'lishi mumkin).")
            if cfg.perm_importance:
                ctx.warn(msg)
            else:
                ctx.log("  " + msg)
        elif cfg.perm_importance:
            ctx.warn(f"{m}: importance hisoblanmadi (permutation yo'q, MDI qo'llab-quvvatlanmaydi).")
    if via_perm and not via_mdi:
        method = "out-of-fold permutation importance (spatial CV, AUC pasayishi)"
    elif via_perm:
        method = (f"out-of-fold permutation ({', '.join(via_perm)}); MDI/gain zaxirasi ({', '.join(via_mdi)})")
    elif via_mdi:
        method = "MDI/gain (permutation importance o'chirilgan yoki hisoblanmadi)"
    else:
        method = "hisoblanmadi"
    ctx.log(f"Feature importance usuli: {method}.")
    if via_perm:
        ctx.log("  Eslatma: permutation importance one-hot kategorik ustunlarni alohida aralashtiradi.")
    return {"method": method, "models": models, "feature_names": list(feature_names)}


def _shap(cfg, ctx, final_models, dataset):
    if not cfg.shap_enabled:
        return None
    try:
        out = compute_shap_summary(final_models, dataset.X, dataset.feature_names, cfg.shap_max_background,
                                   cfg.seed, log_fn=ctx.log, cancel=ctx.cancel)
    except CancelledError:
        raise
    except Exception as e:
        ctx.warn(f"SHAP hisoblanmadi ({type(e).__name__}: {e}).")
        return None
    if out:
        ctx.log("  Eslatma: SHAP birliklari modelga bog'liq (RF: ehtimollik, XGBoost: log-odds) - "
                "modellararo magnitudani bitta o'qda solishtirmang.")
    return out


# ---------------------------------------------------------------------------
# run_training
# ---------------------------------------------------------------------------
def run_training(cfg, *, log_fn=None, progress_fn=None, cancel=None):
    """To'liq o'qitish ish oqimi. Qaytaradi: TrainingResult (docs/ARCHITECTURE.md 3.8). cfg.validate() muammolari =>
    ValueError. Bekor qilinsa CancelledError (xom) ko'tariladi. Log fayli: <cfg.output_dir>/mpm_run_<N>.log (UTF-8)."""
    ctx = _Ctx(log_fn, progress_fn, cancel)
    try:
        return _run_training(cfg, ctx)
    except CancelledError:
        ctx.log("Hisoblash foydalanuvchi tomonidan to'xtatildi.")
        raise
    except Exception as e:
        ctx.log(f"XATO: {type(e).__name__}: {e}")
        raise
    finally:
        ctx.close()


def _run_training(cfg, ctx):
    t_all = time.perf_counter()
    ctx.check()
    cfg, hp, names = _prepare(cfg, ctx)
    set_global_seed(cfg.seed)
    ctx.report(0.0, "Boshlandi")

    data_ = _load_data(cfg, ctx, hp, names)
    raster, fpipe, dataset = data_["raster"], data_["fpipe"], data_["dataset"]
    block_size, groups = _block_size(cfg, ctx, raster, dataset, hp, names)

    random_block, spatial_block = _run_cvs(cfg, ctx, names, hp, dataset, groups)

    bg_sens = None
    if cfg.bg_sensitivity_enabled:
        with ctx.stage("bg_sensitivity"):
            ctx.check()
            ctx.log(f"\n=== Fon tanloviga sezgirlik ({cfg.bg_sensitivity_draws} ta mustaqil tanlov, bazaviy hp) ===")
            bg_sens = run_background_sensitivity(
                data_["aoi_gdf"], data_["positive_gdf"], raster, fpipe, hp, model_names=names,
                n_background=cfg.n_background, min_distance=cfg.min_distance, strategy=cfg.background_strategy,
                block_size=block_size, n_draws=cfg.bg_sensitivity_draws, n_splits=cfg.n_splits,
                n_repeats=cfg.bg_sensitivity_repeats, calibrate=cfg.calibrate,
                calibration_method=cfg.calibration_method, calibration_cv=cfg.calibration_cv, n_jobs=cfg.n_jobs,
                seed=cfg.seed, log_fn=ctx.log, progress_fn=ctx.prog(0.70, 0.80), cancel=ctx.cancel)
    ctx.report(0.80, "Fon sezgirligi tugadi" if cfg.bg_sensitivity_enabled else "Fon sezgirligi o'chirilgan")

    with ctx.stage("final_fit"):
        final_models, final_hp, final_tuning = _final_fit(cfg, ctx, names, hp, data_, groups, block_size)

    with ctx.stage("importance"):
        ctx.check()
        ctx.report(0.92, "Importance hisoblanmoqda")
        importance = _importance(cfg, ctx, names, spatial_block, final_models, dataset.feature_names)
        ctx.report(0.95, "SHAP hisoblanmoqda")
        shap_res = _shap(cfg, ctx, final_models, dataset)
        thresholds = {name: float(m["threshold_youden"]) for name, m in spatial_block["metrics"].items()}
        ctx.log("Youden bo'sag'lari (spatial OOF): " + ", ".join(f"{k}={v:.3f}" for k, v in thresholds.items()))
        ctx.report(0.98, "Importance tayyor")

    ctx.timings["total"] = time.perf_counter() - t_all
    ctx.log("\nVaqtlar (s): " + ", ".join(f"{k}={v:.1f}" for k, v in ctx.timings.items()))
    ctx.log(_NOTE_INDEX)
    ctx.report(1.0, "Tayyor")
    return {
        "cfg": cfg.to_dict(), "band_names": list(raster.band_names), "feature_names": list(dataset.feature_names),
        "categorical_layers": list(raster.categorical), "raster": raster, "pipeline": fpipe, "dataset": dataset,
        "groups": groups, "block_size": float(block_size), "n_positive": dataset.n_pos,
        "n_background": dataset.n_neg, "spatial": spatial_block, "random": random_block,
        "bg_sensitivity": bg_sens, "final_models": final_models, "final_hyperparams": final_hp,
        "importance": importance, "shap": shap_res, "data_dictionary": data_["data_dictionary"],
        "metadata_csv_path": data_["metadata_csv_path"], "diagnostics": data_["diagnostics"],
        "thresholds": thresholds, "versions": collect_versions(), "timings": dict(ctx.timings),
        "log_path": ctx.log_path, "warnings": list(ctx.warnings), "aoi_gdf": data_["aoi_gdf"],
        "positive_gdf": data_["positive_gdf"], "background_gdf": data_["background_gdf"],
        # qo'shimcha (shartnomadan tashqari) kalitlar
        "model_names": list(names), "final_tuning": final_tuning, "pixel_size": raster.pixel_size,
    }


# ---------------------------------------------------------------------------
# run_prediction
# ---------------------------------------------------------------------------
def run_prediction(result, *, out_dir=None, class_method=None, n_classes=None, class_breaks=None, batch_size=8192,
                   log_fn=None, progress_fn=None, cancel=None):
    """Butun maydon bo'yicha prospektivlik indeksi xaritalari + sinflash + maydon statistikasi + success-rate.
    class_* berilmasa result["cfg"] dan olinadi. out_dir bo'lsa GeoTIFF'lar va predictor_data_dictionary.csv saqlanadi.
    Qaytaradi: {"maps","uncertainty","valid_mask","class_map","class_breaks","class_stats","success_curve","saved_paths"}."""
    log = log_fn or noop_log
    cfg = result.get("cfg") or {}
    method = class_method or cfg.get("class_method") or "quantile"
    ncls = int(n_classes if n_classes is not None else cfg.get("n_classes", 5))
    breaks = class_breaks if class_breaks is not None else cfg.get("class_breaks")
    raster, dataset = result["raster"], result["dataset"]
    log("Prognoz xaritalari hisoblanmoqda (chiqish: prospektivlik indeksi 0-1, ehtimollik emas)...")
    pred = predict_probability_maps(raster, result["pipeline"], result["final_models"], batch_size=batch_size,
                                    log_fn=log, progress_fn=sub_progress(progress_fn, 0.0, 0.85), cancel=cancel)
    check_cancel(cancel)
    ens = pred["maps"][ENSEMBLE_NAME]
    class_map, used = classify_map(ens, method, ncls, breaks)
    pos = np.asarray(dataset.y) == 1
    rows, cols = dataset.rows[pos], dataset.cols[pos]
    stats = class_area_stats(class_map, raster.pixel_size, rows=rows, cols=cols, n_classes=len(used) + 1)
    curve = success_rate_curve(ens, rows, cols)
    log(f"Sinflash: {method}, {len(used) + 1} sinf; success-rate AUC={curve['auc']:.3f} ({curve['n_pos']} kon). "
        f"Eslatma: success-rate o'qitish nuqtalarida hisoblangan (optimistik), mustaqil baho emas.")
    if "CNN" in result["final_models"] and len(result["final_models"]) > 1:
        log("  OGOHLANTIRISH: " + _NOTE_CNN)
    saved = []
    if out_dir:
        check_cancel(cancel)
        saved = save_rasters(out_dir, pred["maps"], raster.profile, class_map=class_map,
                             uncertainty=pred["uncertainty"], log_fn=log)
        dd = result.get("data_dictionary")
        if dd is not None:
            saved.append(save_data_dictionary(dd, out_dir, log_fn=log))
    if progress_fn is not None:
        progress_fn(1.0, "Prognoz tayyor")
    return {"maps": pred["maps"], "uncertainty": pred["uncertainty"], "valid_mask": pred["valid_mask"],
            "class_map": class_map, "class_breaks": used, "class_stats": stats, "success_curve": curve,
            "saved_paths": saved, "class_method": method, "n_classes": len(used) + 1}


# ---------------------------------------------------------------------------
# export_results
# ---------------------------------------------------------------------------
def _clean(o):
    """JSON uchun: numpy/DataFrame/NaN -> oddiy Python (NaN/inf -> None)."""
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
        return _clean(o.tolist())
    if isinstance(o, pd.DataFrame):
        return _clean(o.to_dict(orient="records"))
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    return str(o)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, ensure_ascii=False, indent=2)
    return path


def _importance_df(result):
    imp = result.get("importance") or {}
    feats = imp.get("feature_names") or result.get("feature_names") or []
    rows = []
    for m, d in (imp.get("models") or {}).items():
        mean, std = np.asarray(d["mean"], dtype=float), np.asarray(d["std"], dtype=float)
        for i, f in enumerate(feats):
            rows.append({"model": m, "feature": f, "mean": mean[i], "std": std[i] if i < std.size else np.nan,
                         "source": d.get("source", "")})
    return pd.DataFrame(rows, columns=["model", "feature", "mean", "std", "source"])


def _tuning_tables(result):
    """(tuned_params DataFrame, tuning_trials DataFrame) - nested (tashqi fold) va yakuniy yozuvlar."""
    summ, trials = [], []

    def add(model, scope, rec):
        base = {"model": model, "scope": scope, "repeat": rec.get("repeat"), "fold": rec.get("fold")}
        summ.append({**base, "best_score": rec.get("best_score"), "base_score": rec.get("base_score"),
                     "n_trials": rec.get("n_trials"), "scoring": rec.get("scoring"),
                     "best_params": json.dumps(_clean(rec.get("best_params")), ensure_ascii=False),
                     "fallback": rec.get("fallback") or rec.get("skipped_reason") or ""})
        for i, t in enumerate(rec.get("trials") or []):
            trials.append({**base, "trial": i, "score": t.get("score"), "std": t.get("std"),
                           "params": json.dumps(_clean(t.get("params")), ensure_ascii=False)})

    for m, recs in ((result.get("spatial") or {}).get("tuned_params") or {}).items():
        for rec in recs:
            if isinstance(rec, dict):
                add(m, "nested", rec)
    for m, rec in (result.get("final_tuning") or {}).items():
        add(m, "final", rec)
    df_s, df_t = pd.DataFrame(summ), pd.DataFrame(trials)
    for df, cols in ((df_s, ("repeat", "fold", "n_trials")), (df_t, ("repeat", "fold", "trial"))):
        for c in cols:                          # yakuniy yozuvda repeat/fold yo'q: butun son ustuni float'ga aylanmasin
            if c in df.columns:
                df[c] = pd.array([None if v is None or v != v else int(v) for v in df[c]], dtype="Int64")
    return df_s, df_t


def _oof_df(result):
    sp = result["spatial"]
    ds = result["dataset"]
    df = pd.DataFrame({"x": ds.coords[:, 0], "y": ds.coords[:, 1], "y_true": np.asarray(ds.y, dtype=int),
                       "fold": sp["fold_map"]})
    for name, reps in sp["oof"].items():
        df[name] = np.mean(np.vstack([np.asarray(r, dtype=float) for r in reps]), axis=0)
    df["Ensemble"] = np.asarray(sp["ensemble_oof"], dtype=float)
    return df


def _fold_table_df(result):
    parts = []
    for blk in (result.get("random"), result.get("spatial")):
        if blk is not None:
            t = blk["fold_table"].copy()
            t.insert(0, "cv", blk["mode"])
            parts.append(t)
    return pd.concat(parts, ignore_index=True)


def _summary_text(result, prediction):
    cfg = result.get("cfg") or {}
    sp = result["spatial"]
    lines = [f"MPM ML v{MPM_VERSION} - o'qitish xulosasi ({datetime.datetime.now():%Y-%m-%d %H:%M})", "=" * 64, "",
             _NOTE_INDEX, "",
             f"Ma'lumot: {result['n_positive']} musbat + {result['n_background']} fon nuqta; "
             f"{len(result['feature_names'])} feature ({len(result['band_names'])} band"
             f"{', kategorik: ' + ', '.join(result['categorical_layers']) if result['categorical_layers'] else ''}).",
             f"Spatial blok o'lchami: {result['block_size']:,.0f} m; CV: {sp.get('n_splits')}-fold x "
             f"{sp.get('n_repeats')} takror; seed={cfg.get('seed')}.",
             f"Modellar: {', '.join(result.get('model_names') or result['final_models'])}; yakuniy fon tanlovlari: "
             f"{max(len(v) for v in result['final_models'].values())} ta.", "",
             "Spatial block CV natijalari (asosiy):"]
    for name, m in sp["metrics"].items():
        lo, hi = m["auc_ci95"]
        ci = f"95% CI {lo:.3f}-{hi:.3f}" if np.isfinite(lo) and np.isfinite(hi) else "CI hisoblanmadi"
        lines.append(f"  {name}: AUC={m['auc']:.3f} +/- {m['auc_std']:.3f} ({ci}), PR-AUC={m['pr_auc']:.3f}, "
                     f"Sens/Spec@Youden={m['sensitivity']:.2f}/{m['specificity']:.2f}, Brier={m['brier']:.3f}")
    lines.append("  Eslatma: AUC - repeat'lar o'rtachasi; CI esa mean-proba ustida blok-bootstrap bilan topilgan, "
                 "shuning uchun nuqtaviy qiymat CI chetiga yaqin bo'lishi mumkin.")
    rnd = result.get("random")
    if rnd is not None:
        lines += ["", "Random vs Spatial (optimizm = leakage taxmini):"]
        for name, m in sp["metrics"].items():
            r = rnd["metrics"].get(name, {}).get("auc", float("nan"))
            lines.append(f"  {name}: random AUC={r:.3f}, spatial AUC={m['auc']:.3f}, farq={r - m['auc']:+.3f}")
    bg = result.get("bg_sensitivity")
    if bg:
        lines += ["", f"Fon tanloviga sezgirlik ({bg.get('n_draws')} ta tanlov):"]
        for name, s in bg["summary"].items():
            lines.append(f"  {name}: AUC {s['mean']:.3f} +/- {s['std']:.3f} [{s['min']:.3f}-{s['max']:.3f}]")
    imp = result.get("importance") or {}
    if imp.get("models"):
        lines += ["", f"Feature importance - {imp.get('method')}; eng muhim 5 ta:"]
        feats = imp.get("feature_names") or []
        for m, d in imp["models"].items():
            mean = np.nan_to_num(np.asarray(d["mean"], dtype=float), nan=-np.inf)
            top = np.argsort(-mean)[:5]
            lines.append(f"  {m}: " + ", ".join(f"{feats[i]} ({d['mean'][i]:.3f})" for i in top if np.isfinite(mean[i])))
    ft = result.get("final_tuning") or {}
    if ft:
        lines += ["", "Yakuniy giperparametr qidiruvi:"]
        for m, rec in ft.items():
            lines.append(f"  {m}: {rec.get('scoring')}={rec.get('best_score')} ; {rec.get('best_params')}")
    if prediction is not None:
        cs = prediction.get("success_curve") or {}
        lines += ["", f"Prognoz: {prediction.get('n_classes')} sinf, success-rate AUC={cs.get('auc', float('nan')):.3f} "
                      f"(o'qitish nuqtalarida - optimistik)."]
    if "CNN" in result["final_models"] and len(result["final_models"]) > 1:
        lines += ["", _NOTE_CNN]
    if result.get("warnings"):
        lines += ["", f"Ogohlantirishlar ({len(result['warnings'])}):"] + [f"  - {w}" for w in result["warnings"][:50]]
    lines += ["", "Vaqtlar (s): " + ", ".join(f"{k}={v:.1f}" for k, v in (result.get("timings") or {}).items())]
    return "\n".join(lines) + "\n"


def export_results(result, out_dir, prediction=None, log_fn=None):
    """Natijalarni out_dir'ga yozadi (CSV/XLSX/JSON/TXT). Qaytaradi: yozilgan fayl yo'llari ro'yxati."""
    log = log_fn or noop_log
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    def csv(name, df):
        p = os.path.join(out_dir, name)
        df.to_csv(p, index=False, encoding="utf-8")
        paths.append(p)
        log(f"Saqlandi: {p}")

    def js(name, obj):
        p = _write_json(os.path.join(out_dir, name), obj)
        paths.append(p)
        log(f"Saqlandi: {p}")

    sp, rnd = result["spatial"], result.get("random")
    csv("metrics_spatial.csv", sp["metrics_df"])
    if rnd is not None:
        csv("metrics_random.csv", rnd["metrics_df"])
    if importlib.util.find_spec("openpyxl") is not None:
        xp = os.path.join(out_dir, "metrics_spatial.xlsx")
        try:
            with pd.ExcelWriter(xp, engine="openpyxl") as xw:
                sp["metrics_df"].to_excel(xw, sheet_name="Spatial CV", index=False)
                if rnd is not None:
                    rnd["metrics_df"].to_excel(xw, sheet_name="Random CV", index=False)
                if prediction is not None and prediction.get("class_stats") is not None:
                    prediction["class_stats"].to_excel(xw, sheet_name="Sinflar", index=False)
            paths.append(xp)
            log(f"Saqlandi: {xp}")
        except Exception as e:
            log(f"  OGOHLANTIRISH: XLSX yozib bo'lmadi ({type(e).__name__}: {e}); faqat CSV saqlandi.")
    else:
        log("  OGOHLANTIRISH: openpyxl o'rnatilmagan, XLSX yozilmadi (faqat CSV).")

    cfg = result.get("cfg") or {}
    js("hyperparameters_used.json", {"base": (cfg.get("hyperparams") or {}), "final": result.get("final_hyperparams"),
                                     "tuning": cfg.get("tuning"), "models": result.get("model_names")})
    js("run_config.json", {**cfg, "band_names": result["band_names"], "block_size": result["block_size"],
                           "n_positive": result["n_positive"], "n_background": result["n_background"],
                           "mpm_version": MPM_VERSION})
    imp = _importance_df(result)
    if len(imp):
        csv("importance.csv", imp)
    tp, tt = _tuning_tables(result)
    if len(tp):
        csv("tuned_params.csv", tp)
    if len(tt):
        csv("tuning_trials.csv", tt)
    csv("fold_table.csv", _fold_table_df(result))
    csv("oof_predictions.csv", _oof_df(result))
    if result.get("data_dictionary") is not None:
        paths.append(save_data_dictionary(result["data_dictionary"], out_dir, log_fn=log))
    js("versions.json", result.get("versions") or {})
    if result.get("bg_sensitivity"):
        js("bg_sensitivity.json", result["bg_sensitivity"])
    if prediction is not None and prediction.get("class_stats") is not None:
        csv("class_stats.csv", prediction["class_stats"])
    sp_path = os.path.join(out_dir, "summary.txt")
    with open(sp_path, "w", encoding="utf-8") as f:
        f.write(_summary_text(result, prediction))
    paths.append(sp_path)
    log(f"Saqlandi: {sp_path}")
    return paths


# ---------------------------------------------------------------------------
# estimate_cost_text
# ---------------------------------------------------------------------------
def _fit_mult(model, cfg, calibrated=None):
    """Bitta o'qitishning taxminiy ichki fit soni: kalibrlangan sklearn (calibration_cv + 1), SVM har doim."""
    cal = cfg.calibrate if calibrated is None else calibrated
    if model == "CNN":
        return 1
    return (int(cfg.calibration_cv) + 1) if (cal or model == "SVM") else 1


def _estimate(cfg):
    names = cfg.enabled_models()
    folds = int(cfg.n_splits) * int(cfg.n_repeats)
    cv_folds = folds * (2 if cfg.run_random_cv else 1)
    tune = _tune_names(cfg, names)
    t = cfg.tuning
    per_cand = int(t.n_iter) * int(t.inner_splits)
    nested = (t.mode == "nested")
    n_final = max(1, int(cfg.final_bg_draws))
    rows = []          # (sarlavha, fitlar_soni, ichki_fit_soni)
    cnn_fits = 0
    total_eff = 0

    def add(label, per_model_counts):
        nonlocal cnn_fits, total_eff
        cnt = sum(per_model_counts.values())
        eff = sum(c * _fit_mult(m, cfg) for m, c in per_model_counts.items())
        cnn_fits += per_model_counts.get("CNN", 0)
        total_eff += eff
        rows.append((label, cnt, eff))

    add(f"CV: {'random + ' if cfg.run_random_cv else ''}spatial, {cfg.n_splits}-fold x {cfg.n_repeats} takror "
        f"= {cv_folds} fold", {m: cv_folds for m in names})
    if tune and nested:
        add(f"Nested tuning (spatial, {folds} tashqi fold x {len(tune)} model x n_iter {t.n_iter} x ichki fold "
            f"{t.inner_splits})", {m: folds * per_cand for m in tune})
    if tune:
        add(f"Yakuniy tuning (butun ma'lumotda, {len(tune)} model x {per_cand})", {m: per_cand for m in tune})
    add(f"Yakuniy modellar ({n_final} fon tanlovi)", {m: n_final for m in names})
    if cfg.bg_sensitivity_enabled:
        n_bg = int(cfg.bg_sensitivity_draws) * int(cfg.bg_sensitivity_repeats) * int(cfg.n_splits)
        add(f"Fon sezgirligi ({cfg.bg_sensitivity_draws} tanlov x {cfg.bg_sensitivity_repeats} takror x "
            f"{cfg.n_splits}-fold)", {m: n_bg for m in names})
    # tuning ichida kalibrlash o'chirilgan: ichki fit sonini tuzatish (sklearn modellari uchun mult=1, SVM baribir)
    if tune:
        corr = 0
        for m in tune:
            c = (folds * per_cand if nested else 0) + per_cand
            corr += c * (_fit_mult(m, cfg) - _fit_mult(m, cfg, calibrated=False))
        total_eff -= corr

    lines = ["Taxminiy hisob-kitob (o'qitishlar soni):",
             f"  Modellar: {', '.join(names) if names else 'yo`q'}."]
    for label, cnt, eff in rows:
        lines.append(f"  - {label}: {cnt} ta o'qitish" + (f" (kalibrlash bilan ~{eff} ichki fit)" if eff != cnt else "."))
    lines.append(f"  Jami taxminan {total_eff:,} ichki fit.")
    if cfg.perm_importance:
        lines.append(f"  Permutation importance (spatial CV, 1-takror): har fold uchun feature soni x "
                     f"{cfg.perm_importance_repeats} takror bashorat.")
    if cfg.n_bootstrap:
        lines.append(f"  Blok-bootstrap CI: {cfg.n_bootstrap} resample (katta ma'lumotda sekin bo'lishi mumkin; "
                     f"0 = o'chirish).")
    if total_eff > 4000:
        lines.append("OGOHLANTIRISH: hisoblash juda og'ir (soatlar). CV takrorlari/n_iter/modellar sonini kamaytiring.")
    elif total_eff > 800:
        lines.append("OGOHLANTIRISH: hisoblash o'rtacha og'ir (o'nlab daqiqa bo'lishi mumkin).")
    if tune:
        extra = ""
        if "RandomForest" in tune:
            sp = t.resolved_space("RandomForest").get("n_estimators", {})
            if sp.get("min") is not None and sp.get("max") is not None:
                mid = (float(sp["min"]) + float(sp["max"])) / 2
                extra = (f" RF: har tashqi fold'da taxminan {per_cand} x ~{mid:.0f} = ~{per_cand * mid:,.0f} "
                         f"daraxt quriladi.")
        lines.append(f"OGOHLANTIRISH: tuning qimmat: har tashqi fold'da n_iter x ichki fold = {per_cand} o'qitish/model."
                     f"{extra} Kerak bo'lmasa n_iter/inner_splits ni kamaytiring.")
    if "CNN" in names:
        lines.append(f"OGOHLANTIRISH: CNN qimmat (~{cnn_fits} o'qitish); TensorFlow xotirasi har o'qitishda "
                     f"~{_TF_MB_PER_FIT} MB o'sadi (jami ~{cnn_fits * _TF_MB_PER_FIT:,} MB), ayniqsa CNN tuning va "
                     f"nested CV da. CNN epochs/n_repeats ni kichik tuting.")
        if "CNN" in tune:
            lines.append("OGOHLANTIRISH: CNN tuning yoqilgan: yuzlab o'qitish bo'lishi mumkin - odatda tavsiya etilmaydi.")
    return "\n".join(lines)


def estimate_cost_text(cfg):
    """GUI uchun taxminiy fold/fit soni va xarajat haqida o'zbekcha ogohlantirish matni."""
    if isinstance(cfg, dict):
        cfg = RunConfig.from_dict(cfg)
    try:
        return _estimate(cfg)
    except Exception as e:                      # taxmin hech qachon ishni to'xtatmasin
        return f"Narxni baholab bo'lmadi ({type(e).__name__}: {e})."
