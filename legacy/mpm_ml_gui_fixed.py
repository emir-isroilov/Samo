# -*- coding: utf-8 -*-
"""
MPM ML GUI - Oltin ma'danlashuvi prospektivligini bashoratlash uchun
Random Forest, SVM, XGBoost, CNN va Ansambl (soft-voting) modellari

Muallif uchun eslatma (o'rnatish):
    conda ichida quyidagilarni o'rnating:
    pip install PyQt5 rasterio geopandas shapely scikit-learn xgboost
    pip install tensorflow matplotlib joblib pandas numpy

Ishlash tartibi:
    1. TIFF qatlamlar joylashgan papkani tanlang (har bir .tif -> alohida feature)
    2. Target (ma'lum kon/ma'dan namoyon) nuqtalari joylashgan papkani tanlang (.shp)
    3. Tadqiqot maydoni konturi (AOI) shapefile joylashgan papkani tanlang (.shp)
    4. CRS - EPSG:28411 (Pulkovo 1942 / Gauss-Kruger zone 11) qattiq belgilangan,
       barcha kirish qatlamlari shu tizimga avtomatik reproyeksiya qilinadi
    5. "Modellarni o'qitish" tugmasini bosing
    6. ROC-AUC, Feature Importance natijalarini ko'ring
    7. "Prognoz xarita yaratish" orqali butun maydon uchun ehtimollik xaritasini
       GeoTIFF sifatida saqlang
"""

import sys
import os
import glob
import traceback
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ixtiyoriy kutubxonalar - mavjud bo'lmasa dastur ishlashda davom etadi,
# faqat tegishli model o'chiriladi
# ---------------------------------------------------------------------------
XGBOOST_AVAILABLE = True
try:
    from xgboost import XGBClassifier
except Exception:
    XGBOOST_AVAILABLE = False

TF_AVAILABLE = True
try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow.keras import layers, models as keras_models, callbacks
    tf.get_logger().setLevel("ERROR")
except Exception:
    TF_AVAILABLE = False

import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio import features as rio_features
import geopandas as gpd
from shapely.geometry import Point

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    balanced_accuracy_score, f1_score, brier_score_loss,
)
from scipy.spatial.distance import pdist, squareform
import joblib

SHAP_AVAILABLE = True
try:
    import shap
except Exception:
    SHAP_AVAILABLE = False

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit, QProgressBar,
    QTabWidget, QCheckBox, QGroupBox, QMessageBox, QSpinBox, QDoubleSpinBox,
    QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Global konstantalar
# ---------------------------------------------------------------------------
TARGET_EPSG = 28411   # Pulkovo 1942 / Gauss-Kruger zone 11 (GK 1942 zone 11)
RANDOM_STATE = 42
NODATA_FILL = np.nan


def find_shapefile(folder_path):
    """Berilgan papka ichidan birinchi .shp faylni topadi."""
    if not folder_path or not os.path.isdir(folder_path):
        return None
    shp_list = sorted(glob.glob(os.path.join(folder_path, "*.shp")))
    return shp_list[0] if shp_list else None


def find_tiff_files(folder_path):
    """Berilgan papka ichidagi barcha .tif/.tiff fayllarni topadi (feature qatlamlar)."""
    if not folder_path or not os.path.isdir(folder_path):
        return []
    files = sorted(
        glob.glob(os.path.join(folder_path, "*.tif")) +
        glob.glob(os.path.join(folder_path, "*.tiff"))
    )
    return files


# ---------------------------------------------------------------------------
# Raster stack: barcha TIFF qatlamlarni bitta umumiy grid (reference)ga
# reproyeksiya/resample qilib, 3D massivga (n_bands, rows, cols) yig'adi
# ---------------------------------------------------------------------------
def load_and_align_rasters(tiff_paths, log_fn=print, assume_crs_if_missing=True,
                            fallback_epsg=TARGET_EPSG):
    """
    Birinchi TIFF - referens grid bo'ladi (CRS, transform, shape, resolution).
    Qolgan barcha qatlamlar shu grid ustiga bilinear resample qilinadi.
    Barchasi TARGET_EPSG ga reproyeksiya qilinadi.

    Agar biror TIFF faylida CRS metadata yozilmagan bo'lsa (masalan, Surfer yoki
    boshqa dasturdan georeferences saqlanmagan holda eksport qilingan grid):
      - assume_crs_if_missing=True bo'lsa -> fayl allaqachon fallback_epsg
        tizimida deb hisoblanadi (reproyeksiya qilinmaydi, faqat CRS "yopishtiriladi"),
        chunki koordinatalarning o'zi to'g'ri, faqat metadata yetishmayapti.
      - False bo'lsa -> xatolik chiqariladi.

    Natija: stack (n_bands, H, W) float32, ref_profile (rasterio profile), band_names,
            ref_transform, tech_metadata (har bir band uchun avtomatik texnik metadata
            ro'yxati - taqrizda talab qilingan predictor/data dictionary'ning texnik
            qismi: manba CRS, resolution, bounds, nodata, va reproject'dan keyingi
            qiymat statistikasi)
    """
    if not tiff_paths:
        raise ValueError("TIFF fayllar topilmadi.")

    band_names = [os.path.splitext(os.path.basename(p))[0] for p in tiff_paths]
    tech_metadata = []

    def _resolve_crs(src, path):
        """CRS mavjud bo'lmasa, fallback_epsg ni qo'llaydi (assume_crs_if_missing=True bo'lsa)."""
        if src.crs is not None:
            return src.crs
        if assume_crs_if_missing:
            log_fn(f"  OGOHLANTIRISH: '{os.path.basename(path)}' faylida CRS yozilmagan. "
                   f"Fayl koordinatalari allaqachon EPSG:{fallback_epsg} tizimida deb "
                   f"qabul qilinmoqda (reproyeksiya qilinmaydi, faqat CRS belgilanadi).")
            return rasterio.crs.CRS.from_epsg(fallback_epsg)
        raise ValueError(
            f"'{path}' faylida CRS aniqlanmagan. "
            "Iltimos, TIFF fayllarga CRS yozilganini tekshiring, yoki dasturda "
            "'CRS yo'q bo'lsa EPSG:28411 deb qabul qilinsin' katagini belgilang."
        )

    def _read_and_sanitize(path, crs):
        """
        Bandni o'qib, nodata/inf/juda katta (senator) qiymatlarni NaN ga aylantiradi.
        Bu reproject (bilinear) jarayonida noto'g'ri qiymatlarning qo'shni piksellarga
        "sizib chiqishi"ning oldini oladi - ba'zi geofizik filtr gridlarida (masalan,
        TDR, Euler dekonvolyutsiya) chekka effektlari tufayli inf/-inf yoki juda
        katta sentinel qiymatlar uchrashi mumkin.
        """
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float64)
            nodata_val = src.nodata
            if nodata_val is not None:
                arr[np.isclose(arr, nodata_val, equal_nan=False)] = np.nan
            # inf/-inf va haddan tashqari katta (float32 range'dan chetga chiqqan) qiymatlar
            arr[~np.isfinite(arr)] = np.nan
            extreme = np.abs(arr) > 1e15
            if np.any(extreme):
                arr[extreme] = np.nan
            return arr.astype(np.float32), src.transform, src.width, src.height, src.bounds

    # --- Referens grid: birinchi faylni TARGET_EPSG ga reproyeksiya qilib olamiz
    ref_arr, ref_src_transform, ref_src_w, ref_src_h, ref_src_bounds = _read_and_sanitize(
        tiff_paths[0], None
    )
    with rasterio.open(tiff_paths[0]) as ref_src:
        ref_src_crs = _resolve_crs(ref_src, tiff_paths[0])
        ref_profile_base = ref_src.profile.copy()

    ref_transform, ref_width, ref_height = calculate_default_transform(
        ref_src_crs, f"EPSG:{TARGET_EPSG}",
        ref_src_w, ref_src_h, *ref_src_bounds
    )
    ref_profile = ref_profile_base
    ref_profile.update({
        "crs": f"EPSG:{TARGET_EPSG}",
        "transform": ref_transform,
        "width": ref_width,
        "height": ref_height,
        "count": 1,
        "dtype": "float32",
        "nodata": -9999.0,
    })

    ref_band = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
    reproject(
        source=ref_arr,
        destination=ref_band,
        src_transform=ref_src_transform,
        src_crs=ref_src_crs,
        src_nodata=np.nan,
        dst_transform=ref_transform,
        dst_crs=f"EPSG:{TARGET_EPSG}",
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    stack = np.full((len(tiff_paths), ref_height, ref_width), np.nan, dtype=np.float32)
    stack[0] = ref_band
    log_fn(f"[1/{len(tiff_paths)}] Referens qatlam: {band_names[0]} "
           f"({ref_height} x {ref_width}, EPSG:{TARGET_EPSG})")

    def _tech_meta_entry(path, src_crs, src_transform, src_w, src_h, src_bounds, dst_band):
        with rasterio.open(path) as src:
            px_x = abs(src_transform.a)
            px_y = abs(src_transform.e)
            nodata_val = src.nodata
            dtype = src.dtypes[0]
        valid = dst_band[np.isfinite(dst_band)]
        return {
            "band_name": os.path.splitext(os.path.basename(path))[0],
            "file_path": path,
            "source_crs": str(src_crs),
            "source_resolution_x_m": round(float(px_x), 3),
            "source_resolution_y_m": round(float(px_y), 3),
            "source_width_px": int(src_w),
            "source_height_px": int(src_h),
            "source_bounds": tuple(round(v, 2) for v in src_bounds),
            "source_nodata": nodata_val,
            "source_dtype": dtype,
            "reprojected_resolution_m": round(float(abs(ref_transform.a)), 3),
            "valid_pixels_pct": round(100.0 * len(valid) / dst_band.size, 2) if dst_band.size else 0.0,
            "value_min": round(float(np.min(valid)), 4) if len(valid) else None,
            "value_max": round(float(np.max(valid)), 4) if len(valid) else None,
            "value_mean": round(float(np.mean(valid)), 4) if len(valid) else None,
            "value_std": round(float(np.std(valid)), 4) if len(valid) else None,
        }

    tech_metadata.append(_tech_meta_entry(
        tiff_paths[0], ref_src_crs, ref_src_transform, ref_src_w, ref_src_h,
        ref_src_bounds, ref_band
    ))

    for i, p in enumerate(tiff_paths[1:], start=1):
        src_arr, src_transform, src_w, src_h, src_bounds = _read_and_sanitize(p, None)
        with rasterio.open(p) as src:
            src_crs = _resolve_crs(src, p)
        dst_band = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst_band,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=ref_transform,
            dst_crs=f"EPSG:{TARGET_EPSG}",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        stack[i] = dst_band
        tech_metadata.append(_tech_meta_entry(p, src_crs, src_transform, src_w, src_h,
                                               src_bounds, dst_band))
        log_fn(f"[{i+1}/{len(tiff_paths)}] Qatlam moslashtirildi: {band_names[i]}")

    # Yakuniy xavfsizlik to'sig'i: reproject/resampling natijasida tasodifiy
    # hosil bo'lishi mumkin bo'lgan har qanday inf/juda katta qiymatni ham tozalaymiz
    stack[~np.isfinite(stack)] = np.nan

    return stack, ref_profile, band_names, ref_transform, tech_metadata


# ---------------------------------------------------------------------------
# PREDICTOR DATA DICTIONARY
#
# Taqrizning 3-band tanqidi: har bir predictor qatlami uchun manba, sana,
# masshtab, CRS, transformatsiya kabi ma'lumotlar hujjatlashtirilmagan.
# Avtomatik texnik metadata (tech_metadata, yuqorida) buning faqat bir
# qismini yopadi - manba tashkilot, so'rov sanasi, qo'llanilgan
# transformatsiya turi (RTP/TDR/HGM va h.k.) kabi "qo'lda" ma'lumotlar TIFF
# fayl ichida umuman yo'q. Shuning uchun quyidagi funksiya TIFF papkasi
# ichidan ixtiyoriy "metadata.csv" faylni qidiradi (band_name, source,
# survey_date, transformation, notes ustunlari bilan); topilmasa, to'ldirish
# uchun BO'SH SHABLON yaratadi va foydalanuvchini ogohlantiradi.
# ---------------------------------------------------------------------------
MANUAL_METADATA_FIELDS = ["band_name", "source_owner", "survey_or_scene_id",
                           "survey_date", "original_scale_or_resolution",
                           "transformation_applied", "notes"]


def load_or_create_manual_metadata(tiff_folder, band_names, log_fn=print):
    """
    metadata.csv topilsa -> o'qiydi va band_name bo'yicha dict qaytaradi.
    Topilmasa -> band_name ustuni to'ldirilgan, qolgani bo'sh shablon
    metadata.csv yaratadi va OGOHLANTIRISH beradi (chunki bu holatda
    predictor data dictionary texnik qismidan tashqari bo'sh qoladi).
    """
    meta_path = os.path.join(tiff_folder, "metadata.csv")
    if os.path.isfile(meta_path):
        try:
            df = pd.read_csv(meta_path)
            missing_cols = [c for c in MANUAL_METADATA_FIELDS if c not in df.columns]
            if missing_cols:
                log_fn(f"  OGOHLANTIRISH: metadata.csv'da ustunlar yetishmayapti: {missing_cols}")
            df = df.set_index("band_name") if "band_name" in df.columns else df
            manual = {name: (df.loc[name].to_dict() if name in df.index else {})
                      for name in band_names}
            log_fn(f"  metadata.csv topildi va o'qildi: {meta_path}")
            return manual, meta_path
        except Exception as e:
            log_fn(f"  OGOHLANTIRISH: metadata.csv o'qib bo'lmadi ({e}), bo'sh shablon ishlatiladi.")

    # shablon yaratamiz, lekin to'ldirmaymiz (foydalanuvchi keyinroq to'ldirishi kerak)
    template_rows = [{f: (name if f == "band_name" else "") for f in MANUAL_METADATA_FIELDS}
                      for name in band_names]
    try:
        pd.DataFrame(template_rows).to_csv(meta_path, index=False)
        log_fn(f"  OGOHLANTIRISH: metadata.csv topilmadi. Bo'sh shablon yaratildi: {meta_path}\n"
               f"  Predictor data dictionary'ni to'liq qilish uchun uni source_owner, "
               f"survey_date, transformation_applied va boshqa ustunlar bilan to'ldiring.")
    except Exception as e:
        log_fn(f"  OGOHLANTIRISH: metadata.csv shablonini yaratib bo'lmadi ({e}).")
    return {name: {} for name in band_names}, meta_path


def build_data_dictionary(band_names, tech_metadata, manual_metadata):
    """Texnik (avtomatik) va qo'lda kiritilgan metadata'ni bitta jadvalga birlashtiradi."""
    rows = []
    for name, tech in zip(band_names, tech_metadata):
        row = dict(tech)
        manual = manual_metadata.get(name, {}) or {}
        for f in MANUAL_METADATA_FIELDS:
            if f == "band_name":
                continue
            row[f] = manual.get(f, "")
        rows.append(row)
    return pd.DataFrame(rows)


def save_data_dictionary(df, out_dir, log_fn=print):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "predictor_data_dictionary.csv")
    df.to_csv(path, index=False)
    log_fn(f"Predictor data dictionary saqlandi: {path}")
    return path


def sample_points_from_stack(points_gdf, stack, transform):
    """Nuqta koordinatalari bo'yicha stackdan feature qiymatlarini oladi."""
    coords = [(geom.x, geom.y) for geom in points_gdf.geometry]
    rows_cols = [rasterio.transform.rowcol(transform, x, y) for x, y in coords]
    n_bands, H, W = stack.shape
    values = np.full((len(coords), n_bands), np.nan, dtype=np.float32)
    for idx, (row, col) in enumerate(rows_cols):
        if 0 <= row < H and 0 <= col < W:
            values[idx, :] = stack[:, row, col]
    return values


# ---------------------------------------------------------------------------
# Global seed - barcha tasodifiy jarayonlarni (numpy, python random, TensorFlow)
# bitta qiymatga mahkamlaydi, shunda bir xil ma'lumot bilan har doim bir xil
# natija chiqadi (to'liq reproduktivlik)
# ---------------------------------------------------------------------------
def set_global_seed(seed=RANDOM_STATE):
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass


from sklearn.model_selection import RepeatedStratifiedKFold


def generate_background_points(aoi_gdf, positive_gdf, n_points, min_distance, random_state=RANDOM_STATE):
    """
    AOI poligoni ichida, ma'lum (musbat) nuqtalardan min_distance dan yaqin
    bo'lmagan tasodifiy fon (pseudo-absence) nuqtalarini generatsiya qiladi.
    """
    rng = np.random.default_rng(random_state)
    aoi_union = aoi_gdf.geometry.unary_union
    minx, miny, maxx, maxy = aoi_union.bounds
    positive_union = positive_gdf.geometry.unary_union.buffer(min_distance)

    bg_points = []
    max_attempts = n_points * 200
    attempts = 0
    while len(bg_points) < n_points and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        pt = Point(x, y)
        if aoi_union.contains(pt) and not positive_union.contains(pt):
            bg_points.append(pt)

    return gpd.GeoDataFrame(geometry=bg_points, crs=aoi_gdf.crs)


# ---------------------------------------------------------------------------
# BACKGROUND (PSEUDO-ABSENCE) SEZGIRLIK TAHLILI
#
# Taqrizning 2-band tanqidi: 80 background nuqta faqat BITTA marta,
# BITTA random_state bilan generatsiya qilinadi va natija (AUC, xarita) shu
# bitta tanlovga qanchalik bog'liqligi hech qachon tekshirilmagan. Quyidagi
# funksiya butun pipeline'ni (background generatsiya -> feature extraction ->
# spatial-block CV) bir necha marta, HAR SAFAR BOSHQA background tanlovi
# bilan takrorlaydi va natijalar taqsimotini (AUC mean/std/min/max har bir
# tanlov bo'yicha) qaytaradi - bu "sensitivity to background sample" degan
# talabga javob beradi.
# ---------------------------------------------------------------------------
def run_background_sensitivity(aoi_gdf, positive_gdf, stack, transform,
                                n_background, min_distance, block_size,
                                n_draws, n_splits, n_repeats_per_draw,
                                use_xgb, use_cnn, calibrate=True, log_fn=print):
    n_pos_valid = None
    per_draw_results = []  # list of dict: {"draw": i, "auc": {name: auc}, "seed": seed}

    for draw in range(n_draws):
        seed = RANDOM_STATE + 1000 + draw  # asosiy tahlilda ishlatilgan seeddan farqli
        log_fn(f"  [Background sensitivity] {draw + 1}/{n_draws}-tanlov (seed={seed})...")

        bg_gdf = generate_background_points(aoi_gdf, positive_gdf, n_background,
                                             min_distance, random_state=seed)
        X_pos = sample_points_from_stack(positive_gdf, stack, transform)
        X_neg = sample_points_from_stack(bg_gdf, stack, transform)
        coords_pos = np.array([(g.x, g.y) for g in positive_gdf.geometry])
        coords_neg = np.array([(g.x, g.y) for g in bg_gdf.geometry])

        valid_pos = np.all(np.isfinite(X_pos), axis=1)
        valid_neg = np.all(np.isfinite(X_neg), axis=1)
        X_pos, X_neg = X_pos[valid_pos], X_neg[valid_neg]
        coords_pos, coords_neg = coords_pos[valid_pos], coords_neg[valid_neg]
        n_pos_valid = len(X_pos)

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
        coords = np.vstack([coords_pos, coords_neg])
        groups = assign_spatial_blocks(coords, block_size)
        if len(np.unique(groups)) < n_splits:
            log_fn(f"    Ogohlantirish: bu tanlovda bloklar yetarli emas, o'tkazib yuborildi.")
            continue

        oof, _, _ = run_cv_training(X, y, use_xgb, use_cnn, n_splits=n_splits,
                                     n_repeats=n_repeats_per_draw, groups=groups,
                                     cv_mode="spatial", calibrate=calibrate, log_fn=lambda m: None)
        roc_res, _ = compute_roc_and_stats(y, oof, log_fn=lambda m: None)
        auc_by_model = {name: res["auc"] for name, res in roc_res.items()}
        per_draw_results.append({"draw": draw, "seed": seed, "auc": auc_by_model})
        log_fn(f"    Ensemble AUC = {auc_by_model.get('Ensemble (soft-voting)', float('nan')):.3f}")

    if not per_draw_results:
        return None

    model_names = list(per_draw_results[0]["auc"].keys())
    summary = {}
    for name in model_names:
        vals = np.array([d["auc"][name] for d in per_draw_results if name in d["auc"]])
        summary[name] = {
            "mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "min": float(np.min(vals)), "max": float(np.max(vals)),
            "values": vals.tolist(),
        }
        log_fn(f"  [Background sensitivity] {name}: AUC {np.mean(vals):.3f} \u00b1 {np.std(vals):.3f} "
               f"(min={np.min(vals):.3f}, max={np.max(vals):.3f}, {len(vals)} tanlov)")

    return {"per_draw": per_draw_results, "summary": summary, "n_positive": n_pos_valid}


# ---------------------------------------------------------------------------
# SPATIAL AUTOCORRELATION / BLOCK CV
#
# Taqrizning asosiy tanqidi: RepeatedStratifiedKFold faqat class balansini
# saqlaydi, lekin bir-biriga yaqin nuqtalarni (va ularning deyarli bir xil
# raster qiymatlarini, spatial autocorrelation tufayli) bir vaqtning o'zida
# train va validation to'plamlariga tushirib yuborishi mumkin. Bu quyida:
#   1) predictor rasterlar asosida empirik semivariogram orqali
#      autocorrelation range (blok o'lchami) baholanadi,
#   2) shu o'lchamdagi kvadrat bloklar bo'yicha nuqtalar guruhlanadi,
#   3) guruhlar (bloklar), NUQTALAR EMAS, fold'larga taqsimlanadi -
#      shu bilan yaqin nuqtalar bir xil fold ichida qoladi.
# ---------------------------------------------------------------------------
def estimate_autocorrelation_range(stack, transform, band_index=0,
                                    n_lags=15, max_pairs=250000,
                                    random_state=RANDOM_STATE, log_fn=print):
    """
    Berilgan predictor band bo'yicha empirik semivariogramma hisoblab,
    "range" (masofa, undan keyin gamma(h) sill'ning ~95% iga yetadi) ni
    baholaydi. Natija - metrlarda taxminiy spatial autocorrelation radiusi,
    bu spatial block CV uchun blok o'lchamini tanlashda boshlang'ich nuqta
    bo'lib xizmat qiladi (yakuniy qarorni geolog/muallif tasdiqlashi kerak).
    """
    n_bands, H, W = stack.shape
    band = stack[band_index]
    rows, cols = np.where(np.isfinite(band))
    if len(rows) < 50:
        log_fn("  Variogram: yetarli valid piksel yo'q, standart blok o'lchami ishlatiladi.")
        return None

    rng = np.random.default_rng(random_state)
    max_pixels = int(np.sqrt(max_pairs)) 
    if len(rows) > max_pixels:
        sel = rng.choice(len(rows), size=max_pixels, replace=False)
        rows, cols = rows[sel], cols[sel]

    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs, ys = np.array(xs), np.array(ys)
    zs = band[rows, cols]

    coords = np.column_stack([xs, ys])
    dist = pdist(coords)
    zdiff2 = pdist(zs.reshape(-1, 1), metric="sqeuclidean")

    max_dist = np.percentile(dist, 60)  # uzoq masofalarni chetlab, lokal strukturaga e'tibor
    bins = np.linspace(0, max_dist, n_lags + 1)
    bin_idx = np.digitize(dist, bins) - 1

    lag_centers, gammas = [], []
    for b in range(n_lags):
        mask = bin_idx == b
        if mask.sum() < 30:
            continue
        gamma = 0.5 * np.mean(zdiff2[mask])
        lag_centers.append((bins[b] + bins[b + 1]) / 2)
        gammas.append(gamma)

    if len(gammas) < 3:
        log_fn("  Variogram: yetarli lag'lar hosil bo'lmadi, standart blok o'lchami ishlatiladi.")
        return None

    lag_centers, gammas = np.array(lag_centers), np.array(gammas)
    sill = np.percentile(gammas, 90)
    target = 0.95 * sill
    reached = np.where(gammas >= target)[0]
    est_range = lag_centers[reached[0]] if len(reached) > 0 else lag_centers[-1]

    log_fn(f"  Empirik semivariogram: taxminiy autocorrelation range \u2248 {est_range:,.0f} m "
           f"(band: {band_index}). Spatial block CV uchun boshlang'ich blok o'lchami sifatida "
           f"taklif qilinadi - yakuniy qiymatni tekshirib tasdiqlang.")
    return float(est_range)


def assign_spatial_blocks(coords, block_size):
    """
    Har bir nuqtani (x, y) kvadrat blok ID'siga biriktiradi:
    block_id = (floor(x/block_size), floor(y/block_size)).
    Bir xil blokdagi nuqtalar CV'da har doim bitta fold'da qoladi -
    shu bilan train/validation orasidagi spatial leakage oldini oladi.
    """
    bx = np.floor(coords[:, 0] / block_size).astype(np.int64)
    by = np.floor(coords[:, 1] / block_size).astype(np.int64)
    # unique integer group id har bir (bx,by) juftligi uchun
    block_ids = bx.astype(str) + "_" + by.astype(str)
    _, groups = np.unique(block_ids, return_inverse=True)
    return groups


def repeated_group_kfold_splits(groups, n_splits, n_repeats, random_state=RANDOM_STATE):
    """
    GroupKFold'ning "repeated" versiyasi: har bir repeat'da bloklar tartibi
    boshqacha tasodifiy aralashtiriladi, so'ng ketma-ket n_splits qismga
    bo'linadi. Natijada bitta blokning barcha nuqtalari doim bitta fold'da
    qoladi (spatial leakage yo'q), va bir nechta repeat orqali fold
    tarkibidagi tasodifiylikka bog'liq tebranish kamayadi.
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    if n_groups < n_splits:
        raise ValueError(
            f"Spatial bloklar soni ({n_groups}) k-fold soni ({n_splits}) dan kam. "
            "Blok o'lchamini kichraytiring yoki k-fold sonini kamaytiring."
        )
    rng = np.random.default_rng(random_state)
    for repeat in range(n_repeats):
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)
        fold_of_group = {g: i % n_splits for i, g in enumerate(shuffled)}
        fold_assignment = np.array([fold_of_group[g] for g in groups])
        for fold in range(n_splits):
            val_mask = fold_assignment == fold
            train_idx = np.where(~val_mask)[0]
            val_idx = np.where(val_mask)[0]
            yield train_idx, val_idx


# ---------------------------------------------------------------------------
# CNN (1D-Conv) qurish - tabular feature vektorini "spektr" sifatida qaraydi
# ---------------------------------------------------------------------------
def build_cnn_model(n_features):
    model = keras_models.Sequential([
        layers.Input(shape=(n_features, 1)),
        layers.Conv1D(16, 3, padding="same", activation="relu"),
        layers.Conv1D(32, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_cnn_fold(X_train, y_train, X_val, n_features, epochs=150):
    model = build_cnn_model(n_features)
    es = callbacks.EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)
    model.fit(
        X_train.reshape(-1, n_features, 1), y_train,
        epochs=epochs, batch_size=8, verbose=0, callbacks=[es]
    )
    val_proba = model.predict(X_val.reshape(-1, n_features, 1), verbose=0).ravel()
    return val_proba, model


# ---------------------------------------------------------------------------
# Cross-validation asosida har bir model uchun out-of-fold ehtimolliklarni
# hisoblash (20 ta musbat nuqta kabi kichik namunalar uchun to'g'ri yondashuv).
#
# n_repeats > 1 bo'lsa - Repeated Stratified K-Fold qo'llaniladi: butun CV
# jarayoni turli fold-bo'linishlar bilan bir necha marta takrorlanadi. Bu
# kichik namunada bitta tasodifiy fold-bo'linishga bog'liq tebranishni
# kamaytiradi va AUC ning o'rtacha ± standart og'ishini hisoblash imkonini
# beradi - bu bitta nuqtaviy qiymatdan (masalan, faqat "0.782") ancha
# ishonchli va ilmiy maqola uchun to'g'ri statistik baholash usuli.
# ---------------------------------------------------------------------------
def _make_cv_split_generator(X, y, groups, n_splits, n_repeats, cv_mode):
    """
    cv_mode="random"  -> RepeatedStratifiedKFold (eski, faqat class balansi)
    cv_mode="spatial" -> repeated_group_kfold_splits (spatial bloklar bo'yicha,
                          bir xil blokdagi nuqtalar hech qachon train/val
                          orasida bo'linmaydi)
    Har ikkalasi ham (train_idx, val_idx) juftliklarini generatsiya qiladi va
    split_num // n_splits orqali repeat raqamini aniqlash mumkin bo'lishi
    uchun ketma-ket (repeat-ichida-fold) tartibda qaytariladi.
    """
    if cv_mode == "spatial":
        if groups is None:
            raise ValueError("cv_mode='spatial' uchun 'groups' (spatial block id'lari) shart.")
        yield from repeated_group_kfold_splits(groups, n_splits, n_repeats, random_state=RANDOM_STATE)
    else:
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                        random_state=RANDOM_STATE)
        yield from rskf.split(X, y)


def _fit_predict_calibrated(base_estimator, X_train, y_train, X_val, calibrate,
                             calib_method="sigmoid", calib_cv=3):
    """
    calibrate=True bo'lsa, bazaviy modelni CalibratedClassifierCV bilan o'raydi.
    CalibratedClassifierCV o'zi ICHKI cv (calib_cv) orqali kalibratorni alohida
    (model o'qitilgan train qismidan chiqib ketmagan holda) o'qitadi - shuning
    uchun bu yerda ehtimollik "ikki marta ko'rilgan" ma'lumotda hisoblanmaydi.
    Kichik namunada (n_train ~ 60-90) sigmoid (Platt) usuli isotonic'dan
    barqarorroq, shuning uchun standart usul sifatida tanlangan.
    """
    if not calibrate:
        base_estimator.fit(X_train, y_train)
        return base_estimator.predict_proba(X_val)[:, 1], base_estimator

    # Har bir klassdan calib_cv marta kamida 1 ta namuna bo'lishi kerak
    min_class_count = min(np.bincount(y_train.astype(int)))
    eff_cv = max(2, min(calib_cv, int(min_class_count)))
    calibrated = CalibratedClassifierCV(base_estimator, method=calib_method, cv=eff_cv)
    calibrated.fit(X_train, y_train)
    return calibrated.predict_proba(X_val)[:, 1], calibrated


def run_cv_training(X, y, use_xgb, use_cnn, n_splits=5, n_repeats=1,
                     groups=None, cv_mode="random", calibrate=True,
                     compute_perm_importance=False, perm_importance_n_repeats=5,
                     log_fn=print):
    """
    cv_mode="random"  - eski xulq: RepeatedStratifiedKFold (faqat benchmark
                         sifatida saqlanadi, taqrizda tavsiya etilganidek).
    cv_mode="spatial" - asosiy natija: bloklar (guruhlar) darajasida bo'linadi,
                         shu bilan spatial autocorrelation orqali sodir
                         bo'ladigan train/validation leakage oldini oladi.
    calibrate=True    - har bir model ehtimolligi probability calibration
                         (Platt scaling) orqali 0-1 oralig'ida solishtiriladigan
                         qilinadi, shundan keyingina ensemble o'rtachalanadi.
                         Bu taqrizning "calibration SVM/RF/XGBoost не показана"
                         degan tanqidiga javob beradi.
    compute_perm_importance=True - MDI/gain o'rniga, HAR BIR spatial fold uchun
                         validation (OUT-OF-FOLD) qismida permutation importance
                         hisoblanadi va keyin fold'lar bo'yicha mean ± std
                         qaytariladi. Bu taqrizning "MDI importance RF и gain
                         importance XGBoost смещены... используйте out-of-fold
                         permutation/SHAP" talabiga to'g'ridan-to'g'ri javob
                         beradi - chunki MDI/gain training data'da hisoblanib,
                         correlated predictorlarda ishonchsiz bo'lishi mumkin,
                         permutation esa har doim MODEL KO'RMAGAN (val) qismda
                         hisoblanadi.
    """
    n_features = X.shape[1]

    model_names = ["RandomForest", "SVM"]
    if use_xgb and XGBOOST_AVAILABLE:
        model_names.append("XGBoost")
    if use_cnn and TF_AVAILABLE:
        model_names.append("CNN")

    # Har bir model uchun har bir repeat'dagi out-of-fold bashoratlarini saqlaymiz
    oof_repeats = {name: [] for name in model_names}
    # Har bir fold'dagi out-of-fold permutation importance (faqat 1-repeat, RF/XGB uchun)
    perm_importance_folds = {name: [] for name in ["RandomForest", "XGBoost"]}
    # Har bir nuqta qaysi fold'da validation'da bo'lganini kuzatish uchun
    # (fold xaritasini chizish va reviewerga ko'rsatish uchun, faqat 1-repeat)
    fold_map_first_repeat = np.full(len(y), -1, dtype=int)

    total_folds = n_splits * n_repeats
    fold_num = 0
    current_repeat = -1
    oof_this_repeat = None

    split_gen = _make_cv_split_generator(X, y, groups, n_splits, n_repeats, cv_mode)

    for split_num, (train_idx, val_idx) in enumerate(split_gen):
        repeat_id = split_num // n_splits
        fold_in_repeat = split_num % n_splits

        if repeat_id != current_repeat:
            # yangi repeat boshlanmoqda - avvalgisini yakunlab, yangisini ochamiz
            if oof_this_repeat is not None:
                for name in model_names:
                    oof_repeats[name].append(oof_this_repeat[name])
            current_repeat = repeat_id
            oof_this_repeat = {name: np.full(len(y), np.nan) for name in model_names}
            if n_repeats > 1:
                log_fn(f"--- [{cv_mode}] Takror {repeat_id + 1}/{n_repeats} ---")

        if repeat_id == 0:
            fold_map_first_repeat[val_idx] = fold_in_repeat

        fold_num += 1
        log_fn(f"  [{cv_mode}] Fold {fold_in_repeat + 1}/{n_splits} o'qitilmoqda... "
               f"(jami {fold_num}/{total_folds})")

        # Har bir fold uchun seedni qayta mahkamlaymiz - shunda CNN vazn
        # boshlang'ich qiymatlari HAM to'liq reproduktiv bo'ladi
        set_global_seed(RANDOM_STATE + split_num)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        rf_base = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
        )
        rf_proba, rf_fitted = _fit_predict_calibrated(rf_base, X_train, y_train, X_val, calibrate)
        oof_this_repeat["RandomForest"][val_idx] = rf_proba

        svm_base = SVC(kernel="rbf", C=1.0, gamma="scale",
                        probability=not calibrate,  # calibrate=True bo'lsa, CalibratedClassifierCV
                        class_weight="balanced", random_state=RANDOM_STATE)  # o'zi kalibrlaydi
        svm_proba, _ = _fit_predict_calibrated(svm_base, X_train_s, y_train, X_val_s, calibrate)
        oof_this_repeat["SVM"][val_idx] = svm_proba

        xgb_fitted = None
        if use_xgb and XGBOOST_AVAILABLE:
            pos = max(y_train.sum(), 1)
            neg = max(len(y_train) - y_train.sum(), 1)
            xgb_base = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=neg / pos, eval_metric="logloss",
                random_state=RANDOM_STATE, n_jobs=1
            )
            xgb_proba, xgb_fitted = _fit_predict_calibrated(xgb_base, X_train, y_train, X_val, calibrate)
            oof_this_repeat["XGBoost"][val_idx] = xgb_proba

        # --- Out-of-fold permutation importance (faqat 1-repeat, hisoblash
        # vaqtini nazorat qilish uchun; bu allaqachon N marta takrorlangan
        # CV'ning bir qismi bo'lgani uchun fold'lar bo'yicha statistika
        # baribir olinadi)
        if compute_perm_importance and repeat_id == 0 and len(val_idx) >= 5:
            try:
                pr = permutation_importance(
                    rf_fitted, X_val, y_val, scoring="roc_auc",
                    n_repeats=perm_importance_n_repeats, random_state=RANDOM_STATE, n_jobs=1
                )
                perm_importance_folds["RandomForest"].append(pr.importances_mean)
            except Exception as e:
                log_fn(f"    Ogohlantirish: RF permutation importance hisoblanmadi ({e})")
            if xgb_fitted is not None:
                try:
                    pr = permutation_importance(
                        xgb_fitted, X_val, y_val, scoring="roc_auc",
                        n_repeats=perm_importance_n_repeats, random_state=RANDOM_STATE, n_jobs=1
                    )
                    perm_importance_folds["XGBoost"].append(pr.importances_mean)
                except Exception as e:
                    log_fn(f"    Ogohlantirish: XGBoost permutation importance hisoblanmadi ({e})")

        if use_cnn and TF_AVAILABLE:
            # CNN (Keras) sklearn API'ga mos emas, shuning uchun CalibratedClassifierCV
            # bilan avtomatik o'ralmaydi; sigmoid chiqishi o'z-o'zidan taxminiy
            # kalibrlangan hisoblanadi, lekin RF/SVM/XGB bilan bir xil kafolat yo'q -
            # bu cheklov natijalar bo'limida alohida qayd etilishi kerak.
            cnn_proba, _ = train_cnn_fold(X_train_s, y_train, X_val_s, n_features)
            oof_this_repeat["CNN"][val_idx] = cnn_proba

    # oxirgi repeatni yakunlaymiz
    if oof_this_repeat is not None:
        for name in model_names:
            oof_repeats[name].append(oof_this_repeat[name])

    perm_importance_summary = {}
    if compute_perm_importance:
        for name, fold_list in perm_importance_folds.items():
            if len(fold_list) == 0:
                continue
            arr = np.vstack(fold_list)  # (n_folds, n_features)
            perm_importance_summary[name] = {
                "mean": np.mean(arr, axis=0),
                "std": np.std(arr, axis=0),
                "n_folds": arr.shape[0],
            }

    return oof_repeats, fold_map_first_repeat, perm_importance_summary


def _extra_metrics(y_true, proba, threshold=0.5):
    """PR-AUC, balanced accuracy, F1, Brier score - taqrizda so'ralgan qo'shimcha
    metrikalar (faqat ROC-AUC yetarli emas, ayniqsa kichik/imbalanced sinf uchun)."""
    pr_auc = average_precision_score(y_true, proba)
    y_pred = (proba >= threshold).astype(int)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, proba)
    return {"pr_auc": pr_auc, "balanced_accuracy": bal_acc, "f1": f1, "brier": brier}


def _mean_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))
    n = len(values)
    half_width = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return mean, std, (mean - half_width, mean + half_width)


def compute_roc_and_stats(y_true, oof_repeats, log_fn=print):
    """
    Har bir model uchun:
      - repeat-lar bo'yicha o'rtachalashtirilgan ehtimollik -> yagona ROC egri chizig'i
      - har bir repeat uchun alohida AUC -> mean ± std (barqarorlik ko'rsatkichi)
      - PR-AUC, balanced accuracy, F1, Brier score - har biri ham repeat bo'yicha
        mean ± std va taxminiy 95% ishonch oralig'i bilan (taqrizning "добавить
        PR-AUC, sensitivity/specificity, balanced accuracy, F1, Brier score/
        calibration curve, confidence intervals" talabiga javob beradi)
    Ansambl xuddi shu tarzda, lekin har bir repeat ichida barcha modellar
    o'rtachalashtirilib, keyin repeat-lar bo'yicha statistikasi hisoblanadi.
    """
    results = {}
    n_repeats = len(next(iter(oof_repeats.values())))
    model_names = list(oof_repeats.keys())

    ensemble_per_repeat = []

    for name in model_names:
        repeat_arrays = oof_repeats[name]  # list of arrays (len(y),) - bittasi har repeat uchun
        per_repeat_auc = []
        per_repeat_extra = {"pr_auc": [], "balanced_accuracy": [], "f1": [], "brier": []}
        for r_proba in repeat_arrays:
            fpr_r, tpr_r, _ = roc_curve(y_true, r_proba)
            per_repeat_auc.append(auc(fpr_r, tpr_r))
            extra = _extra_metrics(y_true, r_proba)
            for k, v in extra.items():
                per_repeat_extra[k].append(v)

        mean_proba = np.mean(np.vstack(repeat_arrays), axis=0)
        fpr, tpr, _ = roc_curve(y_true, mean_proba)
        auc_mean, auc_std, auc_ci = _mean_ci95(per_repeat_auc)
        results[name] = {
            "fpr": fpr, "tpr": tpr,
            "auc": auc_mean, "auc_std": auc_std, "auc_ci95": auc_ci,
            "auc_single": float(auc(fpr, tpr)),  # o'rtachalashtirilgan proba'dan hisoblangan AUC
            "mean_proba": mean_proba,
        }
        for k in per_repeat_extra:
            m, s, ci = _mean_ci95(per_repeat_extra[k])
            results[name][k] = m
            results[name][k + "_std"] = s
            results[name][k + "_ci95"] = ci
        log_fn(f"  {name}: AUC={auc_mean:.3f}\u00b1{auc_std:.3f} (95%CI {auc_ci[0]:.3f}-{auc_ci[1]:.3f}) "
               f"| PR-AUC={results[name]['pr_auc']:.3f} | BalAcc={results[name]['balanced_accuracy']:.3f} "
               f"| F1={results[name]['f1']:.3f} | Brier={results[name]['brier']:.3f}")

    # --- Ansambl: har bir repeat ichida barcha modellarni o'rtachalashtiramiz
    for r in range(n_repeats):
        r_stack = np.vstack([oof_repeats[name][r] for name in model_names])
        ensemble_per_repeat.append(np.mean(r_stack, axis=0))

    ens_auc_per_repeat = []
    ens_extra_per_repeat = {"pr_auc": [], "balanced_accuracy": [], "f1": [], "brier": []}
    for r_proba in ensemble_per_repeat:
        fpr_r, tpr_r, _ = roc_curve(y_true, r_proba)
        ens_auc_per_repeat.append(auc(fpr_r, tpr_r))
        extra = _extra_metrics(y_true, r_proba)
        for k, v in extra.items():
            ens_extra_per_repeat[k].append(v)

    ensemble_mean_proba = np.mean(np.vstack(ensemble_per_repeat), axis=0)
    fpr, tpr, _ = roc_curve(y_true, ensemble_mean_proba)
    auc_mean, auc_std, auc_ci = _mean_ci95(ens_auc_per_repeat)
    results["Ensemble (soft-voting)"] = {
        "fpr": fpr, "tpr": tpr,
        "auc": auc_mean, "auc_std": auc_std, "auc_ci95": auc_ci,
        "auc_single": float(auc(fpr, tpr)),
        "mean_proba": ensemble_mean_proba,
    }
    for k in ens_extra_per_repeat:
        m, s, ci = _mean_ci95(ens_extra_per_repeat[k])
        results["Ensemble (soft-voting)"][k] = m
        results["Ensemble (soft-voting)"][k + "_std"] = s
        results["Ensemble (soft-voting)"][k + "_ci95"] = ci
    log_fn(f"  Ensemble (soft-voting): AUC={auc_mean:.3f}\u00b1{auc_std:.3f} "
           f"(95%CI {auc_ci[0]:.3f}-{auc_ci[1]:.3f}) | PR-AUC={results['Ensemble (soft-voting)']['pr_auc']:.3f} "
           f"| BalAcc={results['Ensemble (soft-voting)']['balanced_accuracy']:.3f} "
           f"| F1={results['Ensemble (soft-voting)']['f1']:.3f} "
           f"| Brier={results['Ensemble (soft-voting)']['brier']:.3f}")

    return results, ensemble_mean_proba


# ---------------------------------------------------------------------------
# Yakuniy modellarni TO'LIQ ma'lumotda qayta o'qitish (butun maydon uchun
# prognoz xarita yaratishdan oldin) va butun raster stack ustida bashorat
# ---------------------------------------------------------------------------
def fit_final_models(X, y, use_xgb, use_cnn, calibrate=True, log_fn=print):
    """
    MUHIM TUZATISH: avvalgi versiyada bu yerdagi RF (n_estimators=800) va XGBoost
    (n_estimators=100, max_depth=6, learning_rate=0.1) giperparametrlari
    run_cv_training() dagi CV bosqichi giperparametrlaridan (RF=500, XGB
    n_estimators=300/max_depth=4/learning_rate=0.05) FARQ QILAR EDI - bu esa
    maqola matnidagi "Identical hyperparameters were used for both the
    cross-validation stage and the final full-dataset model" degan aniq
    da'voga zid edi. Endi ikkalasida ham AYNAN BIR XIL giperparametrlar
    ishlatiladi - reproducibility da'vosi endi kodga mos keladi.
    """
    set_global_seed(RANDOM_STATE)
    n_features = X.shape[1]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    final_models = {}

    rf_base = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
    )
    _, rf_fitted = _fit_predict_calibrated(rf_base, X, y, X[:1], calibrate)
    final_models["RandomForest"] = ("raw", rf_fitted)

    svm_base = SVC(kernel="rbf", C=1.0, gamma="scale", probability=not calibrate,
                    class_weight="balanced", random_state=RANDOM_STATE)
    _, svm_fitted = _fit_predict_calibrated(svm_base, X_s, y, X_s[:1], calibrate)
    final_models["SVM"] = ("scaled", svm_fitted)

    if use_xgb and XGBOOST_AVAILABLE:
        pos = max(y.sum(), 1)
        neg = max(len(y) - y.sum(), 1)
        xgb_base = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg / pos, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=1
        )
        _, xgb_fitted = _fit_predict_calibrated(xgb_base, X, y, X[:1], calibrate)
        final_models["XGBoost"] = ("raw", xgb_fitted)

    if use_cnn and TF_AVAILABLE:
        cnn = build_cnn_model(n_features)
        es = callbacks.EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)
        cnn.fit(X_s.reshape(-1, n_features, 1), y, epochs=200, batch_size=8,
                verbose=0, callbacks=[es])
        final_models["CNN"] = ("scaled_cnn", cnn)

    return final_models, scaler


def compute_shap_summary(final_models, X, band_names, max_background=50, log_fn=print):
    """
    RF/XGBoost yakuniy modellari uchun SHAP qiymatlarini hisoblaydi (agar
    `shap` kutubxonasi o'rnatilgan bo'lsa). Bu MDI/gain'ga muqobil,
    interpretatsiya jihatidan ancha ishonchli usul - taqrizning "используйте
    out-of-fold permutation/SHAP" tavsiyasining ikkinchi yarmi.

    calibrate=True bo'lsa, final model CalibratedClassifierCV bilan o'ralgan
    bo'ladi; SHAP TreeExplainer daraxt asosidagi model kutgani uchun,
    kalibratsiya ichidagi birinchi ichki-fold bazaviy modelini (taxminiy
    vakil sifatida) chiqarib olishga harakat qilamiz. Bu yondashuv aniq emas
    (faqat bitta ichki-fold modelini aks ettiradi) - shuning uchun asosiy
    interpretatsiya out-of-fold permutation importance'ga tayanishi kerak,
    SHAP faqat QO'SHIMCHA tekshiruv sifatida taqdim etiladi.
    """
    if not SHAP_AVAILABLE:
        log_fn("  SHAP kutubxonasi o'rnatilmagan ('pip install shap') - SHAP tahlili o'tkazib yuborildi.")
        return None

    def _extract_tree_model(wrapped):
        if hasattr(wrapped, "feature_importances_"):
            return wrapped
        try:  # CalibratedClassifierCV -> ichki fold bazaviy modeli
            cc = wrapped.calibrated_classifiers_[0]
            base = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if base is not None and hasattr(base, "feature_importances_"):
                return base
        except Exception:
            pass
        return None

    shap_out = {}
    bg_idx = np.random.default_rng(RANDOM_STATE).choice(
        len(X), size=min(max_background, len(X)), replace=False
    )
    X_bg = X[bg_idx]

    for name in ("RandomForest", "XGBoost"):
        if name not in final_models:
            continue
        tree_model = _extract_tree_model(final_models[name][1])
        if tree_model is None:
            log_fn(f"  OGOHLANTIRISH: {name} uchun SHAP - kalibratsiya ichidan bazaviy "
                   f"daraxt modelini chiqarib olib bo'lmadi, o'tkazib yuborildi.")
            continue
        try:
            explainer = shap.TreeExplainer(tree_model)
            sv = explainer.shap_values(X_bg)
            if isinstance(sv, list):  # ba'zi versiyalarda [class0, class1] ro'yxati qaytadi
                sv = sv[1] if len(sv) > 1 else sv[0]
            mean_abs = np.mean(np.abs(sv), axis=0)
            shap_out[name] = {"mean_abs_shap": mean_abs, "shap_values": sv, "X_background": X_bg}
            log_fn(f"  SHAP hisoblandi: {name} ({len(X_bg)} ta background nuqta).")
        except Exception as e:
            log_fn(f"  OGOHLANTIRISH: {name} uchun SHAP hisoblanmadi ({e}).")

    return shap_out if shap_out else None


def predict_probability_map(stack, band_names, final_models, scaler, log_fn=print, batch_size=200000):
    """
    Butun raster stack ustida har bir piksel uchun ehtimollikni bashorat qiladi.
    NaN (nodata) piksellar chetlab o'tiladi. Natija: dict {model_name: 2D array}, ensemble.
    """
    n_bands, H, W = stack.shape
    flat = stack.reshape(n_bands, -1).T  # (n_pixels, n_bands)
    # isnan emas, isfinite orqali tekshiramiz - shu bilan inf/-inf qiymatlar ham
    # (masalan, geofizik filtr gridlarining chekka effektlari) chetlab o'tiladi
    valid_mask = np.all(np.isfinite(flat), axis=1)
    valid_idx = np.where(valid_mask)[0]

    log_fn(f"Bashorat qilinmoqda: {len(valid_idx):,} piksel (jami {flat.shape[0]:,} dan)...")

    prob_maps = {}
    all_probs = []

    for name, (mode, model) in final_models.items():
        proba_flat = np.full(flat.shape[0], np.nan, dtype=np.float32)
        for start in range(0, len(valid_idx), batch_size):
            batch_idx = valid_idx[start:start + batch_size]
            X_batch = flat[batch_idx]
            if mode == "raw":
                p = model.predict_proba(X_batch)[:, 1]
            elif mode == "scaled":
                X_batch_s = scaler.transform(X_batch)
                p = model.predict_proba(X_batch_s)[:, 1]
            elif mode == "scaled_cnn":
                X_batch_s = scaler.transform(X_batch)
                p = model.predict(
                    X_batch_s.reshape(-1, n_bands, 1), verbose=0
                ).ravel()
            proba_flat[batch_idx] = p
        prob_maps[name] = proba_flat.reshape(H, W)
        all_probs.append(proba_flat)
        log_fn(f"  {name} bashorati tayyor.")

    ensemble_flat = np.nanmean(np.vstack(all_probs), axis=0)
    prob_maps["Ensemble (soft-voting)"] = ensemble_flat.reshape(H, W)

    return prob_maps


def save_probability_maps(prob_maps, ref_profile, out_dir, log_fn=print):
    os.makedirs(out_dir, exist_ok=True)
    profile = ref_profile.copy()
    profile.update(count=1, dtype="float32", nodata=-9999.0)
    saved_paths = []
    for name, arr in prob_maps.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        out_path = os.path.join(out_dir, f"prognoz_{safe_name}.tif")
        out_arr = np.where(np.isnan(arr), -9999.0, arr).astype("float32")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_arr, 1)
        saved_paths.append(out_path)
        log_fn(f"Saqlandi: {out_path}")
    return saved_paths


# ===========================================================================
# QThread ISHCHILARI - GUI qotib qolmasligi uchun og'ir hisoblarni fon
# jarayonida bajaradi
# ===========================================================================
class TrainingWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, tiff_folder, points_folder, aoi_folder,
                 n_background, min_distance, n_splits, use_xgb, use_cnn,
                 assume_crs_if_missing=True, n_repeats=1, block_size=0,
                 bg_sensitivity_enabled=False, bg_sensitivity_draws=5,
                 bg_sensitivity_repeats=3, calibrate=True):
        super().__init__()
        self.tiff_folder = tiff_folder
        self.points_folder = points_folder
        self.aoi_folder = aoi_folder
        self.n_background = n_background
        self.min_distance = min_distance
        self.n_splits = n_splits
        self.use_xgb = use_xgb
        self.use_cnn = use_cnn
        self.assume_crs_if_missing = assume_crs_if_missing
        self.n_repeats = n_repeats
        self.block_size = block_size  # 0/None => avtomatik (variogram) baholash
        self.bg_sensitivity_enabled = bg_sensitivity_enabled
        self.bg_sensitivity_draws = bg_sensitivity_draws
        self.bg_sensitivity_repeats = bg_sensitivity_repeats
        self.calibrate = calibrate

    def log(self, msg):
        self.log_signal.emit(str(msg))

    def run(self):
        try:
            set_global_seed(RANDOM_STATE)
            self.progress_signal.emit(5)
            tiff_paths = find_tiff_files(self.tiff_folder)
            if not tiff_paths:
                raise ValueError("TIFF papkasida .tif fayllar topilmadi.")
            self.log(f"{len(tiff_paths)} ta TIFF qatlam topildi.")

            points_shp = find_shapefile(self.points_folder)
            aoi_shp = find_shapefile(self.aoi_folder)
            if not points_shp:
                raise ValueError("Target nuqtalar papkasida .shp fayl topilmadi.")
            if not aoi_shp:
                raise ValueError("AOI (maydon konturi) papkasida .shp fayl topilmadi.")

            self.log("Raster qatlamlar yuklanmoqda va moslashtirilmoqda (EPSG:28411)...")
            stack, ref_profile, band_names, transform, tech_metadata = load_and_align_rasters(
                tiff_paths, log_fn=self.log,
                assume_crs_if_missing=self.assume_crs_if_missing,
                fallback_epsg=TARGET_EPSG,
            )
            self.progress_signal.emit(30)

            manual_metadata, meta_csv_path = load_or_create_manual_metadata(
                self.tiff_folder, band_names, log_fn=self.log
            )
            data_dictionary = build_data_dictionary(band_names, tech_metadata, manual_metadata)
            n_incomplete = sum(
                1 for name in band_names
                if not str(manual_metadata.get(name, {}).get("source_owner", "")).strip()
            )
            if n_incomplete > 0:
                self.log(f"  OGOHLANTIRISH: {n_incomplete}/{len(band_names)} qatlam uchun "
                         f"qo'lda metadata (manba, sana, transformatsiya) to'ldirilmagan. "
                         f"'{meta_csv_path}' faylini to'ldiring - maqolaga chiqarishdan oldin "
                         f"data dictionary to'liq bo'lishi kerak.")
            self.progress_signal.emit(35)

            self.log(f"Target nuqtalar yuklanmoqda: {points_shp}")
            positive_gdf = gpd.read_file(points_shp)
            if positive_gdf.crs is None:
                if self.assume_crs_if_missing:
                    self.log(f"  OGOHLANTIRISH: target nuqtalar shapefile CRS ga ega emas. "
                             f"EPSG:{TARGET_EPSG} deb qabul qilinmoqda.")
                    positive_gdf = positive_gdf.set_crs(epsg=TARGET_EPSG)
                else:
                    raise ValueError("Target nuqtalar shapefile CRS ga ega emas.")
            positive_gdf = positive_gdf.to_crs(epsg=TARGET_EPSG)
            positive_gdf = positive_gdf[positive_gdf.geometry.type == "Point"]
            self.log(f"  {len(positive_gdf)} ta musbat (ma'lum kon/namoyon) nuqta.")

            self.log(f"AOI kontur yuklanmoqda: {aoi_shp}")
            aoi_gdf = gpd.read_file(aoi_shp)
            if aoi_gdf.crs is None:
                if self.assume_crs_if_missing:
                    self.log(f"  OGOHLANTIRISH: AOI shapefile CRS ga ega emas. "
                             f"EPSG:{TARGET_EPSG} deb qabul qilinmoqda.")
                    aoi_gdf = aoi_gdf.set_crs(epsg=TARGET_EPSG)
                else:
                    raise ValueError("AOI shapefile CRS ga ega emas.")
            aoi_gdf = aoi_gdf.to_crs(epsg=TARGET_EPSG)

            self.log(f"Fon (bepusht) nuqtalar generatsiya qilinmoqda: {self.n_background} ta, "
                     f"min. masofa = {self.min_distance} m")
            background_gdf = generate_background_points(
                aoi_gdf, positive_gdf, self.n_background, self.min_distance
            )
            self.log(f"  {len(background_gdf)} ta fon nuqtasi yaratildi.")
            self.progress_signal.emit(45)

            X_pos = sample_points_from_stack(positive_gdf, stack, transform)
            X_neg = sample_points_from_stack(background_gdf, stack, transform)
            coords_pos_all = np.array([(geom.x, geom.y) for geom in positive_gdf.geometry])
            coords_neg_all = np.array([(geom.x, geom.y) for geom in background_gdf.geometry])

            valid_pos = np.all(np.isfinite(X_pos), axis=1)
            valid_neg = np.all(np.isfinite(X_neg), axis=1)
            n_dropped = (~valid_pos).sum() + (~valid_neg).sum()
            if n_dropped > 0:
                self.log(f"  Ogohlantirish: {n_dropped} ta nuqta raster chegarasidan "
                         f"tashqarida/nodata, tashlab yuborildi.")
            X_pos, X_neg = X_pos[valid_pos], X_neg[valid_neg]
            coords_pos, coords_neg = coords_pos_all[valid_pos], coords_neg_all[valid_neg]

            if len(X_pos) < 10:
                raise ValueError(
                    f"Musbat nuqtalar soni juda kam ({len(X_pos)}). "
                    "Modelni ishonchli o'qitish uchun kamida ~10-15 nuqta kerak."
                )

            X = np.vstack([X_pos, X_neg])
            y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
            coords = np.vstack([coords_pos, coords_neg])
            self.log(f"Yakuniy dataset: {len(X_pos)} musbat + {len(X_neg)} fon = {len(X)} nuqta, "
                     f"{X.shape[1]} feature.")

            if self.use_cnn and not TF_AVAILABLE:
                self.log("OGOHLANTIRISH: TensorFlow o'rnatilmagan, CNN o'tkazib yuboriladi.")
            if self.use_xgb and not XGBOOST_AVAILABLE:
                self.log("OGOHLANTIRISH: XGBoost o'rnatilmagan, XGBoost o'tkazib yuboriladi.")

            # -----------------------------------------------------------
            # SPATIAL BLOK O'LCHAMINI ANIQLASH (variogram yoki foydalanuvchi
            # tomonidan qo'lda kiritilgan qiymat)
            # -----------------------------------------------------------
            if self.block_size and self.block_size > 0:
                block_size = float(self.block_size)
                self.log(f"Spatial blok o'lchami (qo'lda belgilangan): {block_size:,.0f} m")
            else:
                self.log("Spatial blok o'lchami avtomatik baholanmoqda (empirik semivariogram)...")
                est = estimate_autocorrelation_range(stack, transform, band_index=0, log_fn=self.log)
                block_size = est if est else 1000.0
                self.log(f"  Ishlatiladigan blok o'lchami: {block_size:,.0f} m")

            groups = assign_spatial_blocks(coords, block_size)
            n_blocks = len(np.unique(groups))
            self.log(f"Spatial bloklar soni: {n_blocks} ({len(y)} nuqta uchun)")
            attempts = 0
            while n_blocks < self.n_splits and block_size > 10 and attempts < 20:
                block_size /= 2.0
                groups = assign_spatial_blocks(coords, block_size)
                n_blocks = len(np.unique(groups))
                attempts += 1
            if n_blocks < self.n_splits:
                raise ValueError(
                    f"Spatial bloklar soni ({n_blocks}) k-fold soniga ({self.n_splits}) "
                    "yetmayapti hatto blok o'lchami kichraytirilgandan keyin ham. "
                    "K-fold sonini kamaytiring yoki blok o'lchamini qo'lda kiriting."
                )
            if attempts > 0:
                self.log(f"  Bloklar yetarli bo'lishi uchun blok o'lchami avtomatik "
                         f"kichraytirildi: {block_size:,.0f} m ({n_blocks} blok)")

            self.progress_signal.emit(50)

            # -----------------------------------------------------------
            # 1) RANDOM CV - faqat qo'shimcha benchmark sifatida (eski usul)
            # 2) SPATIAL BLOCK CV - asosiy, taqrizda talab qilingan natija
            # -----------------------------------------------------------
            self.log(f"\n=== RANDOM Cross-Validation (benchmark, {self.n_splits}-fold x "
                     f"{self.n_repeats} takror) ===")
            oof_random, foldmap_random, _ = run_cv_training(
                X, y, self.use_xgb, self.use_cnn,
                n_splits=self.n_splits, n_repeats=self.n_repeats,
                cv_mode="random", calibrate=self.calibrate, log_fn=self.log
            )
            roc_results_random, _ = compute_roc_and_stats(y, oof_random, log_fn=self.log)
            self.progress_signal.emit(65)

            self.log(f"\n=== SPATIAL BLOCK Cross-Validation (asosiy natija, "
                     f"{self.n_splits}-fold x {self.n_repeats} takror, blok={block_size:,.0f} m) ===")
            oof_spatial, foldmap_spatial, perm_importance = run_cv_training(
                X, y, self.use_xgb, self.use_cnn,
                n_splits=self.n_splits, n_repeats=self.n_repeats,
                groups=groups, cv_mode="spatial", calibrate=self.calibrate,
                compute_perm_importance=True, perm_importance_n_repeats=5,
                log_fn=self.log
            )
            roc_results, ensemble_oof = compute_roc_and_stats(y, oof_spatial, log_fn=self.log)

            self.log("\n--- Random vs Spatial-block AUC solishtiruvi ---")
            for name in roc_results.keys():
                r_auc = roc_results_random.get(name, {}).get("auc", float("nan"))
                s_auc = roc_results.get(name, {}).get("auc", float("nan"))
                gap = r_auc - s_auc
                self.log(f"  {name}: random AUC={r_auc:.3f}  |  spatial AUC={s_auc:.3f}  "
                         f"|  optimizm (leakage taxmini) = {gap:+.3f}")

            self.progress_signal.emit(80)

            bg_sensitivity = None
            if self.bg_sensitivity_enabled:
                self.log(f"\n=== Background (fon) tanloviga sezgirlik tahlili "
                         f"({self.bg_sensitivity_draws} ta mustaqil tanlov) ===")
                bg_sensitivity = run_background_sensitivity(
                    aoi_gdf, positive_gdf, stack, transform,
                    n_background=self.n_background, min_distance=self.min_distance,
                    block_size=block_size, n_draws=self.bg_sensitivity_draws,
                    n_splits=self.n_splits, n_repeats_per_draw=self.bg_sensitivity_repeats,
                    use_xgb=self.use_xgb, use_cnn=self.use_cnn,
                    calibrate=self.calibrate, log_fn=self.log,
                )
            self.progress_signal.emit(85)

            self.log("Yakuniy modellar to'liq datasetda qayta o'qitilmoqda...")
            final_models, scaler = fit_final_models(
                X, y, self.use_xgb, self.use_cnn, calibrate=self.calibrate, log_fn=self.log
            )

            # ---------------------------------------------------------------
            # FEATURE IMPORTANCE: MDI/gain (training-data-based, correlated
            # predictorlarda noto'g'ri bo'lishi mumkin) o'rniga birinchi
            # navbatda OUT-OF-FOLD PERMUTATION IMPORTANCE ishlatiladi (yuqorida
            # spatial CV davomida hisoblangan - hech qachon model ko'rgan
            # ma'lumotda emas). SHAP (agar kutubxona mavjud bo'lsa) qo'shimcha
            # tekshiruv sifatida yakuniy modeldan hisoblanadi.
            # ---------------------------------------------------------------
            rf_importance = perm_importance.get("RandomForest", {}).get("mean")
            rf_importance_std = perm_importance.get("RandomForest", {}).get("std")
            xgb_importance = perm_importance.get("XGBoost", {}).get("mean")
            xgb_importance_std = perm_importance.get("XGBoost", {}).get("std")
            importance_method = "out-of-fold permutation (spatial CV)"

            if rf_importance is None and "RandomForest" in final_models:
                # zaxira variant: agar biror sababdan permutation hisoblanmagan
                # bo'lsa, faqat OGOHLANTIRISH bilan MDI'ga qaytiladi (SHAP/permutation
                # tavsiya etiladi, lekin dastur sinmasligi kerak)
                try:
                    base = final_models["RandomForest"][1]
                    base = getattr(base, "estimator", base)
                    rf_importance = base.feature_importances_
                    self.log("  OGOHLANTIRISH: RF uchun permutation importance mavjud emas, "
                             "zaxira sifatida MDI ishlatildi (ehtiyot bo'ling - correlated "
                             "predictorlarda noto'g'ri bo'lishi mumkin).")
                    importance_method = "MDI (zaxira, permutation muvaffaqiyatsiz)"
                except Exception:
                    pass

            if xgb_importance is None and "XGBoost" in final_models:
                try:
                    base = final_models["XGBoost"][1]
                    base = getattr(base, "estimator", base)
                    xgb_importance = base.feature_importances_
                    self.log("  OGOHLANTIRISH: XGBoost uchun permutation importance mavjud emas, "
                             "zaxira sifatida gain-based importance ishlatildi.")
                except Exception:
                    pass

            shap_values = compute_shap_summary(final_models, X, band_names, log_fn=self.log)

            self.progress_signal.emit(95)

            result = {
                "stack": stack,
                "ref_profile": ref_profile,
                "band_names": band_names,
                "transform": transform,
                "roc_results": roc_results,
                "final_models": final_models,
                "scaler": scaler,
                "rf_importance": rf_importance,
                "xgb_importance": xgb_importance,
                "X": X, "y": y,
                "n_positive": len(X_pos), "n_background": len(X_neg),
                "roc_results_random": roc_results_random,
                "coords": coords,
                "groups": groups,
                "block_size": block_size,
                "foldmap_spatial": foldmap_spatial,
                "foldmap_random": foldmap_random,
                "bg_sensitivity": bg_sensitivity,
                "data_dictionary": data_dictionary,
                "metadata_csv_path": meta_csv_path,
                "rf_importance_std": rf_importance_std,
                "xgb_importance_std": xgb_importance_std,
                "importance_method": importance_method,
                "shap_values": shap_values,
            }
            self.progress_signal.emit(100)
            self.finished_signal.emit(result)

        except Exception as e:
            self.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")


class PredictionWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, stack, band_names, final_models, scaler, ref_profile, out_dir,
                 data_dictionary=None):
        super().__init__()
        self.stack = stack
        self.band_names = band_names
        self.final_models = final_models
        self.scaler = scaler
        self.ref_profile = ref_profile
        self.out_dir = out_dir
        self.data_dictionary = data_dictionary

    def log(self, msg):
        self.log_signal.emit(str(msg))

    def run(self):
        try:
            prob_maps = predict_probability_map(
                self.stack, self.band_names, self.final_models, self.scaler, log_fn=self.log
            )
            saved_paths = save_probability_maps(
                prob_maps, self.ref_profile, self.out_dir, log_fn=self.log
            )
            if self.data_dictionary is not None:
                dict_path = save_data_dictionary(self.data_dictionary, self.out_dir, log_fn=self.log)
                saved_paths.append(dict_path)
            self.finished_signal.emit({"prob_maps": prob_maps, "saved_paths": saved_paths})
        except Exception as e:
            self.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")


# ===========================================================================
# GUI - ASOSIY OYNA
# ===========================================================================
class FolderPicker(QWidget):
    """Papka tanlash uchun label + lineedit + browse tugmasi."""
    def __init__(self, label_text):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(label_text)
        self.label.setFixedWidth(220)
        self.line_edit = QLineEdit()
        self.btn = QPushButton("Tanlash...")
        self.btn.clicked.connect(self.browse)
        layout.addWidget(self.label)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.btn)
        self.setLayout(layout)

    def browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Papkani tanlang")
        if folder:
            self.line_edit.setText(folder)

    def path(self):
        return self.line_edit.text().strip()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPM ML — Oltin Ma'danlashuvi Prospektivligini Bashoratlash "
                             "(RF / SVM / XGBoost / CNN / Ansambl)")
        self.resize(1150, 800)

        self.training_result = None
        self.training_worker = None
        self.prediction_worker = None

        self._build_ui()

    # -------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.tab_data = QWidget()
        self.tab_roc = QWidget()
        self.tab_spatial = QWidget()
        self.tab_importance = QWidget()
        self.tab_map = QWidget()

        self.tabs.addTab(self.tab_data, "1. Ma'lumotlar va O'qitish")
        self.tabs.addTab(self.tab_roc, "2. ROC-AUC natijalar")
        self.tabs.addTab(self.tab_spatial, "3. Spatial CV diagnostika")
        self.tabs.addTab(self.tab_importance, "4. Feature Importance")
        self.tabs.addTab(self.tab_map, "5. Prognoz xarita")

        self._build_data_tab()
        self._build_roc_tab()
        self._build_spatial_tab()
        self._build_importance_tab()
        self._build_map_tab()

    # -------------------------------------------------------------
    def _build_data_tab(self):
        layout = QVBoxLayout()
        self.tab_data.setLayout(layout)

        # --- Kirish papkalari
        input_group = QGroupBox("Kirish ma'lumotlari (barchasi EPSG:28411 / GK 1942 zone 11 ga "
                                 "avtomatik reproyeksiya qilinadi)")
        input_layout = QVBoxLayout()
        input_group.setLayout(input_layout)

        self.picker_tiff = FolderPicker("TIFF qatlamlar papkasi:")
        self.picker_points = FolderPicker("Target nuqtalar papkasi (.shp):")
        self.picker_aoi = FolderPicker("Maydon konturi (AOI) papkasi (.shp):")
        input_layout.addWidget(self.picker_tiff)
        input_layout.addWidget(self.picker_points)
        input_layout.addWidget(self.picker_aoi)
        layout.addWidget(input_group)

        # --- Parametrlar
        param_group = QGroupBox("Model parametrlari")
        param_layout = QGridLayout()
        param_group.setLayout(param_layout)

        param_layout.addWidget(QLabel("Fon (pseudo-absence) nuqtalar soni:"), 0, 0)
        self.spin_background = QSpinBox()
        self.spin_background.setRange(10, 5000)
        self.spin_background.setValue(80)
        param_layout.addWidget(self.spin_background, 0, 1)

        param_layout.addWidget(QLabel("Min. masofa (musbat nuqtadan, metr):"), 0, 2)
        self.spin_min_dist = QSpinBox()
        self.spin_min_dist.setRange(0, 20000)
        self.spin_min_dist.setValue(500)
        self.spin_min_dist.setSingleStep(50)
        param_layout.addWidget(self.spin_min_dist, 0, 3)

        param_layout.addWidget(QLabel("K-fold (cross-validation) soni:"), 1, 0)
        self.spin_kfold = QSpinBox()
        self.spin_kfold.setRange(3, 10)
        self.spin_kfold.setValue(5)
        param_layout.addWidget(self.spin_kfold, 1, 1)

        param_layout.addWidget(QLabel("CV takrorlash soni (barqarorlik uchun):"), 1, 2)
        self.spin_repeats = QSpinBox()
        self.spin_repeats.setRange(1, 30)
        self.spin_repeats.setValue(10)
        self.spin_repeats.setToolTip(
            "1 dan katta bo'lsa, Repeated Stratified K-Fold qo'llaniladi: butun "
            "cross-validation turli tasodifiy fold-bo'linishlar bilan bir necha "
            "marta takrorlanadi va AUC uchun o'rtacha ± standart og'ish hisoblanadi. "
            "Bu kichik namunada (masalan, ~20 nuqta) bitta tasodifiy fold-bo'linishga "
            "bog'liq tebranishni kamaytiradi. Tavsiya: 10. CNN yoqilgan bo'lsa, "
            "hisoblash vaqti taxminan shu songa proporsional oshadi."
        )
        param_layout.addWidget(self.spin_repeats, 1, 3)

        param_layout.addWidget(QLabel("Spatial blok o'lchami, m (0 = avtomatik/variogram):"), 4, 0)
        self.spin_block_size = QSpinBox()
        self.spin_block_size.setRange(0, 100000)
        self.spin_block_size.setValue(0)
        self.spin_block_size.setSingleStep(100)
        self.spin_block_size.setToolTip(
            "Spatial block cross-validation uchun blok o'lchami. 0 bo'lsa, "
            "birinchi predictor rasterning empirik semivariogrammasidan "
            "avtomatik baholanadi. Bir xil blokdagi nuqtalar CV'da hech qachon "
            "train va validation orasida bo'linmaydi - shu bilan spatial "
            "autocorrelation orqali yuzaga keladigan leakage oldini oladi."
        )
        param_layout.addWidget(self.spin_block_size, 4, 1, 1, 3)

        self.check_bg_sensitivity = QCheckBox("Background sezgirlik tahlili ishlatilsin")
        self.check_bg_sensitivity.setToolTip(
            "Yoqilgan bo'lsa, dastur butun pipeline'ni (fon nuqtalar generatsiyasidan "
            "boshlab) bir necha marta, har safar boshqa tasodifiy fon tanlovi bilan "
            "takrorlaydi va AUC natijalarining fon tanloviga qanchalik bog'liqligini "
            "ko'rsatadi. Hisoblash vaqtini oshiradi."
        )
        param_layout.addWidget(self.check_bg_sensitivity, 5, 0, 1, 2)

        param_layout.addWidget(QLabel("  Sinovlar soni:"), 5, 2)
        self.spin_bg_draws = QSpinBox()
        self.spin_bg_draws.setRange(2, 30)
        self.spin_bg_draws.setValue(5)
        param_layout.addWidget(self.spin_bg_draws, 5, 3)

        self.check_calibrate = QCheckBox(
            "Ehtimolliklarni kalibrlash (Platt scaling) va shundan keyin ensemble'ga o'rtachalash"
        )
        self.check_calibrate.setChecked(True)
        self.check_calibrate.setToolTip(
            "Yoqilgan bo'lsa, RF/SVM/XGBoost ehtimolliklari CalibratedClassifierCV "
            "(sigmoid/Platt) orqali bir xil, solishtiriladigan shkalaga keltiriladi, "
            "shundan keyingina soft-voting ensemble hisoblanadi. Bu diapazoni turlicha "
            "(masalan RF 0.00-0.90 vs SVM 0.05-0.55) xom ehtimolliklarni to'g'ridan-to'g'ri "
            "o'rtachalash muammosini bartaraf etadi."
        )
        param_layout.addWidget(self.check_calibrate, 6, 0, 1, 4)

        self.check_xgb = QCheckBox("XGBoost ishlatilsin")
        self.check_xgb.setChecked(XGBOOST_AVAILABLE)
        self.check_xgb.setEnabled(XGBOOST_AVAILABLE)
        if not XGBOOST_AVAILABLE:
            self.check_xgb.setText("XGBoost ishlatilsin (o'rnatilmagan!)")
        param_layout.addWidget(self.check_xgb, 2, 2)

        self.check_cnn = QCheckBox("CNN ishlatilsin")
        self.check_cnn.setChecked(TF_AVAILABLE)
        self.check_cnn.setEnabled(TF_AVAILABLE)
        if not TF_AVAILABLE:
            self.check_cnn.setText("CNN ishlatilsin (TensorFlow o'rnatilmagan!)")
        param_layout.addWidget(self.check_cnn, 2, 3)

        self.check_assume_crs = QCheckBox(
            "TIFF/SHP faylida CRS yozilmagan bo'lsa, EPSG:28411 (GK 1942 zone 11) "
            "deb qabul qilinsin"
        )
        self.check_assume_crs.setChecked(True)
        self.check_assume_crs.setToolTip(
            "Yoqilgan bo'lsa: CRS metadata yo'q fayllar reproyeksiya qilinmaydi, "
            "koordinatalari allaqachon EPSG:28411 da deb hisoblanadi (odatiy holat, "
            "masalan Surfer'dan eksport qilingan grid fayllarida).\n"
            "O'chirilgan bo'lsa: CRS yo'q fayl uchrasa, dastur xatolik beradi."
        )
        param_layout.addWidget(self.check_assume_crs, 3, 0, 1, 4)

        layout.addWidget(param_group)

        # --- Boshqaruv
        control_layout = QHBoxLayout()
        self.btn_train = QPushButton("Modellarni o'qitish (Train + Cross-Validation)")
        self.btn_train.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_train.clicked.connect(self.start_training)
        control_layout.addWidget(self.btn_train)
        layout.addLayout(control_layout)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

    def _build_roc_tab(self):
        layout = QVBoxLayout()
        self.tab_roc.setLayout(layout)
        self.roc_figure = Figure(figsize=(7, 6))
        self.roc_canvas = FigureCanvas(self.roc_figure)
        layout.addWidget(self.roc_canvas)
        self.roc_summary_label = QLabel("Modellarni o'qitgandan so'ng ROC-AUC natijalari shu yerda ko'rinadi.")
        layout.addWidget(self.roc_summary_label)

    def _build_spatial_tab(self):
        layout = QVBoxLayout()
        self.tab_spatial.setLayout(layout)
        self.spatial_figure = Figure(figsize=(9, 5))
        self.spatial_canvas = FigureCanvas(self.spatial_figure)
        layout.addWidget(self.spatial_canvas)
        self.spatial_summary_label = QLabel(
            "Spatial block CV va random CV solishtiruvi shu yerda ko'rinadi. "
            "Katta farq (random AUC >> spatial AUC) - spatial autocorrelation "
            "tufayli random-CV optimistik ekanligini ko'rsatadi."
        )
        self.spatial_summary_label.setWordWrap(True)
        layout.addWidget(self.spatial_summary_label)

    def _build_importance_tab(self):
        layout = QVBoxLayout()
        self.tab_importance.setLayout(layout)
        self.imp_figure = Figure(figsize=(8, 6))
        self.imp_canvas = FigureCanvas(self.imp_figure)
        layout.addWidget(self.imp_canvas)

    def _build_map_tab(self):
        layout = QVBoxLayout()
        self.tab_map.setLayout(layout)

        out_layout = QHBoxLayout()
        self.picker_output = FolderPicker("Chiqish papkasi (GeoTIFF saqlash):")
        out_layout.addWidget(self.picker_output)
        layout.addLayout(out_layout)

        self.btn_predict_map = QPushButton("Prognoz xarita yaratish (butun maydon uchun)")
        self.btn_predict_map.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_predict_map.clicked.connect(self.start_map_prediction)
        self.btn_predict_map.setEnabled(False)
        layout.addWidget(self.btn_predict_map)

        self.map_figure = Figure(figsize=(7, 6))
        self.map_canvas = FigureCanvas(self.map_figure)
        layout.addWidget(self.map_canvas)

        self.map_log = QTextEdit()
        self.map_log.setReadOnly(True)
        self.map_log.setMaximumHeight(150)
        layout.addWidget(self.map_log)

    # -------------------------------------------------------------
    def log(self, msg):
        self.log_box.append(str(msg))

    def start_training(self):
        tiff_folder = self.picker_tiff.path()
        points_folder = self.picker_points.path()
        aoi_folder = self.picker_aoi.path()

        if not tiff_folder or not points_folder or not aoi_folder:
            QMessageBox.warning(self, "Xatolik", "Iltimos, barcha uchta papkani tanlang.")
            return

        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.btn_train.setEnabled(False)

        self.training_worker = TrainingWorker(
            tiff_folder=tiff_folder,
            points_folder=points_folder,
            aoi_folder=aoi_folder,
            n_background=self.spin_background.value(),
            min_distance=self.spin_min_dist.value(),
            n_splits=self.spin_kfold.value(),
            use_xgb=self.check_xgb.isChecked(),
            use_cnn=self.check_cnn.isChecked(),
            assume_crs_if_missing=self.check_assume_crs.isChecked(),
            n_repeats=self.spin_repeats.value(),
            block_size=self.spin_block_size.value(),
            bg_sensitivity_enabled=self.check_bg_sensitivity.isChecked(),
            bg_sensitivity_draws=self.spin_bg_draws.value(),
            bg_sensitivity_repeats=3,
            calibrate=self.check_calibrate.isChecked(),
        )
        self.training_worker.log_signal.connect(self.log)
        self.training_worker.progress_signal.connect(self.progress_bar.setValue)
        self.training_worker.finished_signal.connect(self.on_training_finished)
        self.training_worker.error_signal.connect(self.on_error)
        self.training_worker.start()

    def on_error(self, msg):
        self.btn_train.setEnabled(True)
        QMessageBox.critical(self, "Xatolik yuz berdi", msg)
        self.log("XATOLIK: " + msg)

    def on_training_finished(self, result):
        self.btn_train.setEnabled(True)
        self.training_result = result
        self.btn_predict_map.setEnabled(True)
        self.log("\n=== O'QITISH YAKUNLANDI ===")
        # Asosiy ROC-AUC grafigi - SPATIAL BLOCK CV natijalari (taqrizda talab
        # qilingan asosiy, leakage'siz baholash)
        self.draw_roc(result["roc_results"], title_suffix="Spatial block CV (asosiy)")
        self.draw_spatial_diagnostics(result)
        self.draw_importance(result)
        self.tabs.setCurrentWidget(self.tab_roc)

    def draw_roc(self, roc_results, title_suffix=""):
        self.roc_figure.clear()
        ax = self.roc_figure.add_subplot(111)
        for name, res in roc_results.items():
            lw = 3 if "Ensemble" in name else 1.5
            std = res.get("auc_std", 0.0)
            label = f"{name} (AUC = {res['auc']:.3f} \u00b1 {std:.3f})"
            ax.plot(res["fpr"], res["tpr"], lw=lw, label=label)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        title = "ROC egri chiziqlari (Repeated out-of-fold cross-validation)"
        if title_suffix:
            title += f" \u2014 {title_suffix}"
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        self.roc_figure.tight_layout()
        self.roc_canvas.draw()

        summary = "  |  ".join(
            f"{n}: AUC={r['auc']:.3f}\u00b1{r.get('auc_std', 0.0):.3f}"
            for n, r in roc_results.items()
        )
        self.roc_summary_label.setText(summary)

    def draw_spatial_diagnostics(self, result):
        """
        1) Random-CV va Spatial-block-CV AUC'larini yonma-yon ustunli
           diagramma sifatida solishtiradi (leakage optimizmini vizual
           ko'rsatish uchun).
        2) Nuqtalarni birinchi repeat'dagi spatial-block validation fold
           raqami bo'yicha rangga bo'lib xaritada ko'rsatadi - shu bilan
           bloklarning geografik taqsimotini tekshirish mumkin.
        """
        self.spatial_figure.clear()
        roc_spatial = result["roc_results"]
        roc_random = result.get("roc_results_random", {})

        ax1 = self.spatial_figure.add_subplot(1, 2, 1)
        names = list(roc_spatial.keys())
        x = np.arange(len(names))
        width = 0.35
        auc_random = [roc_random.get(n, {}).get("auc", np.nan) for n in names]
        auc_spatial = [roc_spatial[n]["auc"] for n in names]
        ax1.bar(x - width / 2, auc_random, width, label="Random CV", color="lightgray")
        ax1.bar(x + width / 2, auc_spatial, width, label="Spatial block CV", color="firebrick")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax1.set_ylabel("AUC")
        ax1.set_ylim(0.4, 1.0)
        ax1.axhline(0.5, color="gray", linestyle="--", lw=1)
        ax1.set_title("Random vs Spatial-block AUC")
        ax1.legend(fontsize=8)

        ax2 = self.spatial_figure.add_subplot(1, 2, 2)
        coords = result.get("coords")
        foldmap = result.get("foldmap_spatial")
        if coords is not None and foldmap is not None:
            sc = ax2.scatter(coords[:, 0], coords[:, 1], c=foldmap, cmap="tab10", s=25)
            n_pos = result["n_positive"]
            ax2.scatter(coords[:n_pos, 0], coords[:n_pos, 1], facecolors="none",
                        edgecolors="black", s=60, linewidths=1.2, label="Positive (known)")
            ax2.set_title(f"Spatial fold xaritasi (blok={result.get('block_size', 0):,.0f} m)")
            ax2.set_xlabel("X (EPSG:28411)")
            ax2.set_ylabel("Y (EPSG:28411)")
            ax2.legend(fontsize=7, loc="upper right")
            ax2.set_aspect("equal", adjustable="datalim")

        self.spatial_figure.tight_layout()
        self.spatial_canvas.draw()

        lines = []
        for n in names:
            r = roc_random.get(n, {}).get("auc", float("nan"))
            s = roc_spatial[n]["auc"]
            lines.append(f"{n}: random={r:.3f}, spatial={s:.3f}, optimizm={r - s:+.3f}")

        bg_sens = result.get("bg_sensitivity")
        if bg_sens is not None:
            lines.append("—")
            lines.append("Background sezgirlik (spatial CV, har xil fon tanlovlari):")
            for n, s in bg_sens["summary"].items():
                lines.append(f"{n}: AUC {s['mean']:.3f}\u00b1{s['std']:.3f} "
                              f"[{s['min']:.3f}-{s['max']:.3f}]")

        self.spatial_summary_label.setText(" | ".join(lines))

    def draw_importance(self, result):
        """
        Feature importance'ni ko'rsatadi. Endi MDI/gain o'rniga birinchi
        navbatda OUT-OF-FOLD PERMUTATION IMPORTANCE (fold'lar bo'yicha std
        bilan, error bar sifatida) chiziladi; agar SHAP mavjud bo'lsa,
        qo'shimcha qator sifatida SHAP |mean| qiymatlari ham ko'rsatiladi.
        Bu taqrizning "используйте out-of-fold permutation/SHAP" talabiga
        javob beradi va MDI/gain'ning correlated-predictor muammosini
        oldini oladi.

        MUHIM: bo'sh (ma'lumotsiz) subplot hosil bo'lishining oldini olish
        uchun endi faqat HAQIQATDA mazmunli array ("mean_abs_shap" mavjud va
        bo'sh emas) qaytargan SHAP natijalari uchungina alohida qator
        band qilinadi. Agar SHAP hech biri uchun ishlamagan bo'lsa (masalan,
        'shap' kutubxonasi o'rnatilmagan yoki kalibratsiya wrapper'idan
        bazaviy daraxt modelini chiqarib bo'lmagan), unda ikkinchi qator
        umuman ochilmaydi va bo'sh oq maydon qolmaydi.
        """
        band_names = result["band_names"]
        rf_importance = result.get("rf_importance")
        rf_std = result.get("rf_importance_std")
        xgb_importance = result.get("xgb_importance")
        xgb_std = result.get("xgb_importance_std")
        method = result.get("importance_method", "")
        shap_values_raw = result.get("shap_values") or {}

        # faqat haqiqatan mazmunli SHAP natijalarini qoldiramiz
        shap_values = {
            name: sv for name, sv in shap_values_raw.items()
            if isinstance(sv, dict) and sv.get("mean_abs_shap") is not None
            and len(np.asarray(sv["mean_abs_shap"])) == len(band_names)
        }

        self.imp_figure.clear()
        rows = []
        if rf_importance is not None:
            rows.append(("RandomForest", rf_importance, rf_std, "steelblue"))
        if xgb_importance is not None:
            rows.append(("XGBoost", xgb_importance, xgb_std, "darkorange"))

        n_shap = len(shap_values)
        n_cols = max(len(rows), n_shap, 1)
        n_plot_rows = 2 if n_shap > 0 else 1

        if not rows and n_shap == 0:
            ax = self.imp_figure.add_subplot(111)
            ax.text(0.5, 0.5, "Feature importance hisoblanmadi.",
                    ha="center", va="center", fontsize=10)
            ax.axis("off")
            self.imp_canvas.draw()
            return

        idx = 1
        for name, imp, std, color in rows:
            ax = self.imp_figure.add_subplot(n_plot_rows, n_cols, idx)
            order = np.argsort(imp)
            xerr = std[order] if std is not None else None
            ax.barh(np.array(band_names)[order], imp[order], xerr=xerr,
                    color=color, ecolor="black", capsize=2)
            ax.set_title(f"{name} \u2014 Permutation Importance\n(\u0394 ROC-AUC, OOF)", fontsize=9)
            idx += 1

        if n_shap > 0:
            idx = n_cols + 1
            for name, sv in shap_values.items():
                ax = self.imp_figure.add_subplot(n_plot_rows, n_cols, idx)
                mean_abs = np.asarray(sv["mean_abs_shap"])
                order = np.argsort(mean_abs)
                color = "steelblue" if name == "RandomForest" else "darkorange"
                ax.barh(np.array(band_names)[order], mean_abs[order], color=color, alpha=0.7)
                ax.set_title(f"{name} \u2014 SHAP |mean|\n(qo'shimcha tekshiruv)", fontsize=9)
                idx += 1

        if method:
            self.imp_figure.suptitle(f"Asosiy usul: {method}", fontsize=8, y=1.02)

        self.imp_figure.tight_layout()
        self.imp_canvas.draw()

    # -------------------------------------------------------------
    def start_map_prediction(self):
        if self.training_result is None:
            QMessageBox.warning(self, "Xatolik", "Avval modellarni o'qiting.")
            return
        out_dir = self.picker_output.path()
        if not out_dir:
            QMessageBox.warning(self, "Xatolik", "Chiqish papkasini tanlang.")
            return

        self.map_log.clear()
        self.btn_predict_map.setEnabled(False)

        r = self.training_result
        self.prediction_worker = PredictionWorker(
            stack=r["stack"], band_names=r["band_names"],
            final_models=r["final_models"], scaler=r["scaler"],
            ref_profile=r["ref_profile"], out_dir=out_dir,
            data_dictionary=r.get("data_dictionary"),
        )
        self.prediction_worker.log_signal.connect(lambda m: self.map_log.append(m))
        self.prediction_worker.finished_signal.connect(self.on_map_finished)
        self.prediction_worker.error_signal.connect(self.on_error)
        self.prediction_worker.start()

    def on_map_finished(self, result):
        self.btn_predict_map.setEnabled(True)
        prob_maps = result["prob_maps"]
        self.map_log.append(f"\nJami {len(result['saved_paths'])} ta GeoTIFF saqlandi.")

        self.map_figure.clear()
        ensemble_key = "Ensemble (soft-voting)"
        if ensemble_key in prob_maps:
            ax = self.map_figure.add_subplot(111)
            im = ax.imshow(prob_maps[ensemble_key], cmap="RdYlGn_r", vmin=0, vmax=1)
            ax.set_title("Ansambl (soft-voting) — Ma'danlashuv ehtimolligi xaritasi")
            self.map_figure.colorbar(im, ax=ax, label="Ehtimollik (0-1)")
            self.map_figure.tight_layout()
            self.map_canvas.draw()

        QMessageBox.information(self, "Tayyor", "Prognoz xaritalari muvaffaqiyatli yaratildi va saqlandi.")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
