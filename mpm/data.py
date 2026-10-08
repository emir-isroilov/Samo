# -*- coding: utf-8 -*-
"""
Ma'lumotlar moduli: raster yuklash va moslashtirish, FeaturePipeline (one-hot), musbat/fon nuqtalar,
Dataset, patchlar va ma'lumotlar diagnostikasi (statistika, korrelyatsiya, VIF).

Asl koddagi tuzatilgan xatolar:
  BUG-10  kategorik qatlamlar nearest bilan resample qilinadi, modelga one-hot beriladi;
  BUG-11  fon nuqtalar valid_mask ichida, vektorlashtirilgan batch generatsiya, kerakli soniga yetadi;
  BUG-14  toza GeoTIFF profili, dublikat musbatlar, MultiPoint, AOI tashqarisidagi nuqtalar,
          faylni ikki marta o'qimaslik, ishlatilmagan importlar yo'q.
"""
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

from .common import NODATA_OUT, RANDOM_STATE, TARGET_EPSG, check_cancel, noop_log
from .config import BACKGROUND_STRATEGIES

_MAX_ABS = 1e15              # |x| shundan katta bo'lsa sentinel deb NaN qilinadi
_INT_TOL = 1e-3              # kategorik qiymat butun son deb hisoblanishi uchun tolerantlik
_CAT_MAX_UNIQUE_FLOAT = 20   # float qatlam kategorik deb taklif qilinishi uchun maks. noyob qiymatlar
_CAT_MAX_UNIQUE_INT = 30     # butun dtype uchun (FeaturePipeline.max_levels bilan mos)
_PATCH_CHUNK_ELEMS = 8_000_000   # extract_patches: bir chunk'dagi maks. element soni (~32 MB)
_MAX_BATCH = 2_000_000       # fon nomzodlari: bir batch maks. hajmi
_GRID_JITTER = 0.8           # grid strategiyasida katak ichidagi og'ish (katak ulushi)
_VIF_RIDGE = 1e-8
_CONST_RTOL = 1e-12          # std <= bu * |mean| bo'lsa band o'zgarmas deb hisoblanadi


# ---------------------------------------------------------------------------
# Raster stack
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class RasterStack:
    """Bitta umumiy gridga moslashtirilgan qatlamlar to'plami."""
    stack: np.ndarray                 # (n_bands, H, W) float32, NaN = nodata
    band_names: list
    profile: dict                     # toza GeoTIFF profili (tiled/blocksize YO'Q)
    transform: Affine
    crs_epsg: int = TARGET_EPSG
    categorical: list = field(default_factory=list)
    tech_metadata: list = field(default_factory=list)

    def __post_init__(self):
        if self.stack.ndim != 3:
            raise ValueError(f"stack 3 o'lchamli (n_bands, H, W) bo'lishi kerak, berilgan: {self.stack.shape}")
        if len(self.band_names) != self.stack.shape[0]:
            raise ValueError(f"band_names soni ({len(self.band_names)}) stack bandlari "
                             f"({self.stack.shape[0]}) ga teng emas.")

    @property
    def n_bands(self):
        return int(self.stack.shape[0])

    @property
    def height(self):
        return int(self.stack.shape[1])

    @property
    def width(self):
        return int(self.stack.shape[2])

    @property
    def pixel_size(self):
        """Piksel o'lchami (metr): piksel yuzining kvadrat ildizi (kvadrat bo'lmagan piksellarda ham to'g'ri maydon)."""
        t = self.transform
        return float(np.sqrt(abs(t.a * t.e - t.b * t.d)))


def _stem(path):
    return os.path.splitext(os.path.basename(str(path)))[0]


def _names_list(names):
    """None / ro'yxat / massiv -> str ro'yxati."""
    return [] if names is None else [str(v) for v in names]


def _list_files(folder, suffixes):
    """Yashirin fayllar (masalan macOS '._x.tif') o'tkazib yuboriladi - asl koddagi glob ham ularni olmagan."""
    if not folder or not os.path.isdir(folder):
        return []
    names = [n for n in os.listdir(folder)
             if not n.startswith(".") and n.lower().endswith(suffixes) and os.path.isfile(os.path.join(folder, n))]
    return [os.path.join(folder, n) for n in sorted(names, key=lambda s: (s.lower(), s))]


def find_shapefile(folder):
    """Papkadagi birinchi .shp fayl (alifbo bo'yicha) yoki None."""
    files = _list_files(folder, (".shp",))
    return files[0] if files else None


def find_tiff_files(folder):
    """Papkadagi barcha .tif/.tiff fayllar (alifbo tartibida; birinchisi referens grid bo'ladi)."""
    return _list_files(folder, (".tif", ".tiff"))


def suggest_categorical_layers(tiff_paths):
    """Kategorik bo'lishi mumkin qatlamlar (band nomlari): butun dtype yoki (barcha qiymatlar butun va
    noyob qiymatlar <= 20). Butun dtype uchun noyob qiymatlar <= 30 sharti ham qo'yiladi (DEM kabi
    qatlamlar kategorik deb taklif qilinmasligi uchun). Katta fayllar kamaytirilgan o'lchamda o'qiladi."""
    out = []
    for path in tiff_paths:
        try:
            with rasterio.open(path) as src:
                is_int = np.issubdtype(np.dtype(src.dtypes[0]), np.integer)
                shape = (min(src.height, 1024), min(src.width, 1024))
                arr = src.read(1, out_shape=shape, masked=True, resampling=Resampling.nearest)
        except Exception:
            continue    # o'qib bo'lmaydigan fayl bu yerda emas, load_and_align_rasters'da aniq xato beradi
        vals = np.ma.compressed(arr).astype(np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        n_uniq = np.unique(vals).size
        if is_int:
            ok = n_uniq <= _CAT_MAX_UNIQUE_INT
        else:
            ok = n_uniq <= _CAT_MAX_UNIQUE_FLOAT and bool(np.all(vals == np.rint(vals)))
        if ok:
            out.append(_stem(path))
    return out


def _resolve_crs(crs, path, assume_crs_if_missing, fallback_epsg, log):
    """CRS yo'q bo'lsa fallback_epsg qo'llanadi (ogohlantirish bilan) yoki xato."""
    if crs is not None:
        return crs
    name = os.path.basename(str(path))
    if assume_crs_if_missing:
        log(f"  OGOHLANTIRISH: '{name}' faylida CRS yozilmagan. Fayl koordinatalari allaqachon "
            f"EPSG:{fallback_epsg} tizimida deb qabul qilinmoqda (reproyeksiya qilinmaydi, faqat CRS belgilanadi).")
        return CRS.from_epsg(fallback_epsg)
    raise ValueError(f"'{path}' faylida CRS aniqlanmagan. TIFF fayllarga CRS yozilganini tekshiring, yoki "
                     f"'CRS yo'q bo'lsa EPSG:{TARGET_EPSG} deb qabul qilinsin' katagini belgilang.")


def _is_target_crs(crs):
    try:
        return crs.to_epsg() == TARGET_EPSG or crs == CRS.from_epsg(TARGET_EPSG)
    except Exception:
        return False


def _same_transform(a, b, tol=1e-6):
    return bool(np.allclose(tuple(a)[:6], tuple(b)[:6], rtol=0.0, atol=tol))


def _read_band(path):
    """Faylni BIR marta ochadi: 1-band (nodata/inf/|x|>1e15 -> NaN, float32) va manba ma'lumotlari."""
    with rasterio.open(path) as src:
        info = {"crs": src.crs, "transform": src.transform, "width": src.width, "height": src.height,
                "bounds": tuple(src.bounds), "nodata": src.nodata, "dtype": src.dtypes[0], "count": src.count}
        masked = src.read(1, masked=True)
    with np.errstate(over="ignore", invalid="ignore"):
        arr = np.ma.filled(masked.astype(np.float32), np.nan)
        arr[~np.isfinite(arr)] = np.nan
        arr[np.abs(arr) > _MAX_ABS] = np.nan
    return arr, info


def _align_band(arr, crs, info, ref_transform, ref_shape, dst_crs, categorical):
    """Bandni referens gridga o'tkazadi. Grid bir xil bo'lsa resample QILINMAYDI."""
    if (_is_target_crs(crs) and tuple(arr.shape) == tuple(ref_shape)
            and _same_transform(info["transform"], ref_transform)):
        return arr
    dst = np.full(ref_shape, np.nan, dtype=np.float32)
    reproject(source=arr, destination=dst, src_transform=info["transform"], src_crs=crs, src_nodata=np.nan,
              dst_transform=ref_transform, dst_crs=dst_crs, dst_nodata=np.nan,
              resampling=Resampling.nearest if categorical else Resampling.bilinear)
    dst[~np.isfinite(dst)] = np.nan
    return dst


def _sig(x, digits=6):
    """Ahamiyatli raqamlar bo'yicha yaxlitlash (kichik/katta masshtabli qatlamlar 0 ga aylanib ketmasin)."""
    return float(f"{float(x):.{digits}g}")


def _tech_entry(path, crs, info, ref_transform, band):
    valid = band[np.isfinite(band)]
    has = valid.size > 0
    return {
        "band_name": _stem(path),
        "file_path": path,
        "source_crs": str(crs),
        "source_resolution_x_m": round(float(abs(info["transform"].a)), 3),
        "source_resolution_y_m": round(float(abs(info["transform"].e)), 3),
        "source_width_px": int(info["width"]),
        "source_height_px": int(info["height"]),
        "source_bounds": tuple(round(v, 2) for v in info["bounds"]),
        "source_nodata": info["nodata"],
        "source_dtype": info["dtype"],
        "reprojected_resolution_m": round(float(abs(ref_transform.a)), 3),
        "valid_pixels_pct": round(100.0 * valid.size / band.size, 2) if band.size else 0.0,
        "value_min": _sig(valid.min()) if has else None,
        "value_max": _sig(valid.max()) if has else None,
        "value_mean": _sig(valid.mean(dtype=np.float64)) if has else None,
        "value_std": _sig(valid.std(dtype=np.float64)) if has else None,
    }


def load_and_align_rasters(tiff_paths, categorical=(), log_fn=None, assume_crs_if_missing=True,
                           fallback_epsg=TARGET_EPSG, cancel=None):
    """
    Birinchi TIFF - referens grid (EPSG:28411 ga). Qolgan qatlamlar shu gridga moslanadi: kategorik
    qatlamlar Resampling.nearest, qolganlari bilinear. nodata/inf/|x|>1e15 -> NaN. Har bir fayl BIR marta
    ochiladi. CRS yo'q fayl: assume_crs_if_missing=True bo'lsa fallback_epsg deb olinadi (ogohlantirish bilan).
    band_names tartibi tiff_paths tartibiga mos (find_tiff_files alifbo tartibida beradi).
    """
    log = log_fn or noop_log
    paths = [str(p) for p in tiff_paths]
    if not paths:
        raise ValueError("TIFF fayllar topilmadi.")
    band_names = [_stem(p) for p in paths]
    dup = sorted({n for n in band_names if band_names.count(n) > 1})
    if dup:
        raise ValueError(f"Band nomlari takrorlanmoqda (bir xil nomli .tif/.tiff fayllar): {dup}")
    wanted = _names_list(categorical)
    cat = [b for b in band_names if b in wanted]
    unknown = [b for b in wanted if b not in band_names]
    if unknown:
        log(f"  OGOHLANTIRISH: kategorik deb belgilangan qatlamlar topilmadi, e'tiborsiz qoldirildi: {unknown}")

    dst_crs = CRS.from_epsg(TARGET_EPSG)
    n = len(paths)
    stack = None
    ref_transform = None
    tech_metadata = []
    for i, path in enumerate(paths):
        check_cancel(cancel)
        arr, info = _read_band(path)
        if info["transform"].is_identity:
            raise ValueError(f"'{os.path.basename(path)}' georeferenslanmagan (transform yo'q).")
        crs = _resolve_crs(info["crs"], path, assume_crs_if_missing, fallback_epsg, log)
        if info["count"] > 1:
            log(f"  OGOHLANTIRISH: '{os.path.basename(path)}' faylida {info['count']} ta band bor; faqat 1-band ishlatiladi.")
        if i == 0:
            if _is_target_crs(crs):
                ref_transform, height, width = info["transform"], info["height"], info["width"]
            else:
                ref_transform, width, height = calculate_default_transform(
                    crs, dst_crs, info["width"], info["height"], *info["bounds"])
            stack = np.full((n, height, width), np.nan, dtype=np.float32)
        band = _align_band(arr, crs, info, ref_transform, stack.shape[1:], dst_crs, band_names[i] in cat)
        stack[i] = band
        tech_metadata.append(_tech_entry(path, crs, info, ref_transform, band))
        if not np.isfinite(band).any():
            log(f"  OGOHLANTIRISH: '{band_names[i]}' qatlami referens gridga tushmadi yoki butunlay nodata.")
        if i == 0:
            log(f"[1/{n}] Referens qatlam: {band_names[0]} ({stack.shape[1]} x {stack.shape[2]}, "
                f"EPSG:{TARGET_EPSG})")
        else:
            how = "nearest" if band_names[i] in cat else "bilinear"
            log(f"[{i + 1}/{n}] Qatlam moslashtirildi: {band_names[i]} ({how})")
        del arr

    profile = {"driver": "GTiff", "dtype": "float32", "count": 1, "width": int(stack.shape[2]),
               "height": int(stack.shape[1]), "crs": f"EPSG:{TARGET_EPSG}", "transform": ref_transform,
               "nodata": NODATA_OUT, "compress": "lzw"}
    return RasterStack(stack=stack, band_names=band_names, profile=profile, transform=ref_transform,
                       crs_epsg=TARGET_EPSG, categorical=cat, tech_metadata=tech_metadata)


def valid_pixel_mask(stack):
    """(H, W) bool: barcha bandlar chekli bo'lgan piksellar. stack: ndarray (n,H,W) yoki RasterStack."""
    arr = getattr(stack, "stack", stack)
    mask = np.ones(arr.shape[1:], dtype=bool)
    for band in arr:
        mask &= np.isfinite(band)
    return mask


# ---------------------------------------------------------------------------
# Data dictionary (metadata)
# ---------------------------------------------------------------------------
MANUAL_METADATA_FIELDS = ["band_name", "source_owner", "survey_or_scene_id", "survey_date",
                          "original_scale_or_resolution", "transformation_applied", "notes"]


_METADATA_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "cp1252")   # Excel mintaqaviy sozlamalariga qarab
_METADATA_DELIMS = (",", ";")


def _mixed_script_words(text):
    """Bitta so'z ichida kirill VA lotin harflari aralashgan so'zlar soni (noto'g'ri kodlash belgisi)."""
    return sum(1 for w in re.findall(r"\w+", text)
               if re.search(r"[A-Za-z]", w) and re.search(r"[\u0400-\u04FF]", w))


def _read_metadata_csv(path):
    """metadata.csv ni ajratgich (',' / ';') va kodlashni (utf-8-sig, utf-8, cp1251, cp1252) avtomatik aniqlab o'qiydi.
    Qaytaradi: (DataFrame (dtype=str), kodlash, ajratgich). 'band_name' ustuni topilmasa ham, o'qish muvaffaqiyatli
    bo'lgan birinchi variant qaytariladi (chaqiruvchi ustun yo'qligini aniq xabar bilan bildiradi).
    O'qib bo'lmasa (ikkilik/NUL baytli fayl, hech bir kodlash mos kelmaydi, jadval sifatida tahlil bo'lmaydi) ValueError."""
    with open(path, "rb") as f:
        raw = f.read()
    last_err = None
    fallback = None
    candidates = []                                 # (kodlash, matn): muvaffaqiyatli dekodlanganlar, tartib bo'yicha
    for enc in _METADATA_ENCODINGS:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        if "\x00" in text:                         # matn emas (masalan UTF-16/ikkilik fayl)
            last_err = ValueError("faylda NUL baytlar bor: matnli CSV emas (UTF-16/ikkilik fayl bo'lishi mumkin)")
            continue
        candidates.append((enc, text))
    # cp1251/cp1252 bir xil baytlarni dekodlaydi: so'z ichida kirill va lotin harflari aralashgan bo'lsa (masalan
    # "Geolуgico") bu noto'g'ri kodlash belgisi - aralashmagan variant oldinga o'tadi (teng bo'lsa asl tartib saqlanadi).
    candidates = [c for c in candidates if c[0].startswith("utf")] + sorted(
        (c for c in candidates if not c[0].startswith("utf")), key=lambda c: _mixed_script_words(c[1]))
    for enc, text in candidates:
        for sep in _METADATA_DELIMS:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
            except Exception as e:
                last_err = e
                continue
            df.columns = [str(c).strip() for c in df.columns]
            if "band_name" in df.columns:
                return df, enc, sep
            if fallback is None:
                fallback = (df, enc, sep)
    if fallback is not None:
        return fallback
    reason = last_err if last_err is not None else "fayl bo'sh"
    raise ValueError(f"kodlash ({', '.join(_METADATA_ENCODINGS)}) va ajratgich (',' yoki ';') aniqlanmadi: {reason}")


def load_or_create_manual_metadata(tiff_folder, band_names, log_fn=None):
    """
    metadata.csv topilsa o'qiydi -> {band_name: {...}}; ajratgich (',' yoki ';') va kodlash (utf-8-sig, utf-8,
    cp1251, cp1252) avtomatik aniqlanadi (Excel mintaqaviy sozlamasi). Topilmasa KIRISH TIFF papkasiga bo'sh shablon
    yozadi va log'da buni aniq aytadi. Mavjud (lekin o'qib bo'lmaydigan) fayl HECH QACHON ustiga yozilmaydi.
    """
    log = log_fn or noop_log
    meta_path = os.path.join(tiff_folder, "metadata.csv")
    empty = {name: {} for name in band_names}
    if os.path.isfile(meta_path):
        try:
            df, enc, sep = _read_metadata_csv(meta_path)
        except Exception as e:
            log(f"  OGOHLANTIRISH: metadata.csv o'qib bo'lmadi ({e}); fayl o'zgartirilmadi, bo'sh metadata ishlatiladi. "
                f"Faylni Excel'da 'CSV UTF-8' formatida qayta saqlang.")
            return empty, meta_path
        if "band_name" not in df.columns:
            log("  OGOHLANTIRISH: metadata.csv'da 'band_name' ustuni yo'q; bo'sh metadata ishlatiladi.")
            return empty, meta_path
        missing_cols = [c for c in MANUAL_METADATA_FIELDS if c not in df.columns]
        if missing_cols:
            log(f"  OGOHLANTIRISH: metadata.csv'da ustunlar yetishmayapti: {missing_cols}")
        df["band_name"] = df["band_name"].str.strip()
        df = df.drop_duplicates("band_name").set_index("band_name")
        manual = {name: (df.loc[name].to_dict() if name in df.index else {}) for name in band_names}
        fmt = ""
        if sep != "," or enc not in ("utf-8-sig", "utf-8"):
            fmt = f" (avtomatik aniqlandi: ajratgich '{sep}', kodlash {enc})"
        log(f"  metadata.csv topildi va o'qildi: {meta_path}{fmt}")
        return manual, meta_path

    rows = [{f: (name if f == "band_name" else "") for f in MANUAL_METADATA_FIELDS} for name in band_names]
    try:
        pd.DataFrame(rows).to_csv(meta_path, index=False, encoding="utf-8-sig")     # Excel kirill/o'zbek harflarini tanisin
        log(f"  OGOHLANTIRISH: metadata.csv topilmadi. Bo'sh shablon KIRISH TIFF papkasiga yozildi: {meta_path} "
            f"(yon ta'sir: kirish papkasida yangi fayl paydo bo'ldi).\n"
            f"  Predictor data dictionary'ni to'liq qilish uchun uni source_owner, survey_date, "
            f"transformation_applied va boshqa ustunlar bilan to'ldiring.")
    except Exception as e:
        log(f"  OGOHLANTIRISH: metadata.csv shablonini yaratib bo'lmadi ({e}); kirish papkasi '{tiff_folder}' "
            f"yozishga ruxsat bermaydi bo'lishi mumkin.")
    return empty, meta_path


def build_data_dictionary(band_names, tech_metadata, manual_metadata):
    """Texnik (avtomatik) va qo'lda kiritilgan metadata'ni bitta jadvalga birlashtiradi."""
    if len(band_names) != len(tech_metadata):
        raise ValueError(f"band_names ({len(band_names)}) va tech_metadata ({len(tech_metadata)}) soni teng emas.")
    rows = []
    for name, tech in zip(band_names, tech_metadata):
        row = dict(tech)
        manual = manual_metadata.get(name, {}) or {}
        for f in MANUAL_METADATA_FIELDS:
            if f != "band_name":
                row[f] = manual.get(f, "")
        rows.append(row)
    return pd.DataFrame(rows)


def save_data_dictionary(df, out_dir, log_fn=None):
    log = log_fn or noop_log
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "predictor_data_dictionary.csv")
    df.to_csv(path, index=False)
    log(f"Predictor data dictionary saqlandi: {path}")
    return path


# ---------------------------------------------------------------------------
# FeaturePipeline: raster bandlar -> model feature'lari (kategorik: one-hot)
# ---------------------------------------------------------------------------
def _as_stack(stack):
    arr = getattr(stack, "stack", stack)
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"stack 3 o'lchamli (n_bands, H, W) bo'lishi kerak, berilgan: {arr.shape}")
    return arr


def _onehot(v, levels):
    """v (har qanday shakl, float) -> (len(levels), *v.shape) float32. NaN -> NaN; noma'lum daraja -> 0."""
    v = np.asarray(v)
    out = np.zeros((len(levels),) + v.shape, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        r = np.rint(v)
        intlike = np.abs(v - r) <= _INT_TOL
    nan = ~np.isfinite(v)
    for i, lv in enumerate(levels):
        out[i] = (r == lv) & intlike
    if nan.any():
        out[:, nan] = np.nan
    return out


class FeaturePipeline:
    """
    Bandlar -> feature'lar: raqamli band o'zgarishsiz, kategorik band one-hot (darajalar fit paytida
    valid piksellardan olinadi; noma'lum daraja => barcha ustunlar 0, NaN => NaN).
    """

    def __init__(self, band_names, categorical=(), max_levels=30):
        self.band_names = [str(b) for b in band_names]
        if len(set(self.band_names)) != len(self.band_names):
            raise ValueError("band_names takrorlanmasligi kerak.")
        wanted = _names_list(categorical)
        unknown = [b for b in wanted if b not in self.band_names]
        if unknown:
            raise ValueError(f"Kategorik qatlamlar band_names ichida yo'q: {unknown}")
        self.categorical = [b for b in self.band_names if b in wanted]
        self.max_levels = int(max_levels)
        self.levels = {}
        self._fitted = not self.categorical   # kategorik bo'lmasa o'rganiladigan narsa yo'q
        self._names = []
        self._feature_bands = []
        if self._fitted:
            self._build_names()

    # ---- ichki
    def _build_names(self):
        names, bands = [], []
        for b in self.band_names:
            if b in self.levels:
                names += [f"{b}=={int(lv)}" for lv in self.levels[b]]
                bands += [b] * len(self.levels[b])
            else:
                names.append(b)
                bands.append(b)
        self._names, self._feature_bands = names, bands

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("FeaturePipeline hali fit qilinmagan (kategorik darajalar noma'lum).")

    # ---- ochiq
    @property
    def feature_names(self):
        self._require_fitted()
        return list(self._names)

    @property
    def feature_bands(self):
        """Har bir feature qaysi band'dan kelgani (one-hot ustunlari uchun band nomi takrorlanadi)."""
        self._require_fitted()
        return list(self._feature_bands)

    @property
    def n_features(self):
        self._require_fitted()
        return len(self._names)

    def fit(self, stack):
        """Kategorik bandlar uchun barcha bandlar chekli bo'lgan piksellardan noyob BUTUN darajalarni oladi."""
        arr = _as_stack(stack)
        if arr.shape[0] != len(self.band_names):
            raise ValueError(f"stack bandlari ({arr.shape[0]}) band_names ({len(self.band_names)}) ga mos emas.")
        levels = {}
        if self.categorical:
            valid = valid_pixel_mask(arr)
            if not valid.any():
                raise ValueError("Barcha bandlar bir vaqtda chekli bo'lgan piksel yo'q; kategorik darajalarni "
                                 "aniqlab bo'lmadi (qatlamlar kesishmasligini tekshiring).")
            for b in self.categorical:
                vals = arr[self.band_names.index(b)][valid].astype(np.float64)
                r = np.rint(vals)
                if np.any(np.abs(vals - r) > _INT_TOL):
                    raise ValueError(f"'{b}' kategorik qatlam butun sonli emas (bilinear resample yoki "
                                     f"uzluksiz qatlam bo'lishi mumkin). Uni kategorik deb belgilamang.")
                uniq = np.unique(r)
                if uniq.size > self.max_levels:
                    raise ValueError(f"'{b}' kategorik qatlamda {uniq.size} ta daraja bor (maks. {self.max_levels}). "
                                     f"Bu qatlam uzluksiz bo'lishi mumkin - kategorik belgisini olib tashlang.")
                levels[b] = [int(v) for v in uniq]
        self.levels = levels
        self._fitted = True
        self._build_names()
        return self

    def transform_pixels(self, pix):
        """pix (m, n_bands) -> (m, n_features) float32. NaN saqlanadi; kategorikda noma'lum daraja => 0."""
        self._require_fitted()
        pix = np.asarray(pix)
        if pix.ndim != 2 or pix.shape[1] != len(self.band_names):
            raise ValueError(f"pix shakli (m, {len(self.band_names)}) bo'lishi kerak, berilgan: {pix.shape}")
        out = np.empty((pix.shape[0], len(self._names)), dtype=np.float32)
        k = 0
        for j, b in enumerate(self.band_names):
            if b in self.levels:
                lv = self.levels[b]
                out[:, k:k + len(lv)] = _onehot(pix[:, j], lv).T
                k += len(lv)
            else:
                out[:, k] = pix[:, j]
                k += 1
        return out

    def transform_stack(self, stack):
        """stack (n_bands, H, W) -> (n_features, H, W) float32. NaN saqlanadi."""
        self._require_fitted()
        arr = _as_stack(stack)
        if arr.shape[0] != len(self.band_names):
            raise ValueError(f"stack bandlari ({arr.shape[0]}) band_names ({len(self.band_names)}) ga mos emas.")
        out = np.empty((len(self._names),) + arr.shape[1:], dtype=np.float32)
        k = 0
        for j, b in enumerate(self.band_names):
            if b in self.levels:
                lv = self.levels[b]
                out[k:k + len(lv)] = _onehot(arr[j], lv)
                k += len(lv)
            else:
                out[k] = arr[j]
                k += 1
        return out

    def to_dict(self):
        """JSON'ga yaroqli ko'rinish (persist uchun)."""
        d = {"version": 1, "band_names": list(self.band_names), "categorical": list(self.categorical),
             "max_levels": int(self.max_levels), "fitted": bool(self._fitted),
             "levels": {b: [int(v) for v in lv] for b, lv in self.levels.items()}}
        if self._fitted:
            d["feature_names"] = list(self._names)
        return d

    @classmethod
    def from_dict(cls, d):
        if "band_names" not in d:
            raise ValueError("FeaturePipeline lug'atida 'band_names' yo'q.")
        obj = cls(d["band_names"], d.get("categorical", ()), d.get("max_levels", 30))
        if d.get("fitted", True) and obj.categorical:
            missing = [b for b in obj.categorical if b not in (d.get("levels") or {})]
            if missing:
                raise ValueError(f"FeaturePipeline lug'atida darajalar yetishmayapti: {missing}")
            obj.levels = {b: [int(v) for v in d["levels"][b]] for b in obj.categorical}
            obj._fitted = True
            obj._build_names()
        return obj


# ---------------------------------------------------------------------------
# Geometriya yordamchilari
# ---------------------------------------------------------------------------
def xy_to_rowcol(transform, x, y):
    """Koordinatalar -> (rows, cols) int64 (floor, vektorlashtirilgan). Chegara tekshirilmaydi;
    chekli bo'lmagan koordinata -1 ga o'tadi (ya'ni raster tashqarisi)."""
    inv = ~transform
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    with np.errstate(invalid="ignore", over="ignore"):
        cols = np.floor(inv.a * x + inv.b * y + inv.c)
        rows = np.floor(inv.d * x + inv.e * y + inv.f)
    bad = ~(np.isfinite(rows) & np.isfinite(cols))
    rows[bad] = -1.0
    cols[bad] = -1.0
    lim = 1e15
    return (np.clip(rows, -lim, lim).astype(np.int64), np.clip(cols, -lim, lim).astype(np.int64))


def _gdf_xy(gdf):
    """GeoDataFrame/GeoSeries (Point) -> (n, 2) float64."""
    geom = gdf.geometry
    return np.column_stack([geom.x.to_numpy(dtype=np.float64), geom.y.to_numpy(dtype=np.float64)])


def _as_xy(xy):
    if hasattr(xy, "geometry"):
        return _gdf_xy(xy)
    arr = np.asarray(xy, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"xy shakli (n, 2) bo'lishi kerak, berilgan: {arr.shape}")
    return arr


def _union_all(gdf):
    geoms = gdf.geometry
    return geoms.union_all() if hasattr(geoms, "union_all") else geoms.unary_union


def _fix_polygon(geom):
    """Noto'g'ri (o'zi bilan kesishuvchi) poligonni tuzatadi."""
    if geom.is_valid:
        return geom
    try:
        fixed = shapely.make_valid(geom)
    except AttributeError:
        return geom.buffer(0)
    if fixed.geom_type in ("Polygon", "MultiPolygon"):
        return fixed
    parts = [g for g in getattr(fixed, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(parts) if parts else geom.buffer(0)


def _prepare(geom):
    try:
        shapely.prepare(geom)    # shapely>=2: tezkor takroriy tekshiruv uchun
    except AttributeError:
        pass
    return geom


def _contains_xy(geom, x, y):
    """(x, y) nuqtalar geom ichidami. shapely>=2: contains_xy; zaxira: shapely.vectorized, keyin prepared geometry."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return np.zeros(0, dtype=bool)
    fn = getattr(shapely, "contains_xy", None)
    if fn is not None:
        return np.asarray(fn(geom, x, y), dtype=bool)
    try:
        from shapely.vectorized import contains as vec_contains
    except ImportError:
        vec_contains = None
    if vec_contains is not None:
        return np.asarray(vec_contains(geom, x, y), dtype=bool)
    prepared = prep(geom)
    return np.fromiter((prepared.contains(Point(px, py)) for px, py in zip(x, y)), dtype=bool, count=x.size)


def _to_target_crs(gdf, name, assume_crs_if_missing, log):
    if gdf.crs is None:
        if not assume_crs_if_missing:
            raise ValueError(f"'{name}' faylida CRS aniqlanmagan. Faylga CRS yozing yoki "
                             f"'CRS yo'q bo'lsa EPSG:{TARGET_EPSG} deb qabul qilinsin' katagini belgilang.")
        log(f"  OGOHLANTIRISH: '{name}' faylida CRS yozilmagan; koordinatalar EPSG:{TARGET_EPSG} deb qabul qilindi.")
        return gdf.set_crs(epsg=TARGET_EPSG)
    if gdf.crs.to_epsg() != TARGET_EPSG:
        log(f"  '{name}' EPSG:{TARGET_EPSG} ga o'tkazildi (manba: {gdf.crs.to_string()}).")
        return gdf.to_crs(epsg=TARGET_EPSG)
    return gdf


def _in_target_crs(gdf):
    """CRS berilgan va EPSG:28411 dan farq qilsa, 28411 ga o'tkazadi (CRS yo'q => 28411 deb olinadi)."""
    if gdf is not None and gdf.crs is not None and gdf.crs.to_epsg() != TARGET_EPSG:
        return gdf.to_crs(epsg=TARGET_EPSG)
    return gdf


def _aoi_geometry(aoi_gdf):
    """AOI GeoDataFrame -> tuzatilgan va prepared birlashma geometriyasi (EPSG:28411 deb olinadi).
    Noto'g'ri geometriyalar birlashtirishdan oldin tuzatiladi (aks holda union TopologyException berishi mumkin)."""
    if aoi_gdf is None or len(aoi_gdf) == 0:
        raise ValueError("AOI bo'sh.")
    geoms = [_fix_polygon(g) for g in _in_target_crs(aoi_gdf).geometry if g is not None and not g.is_empty]
    if not geoms:
        raise ValueError("AOI geometriyasi bo'sh.")
    return _prepare(_fix_polygon(_union_all(gpd.GeoSeries(geoms))))


def load_aoi(aoi_shp, assume_crs_if_missing=True, log_fn=None):
    """AOI shapefile -> poligonlar GeoDataFrame'i (EPSG:28411). Noto'g'ri geometriya tuzatiladi."""
    log = log_fn or noop_log
    gdf = gpd.read_file(aoi_shp)
    name = os.path.basename(str(aoi_shp))
    if len(gdf) == 0:
        raise ValueError(f"AOI fayli bo'sh: {aoi_shp}")
    gdf = _to_target_crs(gdf, name, assume_crs_if_missing, log)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        raise ValueError(f"AOI faylida poligon topilmadi: {aoi_shp}")
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(_fix_polygon)
    return gdf.reset_index(drop=True)


def load_positive_points(points_shp, aoi_gdf=None, assume_crs_if_missing=True, log_fn=None):
    """
    Musbat nuqtalar (EPSG:28411). MultiPoint -> alohida nuqtalar; faqat Point qoladi;
    AOI tashqarisidagilar ogohlantirish bilan tashlanadi.
    """
    log = log_fn or noop_log
    gdf = gpd.read_file(points_shp)
    name = os.path.basename(str(points_shp))
    if len(gdf) == 0:
        raise ValueError(f"Musbat nuqtalar fayli bo'sh: {points_shp}")
    gdf = _to_target_crs(gdf, name, assume_crs_if_missing, log)
    empty = gdf.geometry.isna() | gdf.geometry.is_empty
    if empty.any():
        log(f"  OGOHLANTIRISH: {int(empty.sum())} ta bo'sh geometriya tashlandi.")
    gdf = gdf[~empty]
    if gdf.empty:
        raise ValueError(f"Musbat nuqtalar faylida geometriya topilmadi: {points_shp}")
    n_multi = int((gdf.geom_type == "MultiPoint").sum())
    if n_multi:
        log(f"  {n_multi} ta MultiPoint alohida nuqtalarga ajratildi.")
    gdf = gdf.explode(index_parts=False)
    is_pt = gdf.geom_type == "Point"
    if (~is_pt).any():
        log(f"  OGOHLANTIRISH: {int((~is_pt).sum())} ta Point bo'lmagan geometriya tashlandi.")
    gdf = gdf[is_pt]
    if gdf.empty:
        raise ValueError(f"Musbat nuqtalar faylida Point geometriya topilmadi: {points_shp}")
    if aoi_gdf is not None:
        xy = _gdf_xy(gdf)
        inside = _contains_xy(_aoi_geometry(aoi_gdf), xy[:, 0], xy[:, 1])
        if not inside.all():
            log(f"  OGOHLANTIRISH: {int((~inside).sum())} ta musbat nuqta AOI tashqarisida, tashlandi.")
        gdf = gdf[inside]
        if gdf.empty:
            raise ValueError("Barcha musbat nuqtalar AOI tashqarisida; AOI va nuqtalar CRS/hududini tekshiring.")
    return gdf.reset_index(drop=True)


def dedupe_points_by_pixel(points_gdf, transform, shape, log_fn=None):
    """Bir pikselga tushgan musbat nuqtalardan birinchisini qoldiradi. Raster tashqarisidagilar tegilmaydi.
    shape = (H, W) (yoki stack.shape)."""
    log = log_fn or noop_log
    height, width = int(shape[-2]), int(shape[-1])
    xy = _gdf_xy(points_gdf)
    rows, cols = xy_to_rowcol(transform, xy[:, 0], xy[:, 1])
    inb = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    keep = np.ones(len(rows), dtype=bool)
    idx = np.flatnonzero(inb)
    if idx.size:
        _, first = np.unique(rows[idx] * width + cols[idx], return_index=True)
        dup = np.ones(idx.size, dtype=bool)
        dup[first] = False
        keep[idx[dup]] = False
    n_removed = int((~keep).sum())
    if n_removed:
        log(f"  OGOHLANTIRISH: {n_removed} ta musbat nuqta boshqasi bilan bir pikselga tushgani uchun olib tashlandi.")
    return points_gdf[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fon (pseudo-absence) nuqtalar
# ---------------------------------------------------------------------------
def _pixel_key(rows, cols):
    return rows.astype(np.int64) * 4294967296 + cols.astype(np.int64)


class _BgSampler:
    """Fon nomzodlarini yaratadi va tekshiradi: AOI ichida, valid_mask ichida, musbatlardan
    min_distance dan uzoqda, (transform berilsa) har pikselda bittadan va musbat pikselidan tashqarida."""

    def __init__(self, aoi, pos_xy, min_distance, valid_mask, transform):
        self.aoi = aoi
        self.bounds = aoi.bounds
        self.min_distance = float(min_distance)
        self.tree = cKDTree(pos_xy) if len(pos_xy) else None
        self.valid_mask = valid_mask
        self.transform = transform
        self.used = np.empty(0, dtype=np.int64)
        if transform is not None and len(pos_xy):
            self.used = np.unique(_pixel_key(*xy_to_rowcol(transform, pos_xy[:, 0], pos_xy[:, 1])))
        self.pix_r = self.pix_c = None
        if valid_mask is not None:
            self._init_pixels()

    def _init_pixels(self):
        """AOI oynasidagi yaroqli piksellar ro'yxati (past qabul qilish darajasida ham samarali)."""
        height, width = self.valid_mask.shape
        minx, miny, maxx, maxy = self.bounds
        rr, cc = xy_to_rowcol(self.transform, np.array([minx, minx, maxx, maxx]), np.array([miny, maxy, miny, maxy]))
        r0, r1 = max(int(rr.min()), 0), min(int(rr.max()), height - 1)
        c0, c1 = max(int(cc.min()), 0), min(int(cc.max()), width - 1)
        if r0 > r1 or c0 > c1:
            raise ValueError("AOI raster chegarasi bilan kesishmaydi (CRS/hududni tekshiring).")
        step = max(1, 4_000_000 // (c1 - c0 + 1))
        rows, cols = [], []
        for s in range(r0, r1 + 1, step):
            nz_r, nz_c = np.nonzero(self.valid_mask[s:min(s + step, r1 + 1), c0:c1 + 1])
            rows.append((nz_r + s).astype(np.int32))
            cols.append((nz_c + c0).astype(np.int32))
        self.pix_r, self.pix_c = np.concatenate(rows), np.concatenate(cols)
        if self.pix_r.size == 0:
            raise ValueError("valid_mask AOI ichida yaroqli piksel yo'q (barcha qatlamlarda nodata).")

    def draw(self, rng, count):
        """Tekis nomzodlar: valid piksellardan (piksel ichida tasodifiy joy) yoki AOI bbox'idan."""
        if self.pix_r is not None:
            k = rng.integers(0, self.pix_r.size, size=count)
            c = self.pix_c[k] + rng.random(count)
            r = self.pix_r[k] + rng.random(count)
            t = self.transform
            return t.a * c + t.b * r + t.c, t.d * c + t.e * r + t.f
        minx, miny, maxx, maxy = self.bounds
        return rng.uniform(minx, maxx, count), rng.uniform(miny, maxy, count)

    def accept(self, x, y, register=True):
        """Yaroqli nomzodlar (x, y, d); d - eng yaqin musbat nuqtagacha masofa (musbat yo'q bo'lsa inf)."""
        if self.valid_mask is not None:
            height, width = self.valid_mask.shape
            r, c = xy_to_rowcol(self.transform, x, y)
            ok = (r >= 0) & (r < height) & (c >= 0) & (c < width)
            ok[ok] = self.valid_mask[r[ok], c[ok]]
            x, y = x[ok], y[ok]
        inside = _contains_xy(self.aoi, x, y)
        x, y = x[inside], y[inside]
        if self.tree is not None and x.size:
            d = self.tree.query(np.column_stack([x, y]), k=1)[0]
            ok = d >= self.min_distance
            x, y, d = x[ok], y[ok], d[ok]
        else:
            d = np.full(x.size, np.inf)
        if self.transform is not None and x.size:
            keys = _pixel_key(*xy_to_rowcol(self.transform, x, y))
            first = np.zeros(keys.size, dtype=bool)
            first[np.unique(keys, return_index=True)[1]] = True
            ok = first & ~np.isin(keys, self.used)
            x, y, d = x[ok], y[ok], d[ok]
            if register:
                self.used = np.concatenate([self.used, keys[ok]])
        return x, y, d


def _bg_collect(sampler, rng, target, max_attempts, cancel=None):
    """Tekis nomzodlarni batch'lab yig'adi, to'g'ri nuqtalar `target` ga yetguncha (yoki urinishlar tugaguncha)."""
    xs, ys, ds = [], [], []
    have = attempts = 0
    while have < target and attempts < max_attempts:
        check_cancel(cancel)
        batch = int(min(max(4096, 6 * (target - have)), _MAX_BATCH, max_attempts - attempts))
        x, y, d = sampler.accept(*sampler.draw(rng, batch))
        attempts += batch
        xs.append(x)
        ys.append(y)
        ds.append(d)
        have += x.size
    if not xs:
        return np.empty(0), np.empty(0), np.empty(0), attempts
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ds), attempts


def _bg_grid(sampler, rng, n_points, max_cells, cancel=None):
    """AOI ustida jitterli to'r: yetarli nomzod chiqquncha katakni kichraytiradi, keyin n_points tanlanadi."""
    minx, miny, maxx, maxy = sampler.bounds
    w, h = maxx - minx, maxy - miny
    cell = float(np.sqrt(max(sampler.aoi.area, 1e-12) / (1.5 * n_points)))
    best, attempts = (np.empty(0), np.empty(0)), 0
    for _ in range(60):
        check_cancel(cancel)
        nx, ny = int(np.ceil(w / cell)), int(np.ceil(h / cell))
        if nx * ny > max_cells:
            if attempts:
                break
            cell *= np.sqrt(nx * ny / max_cells) * 1.01     # birinchi to'r juda zich (ingichka AOI): hajmni cheklaymiz
            continue
        gx, gy = np.meshgrid(minx + (np.arange(nx) + 0.5) * cell, miny + (np.arange(ny) + 0.5) * cell)
        jx = (rng.random(gx.size) - 0.5) * cell * _GRID_JITTER
        jy = (rng.random(gy.size) - 0.5) * cell * _GRID_JITTER
        attempts += gx.size
        x, y, _d = sampler.accept(gx.ravel() + jx, gy.ravel() + jy, register=False)
        if x.size > best[0].size:
            best = (x, y)
        if x.size >= n_points:
            break
        cell *= 0.8
    x, y = best
    if x.size > n_points:
        sel = np.sort(rng.choice(x.size, size=n_points, replace=False))
        x, y = x[sel], y[sel]
    return x, y, attempts


def generate_background_points(aoi_gdf, positive_gdf, n_points, min_distance, random_state=RANDOM_STATE,
                               strategy="random", valid_mask=None, transform=None, max_attempts_factor=200,
                               log_fn=None, cancel=None):
    """
    AOI ichida, musbat nuqtalardan kamida min_distance uzoqlikdagi fon (pseudo-absence) nuqtalar.
    strategy: "random" (AOI ∩ valid_mask ichida tekis), "grid" (jitterli to'r, keyin tanlanadi),
    "distance_weighted" (musbatlardan masofaga proporsional ehtimol). valid_mask (H,W) bo'lsa transform ham
    kerak; transform berilganda har pikselga bittadan nuqta tushadi va musbat pikselidan tashqarida.
    Natija n_points ga yetadi, yetmasa ogohlantirish beriladi.
    """
    log = log_fn or noop_log
    if strategy not in BACKGROUND_STRATEGIES:
        raise ValueError(f"Fon strategiyasi noto'g'ri: '{strategy}' (mumkin: {list(BACKGROUND_STRATEGIES)})")
    n_points = int(n_points)
    if n_points < 1:
        raise ValueError("n_points kamida 1 bo'lishi kerak.")
    if min_distance < 0:
        raise ValueError("min_distance manfiy bo'lmasligi kerak.")
    if valid_mask is not None:
        if transform is None:
            raise ValueError("valid_mask berilsa, transform ham berilishi kerak.")
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.ndim != 2:
            raise ValueError(f"valid_mask 2 o'lchamli (H, W) bo'lishi kerak, berilgan: {valid_mask.shape}")

    rng = np.random.default_rng(int(random_state) % (2 ** 32 - 1))
    positive_gdf = _in_target_crs(positive_gdf)
    pos_xy = _gdf_xy(positive_gdf) if positive_gdf is not None and len(positive_gdf) else np.empty((0, 2))
    sampler = _BgSampler(_aoi_geometry(aoi_gdf), pos_xy, min_distance, valid_mask, transform)
    max_attempts = max(int(n_points * max_attempts_factor), 1000)
    if valid_mask is not None:
        # har pikselga bittadan nuqta: mavjud piksellardan ko'p so'ralsa behuda uzoq urinmaymiz
        max_attempts = min(max_attempts, 50 * int(valid_mask.sum()) + 10_000)

    if strategy == "grid":
        x, y, attempts = _bg_grid(sampler, rng, n_points, max_cells=max_attempts, cancel=cancel)
    else:
        pool = n_points if strategy == "random" else max(20 * n_points, 2000)
        x, y, d, attempts = _bg_collect(sampler, rng, pool, max_attempts, cancel=cancel)
        if strategy == "distance_weighted" and not len(pos_xy):
            log("  OGOHLANTIRISH: musbat nuqta yo'q, distance_weighted o'rniga tekis tanlov ishlatildi.")
        if x.size > n_points:
            if strategy == "distance_weighted" and len(pos_xy):
                w = d / d.max() + 1e-6 if d.max() > 0 else np.ones(d.size)
                sel = rng.choice(x.size, size=n_points, replace=False, p=w / w.sum())
            else:
                sel = np.arange(n_points)
            x, y = x[sel], y[sel]

    if x.size < n_points:
        log(f"  OGOHLANTIRISH: faqat {x.size}/{n_points} ta fon nuqta topildi ({attempts} urinish). "
            f"min_distance ni kamaytiring yoki AOI/valid_mask ni tekshiring.")
    else:
        log(f"  Fon nuqtalar ({strategy}): {x.size} ta ({attempts} urinish).")
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=_in_target_crs(aoi_gdf).crs)


# ---------------------------------------------------------------------------
# Piksel namunalari, patchlar, Dataset
# ---------------------------------------------------------------------------
def sample_features(xy, raster_or_stack, transform=None):
    """
    Koordinatalar bo'yicha piksel qiymatlari: (values (n, n_bands) float32, rows, cols). Raster tashqarisi => NaN
    (rows/cols hisoblangan holda qaytariladi). xy: (n, 2) massiv yoki Point GeoDataFrame.
    """
    stack = np.asarray(getattr(raster_or_stack, "stack", raster_or_stack))
    if stack.ndim != 3:
        raise ValueError(f"stack 3 o'lchamli (n_bands, H, W) bo'lishi kerak, berilgan: {stack.shape}")
    if transform is None:
        transform = getattr(raster_or_stack, "transform", None)
    if transform is None:
        raise ValueError("transform berilmagan.")
    xy = _as_xy(xy)
    n_bands, height, width = stack.shape
    rows, cols = xy_to_rowcol(transform, xy[:, 0], xy[:, 1])
    inb = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    values = np.full((len(rows), n_bands), np.nan, dtype=np.float32)
    if inb.any():
        values[inb] = stack[:, rows[inb], cols[inb]].T
    return values, rows, cols


def extract_patches(feature_stack, rows, cols, window):
    """(m, w, w, p) float32 patchlar. Chegaradan tashqari va nodata piksellar NaN. Xotira uchun chunk bilan."""
    fs = np.asarray(feature_stack)
    if fs.ndim != 3:
        raise ValueError(f"feature_stack 3 o'lchamli (p, H, W) bo'lishi kerak, berilgan: {fs.shape}")
    if not np.issubdtype(fs.dtype, np.floating):
        fs = fs.astype(np.float32)          # NaN bilan to'ldirish uchun
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError(f"Oyna o'lchami toq musbat son bo'lishi kerak: {window}")
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    if rows.shape != cols.shape:
        raise ValueError("rows va cols uzunligi teng bo'lishi kerak.")
    p, height, width = fs.shape
    offs = np.arange(-(window // 2), window // 2 + 1, dtype=np.int64)
    m = rows.size
    out = np.empty((m, window, window, p), dtype=np.float32)
    chunk = max(1, _PATCH_CHUNK_ELEMS // (window * window * p))
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        rr = rows[s:e, None, None] + offs[None, :, None]       # (c, w, 1)
        cc = cols[s:e, None, None] + offs[None, None, :]       # (c, 1, w)
        inb = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        vals = np.moveaxis(fs[:, np.clip(rr, 0, height - 1), np.clip(cc, 0, width - 1)], 0, -1)
        vals[~inb] = np.nan
        out[s:e] = vals
    return out


@dataclass(eq=False)
class Dataset:
    """Model uchun jadval: musbatlar BIRINCHI, keyin fon."""
    X: np.ndarray                  # (n, n_features) float32
    y: np.ndarray                  # (n,) int8
    coords: np.ndarray             # (n, 2) float64
    rows: np.ndarray               # (n,) int64
    cols: np.ndarray               # (n,) int64
    feature_names: list
    feature_stack: object = None   # (n_features, H, W) yoki None - faqat patch kerak bo'lsa

    def subset(self, idx):
        """Indeks yoki bool maska bo'yicha qism; feature_stack nusxalanmaydi (ulashiladi)."""
        idx = np.asarray(idx)
        return Dataset(X=self.X[idx], y=self.y[idx], coords=self.coords[idx], rows=self.rows[idx],
                       cols=self.cols[idx], feature_names=self.feature_names, feature_stack=self.feature_stack)

    def get_patches(self, window):
        """(n, w, w, n_features) float32 patchlar (chegara/nodata NaN)."""
        if self.feature_stack is None:
            raise ValueError("feature_stack yo'q: build_dataset(..., need_feature_stack=True) bilan yarating.")
        return extract_patches(self.feature_stack, self.rows, self.cols, window)

    @property
    def n_pos(self):
        return int((self.y == 1).sum())

    @property
    def n_neg(self):
        return int((self.y == 0).sum())

    @property
    def n(self):
        return int(self.y.shape[0])


def build_dataset(raster, pipeline, positive_gdf, background_gdf, log_fn=None, need_feature_stack=False):
    """Musbat + fon nuqtalardan Dataset. Chekli bo'lmagan (nodata/chegaradan tashqari) qatorlar tashlanadi."""
    log = log_fn or noop_log
    if list(pipeline.band_names) != list(raster.band_names):
        raise ValueError("FeaturePipeline band_names raster band_names bilan mos emas.")
    xy_pos, xy_neg = _gdf_xy(positive_gdf), _gdf_xy(background_gdf)
    coords = np.vstack([xy_pos, xy_neg])
    y = np.concatenate([np.ones(len(xy_pos), dtype=np.int8), np.zeros(len(xy_neg), dtype=np.int8)])
    values, rows, cols = sample_features(coords, raster.stack, raster.transform)
    X = pipeline.transform_pixels(values)
    keep = np.isfinite(X).all(axis=1)
    n_bad_pos, n_bad_neg = int((~keep[y == 1]).sum()), int((~keep[y == 0]).sum())
    if n_bad_pos or n_bad_neg:
        log(f"  OGOHLANTIRISH: yaroqsiz piksellarga (nodata/raster tashqarisi) tushgan nuqtalar tashlandi: "
            f"{n_bad_pos} ta musbat, {n_bad_neg} ta fon.")
    y = y[keep]
    if not (y == 1).any():
        raise ValueError("Yaroqli musbat nuqta qolmadi (barchasi nodata yoki raster tashqarisida).")
    if not (y == 0).any():
        raise ValueError("Yaroqli fon nuqta qolmadi (fon nuqtalar sonini yoki valid_mask ni tekshiring).")
    feature_stack = pipeline.transform_stack(raster.stack) if need_feature_stack else None
    return Dataset(X=X[keep], y=y, coords=coords[keep], rows=rows[keep], cols=cols[keep],
                   feature_names=list(pipeline.feature_names), feature_stack=feature_stack)


# ---------------------------------------------------------------------------
# Diagnostika
# ---------------------------------------------------------------------------
def _vif_from_corr(corr):
    """VIF = diag(R^-1). Kichik ridge bilan barqarorlashtirilgan; singular bo'lsa pinv."""
    k = corr.shape[0]
    try:
        inv = np.linalg.inv(corr + _VIF_RIDGE * np.eye(k))
        if not np.all(np.isfinite(inv)):
            raise np.linalg.LinAlgError("inf")
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(corr)
    return np.maximum(np.diag(inv), 1.0)


def _vif_flag(v):
    if not np.isfinite(v):
        return "o'zgarmas"
    return "yuqori" if v >= 10 else ("o'rta" if v >= 5 else "past")


def _layer_stats(raster):
    rows = []
    for j, name in enumerate(raster.band_names):
        band = raster.stack[j]
        v = band[np.isfinite(band)]
        has = v.size > 0
        rows.append({"band": name, "valid_pct": round(100.0 * v.size / band.size, 2) if band.size else 0.0,
                     "min": float(v.min()) if has else np.nan, "max": float(v.max()) if has else np.nan,
                     "mean": float(v.mean(dtype=np.float64)) if has else np.nan,
                     "std": float(v.std(dtype=np.float64)) if has else np.nan,
                     "kind": "kategorik" if name in raster.categorical else "raqamli"})
    return pd.DataFrame(rows, columns=["band", "valid_pct", "min", "max", "mean", "std", "kind"])


def _feature_effects(dataset):
    """Har feature uchun musbat/fon farqi: o'rtachalar, Cohen d va bir o'zgaruvchili AUC (Mann-Whitney)."""
    X = dataset.X.astype(np.float64)
    pos = dataset.y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:      # bir sinfli qism: farq hisoblanmaydi
        nan = np.full(X.shape[1], np.nan)
        return pd.DataFrame({"feature": dataset.feature_names, "mean_pos": nan, "mean_neg": nan,
                             "cohen_d": nan, "auc": nan})
    mp, mn = X[pos].mean(axis=0), X[~pos].mean(axis=0)
    d = np.full(X.shape[1], np.nan)
    if n1 > 1 and n0 > 1:
        pooled = np.sqrt(((n1 - 1) * X[pos].var(axis=0, ddof=1) + (n0 - 1) * X[~pos].var(axis=0, ddof=1))
                         / (n1 + n0 - 2))
        np.divide(mp - mn, pooled, out=d, where=pooled > 0)
    auc = (rankdata(X, axis=0)[pos].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return pd.DataFrame({"feature": dataset.feature_names, "mean_pos": mp, "mean_neg": mn,
                         "cohen_d": d, "auc": auc})


def _dataset_summary(dataset):
    p = len(dataset.feature_names)
    epv = dataset.n_pos / p if p else float("nan")
    warns = []
    if p and epv < 10:
        warns.append(f"Musbat nuqtalar ({dataset.n_pos}) feature'lar soniga ({p}) nisbatan kam (EPV={epv:.1f} < 10): "
                     f"overfitting xavfi yuqori, feature'larni kamaytirish yoki ko'proq musbat nuqta tavsiya etiladi.")
    return {"n": dataset.n, "n_pos": dataset.n_pos, "n_neg": dataset.n_neg, "n_features": p,
            "epv": float(epv), "warnings": warns}


def data_diagnostics(raster, dataset=None, max_pixels=50000, seed=RANDOM_STATE):
    """
    Ma'lumotlar diagnostikasi.
      "layer_stats": DataFrame(band, valid_pct, min, max, mean, std, kind)
      "corr": DataFrame - Pearson korrelyatsiya (faqat raqamli bandlar)
      "vif": DataFrame(band, vif, flag) - ridge-barqaror VIF (flag: past/o'rta/yuqori/o'zgarmas)
      "high_corr_pairs": DataFrame(band_a, band_b, corr) - |r| >= 0.9
      "n_sample": korrelyatsiya/VIF uchun ishlatilgan piksellar soni (<= max_pixels)
    dataset berilsa qo'shimcha: "dataset_summary" (n, n_pos, n_neg, EPV, ogohlantirishlar) va
    "feature_effects" (musbat/fon farqi, bir o'zgaruvchili AUC).
    """
    max_pixels = int(max_pixels)
    if max_pixels < 2:
        raise ValueError("max_pixels kamida 2 bo'lishi kerak.")
    num_idx = [j for j, b in enumerate(raster.band_names) if b not in raster.categorical]
    names = [raster.band_names[j] for j in num_idx]
    rng = np.random.default_rng(int(seed) % (2 ** 32 - 1))

    sample = np.empty((0, len(num_idx)))
    if num_idx:
        valid = np.ones(raster.stack.shape[1:], dtype=bool)
        for j in num_idx:
            valid &= np.isfinite(raster.stack[j])
        flat = np.flatnonzero(valid)
        if flat.size > max_pixels:
            flat = np.sort(rng.choice(flat, size=max_pixels, replace=False))
        r, c = np.divmod(flat, raster.width)
        sample = raster.stack[np.asarray(num_idx)[:, None], r[None, :], c[None, :]].T.astype(np.float64)

    k = sample.shape[0]
    corr = np.full((len(names), len(names)), np.nan)
    const = np.ones(len(names), dtype=bool)
    if k >= 2:
        mean, std = sample.mean(axis=0), sample.std(axis=0)
        const = std <= _CONST_RTOL * np.abs(mean)       # nisbiy: qatlam masshtabiga bog'liq emas
        z = (sample - mean) / np.where(const, 1.0, std)
        corr = np.clip(z.T @ z / k, -1.0, 1.0)
        corr[const, :] = np.nan
        corr[:, const] = np.nan
        np.fill_diagonal(corr, np.where(const, np.nan, 1.0))
    ok = ~const
    enough = k >= 2 and k > int(ok.sum())        # k <= band soni => korrelyatsiya matritsasi singular
    vif = np.full(len(names), np.nan)
    if ok.any() and enough:
        vif[ok] = _vif_from_corr(corr[np.ix_(ok, ok)])
    little = "ma'lumot yetarli emas"
    flags = [little if k < 2 else ("o'zgarmas" if c else (_vif_flag(v) if enough else little))
             for v, c in zip(vif, const)]
    vif_df = pd.DataFrame({"band": names, "vif": vif, "flag": flags}, columns=["band", "vif", "flag"])

    pairs = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if np.isfinite(corr[a, b]) and abs(corr[a, b]) >= 0.9:
                pairs.append({"band_a": names[a], "band_b": names[b], "corr": float(corr[a, b])})
    pairs_df = pd.DataFrame(pairs, columns=["band_a", "band_b", "corr"])
    if len(pairs_df):
        pairs_df = pairs_df.reindex(pairs_df["corr"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    out = {"layer_stats": _layer_stats(raster), "corr": pd.DataFrame(corr, index=names, columns=names),
           "vif": vif_df, "high_corr_pairs": pairs_df, "n_sample": int(k)}
    if dataset is not None:
        out["dataset_summary"] = _dataset_summary(dataset)
        out["feature_effects"] = _feature_effects(dataset)
    return out
