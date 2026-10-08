# MPM ML GUI v2 — arxitektura va modul shartnomasi

Bu hujjat **yagona shartnoma**: har bir modul shu yerdagi imzo va ma'lumot sxemalariga rioya qiladi.
Hujjat haqiqiy kodga (v2.0.0) moslab yangilangan: shartnoma va kod farq qilsa, **kod haqiqat**.
Eski kod: `legacy/mpm_ml_gui_fixed.py` (faqat ma'lumot uchun; undan mantiqni olish mumkin, lekin
xatolarini ko'chirmang). Foydalanuvchi qo'llanmasi: [`USER_GUIDE.md`](USER_GUIDE.md), o'zgarishlar tarixi:
[`CHANGELOG.md`](CHANGELOG.md), legacy'dan nima tuzatilgani: [4-bo'lim](#4-ozgarishlar-jadvali-asl-koddan-legacy-nima-tuzatildi-va-nima-qoshildi).

## 0. Umumiy qoidalar

* Python >= 3.9 (`from __future__ import annotations`), scikit-learn >= 1.3 (sinov: 1.9.1), xgboost 2/3,
  TensorFlow >= 2.16 / Keras 3 (sinov: tensorflow-cpu 2.21, Keras 3.15; CNN `keras.utils.PyDataset` ishlatadi),
  shap >= 0.42 (sinov: 0.52), shapely 1.8/2.x, geopandas 0.14/1.x, rasterio >= 1.3.
  **Eskirgan API ishlatilmaydi**: `SVC(probability=...)` YO'Q (har doim `CalibratedClassifierCV`),
  `GeoSeries.unary_union` YO'Q (`union_all()`, zaxira: `unary_union`), sklearn `permutation_importance` YO'Q (o'zimiznikini yozamiz).
* `tensorflow`, `xgboost`, `shap`, `PyQt5` modul darajasida import QILINMAYDI (faqat `mpm.gui.*` va `mpm.workers`
  PyQt5 import qiladi). Ixtiyoriylar `mpm.common.get_tf()/get_xgboost()/get_shap()` orqali, funksiya ichida olinadi
  (muvaffaqiyatsiz import `None` sifatida keshlanadi). Buni `tests/test_cnn.py::test_tensorflow_not_imported_at_module_level`,
  `tests/test_explain.py::test_heavy_libs_not_imported_at_module_level`, `tests/test_plots.py::test_no_forbidden_imports`,
  `tests/test_data.py::test_module_rules` tekshiradi.
* `print` ishlatilmaydi: har bir funksiya `log_fn=None` oladi (None => `noop_log`), uzoq funksiyalar
  `progress_fn(frac:0..1, msg:str)=None` va `cancel=None` (`CancelToken`) oladi; tsikllarda `check_cancel(cancel)`.
* Tasodifiylik: har bir funksiya `seed`/`random_state` oladi; global holat yo'q. Reproduktivlik: bir xil
  ma'lumot + bir xil seed => bir xil natija (RF uchun `n_jobs` ham natijaga ta'sir qilmaydi: `models._DeterministicRF`).
* Hamma foydalanuvchiga ko'rinadigan matn, izoh va docstring **o'zbek tilida (lotin)**, ixcham (original kod uslubi).
* Terminologiya: chiqish "ehtimollik" emas, **"prospektivlik indeksi"** (pos:fon nisbati sun'iy). Fayl nomlari
  `prognoz_*.tif` saqlanadi, lekin legend/label/hujjatda "prospektivlik indeksi (0-1)" yoziladi
  (`plots.INDEX_LABEL`, `pipeline._NOTE_INDEX`, GeoTIFF band tavsifi, bundle `README.txt`, `summary.txt`).
* Test: `tests/test_<modul>.py`, `pytest`, tez bo'lishi shart (<60 s/fayl);
  TensorFlow/CNN/to'liq pipeline testlari `@pytest.mark.slow`. Sintetik ma'lumot: `tests/synth.py`
  (`make_synthetic_project`), fixture'lar: `tests/conftest.py` (`synth_project`, `synth_project_cat`;
  `QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg` o'rnatiladi).
* Ishga tushirish muhiti: `PY=/tmp/claude-0/-home-user-Samo/6fae25e6-7e1b-535c-aa07-7f13809fb8d9/scratchpad/venv/bin/python`
  (numpy, pandas, sklearn, scipy, rasterio, geopandas, shapely, xgboost, tensorflow-cpu, shap, PyQt5, matplotlib, pytest, openpyxl).
  Repo ildizidan: `cd /home/user/Samo && $PY -m pytest tests/test_x.py`. GUI: `QT_QPA_PLATFORM=offscreen`.
  Mashinada 4 CPU / 15 GB — og'ir testlarni (TF) kichik tuting (epochs<=3, kichik tarmoq), bir vaqtda ko'p
  parallel og'ir jarayon ishga tushirmang.
* Boshqa agent fayllariga TEGMANG: faqat o'zingizga ajratilgan fayllarni yozing. Boshqa modulning xatosini
  topsangiz — o'zgartirmasdan, yakuniy hisobotda aniq yozing (fayl, qator, sabab, taklif).

## 1. Tuzatilishi shart bo'lgan xatolar (BUG) — `legacy/` dagi asl kodga nisbatan

Qisqa jadval (qaror). Har birining aniq moduli, funksiyasi va testi — [4-bo'lim](#4-ozgarishlar-jadvali-asl-koddan-legacy-nima-tuzatildi-va-nima-qoshildi)da.

| ID | Muammo | Yechim (modul) |
|---|---|---|
| BUG-01 | yangi `shap` RF uchun `(n,p,2)` qaytaradi -> `draw_importance` TypeError -> PyQt5 slotida abort | `explain.normalize_shap_values` (3D/list/2D), `plots` himoyasi, GUI global `sys.excepthook` + slotlarda try/except |
| BUG-02 | `on_error` prognoz tugmasini qulflab qo'yadi | GUI: barcha tugmalar holati bitta `_set_busy(False)` orqali tiklanadi |
| BUG-03 | MDI zaxirasi `CalibratedClassifierCV.estimator` (o'qitilmagan) ni o'qiydi | `ModelWrapper.importance_mdi()` / `tree_model()` o'qitilgan bazaviy modeldan |
| BUG-04 | CNN `EarlyStopping(monitor="loss")` — train loss; CV 150 vs final 200 epoch | validatsiya ajratmasi (stratified) + `val_loss` + bir xil giperparametr; `clear_session` |
| BUG-05 | spatial fold'larda musbat nuqtasiz validation fold chiqadi | `spatial.stratified_group_splits` (StratifiedGroupKFold, har repeat yangi seed) |
| BUG-06 | `SVC(probability=...)` sklearn 1.11 da olib tashlanadi; `unary_union` eskirgan | `CalibratedClassifierCV(SVC())` har doim; `union_all()` |
| BUG-07 | CNN `predict` default batch=32 (28x sekin); RF/XGB bashorat n_jobs=1 | `batch_size=8192` va `n_jobs` |
| BUG-08 | CI faqat repeat'lar bo'yicha (sun'iy tor) | blok-bootstrap CI (`cv.compute_metrics`), repeat-std alohida |
| BUG-09 | variogram faqat 1-band (alifbo bo'yicha birinchi), kam nuqta | barcha raqamli bandlar medianasi / tanlangan band, ko'proq nuqta |
| BUG-10 | kategorik qatlamlar bilinear resample qilinadi | nearest + one-hot (`data.FeaturePipeline`) |
| BUG-11 | fon nuqtalar nodata joyga tushib tashlanadi (soni kamayadi); sekin sikl | valid-mask + vektorlashtirilgan generatsiya, kerakli soniga yetadi |
| BUG-12 | kalibrlangan "final" model aslida 3 ta 2/3-ma'lumotli model; SHAP faqat 1-ichki fold | `CalibratedClassifierCV(ensemble=False)`: bazaviy model to'liq train ma'lumotida |
| BUG-13 | musbat nuqta kam bo'lsa kalibrlash crash | `eff_cv` adaptiv; <2 bo'lsa kalibrlashsiz + ogohlantirish |
| BUG-14 | mayda: ishlatilmagan importlar, `nanmean` ogohlantirishi, bir pikseldagi dublikat musbatlar, MultiPoint, AOI tashqarisidagi nuqtalar, source profile (tiled/blocksize) bilan yozish, oyna yopilishi, `bg_sensitivity_repeats=3` qattiq, fold xaritasida legend yo'q, suptitle kesilishi, legend ustma-ust | tegishli modullarda |
| BUG-15 | giperparametrlar CV va final'da ikki joyda takror (drift xavfi) | yagona `hp` lug'ati (`config.PARAM_SPECS`) har ikkalasiga uzatiladi |

## 2. Qo'shiladigan imkoniyatlar (ENH)

| ID | Imkoniyat |
|---|---|
| ENH-01 | To'liq giperparametr paneli (barcha `PARAM_SPECS` avtomatik), RF/SVM ham yoqish-o'chirish, preset JSON saqlash/yuklash, standartga qaytarish, seed va n_jobs GUI'da, ishlatilgan giperparametrlar natija bilan JSON'ga yoziladi |
| ENH-02 | Giperparametr qidirish: nested spatial CV bilan tasodifiy qidiruv (GUI: yoqish, modellar, n_iter, ichki fold, scoring, oraliqlarni tahrirlash) |
| ENH-03 | Haqiqiy CNN: patch2d (WxW oyna, 2D-Conv, augmentatsiya, val-split early stopping), eski tabular1d saqlanadi |
| ENH-04 | GUI natijalar: metrikalar jadvali (AUC+CI, PR-AUC, BalAcc, F1, Brier, sens/spec @Youden), PR, kalibrlash (reliability), confusion matritsa, CSV/XLSX eksport, `NavigationToolbar`, barcha rasmlarni PNG/PDF saqlash |
| ENH-05 | Model saqlash/yuklash (bundle) va yangi maydonga qo'llash |
| ENH-06 | Stop tugmasi, fold bo'yicha progress + ETA, n_jobs, seed, log fayli, global excepthook, yopishda tasdiq |
| ENH-07 | Ma'lumotlar tahlili (layer stats, korrelyatsiya heatmap, VIF, data dictionary jadvali), kategorik qatlamlar, musbat nuqta dublikati nazorati, fon strategiyalari (random/grid/distance_weighted), yakuniy model uchun K-fon ansambli, xarita overlay (koordinatalar, AOI, nuqtalar), sinflangan xarita + maydon statistikasi, noaniqlik xaritasi, success-rate egri chizig'i, SHAP beeswarm/dependence |
| ENH-08 | "Prospektivlik indeksi" terminologiyasi + GUI/hujjatda izoh |
| ENH-09 | `requirements.txt`, avtomatik testlar, README |

## 3. Modul xaritasi va shartnomalar

```
mpm/
  common.py      konstantalar, CancelToken, sub_progress, lazy import, set_global_seed, safe_name, collect_versions
  base.py        ModelWrapper abstrakt sinfi (fit/predict_proba_pos/tree_model/importance_mdi/save/load)
  config.py      PARAM_SPECS, RunConfig, TuningConfig, validate_hyperparams, preset IO
  data.py        raster yuklash, FeaturePipeline, nuqtalar, fon, Dataset, patchlar, diagnostika
  spatial.py     bloklar, variogram, stratified group splits, bootstrap indekslari
  models.py      make_model(): RF/SVM/XGBoost wrapper'lari (+kalibrlash)
  cnn.py         CNNModel (patch2d / tabular1d)
  explain.py     OOF permutation importance, SHAP, MDI
  tuning.py      tasodifiy qidiruv (ichki spatial CV)
  cv.py          run_cv, compute_metrics, background sensitivity
  predict.py     xarita bashorati, sinflash, noaniqlik, success-rate, GeoTIFF saqlash
  persist.py     bundle saqlash/yuklash/qo'llash
  pipeline.py    run_training / run_prediction / export_results / estimate_cost_text
  workers.py     QThread ishchilar (cancel, progress)
  gui/ widgets.py plots.py param_panel.py result_tabs.py main_window.py
mpm_ml_gui.py    (ildizda) launcher
```

Modullar bog'liqligi (o'qlar "import qiladi" ma'nosida): `pipeline` -> {`cv`, `data`, `explain`, `models`, `predict`,
`spatial`, `config`}; `cv` -> {`models`, `spatial`, `tuning` (lazy), `explain` (lazy)}; `persist` -> {`data`, `predict`};
`workers` -> `pipeline`/`persist` (lazy, ishchi ichida); `gui.*` -> `config`, `common`, `explain` (faqat
`normalize_shap_values`). `mpm.gui.plots` Qt'ga bog'liq emas.

### 3.1 `mpm/data.py`

```python
@dataclass
class RasterStack:
    stack: np.ndarray        # (n_bands, H, W) float32, NaN = nodata/inf/sentinel
    band_names: list[str]    # fayl nomlari (kengaytmasiz), alifbo tartibida
    profile: dict            # TOZA GeoTIFF profili: driver="GTiff", dtype="float32", count=1, width, height,
                             # crs="EPSG:28411", transform, nodata=-9999.0, compress="lzw" (tiled/blocksize YO'Q)
    transform: "Affine"
    crs_epsg: int
    categorical: list[str]
    tech_metadata: list[dict]   # har band uchun: band_name, file_path, source_crs, source_resolution_x_m/y_m,
                                # source_width_px/height_px, source_bounds, source_nodata, source_dtype,
                                # reprojected_resolution_m, valid_pixels_pct, value_min/max/mean/std
    # xossalar: .n_bands .height .width .pixel_size (m)

def find_shapefile(folder) -> str|None        # papkadagi birinchi .shp (alifbo, katta-kichik harf farqsiz); yashirin fayllar o'tkaziladi
def find_tiff_files(folder) -> list[str]      # .tif/.tiff, alifbo tartibida (birinchisi = referens grid)
def suggest_categorical_layers(tiff_paths) -> list[str]
    # heuristika (<=1024x1024 ga kichraytirib o'qiydi): integer dtype va noyob qiymatlar <= 30, yoki float dtype,
    # barcha qiymatlar butun va noyob qiymatlar <= 20 -> band nomlari
def load_and_align_rasters(tiff_paths, categorical=(), log_fn=None, assume_crs_if_missing=True,
                           fallback_epsg=TARGET_EPSG, cancel=None) -> RasterStack
    # birinchi fayl = referens grid (EPSG:28411 ga). Kategorik: Resampling.nearest, qolgani bilinear.
    # nodata/inf/|x|>1e15 -> NaN. CRS yo'q fayllar fallback_epsg deb olinadi (ogohlantirish bilan) yoki
    # assume_crs_if_missing=False bo'lsa ValueError. Grid referensga mos bo'lsa resample QILINMAYDI.
    # Ko'p bandli faylda faqat 1-band (ogohlantirish); takroriy band nomi, georeferenssiz fayl => ValueError.
    # Faylni ikki marta o'qimaslik; yakuniy stack inf-siz.
def valid_pixel_mask(stack) -> (H,W) bool      # barcha bandlar chekli (ndarray yoki RasterStack)
MANUAL_METADATA_FIELDS = ["band_name", "source_owner", "survey_or_scene_id", "survey_date",
                          "original_scale_or_resolution", "transformation_applied", "notes"]
load_or_create_manual_metadata(tiff_folder, band_names, log_fn=None) -> (dict, path)
    # <tiff_folder>/metadata.csv topilsa o'qiydi: ajratgich (',' yoki ';') va kodlash (utf-8-sig, utf-8, cp1251, cp1252)
    # AVTOMATIK aniqlanadi (Excel mintaqaviy sozlamasi; cp1251/cp1252 noaniqligida so'z ichida kirill+lotin aralashgan
    # variant rad etiladi); noan'anaviy format log'da yoziladi. Topilmasa bo'sh SHABLONNI SHU (KIRISH TIFF) PAPKAGA
    # YOZADI (yon ta'sir!) va log'da buni aniq aytadi. Mavjud, lekin o'qib bo'lmaydigan (NUL baytli/bo'sh/jadval emas) yoki
    # 'band_name' ustunsiz fayl HECH QACHON ustiga yozilmaydi: aniq ogohlantirish + bo'sh metadata.
build_data_dictionary(band_names, tech_metadata, manual_metadata) -> DataFrame   # tech_metadata + qo'lda maydonlar
save_data_dictionary(df, out_dir, log_fn=None) -> path        # <out_dir>/predictor_data_dictionary.csv

class FeaturePipeline:
    def __init__(self, band_names, categorical=(), max_levels=30)
    def fit(self, stack) -> self          # kategorik bandlar uchun valid piksellardan noyob BUTUN darajalar;
                                          # butun bo'lmagan qiymat yoki > max_levels daraja => ValueError
    levels: dict[str, list[int]]          # kategorik band -> darajalar
    feature_names: list[str]              # raqamli band nomi; kategorik: f"{band}=={int(level)}"
    feature_bands: list[str]              # har feature qaysi banddan (one-hot ustunlarida band nomi takrorlanadi)
    n_features: int ; band_names ; categorical
    def transform_pixels(self, pix) -> (m, n_features) float32   # pix: (m, n_bands); noma'lum daraja => hammasi 0
    def transform_stack(self, stack) -> (n_features, H, W) float32   # NaN saqlanadi
    def to_dict(self) / @classmethod from_dict(d)                   # JSON-ga yaroqli (persist uchun)

def xy_to_rowcol(transform, x, y) -> (rows:int64[], cols:int64[])   # vektorlashtirilgan, floor, chegara tekshiruvisiz
def load_aoi(aoi_shp, assume_crs_if_missing=True, log_fn=None) -> GeoDataFrame            # EPSG:28411, faqat poligon
def load_positive_points(points_shp, aoi_gdf=None, assume_crs_if_missing=True, log_fn=None) -> GeoDataFrame
    # EPSG:28411; MultiPoint -> explode; faqat Point; AOI tashqarisidagilar ogohlantirish bilan tashlanadi
def dedupe_points_by_pixel(points_gdf, transform, shape, log_fn=None) -> GeoDataFrame        # bir pikseldagi dublikat musbatlar
def generate_background_points(aoi_gdf, positive_gdf, n_points, min_distance, random_state=RANDOM_STATE,
                               strategy="random", valid_mask=None, transform=None, max_attempts_factor=200,
                               log_fn=None) -> GeoDataFrame
    # strategy: "random" (AOI ∩ valid_mask ichida tekis), "grid" (AOI ichida jitterli to'r, keyin tanlab olinadi),
    # "distance_weighted" (musbat nuqtalardan masofaga proporsional ehtimol). min_distance dan yaqinlari rad etiladi
    # (cKDTree). valid_mask berilsa transform majburiy; transform berilganda har pikselga bittadan nuqta
    # tushadi va musbat piksellaridan tashqarida. Natija n_points ga yetadi (yetmasa ogohlantirish). Vektorlashtirilgan.
def sample_features(xy, raster_or_stack, transform=None) -> (values (n,n_bands) NaN tashqarida, rows, cols)
def extract_patches(feature_stack, rows, cols, window) -> (m, w, w, p) float32   # NaN bilan to'ldirilgan chegara

@dataclass
class Dataset:
    X: np.ndarray            # (n, n_features) float32
    y: np.ndarray            # (n,) int8   (musbatlar BIRINCHI, keyin fon)
    coords: np.ndarray       # (n, 2) float64
    rows: np.ndarray ; cols: np.ndarray   # int64 piksel indekslari
    feature_names: list[str]
    feature_stack: np.ndarray|None        # (n_features,H,W) — faqat patch kerak bo'lsa
    def subset(self, idx) -> "Dataset"    # feature_stack'ni nusxalamaydi (ulashadi)
    def get_patches(self, window) -> (n, w, w, n_features) float32    # NaN-padded (chegara/nodata NaN)
    @property n_pos, n_neg, n

def build_dataset(raster: RasterStack, pipeline: FeaturePipeline, positive_gdf, background_gdf,
                  log_fn=None, need_feature_stack=False) -> Dataset
    # yaroqsiz (chekli bo'lmagan) qatorlarni tashlaydi va nechtasi tashlanganini loglaydi
def data_diagnostics(raster: RasterStack, dataset: Dataset|None = None, max_pixels=50000,
                     seed=RANDOM_STATE) -> dict
    # {"layer_stats": DataFrame(band, valid_pct, min, max, mean, std, kind: "raqamli"|"kategorik"),
    #  "corr": DataFrame (pearson, faqat raqamli bandlar), "vif": DataFrame(band, vif, flag: past|o'rta|yuqori|o'zgarmas|
    #  "ma'lumot yetarli emas"), "high_corr_pairs": DataFrame(band_a, band_b, corr) (|r|>=0.9), "n_sample": int,
    #  dataset berilsa qo'shimcha: "dataset_summary" {n, n_pos, n_neg, n_features, epv, warnings (EPV<10)},
    #  "feature_effects" DataFrame(feature, mean_pos, mean_neg, cohen_d, auc)}
```

### 3.2 `mpm/spatial.py`

```python
def estimate_autocorrelation_range(stack, transform, band_indices=None, n_lags=15, n_points=1500,
                                   random_state=RANDOM_STATE, log_fn=None) -> float|None
    # band_indices None => barcha bandlar; har biri uchun n_points (50..4000) tasodifiy valid piksel, empirik semivariogram
    # (maks. lag = hudud diagonalining 1/3), sferik model range'i; natija MEDIANA (metr). Fazoviy strukturasiz
    # (shovqin/o'zgarmas) bandlar e'tiborga olinmaydi. Hech biri baholanmasa None (pipeline zaxira 1000 m ishlatadi).
def assign_spatial_blocks(coords, block_size) -> int64 ndarray (n,)     # kvadrat to'r kataklari, zich raqamlangan
def adapt_block_size(coords, y, n_splits, block_size, min_size=10.0, log_fn=None) -> (block_size, groups)
    # n_blocks >= n_splits VA musbatlar kamida n_splits ta blokda bo'lguncha yarmiga kamaytiradi (max 30 marta,
    # min_size dan kichik emas); musbat nuqtalar < n_splits yoki fon yo'q bo'lsa ham ValueError(aniq xabar bilan)
def stratified_group_splits(y, groups, n_splits, n_repeats, random_state=RANDOM_STATE, log_fn=None, max_attempts=50)
    # generator: (repeat, fold, train_idx, val_idx). Har repeat'da yangi seed (random_state + repeat*1000 + urinish) bilan
    # StratifiedGroupKFold(shuffle=True). Har bir val fold'da >=1 musbat bo'lishi kerak: yetmasa qayta urinadi (max_attempts),
    # keyin musbat blokni (>=2 fold'dagidan) butunlay musbatsiz fold'ga ko'chiradi, bo'lmasa ogohlantirib davom etadi.
    # Bitta blok HECH QACHON train va val orasida bo'linmaydi. Argumentlar chaqiruv paytidayoq tekshiriladi.
def random_stratified_splits(y, n_splits, n_repeats, random_state=RANDOM_STATE)   # xuddi shu format (RepeatedStratifiedKFold)
def block_bootstrap_indices(groups, n_boot, random_state=RANDOM_STATE)
    # generator: har bir bootstrap uchun nuqta indekslari (bloklar almashtirish bilan tanlanadi)
def fold_report(y, coords, groups, splits) -> DataFrame(repeat, fold, n_train, n_val, pos_train, pos_val, n_blocks_val)
```

### 3.3 `mpm/models.py` va `mpm/cnn.py`

```python
# models.py
def make_model(name, params, *, calibrate=True, calibration_method="sigmoid", calibration_cv=3,
               n_jobs=1, seed=RANDOM_STATE, n_features=None) -> ModelWrapper
    # name in MODEL_NAMES. "CNN" => cnn.CNNModel (lazy import). Mavjud bo'lmagan kutubxona => RuntimeError(aniq xabar).
def normalize_params(model, params) -> dict       # to'liq, tur jihatidan tozalangan; noma'lum kalit => ValueError
class SklearnModel(ModelWrapper)         # RF / SVM / XGBoost; kirish: XOM feature'lar (NaN/inf => ValueError)
    # RF: _DeterministicRF (n_jobs'dan mustaqil, bit-darajasida reproduktiv); SVM: Pipeline(StandardScaler, SVC(...)) —
    # `probability` parametri BERILMAYDI; XGB: raw, scale_pos_weight=None => neg/pos (fit paytida y'dan). params ->
    # sklearn kwargs xaritalash: "none"->None, max_features "all"->None, SVM gamma None->"scale",
    # RF max_samples faqat bootstrap=True bilan.
    # calibrate=True: CalibratedClassifierCV(base, method, cv=StratifiedKFold(eff_cv, shuffle=True), ensemble=False).
    # eff_cv = min(calibration_cv, n_pos, n_neg); <2 bo'lsa (yoki kalibrlash xato bersa) kalibrlashsiz
    #   (fit_info["calibration"]="skipped" + "calibration_reason"; SVM uchun zaxira _SigmoidMinMaxSVM,
    #   fit_info["svm_fallback"]="sigmoid_minmax" - kalibrlangan ehtimollik EMAS, monoton indeks).
    # calibrate=False va SVM: baribir CalibratedClassifierCV(SVC, sigmoid, ensemble=False) (probabilistik chiqish kerak) —
    #   fit_info["calibration"]="svm_platt_only".
    # fit_info: {"n_train","n_pos","n_neg","calibration","calibration_cv","calibration_reason", ["scale_pos_weight"]}
    # tree_model(): o'qitilgan RF/XGB (kalibrlangan bo'lsa calibrated_classifiers_[0].estimator), SVM => None.
    # importance_mdi(): feature_importances_ (p,) yoki None.
# cnn.py
class CNNModel(ModelWrapper)             # __init__(params, seed=RANDOM_STATE, n_jobs=1) (n_jobs faqat meta'ga yoziladi)
    # input_kind = "patch" (mode=patch2d) yoki "tabular" (mode=tabular1d)
    # patch2d: kirish (n,w,w,p); per-kanal standartlashtirish (train statistikasi, NaN->0 standartlashdan KEYIN);
    #   arxitektura: Conv2D(filters1,k) -> [Conv2D(filters2,k) agar >0] -> GlobalAveragePooling2D -> Dense(dense_units)
    #   -> Dropout -> Dense(1, sigmoid). Augmentatsiya: tasodifiy flip/rot90 (faqat train, params["augment"]).
    # tabular1d: eski usul (Conv1D, feature o'qi bo'ylab) — taqqoslash uchun.
    # O'qitish: stratified val_fraction ajratmasi, EarlyStopping(monitor="val_loss", patience, restore_best_weights),
    #   musbat sinf neg/pos namuna og'irligi bilan (CNN chiqishi KALIBRLANMAYDI), cancel tokeni Keras callback orqali
    #   tekshiriladi. Qurishdan oldin keras.backend.clear_session().
    #   Agar val ajratib bo'lmasa (har sinfdan >=2 namuna kerak) -> train loss bo'yicha ogohlantirish bilan.
    # predict_proba_pos(batch_size=8192) — model.predict(batch_size=batch_size).
    # fit_info: {"epochs_run","best_epoch","val_loss","train_loss","n_train","n_val","early_stopping": "val_loss"|"train_loss",
    #            "class_weight_pos"}
    # save/load: Keras `model.keras` + `meta.json` (standartlashtirish parametrlari, params, fit_info); joblib EMAS.
```

### 3.4 `mpm/explain.py`, `mpm/tuning.py`

```python
# explain.py
def permutation_importance_auc(model, X, y, *, patches=None, n_repeats=5, seed=RANDOM_STATE, batch_size=8192,
                               cancel=None, progress_fn=None) -> ndarray (p,)
    # har feature j uchun: ustunni (patch modelida — kanalni barcha patchlarda bir xil permutatsiya bilan) aralashtiradi,
    # AUC pasayishining o'rtachasi. y bir sinfli bo'lsa NaN massiv qaytaradi (xato emas).
def summarize_perm_importance(per_fold: list[ndarray]) -> {"mean","std","per_fold","n_folds","n_valid_folds"}   # nanmean/nanstd (ddof=0)
def normalize_shap_values(sv, class_index=1, *, n_samples=None, n_features=None) -> ndarray (n, p)
    # list / (n,p) / (n,p,2) / (2,n,p) / Explanation hammasini qamraydi
def compute_shap_summary(final_models, X, feature_names, max_background=50, seed=RANDOM_STATE, log_fn=None, cancel=None)
    -> {model: {"mean_abs_shap": (p,), "shap_values": (k,p), "X_background": (k,p), "feature_names": [...],
                "expected_value": float|None, "units": "ehtimollik"|"log-odds"}} | None
    # faqat RF/XGBoost; final_models: {name: [ModelWrapper,...]} (faqat BIRINCHI bg-draw modeli), tree_model() orqali;
    # fon: X dan seed bo'yicha tasodifiy min(max_background, n) qator (hamma model uchun bir xil).
    # shap yo'q bo'lsa None + log. Barcha shape'lar normalizatsiya qilinadi. RF birligi - ehtimollik, XGBoost - log-odds.
def mdi_importance(model) -> ndarray|None
# tuning.py
def sample_params(model, base_params, space, rng) -> dict        # space = TuningConfig.resolved_space(model); tur/diapazon/log hurmat
def tune_model(name, base_params, dataset, groups, *, tuning, calibrate, calibration_method, calibration_cv,
               n_jobs, seed, log_fn=None, cancel=None, progress_fn=None) -> dict
    # dataset/groups = TRAIN to'plami. Ichki: stratified_group_splits(n_splits=tuning.inner_splits, n_repeats=1);
    # imkonsiz bo'lsa fold soni 2 gacha kamayadi. n_iter ta nomzod: 1-nomzod = base_params o'zi (teng ballda base afzal),
    # qolganlari tasodifiy (takrorsiz). Ball: tuning.scoring ("roc_auc" | "average_precision"), nomzodlar bir xil fold'larda.
    # Qaytaradi: {"best_params": dict (TO'LIQ param lug'ati), "best_score": float, "trials": [{"params","score","std","n_folds"}]
    #   (trials[0] = base), "n_trials": int, "scoring", "base_score", "best_index", "inner_splits_used"}
    # Imkonsiz holatda (ichki bo'linish, noto'g'ri oraliq, patch2d uchun feature_stack yo'q): best_params=base, best_score=NaN,
    #   trials=[], n_trials=0 + "skipped_reason" (hech qachon yiqilmaydi). CNN window o'zgarsa patchlar window bo'yicha
    #   guruhlanib qayta olinadi.
```

### 3.5 `mpm/cv.py`

```python
def run_cv(dataset, groups, cv_mode, model_names, hp, *, n_splits, n_repeats, calibrate=True,
           calibration_method="sigmoid", calibration_cv=3, tuning=None, perm_importance=False,
           perm_repeats=5, n_jobs=1, seed=RANDOM_STATE, log_fn=None, progress_fn=None, cancel=None) -> dict
    # cv_mode "spatial" (stratified_group_splits; groups majburiy) | "random" (random_stratified_splits).
    # tuning: TuningConfig|dict|None; enabled VA mode=="nested" bo'lsa har tashqi fold'da tunable va belgilangan modellar
    #   uchun tune_model(train qismi) - qidiruv ichida kalibrlash O'CHIRILADI (ball tartibga bog'liq, ~4-5x tez), fold modeli
    #   esa calibrate bilan o'qitiladi. mode != "nested" bo'lsa fold ichida tuning QILINMAYDI (log'da yoziladi).
    # Qaytaradi: {"mode", "oof": {name: [ (n,) float64 NaN-siz massivlar, har repeat uchun ]},
    #   "fold_map": (n,) int (1-repeat, fold indeksi 0 dan),
    #   "fold_table": DataFrame (repeat, fold, n_train, n_val, pos_train, pos_val, n_blocks_val, seconds_<model>...),
    #   "perm_importance": {name: summarize_perm_importance(...)} (faqat spatial, faqat 1-repeat, perm_importance=True bo'lsa;
    #       har fold'ning val qismida, barcha modellar uchun - SVM va patch-CNN ham),
    #   "tuned_params": {name: [rec, ...]} (tashqi fold'lar bo'yicha; faqat tuning qilingan modellar),
    #   "feature_names", "model_names", "hp": {name: bazaviy hp}, "n_splits", "n_repeats", "warnings": [str]}
    # tuned_params yozuvi (rec) = tune_model natijasi + {"repeat","fold"}: "best_params" (to'liq hp), "best_score", "base_score",
    #   "scoring", "trials": [{"params","score","std","n_folds"}], "n_trials", "best_index", "inner_splits_used".
    #   Zaxira yozuv (tuning bajarilmadi): "best_params"=bazaviy, "best_score"=NaN, "trials"=[], "n_trials"=0 va
    #   "fallback": sabab matni (ValueError) yoki "skipped_reason" (tune_model o'zi qaytargan).
    # Progress: har fold'dan keyin (frac, "Fold i/N ... ETA mm:ss"); tuning bo'lsa fold ichida ham. Seed har fold uchun seed+split_num.
    # Model xatosi jim yutilmaydi: aniq log + RuntimeError.
def compute_metrics(y, oof, groups=None, n_boot=1000, seed=RANDOM_STATE, log_fn=None) -> (dict, ensemble_mean_proba)
    # kalitlar: har model + ENSEMBLE_NAME ("Ensemble (soft-voting)": har repeat ichida modellar oddiy o'rtachasi). Har biri uchun:
    #  "auc"(repeat-o'rtacha), "auc_std"(repeat; n_repeats==1 bo'lsa NaN - 0.0 EMAS, barcha "*_std" kalitlari ham), "auc_ci95"(blok-bootstrap, (lo,hi); n_boot=0 => (nan,nan)), "auc_single"
    #  (o'rtacha OOF ustida), "auc_repeats" (har repeat AUC massivi), "pr_auc","balanced_accuracy","f1","brier",
    #  "balanced_accuracy_youden","f1_youden" va ularning "_std", "_ci95" variantlari,
    #  "threshold_youden", "sensitivity","specificity" (Youden bo'sag'ida), "confusion" ([[tn,fp],[fn,tp]]),
    #  "fpr","tpr","precision","recall" (massivlar), "calibration": {"prob_pred","prob_true","counts"} (kvantil bin'lar;
    #  bin soni = clip(n_pos // 5, 3, 10)), "mean_proba" (n,).
    #  balanced_accuracy/f1 bo'sag'i 0.5; "*_youden" - o'rtacha OOF'dan topilgan Youden bo'sag'i.
    #  Eslatma: "auc" repeat-o'rtacha, CI esa o'rtacha bashorat ustida bootstrap: nuqta qiymat CI chetiga yaqin (yoki
    #  tashqarisida) chiqishi mumkin.
def metrics_dataframe(results, label="") -> DataFrame   # bir qator har model: [CV,] Model, AUC, AUC_std (1 repeat: NaN), AUC_CI_lo, AUC_CI_hi, PR_AUC, BalAcc, F1, Brier, Sens, Spec, Thr
    # NaN: CSV/XLSX'da bo'sh katak, summary.txt'da "(std: -, 1 takror)", GUI jadvalida bo'sh katak, grafiklarda xato chizig'i yo'q.
def youden_threshold(y, p) -> float
def run_background_sensitivity(aoi_gdf, positive_gdf, raster, pipeline, hp, *, model_names, n_background, min_distance,
        strategy, block_size, n_draws, n_splits, n_repeats, calibrate, calibration_method, calibration_cv,
        n_jobs, seed, log_fn=None, progress_fn=None, cancel=None, feature_stack=None) -> dict|None
    # har draw: seed + 1000 + draw; bazaviy hp (tuning YO'Q), spatial CV, n_boot=0. Imkonsiz draw o'tkazib yuboriladi.
    # feature_stack (ixtiyoriy, CNN patch2d): tayyor (n_features,H,W) stek (pipeline dataset.feature_stack'ni uzatadi); berilmasa
    #   bir marta quriladi; har holda barcha draw'lar orasida ULASHILADI (stek nuqtalarga bog'liq emas) - qayta qurilmaydi.
    # {"per_draw": [{"draw","seed","auc":{name:auc},"n_background","block_size"}], "summary": {name:{"mean","std","min","max","values"}},
    #  "n_positive", "n_draws", "n_draws_requested"}; hech draw bajarilmasa None.
```

### 3.6 `mpm/predict.py`

```python
def predict_probability_maps(raster, pipeline, final_models, *, batch_size=8192, log_fn=None, progress_fn=None,
                             cancel=None) -> dict
    # final_models: {name: [ModelWrapper,...]}. Har model xaritasi = draw'lar bo'yicha o'rtacha. Faqat valid piksellar.
    # Tabular modellar: pipeline.transform_pixels(batch); patch modellar: transform_stack bir marta + extract_patches(batch).
    # {"maps": {name: (H,W) float32 NaN-nodata, ENSEMBLE_NAME ham shu yerda (nanmean, ogohlantirishsiz)},
    #  "uncertainty": (H,W) float32|None (ALOHIDA MODELLAR xaritalari std'i - modellar kelishmovchiligi; <2 model bo'lsa None),
    #  "valid_mask": (H,W) bool}
def classify_map(prob_map, method="quantile", n_classes=5, breaks=None) -> (class_map int8 (0=nodata, 1..n), breaks_used list)
    # quantile (teng maydonli), equal_interval (0-1 teng), fixed (breaks: o'suvchi chegaralar, n = len+1); n <= 127
CLASS_LABELS(n) -> ["Juda past","Past","O'rta","Yuqori","Juda yuqori"] (n=5) yoki "1-sinf".. 
def class_area_stats(class_map, pixel_size, rows=None, cols=None, n_classes=None) -> DataFrame
    # ustunlar: [Sinf, Nomi, Piksel, Maydon_km2, Maydon_%, Konlar_soni, Konlar_%, Boyitish]
    # rows/cols (konlar piksel indekslari) berilmasa Konlar_soni, Konlar_%, Boyitish = NaN (GUI'da bo'sh katak).
    # Boyitish = Konlar_% / Maydon_% (sinf konlarni tasodifiy taqsimotdan necha marta ko'p ushlaydi; 1 = tasodifiy).
def success_rate_curve(prob_map, rows, cols) -> {"area_frac": arr, "capture_frac": arr, "auc": float, "n_pos": int}
    # piksellar indeks bo'yicha kamayish tartibida; musbat nuqtalarning (rows, cols) piksel qiymatlari bo'yicha ushlangan ulush.
    # Nuqtalar O'QITISH nuqtalari => optimistik baho (mustaqil emas). Kon yoki valid piksel yo'q bo'lsa bo'sh massivlar, auc=NaN.
def save_rasters(out_dir, maps, profile, class_map=None, uncertainty=None, log_fn=None) -> list[paths]
    # prognoz_{safe_name}.tif (float32, nodata -9999), prognoz_classes.tif (int8, nodata 0), prognoz_uncertainty.tif;
    # band tavsifida "Prospektivlik indeksi" yoziladi.
```

### 3.7 `mpm/persist.py`

```python
def save_bundle(directory, *, final_models, pipeline, hyperparams_used, cfg_dict, metrics_summary=None,
                thresholds=None, block_size=None, crs_epsg=TARGET_EPSG, notes="", log_fn=None) -> directory
    # directory/manifest.json (OXIRIDA yoziladi), pipeline.json, models/<safe_name>/<draw>/ (wrapper.save), README.txt
    # manifest kalitlari: bundle_version(=1), created_utc, mpm_version, versions, band_names, categorical, feature_names, n_features,
    #   model_names, models {name: {n_draws, dir, class, input_kind, loader: "joblib"|"keras", params}}, hyperparams_used, cfg,
    #   metrics_summary, thresholds, block_size, crs_epsg, ensemble_name, notes, security_warning.
    # directory mavjud FAYL bo'lsa (yoki yo'lning biror qismi fayl) "[Errno 17] File exists" o'rniga aniq ValueError ("... papka emas").
    # Mavjud bundle ustiga yozilsa eski models/ o'chiriladi; MPM bundle bo'lmagan papkaga yozishdan bosh tortadi (ValueError);
    # yarim saqlangan bundle (manifest yo'q) yuklanmaydi.
def load_bundle(directory, log_fn=None) -> dict   # {"final_models","pipeline","manifest","cfg","hyperparams_used","metrics_summary",
                                                  #  "thresholds","block_size","crs_epsg","notes","directory"}
    # versiyalar (bundle_version, kutubxona versiyalari) tekshiriladi/ogohlantiriladi; log'da "faqat ishonchli manbadagi bundle'ni
    # yuklang" eslatmasi; buzilgan fayl => tushunarli xato.
def apply_bundle(bundle, tiff_folder, *, out_dir=None, batch_size=8192, assume_crs_if_missing=True,
                 class_method="quantile", n_classes=5, class_breaks=None, log_fn=None, progress_fn=None, cancel=None) -> dict
    # bundle: load_bundle natijasi yoki papka yo'li. Yangi papkadagi TIFF'larni band nomi bo'yicha (katta-kichik harf farqsiz)
    # moslab, referens grid = tanlangan fayllarning birinchisi (alifbo); band tartibi bundle'niki; yetishmaydigan band =>
    # ValueError (nomlar ro'yxati bilan), ortiqcha TIFF'lar e'tiborsiz. Kategorik darajalar o'qitishdagidan farq qilsa ogohlantiradi.
    # Qaytaradi: predict_probability_maps natijasi ("maps","uncertainty","valid_mask") + "class_map","class_breaks",
    #   "class_stats" (konlarsiz), "raster" (RasterStack), "saved_paths", "matched_files", "ignored_files".
```
Xavfsizlik: joblib/pickle — faqat ishonchli manbadan; `load_bundle` log'ga ogohlantirish yozadi, GUI esa yuklashdan oldin
tasdiq dialogini ko'rsatadi. CNN `.keras` formatida saqlanadi (joblib emas).

### 3.8 `mpm/pipeline.py`

```python
def run_training(cfg: RunConfig, *, log_fn=None, progress_fn=None, cancel=None) -> dict      # TrainingResult
def run_prediction(result, *, out_dir=None, class_method=None, n_classes=None, class_breaks=None, batch_size=8192,
                   log_fn=None, progress_fn=None, cancel=None) -> dict                       # PredictionOutput
def export_results(result, out_dir, prediction=None, log_fn=None) -> list[paths]
def estimate_cost_text(cfg: RunConfig|dict) -> str   # taxminiy o'qitish/ichki fit soni va ogohlantirishlar (GUI cost hint); hech qachon xato bermaydi
```
`cfg` dict ham qabul qilinadi (`RunConfig.from_dict`); `cfg.validate()` muammolari => `ValueError` (ro'yxat bilan).
Bekor qilinsa xom `CancelledError` ko'tariladi (bekor qilish qarori uchun "Pipeline tartibi"dan keyingi izohga qarang).

**TrainingResult** kalitlari:
`cfg(dict), band_names, feature_names, categorical_layers, raster(RasterStack), pipeline(FeaturePipeline),
dataset(Dataset), groups, block_size, n_positive, n_background, spatial(CVBlock), random(CVBlock|None; random CV o'chiq bo'lsa None),
bg_sensitivity(dict|None), final_models({name:[ModelWrapper]}), final_hyperparams({name:[dict]}),
importance({"method", "models": {name: {"mean","std","source": "permutation"|"mdi", ...}}, "feature_names"}),
shap({name:{...}}|None), data_dictionary(DataFrame), metadata_csv_path, diagnostics(dict), thresholds({name: float}),
versions(dict), timings({stage: sec}), log_path(str|None), warnings([str]), aoi_gdf, positive_gdf, background_gdf`
va shartnomadan tashqari qo'shimcha kalitlar: `model_names` (haqiqatda ishlatilgan modellar; mavjud bo'lmagan
kutubxonalar chiqarib tashlanadi), `final_tuning` ({name: tune_model yozuvi}, faqat yakuniy tuning qilingan
modellar uchun), `pixel_size` (m).
`importance["models"]` bo'sh bo'lishi mumkin (faqat-SVM yoki permutation o'chiq holatda; `method="hisoblanmadi"`).
MDI/gain zaxirasida `std` nollar va `source="mdi"` (xato chiziqlari chizilmaydi). `timings` kalitlari: `data`, `diagnostics`,
`blocks`, `random_cv` (yoqilgan bo'lsa), `spatial_cv`, `bg_sensitivity` (yoqilgan bo'lsa), `final_fit`, `importance`, `total`.
**CVBlock**: `{"mode","oof","metrics"(compute_metrics natijasi),"metrics_df","ensemble_oof","fold_map","fold_table",
"perm_importance","tuned_params","feature_names"}` + qo'shimcha `"hp"`, `"n_splits"`, `"n_repeats"`.
**PredictionOutput**: `{"maps","uncertainty","valid_mask","class_map","class_breaks","class_stats"(DataFrame),
"success_curve","saved_paths"}` + qo'shimcha `"class_method"`, `"n_classes"`. `out_dir` berilsa GeoTIFF'lar va
`predictor_data_dictionary.csv` saqlanadi.

Tuning semantikasi (MUHIM): `cfg.tuning.mode="nested"` bo'lsa spatial CV har tashqi fold'ning train qismida tuning qiladi
(baholash tuning protsedurasi bilan, optimistik emas). **Yakuniy tuning HAR DOIM butun ma'lumotda** bajariladi (nested
ham, `mode="final"` ham) — faqat 0-fon tanlovida (CV dataset'i), qolgan fon tanlovlari shu giperparametrlarni ishlatadi.
`mode="final"` bo'lsa spatial CV bazaviy hp bilan baholanadi (nested tuning yo'q): GUI faqat `nested` rejimini ko'rsatadi
(`TuningGroup` rejimni o'zgartirmaydi, preset'dan kelgan qiymatni saqlaydi). Random CV har doim bazaviy hp bilan, tuning va
importance'siz (faqat benchmark). `mode != "nested"` va tuning ishlaydigan model bor bo'lsa `result["warnings"]` ga aniq
ogohlantirish yoziladi ("CV metrikalari bazaviy giperparametrlar bilan, yakuniy model tuned hp bilan"); `summary.txt`
"Random vs Spatial" bo'limida ham izoh bor (nested: random bazaviy, spatial tuning bilan - "optimizm" qisman tuning farqi;
final: ikkala CV bazaviy hp bilan). `summary.txt` da `best_score` `{:.4f}`, `best_params` `{:.4g}` formatida.

Pipeline tartibi (progress oralig'i, monoton 0.0 -> 1.0): rasterlar (0-6%) + metadata (8%) -> nuqtalar/fon/dataset (12%) ->
diagnostika (14%) -> blok o'lchami (16%) -> random CV (16-36%, o'chirilishi mumkin) -> spatial CV (36-70%) -> bg sensitivity
(70-80%, ixtiyoriy) -> yakuniy modellar (80-92%; K fon tanlovi x model, yakuniy tuning shu oraliqda) -> importance (92-95%) ->
SHAP (95-98%) -> yakun (100%). **Bekor qilish qarori:** `check_cancel` importance'ning har modelida, importance -> SHAP
orasida va SHAP ichida chaqiriladi (Stop SHAP bosqichigacha, shu jumladan SHAP vaqtida ham ishlaydi => `CancelledError`).
SHAP tugagach yakunlash (bo'sag'lar, vaqtlar, natija lug'ati; millisekundlar) BEKOR QILINMAYDI: ishning ~99% bajarilgan, shu
paytda Stop bosilgan bo'lsa ham tayyor natija qaytariladi (`finished_signal` kelsa ham Stop bosilgan bo'lishi mumkin) -
natija yo'qotilmaydi. Patch-CNN uchun blok o'lchami < oyna*piksel bo'lsa
ogohlantirish (patchlar train/val orasida ustma-ust tushadi). Barcha ogohlantirishlar (`"Ogohlantirish:"` qatorlari)
`result["warnings"]` ga ham yig'iladi (<= 300 ta). Log fayli: `cfg.output_dir` berilgan bo'lsa `<output_dir>/mpm_run_<n>.log` (UTF-8, ketma-ket n).
CNN va boshqa modellar birga bo'lsa ansambl oddiy o'rtacha: CNN kalibrlanmaydi va neg/pos og'irlik bilan o'qitiladi, shuning uchun
shkalalar farq qilishi mumkin (log va `summary.txt` da ogohlantiriladi).

`export_results(result, out_dir, prediction=None)` yozadigan fayllar: `metrics_spatial.csv`, `metrics_random.csv` (random CV bo'lsa),
`metrics_spatial.xlsx` (openpyxl bo'lsa; varaqlar: "Spatial CV", "Random CV", "Sinflar"), `hyperparameters_used.json`
({"base","final","tuning","models"}), `run_config.json`, `importance.csv` (model, feature, mean, std, source), `tuned_params.csv` va
`tuning_trials.csv` (tuning bo'lsa; "nested" va "final" yozuvlar), `fold_table.csv`, `oof_predictions.csv`,
`predictor_data_dictionary.csv`, `versions.json`, `bg_sensitivity.json` (bo'lsa), `class_stats.csv` (prediction bo'lsa), `summary.txt`.

### 3.9 `mpm/workers.py` (PyQt5)

`TrainingWorker(cfg)`, `PredictionWorker(result, out_dir, class_method=None, n_classes=None, class_breaks=None, batch_size=8192)`,
`ApplyBundleWorker(bundle_dir, tiff_folder, out_dir=None, class_method="quantile", n_classes=5, class_breaks=None, batch_size=8192,
assume_crs_if_missing=True)`, `ExportWorker(result, out_dir, prediction=None)`. Hammasi `QThread` merosxo'ri (`_BaseWorker`);
`cancel()` (CancelToken), `cancel_token`, `is_cancelled`; signal'lar:
`log_signal(str)`, `progress_signal(int 0-100, str)`, `finished_signal(object)` (**dict**; `pyqtSignal(dict)` emas — katta
massivlar/RasterStack nusxalanib xotira ikki baravar bo'lmasligi uchun), `error_signal(str)` (xabar + traceback), `cancelled_signal()`.
`run()` ichida `try/except CancelledError -> cancelled_signal`, qolgan xatolar `error_signal`. `run()` sinxron chaqirilganda ham
ishlaydi (testlar). Pipeline/persist modullari ishchi ichida lazy import qilinadi.
`ApplyBundleWorker` natijasidan katta `"raster"` olib tashlanadi; o'rniga `"transform"`, `"profile"`, `"pixel_size"`, `"crs_epsg"`,
`"band_names"`, `"manifest"`, `"bundle_dir"` qo'shiladi. `ExportWorker` natijasi: `{"paths": [...], "out_dir": ...}`.

### 3.10 `mpm/gui/*`

* `widgets.py`: `FolderPicker(label_text="", parent=None, label_width=220)` (`.path()`, `.setPath()`, signal `changed(str)`);
  `MplCanvas(parent=None, figsize=(6,4.5), dpi=100, with_toolbar=True)` (`.fig` (constrained layout), `.canvas`, `.toolbar`
  (`NavigationToolbar2QT`), `.redraw()` (SINXRON; xato yutiladi, `.last_error`), `.clear()`, `.save_figure(path, dpi=300)`; minimal o'lcham juda
  kichik (200x150)). `.canvas` — `_PhasedCanvas(FigureCanvasQTAgg)`: `draw_idle()` (oyna/tab o'lchami o'zgarishi, zoom/pan) ikki
  hodisa-tsikl bosqichiga bo'linadi (1 layout, 2 rasterlash; `PHASE_PAUSE_MS` pauza), layout dvigateli vaqtincha o'chirilib
  har doim tiklanadi (~0.6 s qotish o'rniga ~0.2-0.3 s bo'laklar);
  `elide_text(text, limit=150)` (oxiriga `...`); `ProgressPanel.stopped(msg, tooltip=None)` (qisqartirilgan matnning to'liq varianti tooltip'da);
  `DataFrameTable(parent=None, max_rows=20000)` (`.set_dataframe(df, float_fmt="{:.3f}"|dict)`, NaN -> bo'sh katak, son/matn bo'yicha
  saralash, Ctrl+C TSV nusxalash, `.save_csv(path=None)` utf-8-sig, to'liq df); `ProgressPanel` (bar + bosqich matni + o'tgan vaqt/ETA;
  `.update_progress(frac|0..100, msg)`, `.reset()`, `.finish(msg)`, `.stopped(msg)`: yangi ish boshlashda `reset()`, bekor/xatoda
  `stopped()`/`finish()`); `LogView(max_lines=20000)` (`.append_line`, `.save_to_file(path=None)`, `.clear_log()`); `format_duration(sec)`.
* `plots.py` (Qt'siz, `matplotlib.figure.Figure` oladi, hammasi `@_safe` bilan o'ralgan: bo'sh/yaroqsiz ma'lumot yoki kutilmagan
  istisnoda figuraga "Ma'lumot yo'q" yozadi, slotni qulatmaydi; figurani qaytaradi): `INDEX_LABEL="Prospektivlik indeksi (0-1)"`,
  `draw_roc(fig, metrics, title_suffix="")`, `draw_pr(fig, metrics, y)`, `draw_calibration(fig, metrics)`,
  `draw_confusion(fig, metrics, model=None)`, `draw_spatial_diagnostics(fig, result)` (TrainingResult bilan),
  `draw_importance(fig, result)`, `draw_shap_beeswarm(fig, result, model=None)`,
  `draw_shap_dependence(fig, result, model=None, feature=None, color_feature="auto")`, `draw_corr_heatmap(fig, corr)`,
  `draw_map(fig, array, transform=None, title="", cmap="RdYlGn_r", vmin=0.0, vmax=1.0, aoi_gdf=None, positives=None,
  background=None, class_labels=None, cbar_label="")`, `draw_success_rate(fig, curve)`, `draw_tuning_trials(fig, tuned_params)`
  (figura o'lchamiga moslashadi), `save_all_figures(figs: dict[str, Figure], out_dir, formats=("png","pdf"), dpi=300, log_fn=None) -> list[paths]`,
  `no_data(fig, msg, detail, title)`. 30 tadan ko'p feature bo'lsa importance/SHAP grafiklarida eng muhim 20 tasi ko'rsatiladi.
  `draw_map` juda katta rasterda (16 mln piksel) har chizishda sekin (~2.4 s): ko'rsatish uchun stride bilan <= ~2500 px ga kamaytiring.
* `param_panel.py`: `HyperParamPanel(QWidget)` — `PARAM_SPECS` dan avtomatik (har model uchun sub-tab, `group` bo'yicha `QGroupBox`;
  int -> QSpinBox, float -> QDoubleSpinBox, bool -> QCheckBox, choice -> QComboBox, optint/optfloat -> "avtomatik/None" checkbox + spin);
  `.get_hyperparams() -> dict` (validatsiyadan o'tgan, to'liq), `.set_hyperparams(hp)` (MERGE: berilmaganlar o'zgarmaydi; ogohlantirishlar
  ro'yxatini qaytaradi), `.reset_defaults(model=None)`, `.get_tuning() -> TuningConfig`, `.set_tuning(t)`, `.set_cost_hint(text)` /
  `.cost_hint()`, `.save_preset_to(path)` / `.load_preset_from(path)` (+ dialogli tugmalar), signal `changed`.
  Bog'liqliklar (qiymat saqlanadi, vidjet o'chadi): `CNN.mode="tabular1d"` => `window`, `augment`; `RandomForest.bootstrap=False` => `max_samples`.
  Modellarni yoqish-o'chirish (`cfg.use_models`) panelda YO'Q — asosiy oynada alohida checkbox'lar.
  `TuningGroup` (yoqish, modellar (CNN "qimmat!"), `n_iter`, ichki fold'lar, scoring, "Qidiruv oraliqlarini tahrirlash..."),
  `SearchSpaceDialog(tuning)` — tunable parametr oraliqlarini jadvalda tahrirlash (xato bo'lsa dialog yopilmaydi).
* `result_tabs.py`: natija tab'lari (har biri `QWidget`; `MplCanvas`/`DataFrameTable` + `plots.draw_*`; bo'sh natijada "Hali natija yo'q"):
  ma'lumotlar tahlili, natijalar (Spatial/Random CV metrikalari, ROC/PR/kalibrlash/confusion, CSV/XLSX va grafiklarni saqlash),
  spatial CV diagnostika (sahifalar: Diagnostika grafigi, Fold jadvali, Fon sezgirligi, Tuning, Ogohlantirishlar (`SpatialTab.WARN_TAB`)),
  feature importance (permutation + SHAP), prognoz xarita (sinflash sozlamalari, xarita, success-rate, sinf statistikasi, eksport).
  **Tembel (lazy) chizish:** `tab.lazy_fill = True` bo'lganda `set_result`/`set_prediction`/`set_diagnostics` faqat yengil qismlarni
  (jadval, yorliq) darhol bajaradi; og'ir canvas chizishlari canvas BIRINCHI KO'RSATILGANDA (`QEvent.Show` -> `QTimer`) `plots.draw_*`
  (artist'lar) + canvas'ning bosqichli `draw_idle()` i orqali bajariladi; ko'rinmagan tab'lar chizilmaydi. API:
  `pending_count()`, `visible_pending_count()`, `flush_one()`, `flush()` (SINXRON, testlar/saqlash uchun). Standart (`lazy_fill=False`) -
  hammasi darhol (sinxron). `_begin()`/`clear()` eskirgan kutayotgan chizishlarni tashlaydi. Tab tanasi (`QScrollArea` ichida):
  canvas minimal o'lchami ~480x240, `_FlatVBox`/`_Tabs` `heightForWidth` ni uzatmaydi (aks holda scroll tana balandligini sizeHint
  bo'yicha hisoblab, canvas pastki qismi ko'rinmay qolardi); uzun izohlar `_NoteBox` (yig'iladigan) da; `MapTab.figures()` kalitlari
  `map_<Model>`, `map_uncertainty`, `map_classes`, `success_rate`.
* `main_window.py`: `MainWindow` (tablar `TAB_TITLES`: "1. Ma'lumotlar", "2. Giperparametrlar", "3. Tahlil", "4. Natijalar",
  "5. Spatial CV", "6. Importance", "7. Xarita", "8. Modellar"; to'liq nomlar `TAB_TOOLTIPS` da, tooltip sifatida), `main(argv=None) -> int`;
  boshlang'ich o'lcham `initial_window_size(avail_w, avail_h)` = min(1400x900, ekran availableGeometry'ning 90%), kamida 980x640;
  global `sys.excepthook`, `closeEvent` (ishchi ishlayotgan bo'lsa tasdiq so'rab `cancel()`), `_set_busy(bool)` (BARCHA tugmalar holati
  bitta joyda), `_collect_config() -> RunConfig` / `_apply_config(cfg)` (NaN/inf qiymat vidjetga TEGMAYDI + ogohlantirish; umuman
  son bo'lmagan qiymat `ValueError`). Barcha uzoq ishlar `workers.*` orqali (GUI qotmaydi).
  1-tab: yuqorida sozlamalar (scroll), o'rtada tugmalar + bir qatorli progress, pastda yig'iladigan `QTabWidget` (Log / Taxminiy
  hisob-kitob) - `main_splitter` (3 bo'lak, oxirgisi yig'iladi). 2-tab: `HyperParamPanel` to'g'ridan-to'g'ri (tashqi scroll yo'q).
  **Tembel to'ldirish:** `_fill_tabs(result)` tab'larni `lazy_fill` bilan to'ldiradi; `fill_pending()` (ko'rinib turgan, chizilishi
  tugamagan canvas'lar), `fill_all_now()` (hammasini sinxron chizadi), `wait_idle()` ko'rinmaganlarni kutmaydi.
  Xato matni progress qatorida 150 belgidan oshsa `...` bilan qisqartiriladi (to'liq matn tooltip'da); `autoexport` xatosi
  "Tayyor" holatini o'zgartirmaydi (faqat log + status). `format_manifest` har bo'lim uchun alohida try/except; `_select_bundle`
  esa buzuq tuzilishli manifest'ni (`band_names` ro'yxat emas va h.k.) rad etadi.
* Ildizda `mpm_ml_gui.py`: `sys.exit(main())` (chiqish kodi `main()` dan).

## 4. O'zgarishlar jadvali: asl koddan (legacy) nima tuzatildi va nima qo'shildi

Asl kod: `legacy/mpm_ml_gui_fixed.py` (bitta fayl, ~2000 qator). "Legacy" ustunida asl xatti-harakat, "Yechim" ustunida
hozirgi modul va funksiya, "Test" ustunida tasdiqlovchi test fayli (va muhim testlar nomi) ko'rsatilgan. Testlarni ishga tushirish:
`QT_QPA_PLATFORM=offscreen python -m pytest tests/<fayl>`.

### 4.1 Tuzatilgan xatolar (BUG-01 ... BUG-15)

| ID | Legacy xatti-harakati | Yechim (modul.funksiya) | Test |
|---|---|---|---|
| BUG-01 | `compute_shap_summary` faqat `list` shaklini kutadi; yangi shap RF uchun `(n,p,2)` massiv qaytaradi, `mean_abs` shakli `(p,2)` bo'lib `draw_importance` `barh` da xato beradi va PyQt5 slotida istisno => dastur abort | `explain.normalize_shap_values` (list/`(n,p)`/`(n,p,2)`/`(2,n,p)`/Explanation), `explain.compute_shap_summary`; `plots._safe` dekoratori (har `draw_*` xatoda "Ma'lumot yo'q" yozadi); `widgets.MplCanvas.redraw` xatoni yutadi; `workers._BaseWorker.run` barcha xatoni `error_signal` ga yo'naltiradi; `main_window` global `sys.excepthook` + slotlarda try/except | `test_explain.py` (`test_normalize_all_layouts`, `test_shap_rf_xgb_real`, `test_shap_failure_does_not_crash`); `test_plots.py` (`test_garbage_never_raises`, `test_importance_shap_shapes`, `test_unexpected_exception_is_not_fatal`); `test_gui_widgets.py::test_redraw_error_does_not_raise`; `test_workers.py::test_training_worker_error_has_traceback` |
| BUG-02 | Prognoz ishchisining `error_signal`i ham o'qitishning `on_error` slotiga ulangan, u faqat `btn_train` ni qayta yoqadi: prognoz xatosidan keyin `btn_predict_map` qulflanib qoladi (bekor qilish/Stop yo'q) | workerlar xato, to'xtatish va tugashni alohida signal bilan xabar qiladi (`error_signal`, `cancelled_signal`, `finished_signal`); `ProgressPanel.stopped()`/`finish()`; `main_window._set_busy(bool)` BARCHA tugmalarni bitta joyda tiklaydi | `test_workers.py` (`test_training_worker_cancel_before_run`, `test_prediction_worker_cancel_and_error`, `test_export_worker_error`); `test_gui_widgets.py` (`test_stopped_and_format`, `test_progress_restart_after_early_stop`); `test_main_window.py` (`_set_busy`) |
| BUG-03 | MDI zaxirasi `CalibratedClassifierCV.estimator` (o'qitilmagan) dan o'qiydi | `base.ModelWrapper.importance_mdi/tree_model`, `models.SklearnModel.tree_model` (o'qitilgan `calibrated_classifiers_[0].estimator`), `explain.mdi_importance`; `pipeline._importance` zaxira sifatida (`source="mdi"`) | `test_models.py::test_tree_model_and_importance`; `test_explain.py::test_mdi_importance`; `test_pipeline.py::test_models_thresholds_importance_shap` |
| BUG-04 | CNN `EarlyStopping(monitor="loss")` (train loss); CV'da 150, final'da 200 epoch; `clear_session` yo'q | `cnn.CNNModel.fit`: stratified `val_fraction` ajratmasi (`_split`), `EarlyStopping(monitor="val_loss", restore_best_weights=True)`, `_clear_session`; `epochs` yagona manba (`config.PARAM_SPECS["CNN"]`, CV va final bir xil) | `test_cnn.py` (`test_early_stopping_val_loss_and_deterministic`, `test_fit_info_val_loss_matches_restored_best_weights`, `test_split_is_stratified_and_deterministic`, `test_single_positive_falls_back_to_train_loss`, `test_next_fit_clear_session_keeps_earlier_models_usable`) |
| BUG-05 | `repeated_group_kfold_splits`: bloklar tasodifiy aralashtirilib `i % k` bilan fold'larga taqsimlanadi (stratifikatsiyasiz), validation fold'da musbat nuqta bo'lmasligi mumkin (AUC aniqlanmaydi) | `spatial.stratified_group_splits` (`StratifiedGroupKFold`, har repeat yangi seed, qayta urinish, musbat blokni ko'chirish), `spatial.adapt_block_size` (musbat nuqtali bloklar >= k) | `test_spatial.py` (`test_every_val_fold_has_positive`, `test_val_folds_have_positive_when_positives_clustered_in_few_blocks`, `test_repair_moves_whole_positive_block_after_retries_exhausted`, `test_adapt_block_size_requires_positive_blocks`); `test_cv.py::test_fold_table` (`pos_val >= 1`) |
| BUG-06 | `SVC(probability=not calibrate)` (sklearn 1.11 da olib tashlanadi), `geometry.unary_union` eskirgan | `models._make_svm` + `SklearnModel.fit`: SVC `probability` parametrisiz, har doim `CalibratedClassifierCV` ichida; `data._union_all` (`union_all()`, zaxira `unary_union`) | `test_models.py` (`test_svm_structure_no_probability_flag`, `test_svm_scaling_inside_pipeline`, `test_no_future_or_deprecation_warnings`); `test_data.py` (`test_load_aoi_and_positive`, `test_invalid_aoi_geometry_is_fixed_before_union`) |
| BUG-07 | `predict_probability_map`: CNN `model.predict(..., verbose=0)` `batch_size` berilmasdan, ya'ni standart 32 bilan chaqiriladi (juda sekin); RF/XGBoost `n_jobs=1` qattiq (GUI'dan o'zgartirib bo'lmaydi) | `predict.predict_probability_maps(batch_size=8192)` (faqat valid indekslar, bo'laklab), `cnn.CNNModel.predict_proba_pos(batch_size=8192)`, `models.make_model(n_jobs=...)` | `test_predict.py` (`test_predict_batches_bound_memory_and_match`, `test_predict_does_not_materialize_full_feature_array`); `test_cnn.py::test_predict_batch_size_does_not_change_result`; `test_models.py` (`test_seed_and_n_jobs_passed`, `test_n_jobs_invariant`) |
| BUG-08 | `_mean_ci95`: CI faqat repeat'lar bo'yicha o'rtacha ± 1.96·std/√n (repeat soni oshsa CI sun'iy torayadi, nuqtalar/bloklar bog'liqligini hisobga olmaydi) | `cv.compute_metrics` / `cv._bootstrap_ci`: blok-bootstrap (`spatial.block_bootstrap_indices`) o'rtacha OOF ustida; repeat-std alohida (`auc_std`) | `test_cv.py` (`test_ci_matches_manual_block_bootstrap`, `test_block_bootstrap_is_wider_for_clustered_data`, `test_ci_is_for_mean_proba_auc_even_with_noisy_repeats`, `test_n_boot_zero_is_fast_mode`); `test_spatial.py::test_block_bootstrap_whole_blocks_with_replacement` |
| BUG-09 | `estimate_autocorrelation_range(band_index=0)`: faqat alifbo bo'yicha birinchi band, `max_pairs=250000` | `spatial.estimate_autocorrelation_range(band_indices=None)`: barcha raqamli bandlar (yoki tanlangan) medianasi, `n_points=1500`, strukturasiz bandlar chiqariladi; `pipeline._block_size` (`cfg.variogram_band`) | `test_spatial.py` (`test_variogram_all_bands_median_ignores_structureless`, `test_variogram_not_dependent_on_band_order_of_first_band`, `test_variogram_stable_across_random_state_and_n_points`); `test_pipeline.py::test_block_size_manual_and_named_band` |
| BUG-10 | Barcha qatlamlar bilinear resample (kategorik qiymatlar buziladi), kategorik qatlam raqamli sifatida o'qitiladi | `data.load_and_align_rasters(categorical=...)` (`Resampling.nearest`), `data.FeaturePipeline` (one-hot `band==level`), `data.suggest_categorical_layers` | `test_data.py` (`test_categorical_nearest`, `test_pipeline_onehot`, `test_pipeline_real_categorical`, `test_suggest_categorical_layers`); `test_pipeline.py::test_tuned_categorical_final_draws` |
| BUG-11 | Fon nuqtalar nodata joyga tushsa keyin tashlanadi (soni kamayadi); nuqtama-nuqta sekin sikl | `data.generate_background_points(valid_mask=, transform=)`: valid-mask ichida vektorlashtirilgan batch generatsiya, n_points ga yetadi; `data.build_dataset` yaroqsizlarni loglab tashlaydi | `test_data.py` (`test_background_respects_valid_mask`, `test_background_nonconvex_aoi_reaches_count`, `test_background_pixels_exclude_positive_pixels`, `test_background_shortage_warns`, `test_background_strategies`) |
| BUG-12 | `CalibratedClassifierCV` standart `ensemble=True`: "final" model aslida 3 ta 2/3-ma'lumotli model; SHAP faqat 1-ichki fold modeliga | `models.SklearnModel._calibrator`: `ensemble=False` (bazaviy model to'liq train'da), `tree_model()` shu modelni qaytaradi; `explain.compute_shap_summary` shundan foydalanadi | `test_models.py::test_calibrated_base_fitted_on_full_train`; `test_explain.py` (`test_shap_rf_xgb_real`, `test_shap_uses_first_draw_model`) |
| BUG-13 | `_fit_predict_calibrated`: `eff_cv = max(2, min(calib_cv, kam sinf soni))` — train'da 1 ta musbat qolsa ham `cv=2` majburlanadi va sklearn xato beradi (crash) | `models.SklearnModel.fit`: `eff_cv = min(calibration_cv, n_pos, n_neg)`; <2 yoki kalibrlash xatosi => kalibrlashsiz (`fit_info["calibration"]="skipped"`, ogohlantirish), SVM uchun `_SigmoidMinMaxSVM` zaxirasi | `test_models.py` (`test_three_positives_no_crash`, `test_two_positives_adaptive_cv`, `test_single_positive_skips_calibration`, `test_small_calibration_cv_skips`, `test_calibration_failure_falls_back`, `test_svm_fallback_is_monotonic_and_picklable`); `test_cv.py::test_calibration_skipped_is_reported` |
| BUG-14 | Mayda: ishlatilmagan importlar; `nanmean` ogohlantirishi; bir pikseldagi dublikat musbatlar; MultiPoint; AOI tashqarisidagi nuqtalar; GeoTIFF'ni manba profili (tiled/blocksize) bilan yozish; `bg_sensitivity_repeats=3` qattiq; fold xaritasida legend yo'q; suptitle kesilishi; legend ma'lumot ustida | `data.dedupe_points_by_pixel`, `data.load_positive_points` (explode, AOI filtri), `data.load_and_align_rasters` (toza `profile`), `predict.save_rasters` (`_clean_profile`), `predict._nanmean_std` (ogohlantirishsiz), `explain._mean_abs`, `config.RunConfig.bg_sensitivity_repeats`, `plots.draw_spatial_diagnostics` va `_legend_outside` (constrained layout) | `test_data.py` (`test_dedupe_points_by_pixel`, `test_load_positive_multipoint_and_nonpoint`, `test_profile_not_inherited_from_tiled_source`); `test_predict.py` (`test_predict_no_warnings`, `test_save_rasters_roundtrip`); `test_explain.py::test_shap_all_nan_column_no_runtime_warning`; `test_plots.py` (`test_spatial_real_legend_not_clipped`, `test_titles_and_footnotes_fit_narrow_canvas`); `test_cv.py::test_bg_sensitivity_pipeline_calls` (`n_repeats` parametr sifatida) |
| BUG-15 | Giperparametrlar CV (`run_cv_training`) va final (`fit_final_models`) da alohida qattiq yozilgan literal'lar (RF 500/`min_samples_leaf=2`, XGBoost, SVM ikki joyda takrorlangan; CNN: CV 150 va final 200 epoch); GUI'dan o'zgartirib bo'lmaydi | `config.PARAM_SPECS` + `config.validate_hyperparams`: yagona `hp` lug'ati `cv.run_cv`, `pipeline._final_fit`, `tuning.tune_model`, `persist.save_bundle` ga uzatiladi; `models.normalize_params` har parametrni modelga majburan xaritalaydi (`_ensure_consumed`) | `test_models.py::test_param_specs_fully_covered`; `test_cnn.py::test_param_coverage_table_is_complete`; `test_cv.py::test_hp_is_validated_and_used`; `test_gui_widgets.py::test_panel_hp_feeds_runconfig` |

### 4.2 Qo'shilgan imkoniyatlar (ENH-01 ... ENH-09)

| ID | Imkoniyat | Modul.funksiya | Test |
|---|---|---|---|
| ENH-01 | To'liq giperparametr paneli, RF/SVM yoqish-o'chirish, preset JSON, standartga qaytarish, seed/n_jobs, ishlatilgan hp natija bilan | `config.PARAM_SPECS/RunConfig/save_preset/load_preset`; `gui.param_panel.HyperParamPanel`; `pipeline.export_results` (`hyperparameters_used.json`, `run_config.json`); `main_window` (modellar checkbox'lari, seed, n_jobs) | `test_gui_widgets.py` (`test_widget_for_every_spec`, `test_save_load_file`, `test_reset_all`, `test_load_run_config_file`); `test_models.py::test_param_specs_fully_covered`; `test_pipeline.py::test_export_files_readable` |
| ENH-02 | Nested spatial CV bilan tasodifiy giperparametr qidirish | `tuning.tune_model/sample_params`; `cv.run_cv(tuning=)`; `pipeline._final_fit/_tune_final`; `config.TuningConfig`; `param_panel.TuningGroup/SearchSpaceDialog`; `plots.draw_tuning_trials` | `test_tuning.py` (butun fayl); `test_cv.py` (`test_nested_tuning_stub`, `test_nested_tuning_real_module`); `test_pipeline.py` (`test_tuned_prediction_and_export`); `test_gui_widgets.py` (`test_edit_spaces_button`, `test_edit_and_accept_updates_spaces`); `test_plots.py` (`test_tuning_real`) |
| ENH-03 | Haqiqiy CNN: patch2d (WxW oyna, 2D-Conv, augmentatsiya, val-split ES), tabular1d saqlanadi | `cnn.CNNModel`; `data.extract_patches/Dataset.get_patches`; `cv._PatchCache`; `explain.permutation_importance_auc(patches=)` | `test_cnn.py` (`test_patch2d_fit_predict_shape`, `test_tabular1d_fit_predict`, `test_integration_with_dataset_patches`); `test_data.py` (`test_extract_patches_matches_bruteforce`); `test_pipeline.py::test_cnn_end_to_end` (slow) |
| ENH-04 | Metrikalar jadvali (AUC+CI, PR-AUC, BalAcc, F1, Brier, Sens/Spec@Youden), PR, kalibrlash, confusion, CSV/XLSX eksport, toolbar, PNG/PDF | `cv.compute_metrics/metrics_dataframe`; `plots.draw_roc/draw_pr/draw_calibration/draw_confusion/save_all_figures`; `widgets.MplCanvas` (`NavigationToolbar2QT`), `DataFrameTable.save_csv`; `pipeline.export_results` | `test_cv.py` (`test_metrics_keys`, `test_metrics_dataframe`, `test_calibration_bins_adapt_to_positives`); `test_plots.py` (`test_roc_real`, `test_pr_real_baseline`, `test_calibration_real_variable_bins`, `test_confusion_real`, `test_save_all_figures`); `test_gui_widgets.py` (`test_save_csv`); `test_pipeline.py::test_export_files_readable` |
| ENH-05 | Model saqlash/yuklash (bundle) va yangi maydonga qo'llash | `persist.save_bundle/load_bundle/apply_bundle`; `workers.ApplyBundleWorker`; GUI "Modellar" tabi | `test_persist.py` (butun fayl: `test_roundtrip_identical_maps`, `test_apply_bundle_other_folder`, `test_load_log_warns_about_trust`); `test_workers.py::test_apply_bundle_worker` |
| ENH-06 | Stop, fold bo'yicha progress + ETA, n_jobs, seed, log fayli, excepthook, yopishda tasdiq | `common.CancelToken/check_cancel/sub_progress`; `workers._BaseWorker.cancel`; `cv.run_cv` (progress+ETA, `cancel`); `pipeline._Ctx` (log fayli `mpm_run_<n>.log`); `widgets.ProgressPanel/LogView`; `main_window` (Stop, `closeEvent`, `sys.excepthook`) | `test_cv.py` (`test_logs_and_progress`, `test_cancel_after_first_fold`); `test_pipeline.py` (`test_progress_monotonic_and_log`, `test_log_file_utf8_and_sequential`, `test_cancel_mid_run_closes_log`); `test_workers.py` (`test_training_worker_cancel_via_progress`); `test_gui_widgets.py` (`test_eta_with_fake_clock`, `test_append_and_clear`, `test_save_utf8`) |
| ENH-07 | Ma'lumotlar tahlili (layer stats, korrelyatsiya, VIF, data dictionary), kategorik qatlamlar, dublikat nazorati, fon strategiyalari, K-fon ansambli, xarita overlay, sinflangan xarita + maydon statistikasi, noaniqlik, success-rate, SHAP beeswarm/dependence | `data.data_diagnostics/build_data_dictionary/FeaturePipeline/dedupe_points_by_pixel/generate_background_points(strategy=)`; `config.RunConfig.final_bg_draws` + `pipeline._final_fit/_draw_dataset`; `plots.draw_corr_heatmap/draw_map/draw_success_rate/draw_shap_beeswarm/draw_shap_dependence`; `predict.classify_map/class_area_stats/success_rate_curve/predict_probability_maps(uncertainty)`; `explain.compute_shap_summary` | `test_data.py` (`test_diagnostics_columns_and_values`, `test_manual_metadata_and_dictionary`, `test_background_strategies`, `test_dedupe_points_by_pixel`); `test_predict.py` (`test_classify_quantile_equal_area`, `test_class_area_stats_with_deposits`, `test_success_rate_known_value`, `test_predict_structure_and_values`); `test_plots.py` (`test_corr_real`, `test_map_real_overlays_and_extent`, `test_success_rate_real`, `test_beeswarm_real`, `test_dependence_real_and_manual`); `test_pipeline.py::test_tuned_categorical_final_draws` |
| ENH-08 | "Prospektivlik indeksi" terminologiyasi + izoh | `plots.INDEX_LABEL`; `pipeline._NOTE_INDEX/_NOTE_CNN` (log, `summary.txt`); `predict.save_rasters` (band tavsifi); `persist._readme_text`; GUI yordam matni | `test_plots.py` (`INDEX_LABEL` tekshiruvlari: `test_wrapped_functions_exposed`, `test_map_real_overlays_and_extent`, `test_success_rate_real`); `test_pipeline.py` (`test_export_files_readable`, `test_prediction_saves_files`); `test_persist.py::test_readme`; `test_predict.py::test_save_rasters_roundtrip` |
| ENH-09 | `requirements.txt`, avtomatik testlar, README | `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `tests/` (13+ fayl), `README.md`, `docs/USER_GUIDE.md`, `docs/CHANGELOG.md` | butun `tests/` to'plami |

## 5. Ma'lum cheklovlar va eslatmalar

* **Yakuniy tuning va baholash.** Yakuniy tuning butun ma'lumotda bajariladi; nested spatial CV shu *protsedurani* baholaydi
  (har tashqi fold'ning train qismida alohida tuning). `tuning.mode="final"` da spatial CV bazaviy hp bilan baholanadi va
  yakuniy modelning natijasi bilan mos kelmaydi: GUI faqat `nested` ni ko'rsatadi.
* **Random va spatial CV adolatli solishtirilmaydi (tuning yoqilganda).** Random CV bazaviy hp bilan, spatial CV nested tuning
  bilan baholanadi; "optimizm = random - spatial" farqi qisman shundan ham bo'lishi mumkin (log'da yoziladi).
* **Permutation importance** faqat spatial CV'ning 1-takrorida, har fold'ning validation qismida hisoblanadi; MDI zaxirasida
  `std = 0` (xato chiziqlari chizilmaydi).
* **1 repeat (`n_repeats=1`) da repeat-std ma'nosiz:** `compute_metrics` barcha `*_std` ni NaN qiladi (0.0 emas); ishonch oralig'i
  (`*_ci95`, blok-bootstrap) saqlanadi. Fon sezgirligida 1 ta draw bo'lsa `summary[name]["std"]` hali 0.0 (qiymatlar soni 1),
  `summary.txt` da esa "(1 tanlov: std yo'q)" deb ko'rsatiladi.
* **Ixtiyoriy kutubxona import xatosi:** `common._lazy_import` import yiqilsa `None` keshlaydi, lekin sababni ham saqlaydi:
  `common.import_error(modname) -> str|None` ("XatoTuri: xabar", <= 300 belgi; modul haqiqatan o'rnatilmagan bo'lsa `None`).
  `pipeline` (XGBoost/TensorFlow/SHAP) va `explain.compute_shap_summary` "o'rnatilmagan" o'rniga modul nomi + haqiqiy xatoni
  ogohlantiradi. CNN yoqilgan bo'lsa `pipeline._prepare` TensorFlow'ni erta import qilib ko'radi (muvaffaqiyatsiz => CNN
  o'tkazib yuboriladi, ish o'rtasida yiqilmaydi).
* **SHAP** faqat RF va XGBoost uchun, faqat birinchi fon tanlovidagi yakuniy model bilan, `shap_max_background` ta qatorda;
  birliklar modelga bog'liq (RF: ehtimollik, XGBoost: log-odds). `common.get_shap()` `None` keshlashi mumkin (o'rnatilmagan bo'lsa).
* **success-rate** o'qitish nuqtalarida hisoblanadi (optimistik), mustaqil validatsiya emas. **noaniqlik xaritasi** — alohida
  modellar xaritalarining std'i (modellar kelishmovchiligi), statistik ishonch oralig'i emas.
* **CNN** chiqishi kalibrlanmaydi va neg/pos og'irlik bilan o'qitiladi; ansambl oddiy o'rtacha bo'lgani uchun shkalalar farq qilishi mumkin.
* **`run_background_sensitivity`** bazaviy hp bilan ishlaydi (tuning yo'q); CNN patch2d bo'lsa `feature_stack` pipeline'dan
  uzatiladi (yoki bir marta quriladi) va barcha draw'lar orasida ulashiladi.
* **Cancel** SHAP bosqichigacha (va SHAP ichida) ishlaydi; SHAP tugagach yakunlash bekor qilinmaydi: `finished_signal` kelsa
  ham Stop bosilgan bo'lishi mumkin (natija saqlanadi).
* **`metadata.csv` yon ta'siri:** `data.load_or_create_manual_metadata` KIRISH TIFF papkasiga bo'sh shablon YOZADI (bor bo'lsa
  tegmaydi; log'da aniq aytiladi). Mavjud fayl ajratgich (',' / ';') va kodlash (utf-8-sig/utf-8/cp1251/cp1252) bo'yicha
  avtomatik aniqlanadi.
* **Versiyalar:** `common.collect_versions` paketni `importlib.metadata` nomi bilan qidiradi (`tensorflow`); faqat `tensorflow-cpu`
  o'rnatilgan muhitda `versions.json`/`manifest.json` da `tensorflow: null` chiqadi va bundle yuklashdagi TensorFlow versiya-farqi
  ogohlantirishi ishlamaydi.
* **GUI tembel chizish:** natija tab'larining og'ir canvas'lari tab (sahifa) birinchi ko'rsatilganda chiziladi (o'qitish tugagach
  qotish yo'q: 100x100 sintetik loyihada QTimer(20 ms) tikkerining eng katta oralig'i ~0.3 s, sinxron to'ldirishda ~2.8 s edi).
  Bitta og'ir grafik ham bosqichlarga bo'linadi (chizish / layout / rasterlash), lekin har bosqich o'zi matplotlib'ning bir
  yo'la ishi: juda ko'p panelli grafikda (20+ feature) bosqich ~0.5 s dan oshishi mumkin. Ko'rinmagan tab'lar chizilmagani uchun
  `figures()` (saqlash) bunga bog'liq emas (yangi Figure'lar yaratadi); testlarda `MainWindow.fill_all_now()` ishlating.
* **Bundle xavfsizligi:** RF/SVM/XGBoost modellari joblib/pickle — begona manbadagi bundle'ni yuklamang.
* **`draw_map`** 16 mln pikselli rasterda sekin: ko'rsatish uchun kamaytirilgan (stride) massiv bering.
* **Tekshirilmagan masshtab.** Yakuniy audit sintetik loyihalarda (<= ~300x300 piksel) o'tkazildi; katta rasterlar (1500x1500+,
  12+ qatlam) bo'yicha unumdorlik/xotira o'lchovlari avtomatik auditdan o'tmadi (`pipeline.estimate_cost_text` taxminiy).
  Katta loyihada avval `n_repeats` va `n_bootstrap` ni kamaytirib sinab ko'ring.
