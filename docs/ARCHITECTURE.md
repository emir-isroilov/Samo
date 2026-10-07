# MPM ML GUI v2 — arxitektura va modul shartnomasi

Bu hujjat **yagona shartnoma**: har bir modul shu yerdagi imzo va ma'lumot sxemalariga rioya qiladi.
Eski kod: `legacy/mpm_ml_gui_fixed.py` (faqat ma'lumot uchun; undan mantiqni olish mumkin, lekin
xatolarini ko'chirmang).

## 0. Umumiy qoidalar

* Python >= 3.9 (`from __future__ import annotations`), scikit-learn >= 1.3 (sinov: 1.9.1), xgboost 2/3,
  TensorFlow/Keras 3 (sinov: 2.21), shap >= 0.42, shapely 1.8/2.x, geopandas 0.14/1.x, rasterio >= 1.3.
  **Eskirgan API ishlatilmaydi**: `SVC(probability=...)` YO'Q (har doim `CalibratedClassifierCV`),
  `GeoSeries.unary_union` YO'Q (`union_all()`, zaxira: `unary_union`), sklearn `permutation_importance` YO'Q (o'zimiznikini yozamiz).
* `tensorflow`, `xgboost`, `shap`, `PyQt5` modul darajasida import QILINMAYDI (faqat `mpm.gui.*` PyQt5 import qiladi).
  Ixtiyoriylar `mpm.common.get_tf()/get_xgboost()/get_shap()` orqali, funksiya ichida olinadi.
* `print` ishlatilmaydi: har bir funksiya `log_fn=None` oladi (None => `noop_log`), uzoq funksiyalar
  `progress_fn(frac:0..1, msg:str)=None` va `cancel=None` (`CancelToken`) oladi; tsikllarda `check_cancel(cancel)`.
* Tasodifiylik: har bir funksiya `seed`/`random_state` oladi; global holat yo'q. Reproduktivlik: bir xil
  ma'lumot + bir xil seed => bir xil natija.
* Hamma foydalanuvchiga ko'rinadigan matn, izoh va docstring **o'zbek tilida (lotin)**, ixcham (original kod uslubi).
* Terminologiya: chiqish "ehtimollik" emas, **"prospektivlik indeksi"** (pos:fon nisbati sun'iy). Fayl nomlari
  `prognoz_*.tif` saqlanadi, lekin legend/label/hujjatda "prospektivlik indeksi (0-1)" yoziladi.
* Test: `tests/test_<modul>.py`, `pytest` (venv'da o'rnatilgan), tez bo'lishi shart (<60 s/fayl);
  TensorFlow/CNN/to'liq pipeline testlari `@pytest.mark.slow`. Sintetik ma'lumot: `tests/synth.py`,
  fixture'lar: `tests/conftest.py` (`synth_project`, `synth_project_cat`).
* Ishga tushirish muhiti: `PY=/tmp/claude-0/-home-user-Samo/6fae25e6-7e1b-535c-aa07-7f13809fb8d9/scratchpad/venv/bin/python`
  (numpy, pandas, sklearn, scipy, rasterio, geopandas, shapely, xgboost, tensorflow, shap, PyQt5, matplotlib, pytest, openpyxl).
  Repo ildizidan: `cd /home/user/Samo && $PY -m pytest tests/test_x.py`. GUI: `QT_QPA_PLATFORM=offscreen`.
  Mashinada 4 CPU / 15 GB — og'ir testlarni (TF) kichik tuting (epochs<=3, kichik tarmoq), bir vaqtda ko'p
  parallel og'ir jarayon ishga tushirmang.
* Boshqa agent fayllariga TEGMANG: faqat o'zingizga ajratilgan fayllarni yozing. Boshqa modulning xatosini
  topsangiz — o'zgartirmasdan, yakuniy hisobotda aniq yozing (fayl, qator, sabab, taklif).

## 1. Tuzatilishi shart bo'lgan xatolar (BUG) — `legacy/` dagi asl kodga nisbatan

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
  common.py      (tayyor)  konstantalar, CancelToken, sub_progress, lazy import, set_global_seed, safe_name
  base.py        (tayyor)  ModelWrapper abstrakt sinfi
  config.py      (tayyor)  PARAM_SPECS, RunConfig, TuningConfig, preset IO
  data.py                  raster yuklash, FeaturePipeline, nuqtalar, fon, Dataset, patchlar, diagnostika
  spatial.py               bloklar, variogram, stratified group splits, bootstrap indekslari
  models.py                make_model(): RF/SVM/XGBoost wrapper'lari (+kalibrlash)
  cnn.py                   CNNModel (patch2d / tabular1d)
  explain.py               OOF permutation importance, SHAP, MDI
  tuning.py                tasodifiy qidiruv (nested spatial CV)
  cv.py                    run_cv, compute_metrics, background sensitivity
  predict.py               xarita bashorati, sinflash, noaniqlik, success-rate, GeoTIFF saqlash
  persist.py               bundle saqlash/yuklash/qo'llash
  pipeline.py              run_training / run_prediction / export_results
  workers.py               QThread ishchilar (cancel, progress)
  gui/ widgets.py plots.py param_panel.py main_window.py
mpm_ml_gui.py              (ildizda) launcher
```

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
    tech_metadata: list[dict]   # eski tech_metadata bilan bir xil kalitlar
    # xossalar: .n_bands .height .width .pixel_size (m)

def find_shapefile(folder) -> str|None
def find_tiff_files(folder) -> list[str]
def suggest_categorical_layers(tiff_paths) -> list[str]
    # heuristika: integer dtype yoki (barcha qiymatlar butun va noyob qiymatlar <= 20) -> band nomlari
def load_and_align_rasters(tiff_paths, categorical=(), log_fn=None, assume_crs_if_missing=True,
                           fallback_epsg=TARGET_EPSG, cancel=None) -> RasterStack
    # birinchi fayl = referens grid (EPSG:28411 ga). Kategorik: Resampling.nearest, qolgani bilinear.
    # nodata/inf/|x|>1e15 -> NaN. CRS yo'q fayllar fallback_epsg deb olinadi (ogohlantirish bilan).
    # Faylni ikki marta o'qimaslik; yakuniy stack inf-siz.
def valid_pixel_mask(stack) -> (H,W) bool      # barcha bandlar chekli
load_or_create_manual_metadata(tiff_folder, band_names, log_fn=None) -> (dict, path)     # eskidek
build_data_dictionary(band_names, tech_metadata, manual_metadata) -> DataFrame           # eskidek
save_data_dictionary(df, out_dir, log_fn=None) -> path                                   # eskidek

class FeaturePipeline:
    def __init__(self, band_names, categorical=(), max_levels=30)
    def fit(self, stack) -> self          # kategorik bandlar uchun valid piksellardan noyob BUTUN darajalar
    feature_names: list[str]              # raqamli band nomi; kategorik: f"{band}=={int(level)}"
    n_features: int ; band_names ; categorical
    def transform_pixels(self, pix) -> (m, n_features) float32   # pix: (m, n_bands); noma'lum daraja => hammasi 0
    def transform_stack(self, stack) -> (n_features, H, W) float32   # NaN saqlanadi
    def to_dict(self) / @classmethod from_dict(d)                   # JSON-ga yaroqli (persist uchun)

def xy_to_rowcol(transform, x, y) -> (rows:int64[], cols:int64[])   # vektorlashtirilgan, floor, chegara tekshiruvisiz
def load_aoi(aoi_shp, assume_crs_if_missing=True, log_fn=None) -> GeoDataFrame            # EPSG:28411
def load_positive_points(points_shp, aoi_gdf=None, assume_crs_if_missing=True, log_fn=None) -> GeoDataFrame
    # EPSG:28411; MultiPoint -> explode; faqat Point; AOI tashqarisidagilar ogohlantirish bilan tashlanadi
def dedupe_points_by_pixel(points_gdf, transform, shape, log_fn=None) -> GeoDataFrame        # bir pikseldagi dublikat musbatlar
def generate_background_points(aoi_gdf, positive_gdf, n_points, min_distance, random_state=RANDOM_STATE,
                               strategy="random", valid_mask=None, transform=None, max_attempts_factor=200,
                               log_fn=None) -> GeoDataFrame
    # strategy: "random" (AOI ∩ valid_mask ichida tekis), "grid" (AOI ichida jitterli to'r, keyin tanlab olinadi),
    # "distance_weighted" (musbat nuqtalardan masofaga proporsional ehtimol). min_distance dan yaqinlari rad etiladi
    # (cKDTree). Natija n_points ga yetadi (yetmasa ogohlantirish). Vektorlashtirilgan (batch) generatsiya.

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

def extract_patches(feature_stack, rows, cols, window) -> (m, w, w, p) float32   # NaN bilan to'ldirilgan chegara
def sample_features(xy, raster_or_stack, transform) -> (values (n,n_bands) NaN tashqarida, rows, cols)
def build_dataset(raster: RasterStack, pipeline: FeaturePipeline, positive_gdf, background_gdf,
                  log_fn=None, need_feature_stack=False) -> Dataset
    # yaroqsiz (chekli bo'lmagan) qatorlarni tashlaydi va nechtasi tashlanganini loglaydi
def data_diagnostics(raster: RasterStack, dataset: Dataset|None = None, max_pixels=50000,
                     seed=RANDOM_STATE) -> dict
    # {"layer_stats": DataFrame(band, valid_pct, min, max, mean, std, kind),
    #  "corr": DataFrame (pearson, faqat raqamli bandlar), "vif": DataFrame(band, vif, flag),
    #  "high_corr_pairs": DataFrame(band_a, band_b, corr) (|r|>=0.9)}
```

### 3.2 `mpm/spatial.py`

```python
def estimate_autocorrelation_range(stack, transform, band_indices=None, n_lags=15, n_points=1500,
                                   random_state=RANDOM_STATE, log_fn=None) -> float|None
    # band_indices None => barcha bandlar; har biri uchun range, natija MEDIANA (metr). Hech biri baholanmasa None.
def assign_spatial_blocks(coords, block_size) -> int ndarray (n,)
def adapt_block_size(coords, y, n_splits, block_size, min_size=10.0, log_fn=None) -> (block_size, groups)
    # n_blocks >= n_splits VA musbatlar kamida n_splits ta blokda bo'lguncha yarmiga kamaytiradi (max 30 marta);
    # bo'lmasa ValueError(aniq xabar bilan)
def stratified_group_splits(y, groups, n_splits, n_repeats, random_state=RANDOM_STATE)
    # generator: (repeat, fold, train_idx, val_idx). Har repeat'da yangi seed bilan StratifiedGroupKFold(shuffle=True).
    # Har bir val fold'da >=1 musbat bo'lishi kerak (yetmasa qayta urinadi, oxirida ogohlantirib davom etadi).
    # Bitta blok HECH QACHON train va val orasida bo'linmaydi.
def random_stratified_splits(y, n_splits, n_repeats, random_state=RANDOM_STATE)   # xuddi shu format
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
class SklearnModel(ModelWrapper)         # RF / SVM / XGBoost
    # RF: raw; SVM: Pipeline(StandardScaler, SVC(...)) — `probability` parametri BERILMAYDI;
    # XGB: raw, scale_pos_weight=None => neg/pos (fit paytida y'dan). params -> sklearn kwargs xaritalash:
    #   "none"->None, max_features "all"->None, SVM gamma None->"scale", RF max_samples faqat bootstrap=True bilan.
    # calibrate=True: CalibratedClassifierCV(base, method, cv=eff_cv, ensemble=False).
    # eff_cv = min(calibration_cv, min sinf soni); <2 bo'lsa kalibrlashsiz (fit_info["calibration"]="skipped").
    # calibrate=False va SVM: baribir CalibratedClassifierCV(SVC, ensemble=False) (probabilistik chiqish kerak) —
    #   fit_info["calibration"]="svm_platt_only".
    # tree_model(): o'qitilgan RF/XGB (kalibrlangan bo'lsa calibrated_classifiers_[0].estimator), SVM => None.
    # importance_mdi(): feature_importances_ (p,) yoki None.
# cnn.py
class CNNModel(ModelWrapper)             # input_kind = "patch" (mode=patch2d) yoki "tabular" (mode=tabular1d)
    # patch2d: kirish (n,w,w,p); per-kanal standartlashtirish (train statistikasi, NaN->0 standartlashdan KEYIN);
    #   arxitektura: Conv2D(filters1,k) -> [Conv2D(filters2,k) agar >0] -> GlobalAveragePooling2D -> Dense(dense_units)
    #   -> Dropout -> Dense(1, sigmoid). Augmentatsiya: tasodifiy flip/rot90 (faqat train, params["augment"]).
    # tabular1d: eski usul (Conv1D, feature o'qi bo'ylab) — taqqoslash uchun.
    # O'qitish: stratified val_fraction ajratmasi, EarlyStopping(monitor="val_loss", patience, restore_best_weights),
    #   cancel tokeni Keras callback orqali tekshiriladi. Oxirida tf.keras.backend.clear_session().
    #   Agar val ajratib bo'lmasa (kam musbat) -> train loss bo'yicha ogohlantirish bilan.
    # predict_proba_pos(batch_size=8192) — model.predict(batch_size=batch_size).
    # fit_info: {"epochs_run", "best_epoch", "val_loss", "n_train", "n_val"}
    # save/load: Keras `.keras` fayli + `meta.json` (standartlashtirish parametrlari, params); joblib EMAS.
```

### 3.4 `mpm/explain.py`, `mpm/tuning.py`

```python
# explain.py
def permutation_importance_auc(model, X, y, *, patches=None, n_repeats=5, seed=RANDOM_STATE) -> ndarray (p,)
    # har feature j uchun: ustunni (patch modelida — kanalni barcha patchlarda bir xil permutatsiya bilan) aralashtiradi,
    # AUC pasayishining o'rtachasi. y bir sinfli bo'lsa NaN massiv qaytaradi (xato emas).
def summarize_perm_importance(per_fold: list[ndarray]) -> {"mean","std","per_fold","n_folds"}   # nanmean/nanstd
def normalize_shap_values(sv, class_index=1) -> ndarray (n, p)     # list / (n,p) / (n,p,2) / (2,n,p) hammasini qamraydi
def compute_shap_summary(final_models, X, feature_names, max_background=50, seed=RANDOM_STATE, log_fn=None)
    -> {model: {"mean_abs_shap": (p,), "shap_values": (n,p), "X_background": (n,p), "feature_names": [...]}} | None
    # faqat RF/XGBoost; final_models: {name: [ModelWrapper,...]} (birinchi bg-draw modeli); tree_model() orqali;
    # shap yo'q bo'lsa None + log. Barcha shape'lar normalizatsiya qilinadi.
def mdi_importance(model) -> ndarray|None
# tuning.py
def sample_params(model, base_params, space, rng) -> dict        # space = TuningConfig.resolved_space(model); tur/diapazon/log hurmat
def tune_model(name, base_params, dataset, groups, *, tuning, calibrate, calibration_method, calibration_cv,
               n_jobs, seed, log_fn=None, cancel=None, progress_fn=None) -> dict
    # dataset/groups = TRAIN to'plami (tashqari fold'ning train qismi). Ichki: stratified_group_splits(n_splits=tuning.inner_splits,
    # n_repeats=1). n_iter ta tasodifiy nomzod + base_params o'zi (1-nomzod). Ball: tuning.scoring.
    # CNN window o'zgarsa, patchlar dataset.get_patches(window) bilan qayta olinadi.
    # Qaytaradi: {"best_params": dict (TO'LIQ param lug'ati), "best_score": float, "trials": [{"params","score","std"}], "n_trials": int}
```

### 3.5 `mpm/cv.py`

```python
def run_cv(dataset, groups, cv_mode, model_names, hp, *, n_splits, n_repeats, calibrate=True,
           calibration_method="sigmoid", calibration_cv=3, tuning=None, perm_importance=False,
           perm_repeats=5, n_jobs=1, seed=RANDOM_STATE, log_fn=None, progress_fn=None, cancel=None) -> dict
    # cv_mode "spatial" (stratified_group_splits) | "random" (random_stratified_splits).
    # tuning: TuningConfig|None; enabled bo'lsa har tashqi fold'da tunable va yoqilgan modellar uchun tune_model(train qismi).
    # Qaytaradi: {"mode", "oof": {name: [ (n,) float64 NaN-siz massivlar, har repeat uchun ]},
    #   "fold_map": (n,) int (1-repeat), "fold_table": DataFrame, "perm_importance": {name: summarize_perm_importance(...)} (faqat spatial,
    #   1-repeat, perm_importance=True va model tree/CNN emas -> RF, XGBoost, SVM uchun ham hisoblash mumkin; CNN ham patch bilan),
    #   "tuned_params": {name: [dict, ...]} (tashqi fold'lar bo'yicha), "feature_names": [...], "model_names": [...]}
    # Progress: har fold'dan keyin (frac, "Fold i/N ... ETA mm:ss"). Seed har fold uchun seed+split_num.
def compute_metrics(y, oof, groups=None, n_boot=1000, seed=RANDOM_STATE, log_fn=None) -> (dict, ensemble_mean_proba)
    # kalitlar: har model + ENSEMBLE_NAME. Har biri uchun:
    #  "auc"(repeat-o'rtacha), "auc_std"(repeat), "auc_ci95"(blok-bootstrap, (lo,hi)), "auc_single",
    #  "pr_auc","balanced_accuracy","f1","brier" va ularning "_std", "_ci95" (bootstrap) variantlari,
    #  "threshold_youden", "sensitivity","specificity" (Youden bo'sag'ida), "confusion" ([[tn,fp],[fn,tp]]),
    #  "fpr","tpr","precision","recall" (massivlar), "calibration": {"prob_pred","prob_true","counts"} (10 bin, kvantil),
    #  "mean_proba" (n,). balanced_accuracy/f1 bo'sag'i 0.5 (alohida kalitlar bilan Youden variantlari: "balanced_accuracy_youden","f1_youden").
def metrics_dataframe(results, label="") -> DataFrame   # bir qator har model: Model, AUC, AUC_std, AUC_CI_lo/hi, PR_AUC, BalAcc, F1, Brier, Sens, Spec, Thr
def youden_threshold(y, p) -> float
def run_background_sensitivity(aoi_gdf, positive_gdf, raster, pipeline, hp, *, model_names, n_background, min_distance,
        strategy, block_size, n_draws, n_splits, n_repeats, calibrate, calibration_method, calibration_cv,
        n_jobs, seed, log_fn=None, progress_fn=None, cancel=None) -> dict|None
    # {"per_draw": [{"draw","seed","auc":{name:auc}}], "summary": {name:{"mean","std","min","max","values"}}, "n_positive"}
```

### 3.6 `mpm/predict.py`

```python
def predict_probability_maps(raster, pipeline, final_models, *, batch_size=8192, log_fn=None, progress_fn=None,
                             cancel=None) -> dict
    # final_models: {name: [ModelWrapper,...]}. Har model xaritasi = draw'lar bo'yicha o'rtacha. Faqat valid piksellar.
    # Tabular modellar: pipeline.transform_pixels(batch); patch modellar: transform_stack bir marta + extract_patches(batch).
    # {"maps": {name: (H,W) float32 NaN-nodata, ENSEMBLE_NAME ham shu yerda (nanmean, ogohlantirishsiz)},
    #  "uncertainty": (H,W) float32|None (alohida modellar std'i; <2 model bo'lsa None), "valid_mask": (H,W) bool}
def classify_map(prob_map, method="quantile", n_classes=5, breaks=None) -> (class_map int8 (0=nodata, 1..n), breaks_used list)
    # quantile (teng maydonli), equal_interval (0-1 teng), fixed (breaks: o'suvchi chegaralar, n = len+1)
CLASS_LABELS(n) -> ["Juda past","Past","O'rta","Yuqori","Juda yuqori"] (n=5) yoki "1-sinf".. 
def class_area_stats(class_map, pixel_size, rows=None, cols=None) -> DataFrame
    # [Sinf, Nomi, Piksel, Maydon_km2, Maydon_%, Konlar_soni, Konlar_%, Boyitish (Konlar_% / Maydon_%)]
def success_rate_curve(prob_map, rows, cols) -> {"area_frac": arr, "capture_frac": arr, "auc": float, "n_pos": int}
    # piksellar ehtimol bo'yicha kamayish tartibida; musbat nuqtalarning (rows, cols) piksel qiymatlari bo'yicha ushlangan ulush
def save_rasters(out_dir, maps, profile, class_map=None, uncertainty=None, log_fn=None) -> list[paths]
    # prognoz_{safe_name}.tif (float32, nodata -9999), prognoz_classes.tif (int8, nodata 0), prognoz_uncertainty.tif
```

### 3.7 `mpm/persist.py`

```python
def save_bundle(directory, *, final_models, pipeline, hyperparams_used, cfg_dict, metrics_summary=None,
                thresholds=None, block_size=None, crs_epsg=TARGET_EPSG, notes="") -> directory
    # directory/manifest.json (versiyalar, vaqt, band_names, feature_names, model ro'yxati, cfg, metrics, ogohlantirish),
    # directory/pipeline.json, directory/models/<name>/<draw>/ (wrapper.save), directory/README.txt
def load_bundle(directory) -> dict   # {"final_models","pipeline","manifest","cfg","hyperparams_used",...}
def apply_bundle(bundle, tiff_folder, *, out_dir=None, batch_size=8192, assume_crs_if_missing=True,
                 class_method="quantile", n_classes=5, class_breaks=None, log_fn=None, progress_fn=None, cancel=None) -> dict
    # yangi papkadagi TIFF'larni band nomi bo'yicha (katta-kichik harf farqsiz) moslab, referens grid = papkadagi 1-fayl;
    # yetishmaydigan band bo'lsa ValueError (nomlar ro'yxati bilan). predict_probability_maps + classify + (out_dir bo'lsa) save_rasters.
```
Xavfsizlik: joblib/pickle — faqat ishonchli manbadan; GUI buni ogohlantiradi.

### 3.8 `mpm/pipeline.py`

```python
def run_training(cfg: RunConfig, *, log_fn=None, progress_fn=None, cancel=None) -> dict      # TrainingResult
def run_prediction(result, *, out_dir=None, class_method=None, n_classes=None, class_breaks=None, batch_size=8192,
                   log_fn=None, progress_fn=None, cancel=None) -> dict                       # PredictionOutput
def export_results(result, out_dir, prediction=None, log_fn=None) -> list[paths]
def estimate_cost_text(cfg: RunConfig) -> str      # taxminiy fold/model soni haqida ogohlantirish matni (GUI uchun)
```
**TrainingResult** kalitlari:
`cfg(dict), band_names, feature_names, categorical_layers, raster(RasterStack), pipeline(FeaturePipeline),
dataset(Dataset), groups, block_size, n_positive, n_background, spatial(CVBlock), random(CVBlock|None),
bg_sensitivity(dict|None), final_models({name:[ModelWrapper]}), final_hyperparams({name:[dict]}),
importance({"method", "models": {name: {"mean","std",...}}, "feature_names"}),
shap({name:{...}}|None), data_dictionary(DataFrame), metadata_csv_path, diagnostics(dict), thresholds({name: float}),
versions(dict), timings({stage: sec}), log_path(str|None), warnings([str]), aoi_gdf, positive_gdf, background_gdf`.
**CVBlock**: `{"mode","oof","metrics"(compute_metrics natijasi),"metrics_df","ensemble_oof","fold_map","fold_table",
"perm_importance","tuned_params","feature_names"}`.
**PredictionOutput**: `{"maps","uncertainty","valid_mask","class_map","class_breaks","class_stats"(DataFrame),
"success_curve","saved_paths"}`.

Pipeline tartibi (progress oralig'i): rasterlar+metadata (0-8%) -> nuqtalar/fon/dataset (8-12%) -> diagnostika (12-14%) ->
blok o'lchami (14-16%) -> random CV (16-36%, o'chirilishi mumkin) -> spatial CV (36-70%) -> bg sensitivity (70-80%) ->
final fit (80-92%) -> importance/SHAP (92-98%) -> yakun (100%). Patch-CNN uchun blok o'lchami < oyna*piksel bo'lsa
ogohlantirish (patchlar train/val orasida ustma-ust tushadi). Barcha ogohlantirishlar `result["warnings"]` ga ham yig'iladi.
Log fayli: `cfg.output_dir` berilgan bo'lsa `<output_dir>/mpm_run_<n>.log`.

### 3.9 `mpm/workers.py` (PyQt5)

`TrainingWorker(cfg)`, `PredictionWorker(result, out_dir, class_*)`, `ApplyBundleWorker(bundle_dir, tiff_folder, out_dir, ...)`,
`ExportWorker(result, out_dir, prediction)`. Hammasida: `cancel()` metodi (CancelToken), signal'lar:
`log_signal(str)`, `progress_signal(int 0-100, str)`, `finished_signal(dict)`, `error_signal(str)`, `cancelled_signal()`.
`run()` ichida `try/except CancelledError -> cancelled_signal`, qolgan xatolar `error_signal` (traceback bilan).

### 3.10 `mpm/gui/*`

* `widgets.py`: `FolderPicker`, `MplCanvas` (Figure + FigureCanvas + `NavigationToolbar2QT`, `.fig`, `.redraw()`),
  `DataFrameTable(QTableWidget)` (`.set_dataframe(df, float_fmt)`, CSV nusxalash), `ProgressPanel` (bar + stage + ETA).
* `plots.py` (Qt'siz, `matplotlib.figure.Figure` oladi; har biri bo'sh/yetarli bo'lmagan ma'lumotda xato bermaydi):
  `draw_roc(fig, metrics, title_suffix="")`, `draw_pr(fig, metrics, y)`, `draw_calibration(fig, metrics)`,
  `draw_confusion(fig, metrics, model)`, `draw_spatial_diagnostics(fig, result)`, `draw_importance(fig, result)`,
  `draw_shap_beeswarm(fig, result, model)`, `draw_shap_dependence(fig, result, model, feature)`,
  `draw_corr_heatmap(fig, corr)`, `draw_map(fig, array, transform, title, cmap, vmin, vmax, aoi_gdf=None, positives=None,
  background=None, class_labels=None, cbar_label="")`, `draw_success_rate(fig, curve)`, `draw_tuning_trials(fig, tuned_params)`,
  `save_all_figures(figs: dict[str, Figure], out_dir, formats=("png","pdf"), dpi=300) -> list[paths]`.
* `param_panel.py`: `HyperParamPanel(QWidget)` — PARAM_SPECS dan avtomatik; `.get_hyperparams() -> dict`, `.set_hyperparams(hp)`,
  `.reset_defaults(model=None)`, `.get_tuning() -> TuningConfig`, `.set_tuning(t)`, signal `changed`. Preset saqlash/yuklash tugmalari ichida.
  `SearchSpaceDialog(tuning)` — tunable parametr oraliqlarini jadvalda tahrirlash.
* `main_window.py`: `MainWindow` (tablar: 1 Ma'lumotlar va o'qitish, 2 Giperparametrlar, 3 Ma'lumotlar tahlili, 4 Natijalar,
  5 Spatial CV diagnostika, 6 Feature importance, 7 Prognoz xarita, 8 Modellar), `main()`; global `sys.excepthook`,
  `closeEvent`, `_set_busy`.
* Ildizda `mpm_ml_gui.py`: `from mpm.gui.main_window import main; main()`.
