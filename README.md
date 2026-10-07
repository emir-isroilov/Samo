# MPM ML GUI v2

Oltin ma'danlashuvi **prospektivligini** bashoratlash dasturi (Mineral Prospectivity Mapping): Random Forest, SVM, XGBoost,
CNN va ularning ansambli, **fazoviy (spatial block) cross-validation** bilan baholanadi. Dastur PyQt5 oynasida ishlaydi.
Bu `legacy/mpm_ml_gui_fixed.py` ning qayta yozilgan, xatolari tuzatilgan va kengaytirilgan versiyasi (v2.0.0).

> **Muhim.** Dastur chiqishi — **prospektivlik indeksi (0-1)**, kon topilish *ehtimolligi emas*. Musbat (kon) va fon nuqtalar
> nisbati sun'iy tanlanadi, shuning uchun indeks faqat maydonlarni o'zaro **taqqoslash va tartiblash** uchun mos.
> CNN chiqishi kalibrlanmaydi, shuning uchun ansambl o'rtachasida shkala farqi bo'lishi mumkin.

## Imkoniyatlar

- **Modellar:** Random Forest, SVM, XGBoost, CNN (`patch2d` — nuqta atrofidagi oyna ustida 2D-Conv, yoki eski `tabular1d`) va
  soft-voting ansambl; ehtimolliklar kalibrlanadi (sigmoid/isotonic).
- **Ishonchli baholash:** spatial block CV (bloklar train/validation orasida bo'linmaydi, blok o'lchami variogramdan avtomatik),
  random CV faqat taqqoslash uchun, blok-bootstrap 95% CI, PR-AUC, Brier, Sens/Spec @Youden, fon nuqtalar tanloviga sezgirlik.
- **Giperparametrlar:** barcha parametrlar GUI'da, preset (JSON) saqlash/yuklash; nested spatial CV bilan avtomatik qidirish (tuning).
- **Talqin:** out-of-fold permutation importance, SHAP (beeswarm, dependence), korrelyatsiya/VIF, kalibrlash va success-rate grafiklari.
- **Xarita:** butun maydon uchun GeoTIFF, sinflangan xarita (kvantil/teng oraliq/qo'lda chegaralar) va maydon statistikasi,
  modellar kelishmovchiligi (noaniqlik) xaritasi.
- **Modelni saqlash:** bundle (model + feature pipeline + sozlamalar) va uni yangi maydonga qo'llash.
- **Qulaylik:** hisoblash alohida oqimda (GUI qotmaydi), Stop tugmasi, progress va ETA, log fayli, hisoblash narxi haqida ogohlantirish.

## O'rnatish

Python 3.9 yoki yangiroq. Virtual muhitda:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`rasterio` va `geopandas` ba'zi tizimlarda `conda install -c conda-forge rasterio geopandas` orqali osonroq o'rnatiladi.
`xgboost`, `tensorflow` (CNN), `shap` va `openpyxl` ixtiyoriy: o'rnatilmasa tegishli model/tahlil ogohlantirish bilan o'chadi.
Faqat CPU uchun `tensorflow` o'rniga `tensorflow-cpu` o'rnatish mumkin. Batafsil: [`requirements.txt`](requirements.txt),
[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) (o'rnatish bo'limi).

## Ishga tushirish

```bash
python mpm_ml_gui.py
```

Kirish ma'lumotlari uchta papkada beriladi:

| Papka | Mazmuni |
|---|---|
| TIFF qatlamlar papkasi | har bir `.tif`/`.tiff` fayl — bitta predictor qatlam (fayl nomi = qatlam nomi) |
| Musbat nuqtalar papkasi | ma'lum kon/namoyon nuqtalarining `.shp` fayli (Point) |
| Maydon konturi (AOI) papkasi | tadqiqot maydoni poligonining `.shp` fayli |

Barcha ma'lumotlar EPSG:28411 (Pulkovo 1942 / Gauss-Kruger zone 11) ga o'tkaziladi. Dasturni ishlatish tartibi va barcha
sozlamalar tavsifi: [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

> **Eslatma (yon ta'sir).** O'qitish boshlanganda TIFF papkasida `metadata.csv` bo'lmasa, dastur u yerga bo'sh shablon yozadi.
> Uni to'ldiring (manba, sana, transformatsiya) va keyingi ishga tushirishda ishlating — natijadagi
> `predictor_data_dictionary.csv` shu ma'lumotdan quriladi.

## Dasturiy (GUI'siz) foydalanish

```python
from mpm.config import RunConfig
from mpm.pipeline import run_training, run_prediction, export_results

cfg = RunConfig(tiff_folder="data/tiff", points_folder="data/pts", aoi_folder="data/aoi",
                output_dir="natija", n_splits=5, n_repeats=3,
                use_models={"RandomForest": True, "SVM": True, "XGBoost": True, "CNN": False})
natija = run_training(cfg, log_fn=print)                    # TrainingResult (lug'at)
print(natija["spatial"]["metrics_df"])                      # asosiy natija: spatial block CV
prognoz = run_prediction(natija, out_dir="natija")          # GeoTIFF'lar + sinflash
export_results(natija, "natija", prediction=prognoz)        # CSV/XLSX/JSON/summary.txt
```

Boshqa funksiyalar (`persist.save_bundle/load_bundle/apply_bundle`) va ma'lumot sxemalari: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Testlar

```bash
pip install -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen python -m pytest                  # to'liq
QT_QPA_PLATFORM=offscreen python -m pytest -m "not slow"    # TensorFlow/to'liq pipeline testlarisiz
python -m pyflakes mpm tests
```

Testlar sintetik loyihada (`tests/synth.py`) ishlaydi, haqiqiy ma'lumot talab qilinmaydi.

## Loyiha tuzilishi

```
mpm_ml_gui.py        ishga tushirish fayli
mpm/                 dastur paketi: data, spatial, models, cnn, explain, tuning, cv, predict, persist, pipeline, workers
mpm/gui/             PyQt5 qatlami: widgets, plots, param_panel, result_tabs, main_window
tests/               avtomatik testlar (pytest)
docs/                USER_GUIDE.md, ARCHITECTURE.md, CHANGELOG.md
legacy/              asl kod (faqat ma'lumot uchun)
```

## Hujjatlar

- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) — foydalanuvchi qo'llanmasi (ma'lumot tayyorlash, tablar, natijalarni o'qish, muammolar).
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — modul shartnomalari va asl koddan nima tuzatilgani (BUG/ENH jadvali, testlar bilan).
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — o'zgarishlar tarixi.

## Xavfsizlik

Saqlangan modellar (RF/SVM/XGBoost) `joblib`/`pickle` formatida: bunday faylni yuklash o'zboshimchalik bilan kod bajarishi mumkin.
Model to'plamini (bundle) **faqat ishonchli manbadan** yuklang; dastur yuklashdan oldin ogohlantiradi.
