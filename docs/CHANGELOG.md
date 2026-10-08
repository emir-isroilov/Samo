# O'zgarishlar tarixi (CHANGELOG)

Format [Keep a Changelog](https://keepachangelog.com/) ga yaqin. Versiya: `mpm.__version__`.
Har bir band qaysi modulda va qaysi testda tasdiqlangani — [`ARCHITECTURE.md`](ARCHITECTURE.md) 4-bo'lim jadvalida.

## [2.0.0] — 2026-10-07

Asl dastur (`legacy/mpm_ml_gui_fixed.py`, bitta fayl) butunlay qayta yozildi: ilmiy qatlam alohida modullarga
ajratildi (`mpm/`), GUI `mpm/gui/` ga ko'chdi, hisoblash `QThread` ishchilarida bajariladi, avtomatik testlar qo'shildi.

### Muhim: natija talqini
- Chiqish endi **"prospektivlik indeksi (0-1)"** deb ataladi, ehtimollik emas: musbat:fon nisbati sun'iy tanlanadi, shuning
  uchun qiymatlar haqiqiy kon topish ehtimolini bildirmaydi, faqat maydonlarni nisbiy tartiblaydi (ENH-08).
  Fayl nomlari (`prognoz_*.tif`) o'zgarmadi.

### Tuzatildi (BUG-01 ... BUG-15)
- **BUG-01** Yangi `shap` RF uchun `(n, p, 2)` qaytarganda `draw_importance` xatosi PyQt5 slotida dasturni abort qilardi:
  `explain.normalize_shap_values` barcha shakllarni qamraydi, barcha grafiklar xatoga chidamli (`plots._safe`), ishchilar xatoni
  signalga o'raydi.
- **BUG-02** Prognoz xatosidan keyin prognoz tugmasi qulflanib qolardi: ishchilar alohida `error_signal` /
  `cancelled_signal` / `finished_signal` chiqaradi, tugmalar holati bitta `_set_busy` orqali tiklanadi.
- **BUG-03** MDI zaxirasi o'qitilmagan `CalibratedClassifierCV.estimator` dan o'qirdi: endi o'qitilgan bazaviy daraxtdan
  (`ModelWrapper.tree_model()` / `importance_mdi()`).
- **BUG-04** CNN erta to'xtatish train loss bo'yicha edi, CV (150) va final (200) epoch farq qilardi: endi stratified
  validatsiya ajratmasi + `val_loss`, bir xil giperparametrlar, har qurishdan oldin `clear_session`.
- **BUG-05** Spatial fold'larda musbat nuqtasiz validation fold chiqishi mumkin edi: `StratifiedGroupKFold` asosida
  `spatial.stratified_group_splits` (qayta urinish va musbat blokni ko'chirish).
- **BUG-06** `SVC(probability=...)` va `unary_union` eskirgan API'lar olib tashlandi (`CalibratedClassifierCV(SVC)`, `union_all()`).
- **BUG-07** Bashorat `batch_size=8192` bilan (CNN uchun 32 edi) va `n_jobs` qo'llab-quvvatlanadi.
- **BUG-08** AUC uchun ishonch oralig'i endi blok-bootstrap (nuqtalar bog'liqligini hisobga oladi); repeat'lar bo'yicha std alohida.
- **BUG-09** Variogram faqat birinchi band bo'yicha emas: barcha raqamli bandlar (yoki tanlangan band) medianasi, ko'proq nuqta.
- **BUG-10** Kategorik qatlamlar bilinear emas, `nearest` bilan moslanadi va modelga one-hot beriladi.
- **BUG-11** Fon nuqtalar yaroqsiz (nodata) piksellarga tushmaydi va so'ralgan soniga yetadi; generatsiya vektorlashtirilgan.
- **BUG-12** Kalibrlangan yakuniy model endi bazaviy modelni to'liq train ma'lumotida o'qitadi (`ensemble=False`); SHAP shu model uchun.
- **BUG-13** Musbat nuqta kam bo'lganda kalibrlash crash o'rniga moslashuvchan `cv`, imkonsiz bo'lsa ogohlantirish bilan o'tkazib yuboriladi.
- **BUG-14** Mayda tuzatishlar: bir pikseldagi dublikat musbatlar, MultiPoint, AOI tashqarisidagi nuqtalar, toza GeoTIFF profili
  (tiled/blocksize ko'chirilmaydi), `nanmean` ogohlantirishi, fon sezgirligi takrorlari GUI'dan boshqariladi, fold xaritasida
  legend, kesilmaydigan sarlavha/legend.
- **BUG-15** Giperparametrlar yagona `PARAM_SPECS` manbasidan olinadi va CV, tuning, yakuniy modelga bir xil uzatiladi.

### Qo'shildi (ENH-01 ... ENH-09)
- **ENH-01** To'liq giperparametr paneli (barcha `PARAM_SPECS` parametrlari avtomatik), preset JSON saqlash/yuklash,
  standartga qaytarish, seed va `n_jobs`; ishlatilgan giperparametrlar natija bilan `hyperparameters_used.json` ga yoziladi.
- **ENH-02** Giperparametrlarni avtomatik qidirish: ichki spatial CV bilan tasodifiy qidiruv, nested baholash
  (har tashqi fold'ning train qismida alohida), yakuniy tuning butun ma'lumotda; qidiruv oraliqlarini tahrirlash dialogi.
- **ENH-03** Haqiqiy CNN: `patch2d` (nuqta atrofidagi WxW oyna, 2D-Conv, flip/rot90 augmentatsiya); eski `tabular1d` taqqoslash uchun qoldirildi.
- **ENH-04** Metrikalar jadvali (AUC + CI, PR-AUC, BalAcc, F1, Brier, Sens/Spec @Youden), ROC, PR, kalibrlash, chalkashlik
  matritsasi, CSV/XLSX eksport, matplotlib navigatsiya paneli, barcha grafiklarni PNG/PDF ga saqlash.
- **ENH-05** Model to'plamini (bundle) saqlash/yuklash va yangi maydonga qo'llash (`persist.save_bundle/load_bundle/apply_bundle`).
- **ENH-06** Stop tugmasi (hamkorlikdagi bekor qilish), fold bo'yicha progress va ETA, log fayli (`mpm_run_<n>.log`),
  global `sys.excepthook`, oynani yopishda tasdiq.
- **ENH-07** Ma'lumotlar tahlili (qatlam statistikasi, korrelyatsiya, VIF, data dictionary), kategorik qatlamlar, fon strategiyalari
  (`random` / `grid` / `distance_weighted`), yakuniy model uchun K ta fon tanlovi ansambli, fon tanloviga sezgirlik tahlili,
  out-of-fold permutation importance + SHAP (beeswarm, dependence), xarita overlay (AOI, nuqtalar), sinflangan xarita va maydon
  statistikasi (`Boyitish` ustuni bilan), noaniqlik xaritasi, success-rate egri chizig'i.
- **ENH-08** "Prospektivlik indeksi" terminologiyasi, CNN kalibrlanmagani va success-rate optimistikligi haqida izohlar.
- **ENH-09** `requirements.txt`, `requirements-dev.txt`, avtomatik testlar (`tests/`), `README.md`, `docs/USER_GUIDE.md`,
  `docs/ARCHITECTURE.md`, shu `CHANGELOG.md`.
- Yangi: `estimate_cost_text` — hisoblash narxi (o'qitishlar soni, xotira) haqida GUI'da ko'rsatiladigan ogohlantirish.
- Yangi: Random CV faqat benchmark sifatida (tuning va importance'siz); random va spatial AUC farqi ("optimizm") logga va `summary.txt` ga yoziladi.
- Yangi: `export_results` — metrikalar (CSV/XLSX), `hyperparameters_used.json`, `run_config.json`, `importance.csv`,
  `tuned_params.csv`, `tuning_trials.csv`, `fold_table.csv`, `oof_predictions.csv`, `predictor_data_dictionary.csv`, `versions.json`,
  `class_stats.csv`, `summary.txt`.

### O'zgartirildi
- Arxitektura: bitta fayl o'rniga `mpm/` paketi (`data`, `spatial`, `models`, `cnn`, `explain`, `tuning`, `cv`, `predict`,
  `persist`, `pipeline`, `workers`, `gui/`); ildizdagi `mpm_ml_gui.py` faqat ishga tushiruvchi.
- Modellar `ModelWrapper` interfeysi ostida (RF / SVM / XGBoost / CNN); SVM har doim `StandardScaler` + Platt kalibrlash bilan.
- `tensorflow`, `xgboost`, `shap` modul darajasida import qilinmaydi (kechiktirilgan yuklash): GUI tez ochiladi, kutubxona yo'q bo'lsa
  tegishli model/tahlil ogohlantirish bilan o'chadi.
- Birinchi TIFF referens grid bo'lib qoladi; har bir TIFF fayl endi bir marta ochiladi (asl kodda birinchi fayl ikki marta ochilardi); grid mos kelsa resample qilinmaydi.
- Bog'liqliklar: `tensorflow>=2.16` (Keras 3 kerak), `shap>=0.42`, `affine`, `openpyxl` ixtiyoriy (batafsil `requirements.txt`).

- GUI (e2e tekshiruvdan keyin): natija tab'lari **tembel** to'ldiriladi (o'qitish tugagach GUI ~3 s qotmaydi: og'ir canvas
  chizishlari tab ochilganda, bosqichma-bosqich bajariladi); tab sarlavhalari qisqartirildi (to'liq nom - tooltip'da), oyna o'lchami
  ekranga moslanadi (min 980x640); 1-tab pastki paneli ixcham (log/hisob-kitob yig'iladi), 2-tab'da ikki qavat scroll yo'q,
  natija tab'larida canvas pastki qismi (x yorliqlari) ko'rinadi; uzun xato matni `...` bilan qisqartiriladi (to'liq matn tooltip'da);
  avto-eksport xatosi "Tayyor" holatini bosmaydi; 1 takrorda AUC_std jadvalda bo'sh va izoh bilan; `mpm_ml_gui.py` chiqish kodini
  qaytaradi; `MapTab.figures()` kalitlari `map_<Model>`; yuklangan konfiguratsiyadagi NaN qiymatlar vidjetga tegmaydi; bir xil TIFF
  papka qayta tanlanganda kategorik tanlov saqlanadi; `format_manifest` har bo'lim uchun xatoga chidamli.

### Yakuniy audit tuzatishlari
Mustaqil auditorlar topgan va takrorlangan nuqsonlar (regressiya testlari: `tests/test_audit_fixes.py`):
- `run_config.json` endi `"kind": "run_config"` bilan yoziladi (GUI/`load_preset` uni to'liq konfiguratsiya sifatida yuklaydi),
  asl `block_size` (0 = avto) saqlanadi, ishlatilgan qiymat esa `block_size_used` kalitida.
- Kvadrat bo'lmagan piksellarda sinf maydoni (km2) endi piksel yuzidan hisoblanadi (`RasterStack.pixel_size = sqrt(yuz)`).
- Qatlam statistikasi/`predictor_data_dictionary.csv` kichik (1e-9) va katta (1e13) masshtabli qatlamlarda ahamiyatli raqamlar
  bilan yoziladi (avval 0 ga yaxlitlanardi).
- Manfiy yoki juda katta `seed` fon generatsiyasi va diagnostikada ham ishlaydi (boshqa modullar kabi modulo).
- `apply_bundle`: kategorik qatlamda butun bo'lmagan qiymatlar haqida ogohlantirish.
- Fon nuqtalar generatsiyasi va blok-bootstrap Stop bilan to'xtatiladi; mavjud piksellardan ko'p fon so'ralganda urinishlar
  soni cheklanadi (avval ~85 s qotardi).
- Chiqish papkasi yozib bo'lmasa, o'qitish/prognoz/qo'llash hisoblashdan OLDIN aniq xato beradi (avval ish oxirida yo'qolardi).
- Natija tab'ida saqlash xatosi keyingi muvaffaqiyatli saqlashdan keyin yashiriladi.

### Olib tashlandi
- `SVC(probability=...)`, `GeoSeries.unary_union` (zaxira sifatida qoldi), sklearn `permutation_importance` (o'rniga AUC
  pasayishiga asoslangan o'z implementatsiyamiz), qattiq yozilgan CV/final giperparametrlari.

### Ma'lum cheklovlar
- Yakuniy tuning butun ma'lumotda bajariladi; nested spatial CV shu protsedurani baholaydi. `tuning.mode="final"` da spatial CV
  bazaviy giperparametrlar bilan baholanadi (GUI faqat `nested` rejimini ko'rsatadi).
- Hisoblash ~95% progressdan keyin bekor qilinmaydi (qolgan ish millisekundlar).
- success-rate o'qitish nuqtalarida hisoblanadi (optimistik); noaniqlik xaritasi — modellar kelishmovchiligi, statistik oralig'i emas.
- CNN chiqishi kalibrlanmaydi, ansambl oddiy o'rtacha bo'lgani uchun shkalalar farq qilishi mumkin.
- Faqat `tensorflow-cpu` o'rnatilgan muhitda `versions.json`/bundle manifestida `tensorflow` versiyasi `null` bo'ladi.
- Tafsilotlar: [`ARCHITECTURE.md`](ARCHITECTURE.md) 5-bo'lim.
