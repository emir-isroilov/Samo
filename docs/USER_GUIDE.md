# MPM ML GUI v2 — foydalanuvchi qo'llanmasi

Bu qo'llanma dasturni birinchi marta ishlatayotgan geolog/tadqiqotchi uchun: o'rnatish, ma'lumotni tayyorlash, har bir tab,
giperparametrlar va tuning, natijalarni to'g'ri o'qish, xaritani talqin qilish, modelni saqlash/qo'llash va muammolarni bartaraf etish.
Texnik tafsilotlar (funksiya imzolari, ma'lumot sxemalari) — [`ARCHITECTURE.md`](ARCHITECTURE.md), o'zgarishlar — [`CHANGELOG.md`](CHANGELOG.md).

## Mundarija

1. [Dastur nima qiladi](#1-dastur-nima-qiladi)
2. [O'rnatish](#2-ornatish)
3. [Ma'lumotni tayyorlash](#3-malumotni-tayyorlash)
4. [Dastur oynasi: tablar](#4-dastur-oynasi-tablar)
5. [Giperparametrlar va tuning](#5-giperparametrlar-va-tuning)
6. [Natijalarni o'qish](#6-natijalarni-oqish)
7. [Xaritani talqin qilish](#7-xaritani-talqin-qilish)
8. [Modelni saqlash va yangi maydonga qo'llash](#8-modelni-saqlash-va-yangi-maydonga-qollash)
9. [Xavfsizlik](#9-xavfsizlik)
10. [Hisoblash vaqti bo'yicha maslahatlar](#10-hisoblash-vaqti-boyicha-maslahatlar)
11. [Chiqish fayllari](#11-chiqish-fayllari)
12. [Muammolarni bartaraf etish](#12-muammolarni-bartaraf-etish)
13. [Dasturiy (GUI'siz) foydalanish](#13-dasturiy-guisiz-foydalanish)
14. [Atamalar lug'ati](#14-atamalar-lugati)

---

## 1. Dastur nima qiladi

Dastur ma'lum kon/namoyon nuqtalari (**musbat** nuqtalar) va tadqiqot maydonidan olingan tasodifiy **fon** (pseudo-absence)
nuqtalarida predictor qatlamlar (TIFF) qiymatlarini oladi, bir nechta mashinaviy o'qitish modeli (Random Forest, SVM, XGBoost, CNN)
va ularning o'rtachasi (ansambl, "soft-voting") ni o'qitadi, sifatini **spatial block cross-validation** bilan baholaydi va butun
maydon uchun **prospektivlik indeksi** xaritasini (GeoTIFF) yaratadi.

Ish oqimi:

```
TIFF qatlamlar + musbat nuqtalar + AOI
   -> bir gridga keltirish (EPSG:28411), fon nuqtalar, dataset
   -> spatial blok o'lchami (variogram), random va spatial CV (+ ixtiyoriy tuning)
   -> yakuniy modellar (butun ma'lumotda), importance / SHAP
   -> prognoz xaritasi, sinflash, maydon statistikasi, eksport, modelni saqlash
```

> ### Prospektivlik indeksi — ehtimollik emas
> Dastur chiqishi **"prospektivlik indeksi (0-1)"**. Bu *kon topilish ehtimolligi emas*:
> * musbat:fon nisbati sun'iy tanlanadi (masalan 22 musbat : 80 fon). Modelning "ehtimolligi" shu o'qitish namunasidagi
>   sinf nisbatiga nisbatan olinadi, haqiqiy geologik chastotaga emas. Fon nuqtalar sonini o'zgartirsangiz, qiymatlar shkalasi ham o'zgaradi;
> * fon nuqtalar "kon yo'q" nuqtalar emas, shunchaki tasodifiy tanlangan joylar (ularning orasida kon bo'lishi mumkin);
> * indeks maydonlarni o'zaro **taqqoslash va tartiblash** uchun: "A hudud B hududdan prospektivroq".
>
> Xaritani "ma'dan topilish ehtimoli 80%" deb o'qimang va maqolada shunday yozmang.

## 2. O'rnatish

### 2.1 Talablar

* Python 3.9 yoki yangiroq (sinalgan: 3.13).
* Operatsion tizim: Linux, Windows, macOS (PyQt5 mavjud bo'lgan tizimlar).
* Katta rasterlar va CNN ko'p operativ xotira talab qiladi (10-bo'limga qarang); CNN CPU'da ham ishlaydi, GPU shart emas.

### 2.2 Virtual muhit va kutubxonalar

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Windows/macOS'da `rasterio` va `geopandas` ba'zan `pip` bilan o'rnatilmaydi; shunda conda ishlating:

```bash
conda create -n mpm python=3.11
conda activate mpm
conda install -c conda-forge rasterio geopandas
pip install -r requirements.txt
```

### 2.3 Ixtiyoriy kutubxonalar

Quyidagilar o'rnatilmasa dastur ishlashda davom etadi, faqat tegishli imkoniyat o'chadi (logda ogohlantirish chiqadi):

| Kutubxona | O'rnatilmasa nima bo'ladi |
|---|---|
| `xgboost` | XGBoost modeli o'tkazib yuboriladi |
| `tensorflow` (>= 2.16, Keras 3; faqat CPU uchun `tensorflow-cpu`) | CNN modeli o'tkazib yuboriladi |
| `shap` | SHAP grafiklari (beeswarm/dependence) hisoblanmaydi; boshqa importance ishlayveradi |
| `openpyxl` | natijalar faqat CSV ga yoziladi (XLSX yozilmaydi) |

Kamida bitta model (RandomForest yoki SVM) ishlatib bo'ladigan bo'lishi shart, aks holda o'qitish xato bilan to'xtaydi.

### 2.4 Ishga tushirish

```bash
python mpm_ml_gui.py
```

### 2.5 O'rnatishni tekshirish (ixtiyoriy)

```bash
pip install -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen python -m pytest -m "not slow"       # tez testlar
```

Testlar sintetik loyihada ishlaydi, ekran (display) shart emas (`QT_QPA_PLATFORM=offscreen`).

## 3. Ma'lumotni tayyorlash

Dastur uchta papka oladi. Hammasi **EPSG:28411** (Pulkovo 1942 / Gauss-Kruger zone 11) tizimiga o'tkaziladi.

### 3.1 TIFF qatlamlar papkasi

* Har bir `.tif`/`.tiff` fayl = **bitta predictor qatlam**. Fayl nomi (kengaytmasiz) qatlam nomi bo'ladi va grafiklarda,
  jadvallarda, `metadata.csv` da shu nom ishlatiladi. Nomlar takrorlanmasligi kerak (`a.tif` va `a.tiff` birga bo'lmasin).
* Fayllar alifbo tartibida o'qiladi (katta-kichik harf farqsiz). **Birinchi fayl referens grid** bo'ladi: boshqa qatlamlar
  shu gridga (piksel o'lchami, qamrov) moslanadi. Chiqish xaritalarining piksel o'lchami va qamrovi shu referensniki bo'ladi: kerakli qatlamni
  alifbo bo'yicha birinchi qiling (masalan `01_dem.tif`).
* Faqat **1-band** o'qiladi. Ko'p bandli faylda 1-banddan boshqasi e'tiborsiz qoldiriladi (ogohlantirish chiqadi).
* Yashirin fayllar (nomi `.` bilan boshlanadigan, masalan macOS `._x.tif`) o'tkazib yuboriladi.
* Fayl georeferenslangan bo'lishi shart. CRS yozilmagan bo'lsa: 1-tabdagi *CRS yo'q bo'lsa EPSG:28411 deb qabul qilish* belgisi yoqilgan
  bo'lsa, koordinatalar allaqachon EPSG:28411 da deb olinadi (reproyeksiya qilinmaydi, ogohlantirish bilan); o'chirilgan bo'lsa xato beriladi.
* **Nodata, cheksiz (inf) va |qiymat| > 1e15 qiymatlar `NaN` ga aylantiriladi.** Modelga faqat **barcha qatlamlar bir vaqtda chekli**
  bo'lgan piksellar (kesishma) qatnashadi: qatlamlar qamrovi bir-biriga mos kelmasa, ishlatiladigan maydon kichrayadi.
  Qatlam maydonning 50% dan kamini qoplasa yoki deyarli o'zgarmas bo'lsa, ogohlantirish chiqadi.
* Resample: odatdagi qatlamlar **bilinear**, kategorik qatlamlar **nearest** bilan moslanadi (3.5-bo'lim).
* Qatlamlar o'qitish va qo'llash paytida bir xil birlik va ma'noda bo'lishi shart.

### 3.2 Musbat nuqtalar papkasi (`.shp`)

* Papkadagi alifbo bo'yicha **birinchi** `.shp` fayl ishlatiladi (shapefile bilan birga `.shx`, `.dbf`, `.prj` bo'lishi kerak).
* Faqat **Point** geometriyalar qoladi. `MultiPoint` alohida nuqtalarga ajratiladi; boshqa geometriya turlari tashlanadi.
* **AOI tashqarisidagi** nuqtalar ogohlantirish bilan tashlanadi (hammasi tashqarida bo'lsa — xato: CRS va hududni tekshiring).
* **Bir pikselga tushgan** nuqtalardan faqat birinchisi qoldiriladi (dublikat nazorati).
* Nodata pikselga tushgan nuqtalar dataset'dan tashlanadi (log'da nechtasi ekani yoziladi).
* Minimal talab: dataset'da kamida **10 ta musbat nuqta** va k-fold sonidan kam bo'lmagan musbat nuqta. Spatial CV uchun musbat
  nuqtalar kamida k-fold ta **turli blokda** bo'lishi kerak (dastur blok o'lchamini kerak bo'lsa avtomatik kichraytiradi).
  Kam nuqtada ishonch oralig'i keng bo'ladi, natijaga ehtiyotkorlik bilan qarang.
* Feature'lar soniga nisbatan musbat nuqtalar kam bo'lsa (nuqtalar/feature < 10) overfitting haqida ogohlantirish chiqadi:
  kamroq/ma'noliroq qatlam tanlang.

### 3.3 Maydon konturi (AOI) papkasi (`.shp`)

* Papkadagi birinchi `.shp` fayl; **poligon** (yoki multipoligon) geometriyalar. Noto'g'ri (o'z-o'zini kesadigan) poligonlar avtomatik tuzatiladi.
* Fon nuqtalar faqat AOI ichida **va** barcha qatlamlar chekli bo'lgan piksellarda tanlanadi.
* AOI qatlamlar qamroviga mos bo'lishi kerak; AOI qatlamlardan ancha katta bo'lsa, fon nuqtalar yetishmasligi mumkin.

### 3.4 `metadata.csv` (data dictionary uchun)

Maqolada har bir predictor qatlamning manbasi, sanasi va qayta ishlanishi ko'rsatilishi kerak. Dastur buning uchun TIFF papkasidagi
`metadata.csv` dan foydalanadi.

> **Yon ta'sir:** o'qitishni boshlaganingizda TIFF papkasida `metadata.csv` bo'lmasa, dastur u yerga **bo'sh shablon yozadi**
> (har qatlam uchun bitta qator). Mavjud faylni hech qachon ustiga yozmaydi. Shablonni to'ldirib, keyingi ishga tushirishda ishlating.

Ustunlar (birinchi qator — sarlavha):

| Ustun | Mazmuni |
|---|---|
| `band_name` | qatlam nomi (fayl nomi, kengaytmasiz) — majburiy, shu bo'yicha moslanadi |
| `source_owner` | ma'lumot egasi/manbasi |
| `survey_or_scene_id` | s'yomka yoki sahna identifikatori |
| `survey_date` | sana |
| `original_scale_or_resolution` | asl masshtab yoki piksel o'lchami |
| `transformation_applied` | qo'llangan qayta ishlash (filtr, normallashtirish...) |
| `notes` | izoh |

Talablar: **UTF-8** kodlash, ajratgich — **vergul** (`,`). Excel ba'zan `;` ajratgich va cp1251 kodlash bilan saqlaydi: dastur
buni aniqlab ogohlantiradi; "CSV UTF-8 (vergul bilan ajratilgan)" formatida qayta saqlang. `source_owner` bo'sh qatlamlar soni
ogohlantirishda ko'rsatiladi. Natija — `predictor_data_dictionary.csv`: avtomatik texnik ma'lumotlar (asl CRS, piksel o'lchami,
qiymatlar statistikasi, valid piksel %) va sizning qo'lda kiritgan ma'lumotlaringiz birga.

### 3.5 Kategorik qatlamlar

Geologik formatsiya, tuproq turi, litologiya kabi **sinf kodlari** qatlamlari "kategorik" deb belgilanishi kerak. Aks holda model
ularni oddiy son (masalan 3 > 2) deb o'qiydi va bilinear resample sinf kodlarini buzadi (2.5 kabi mavjud bo'lmagan qiymatlar paydo bo'ladi).

* TIFF papka tanlanganda dastur kategorik bo'lishi mumkin qatlamlarni taklif qiladi: butun sonli dtype va noyob qiymatlar <= 30,
  yoki float dtype, lekin barcha qiymatlar butun va noyob qiymatlar <= 20. Taklif faqat maslahat: dastur taklif qilmagan, lekin aslida sinf kodlari bo'lgan qatlamni
  qo'lda belgilang; uzluksiz qatlam (masalan DEM) taklif qilingan bo'lsa, belgisini olib tashlang.
* Kategorik qatlam **nearest** bilan resample qilinadi va modelga **one-hot** ustunlar sifatida beriladi: har sinf uchun
  `qatlam==daraja` nomli ustun. Grafiklarda va importance jadvalida shu nomlar ko'rinadi.
* Daraja soni 30 dan oshsa yoki qiymatlar butun bo'lmasa, dastur xato beradi ("uzluksiz qatlam bo'lishi mumkin: kategorik belgisini olib tashlang").
* Yangi maydonga qo'llashda o'qitishda bo'lmagan yangi daraja uchrasa, uning barcha one-hot ustunlari 0 bo'ladi (ogohlantirish bilan).
* Permutation importance one-hot ustunlarni **alohida** aralashtiradi: bitta kategorik qatlamning ahamiyati bir necha ustunga bo'linib ko'rinadi.

### 3.6 Fon (pseudo-absence) nuqtalar

| Sozlama | Ma'nosi |
|---|---|
| Fon nuqtalar soni (standart 80) | nechta fon nuqta; musbat:fon nisbati sun'iy ekanini unutmang |
| Min. masofa (standart 500 m) | fon nuqta musbat nuqtadan kamida shuncha uzoqda bo'ladi |
| Strategiya | `random` — AOI va valid piksellarda tekis; `grid` — AOI ustida jitterli to'r tuziladi, fon nuqtalar shu to'r nuqtalaridan tanlanadi; `distance_weighted` — musbat nuqtalardan uzoqroq joylarga ko'proq ehtimol |
| Yakuniy fon ansambli (K) | yakuniy modelni K ta turli fon tanlovida o'qitib, xaritalarni o'rtachalash (fon tanloviga bog'liqlikni kamaytiradi); K ta narxni oshiradi |

Fon tanlovi natijaga ta'sir qiladi, shuning uchun **fon sezgirligi tahlili**ni (1-tab) yoqib, natija (AUC) bir necha mustaqil fon tanlovida
qanchalik barqarorligini tekshiring (6.6-bo'lim). Har bir piksel faqat bitta nuqta oladi va musbat nuqta pikseliga fon tushmaydi.
Yetarli joy topilmasa dastur nechta fon topilganini ogohlantirib yozadi (min. masofani kamaytiring).

### 3.7 Tekshiruv ro'yxati (o'qitishdan oldin)

- [ ] Barcha TIFF'lar bir-biriga geografik jihatdan mos va CRS bor (yo'q bo'lsa — rostdan ham EPSG:28411 ekaniga ishonch hosil qiling).
- [ ] Referens grid (alifbo bo'yicha birinchi fayl) maqsadli piksel o'lchamida.
- [ ] Kategorik qatlamlar belgilangan.
- [ ] Musbat nuqtalar AOI ichida (dastur kamida 10 ta talab qiladi, ishonchli natija uchun ko'proq), dublikatlar yo'q.
- [ ] `metadata.csv` to'ldirilgan (yoki keyinroq to'ldirishga tayyorsiz).
- [ ] Chiqish papkasi tanlangan (natijalar va log fayli shu yerga yoziladi).

### 3.8 Sinov uchun sintetik ma'lumot

O'zingizning ma'lumotingizsiz dasturni sinab ko'rish uchun repo ichida generator bor:

```python
from tests.synth import make_synthetic_project
loyiha = make_synthetic_project("sinov_loyiha", size=120, n_layers=6, categorical=True)
print(loyiha["tiff"], loyiha["points"], loyiha["aoi"])      # uchta papka yo'li
```

## 4. Dastur oynasi: tablar

Oyna 8 ta tabdan iborat. Uzoq ishlar (o'qitish, prognoz, eksport, bundle'ni qo'llash) alohida oqimda bajariladi — oyna qotmaydi.
Ish davomida sozlamalar va tugmalar bloklanadi; ish tugagach, xato bo'lsa yoki **To'xtatish (Stop)** bosilsa, hammasi qayta yoqiladi.

Tavsiya etiladigan tartib: **1-tab** (papkalar, sozlamalar) -> **2-tab** (giperparametrlar, ixtiyoriy) -> *(ixtiyoriy)* **3-tab**
"Tahlilni hisoblash" -> **1-tab** "O'qitish" -> **4-6 tab** natijalarni o'qish -> **7-tab** xarita -> **8-tab** modelni saqlash.

### 4.1 Tab 1 — Ma'lumotlar va o'qitish

* **Papkalar:** TIFF qatlamlar, musbat nuqtalar (`.shp`), maydon konturi (AOI, `.shp`) va **chiqish papkasi**. Chiqish papkasi
  berilsa, log fayli (`mpm_run_<n>.log`) va o'qitish tugagach natijalar (CSV/XLSX/JSON, `summary.txt`) avtomatik shu yerga yoziladi.
* **Kategorik qatlamlar:** TIFF papka tanlanganda qatlamlar ro'yxati chiqadi, kategorik bo'lishi mumkinlari avtomatik belgilanadi
  (3.5-bo'lim); belgilarni o'zgartirishingiz mumkin.
* **Fon nuqtalar:** soni, min. masofa, strategiya, yakuniy fon ansambli (K) (3.6-bo'lim).
* **Cross-validation:**
  * *k-fold* (standart 5) va *takrorlar soni* (standart 10): CV butun takrorlanadi, AUC "o'rtacha ± std" sifatida beriladi.
    Musbat nuqta kam bo'lganda bitta fold bo'linishiga bog'liq tebranishni kamaytiradi. **Hisoblash vaqti taxminan takrorlar soniga proporsional.**
  * *Spatial blok o'lchami (m)*: **0 = avtomatik** (empirik semivariogram bo'yicha avtokorrelyatsiya masofasi; variogram baholanmasa
    1000 m). Bir blokdagi nuqtalar CV'da hech qachon train va validation orasida bo'linmaydi. Blok o'lchami k-fold uchun yetarli
    blok bermasa, dastur uni avtomatik yarmiga kamaytiradi (log'da ko'rinadi).
  * *Variogram bandi*: `auto` — barcha raqamli bandlarning medianasi (tavsiya), yoki bitta tanlangan band.
  * *Random CV (benchmark)*: yoqilsa, taqqoslash uchun oddiy tasodifiy CV ham bajariladi (6.1-bo'lim). Vaqtni taxminan ikki baravar oshiradi.
  * *Bootstrap soni* (standart 1000): blok-bootstrap ishonch oralig'i uchun resample'lar soni; **0 = ishonch oralig'i hisoblanmaydi** (tez rejim).
* **Modellar:** Random Forest, SVM, XGBoost, CNN yoqish/o'chirish. O'rnatilmagan kutubxonali model o'chirilgan holda ko'rinadi.
  Ansambl modellar o'rtachasidan hosil bo'ladi (kamida bitta model kerak).
* **Kalibrlash:** yoqish, usul (`sigmoid` yoki `isotonic`) va ichki fold soni (`calibration_cv`, standart 3). 6.4-bo'limga qarang.
* **Fon sezgirligi tahlili:** yoqish, tanlovlar soni va har tanlovdagi CV takrorlari soni (standart 5 va 3).
* **Talqin:** permutation importance takrorlari (standart 5), SHAP yoqish va SHAP fon nuqtalari soni (standart 50).
* **Umumiy:** *seed* (natijani takrorlash uchun; bir xil ma'lumot va seed — bir xil natija), *n_jobs* (parallel ishchilar; standart —
  CPU soni, lekin ko'pi bilan 4), CRS taxmin belgisi.
* **Hisoblash narxi:** sozlamalar o'zgarganda taxminiy o'qitishlar soni va ogohlantirishlar ko'rsatiladi (tuning, CNN, ko'p takror
  qimmat ekanini oldindan bilish uchun; 10-bo'limga qarang).
* **Tugmalar:** **O'qitish**; **To'xtatish (Stop)**; **Konfiguratsiyani saqlash/yuklash (JSON)** (barcha sozlamalar — papkalar, CV,
  modellar, giperparametrlar, tuning — bitta faylda). Progress paneli bosqich, foiz, o'tgan vaqt va taxminiy qolgan vaqtni (ETA) ko'rsatadi;
  ETA butun jarayon bo'yicha chiziqli ekstrapolyatsiya, shuning uchun taxminiy. **Log** oynasi va **Logni saqlash** tugmasi.
* **Menyu:** *Fayl* (konfiguratsiyani saqlash/yuklash, chiqish), *Yordam* (dastur haqida, prospektivlik indeksi izohi).

O'qitish boshlanishidan oldin sozlamalar tekshiriladi (kamida bitta model, k-fold >= 2, fon nuqtalar soni >= 5 va h.k.); muammo bo'lsa xabar chiqadi.

**Stop haqida:** to'xtatish hamkorlikda ishlaydi — hisoblash keyingi tekshiruv nuqtasida (fold, model, bosqich chegarasi) to'xtaydi,
shuning uchun darhol emas. O'qitishning ~95% progressdan keyingi qismi (importance/SHAP yakuni) to'xtatilmaydi — u millisekundlar oladi,
shu sabab "tugadi" signali kelsa ham Stop bosilgan bo'lishi mumkin.

### 4.2 Tab 2 — Giperparametrlar

Har bir model uchun alohida sub-tab (Random Forest, SVM, XGBoost, CNN); parametrlar guruhlarga bo'lingan, har birining ustiga sichqoncha
olib borilsa izoh, diapazon va standart qiymat ko'rinadi. Parametrlar to'liq ro'yxati va ma'nosi — 5-bo'lim.

* **Standart qiymatlarga qaytarish (joriy model)** va **Hammasini standartga qaytarish**;
* **Preset saqlash/yuklash (JSON):** giperparametrlar va tuning sozlamalari faylga;
* **Tuning** guruhi (5.4-bo'lim);
* Hisoblash narxi bo'yicha izoh (sozlama o'zgarganda yangilanadi).

Modellarni yoqish/o'chirish bu tabda emas, 1-tabda. Bog'liq parametrlar avtomatik o'chiriladi (qiymati saqlanadi): CNN `tabular1d`
rejimida `window` va `augment`, RF'da bootstrap o'chirilganda `max_samples`. Toq bo'lishi shart parametrlar (CNN `window`, `kernel_size`)
juft kiritilsa +1 qilinadi.

### 4.3 Tab 3 — Ma'lumotlar tahlili

O'qitishdan **oldin** ham, **keyin** ham foydalanish mumkin. **"Tahlilni hisoblash (TIFF'lardan)"** tugmasi TIFF papkasini o'qib
(o'qitmasdan) statistika, korrelyatsiya va VIF ni qayta hisoblaydi. O'qitish tugagach tab o'qitish natijasidagi diagnostika bilan o'zi
to'ladi. Ichki tablar:

* **Qatlam statistikasi:** `valid_pct` (yaroqli piksellar ulushi), min, maks, o'rtacha, std, turi (`raqamli`/`kategorik`);
* **Korrelyatsiya:** Pearson heatmap (faqat raqamli qatlamlar; |r| >= 0.9 juftliklar qora ramka bilan);
* **VIF:** multikollinearlik (`past` < 5, `o'rta` 5-10, `yuqori` >= 10; o'zgarmas qatlam alohida belgilanadi; qatorlar rangga bo'yaladi);
* **Yuqori korrelyatsiya:** |r| >= 0.9 juftliklar jadvali;
* **Data dictionary:** texnik ma'lumotlar (avtomatik) + `metadata.csv` dan qo'lda kiritilgan maydonlar;
* **Musbat/fon farqi** (o'qitishdan keyin): o'rtachalar, Cohen d va bir o'zgaruvchili AUC — tavsifiy, model baholash emas.

Jadvallar: ustun sarlavhasi bo'yicha saralash, Ctrl+C bilan nusxalash, o'ng tugma menyusida "CSV ga saqlash...".

### 4.4 Tab 4 — Natijalar

Yuqorida **CV rejimi** tanlanadi: **Spatial block CV (asosiy)** yoki **Random CV (benchmark)** (random CV o'chirilgan bo'lsa u tanlanmaydi).
Pastda:

* **metrikalar jadvali:** har model va ansambl uchun AUC (o'rtacha), AUC_std, 95% CI, PR-AUC, BalAcc, F1, Brier, Sens/Spec va Youden bo'sag'i (6.2-bo'lim);
  jadval ustida izoh: AUC va CI qanday hisoblangani, tuning bo'lsa baholash qanday o'tgani, CNN haqida eslatma;
* ichki tablar: **ROC**, **PR**, **Kalibrlash**, **Chalkashlik matritsasi** (bu yerda *Model* tanlanadi; standart — ansambl);
* tugmalar: **Jadvalni CSV/XLSX ga saqlash**, **Barcha grafiklarni saqlash (PNG+PDF)** (bu tabning grafiklari va boshqa tablarning grafiklari),
  **Natijalarni eksport (CSV/JSON/XLSX)** (to'liq eksport, 11-bo'lim).

Har grafikda matplotlib navigatsiya paneli bor: kattalashtirish (zoom), surish (pan), tiklash, rasmni saqlash.

### 4.5 Tab 5 — Spatial CV diagnostika

Yuqorida umumiy grafik: random va spatial CV AUC yonma-yon (CI bilan), birinchi takrorning **fold xaritasi** (qaysi nuqta qaysi fold'da)
va (yoqilgan bo'lsa) fon sezgirligi. Pastda ichki tablar:

* **Fold jadvali:** har fold'dagi train/validation nuqtalar, musbat nuqtalar, bloklar soni, har model uchun sekundlar (`seconds_<model>`);
  musbat nuqtasiz validation fold bo'lsa, ogohlantirish ko'rsatiladi;
* **Fon sezgirligi:** AUC taqsimoti (o'rtacha, std, min, maks) va har tanlov natijasi (6.6-bo'lim);
* **Tuning (nested):** har tashqi fold uchun tanlangan va bazaviy ball, o'zgargan parametrlar bo'yicha tanlangan qiymatlar (tuning yoqilgan bo'lsa);
* **Ogohlantirishlar (N):** o'qitish davomida yig'ilgan barcha ogohlantirishlar (kalibrlash o'tkazib yuborilgani, importance zaxirasi,
  CNN leakage xavfi...). Ularni o'qing.

### 4.6 Tab 6 — Feature importance

Tepada **"Importance usuli: ..."** (permutation yoki MDI zaxirasi) va birliklar haqida izoh. Ichki tablar:

* **Importance (perm + SHAP):** permutation importance (dAUC, fold std xato chizig'i) va SHAP |o'rtacha| ustunli grafiklari, modellar yonma-yon;
* **SHAP beeswarm** (*Model* tanlanadi) va **SHAP dependence** (*Model* va *Feature* tanlanadi; standart — eng muhim feature);
* **Jadval:** feature bo'yicha qiymatlar: har model uchun "o'rtacha [manba]" va "std" ustunlari (manba: `permutation` yoki `mdi`; MDI zaxirasida std bo'sh).

SHAP birliklari: RF — ehtimollik, XGBoost — log-odds; modellararo magnitudani bitta o'qda solishtirmang. 30 tadan ko'p feature bo'lsa
grafiklarda eng muhim 20 tasi ko'rsatiladi. Talqin qoidalari — 6.7-bo'lim.

### 4.7 Tab 7 — Prognoz xarita

* **Chiqish papkasi** (GeoTIFF'lar va jadvallar shu yerga saqlanadi; o'qitishda chiqish papkasi berilgan bo'lsa, shu papka taklif qilinadi);
* **Sinflash sozlamalari:** *Usul* — "Kvantil (teng maydonli sinflar)", "Teng intervalli (0-1)" yoki "Qat'iy chegaralar"; *Sinflar soni*
  (2-12; kvantil va teng intervalli uchun); *Qat'iy chegaralar* (faqat "Qat'iy chegaralar" usulida; standart `0.2, 0.4, 0.6, 0.8`):
  chegaralar 0 va 1 orasida, qat'iy o'suvchi, vergul/nuqta-vergul/probel bilan ajratiladi; sinflar soni = chegaralar soni + 1. Noto'g'ri
  kiritilsa, tab ichida xato matni ko'rinadi va prognoz boshlanmaydi;
* **Prognoz xarita yaratish** tugmasi — faqat o'qitilgan natija mavjud bo'lsa yoqiladi; **Natijalarni eksport (CSV/JSON/XLSX)**;
* ichki tablar:
  * **Xarita:** *Qatlam* ro'yxati — har model va ansambl "prospektivlik indeksi", "Noaniqlik (modellar o'rtasida std)", "Sinflangan xarita";
    qoplamalar: "AOI chegarasi" (standart yoqilgan), "Musbat nuqtalar (konlar)", "Fon nuqtalar". Rang shkalasi "Prospektivlik indeksi (0-1)";
  * **Success-rate:** egri chiziq (6.9-bo'lim) va izoh: o'qitish nuqtalarida hisoblangan, optimistik;
  * **Sinf statistikasi:** piksel, maydon (km2, %), kon soni va %, boyitish (6.9-bo'lim); `Konlar_*` bo'sh = noma'lum.

Tab ustida har doim izoh turadi: *"Chiqish - prospektivlik indeksi (0-1), haqiqiy ehtimollik emas: fon nisbati sun'iy"*; CNN boshqa
modellar bilan birga bo'lsa, CNN kalibrlanmagani haqida qo'shimcha eslatma ko'rinadi.
Juda katta rasterlarda xarita ekranda tomoni ~2500 pikselgacha kamaytirilgan (stride) holda chiziladi; saqlangan GeoTIFF'lar va
eksport qilingan rasmlar yuqori aniqlikda. Hamma tablardagi grafiklar va jadvallar bo'sh holatda "Hali natija yo'q" deb ko'rsatadi.

### 4.8 Tab 8 — Modellar

* **Modelni saqlash (bundle):** yakuniy modellar, feature pipeline va sozlamalar papkaga yoziladi (8-bo'lim).
* **Modelni yuklash:** bundle papkasini tanlash; **xavfsizlik ogohlantirishi** ko'rsatiladi va ishonchli manbadan ekaningizni tasdiqlashingiz so'raladi (9-bo'lim).
  Yuklangan bundle haqida ma'lumot: qatlam nomlari, modellar, giperparametrlar, yaratilgan vaqt.
* **Yangi maydonga qo'llash:** yangi TIFF papka + chiqish papkasi tanlanadi; natija 7-tabda ko'rsatiladi.

## 5. Giperparametrlar va tuning

### 5.1 Giperparametr nima

Giperparametrlar — modelning o'qitishdan oldin belgilanadigan sozlamalari (daraxtlar soni, chuqurlik, regulyarizatsiya...).
**Standart qiymatlar asl dasturdagi qiymatlar bilan mos** (RF: 500 daraxt, `min_samples_leaf=2`; XGBoost: 300 daraxt, `max_depth=4`, `learning_rate=0.05`; SVM: `rbf`, `C=1`); ularni sababi bo'lsa o'zgartiring.
Paneldagi har bir qiymat **yagona manba**dan olinadi va bir xil holda ishlatiladi: CV'da, yakuniy modelda, tuning'da va bundle'da
(CV va yakuniy model giperparametrlari farq qilmaydi). Ishlatilgan giperparametrlar `hyperparameters_used.json` ga yoziladi.

### 5.2 Parametrlar jadvallari

Ustunlar: **standart**; **diapazon** (panel qabul qiladigan qiymatlar); **tuning oralig'i** (avtomatik qidiruv standart oralig'i; "-" = qidirilmaydi).
`avtomatik/None` — belgilanmasa model o'zi tanlaydi.

#### Random Forest

| Parametr | Standart | Diapazon / variantlar | Tuning oralig'i |
|---|---|---|---|
| `n_estimators` | 500 | 10 .. 5000 | 100 .. 1000 |
| `criterion` | gini | gini, entropy, log_loss | - |
| `max_depth` | avtomatik/None (cheksiz) | 1 .. 200 | 2 .. 20 |
| `min_samples_split` | 2 | 2 .. 100 | 2 .. 20 |
| `min_samples_leaf` | 2 | 1 .. 100 | 1 .. 10 |
| `max_features` | sqrt | sqrt, log2, all | {sqrt, log2, all} |
| `bootstrap` | ha | ha/yo'q | - |
| `max_samples` | avtomatik/None (to'liq) | 0.1 .. 1 | - |
| `class_weight` | balanced | balanced, balanced_subsample, none | - |
| `ccp_alpha` | 0.0 | 0 .. 0.5 | - |

#### SVM

| Parametr | Standart | Diapazon / variantlar | Tuning oralig'i |
|---|---|---|---|
| `kernel` | rbf | rbf, linear, poly, sigmoid | - |
| `C` | 1.0 | 0.001 .. 10000 | 0.01 .. 100 (log) |
| `gamma` | avtomatik/None ('scale') | 1e-06 .. 100 | 0.0001 .. 1 (log) |
| `degree` | 3 | 1 .. 10 | - |
| `class_weight` | balanced | balanced, none | - |

SVM har doim standartlashtirish (`StandardScaler`) bilan ishlaydi va Platt kalibrlash bilan o'raladi (kalibrlashni o'chirsangiz ham:
SVM ehtimollik chiqishi uchun kalibrlash majburiy).

#### XGBoost

| Parametr | Standart | Diapazon / variantlar | Tuning oralig'i |
|---|---|---|---|
| `n_estimators` | 300 | 10 .. 5000 | 100 .. 800 |
| `learning_rate` | 0.05 | 0.001 .. 1 | 0.01 .. 0.3 (log) |
| `max_depth` | 4 | 1 .. 20 | 2 .. 8 |
| `min_child_weight` | 1.0 | 0 .. 100 | 1 .. 10 |
| `subsample` | 0.8 | 0.1 .. 1 | 0.5 .. 1 |
| `colsample_bytree` | 0.8 | 0.1 .. 1 | 0.4 .. 1 |
| `gamma` | 0.0 | 0 .. 20 | 0 .. 5 |
| `reg_alpha` | 0.0 | 0 .. 100 | 0 .. 5 |
| `reg_lambda` | 1.0 | 0 .. 100 | 0.1 .. 10 (log) |
| `scale_pos_weight` | avtomatik/None (neg/pos) | 0.01 .. 1000 | - |
| `tree_method` | hist | hist, exact, approx | - |

#### CNN

| Parametr | Standart | Diapazon / variantlar | Tuning oralig'i |
|---|---|---|---|
| `mode` | patch2d | patch2d, tabular1d | - |
| `window` | 9 | 3 .. 31 (toq) | {5, 7, 9, 11, 15} |
| `filters1` | 16 | 2 .. 256 | {8, 16, 32} |
| `filters2` | 32 (0 = o'chirilgan) | 0 .. 256 | {0, 16, 32, 64} |
| `kernel_size` | 3 | 1 .. 7 (toq) | - |
| `dense_units` | 32 | 2 .. 512 | {16, 32, 64} |
| `dropout` | 0.3 | 0 .. 0.9 | 0.1 .. 0.6 |
| `l2` | 0.0 | 0 .. 0.1 | - |
| `optimizer` | adam | adam, sgd, rmsprop | - |
| `learning_rate` | 0.001 | 1e-05 .. 0.5 | 0.0001 .. 0.01 (log) |
| `batch_size` | 8 | 1 .. 512 | {4, 8, 16, 32} |
| `epochs` | 150 (maksimum) | 1 .. 2000 | - |
| `patience` | 15 | 1 .. 200 | - |
| `val_fraction` | 0.2 | 0.05 .. 0.5 | - |
| `augment` | ha | ha/yo'q | - |

**CNN haqida:**
* `patch2d` — har nuqta atrofidagi `window` x `window` pikselli oyna (barcha qatlamlar kanal sifatida) ustida 2D konvolyutsiya.
  Bu qatlamlar tartibiga bog'liq emas va fazoviy kontekstni ko'radi. `tabular1d` — eski usul (qatlamlar ketma-ketligi bo'ylab 1D konvolyutsiya,
  qatlam tartibiga bog'liq); faqat taqqoslash uchun.
* O'qitish train qismining `val_fraction` ulushini (stratifikatsiya bilan) validatsiyaga ajratadi va `val_loss` bo'yicha erta to'xtatadi
  (`epochs` — maksimum, haqiqiy son odatda kam). Har sinfdan >= 2 namuna bo'lmasa, train loss bo'yicha to'xtatiladi (ogohlantirish bilan).
* **CNN chiqishi kalibrlanmaydi** va musbat sinf neg/pos namuna og'irligi bilan o'qitiladi. Ansambl oddiy o'rtacha bo'lgani uchun
  CNN indekslari boshqa modellar bilan bir shkalada bo'lmasligi mumkin (log va `summary.txt` da ogohlantirilgan).
* **Leakage xavfi:** spatial blok o'lchami oyna kengligidan (`window` x piksel o'lchami) kichik bo'lsa, train va validation patchlari
  ustma-ust tushadi va CNN natijasi optimistik bo'ladi. Dastur buni ogohlantiradi: blok o'lchamini kattalashtiring yoki oynani kichraytiring.
* CNN ko'p vaqt va xotira oladi (10-bo'lim); birinchi tahlillarda CNN'ni o'chirib turing.

### 5.3 Preset

**Preset saqlash (JSON)** hozirgi giperparametrlar va tuning sozlamalarini faylga yozadi, **Preset yuklash (JSON)** — qaytaradi
(preset ichidagi hamma giperparametr almashadi; tushunarsiz/chegaradan chiqqan qiymatlar tuzatiladi va ogohlantirishlar ko'rsatiladi).
Tab 1 dagi **Konfiguratsiyani saqlash/yuklash** esa barcha sozlamalarni (papkalar, CV, modellar ham) saqlaydi.
Maqola uchun `run_config.json` va `hyperparameters_used.json` ni natija bilan birga saqlab qo'ying.

### 5.4 Tuning: giperparametrlarni avtomatik qidirish

**Nima qiladi.** Tasodifiy qidiruv: `n_iter` ta nomzod giperparametr to'plami (birinchisi — sizning joriy qiymatlaringiz) ichki spatial CV
(`ichki fold'lar`, standart 3) bilan baholanadi; eng yaxshi ball (ROC AUC yoki PR AUC — *Baholash mezoni*) g'olib. Teng ballda joriy
qiymatlar afzal ko'riladi. Faqat jadvaldagi "tuning oralig'i" bor parametrlar qidiriladi.

**Qanday yoqiladi.** 2-tabdagi "Giperparametrlarni avtomatik qidirish" guruhini yoqing, qidiriladigan modellarni (RF, SVM, XGBoost;
CNN — "qimmat!") belgilang, `n_iter`, ichki fold'lar va mezonni tanlang. **"Qidiruv oraliqlarini tahrirlash..."** oynasida har parametr
oralig'ini (min/maks, log shkala yoki variantlar ro'yxati) o'zgartirishingiz mumkin; noto'g'ri qiymat kiritilsa, oyna xatoni ko'rsatadi va yopilmaydi.

**Baholash halolligi (nested).** Spatial CV'da har tashqi fold'ning **faqat train qismida** alohida tuning o'tkaziladi, validation qismi
qidiruvga umuman ko'rinmaydi. Shuning uchun CV natijasi "tuning protsedurasi" ning haqiqiy sifatini ko'rsatadi, optimistik emas.

**Yakuniy model.** Yakuniy tuning **har doim butun ma'lumotda** bajariladi (faqat 1-fon tanlovida; qolgan fon tanlovlari shu
giperparametrlarni ishlatadi) va tanlangan giperparametrlar yakuniy modelga beriladi. Ular `hyperparameters_used.json` (`final`) va
`tuned_params.csv` ga yoziladi.

**Eslatmalar:**
* *Random CV* har doim bazaviy giperparametrlar bilan baholanadi, spatial CV esa tuning bilan: tuning yoqilganda "random - spatial"
  farqining bir qismi shundan bo'lishi mumkin (log'da yoziladi).
* Tuning qidiruvi ichida kalibrlash o'chiriladi (ball faqat tartibga bog'liq, natija bir xil, lekin 4-5 marta tezroq); fold modellari va
  yakuniy model esa siz tanlagan kalibrlash bilan o'qitiladi.
* Ichki bo'linish imkonsiz bo'lsa (musbat bloklar kam), ichki fold soni kamaytiriladi; u ham bo'lmasa joriy giperparametrlar ishlatiladi
  (fold uchun "fallback/skipped" belgilanadi va ogohlantirish yoziladi). Hech qachon yiqilmaydi.
* Tuning vaqtni **bir necha baravar** oshiradi: har tashqi fold uchun `n_iter x ichki fold` ta o'qitish, har model uchun (10-bo'lim).
  Qidiruv oralig'ini toraytirish (masalan RF `n_estimators` 100-300) va `n_iter` ni 10-20 da ushlash tavsiya etiladi.
* Kichik namunada (o'nlab musbat nuqta) tuning ortiqcha moslashishi (ichki ball shovqini) mumkin: foyda ko'pincha kichik. Avval
  standart qiymatlar bilan natijani ko'ring.
* CNN tuning odatda tavsiya etilmaydi (yuzlab o'qitish, TensorFlow xotirasi oshadi).

### 5.5 Qaysi giperparametrni qachon o'zgartirish

Umumiy amaliy qoidalar (mashinaviy o'qitish tajribasiga asoslangan; dasturda avtomatik tekshirilmaydi):

| Holat | Nima qilish |
|---|---|
| RF natijasi beqaror | `n_estimators` ni oshiring; `min_samples_leaf` ni 2-5 qiling (kichik namunada ortiqcha moslashuvni kamaytiradi) |
| Overfitting (train yaxshi, spatial CV past) | `max_depth` cheklash, `min_samples_leaf` oshirish; XGBoost: `max_depth` kamaytirish, `reg_lambda`/`gamma` oshirish, `learning_rate` kamaytirish |
| SVM'da natija yomon | `C` va `gamma` ni tuning orqali tanlang (qatlamlar avtomatik standartlanadi) |
| Fon:musbat nisbat juda katta | `class_weight=balanced` (standart) va `scale_pos_weight=avtomatik` nisbatni hisobga oladi |
| CNN beqaror/sekin | `window` ni kichraytiring, `epochs` va `patience` ni kamaytiring, `filters`/`dense_units` ni kamaytiring |

## 6. Natijalarni o'qish

### 6.1 Spatial CV va random CV: nega ikkita?

Geologik qatlamlarda yaqin joylar bir-biriga o'xshaydi (**spatial avtokorrelyatsiya**). Oddiy (random) CV'da validation nuqtasining qo'shnisi
train'da bo'ladi va model uni "yodlab" oladi: AUC sun'iy yuqori chiqadi. **Spatial block CV** maydonni kvadrat bloklarga bo'ladi va bir blokdagi
nuqtalarni butunlay train yoki validation'ga beradi — bu yangi, ko'rilmagan hududga bashorat qilish sifatini halol baholaydi.

* **Asosiy natija — Spatial block CV.** Maqolaga/xulosaga shu raqamlarni qo'ying.
* Random CV — faqat benchmark. **Optimizm = random AUC - spatial AUC**: farq katta bo'lsa, spatial avtokorrelyatsiya kuchli
  va random CV natijasiga ishonib bo'lmaydi. Farq kichik bo'lsa, model fazoviy tuzilishga kam tayanadi.
* Spatial AUC ~0.5 ga yaqin — model yangi hududlarda bilimni umumlashtira olmayapti (qatlamlar yoki nuqtalar kam).
* Spatial AUC juda yuqori (> 0.95), ayniqsa kam nuqtada — shubha bilan qarang: blok o'lchami kichikmi (leakage), qatlamlardan biri
  to'g'ridan-to'g'ri kon bilan bog'liqmi (masalan konning o'zidan olingan qatlam)?

### 6.2 Metrikalar jadvali

| Ustun | Ma'nosi | Qanday o'qiladi |
|---|---|---|
| **AUC** | ROC egri chizig'i ostidagi yuza, repeat'lar bo'yicha **o'rtacha** | 0.5 = tasodifiy, 1.0 = mukammal. Taxminiy qoida: 0.7-0.8 qoniqarli, 0.8-0.9 yaxshi; kichik namunada ehtiyot bo'ling |
| **AUC_std** | AUC ning CV takrorlari orasidagi og'ishi | fold bo'linishiga bog'liqlik (kichik = barqaror). **Ishonch oralig'i emas** |
| **AUC_CI_lo / AUC_CI_hi** | 95% ishonch oralig'i: **blok-bootstrap** | nuqtalar emas, bloklar qayta tanlanadi (fazoviy bog'liqlikni hisobga oladi). Bu "nuqtalar to'plamining tasodifiyligi"ni ifodalaydi; kam musbat nuqtada keng |
| **PR_AUC** | precision-recall egri chizig'i ostidagi yuza | musbatlar kam bo'lganda ma'lumotliroq; asosiy darajasi = musbat ulushi (sun'iy nisbat!) |
| **BalAcc** | muvozanatli aniqlik, **0.5 bo'sag'ida** | sinflar o'rtacha to'g'ri topilish ulushi |
| **F1** | F1 ko'rsatkichi, 0.5 bo'sag'ida | sun'iy nisbatga bog'liq |
| **Brier** | ehtimolliklar kvadratik xatosi (kichik = yaxshi) | kalibrlash va ajratish birgalikda; sun'iy nisbatga bog'liq |
| **Sens / Spec** | sezgirlik va o'ziga xoslik, **Youden bo'sag'ida** | Sens = topilgan konlar ulushi; Spec = to'g'ri "fon" ulushi |
| **Thr** | Youden bo'sag'i (Sens + Spec maksimal) | o'rtacha OOF bashorat bo'yicha topilgan |

**Muhim eslatmalar:**
* **AUC o'rtacha repeat'lar bo'yicha, CI esa o'rtacha bashorat ustida bootstrap**. Shuning uchun nuqtaviy AUC ba'zan CI chetiga yaqin
  yoki hatto uning tashqarisida chiqadi — bu xato emas, hisoblash usullarining farqi.
* Bootstrap soni 0 bo'lsa CI **bo'sh** ("hisoblanmadi") ko'rsatiladi. Yaroqli resample'lar juda kam bo'lsa (< 30) CI ishonchsiz ekani ogohlantiriladi.
* "Ansambl" qatori — modellar ehtimolliklarining oddiy o'rtachasi (har takror ichida). U har doim eng yaxshi alohida modeldan yaxshi bo'lavermaydi:
  zaif model o'rtachani pasaytirishi mumkin.
* Barcha metrikalar *sun'iy* musbat:fon nisbati bilan hisoblanadi. AUC nisbatga kam sezgir; PR-AUC, F1, Brier, BalAcc — ko'proq. Ularni
  boshqa tadqiqotlar bilan to'g'ridan-to'g'ri solishtirmang.

### 6.3 ROC va PR egri chiziqlari

Egri chiziqlar o'rtacha out-of-fold bashorat bo'yicha chiziladi; legend'da AUC ± std va 95% CI. Ansambl qalin chiziq. ROC'da diagonal —
tasodifiy. PR'da gorizontal chiziq — "baseline" (musbat ulushi). Egri chiziq chap-yuqori burchakka qanchalik yaqin bo'lsa, shuncha yaxshi.

### 6.4 Kalibrlash (reliability diagrammasi)

**Kalibrlash nima.** Model chiqishi (masalan RF "0.7") ko'pincha haqiqiy chastotaga mos kelmaydi. Kalibrlash (sigmoid/Platt yoki isotonic)
chiqishlarni ularning *o'qitish namunasidagi* musbat ulushiga moslashtiradi, shunda RF, SVM va XGBoost chiqishlari bir shkalada bo'ladi va
ansamblga o'rtachalash mantiqiy bo'ladi. Dastur kalibratorni ichki CV bilan o'qitadi, bazaviy modelni esa butun train ma'lumotida.

**Diagrammani o'qish.** X o'qi — bin ichidagi o'rtacha bashorat, Y o'qi — shu binda kuzatilgan musbat ulushi; ideal — diagonal.
Nuqtalar diagonaldan yuqorida — model past baholayapti; pastda — yuqori baholayapti. Bin'lar kvantil bo'yicha; bin soni
musbat nuqtalar soniga moslanadi (`clip(musbatlar_soni // 5, 3, 10)`: har binda ~5 musbat), shuning uchun kam nuqtada diagramma "qirrali".

**Cheklov (muhim).** Kalibrlash faqat o'qitish namunasining sun'iy nisbatiga nisbatan; shuning uchun "kalibrlangan ehtimollik" ham
*haqiqiy kon ehtimolligi emas* (1-bo'limdagi izohga qarang).

* Musbat nuqta juda kam bo'lsa (har sinfda < 2 namuna yoki `calibration_cv` < 2) kalibrlash o'tkazib yuboriladi (ogohlantirish bilan);
  SVM uchun monoton zaxira indeks ishlatiladi (kalibrlangan emas).
* CNN kalibrlanmaydi (yuqoriga qarang).

### 6.5 Confusion matritsa

Youden bo'sag'ida 2x2 jadval: [to'g'ri fon, noto'g'ri musbat; o'tkazib yuborilgan kon, topilgan kon], qator foizlari bilan; sarlavhada
sezgirlik va o'ziga xoslik. Fon nuqtalar "kon yo'q" emasligini eslang: "noto'g'ri musbat" fon nuqtalar ichida hali kashf qilinmagan
kon bo'lishi mumkin — shuning uchun spetsifiklik pastligi har doim yomon model degani emas.

### 6.6 Fon tanloviga sezgirlik

Yoqilganda butun spatial CV bir necha mustaqil fon tanlovida (har safar boshqa tasodifiy fon) takrorlanadi va AUC taqsimoti (o'rtacha,
std, min-maks) ko'rsatiladi. Kichik std — natija fon tanloviga bog'liq emas (yaxshi). Katta tarqalish — natija fon tanloviga sezgir, fon
nuqtalar sonini oshiring yoki yakuniy fon ansamblini (K) ishlating. Bu tahlil bazaviy giperparametrlar bilan ishlaydi (tuning yo'q).

### 6.7 Feature importance: permutation, SHAP va MDI

* **Permutation importance (asosiy).** Har validation fold'da bitta feature qiymatlari aralashtiriladi va AUC qanchalik tushishi o'lchanadi
  (**dAUC**; katta = muhim). Fold'lar bo'yicha o'rtacha va std (xato chizig'i) beriladi. Faqat **spatial CV birinchi takrorida**
  hisoblanadi (vaqtni tejash uchun). dAUC ~0 yoki manfiy — model feature'dan foydalanmaydi (yoki u shovqin).
  *Cheklov:* **o'zaro bog'liq (korrelyatsiyali) feature'lar** ahamiyatni o'zaro bo'lishadi (biri aralashtirilganda ikkinchisi o'rnini bosadi) —
  ikkalasi ham past ko'rinishi mumkin; VIF/korrelyatsiya (3-tab) bilan birga ko'ring. One-hot ustunlar alohida aralashtiriladi.
* **SHAP (qo'shimcha).** Faqat RF va XGBoost. **Yakuniy** (butun ma'lumotda o'qitilgan, birinchi fon tanlovidagi) model uchun, o'qitish
  dataset'idan tasodifiy tanlangan `shap_max_background` ta qatorda hisoblanadi. SHAP — har bir nuqta uchun feature hissasi:
  **beeswarm**'da x o'qi SHAP qiymati (o'ngga = indeksni oshiradi), rang — feature qiymati (qizil yuqori, ko'k past; qatlam yuqori
  bo'lganda indeks oshsa — qizil nuqtalar o'ngda); **dependence**'da feature qiymati va uning ta'siri bog'liqligi (rang — eng bog'liq boshqa feature).
  Birliklar: RF — ehtimollik, XGBoost — log-odds; **modellararo magnitudani solishtirmang**.
  SHAP butun ma'lumot bo'yicha hisoblanadi (out-of-fold emas), shuning uchun u modelning ichki mantig'ini ko'rsatadi, umumlashtirish sifatini emas.
* **MDI/gain (zaxira).** Permutation importance hisoblanmasa (o'chirilgan yoki xato), RF/XGBoost uchun daraxtlardan olingan MDI/gain ishlatiladi
  (natija jadvalida `source = mdi`, xato chiziqlari chizilmaydi). MDI korrelyatsiyalangan va ko'p qiymatli feature'larni ortiqcha baholashga moyil.
  SVM va CNN uchun MDI yo'q: permutation o'chirilgan bo'lsa, faqat-SVM (yoki faqat-CNN) sozlamasida importance bo'sh bo'lishi mumkin.
* **Qaysiga ishonish.** Asosiy xulosa uchun permutation importance; SHAP — qo'shimcha tekshiruv va yo'nalishni ko'rish uchun. Ular katta farq
  qilsa, feature'lar bog'liq bo'lishi mumkin.

### 6.8 Ma'lumotlar tahlili natijalari

* **VIF >= 10** — kuchli multikollinearlik (qatlam boshqalardan deyarli chiziqli hosil bo'lgan); model uchun muammo bo'lmasligi mumkin
  (RF/XGBoost), lekin importance talqinini qiyinlashtiradi. Bir-biriga o'xshash qatlamlardan birini olib tashlashni o'ylab ko'ring.
* **|r| >= 0.9 juftliklar** — takroriy ma'lumot.
* **Deyarli o'zgarmas qatlam** — hech narsa bermaydi; olib tashlang.
* **EPV (musbat nuqtalar / feature) < 10** — overfitting xavfi: qatlamlarni kamaytiring yoki musbat nuqta qo'shing.
* **Feature effects** (Cohen d, bir o'zgaruvchili AUC) — har qatlam musbat va fon nuqtalarni alohida-alohida qanchalik ajratishini ko'rsatadi.

### 6.9 Success-rate va maydon statistikasi

* **Success-rate egri chizig'i.** Piksellar indeks bo'yicha kamayish tartibida; X — eng yuqori indeksli maydon ulushi, Y — shu maydon
  ichida qolgan **konlar ulushi**. Egri chiziq diagonaldan qanchalik yuqori bo'lsa, indeks konlarni shunchalik kichik maydonga to'playdi
  (masalan "maydonning 10% ida konlarning 60% i"). AUC: 0.5 tasodifiy, 1.0 mukammal.
  **Muhim:** konlar o'qitish nuqtalari, shuning uchun bu baho **optimistik** (model ularni ko'rgan) va mustaqil validatsiya o'rnini bosmaydi.
  Halol baho — spatial CV.
* **Sinf statistikasi jadvali:** `Maydon_%` — sinf maydoni ulushi; `Konlar_%` — shu sinfdagi konlar ulushi; **`Boyitish` = `Konlar_%` /
  `Maydon_%`**: 1 — tasodifiy, > 1 — sinf konlarni tasodifiydan ko'p ushlaydi (masalan 3 = 3 baravar). "Juda yuqori" sinfda Boyitish
  yuqori, "Juda past" sinfda < 1 bo'lishi kutiladi. Kon ma'lumoti yo'q holatda (yangi maydonga qo'llash) `Konlar_*` va `Boyitish` bo'sh ko'rsatiladi.
  Bu jadval ham o'qitish konlarida hisoblangan (optimistik).

## 7. Xaritani talqin qilish

* **Chiqish:** har model uchun (RF, SVM, XGBoost, CNN) va ansambl uchun `prognoz_<model>.tif` — **prospektivlik indeksi (0-1)**.
  Yuqori qiymat (qizil) — modelga ko'ra o'qitish konlariga o'xshash sharoit; past (yashil) — o'xshamaydigan. Bu *kon bor degani emas*.
* **Ansambl** — modellar xaritalarining oddiy o'rtachasi; odatda alohida modeldan barqarorroq. CNN kalibrlanmagan, shuning uchun
  CNN qo'shilsa, uning shkalasi boshqalardan farq qilishi mumkin: ansamblni CNN bilan va CNN'siz solishtirib ko'ring.
* **Sinflangan xarita** (`prognoz_classes.tif`, 0 = nodata, 1..n): `quantile` — har sinf maydonning teng ulushi (nisbiy tartib);
  `equal_interval` — mutlaq qiymat bo'yicha; `fixed` — sizning chegaralaringiz. Sinflar nomi n=5 da: Juda past, Past, O'rta, Yuqori,
  Juda yuqori. **Sinf chegaralari model va sozlamaga bog'liq**; ularni maydonlar yoki loyihalar orasida solishtirmang.
* **Noaniqlik xaritasi** (`prognoz_uncertainty.tif`; kamida ikki model bo'lsa): alohida modellar xaritalari orasidagi standart og'ish —
  **modellar kelishmovchiligi**. Yuqori qiymat — modellar bu joyda bir-biriga zid; bu statistik ishonch oralig'i emas.
  Odatda ma'lumotdan uzoq (ekstrapolyatsiya) joylarda kattalashadi.
* **Ehtiyot choralari.** (1) Indeks faqat o'qitish qatlamlari qamrovi (barcha qatlamlar chekli bo'lgan piksellar) ichida beriladi;
  qolgan joy nodata (-9999). (2) Model musbat nuqtalardagi qatlam xususiyatlariga o'xshashlikni topadi; qatlamlar qamrab olmagan
  geologik omillar xaritada ko'rinmaydi. (3) Xaritani spatial CV natijasi bilan birga baholang. (4) Indeksni geologik bilim bilan tekshiring.
* **Format:** barcha chiqish GeoTIFF'lar EPSG:28411, LZW siqish; indeks va noaniqlik xaritalari float32 (nodata -9999), sinf xaritasi int8 (nodata 0). GIS dasturida (QGIS/ArcGIS) ochiladi.

## 8. Modelni saqlash va yangi maydonga qo'llash

**Saqlash (8-tab, "Modelni saqlash (bundle)").** Bundle — papka:

```
manifest.json        versiyalar, vaqt, qatlam nomlari, modellar, giperparametrlar, sozlamalar, metrikalar, Youden bo'sag'lari
pipeline.json        feature pipeline (qatlam nomlari, kategorik qatlamlar va darajalari)
models/<model>/<n>/  yakuniy modellar (RF/SVM/XGBoost: model.joblib; CNN: model.keras + meta.json)
README.txt           qisqa izoh
```

Yangi maydonga qo'llash uchun yangi TIFF papkasi quyidagi talablarga javob berishi kerak:

* **Fayl nomlari (kengaytmasiz, katta-kichik harf farqsiz) o'qitishdagi qatlam nomlariga mos** bo'lishi kerak. Yetishmaydigan qatlam bo'lsa,
  dastur yetishmaydigan va mavjud nomlarni ko'rsatib xato beradi. Ortiqcha TIFF'lar e'tiborsiz qoldiriladi (log'da ko'rsatiladi).
  Bir nomga bir nechta fayl mos kelsa, xato beriladi.
* Qatlam **birligi va ma'nosi** o'qitishdagi bilan bir xil bo'lishi shart (dastur buni tekshira olmaydi).
* Kategorik qatlamlar o'qitishdagi kabi `nearest` bilan moslanadi; yangi daraja uchrasa ogohlantirish chiqadi.
* Referens grid — tanlangan fayllarning alifbo bo'yicha birinchisi; band tartibi bundle'niki.

Natija 7-tabda ko'rsatiladi (konlar yo'q, shuning uchun kon statistikasi va success-rate yo'q) va `out_dir` ga GeoTIFF'lar yoziladi.
Yangi maydon o'qitish maydonidan geologik jihatdan keskin farq qilsa, indeks ishonchsiz bo'ladi (modelning qo'llanilish sohasidan tashqari).

Eslatma: bundle'ni xuddi shu kutubxona versiyalari bilan ishlatish tavsiya etiladi (`manifest.json` -> `versions`); versiyalar farq qilsa,
dastur ogohlantiradi. Faqat `tensorflow-cpu` o'rnatilgan muhitda TensorFlow versiyasi yozilmaydi (`null`) va bu farq tekshirilmaydi.

## 9. Xavfsizlik

RF, SVM va XGBoost modellari `joblib`/`pickle` formatida saqlanadi. **Bunday faylni yuklash o'zboshimchalik bilan kod bajarishi mumkin.**

* Bundle'ni **faqat o'zingiz yaratgan yoki to'liq ishongan manbadan** yuklang. Begona kishidan olingan, internetdan yuklangan yoki
  jamoaviy papkadagi noma'lum bundle'ni ochmang.
* Dastur yuklashdan oldin ogohlantirish va tasdiq so'raydi; log'da ham eslatma yoziladi.
* Boshqalarga bundle uzatsangiz, ularga shu ogohlantirishni ayting. CNN `.keras` formatida (joblib emas) saqlanadi, lekin papkaning qolgan
  modellari (RF/SVM/XGBoost) baribir joblib.
* Jamoa ichida bundle almashishda fayl yaxlitligini (masalan SHA-256 xesh) alohida tekshirish mumkin; dastur o'zi buni qilmaydi.

## 10. Hisoblash vaqti bo'yicha maslahatlar

**Vaqt nimaga bog'liq.** Asosiy narx — o'qitishlar soni (dastur 1-tabda hisoblab ko'rsatadi, `estimate_cost_text`):

| Qism | O'qitishlar soni |
|---|---|
| Cross-validation | `k-fold x takrorlar` (random CV yoqilsa, ikki baravar) har model uchun |
| Kalibrlash | kalibrlangan RF/XGBoost uchun har o'qitish `calibration_cv + 1` ichki fit'ga teng; SVM har doim shunday |
| Nested tuning | `k-fold x takrorlar` tashqi fold x `n_iter x ichki fold` x modellar |
| Yakuniy tuning | `n_iter x ichki fold` x modellar (bir marta) |
| Yakuniy modellar | fon ansambli `K` x modellar |
| Fon sezgirligi | `tanlovlar x takrorlar x k-fold` x modellar |
| Bootstrap CI | `bootstrap soni` resample (katta ma'lumotda sezilarli) |
| Permutation importance | har fold uchun `feature soni x takrorlar` bashorat |

**Misollar** (dastur ko'rsatadigan baho):
* *Standart sozlamalar* (4 model, 5-fold x 10 takror, random CV yoqilgan): ~1300 ichki fit — **o'nlab daqiqa**, CNN esa ~100 ta o'qitish va
  ~800 MB TensorFlow xotirasini o'stiradi.
* *Yengil* (RF + SVM, 5-fold x 2 takror, random CV o'chirilgan): ~90 ichki fit.
* *Standart sozlamalar + tuning yoqilgan* (`n_iter`=20, ichki fold 3): ~20000 ichki fit — **soatlar**. Dastur "hisoblash juda og'ir" deb ogohlantiradi.
* Kichik sintetik loyihada (100x100 piksel, ~22 musbat, 3-fold x 2 takror, RF (100 daraxt) + SVM, random va spatial CV) butun
  o'qitish ~20 sekund oldi (4 CPU). Haqiqiy loyihada vaqt ma'lumot hajmiga va modellarga ko'ra ancha ko'p.

**Tavsiyalar:**
1. **Birinchi o'tish (tekshiruv):** RF + SVM (+ XGBoost), k-fold 5, takrorlar 2-3, bootstrap 200-300, tuning va CNN o'chirilgan. Maqsad —
   ma'lumot va sozlamalarning to'g'riligini tekshirish.
2. **Asosiy natija:** takrorlar 10 (standart), bootstrap 1000, fon sezgirligi yoqilgan (bir marta).
3. **CNN** ni oxirida, `epochs` va `patience` ni kamaytirib, kam takror bilan qo'shing. Patch2d katta oyna va ko'p qatlamda ko'p xotira oladi.
4. **Tuning** ni faqat RF/XGBoost uchun, `n_iter` 10-20, ichki fold 3, oraliqlarni toraytirib yoqing. RF uchun `n_estimators` oralig'i vaqtga ko'p ta'sir qiladi (dastur har tashqi fold uchun quriladigan daraxtlar sonini baholaydi).
5. **n_jobs** — standart CPU soni (ko'pi bilan 4); RF va XGBoost shu qiymatni ishlatadi. RF natijasi `n_jobs` ga bog'liq emas (bir xil seed — bir xil natija).
6. Random CV benchmark'i kerak bo'lmasa, o'chiring (vaqtni yarmiga kamaytiradi).
7. Katta rasterlarda xarita bashorati (ayniqsa CNN patch2d) ko'p xotira oladi: qatlamlar sonini kamaytiring yoki rasterni kichikroq piksel
   kattaligiga o'tkazing. Bashorat bo'laklab (batch) bajariladi, butun feature massivi bir yo'la xotiraga olinmaydi.
8. Uzoq hisob oldidan **Konfiguratsiyani saqlang** va chiqish papkasini belgilang: log fayli va natijalar saqlanadi.

## 11. Chiqish fayllari

**O'qitish tugagach** (chiqish papkasi berilgan bo'lsa), natijalar eksporti:

| Fayl | Mazmuni |
|---|---|
| `metrics_spatial.csv`, `metrics_random.csv` | metrikalar jadvali (random CV bo'lsa) |
| `metrics_spatial.xlsx` | Excel: varaqlar "Spatial CV", "Random CV", "Sinflar" (openpyxl bo'lsa) |
| `hyperparameters_used.json` | `base` (siz kiritgan), `final` (yakuniy modellarda ishlatilgan, tuning bo'lsa tanlangan), `tuning`, `models` |
| `run_config.json` | barcha sozlamalar + qatlam nomlari, blok o'lchami, nuqtalar soni, versiya |
| `importance.csv` | model, feature, mean, std, source (`permutation` yoki `mdi`) |
| `tuned_params.csv`, `tuning_trials.csv` | tuning natijalari (nested va final), faqat tuning bo'lsa |
| `fold_table.csv` | har fold'dagi nuqtalar/bloklar/vaqtlar (random va spatial) |
| `oof_predictions.csv` | har nuqta uchun out-of-fold bashorat (x, y, y_true, fold, model ustunlari, Ensemble) |
| `predictor_data_dictionary.csv` | predictor qatlamlar data dictionary |
| `versions.json` | Python va kutubxona versiyalari (takrorlanuvchanlik) |
| `bg_sensitivity.json` | fon sezgirligi natijalari (yoqilgan bo'lsa) |
| `summary.txt` | xulosa (metrikalar, random vs spatial, importance, ogohlantirishlar) |
| `mpm_run_<n>.log` | to'liq log (UTF-8) |

**Prognoz xaritasi** (chiqish papkasi berilsa): `prognoz_<model>.tif` (har model va `prognoz_Ensemble_soft_voting.tif`),
`prognoz_classes.tif`, `prognoz_uncertainty.tif`, `predictor_data_dictionary.csv`; prognoz keyin eksport qilinsa `class_stats.csv` ham.

## 12. Muammolarni bartaraf etish

Eng yaxshi manba — **log oynasi** va **Spatial CV diagnostika** tabidagi ogohlantirishlar ro'yxati (xabarlar shu yerda to'liq).

| Xabar / belgi | Sabab | Nima qilish |
|---|---|---|
| "TIFF papkasida .tif/.tiff fayllar topilmadi" | yo'l noto'g'ri yoki fayllar boshqa kengaytmada | papkani tekshiring; kengaytma `.tif` yoki `.tiff` |
| "...papkasida .shp fayl topilmadi" | musbat nuqtalar yoki AOI papkasida shapefile yo'q | `.shp` (va `.shx`, `.dbf`) fayl papkaning o'zida bo'lsin (ichki papkada emas) |
| "faylida CRS aniqlanmagan" | fayl CRS'siz | fayl koordinatalari EPSG:28411 ekaniga ishonsangiz CRS taxmin belgisini yoqing, aks holda GIS'da CRS belgilang |
| "Barcha musbat nuqtalar AOI tashqarisida" | CRS/hudud mos emas yoki CRS yo'q | nuqtalar va AOI CRS'ini tekshiring |
| "Musbat nuqtalar soni juda kam (n < 10)" | AOI ichidagi va yaroqli pikseldagi musbat nuqtalar kam | nuqtalar qo'shing; nodata/AOI tashqarisidagilar tashlanganini log'dan ko'ring |
| "Musbat nuqtalar soni (n) k-fold sonidan kam" yoki "Blok o'lchamini ... kamaytirib ham yetarli bloklar hosil bo'lmadi" | musbat nuqtalar kam yoki bir joyda to'plangan | k-fold ni kamaytiring (masalan 3), nuqta qo'shing |
| "Barcha qatlamlar bir vaqtda chekli bo'lgan piksel yo'q" | qatlamlar bir-biri bilan kesishmaydi yoki nodata noto'g'ri | qatlamlar qamrovi/nodata'ni tekshiring (3-tab: valid_pct) |
| "qatlami faqat N% maydonni qoplaydi" | qatlam kichik qamrovli | shu qatlamni olib tashlang yoki qamrovni kengaytiring |
| "...qatlami deyarli o'zgarmas" | qatlam konstanta | olib tashlang (hech narsa bermaydi) |
| "...kategorik qatlam butun sonli emas" / "N ta daraja bor (maks. 30)" | uzluksiz qatlam kategorik deb belgilangan | kategorik belgisini olib tashlang |
| "faqat X/N ta fon nuqta topildi" | min. masofa katta yoki valid maydon kichik | min. masofani kamaytiring yoki fon sonini kamaytiring |
| "metadata.csv ';' ajratgich..." / "UTF-8 emas" | Excel boshqa format bilan saqlagan | "CSV UTF-8 (vergul bilan ajratilgan)" formatida qayta saqlang |
| "XGBoost o'rnatilmagan" / "TensorFlow o'rnatilmagan" | ixtiyoriy kutubxona yo'q | `pip install xgboost` / `pip install tensorflow` (yoki `tensorflow-cpu`), yoki modelni o'chiring |
| "kalibrlash o'tkazib yuborildi" | har sinfdan < 2 namuna (juda kam musbat nuqta) | kutilgan holat kam nuqtada; natija kalibrlanmagan ekanini hisobga oling |
| "blok o'lchami CNN oynasidan kichik" | CNN patchlari train va validation orasida ustma-ust | blok o'lchamini kattalashtiring yoki `window` ni kichraytiring |
| AUC CI "hisoblanmadi" | bootstrap soni 0 yoki yaroqli resample < 2 | bootstrap sonini oshiring (>= 200) |
| "permutation importance mavjud emas, zaxira sifatida MDI/gain ishlatildi" | permutation o'chirilgan yoki hisoblanmagan | permutation'ni yoqing; faqat-SVM'da MDI yo'q |
| "SHAP hisoblanmadi" / SHAP grafigi bo'sh | `shap` yo'q, faqat SVM/CNN tanlangan yoki xato | `pip install shap`; SHAP faqat RF/XGBoost uchun |
| "Yangi papkada bundle uchun kerakli bandlar topilmadi" | fayl nomlari qatlam nomlariga mos emas | nomlarni o'qitishdagi qatlam nomlariga moslang (xabarda ro'yxat bor) |
| "...bandiga bir nechta fayl mos keladi" | nomi faqat harf registri/kengaytmasi bilan farq qiladigan fayllar | takroriy fayllarni olib tashlang |
| "kutubxona versiyalari o'qitishdagidan farq qiladi" | bundle boshqa muhitda yaratilgan | iloji bo'lsa shu versiyalarni o'rnating; natijani tekshiring |
| Spatial AUC ~0.5 | model ajrata olmayapti | qatlamlarni, nuqtalar sonini, kategorik belgilarni tekshiring |
| Spatial AUC juda yuqori (> 0.95) | leakage yoki "javob"ni beruvchi qatlam | blok o'lchami, qatlamlar manbasi (kon bilan bog'liq qatlam) |
| Random AUC >> spatial AUC | spatial avtokorrelyatsiya (kutilgan) | spatial natijaga tayaning |
| Jarayon sekin / ETA noaniq | ko'p fold, tuning, CNN | 10-bo'lim; ETA taxminiy (bosqichlar og'irligi teng emas) |
| Xotira yetishmaydi | katta raster / ko'p qatlam / CNN patch2d | qatlamlar sonini kamaytiring, oynani kichraytiring, boshqa dasturlarni yoping |
| "Stop" bosildi, lekin natija chiqdi | hisoblash ~95% dan keyin to'xtatilmaydi | kutilgan holat; natijani ishlating yoki qayta o'qiting |
| Qt xatosi (platforma plagini topilmadi) | tizim grafik kutubxonalari yo'q | ish stoli muhitida ishga tushiring; serverda sinash uchun `QT_QPA_PLATFORM=offscreen` |

Muammo hal bo'lmasa: log faylini (`mpm_run_<n>.log`), `run_config.json` va `versions.json` ni saqlab qo'ying — bu xatoni qayta ishlab chiqarish uchun yetarli.

## 13. Dasturiy (GUI'siz) foydalanish

GUI faqat qobiq: butun ish oqimi Python'dan ham chaqiriladi (skriptlar, serverda uzoq hisob):

```python
from mpm.config import RunConfig, TuningConfig
from mpm.pipeline import estimate_cost_text, run_training, run_prediction, export_results
from mpm.persist import save_bundle, load_bundle, apply_bundle

cfg = RunConfig(
    tiff_folder="data/tiff", points_folder="data/pts", aoi_folder="data/aoi", output_dir="natija",
    categorical_layers=["geology"],                       # band (fayl) nomlari
    n_splits=5, n_repeats=3, n_bootstrap=500,
    use_models={"RandomForest": True, "SVM": True, "XGBoost": True, "CNN": False},
)
cfg.hyperparams["RandomForest"]["n_estimators"] = 300      # giperparametrlar PARAM_SPECS bo'yicha
# cfg.tuning = TuningConfig(enabled=True, n_iter=10)       # ixtiyoriy: tuning

print(estimate_cost_text(cfg))
natija = run_training(cfg, log_fn=print, progress_fn=lambda f, msg: None)
print(natija["spatial"]["metrics_df"])                     # asosiy natija
prognoz = run_prediction(natija, out_dir="natija")         # GeoTIFF'lar
export_results(natija, "natija", prediction=prognoz)       # CSV/XLSX/JSON/summary.txt

save_bundle("model_bundle", final_models=natija["final_models"], pipeline=natija["pipeline"],
            hyperparams_used=natija["final_hyperparams"], cfg_dict=natija["cfg"],
            thresholds=natija["thresholds"], block_size=natija["block_size"])
yangi = apply_bundle(load_bundle("model_bundle"), "yangi_tiff", out_dir="yangi_natija")
```

Sozlamalarni faylga saqlash: `config.save_preset(path, cfg=cfg)`, o'qish: `config.load_preset(path)`. `run_training` bekor qilish uchun
`cancel=CancelToken()` (`mpm.common`) qabul qiladi. Hamma imzo va natija sxemalari — [`ARCHITECTURE.md`](ARCHITECTURE.md) 3-bo'lim.

## 14. Atamalar lug'ati

| Atama | Ma'nosi |
|---|---|
| **Prospektivlik indeksi** | model chiqishi (0-1): maydonning konlarga o'xshashlik darajasi; ehtimollik emas |
| **Musbat nuqta** | ma'lum kon/namoyon nuqtasi |
| **Fon (pseudo-absence)** | AOI ichidan tasodifiy olingan nuqta ("kon yo'q" emas) |
| **AOI** | tadqiqot maydoni konturi (poligon) |
| **Out-of-fold (OOF)** | validation fold'da, model ko'rmagan nuqta uchun bashorat |
| **Spatial block CV** | maydon bloklarga bo'linib, bloklar butunlay train yoki validation'ga beriladigan cross-validation |
| **Spatial avtokorrelyatsiya** | yaqin joylarning o'xshashligi; random CV'ni optimistik qiladi |
| **Variogram / range** | avtokorrelyatsiya masofasi; blok o'lchamini avtomatik tanlashda ishlatiladi |
| **AUC / PR-AUC** | ajratish sifati ko'rsatkichlari (ROC va precision-recall egri chizig'i ostidagi yuza) |
| **Youden bo'sag'i** | sezgirlik + o'ziga xoslik maksimal bo'ladigan bo'sag' |
| **Blok-bootstrap CI** | bloklarni qayta tanlab hisoblangan ishonch oralig'i |
| **Kalibrlash** | model chiqishlarini namunadagi chastotaga moslashtirish (sigmoid/Platt yoki isotonic) |
| **Permutation importance** | feature aralashtirilganda AUC pasayishi |
| **SHAP** | har nuqta uchun feature hissasi (Shapley qiymatlari) |
| **MDI / gain** | daraxt modellarining ichki importance'i |
| **VIF** | multikollinearlik ko'rsatkichi |
| **EPV** | musbat nuqtalar soni / feature soni |
| **Nested CV (tuning)** | tuning har tashqi fold'ning train qismida alohida bajariladigan baholash |
| **Success-rate** | maydon ulushi bo'yicha ushlangan konlar ulushi egri chizig'i |
| **Boyitish** | sinfdagi konlar ulushining sinf maydoni ulushiga nisbati |
| **Bundle** | saqlangan modellar, feature pipeline va sozlamalar papkasi |
