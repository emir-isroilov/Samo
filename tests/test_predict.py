# -*- coding: utf-8 -*-
"""mpm.predict testlari: predict_probability_maps (tabular / patch, batch, nanmean, noaniqlik, progress/cancel),
classify_map, CLASS_LABELS, class_area_stats, success_rate_curve, save_rasters. CNN testlari - slow."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from scipy.special import expit

from mpm import models as models_mod
from mpm import predict
from mpm.base import ModelWrapper
from mpm.common import ENSEMBLE_NAME, NODATA_OUT, CancelledError, CancelToken
from mpm.config import default_hyperparams
from mpm.data import FeaturePipeline, RasterStack, extract_patches, valid_pixel_mask
from tests.synth import X0, Y0

H, W, NB = 40, 50, 4


# ---------------------------------------------------------------------------
# Yordamchilar
# ---------------------------------------------------------------------------
def make_raster(h=H, w=W, n=NB, seed=0, nodata=True):
    rng = np.random.default_rng(seed)
    stack = gaussian_filter(rng.normal(size=(n, h, w)), sigma=(0, 2, 2)).astype(np.float32)
    if nodata:
        stack[2, :3, :3] = np.nan            # burchakdagi nodata
        stack[1, 30:33, 40:44] = np.nan      # ichkaridagi nodata (boshqa band'da)
    transform = from_origin(X0, Y0 + h * 100.0, 100.0, 100.0)
    profile = {"driver": "GTiff", "dtype": "float32", "count": 1, "width": w, "height": h, "crs": "EPSG:28411",
               "transform": transform, "nodata": NODATA_OUT, "compress": "lzw"}
    return RasterStack(stack=stack, band_names=[f"b{i}" for i in range(n)], profile=profile, transform=transform,
                       crs_epsg=28411, categorical=[], tech_metadata=[])


def fit_models(raster, pipe, names=("RandomForest", "SVM"), n_draws=2, n_samples=150, seed=0):
    """Rastr pikselli namunalarida (signal: b0 + b1 > 0) kichik modellar o'qitadi: {nom: [draw, ...]}."""
    valid = valid_pixel_mask(raster)
    rr, cc = np.nonzero(valid)
    hp = default_hyperparams()
    hp["RandomForest"]["n_estimators"] = 15
    hp["XGBoost"]["n_estimators"] = 15
    out = {n: [] for n in names}
    for d in range(n_draws):
        rng = np.random.default_rng(seed + d)
        sel = rng.choice(rr.size, size=n_samples, replace=False)
        X = pipe.transform_pixels(raster.stack[:, rr[sel], cc[sel]].T)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        for n in names:
            out[n].append(models_mod.make_model(n, hp[n], seed=seed + d).fit(X, y))
    return out


@pytest.fixture(scope="module")
def raster():
    return make_raster()


@pytest.fixture(scope="module")
def pipe(raster):
    return FeaturePipeline(raster.band_names, raster.categorical).fit(raster)


@pytest.fixture(scope="module")
def fm(raster, pipe):
    return fit_models(raster, pipe)


@pytest.fixture(scope="module")
def pred(raster, pipe, fm):
    return predict.predict_probability_maps(raster, pipe, fm)


class FakeTab(ModelWrapper):
    """Tabular soxta model: predict_proba_pos(X) = fn(X)."""
    input_kind = "tabular"

    def __init__(self, fn, name="FakeTab"):
        super().__init__(name, {})
        self.fn = fn
        self.is_fitted = True
        self.calls = []

    def fit(self, X, y, patches=None, cancel=None, log_fn=None):
        return self

    def predict_proba_pos(self, X=None, patches=None, batch_size=8192):
        self.calls.append((len(X), batch_size))
        return np.asarray(self.fn(X), dtype=np.float64)


class FakePatch(ModelWrapper):
    """Patch soxta model: markaziy 0-kanal + 1-kanalning oyna bo'yicha nanmean'i -> sigmoid."""
    input_kind = "patch"

    def __init__(self, window=3, name="FakePatch"):
        super().__init__(name, {"window": window})
        self.is_fitted = True
        self.calls = []

    def fit(self, X, y, patches=None, cancel=None, log_fn=None):
        return self

    def predict_proba_pos(self, X=None, patches=None, batch_size=8192):
        assert X is None and patches is not None
        w = self.params["window"]
        assert patches.shape[1:3] == (w, w)
        self.calls.append((len(patches), batch_size))
        return expit(patches[:, w // 2, w // 2, 0] + np.nanmean(patches[..., 1], axis=(1, 2)))


def expected_fake_patch(raster, r, c, w):
    """Bitta piksel uchun FakePatch natijasi - qo'lda (extract_patches'siz) hisoblangan."""
    s = raster.stack
    h = w // 2
    vals = []
    for rr in range(r - h, r + h + 1):
        for cc in range(c - h, c + h + 1):
            if 0 <= rr < s.shape[1] and 0 <= cc < s.shape[2] and np.isfinite(s[1, rr, cc]):   # 1-kanal NaN'lari o'tkaziladi
                vals.append(s[1, rr, cc])
    return float(expit(float(s[0, r, c]) + np.mean(vals, dtype=np.float32)))


# ---------------------------------------------------------------------------
# CLASS_LABELS
# ---------------------------------------------------------------------------
def test_class_labels():
    assert predict.CLASS_LABELS(5) == ["Juda past", "Past", "O'rta", "Yuqori", "Juda yuqori"]
    assert predict.CLASS_LABELS(3) == ["1-sinf", "2-sinf", "3-sinf"]
    assert predict.CLASS_LABELS(1) == ["1-sinf"]
    for bad in (0, -2):
        with pytest.raises(ValueError):
            predict.CLASS_LABELS(bad)


# ---------------------------------------------------------------------------
# classify_map
# ---------------------------------------------------------------------------
def rand_map(seed=0, shape=(80, 90), nan_block=True):
    m = np.random.default_rng(seed).normal(size=shape).astype(np.float32)
    if nan_block:
        m[:7, :9] = np.nan
    return m


def test_classify_quantile_equal_area():
    pm = rand_map()
    cm, br = predict.classify_map(pm, "quantile", 5)
    assert cm.dtype == np.int8 and cm.shape == pm.shape
    assert np.array_equal(cm == 0, np.isnan(pm))                       # nodata = 0
    counts = np.bincount(cm[cm > 0], minlength=6)[1:]
    assert counts.sum() == np.isfinite(pm).sum()
    assert counts.max() - counts.min() <= 2                           # teng maydon
    assert len(br) == 4 and all(isinstance(b, float) for b in br) and br == sorted(br)
    # sinf chegaralar bilan mos: chegaraga teng qiymat yuqori sinfga
    ok = np.isfinite(pm)
    np.testing.assert_array_equal(cm[ok], np.digitize(pm[ok], br) + 1)
    # monotonlik: qiymat oshsa sinf kamaymaydi
    order = np.argsort(pm[ok])
    assert np.all(np.diff(cm[ok][order]) >= 0)


@pytest.mark.parametrize("n", [2, 3, 4, 7])
def test_classify_quantile_n_classes(n):
    cm, br = predict.classify_map(rand_map(1), "quantile", n)
    assert len(br) == n - 1 and set(np.unique(cm)) == set(range(0, n + 1))
    counts = np.bincount(cm[cm > 0])[1:]
    assert counts.max() - counts.min() <= 2


def test_classify_equal_interval():
    pm = np.array([[0.0, 0.1, 0.24, 0.25], [0.5, 0.74, 0.75, 1.0], [np.nan, np.inf, -np.inf, 0.9]], dtype=np.float32)
    cm, br = predict.classify_map(pm, "equal_interval", 4)
    assert br == [0.25, 0.5, 0.75]
    np.testing.assert_array_equal(cm, [[1, 1, 1, 2], [3, 3, 4, 4], [0, 0, 0, 4]])
    cm5, br5 = predict.classify_map(pm, "equal_interval")                  # standart n=5
    assert br5 == pytest.approx([0.2, 0.4, 0.6, 0.8])


def test_classify_equal_interval_ignores_breaks_arg():
    pm = rand_map(2)
    a = predict.classify_map(np.clip(pm, 0, 1), "equal_interval", 3, breaks=[0.1])
    b = predict.classify_map(np.clip(pm, 0, 1), "equal_interval", 3)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1]


def test_classify_fixed():
    pm = np.array([[0.0, 0.19, 0.2], [0.49, 0.5, 0.99], [np.nan, 1.0, 5.0]], dtype=np.float32)
    cm, br = predict.classify_map(pm, "fixed", n_classes=99, breaks=[0.2, 0.5])      # n_classes e'tiborsiz
    assert br == [0.2, 0.5]
    np.testing.assert_array_equal(cm, [[1, 1, 2], [2, 3, 3], [0, 3, 3]])
    cm1, br1 = predict.classify_map(pm, "fixed", breaks=(0.5,))
    assert br1 == [0.5] and set(np.unique(cm1)) == {0, 1, 2}


@pytest.mark.parametrize("breaks", [None, [], [0.5, 0.2], [0.3, 0.3], [0.1, np.nan], [0.1, np.inf]])
def test_classify_fixed_invalid_breaks(breaks):
    with pytest.raises(ValueError):
        predict.classify_map(rand_map(), "fixed", breaks=breaks)


def test_classify_validation():
    pm = rand_map()
    with pytest.raises(ValueError, match="usuli"):
        predict.classify_map(pm, "jenks")
    for bad in (1, 0, -3, 128, 1000):
        for m in ("quantile", "equal_interval"):
            with pytest.raises(ValueError):
                predict.classify_map(pm, m, bad)
    with pytest.raises(ValueError):
        predict.classify_map(pm, "quantile", "x")
    with pytest.raises(ValueError):
        predict.classify_map(np.array([["a"]]), "equal_interval")
    # eng ko'p sinf (127) ishlaydi
    cm, br = predict.classify_map(rand_map(), "equal_interval", 127)
    assert len(br) == 126 and cm.max() <= 127


def test_classify_no_valid_pixels():
    nan = np.full((5, 6), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="yaroqli"):
        predict.classify_map(nan, "quantile", 5)
    for m, kw in (("equal_interval", {}), ("fixed", {"breaks": [0.3]})):
        cm, br = predict.classify_map(nan, m, 5, **kw)
        assert cm.shape == nan.shape and not cm.any() and cm.dtype == np.int8


def test_classify_constant_and_tied_maps_do_not_crash():
    flat = np.full((20, 20), 0.4, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cm, br = predict.classify_map(flat, "quantile", 5)
    assert (cm > 0).all() and len(br) == 4                           # hammasi sinflangan, xato yo'q
    tied = np.repeat(np.array([0.1, 0.1, 0.1, 0.9]), 25).reshape(10, 10).astype(np.float32)
    cm, br = predict.classify_map(tied, "quantile", 4)
    assert (cm > 0).all() and br == sorted(br)


def test_classify_accepts_float64_and_int_maps():
    cm, _ = predict.classify_map(np.linspace(0, 1, 100).reshape(10, 10), "equal_interval", 2)
    assert set(np.unique(cm)) == {1, 2} and cm.dtype == np.int8
    cm, _ = predict.classify_map(np.arange(100).reshape(10, 10), "fixed", breaks=[50])
    assert set(np.unique(cm)) == {1, 2}


# ---------------------------------------------------------------------------
# class_area_stats
# ---------------------------------------------------------------------------
STATS_COLS = ["Sinf", "Nomi", "Piksel", "Maydon_km2", "Maydon_%", "Konlar_soni", "Konlar_%", "Boyitish"]


def test_class_area_stats_sums():
    pm = rand_map()
    cm, _ = predict.classify_map(pm, "quantile", 5)
    st = predict.class_area_stats(cm, 100.0)
    assert list(st.columns) == STATS_COLS
    assert st["Sinf"].tolist() == [1, 2, 3, 4, 5]
    assert st["Nomi"].tolist() == predict.CLASS_LABELS(5)
    n_valid = int(np.isfinite(pm).sum())
    assert st["Piksel"].sum() == n_valid == int((cm > 0).sum())
    assert st["Maydon_%"].sum() == pytest.approx(100.0)
    np.testing.assert_allclose(st["Maydon_km2"], st["Piksel"] * 100.0 * 100.0 / 1e6)
    assert st["Maydon_km2"].sum() == pytest.approx(n_valid * 0.01)
    # kon ma'lumoti berilmagan: Konlar_* NaN (noma'lum), 0 emas
    assert st[["Konlar_soni", "Konlar_%", "Boyitish"]].isna().all().all()


def test_class_area_stats_pixel_size_scaling():
    cm = np.array([[1, 1], [2, 0]], dtype=np.int8)
    a = predict.class_area_stats(cm, 1000.0)
    assert a["Maydon_km2"].tolist() == pytest.approx([2.0, 1.0])
    assert a["Maydon_%"].tolist() == pytest.approx([200 / 3, 100 / 3])


def test_class_area_stats_with_deposits():
    cm = np.array([[1, 1, 1, 1],
                   [2, 2, 2, 0],
                   [3, 3, 0, 0]], dtype=np.int8)
    # konlar: (0,0) sinf1; (1,0),(1,1) sinf2; (2,0) sinf3; (1,3) nodata -> hisobga olinmaydi; tashqarida: (5,5), (-1,0)
    rows = np.array([0, 1, 1, 2, 1, 5, -1])
    cols = np.array([0, 0, 1, 0, 3, 5, 0])
    st = predict.class_area_stats(cm, 100.0, rows, cols)
    assert st["Piksel"].tolist() == [4, 3, 2]
    assert st["Konlar_soni"].tolist() == [1, 2, 1]
    assert st["Konlar_%"].tolist() == pytest.approx([25.0, 50.0, 25.0])
    assert st["Konlar_%"].sum() == pytest.approx(100.0)
    assert st["Maydon_%"].tolist() == pytest.approx([400 / 9, 300 / 9, 200 / 9])
    np.testing.assert_allclose(st["Boyitish"], st["Konlar_%"] / st["Maydon_%"])
    assert st["Konlar_soni"].dtype.kind == "i"


def test_class_area_stats_deposits_edge_cases():
    cm = np.array([[1, 2], [2, 0]], dtype=np.int8)
    st = predict.class_area_stats(cm, 50.0, [], [])                  # kon yo'q: Konlar 0, % va boyitish NaN
    assert st["Konlar_soni"].tolist() == [0, 0] and st["Konlar_%"].isna().all() and st["Boyitish"].isna().all()
    st = predict.class_area_stats(cm, 50.0, np.array([0.0, np.nan]), np.array([1.0, 0.0]))   # float indekslar, NaN tashlanadi
    assert st["Konlar_soni"].tolist() == [0, 1]
    # bo'sh sinf: maydon 0 => boyitish NaN
    cm2 = np.array([[1, 3]], dtype=np.int8)
    st = predict.class_area_stats(cm2, 10.0, [0], [0])
    assert st["Piksel"].tolist() == [1, 0, 1] and np.isnan(st["Boyitish"].iloc[1])


def test_class_area_stats_n_classes_and_empty():
    cm = np.array([[1, 1], [2, 0]], dtype=np.int8)
    st = predict.class_area_stats(cm, 100.0, n_classes=5)            # bo'sh yuqori sinflar ham qatorda
    assert len(st) == 5 and st["Piksel"].tolist() == [2, 1, 0, 0, 0]
    assert st["Nomi"].tolist() == predict.CLASS_LABELS(5)
    empty = predict.class_area_stats(np.zeros((3, 3), dtype=np.int8), 100.0)
    assert list(empty.columns) == STATS_COLS and len(empty) == 0


def test_class_area_stats_validation():
    cm = np.array([[1, 2]], dtype=np.int8)
    with pytest.raises(ValueError):
        predict.class_area_stats(cm, 0.0)
    with pytest.raises(ValueError):
        predict.class_area_stats(cm, float("nan"))
    with pytest.raises(ValueError):
        predict.class_area_stats(cm, 100.0, rows=[0])                 # cols yo'q
    with pytest.raises(ValueError):
        predict.class_area_stats(cm, 100.0, rows=[0, 0], cols=[0])
    with pytest.raises(ValueError):
        predict.class_area_stats(np.array([1, 2]), 100.0)             # 1 o'lchamli
    with pytest.raises(ValueError):
        predict.class_area_stats(np.array([[0.5, 1.0]]), 100.0)       # float
    with pytest.raises(ValueError):
        predict.class_area_stats(np.array([[-1, 1]], dtype=np.int8), 100.0)


# ---------------------------------------------------------------------------
# success_rate_curve
# ---------------------------------------------------------------------------
def test_success_rate_perfect_vs_random():
    rng = np.random.default_rng(0)
    pm = rng.random((100, 100)).astype(np.float32)
    top = np.argsort(pm.ravel())[::-1][:40]
    r, c = np.divmod(top, 100)
    perfect = predict.success_rate_curve(pm, r, c)
    assert perfect["auc"] > 0.995 and perfect["n_pos"] == 40
    ar, ac = rng.integers(0, 100, 600), rng.integers(0, 100, 600)
    rnd = predict.success_rate_curve(pm, ar, ac)
    assert 0.45 < rnd["auc"] < 0.55 and rnd["n_pos"] == 600
    worst = np.argsort(pm.ravel())[:40]
    wr, wc = np.divmod(worst, 100)
    assert predict.success_rate_curve(pm, wr, wc)["auc"] < 0.005
    for res in (perfect, rnd):
        x, y = res["area_frac"], res["capture_frac"]
        assert x.shape == y.shape and x[0] == 0 and y[0] == 0 and x[-1] == 1 and y[-1] == 1
        assert np.all(np.diff(x) >= 0) and np.all(np.diff(y) >= 0) and x.min() >= 0 and y.max() <= 1
        assert res["auc"] == pytest.approx(float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2)))


def test_success_rate_known_value():
    # 10 piksel (qiymat 0..9), konlar qiymati 9 va 7: eng yuqori pikselda 1/2, 3-pikselda (qiymat 7) 2/2
    pm = np.arange(10, dtype=np.float32).reshape(2, 5)
    res = predict.success_rate_curve(pm, [1, 1], [4, 2])
    x, y = res["area_frac"], res["capture_frac"]
    np.testing.assert_allclose(x, [0, 0.1, 0.2, 0.3, 1.0])
    np.testing.assert_allclose(y, [0, 0.5, 0.5, 1.0, 1.0])
    assert res["auc"] == pytest.approx(0.1 * 0.25 + 0.1 * 0.5 + 0.1 * 0.75 + 0.7 * 1.0)
    assert res["n_pos"] == 2


def test_success_rate_nodata_excluded_from_area():
    pm = np.arange(100, dtype=np.float32).reshape(10, 10)
    pm[:5] = np.nan                                       # yarmi nodata
    res = predict.success_rate_curve(pm, [9, 9], [9, 8])
    assert res["area_frac"][-1] == 1.0
    np.testing.assert_allclose(res["area_frac"][1], 1 / 50)              # maydon faqat valid piksellar bo'yicha
    assert res["n_pos"] == 2
    ref = np.arange(50, dtype=np.float32).reshape(5, 10)                   # nodata'siz ekvivalent
    assert predict.success_rate_curve(ref, [4, 4], [9, 8])["auc"] == pytest.approx(res["auc"])


def test_success_rate_drops_nan_and_outside_positives():
    pm = np.arange(100, dtype=np.float32).reshape(10, 10)
    pm[0, 0] = np.nan
    rows = np.array([9, 0, 9, 50, -1, 9])
    cols = np.array([9, 0, 8, 3, 3, 99])
    res = predict.success_rate_curve(pm, rows, cols)
    assert res["n_pos"] == 2                                       # NaN piksel, ikkita tashqarisi va ustun>=W tashlandi
    only = predict.success_rate_curve(pm, [9, 9], [9, 8])
    assert res["auc"] == pytest.approx(only["auc"])
    # float indekslar (NaN bilan) ham qabul qilinadi
    res2 = predict.success_rate_curve(pm, np.array([9.0, np.nan, 9.0]), np.array([9.0, 3.0, 8.0]))
    assert res2["n_pos"] == 2 and res2["auc"] == pytest.approx(only["auc"])


def test_success_rate_ties():
    pm = np.zeros((10, 10), dtype=np.float32)
    pm[:5] = 1.0
    res = predict.success_rate_curve(pm, [0, 1, 2], [0, 1, 2])           # barcha konlar platoda (yuqori yarim)
    assert res["auc"] == pytest.approx(0.75)
    const = predict.success_rate_curve(np.full((8, 8), 0.3, dtype=np.float32), [1, 2, 3], [1, 2, 3])
    assert const["auc"] == pytest.approx(0.5)
    np.testing.assert_allclose(const["area_frac"], [0.0, 1.0])
    np.testing.assert_allclose(const["capture_frac"], [0.0, 1.0])


def test_success_rate_empty_cases():
    pm = np.random.default_rng(0).random((10, 10)).astype(np.float32)
    for rows, cols in (([], []), ([50], [50])):
        res = predict.success_rate_curve(pm, rows, cols)
        assert res["n_pos"] == 0 and np.isnan(res["auc"]) and res["area_frac"].size == 0 and res["capture_frac"].size == 0
    res = predict.success_rate_curve(np.full((4, 4), np.nan, dtype=np.float32), [1], [1])
    assert res["n_pos"] == 0 and np.isnan(res["auc"])
    with pytest.raises(ValueError):
        predict.success_rate_curve(pm, [1, 2], [1])
    with pytest.raises(ValueError):
        predict.success_rate_curve(pm.ravel(), [1], [1])


# ---------------------------------------------------------------------------
# save_rasters
# ---------------------------------------------------------------------------
def test_save_rasters_roundtrip(tmp_path, raster, pred):
    cm, _ = predict.classify_map(pred["maps"][ENSEMBLE_NAME], "quantile", 5)
    prof = dict(raster.profile, tiled=True, blockxsize=16, blockysize=16, interleave="pixel", count=3,
                dtype="float64", predictor=3)             # "iflos" manba profili: ko'chirilmasligi kerak
    log = []
    paths = predict.save_rasters(str(tmp_path / "out"), pred["maps"], prof, class_map=cm,
                                 uncertainty=pred["uncertainty"], log_fn=log.append)
    names = [p.rsplit("/", 1)[-1] for p in paths]
    assert names == ["prognoz_RandomForest.tif", "prognoz_SVM.tif", "prognoz_Ensemble_soft_voting.tif",
                     "prognoz_classes.tif", "prognoz_uncertainty.tif"]
    assert len(log) == 5 and all(p in " ".join(log) for p in paths)
    for name, p in zip(["RandomForest", "SVM", ENSEMBLE_NAME], paths[:3]):
        with rasterio.open(p) as src:
            assert src.count == 1 and src.dtypes[0] == "float32" and src.nodata == NODATA_OUT
            assert src.crs.to_epsg() == 28411 and src.transform == raster.transform
            assert (src.height, src.width) == (H, W) and not src.profile["tiled"]
            assert src.profile["compress"] == "lzw" and src.profile["interleave"] == "band"
            assert "Prospektivlik indeksi" in (src.descriptions[0] or "")
            arr = src.read(1)
            m = src.read(1, masked=True)
        want = np.where(np.isnan(pred["maps"][name]), NODATA_OUT, pred["maps"][name])
        np.testing.assert_array_equal(arr, want.astype(np.float32))
        assert np.array_equal(m.mask, np.isnan(pred["maps"][name]))      # nodata maska NaN bilan mos
    with rasterio.open(paths[3]) as src:
        assert src.dtypes[0] == "int8" and src.nodata == 0 and src.count == 1
        np.testing.assert_array_equal(src.read(1), cm)
        assert src.read(1, masked=True).mask.sum() == (cm == 0).sum()
    with rasterio.open(paths[4]) as src:
        assert src.dtypes[0] == "float32" and src.nodata == NODATA_OUT
        np.testing.assert_array_equal(src.read(1), np.where(np.isnan(pred["uncertainty"]), NODATA_OUT,
                                                            pred["uncertainty"]).astype(np.float32))


def test_save_rasters_only_maps_and_creates_dir(tmp_path, raster):
    out = tmp_path / "a" / "b"
    m = {"M": np.full((H, W), 0.5, dtype=np.float32)}
    paths = predict.save_rasters(str(out), m, raster.profile)
    assert [p.rsplit("/", 1)[-1] for p in paths] == ["prognoz_M.tif"] and out.is_dir()


def test_save_rasters_inf_and_nan_become_nodata(tmp_path, raster):
    a = np.full((H, W), 0.25, dtype=np.float32)
    a[0, 0], a[0, 1], a[0, 2] = np.nan, np.inf, -np.inf
    p = predict.save_rasters(str(tmp_path), {"X": a}, raster.profile)[0]
    with rasterio.open(p) as src:
        arr = src.read(1)
    assert (arr[0, :3] == NODATA_OUT).all() and arr[1, 1] == 0.25


def test_save_rasters_name_collision_does_not_overwrite(tmp_path, raster):
    maps = {"A b": np.zeros((H, W), np.float32), "A_b": np.ones((H, W), np.float32)}
    paths = predict.save_rasters(str(tmp_path), maps, raster.profile)
    assert len(set(paths)) == 2
    with rasterio.open(paths[0]) as s0, rasterio.open(paths[1]) as s1:
        assert s0.read(1).max() == 0.0 and s1.read(1).min() == 1.0


def test_save_rasters_validation(tmp_path, raster):
    ok = np.zeros((H, W), np.float32)
    with pytest.raises(ValueError, match="o'lcham"):
        predict.save_rasters(str(tmp_path), {"A": ok, "B": np.zeros((H, W + 1), np.float32)}, raster.profile)
    with pytest.raises(ValueError, match="mos emas"):
        predict.save_rasters(str(tmp_path), {"A": ok}, raster.profile, class_map=np.zeros((3, 3), np.int8))
    with pytest.raises(ValueError, match="mos emas"):
        predict.save_rasters(str(tmp_path), {"A": ok}, raster.profile, uncertainty=np.zeros((3, 3), np.float32))
    with pytest.raises(ValueError, match="profile"):                 # xarita o'lchami profilga mos emas
        predict.save_rasters(str(tmp_path), {"A": np.zeros((H + 1, W), np.float32)}, raster.profile)
    with pytest.raises(ValueError, match="transform"):
        predict.save_rasters(str(tmp_path), {"A": ok}, {"crs": "EPSG:28411"})
    with pytest.raises(ValueError):
        predict.save_rasters(str(tmp_path), {"A": ok}, raster.profile, class_map=np.full((H, W), 200, dtype=np.int16))
    with pytest.raises(ValueError):
        predict.save_rasters(str(tmp_path), {"A": ok}, raster.profile, class_map=np.zeros((H, W), np.float32))
    assert predict.save_rasters(str(tmp_path / "none"), {}, raster.profile) == []


# ---------------------------------------------------------------------------
# predict_probability_maps: tabular
# ---------------------------------------------------------------------------
def test_predict_structure_and_values(raster, pipe, fm, pred):
    assert set(pred) == {"maps", "uncertainty", "valid_mask"}
    valid = valid_pixel_mask(raster)
    np.testing.assert_array_equal(pred["valid_mask"], valid)
    assert pred["valid_mask"].dtype == bool and 0 < valid.sum() < valid.size
    assert list(pred["maps"]) == ["RandomForest", "SVM", ENSEMBLE_NAME]
    rr, cc = np.nonzero(valid)
    X = pipe.transform_pixels(raster.stack[:, rr, cc].T)
    refs = {}
    for name, draws in fm.items():
        m = pred["maps"][name]
        assert m.shape == (H, W) and m.dtype == np.float32
        assert np.isnan(m[~valid]).all() and np.isfinite(m[valid]).all()
        assert m[valid].min() >= 0 and m[valid].max() <= 1
        refs[name] = np.mean([d.predict_proba_pos(X) for d in draws], axis=0)       # draw'lar o'rtachasi
        np.testing.assert_allclose(m[rr, cc], refs[name], atol=1e-6)
    ens = pred["maps"][ENSEMBLE_NAME]
    assert ens.dtype == np.float32 and np.isnan(ens[~valid]).all()
    np.testing.assert_allclose(ens[rr, cc], np.mean(list(refs.values()), axis=0), atol=1e-6)
    unc = pred["uncertainty"]
    assert unc.shape == (H, W) and unc.dtype == np.float32 and np.isnan(unc[~valid]).all()
    np.testing.assert_allclose(unc[rr, cc], np.std(list(refs.values()), axis=0), atol=1e-6)      # modellar std (ddof=0)
    assert (unc[valid] >= 0).all() and unc[valid].max() > 0


def test_predict_no_warnings(raster, pipe, fm):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # nanmean "All-NaN slice" va boshqa ogohlantirishlarsiz
        predict.predict_probability_maps(raster, pipe, fm)


def test_predict_batches_bound_memory_and_match(raster, pipe, fm, pred, monkeypatch):
    sizes = []
    orig = pipe.transform_pixels

    def spy(pix):
        sizes.append(len(pix))
        return orig(pix)

    monkeypatch.setattr(pipe, "transform_pixels", spy)
    small = predict.predict_probability_maps(raster, pipe, fm, batch_size=97)
    n_valid = int(pred["valid_mask"].sum())
    assert max(sizes) <= 97 and sum(sizes) == n_valid and len(sizes) == int(np.ceil(n_valid / 97))   # X bir marta/batch
    for name in pred["maps"]:
        np.testing.assert_allclose(small["maps"][name], pred["maps"][name], atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(small["uncertainty"], pred["uncertainty"], atol=1e-6, equal_nan=True)
    # modellarga batch_size uzatiladi (BUG-07: CNN uchun 32 emas)
    probe = FakeTab(lambda X: np.full(len(X), 0.5))
    predict.predict_probability_maps(raster, pipe, {"F": [probe]}, batch_size=500)
    assert probe.calls and all(n <= 500 and bs == 500 for n, bs in probe.calls)
    assert sum(n for n, _ in probe.calls) == n_valid


def test_predict_does_not_materialize_full_feature_array(raster, pipe, fm, monkeypatch):
    """transform_stack tabular modellarda chaqirilmaydi, transform_pixels esa hech qachon butun rasterni olmaydi."""
    monkeypatch.setattr(pipe, "transform_stack", lambda *_a, **_k: pytest.fail("tabular modelga transform_stack kerak emas"))
    predict.predict_probability_maps(raster, pipe, fm, batch_size=300)


def test_predict_single_model_no_uncertainty(raster, pipe, fm, pred):
    one = predict.predict_probability_maps(raster, pipe, {"SVM": fm["SVM"]})
    assert one["uncertainty"] is None and list(one["maps"]) == ["SVM", ENSEMBLE_NAME]
    np.testing.assert_array_equal(one["maps"]["SVM"], one["maps"][ENSEMBLE_NAME])       # bitta model: ansambl = model
    np.testing.assert_allclose(one["maps"]["SVM"], pred["maps"]["SVM"], atol=1e-6, equal_nan=True)


def test_predict_accepts_single_wrapper_and_ndarray_raster(raster, pipe, fm):
    out = predict.predict_probability_maps(raster, pipe, {"RandomForest": fm["RandomForest"][0]})
    assert set(out["maps"]) == {"RandomForest", ENSEMBLE_NAME}
    arr = predict.predict_probability_maps(raster.stack, pipe, {"RandomForest": fm["RandomForest"][:1]})
    np.testing.assert_array_equal(arr["maps"]["RandomForest"], out["maps"]["RandomForest"])


def test_predict_ensemble_nanmean_all_nan_stays_nan(raster, pipe):
    """Model(lar) ba'zi pikselda NaN bersa: boshqalar o'rtachasi; hamma NaN bersa NaN (ogohlantirishsiz)."""
    b0 = raster.stack[0]
    q1, q2 = np.percentile(b0[valid_pixel_mask(raster)], [40, 80])
    a = FakeTab(lambda X: np.where(X[:, 0] > q1, np.nan, 0.2))
    b = FakeTab(lambda X: np.where(X[:, 0] > q2, np.nan, 0.6))
    log = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = predict.predict_probability_maps(raster, pipe, {"A": [a], "B": [b]}, log_fn=log.append)
    valid = out["valid_mask"]
    ens, unc = out["maps"][ENSEMBLE_NAME], out["uncertainty"]
    both = valid & (b0 <= q1)
    only_b = valid & (b0 > q1) & (b0 <= q2)
    neither = valid & (b0 > q2)
    assert both.any() and only_b.any() and neither.any()
    np.testing.assert_allclose(ens[both], 0.4, atol=1e-6)
    np.testing.assert_allclose(ens[only_b], 0.6, atol=1e-6)
    assert np.isnan(ens[neither]).all() and np.isnan(ens[~valid]).all()
    np.testing.assert_allclose(unc[both], 0.2, atol=1e-6)         # std(0.2, 0.6) = 0.2
    assert np.isnan(unc[only_b]).all()                             # bitta chekli model: std aniqlanmagan
    assert any("chekli emas" in m for m in log)


def test_predict_progress_and_cancel(raster, pipe, fm):
    calls = []
    predict.predict_probability_maps(raster, pipe, fm, batch_size=2000, progress_fn=lambda f, m: calls.append((f, m)))
    fr = [f for f, _ in calls]
    n_valid = int(valid_pixel_mask(raster).sum())
    assert len(fr) == int(np.ceil(n_valid / 2000)) * 2          # har (batch, model) dan keyin
    assert fr == sorted(fr) and fr[-1] == pytest.approx(1.0) and 0 < fr[0] < 1
    assert all("Bashorat" in m for _, m in calls) and any("RandomForest" in m for _, m in calls)
    # bekor qilish: boshlanishdan oldin
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledError):
        predict.predict_probability_maps(raster, pipe, fm, cancel=tok)
    # bekor qilish: jarayon o'rtasida (progress ichidan)
    tok2, seen = CancelToken(), []

    def prog(f, m):
        seen.append(f)
        if len(seen) == 2:
            tok2.cancel()

    with pytest.raises(CancelledError):
        predict.predict_probability_maps(raster, pipe, fm, batch_size=500, progress_fn=prog, cancel=tok2)
    assert len(seen) == 2                                        # keyingi birlikka o'tmasdan to'xtadi


def test_predict_logs(raster, pipe, fm):
    log = []
    predict.predict_probability_maps(raster, pipe, fm, log_fn=log.append)
    assert any("piksel" in m and "RandomForest" in m for m in log)


def test_predict_validation(raster, pipe, fm):
    with pytest.raises(ValueError, match="final_models"):
        predict.predict_probability_maps(raster, pipe, {})
    with pytest.raises(ValueError, match="bo'sh"):
        predict.predict_probability_maps(raster, pipe, {"RandomForest": []})
    unfit = models_mod.make_model("RandomForest", default_hyperparams()["RandomForest"])
    with pytest.raises(ValueError, match="o'qitilmagan"):
        predict.predict_probability_maps(raster, pipe, {"RandomForest": [unfit]})
    with pytest.raises(ValueError, match="band"):
        predict.predict_probability_maps(raster, FeaturePipeline(["x0", "x1", "x2", "x3"]), fm)
    with pytest.raises(ValueError, match="ansambl"):
        predict.predict_probability_maps(raster, pipe, {ENSEMBLE_NAME: fm["SVM"]})
    for bad in (0, -5, None, "abc"):
        with pytest.raises(ValueError, match="batch_size"):
            predict.predict_probability_maps(raster, pipe, fm, batch_size=bad)
    dead = make_raster()
    dead.stack[0] = np.nan
    with pytest.raises(ValueError, match="yaroqli piksel"):
        predict.predict_probability_maps(dead, pipe, fm)
    odd = FakeTab(lambda X: X[:, 0])
    odd.input_kind = "voxel"
    with pytest.raises(ValueError, match="input_kind"):
        predict.predict_probability_maps(raster, pipe, {"F": [odd]})


# ---------------------------------------------------------------------------
# predict_probability_maps: patch modellar
# ---------------------------------------------------------------------------
def test_predict_patch_model_values_and_batches(raster, pipe, monkeypatch):
    n_stack = []
    orig = pipe.transform_stack

    def spy(s):
        n_stack.append(1)
        return orig(s)

    monkeypatch.setattr(pipe, "transform_stack", spy)
    fp = FakePatch(window=5)
    out = predict.predict_probability_maps(raster, pipe, {"P": [fp]}, batch_size=300)
    assert len(n_stack) == 1                                       # feature stack BIR marta
    valid = out["valid_mask"]
    m = out["maps"]["P"]
    assert np.isnan(m[~valid]).all() and np.isfinite(m[valid]).all()
    n_valid = int(valid.sum())
    assert all(n <= 300 and bs <= 300 for n, bs in fp.calls) and sum(n for n, _ in fp.calls) == n_valid
    # qo'lda hisoblangan qiymatlar: burchak/chegara (NaN-padded), nodata yonidagi va ichki piksellar
    for r, c in [(0, 3), (H - 1, W - 1), (3, 3), (29, 41), (34, 39), (20, 20), (H - 1, 0)]:
        if valid[r, c]:
            assert m[r, c] == pytest.approx(expected_fake_patch(raster, r, c, 5), abs=1e-5), (r, c)
    rng = np.random.default_rng(0)
    rr, cc = np.nonzero(valid)
    for k in rng.choice(rr.size, 25, replace=False):
        assert m[rr[k], cc[k]] == pytest.approx(expected_fake_patch(raster, rr[k], cc[k], 5), abs=1e-5)
    # batch hajmiga bog'liq emas
    big = predict.predict_probability_maps(raster, pipe, {"P": [FakePatch(window=5)]}, batch_size=8192)
    np.testing.assert_allclose(big["maps"]["P"], m, atol=1e-6, equal_nan=True)


def test_predict_patch_batch_capped_for_large_windows(raster, pipe, monkeypatch):
    monkeypatch.setattr(predict, "_PATCH_BATCH_ELEMS", 9 * 9 * NB * 50)      # ~50 piksel/batch
    fp = FakePatch(window=9)
    out = predict.predict_probability_maps(raster, pipe, {"P": [fp]}, batch_size=8192)
    assert max(n for n, _ in fp.calls) <= 50 and all(bs <= 50 for _, bs in fp.calls)
    assert sum(n for n, _ in fp.calls) == int(out["valid_mask"].sum())


def test_predict_mixed_tabular_and_patch_models(raster, pipe, fm):
    fp = FakePatch(window=3)
    out = predict.predict_probability_maps(raster, pipe, {"RandomForest": fm["RandomForest"], "P": [fp]},
                                           batch_size=400)
    solo_rf = predict.predict_probability_maps(raster, pipe, {"RandomForest": fm["RandomForest"]})
    solo_p = predict.predict_probability_maps(raster, pipe, {"P": [FakePatch(window=3)]})
    valid = out["valid_mask"]
    np.testing.assert_allclose(out["maps"]["RandomForest"], solo_rf["maps"]["RandomForest"], atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(out["maps"]["P"], solo_p["maps"]["P"], atol=1e-6, equal_nan=True)
    exp = (out["maps"]["RandomForest"] + out["maps"]["P"]) / 2.0
    np.testing.assert_allclose(out["maps"][ENSEMBLE_NAME][valid], exp[valid], atol=1e-6)
    assert out["uncertainty"] is not None


def test_predict_draws_with_different_windows_are_averaged(raster, pipe):
    a, b = FakePatch(window=3), FakePatch(window=5)
    c = FakePatch(window=3)
    out = predict.predict_probability_maps(raster, pipe, {"P": [a, b, c]})
    ra = predict.predict_probability_maps(raster, pipe, {"P": [FakePatch(window=3)]})["maps"]["P"]
    rb = predict.predict_probability_maps(raster, pipe, {"P": [FakePatch(window=5)]})["maps"]["P"]
    exp = (2 * ra + rb) / 3.0                                      # vaznlar: 2/3 (oyna 3) va 1/3 (oyna 5)
    np.testing.assert_allclose(out["maps"]["P"], exp, atol=1e-6, equal_nan=True)
    assert a.calls and b.calls


def test_predict_patch_with_categorical_pipeline():
    r = make_raster(nodata=False)
    cat = np.floor(np.clip((r.stack[3] + 1.5) * 1.3, 0, 3.99)).astype(np.float32) + 1.0        # darajalar 1..4
    r.stack[3] = cat
    r.categorical = ["b3"]
    pp = FeaturePipeline(r.band_names, ["b3"]).fit(r)
    assert pp.n_features == 3 + len(pp.levels["b3"])

    class Echo(FakePatch):
        def predict_proba_pos(self, X=None, patches=None, batch_size=8192):
            assert patches.shape[-1] == pp.n_features          # patchlar one-hot feature'lar bo'yicha
            return super().predict_proba_pos(X, patches, batch_size)

    out = predict.predict_probability_maps(r, pp, {"P": [Echo(window=3)]})
    assert np.isfinite(out["maps"]["P"]).all()


# ---------------------------------------------------------------------------
# CNN (slow)
# ---------------------------------------------------------------------------
def _tiny_cnn_params(mode, window=5):
    return {"mode": mode, "window": window, "filters1": 4, "filters2": 0, "dense_units": 4, "epochs": 2,
            "batch_size": 16, "patience": 2, "augment": False}


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["patch2d", "tabular1d"])
def test_predict_with_cnn(raster, pipe, fm, mode):
    pytest.importorskip("tensorflow")
    valid = valid_pixel_mask(raster)
    rr, cc = np.nonzero(valid)
    sel = np.random.default_rng(1).choice(rr.size, 80, replace=False)
    fstack = pipe.transform_stack(raster)
    X = pipe.transform_pixels(raster.stack[:, rr[sel], cc[sel]].T)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    window = 5
    cnn = models_mod.make_model("CNN", _tiny_cnn_params(mode, window), seed=3, n_features=pipe.n_features)
    cnn.fit(X, y, patches=extract_patches(fstack, rr[sel], cc[sel], window) if mode == "patch2d" else None)
    expected_kind = "patch" if mode == "patch2d" else "tabular"
    assert cnn.input_kind == expected_kind
    out = predict.predict_probability_maps(raster, pipe, {"RandomForest": fm["RandomForest"], "CNN": [cnn]},
                                           batch_size=700)
    m = out["maps"]["CNN"]
    assert m.dtype == np.float32 and np.isnan(m[~valid]).all() and np.isfinite(m[valid]).all()
    Xall = pipe.transform_pixels(raster.stack[:, rr, cc].T)
    if mode == "patch2d":
        ref = cnn.predict_proba_pos(None, patches=extract_patches(fstack, rr, cc, window), batch_size=8192)
    else:
        ref = cnn.predict_proba_pos(Xall, batch_size=8192)
    np.testing.assert_allclose(m[rr, cc], ref, atol=1e-5)
    assert out["uncertainty"] is not None and np.isfinite(out["uncertainty"][valid]).all()
    np.testing.assert_allclose(out["maps"][ENSEMBLE_NAME][valid],
                               (out["maps"]["RandomForest"][valid] + m[valid]) / 2.0, atol=1e-6)
