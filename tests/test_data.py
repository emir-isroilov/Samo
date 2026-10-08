# -*- coding: utf-8 -*-
"""mpm.data testlari: alignment, kategorik nearest, FeaturePipeline, nuqtalar, fon, patchlar, diagnostika."""
import inspect
import json
import os
import subprocess
import sys
import types
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
from rasterio.transform import from_origin, rowcol, xy as rio_xy
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box

from mpm import data as D
from mpm.common import TARGET_EPSG, CancelledError, CancelToken
from tests.synth import X0, Y0, make_synthetic_project

PY = sys.executable


def _write_tif(path, arr, transform, crs="EPSG:28411", nodata=-9999, dtype="float32", **kw):
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                transform=transform, nodata=nodata)
    if crs is not None:
        prof["crs"] = crs
    prof.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(arr.astype(dtype), 1)
    return str(path)


def _collect_log():
    msgs = []
    return msgs, msgs.append


@pytest.fixture(scope="module")
def raster(synth_project):
    return D.load_and_align_rasters(D.find_tiff_files(synth_project["tiff"]))


@pytest.fixture(scope="module")
def aoi_gdf(synth_project):
    return D.load_aoi(D.find_shapefile(synth_project["aoi"]))


@pytest.fixture(scope="module")
def pos_gdf(synth_project, aoi_gdf):
    return D.load_positive_points(D.find_shapefile(synth_project["points"]), aoi_gdf)


@pytest.fixture(scope="module")
def vmask(raster):
    return D.valid_pixel_mask(raster)


@pytest.fixture(scope="module")
def pipeline(raster):
    return D.FeaturePipeline(raster.band_names, raster.categorical).fit(raster)


# ---------------------------------------------------------------------------
# Modul qoidalari
# ---------------------------------------------------------------------------
def test_module_rules():
    """Og'ir kutubxonalar import qilinmaydi, print ishlatilmaydi."""
    code = ("import sys, mpm.data; bad=[m for m in ('tensorflow','xgboost','shap','PyQt5') if m in sys.modules];"
            "sys.exit(1 if bad else 0)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert subprocess.run([PY, "-c", code], cwd=root).returncode == 0
    assert "print(" not in inspect.getsource(D)


# ---------------------------------------------------------------------------
# Fayl qidirish
# ---------------------------------------------------------------------------
def test_find_files(synth_project, tmp_path):
    tiffs = D.find_tiff_files(synth_project["tiff"])
    assert [os.path.basename(p) for p in tiffs] == [f"layer{i}.tif" for i in range(1, 7)]
    assert D.find_shapefile(synth_project["points"]).endswith("pts.shp")
    assert D.find_tiff_files(str(tmp_path / "yoq")) == [] and D.find_shapefile("") is None
    for name in ("B.TIF", "a.tif", "c.tiff", "x.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [os.path.basename(p) for p in D.find_tiff_files(str(tmp_path))] == ["a.tif", "B.TIF", "c.tiff"]


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def test_alignment_basic(raster, synth_project):
    assert raster.stack.shape == (6, 120, 120) and raster.stack.dtype == np.float32
    assert raster.band_names == synth_project["layer_names"]
    assert (raster.n_bands, raster.height, raster.width) == (6, 120, 120)
    assert raster.pixel_size == 100.0 and raster.crs_epsg == TARGET_EPSG
    assert raster.transform == synth_project["transform"]
    assert raster.categorical == []
    assert not np.isinf(raster.stack).any()
    layer3 = raster.stack[raster.band_names.index("layer3")]
    assert np.isnan(layer3[:6, :6]).all()                       # nodata (-9999) -> NaN
    assert np.isfinite(layer3[6:, 6:]).all() and np.nanmin(layer3) > -100
    assert np.isfinite(raster.stack[0]).all()
    keys = {"band_name", "file_path", "source_crs", "source_resolution_x_m", "source_resolution_y_m",
            "source_width_px", "source_height_px", "source_bounds", "source_nodata", "source_dtype",
            "reprojected_resolution_m", "valid_pixels_pct", "value_min", "value_max", "value_mean", "value_std"}
    assert len(raster.tech_metadata) == 6 and all(set(t) == keys for t in raster.tech_metadata)
    assert raster.tech_metadata[2]["valid_pixels_pct"] == pytest.approx(100 * (1 - 36 / 14400), abs=0.01)


def test_clean_profile_write_read(raster, tmp_path):
    prof = raster.profile
    assert {"tiled", "blockxsize", "blockysize"}.isdisjoint(prof)
    assert prof["driver"] == "GTiff" and prof["compress"] == "lzw" and prof["nodata"] == -9999.0
    assert prof["dtype"] == "float32" and prof["count"] == 1 and prof["crs"] == f"EPSG:{TARGET_EPSG}"
    path = str(tmp_path / "out.tif")
    arr = np.where(np.isfinite(raster.stack[2]), raster.stack[2], prof["nodata"]).astype("float32")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)
    with rasterio.open(path) as src:
        assert src.profile["compress"].lower() == "lzw" and not src.profile.get("tiled", False)
        assert src.crs.to_epsg() == TARGET_EPSG and src.nodata == -9999.0 and src.transform == raster.transform
        assert (src.width, src.height, src.count, src.dtypes[0]) == (120, 120, 1, "float32")
        np.testing.assert_array_equal(src.read(1), arr)


def test_profile_not_inherited_from_tiled_source(tmp_path):
    """Manba tiled/blocksize bo'lsa ham chiqish profili toza (BUG-14)."""
    tr = from_origin(X0, Y0 + 12000, 100, 100)
    rng = np.random.default_rng(0)
    p = _write_tif(tmp_path / "t.tif", rng.normal(size=(120, 120)), tr, tiled=True, blockxsize=64,
                   blockysize=64, compress="deflate")
    rs = D.load_and_align_rasters([p])
    assert {"tiled", "blockxsize", "blockysize", "interleave"}.isdisjoint(rs.profile)
    assert rs.profile["compress"] == "lzw"
    out = str(tmp_path / "o.tif")
    with rasterio.open(out, "w", **rs.profile) as dst:
        dst.write(np.nan_to_num(rs.stack[0], nan=-9999.0), 1)
    with rasterio.open(out) as src:
        assert not src.profile.get("tiled", False)


def test_alignment_missing_crs(tmp_path):
    a = make_synthetic_project(str(tmp_path / "a"), size=40, n_layers=2, nodata_patch=False, with_crs=False)
    b = make_synthetic_project(str(tmp_path / "b"), size=40, n_layers=2, nodata_patch=False, with_crs=True)
    msgs, log = _collect_log()
    rs = D.load_and_align_rasters(D.find_tiff_files(a["tiff"]), log_fn=log)
    ref = D.load_and_align_rasters(D.find_tiff_files(b["tiff"]))
    np.testing.assert_array_equal(rs.stack, ref.stack)
    assert sum("CRS yozilmagan" in m for m in msgs) == 2          # har bir fayl uchun ogohlantirish
    assert rs.profile["crs"] == f"EPSG:{TARGET_EPSG}"
    with pytest.raises(ValueError, match="CRS"):
        D.load_and_align_rasters(D.find_tiff_files(a["tiff"]), assume_crs_if_missing=False)


def test_alignment_different_grid_bilinear(tmp_path):
    """Boshqa piksel o'lchami/siljish: chiziqli maydon bilinear bilan aniq tiklanadi."""
    t_ref = from_origin(X0, Y0 + 6000, 100, 100)
    _write_tif(tmp_path / "a_ref.tif", np.zeros((60, 60)), t_ref)
    t2 = from_origin(X0 - 333.0, Y0 + 6400, 70, 70)
    h = w = 100
    cx, cy = np.meshgrid(X0 - 333.0 + (np.arange(w) + 0.5) * 70, Y0 + 6400 - (np.arange(h) + 0.5) * 70)
    _write_tif(tmp_path / "b_lin.tif", 0.001 * (cx - X0) + 0.002 * (cy - Y0), t2)
    rs = D.load_and_align_rasters(D.find_tiff_files(str(tmp_path)))
    gx, gy = np.meshgrid(X0 + (np.arange(60) + 0.5) * 100, Y0 + 6000 - (np.arange(60) + 0.5) * 100)
    expected = 0.001 * (gx - X0) + 0.002 * (gy - Y0)
    assert np.isfinite(rs.stack[1]).all()
    np.testing.assert_allclose(rs.stack[1], expected, atol=0.02)
    assert rs.tech_metadata[1]["source_resolution_x_m"] == 70.0 and rs.tech_metadata[1]["reprojected_resolution_m"] == 100.0


def test_alignment_reference_in_other_crs(tmp_path):
    """Birinchi fayl boshqa CRS'da (UTM 41N): referens grid EPSG:28411 ga o'tkaziladi."""
    t = from_origin(500_000 + 0.0, 4_600_000.0, 120, 120)
    rng = np.random.default_rng(1)
    p = _write_tif(tmp_path / "u.tif", rng.normal(size=(80, 80)), t, crs="EPSG:32641")
    rs = D.load_and_align_rasters([p])
    assert rs.profile["crs"] == f"EPSG:{TARGET_EPSG}" and rs.crs_epsg == TARGET_EPSG
    assert rs.transform != t and 80 <= rs.pixel_size <= 160
    assert np.isfinite(rs.stack[0]).mean() > 0.6
    assert rs.tech_metadata[0]["source_crs"] == "EPSG:32641"


def test_categorical_nearest(tmp_path):
    """BUG-10: kategorik qatlam nearest bilan - qiymatlar {1..4} dan chiqmaydi; bilinear'da chiqadi."""
    t_ref = from_origin(X0, Y0 + 6000, 100, 100)
    _write_tif(tmp_path / "a_ref.tif", np.zeros((60, 60)), t_ref)
    rr, cc = np.meshgrid(np.arange(90), np.arange(90), indexing="ij")
    cat = (1 + ((rr // 10 + cc // 10) % 4)).astype("float32")
    _write_tif(tmp_path / "z_cat.tif", cat, from_origin(X0 - 150.0, Y0 + 6250, 70, 70))
    paths = D.find_tiff_files(str(tmp_path))
    msgs, log = _collect_log()
    rs = D.load_and_align_rasters(paths, categorical=["z_cat"], log_fn=log)
    vals = rs.stack[1][np.isfinite(rs.stack[1])]
    assert vals.size > 0 and set(np.unique(vals)) <= {1.0, 2.0, 3.0, 4.0}
    assert rs.categorical == ["z_cat"] and any("nearest" in m for m in msgs)
    bil = D.load_and_align_rasters(paths).stack[1]
    bv = bil[np.isfinite(bil)]
    assert np.any(bv != np.rint(bv))                              # bilinear oraliq qiymat hosil qiladi


def test_categorical_unknown_name_is_ignored_with_warning(synth_project_cat):
    msgs, log = _collect_log()
    rs = D.load_and_align_rasters(D.find_tiff_files(synth_project_cat["tiff"]), categorical=["yoq_qatlam"],
                                  log_fn=log)
    assert rs.categorical == [] and any("yoq_qatlam" in m for m in msgs)
    rs2 = D.load_and_align_rasters(D.find_tiff_files(synth_project_cat["tiff"]), categorical=np.array(["geology_cat"]))
    assert rs2.categorical == ["geology_cat"]


def test_sanitize_inf_and_extreme(tmp_path):
    arr = np.arange(36, dtype="float32").reshape(6, 6)
    arr[0, 0], arr[0, 1], arr[0, 2], arr[1, 0] = np.inf, -np.inf, 3e38, np.nan
    p = _write_tif(tmp_path / "s.tif", arr, from_origin(X0, Y0 + 600, 100, 100), nodata=None)
    rs = D.load_and_align_rasters([p])
    assert np.isnan(rs.stack[0].ravel()[[0, 1, 2, 6]]).all() and np.isfinite(rs.stack[0]).sum() == 32
    assert not np.isinf(rs.stack).any() and rs.tech_metadata[0]["valid_pixels_pct"] == pytest.approx(100 * 32 / 36, abs=0.01)


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_alignment_errors(tmp_path, synth_project):
    with pytest.raises(ValueError, match="topilmadi"):
        D.load_and_align_rasters([])
    tr = from_origin(X0, Y0 + 600, 100, 100)
    a = _write_tif(tmp_path / "a.tif", np.ones((6, 6)), tr)
    b = _write_tif(tmp_path / "a.tiff", np.ones((6, 6)), tr)
    with pytest.raises(ValueError, match="takrorlan"):
        D.load_and_align_rasters([a, b])
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledError):
        D.load_and_align_rasters(D.find_tiff_files(synth_project["tiff"]), cancel=tok)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(tmp_path / "ng.tif", "w", driver="GTiff", height=4, width=4, count=1, dtype="float32") as dst:
            dst.write(np.ones((4, 4), dtype="float32"), 1)
    with pytest.raises(ValueError, match="georeferens"):
        D.load_and_align_rasters([str(tmp_path / "ng.tif")])


def test_all_nodata_layer_warns(tmp_path):
    tr = from_origin(X0, Y0 + 600, 100, 100)
    a = _write_tif(tmp_path / "a.tif", np.ones((6, 6)), tr)
    b = _write_tif(tmp_path / "b.tif", np.full((6, 6), -9999.0), tr)
    msgs, log = _collect_log()
    rs = D.load_and_align_rasters([a, b], log_fn=log)
    assert np.isnan(rs.stack[1]).all() and any("butunlay nodata" in m for m in msgs)
    assert rs.tech_metadata[1]["value_min"] is None


def test_each_file_opened_once(synth_project, monkeypatch):
    calls = []
    real_open = rasterio.open

    def counting_open(path, *a, **k):
        calls.append(str(path))
        return real_open(path, *a, **k)

    monkeypatch.setattr(rasterio, "open", counting_open)
    paths = D.find_tiff_files(synth_project["tiff"])
    D.load_and_align_rasters(paths)
    assert sorted(calls) == sorted(paths)


def test_suggest_categorical_layers(synth_project, synth_project_cat, tmp_path):
    assert D.suggest_categorical_layers(D.find_tiff_files(synth_project["tiff"])) == []
    assert D.suggest_categorical_layers(D.find_tiff_files(synth_project_cat["tiff"])) == ["geology_cat"]
    tr = from_origin(X0, Y0 + 3000, 100, 100)
    rng = np.random.default_rng(2)
    _write_tif(tmp_path / "lc.tif", rng.integers(1, 6, size=(30, 30)), tr, dtype="uint8", nodata=0)
    _write_tif(tmp_path / "dem.tif", rng.integers(0, 3000, size=(30, 30)), tr, dtype="int16", nodata=-32768)
    _write_tif(tmp_path / "cont.tif", rng.normal(size=(30, 30)), tr)
    _write_tif(tmp_path / "many.tif", rng.integers(0, 60, size=(30, 30)), tr)       # float, 60 ta noyob qiymat
    assert D.suggest_categorical_layers(D.find_tiff_files(str(tmp_path))) == ["lc"]
    assert D.suggest_categorical_layers([str(tmp_path / "yoq.tif")]) == []


def test_valid_pixel_mask(raster):
    m = D.valid_pixel_mask(raster)
    assert m.shape == (120, 120) and m.dtype == bool
    assert not m[:6, :6].any() and m[6:, :].all() and m[:, 6:].all()
    np.testing.assert_array_equal(D.valid_pixel_mask(raster.stack), m)


# ---------------------------------------------------------------------------
# Koordinatalar va namunalash
# ---------------------------------------------------------------------------
def test_xy_to_rowcol_matches_rasterio(synth_project):
    tr = synth_project["transform"]
    rng = np.random.default_rng(0)
    x = rng.uniform(X0 - 1500, X0 + 13500, 500)
    y = rng.uniform(Y0 - 1500, Y0 + 13500, 500)
    r, c = D.xy_to_rowcol(tr, x, y)
    er, ec = rowcol(tr, x, y)
    assert r.dtype == np.int64 and c.dtype == np.int64
    np.testing.assert_array_equal(r, np.asarray(er))
    np.testing.assert_array_equal(c, np.asarray(ec))
    r, c = D.xy_to_rowcol(tr, [X0 - 1.0, np.nan, np.inf], [Y0 + 5.0, 1.0, 1.0])
    assert c[0] == -1 and (r[1:] < 0).all() and (c[1:] < 0).all()      # chegaradan tashqari / chekli emas
    r, c = D.xy_to_rowcol(tr, X0 + 50.0, Y0 + 12000 - 50.0)
    assert r.tolist() == [0] and c.tolist() == [0]


def test_sample_features(raster, aoi_gdf):
    tr = raster.transform
    rc = [(0, 0), (5, 7), (119, 119), (60, 3)]
    xy = np.array([rio_xy(tr, r, c) for r, c in rc] + [(X0 - 500.0, Y0 + 5000.0), (X0 + 5000.0, Y0 + 20000.0)])
    values, rows, cols = D.sample_features(xy, raster, None)
    assert values.shape == (6, 6) and values.dtype == np.float32
    for i, (r, c) in enumerate(rc):
        np.testing.assert_array_equal(values[i], raster.stack[:, r, c])
        assert (rows[i], cols[i]) == (r, c)
    assert np.isnan(values[4:]).all() and (rows[4:] >= 0).sum() <= 1        # raster tashqarisi NaN
    assert np.isnan(values[0, 2]) and np.isfinite(values[0, [0, 1, 3, 4, 5]]).all()   # nodata burchagi
    v2, _, _ = D.sample_features(xy, raster.stack, tr)
    np.testing.assert_array_equal(v2, values)
    gdf = gpd.GeoDataFrame(geometry=[Point(*p) for p in xy], crs=aoi_gdf.crs)
    v3, _, _ = D.sample_features(gdf, raster, tr)
    np.testing.assert_array_equal(v3, values)
    with pytest.raises(ValueError, match="transform"):
        D.sample_features(xy, raster.stack)
    with pytest.raises(ValueError, match="xy"):
        D.sample_features(np.zeros((3, 3)), raster)
    v4, r4, _ = D.sample_features(np.empty((0, 2)), raster)
    assert v4.shape == (0, 6) and r4.size == 0


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------
def _cat_stack():
    rng = np.random.default_rng(5)
    stack = rng.normal(size=(3, 12, 14)).astype(np.float32)
    stack[1] = rng.integers(1, 4, size=(12, 14))          # kategorik: {1, 2, 3}
    stack[0, :2, :2] = np.nan
    return stack


def test_pipeline_numeric_only(raster):
    p = D.FeaturePipeline(raster.band_names)
    assert p.feature_names == raster.band_names and p.n_features == 6 and p.categorical == []
    np.testing.assert_array_equal(p.transform_stack(raster.stack), raster.stack)
    pix = raster.stack[:, 10:12, 10].T
    np.testing.assert_array_equal(p.transform_pixels(pix), pix)
    assert p.fit(raster) is p


def test_pipeline_onehot():
    stack = _cat_stack()
    p = D.FeaturePipeline(["a", "geo", "b"], categorical=["geo"]).fit(stack)
    assert p.feature_names == ["a", "geo==1", "geo==2", "geo==3", "b"] and p.n_features == 5
    assert p.feature_bands == ["a", "geo", "geo", "geo", "b"]
    pix = np.array([[0.5, 2, 1.5], [0.1, 1, -1.0], [0.2, 3, 0.0], [0.3, 7, 0.0],      # 7 - noma'lum daraja
                    [0.4, np.nan, 0.0], [0.4, 2.5, 0.0], [np.nan, 1, 0.0]], dtype=np.float64)
    out = p.transform_pixels(pix)
    assert out.dtype == np.float32 and out.shape == (7, 5)
    np.testing.assert_array_equal(out[0], [0.5, 0, 1, 0, 1.5])
    np.testing.assert_array_equal(out[1, 1:4], [1, 0, 0])
    np.testing.assert_array_equal(out[2, 1:4], [0, 0, 1])
    np.testing.assert_array_equal(out[3, 1:4], [0, 0, 0])             # noma'lum daraja => hammasi 0
    assert np.isnan(out[4, 1:4]).all() and out[4, 0] == pytest.approx(0.4)   # NaN => NaN
    np.testing.assert_array_equal(out[5, 1:4], [0, 0, 0])             # butun bo'lmagan qiymat => 0
    assert np.isnan(out[6, 0]) and out[6, 1] == 1
    ts = p.transform_stack(stack)
    assert ts.shape == (5, 12, 14) and ts.dtype == np.float32
    flat = p.transform_pixels(stack.reshape(3, -1).T)
    np.testing.assert_array_equal(ts.reshape(5, -1).T, flat)          # NaN joylari ham mos
    assert np.isnan(ts[:, 0, 0]).sum() == 1 and (np.nansum(ts[1:4], axis=0) == 1).all()


def test_pipeline_levels_only_from_valid_pixels():
    stack = _cat_stack()
    stack[1, 5, 5] = 9.0            # 9-daraja faqat raqamli band nodata bo'lgan joyda
    stack[2, 5, 5] = np.nan
    p = D.FeaturePipeline(["a", "geo", "b"], ["geo"]).fit(stack)
    assert p.levels["geo"] == [1, 2, 3]


def test_pipeline_roundtrip():
    stack = _cat_stack()
    p = D.FeaturePipeline(["a", "geo", "b"], ["geo"], max_levels=10).fit(stack)
    d = json.loads(json.dumps(p.to_dict()))
    q = D.FeaturePipeline.from_dict(d)
    assert q.feature_names == p.feature_names and q.band_names == p.band_names
    assert q.categorical == ["geo"] and q.max_levels == 10 and q.n_features == p.n_features
    np.testing.assert_array_equal(q.transform_stack(stack), p.transform_stack(stack))
    assert q.to_dict() == p.to_dict()
    u = D.FeaturePipeline.from_dict(json.loads(json.dumps(D.FeaturePipeline(["a", "geo"], ["geo"]).to_dict())))
    with pytest.raises(RuntimeError):
        _ = u.feature_names
    assert u.fit(stack[:2]).feature_names[0] == "a"
    assert D.FeaturePipeline.from_dict(D.FeaturePipeline(["x", "y"]).to_dict()).feature_names == ["x", "y"]
    with pytest.raises(ValueError):
        D.FeaturePipeline.from_dict({"categorical": ["a"]})
    with pytest.raises(ValueError, match="yetishmayapti"):
        D.FeaturePipeline.from_dict({"band_names": ["a", "b"], "categorical": ["b"], "fitted": True, "levels": {}})


def test_pipeline_errors():
    stack = _cat_stack()
    with pytest.raises(ValueError, match="yo'q"):
        D.FeaturePipeline(["a", "b"], ["c"])
    with pytest.raises(ValueError, match="takror"):
        D.FeaturePipeline(["a", "a"])
    with pytest.raises(RuntimeError, match="fit"):
        D.FeaturePipeline(["a", "geo", "b"], ["geo"]).transform_pixels(np.zeros((1, 3)))
    bad = stack.copy()
    bad[1] = np.random.default_rng(0).uniform(1, 3, size=(12, 14))
    with pytest.raises(ValueError, match="butun"):
        D.FeaturePipeline(["a", "geo", "b"], ["geo"]).fit(bad)
    many = stack.copy()
    many[1] = np.arange(12 * 14).reshape(12, 14) % 5
    with pytest.raises(ValueError, match="daraja"):
        D.FeaturePipeline(["a", "geo", "b"], ["geo"], max_levels=3).fit(many)
    with pytest.raises(ValueError, match="mos emas"):
        D.FeaturePipeline(["a", "geo"], ["geo"]).fit(stack)
    empty = stack.copy()
    empty[0] = np.nan
    with pytest.raises(ValueError, match="piksel yo'q"):
        D.FeaturePipeline(["a", "geo", "b"], ["geo"]).fit(empty)
    p = D.FeaturePipeline(["a", "geo", "b"], ["geo"]).fit(stack)
    with pytest.raises(ValueError, match="shakli"):
        p.transform_pixels(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="mos emas"):
        p.transform_stack(stack[:2])
    with pytest.raises(ValueError):
        p.transform_stack(np.zeros((3, 4)))


def test_pipeline_real_categorical(synth_project_cat):
    rs = D.load_and_align_rasters(D.find_tiff_files(synth_project_cat["tiff"]), categorical=["geology_cat"])
    p = D.FeaturePipeline(rs.band_names, rs.categorical).fit(rs)
    assert p.feature_names[:4] == [f"geology_cat=={i}" for i in (1, 2, 3, 4)] and p.n_features == 8
    ts = p.transform_stack(rs)
    assert set(np.unique(ts[:4][np.isfinite(ts[:4])])) == {0.0, 1.0}


# ---------------------------------------------------------------------------
# Patchlar
# ---------------------------------------------------------------------------
def _brute_patches(fs, rows, cols, w):
    p, H, W = fs.shape
    h = w // 2
    out = np.full((len(rows), w, w, p), np.nan, dtype=np.float32)
    for k, (r, c) in enumerate(zip(rows, cols)):
        for i in range(w):
            for j in range(w):
                rr, cc = r - h + i, c - h + j
                if 0 <= rr < H and 0 <= cc < W:
                    out[k, i, j] = fs[:, rr, cc]
    return out


def test_extract_patches_matches_bruteforce(monkeypatch):
    rng = np.random.default_rng(3)
    fs = rng.normal(size=(3, 15, 18)).astype(np.float32)
    fs[:, 4:6, 7:9] = np.nan
    rows = np.array([0, 0, 7, 14, 14, 5, -3, 20, 8])
    cols = np.array([0, 17, 8, 0, 17, 7, 4, 4, -1])
    for w in (1, 3, 5, 9):
        got = D.extract_patches(fs, rows, cols, w)
        assert got.shape == (9, w, w, 3) and got.dtype == np.float32
        np.testing.assert_array_equal(got, _brute_patches(fs, rows, cols, w))
    monkeypatch.setattr(D, "_PATCH_CHUNK_ELEMS", 5 * 5 * 3 * 2)         # bir necha chunk
    np.testing.assert_array_equal(D.extract_patches(fs, rows, cols, 5), _brute_patches(fs, rows, cols, 5))
    assert D.extract_patches(fs, [], [], 5).shape == (0, 5, 5, 3)


def test_extract_patches_errors():
    fs = np.zeros((2, 6, 6), dtype=np.float32)
    for bad in (4, 0, -3):
        with pytest.raises(ValueError, match="toq"):
            D.extract_patches(fs, [1], [1], bad)
    with pytest.raises(ValueError, match="uzunligi"):
        D.extract_patches(fs, [1, 2], [1], 3)
    with pytest.raises(ValueError, match="3 o'lchamli"):
        D.extract_patches(np.zeros((6, 6)), [1], [1], 3)


def test_dataset_patches_nodata_and_border(raster, pipeline, pos_gdf, aoi_gdf, vmask):
    bg = D.generate_background_points(aoi_gdf, pos_gdf, 40, 300.0, 1, "random", vmask, raster.transform)
    ds = D.build_dataset(raster, pipeline, pos_gdf, bg, need_feature_stack=True)
    patches = ds.get_patches(9)
    assert patches.shape == (ds.n, 9, 9, 6) and patches.dtype == np.float32
    c = ds.feature_stack[:, ds.rows[0], ds.cols[0]]
    np.testing.assert_array_equal(patches[0, 4, 4], c)                                   # markaz = nuqta o'zi
    # nodata burchagi (layer3, [:6, :6]) va chegara
    rows, cols = np.array([7, 0]), np.array([7, 0])
    pt = D.extract_patches(ds.feature_stack, rows, cols, 9)
    l3 = pipeline.feature_names.index("layer3")
    assert np.isnan(pt[0, :3, :3, l3]).all()                    # patch qatorlari 3..5 nodata ichida
    others = [i for i in range(6) if i != l3]
    assert np.isfinite(pt[0][..., others]).all() and np.isfinite(pt[0, 3:, 3:, l3]).all()
    assert np.isnan(pt[1, :4]).all() and np.isnan(pt[1, :, :4]).all()          # yuqori/chap chegara NaN
    assert np.isfinite(pt[1, 4:, 4:][..., others]).all()
    ds2 = D.Dataset(ds.X, ds.y, ds.coords, ds.rows, ds.cols, ds.feature_names)
    with pytest.raises(ValueError, match="feature_stack"):
        ds2.get_patches(5)


# ---------------------------------------------------------------------------
# AOI va musbat nuqtalar
# ---------------------------------------------------------------------------
def test_load_aoi_and_positive(synth_project, aoi_gdf):
    assert aoi_gdf.crs.to_epsg() == TARGET_EPSG and aoi_gdf.geom_type.tolist() == ["Polygon"]
    raw = gpd.read_file(D.find_shapefile(synth_project["points"]))
    msgs, log = _collect_log()
    pos = D.load_positive_points(D.find_shapefile(synth_project["points"]), aoi_gdf, log_fn=log)
    assert pos.crs.to_epsg() == TARGET_EPSG and 0 < len(pos) < len(raw) == synth_project["n_pos"]
    xy = D._gdf_xy(pos)
    assert D._contains_xy(D._union_all(aoi_gdf), xy[:, 0], xy[:, 1]).all()
    assert any(f"{len(raw) - len(pos)} ta musbat nuqta AOI tashqarisida" in m for m in msgs)
    no_aoi = D.load_positive_points(D.find_shapefile(synth_project["points"]))
    assert len(no_aoi) == len(raw)


def test_load_positive_multipoint_and_nonpoint(tmp_path):
    g = gpd.GeoDataFrame({"id": [1, 2, 3, 4]}, geometry=[
        MultiPoint([(X0 + 500, Y0 + 500), (X0 + 700, Y0 + 700)]), Point(X0 + 900, Y0 + 900),
        LineString([(X0, Y0), (X0 + 1, Y0 + 1)]), Point(X0 + 100, Y0 + 100)], crs="EPSG:28411")
    p = str(tmp_path / "mix.gpkg")
    g.to_file(p, driver="GPKG")
    msgs, log = _collect_log()
    out = D.load_positive_points(p, log_fn=log)
    assert len(out) == 4 and set(out.geom_type) == {"Point"} and out.index.tolist() == [0, 1, 2, 3]
    assert any("MultiPoint" in m for m in msgs) and any("Point bo'lmagan" in m for m in msgs)
    line_only = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:28411")
    line_only.to_file(tmp_path / "l.shp")
    with pytest.raises(ValueError, match="Point"):
        D.load_positive_points(str(tmp_path / "l.shp"))


def test_load_points_crs_handling(tmp_path):
    pt = gpd.GeoDataFrame(geometry=[Point(X0 + 500, Y0 + 500)], crs="EPSG:28411")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpd.GeoDataFrame(geometry=[Point(X0 + 500, Y0 + 500)]).to_file(tmp_path / "nocrs.shp")
    msgs, log = _collect_log()
    out = D.load_positive_points(str(tmp_path / "nocrs.shp"), log_fn=log)
    assert out.crs.to_epsg() == TARGET_EPSG and any("CRS yozilmagan" in m for m in msgs)
    with pytest.raises(ValueError, match="CRS"):
        D.load_positive_points(str(tmp_path / "nocrs.shp"), assume_crs_if_missing=False)
    pt.to_crs(32641).to_file(tmp_path / "utm.shp")
    out = D.load_positive_points(str(tmp_path / "utm.shp"), log_fn=log)
    assert out.crs.to_epsg() == TARGET_EPSG
    assert abs(out.geometry.x.iloc[0] - (X0 + 500)) < 1.0 and abs(out.geometry.y.iloc[0] - (Y0 + 500)) < 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pt.to_file(tmp_path / "ok.shp")
    far = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:28411")
    with pytest.raises(ValueError, match="AOI tashqarisida"):
        D.load_positive_points(str(tmp_path / "ok.shp"), far)


def test_load_aoi_validation(tmp_path):
    bow = Polygon([(X0, Y0), (X0 + 10, Y0 + 10), (X0 + 10, Y0), (X0, Y0 + 10)])      # noto'g'ri (bowtie)
    assert not bow.is_valid
    gpd.GeoDataFrame(geometry=[bow], crs="EPSG:28411").to_file(tmp_path / "bow.shp")
    aoi = D.load_aoi(str(tmp_path / "bow.shp"))
    assert aoi.geometry.is_valid.all() and aoi.geometry.area.sum() == pytest.approx(50.0)
    gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:28411").to_file(tmp_path / "pt.shp")
    with pytest.raises(ValueError, match="poligon"):
        D.load_aoi(str(tmp_path / "pt.shp"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpd.GeoDataFrame(geometry=[box(0, 0, 5, 5)]).to_file(tmp_path / "nc.shp")
    assert D.load_aoi(str(tmp_path / "nc.shp")).crs.to_epsg() == TARGET_EPSG
    with pytest.raises(ValueError, match="CRS"):
        D.load_aoi(str(tmp_path / "nc.shp"), assume_crs_if_missing=False)


def test_dedupe_points_by_pixel(raster):
    tr = raster.transform
    x0, y0 = rio_xy(tr, 50, 50)
    x1, y1 = rio_xy(tr, 60, 60)
    pts = [Point(x0 - 20, y0 + 10), Point(x1, y1), Point(x0 + 30, y0 - 30), Point(X0 - 900, Y0),
           Point(X0 - 900, Y0), Point(x0 + 60, y0)]
    gdf = gpd.GeoDataFrame({"k": range(6)}, geometry=pts, crs="EPSG:28411")
    msgs, log = _collect_log()
    out = D.dedupe_points_by_pixel(gdf, tr, raster.stack.shape, log_fn=log)
    assert out["k"].tolist() == [0, 1, 3, 4, 5] and out.index.tolist() == list(range(5))
    assert len(msgs) == 1 and "1 ta" in msgs[0]
    again = D.dedupe_points_by_pixel(out, tr, (120, 120))
    assert len(again) == 5 and again.crs == gdf.crs


def test_contains_xy_fallbacks(monkeypatch):
    geom = box(0, 0, 10, 10).difference(box(4, 4, 6, 6))
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-2, 12, 400), rng.uniform(-2, 12, 400)
    ref = D._contains_xy(geom, x, y)
    expected = np.array([geom.contains(Point(px, py)) for px, py in zip(x, y)])
    assert ref.dtype == bool and 0 < ref.sum() < 400
    np.testing.assert_array_equal(ref, expected)
    assert not D._contains_xy(geom, [5.0], [5.0]).any() and D._contains_xy(geom, [], []).size == 0
    with monkeypatch.context() as m:
        m.delattr(shapely, "contains_xy")                       # shapely 1.8 holati
        called = []

        def fake_vec_contains(g, xs, ys):
            called.append(1)
            return np.array([g.contains(Point(a, b)) for a, b in zip(xs, ys)])

        fake = types.ModuleType("shapely.vectorized")
        fake.contains = fake_vec_contains
        m.setitem(sys.modules, "shapely.vectorized", fake)
        np.testing.assert_array_equal(D._contains_xy(geom, x, y), ref)          # shapely.vectorized
        assert called
        m.setitem(sys.modules, "shapely.vectorized", None)
        np.testing.assert_array_equal(D._contains_xy(geom, x, y), ref)          # prepared geometry
    old_series = types.SimpleNamespace(unary_union=box(0, 0, 2, 2).union(box(1, 1, 3, 3)))   # geopandas<1: union_all yo'q
    assert D._union_all(types.SimpleNamespace(geometry=old_series)).area == pytest.approx(7.0)
    new_gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2), box(1, 1, 3, 3)])
    assert D._union_all(new_gdf).area == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Fon nuqtalar
# ---------------------------------------------------------------------------
def _check_bg(bg, n, aoi_gdf, pos_gdf, min_distance, raster=None, vmask=None):
    assert len(bg) == n and set(bg.geom_type) == {"Point"} and bg.crs == aoi_gdf.crs
    xy = D._gdf_xy(bg)
    assert D._contains_xy(D._union_all(aoi_gdf), xy[:, 0], xy[:, 1]).all()
    dist = cKDTree(D._gdf_xy(pos_gdf)).query(xy)[0]
    assert dist.min() >= min_distance
    if vmask is not None:
        r, c = D.xy_to_rowcol(raster.transform, xy[:, 0], xy[:, 1])
        assert vmask[r, c].all()
        assert np.unique(r * 1000 + c).size == n                   # har pikselga bittadan


@pytest.mark.parametrize("strategy", ["random", "grid", "distance_weighted"])
def test_background_strategies(strategy, raster, aoi_gdf, pos_gdf, vmask):
    msgs, log = _collect_log()
    bg = D.generate_background_points(aoi_gdf, pos_gdf, 120, 800.0, 7, strategy, vmask, raster.transform, log_fn=log)
    _check_bg(bg, 120, aoi_gdf, pos_gdf, 800.0, raster, vmask)
    assert not any("OGOHLANTIRISH" in m for m in msgs)
    again = D.generate_background_points(aoi_gdf, pos_gdf, 120, 800.0, 7, strategy, vmask, raster.transform)
    np.testing.assert_array_equal(D._gdf_xy(again), D._gdf_xy(bg))          # bir xil seed => bir xil natija
    other = D.generate_background_points(aoi_gdf, pos_gdf, 120, 800.0, 8, strategy, vmask, raster.transform)
    assert not np.array_equal(D._gdf_xy(other), D._gdf_xy(bg))
    plain = D.generate_background_points(aoi_gdf, pos_gdf, 120, 800.0, 7, strategy)        # maskasiz
    _check_bg(plain, 120, aoi_gdf, pos_gdf, 800.0)


def test_background_strategy_character(raster, aoi_gdf, pos_gdf, vmask):
    """distance_weighted musbatlardan uzoqroq, grid esa bir tekisroq (yaqin qo'shnigacha masofa katta)."""
    ptree = cKDTree(D._gdf_xy(pos_gdf))

    def stats(strategy, seed):
        xy = D._gdf_xy(D.generate_background_points(aoi_gdf, pos_gdf, 100, 500.0, seed, strategy, vmask, raster.transform))
        return ptree.query(xy)[0].mean(), cKDTree(xy).query(xy, k=2)[0][:, 1].mean()

    res = {s: np.mean([stats(s, seed) for seed in range(5)], axis=0) for s in ("random", "grid", "distance_weighted")}
    assert res["distance_weighted"][0] > res["random"][0] * 1.1
    assert res["grid"][1] > res["random"][1] * 1.1


def test_background_respects_valid_mask(raster, aoi_gdf, pos_gdf):
    """Maskaning katta qismi yaroqsiz bo'lsa ham son yetadi va nuqtalar faqat yaroqli joyda (BUG-11)."""
    mask = np.zeros((120, 120), dtype=bool)
    mask[:, 5:40] = True
    mask[50:70, :] = False
    for strategy in ("random", "grid", "distance_weighted"):
        bg = D.generate_background_points(aoi_gdf, pos_gdf, 150, 300.0, 3, strategy, mask, raster.transform)
        _check_bg(bg, 150, aoi_gdf, pos_gdf, 300.0, raster, mask)


def test_background_nonconvex_aoi_reaches_count(aoi_gdf):
    """L-shakldagi va teshikli AOI: bbox rejection kam qabul qilsa ham son yetadi; teshikka tushmaydi."""
    ring = box(X0, Y0, X0 + 10000, Y0 + 10000).difference(box(X0 + 2000, Y0 + 2000, X0 + 9000, Y0 + 9000))
    ring_gdf = gpd.GeoDataFrame(geometry=[ring], crs="EPSG:28411")
    pos = gpd.GeoDataFrame(geometry=[Point(X0 + 500, Y0 + 500), Point(X0 + 9500, Y0 + 9500)], crs="EPSG:28411")
    for strategy in ("random", "grid", "distance_weighted"):
        bg = D.generate_background_points(ring_gdf, pos, 200, 400.0, 11, strategy)
        _check_bg(bg, 200, ring_gdf, pos, 400.0)
        xy = D._gdf_xy(bg)
        assert not ((xy[:, 0] > X0 + 2000) & (xy[:, 0] < X0 + 9000) & (xy[:, 1] > Y0 + 2000) & (xy[:, 1] < Y0 + 9000)).any()


def test_background_pixels_exclude_positive_pixels(raster, aoi_gdf, pos_gdf, vmask):
    bg = D.generate_background_points(aoi_gdf, pos_gdf, 400, 0.0, 5, "random", vmask, raster.transform)
    _check_bg(bg, 400, aoi_gdf, pos_gdf, 0.0, raster, vmask)
    pxy = D._gdf_xy(pos_gdf)
    pr, pc = D.xy_to_rowcol(raster.transform, pxy[:, 0], pxy[:, 1])
    xy = D._gdf_xy(bg)
    r, c = D.xy_to_rowcol(raster.transform, xy[:, 0], xy[:, 1])
    assert not np.isin(r * 1000 + c, pr * 1000 + pc).any()


def test_background_shortage_warns(raster, aoi_gdf, pos_gdf, vmask):
    msgs, log = _collect_log()
    bg = D.generate_background_points(aoi_gdf, pos_gdf, 50, 1e7, 1, "random", vmask, raster.transform,
                                      max_attempts_factor=5, log_fn=log)
    assert len(bg) == 0 and bg.crs == aoi_gdf.crs and any("0/50" in m and "OGOHLANTIRISH" in m for m in msgs)
    msgs.clear()
    mask = np.zeros((120, 120), dtype=bool)
    mask[60, 60:63] = True                                       # faqat 3 ta yaroqli piksel
    bg = D.generate_background_points(aoi_gdf, pos_gdf, 20, 0.0, 1, "random", mask, raster.transform,
                                      max_attempts_factor=20, log_fn=log)
    assert 0 < len(bg) <= 3 and any("OGOHLANTIRISH" in m for m in msgs)
    for strategy in ("grid", "distance_weighted"):
        short = D.generate_background_points(aoi_gdf, pos_gdf, 20, 1e7, 1, strategy, max_attempts_factor=5, log_fn=log)
        assert len(short) == 0


def test_background_without_positives(aoi_gdf):
    empty = gpd.GeoDataFrame(geometry=[], crs=aoi_gdf.crs)
    msgs, log = _collect_log()
    for pos in (None, empty):
        for strategy in ("random", "grid", "distance_weighted"):
            bg = D.generate_background_points(aoi_gdf, pos, 30, 500.0, 2, strategy, log_fn=log)
            assert len(bg) == 30
    assert any("musbat nuqta yo'q" in m for m in msgs)


def test_background_errors(raster, aoi_gdf, pos_gdf, vmask):
    with pytest.raises(ValueError, match="strategiyasi"):
        D.generate_background_points(aoi_gdf, pos_gdf, 10, 100.0, strategy="noma'lum")
    with pytest.raises(ValueError, match="n_points"):
        D.generate_background_points(aoi_gdf, pos_gdf, 0, 100.0)
    with pytest.raises(ValueError, match="min_distance"):
        D.generate_background_points(aoi_gdf, pos_gdf, 10, -1.0)
    with pytest.raises(ValueError, match="transform"):
        D.generate_background_points(aoi_gdf, pos_gdf, 10, 100.0, valid_mask=vmask)
    with pytest.raises(ValueError, match="2 o'lchamli"):
        D.generate_background_points(aoi_gdf, pos_gdf, 10, 100.0, valid_mask=np.ones(5, bool), transform=raster.transform)
    with pytest.raises(ValueError, match="yaroqli piksel yo'q"):
        D.generate_background_points(aoi_gdf, pos_gdf, 10, 100.0, valid_mask=np.zeros((120, 120), bool),
                                     transform=raster.transform)
    far = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:28411")
    with pytest.raises(ValueError, match="kesishmaydi"):
        D.generate_background_points(far, None, 10, 100.0, valid_mask=vmask, transform=raster.transform)
    with pytest.raises(ValueError, match="AOI bo'sh"):
        D.generate_background_points(gpd.GeoDataFrame(geometry=[], crs="EPSG:28411"), pos_gdf, 10, 100.0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bg_gdf(raster, aoi_gdf, pos_gdf, vmask):
    return D.generate_background_points(aoi_gdf, pos_gdf, 60, 500.0, 42, "random", vmask, raster.transform)


def test_build_dataset(raster, pipeline, pos_gdf, bg_gdf):
    ds = D.build_dataset(raster, pipeline, pos_gdf, bg_gdf)
    assert (ds.n_pos, ds.n_neg, ds.n) == (len(pos_gdf), len(bg_gdf), len(pos_gdf) + len(bg_gdf))
    assert ds.X.dtype == np.float32 and ds.y.dtype == np.int8 and ds.coords.dtype == np.float64
    assert ds.rows.dtype == np.int64 and ds.cols.dtype == np.int64
    assert ds.X.shape == (ds.n, 6) and ds.feature_names == raster.band_names and ds.feature_stack is None
    assert (ds.y[:ds.n_pos] == 1).all() and (ds.y[ds.n_pos:] == 0).all()           # musbatlar birinchi
    np.testing.assert_array_equal(ds.coords[:ds.n_pos], D._gdf_xy(pos_gdf))
    r, c = D.xy_to_rowcol(raster.transform, ds.coords[:, 0], ds.coords[:, 1])
    np.testing.assert_array_equal(r, ds.rows)
    np.testing.assert_array_equal(c, ds.cols)
    np.testing.assert_array_equal(ds.X, raster.stack[:, ds.rows, ds.cols].T)
    assert np.isfinite(ds.X).all()
    full = D.build_dataset(raster, pipeline, pos_gdf, bg_gdf, need_feature_stack=True)
    assert full.feature_stack.shape == (6, 120, 120) and full.feature_stack.dtype == np.float32


def test_build_dataset_drops_invalid_rows(raster, pipeline, pos_gdf, bg_gdf):
    x_nd, y_nd = rio_xy(raster.transform, 2, 2)                 # layer3 nodata burchagi
    extra = gpd.GeoDataFrame(geometry=[Point(x_nd, y_nd), Point(X0 - 5000, Y0 - 5000)], crs=pos_gdf.crs)
    pos = pd.concat([pos_gdf, extra], ignore_index=True)
    bg_bad = pd.concat([bg_gdf, gpd.GeoDataFrame(geometry=[Point(x_nd, y_nd)], crs=pos_gdf.crs)], ignore_index=True)
    msgs, log = _collect_log()
    ds = D.build_dataset(raster, pipeline, pos, bg_bad, log_fn=log)
    assert ds.n_pos == len(pos_gdf) and ds.n_neg == len(bg_gdf)
    assert any("2 ta musbat, 1 ta fon" in m for m in msgs)
    assert np.isfinite(ds.X).all()
    with pytest.raises(ValueError, match="musbat"):
        D.build_dataset(raster, pipeline, extra, bg_gdf)
    with pytest.raises(ValueError, match="fon"):
        D.build_dataset(raster, pipeline, pos_gdf, extra)
    wrong = D.FeaturePipeline(["a", "b"])
    with pytest.raises(ValueError, match="band_names"):
        D.build_dataset(raster, wrong, pos_gdf, bg_gdf)


def test_build_dataset_with_categorical(synth_project_cat):
    rs = D.load_and_align_rasters(D.find_tiff_files(synth_project_cat["tiff"]), categorical=["geology_cat"])
    pipe = D.FeaturePipeline(rs.band_names, rs.categorical).fit(rs)
    aoi = D.load_aoi(D.find_shapefile(synth_project_cat["aoi"]))
    pos = D.load_positive_points(D.find_shapefile(synth_project_cat["points"]), aoi)
    bg = D.generate_background_points(aoi, pos, 40, 300.0, 1, "grid", D.valid_pixel_mask(rs), rs.transform)
    ds = D.build_dataset(rs, pipe, pos, bg, need_feature_stack=True)
    assert ds.X.shape[1] == pipe.n_features == 8 and ds.feature_names == pipe.feature_names
    assert np.all(ds.X[:, :4].sum(axis=1) == 1)                 # har bir nuqta bitta darajada
    assert D.extract_patches(ds.feature_stack, ds.rows, ds.cols, 5).shape == (ds.n, 5, 5, 8)


def test_dataset_subset(raster, pipeline, pos_gdf, bg_gdf):
    ds = D.build_dataset(raster, pipeline, pos_gdf, bg_gdf, need_feature_stack=True)
    idx = np.array([0, 3, ds.n - 1, ds.n_pos])
    sub = ds.subset(idx)
    assert sub.feature_stack is ds.feature_stack and sub.n == 4 and sub.feature_names == ds.feature_names
    np.testing.assert_array_equal(sub.X, ds.X[idx])
    np.testing.assert_array_equal(sub.coords, ds.coords[idx])
    np.testing.assert_array_equal(sub.rows, ds.rows[idx])
    assert sub.n_pos == 2 and sub.n_neg == 2
    msk = ds.y == 0
    neg = ds.subset(msk)
    assert neg.n_pos == 0 and neg.n_neg == ds.n_neg
    np.testing.assert_array_equal(sub.get_patches(5), D.extract_patches(ds.feature_stack, ds.rows[idx], ds.cols[idx], 5))
    np.testing.assert_array_equal(ds.subset(np.array([], dtype=int)).X, np.empty((0, 6)))


# ---------------------------------------------------------------------------
# Diagnostika
# ---------------------------------------------------------------------------
def _manual_raster(arrays, names, categorical=()):
    stack = np.stack(arrays).astype(np.float32)
    return D.RasterStack(stack=stack, band_names=list(names), profile={}, transform=from_origin(X0, Y0 + 6000, 100, 100),
                         categorical=list(categorical))


def test_diagnostics_columns_and_values():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(60, 60))
    b = 2 * a + 0.01 * rng.normal(size=(60, 60))               # a bilan deyarli bir xil
    c = rng.normal(size=(60, 60))
    const = np.full((60, 60), 3.0)
    cat = rng.integers(1, 5, size=(60, 60))
    e = rng.normal(size=(60, 60))
    e[:20] = np.nan
    rs = _manual_raster([a, b, c, const, cat, e], ["a", "b", "c", "const", "cat", "e"], categorical=["cat"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # RuntimeWarning (nanmean va h.k.) bo'lmasligi kerak
        diag = D.data_diagnostics(rs, max_pixels=500, seed=1)
    assert set(diag) >= {"layer_stats", "corr", "vif", "high_corr_pairs"}
    ls = diag["layer_stats"]
    assert list(ls.columns) == ["band", "valid_pct", "min", "max", "mean", "std", "kind"]
    assert ls["band"].tolist() == ["a", "b", "c", "const", "cat", "e"]
    assert ls.set_index("band").loc["e", "valid_pct"] == pytest.approx(100 * 40 / 60, abs=0.01)
    assert ls.set_index("band").loc["cat", "kind"] == "kategorik" and ls.set_index("band").loc["a", "kind"] == "raqamli"
    assert ls.set_index("band").loc["const", "std"] == 0.0
    corr = diag["corr"]
    assert corr.index.tolist() == corr.columns.tolist() == ["a", "b", "c", "const", "e"]    # kategorik yo'q
    assert corr.loc["a", "a"] == 1.0 and corr.loc["a", "b"] > 0.99 and abs(corr.loc["a", "c"]) < 0.2
    assert corr.loc["const"].isna().all() and corr["const"].isna().all()
    assert np.allclose(corr.drop("const").drop(columns="const").values, corr.drop("const").drop(columns="const").values.T)
    vif = diag["vif"]
    assert list(vif.columns) == ["band", "vif", "flag"] and vif["band"].tolist() == ["a", "b", "c", "const", "e"]
    v = vif.set_index("band")
    assert v.loc["a", "vif"] > 100 and v.loc["a", "flag"] == "yuqori" and v.loc["b", "flag"] == "yuqori"
    assert v.loc["c", "vif"] < 1.5 and v.loc["c", "flag"] == "past"
    assert np.isnan(v.loc["const", "vif"]) and v.loc["const", "flag"] == "o'zgarmas"
    pairs = diag["high_corr_pairs"]
    assert list(pairs.columns) == ["band_a", "band_b", "corr"] and len(pairs) == 1
    assert (pairs.loc[0, "band_a"], pairs.loc[0, "band_b"]) == ("a", "b") and pairs.loc[0, "corr"] > 0.99
    assert 2 <= diag["n_sample"] <= 500 and diag["n_sample"] == 500
    again = D.data_diagnostics(rs, max_pixels=500, seed=1)
    pd.testing.assert_frame_equal(again["corr"], diag["corr"])
    assert D.data_diagnostics(rs, max_pixels=10 ** 6)["n_sample"] == 40 * 60


def test_diagnostics_vif_formula_and_singular():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(50, 50))
    b = 0.8 * a + 0.6 * rng.normal(size=(50, 50))
    rs = _manual_raster([a, b], ["a", "b"])
    diag = D.data_diagnostics(rs)
    r = diag["corr"].loc["a", "b"]
    np.testing.assert_allclose(diag["vif"]["vif"].to_numpy(), 1 / (1 - r ** 2), rtol=1e-4)
    assert (diag["vif"]["flag"] == "past").all() or (diag["vif"]["flag"] == "o'rta").all()
    dup = _manual_raster([a, a, rng.normal(size=(50, 50))], ["a", "a2", "c"])     # singular
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d2 = D.data_diagnostics(dup)
    v = d2["vif"].set_index("band")
    assert np.isfinite(v["vif"]).all() and v.loc["a", "vif"] > 1e5 and v.loc["a", "flag"] == "yuqori"
    assert v.loc["c", "vif"] < 2
    assert len(d2["high_corr_pairs"]) == 1


def test_diagnostics_edge_cases():
    rng = np.random.default_rng(2)
    one = D.data_diagnostics(_manual_raster([rng.normal(size=(20, 20))], ["a"]))
    assert one["vif"]["vif"].tolist() == [pytest.approx(1.0)] and one["vif"]["flag"].tolist() == ["past"]
    assert one["corr"].shape == (1, 1) and one["high_corr_pairs"].empty
    only_cat = D.data_diagnostics(_manual_raster([rng.integers(1, 3, size=(20, 20))], ["g"], ["g"]))
    assert only_cat["corr"].empty and only_cat["vif"].empty and only_cat["layer_stats"].shape[0] == 1
    assert list(only_cat["vif"].columns) == ["band", "vif", "flag"] and only_cat["n_sample"] == 0
    allnan = _manual_raster([np.full((20, 20), np.nan), rng.normal(size=(20, 20))], ["x", "y"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = D.data_diagnostics(allnan)
    ls = d["layer_stats"].set_index("band")
    assert ls.loc["x", "valid_pct"] == 0.0 and np.isnan(ls.loc["x", "mean"]) and d["n_sample"] == 0
    assert d["vif"]["flag"].tolist() == ["ma'lumot yetarli emas"] * 2
    with pytest.raises(ValueError, match="max_pixels"):
        D.data_diagnostics(allnan, max_pixels=1)


def test_diagnostics_real_data_and_dataset(raster, pipeline, pos_gdf, bg_gdf):
    ds = D.build_dataset(raster, pipeline, pos_gdf, bg_gdf)
    diag = D.data_diagnostics(raster, ds, max_pixels=3000)
    assert diag["n_sample"] == 3000 and diag["layer_stats"].shape == (6, 7)
    assert (diag["vif"]["flag"] == "past").all() and diag["vif"]["vif"].between(1, 2).all()
    assert diag["dataset_summary"]["n_pos"] == ds.n_pos and diag["dataset_summary"]["n_features"] == 6
    assert diag["dataset_summary"]["epv"] == pytest.approx(ds.n_pos / 6)
    fe = diag["feature_effects"]
    assert fe["feature"].tolist() == raster.band_names and fe["auc"].between(0, 1).all()
    assert fe.set_index("feature").loc["layer1", "auc"] > 0.7           # signal qatlami
    assert "dataset_summary" not in D.data_diagnostics(raster, max_pixels=100)
    tiny = ds.subset(np.r_[np.arange(5), np.arange(ds.n_pos, ds.n_pos + 5)])
    assert any("EPV" in w for w in D.data_diagnostics(raster, tiny, max_pixels=100)["dataset_summary"]["warnings"])


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def test_manual_metadata_and_dictionary(raster, tmp_path):
    folder = str(tmp_path)
    msgs, log = _collect_log()
    manual, path = D.load_or_create_manual_metadata(folder, raster.band_names, log_fn=log)
    assert os.path.isfile(path) and path.endswith("metadata.csv") and all(v == {} for v in manual.values())
    assert any("shablon" in m for m in msgs)
    assert any("KIRISH TIFF papkasiga" in m and path in m for m in msgs)          # yon ta'sir log'da aniq aytilgan
    tpl = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert tpl["band_name"].tolist() == raster.band_names and list(tpl.columns) == D.MANUAL_METADATA_FIELDS
    tpl.loc[0, "source_owner"] = "Geologiya xizmati"
    tpl.loc[1, "survey_date"] = "2020-05"
    tpl.to_csv(path, index=False, encoding="utf-8-sig")
    manual, _ = D.load_or_create_manual_metadata(folder, raster.band_names, log_fn=log)
    assert manual["layer1"]["source_owner"] == "Geologiya xizmati" and manual["layer2"]["survey_date"] == "2020-05"
    df = D.build_data_dictionary(raster.band_names, raster.tech_metadata, manual)
    assert len(df) == 6 and df.loc[0, "source_owner"] == "Geologiya xizmati" and df.loc[2, "notes"] == ""
    assert df["band_name"].tolist() == raster.band_names and "valid_pixels_pct" in df.columns
    out = D.save_data_dictionary(df, str(tmp_path / "sub" / "dir"), log_fn=log)
    assert os.path.basename(out) == "predictor_data_dictionary.csv" and len(pd.read_csv(out)) == 6
    with pytest.raises(ValueError, match="soni teng emas"):
        D.build_data_dictionary(raster.band_names[:3], raster.tech_metadata, manual)


def test_manual_metadata_never_overwrites_existing_file(tmp_path):
    path = tmp_path / "metadata.csv"
    junk = b"\xff\xfe\x00\x81garbage,\x00"
    path.write_bytes(junk)
    msgs, log = _collect_log()
    manual, _ = D.load_or_create_manual_metadata(str(tmp_path), ["a", "b"], log_fn=log)
    assert manual == {"a": {}, "b": {}} and path.read_bytes() == junk and any("o'qib bo'lmadi" in m for m in msgs)
    junk2 = b"\xff\xfe garbage \x81\n\x01\x02"                  # NUL yo'q: cp1251 bilan "o'qiladi", lekin jadval emas
    path.write_bytes(junk2)
    msgs.clear()
    manual, _ = D.load_or_create_manual_metadata(str(tmp_path), ["a", "b"], log_fn=log)
    assert manual == {"a": {}, "b": {}} and path.read_bytes() == junk2 and any("OGOHLANTIRISH" in m for m in msgs)
    path.write_bytes(b"")
    msgs.clear()
    manual, _ = D.load_or_create_manual_metadata(str(tmp_path), ["a"], log_fn=log)
    assert manual == {"a": {}} and path.read_bytes() == b"" and any("o'qib bo'lmadi" in m for m in msgs)
    path.write_text("x,y\n1,2\n", encoding="utf-8")
    manual, _ = D.load_or_create_manual_metadata(str(tmp_path), ["a"], log_fn=log)
    assert manual == {"a": {}} and path.read_text(encoding="utf-8") == "x,y\n1,2\n"
    path.write_text("band_name,notes\na,birinchi\na,ikkinchi\n b ,xx\n", encoding="utf-8")
    manual, _ = D.load_or_create_manual_metadata(str(tmp_path), ["a", "b", "c"], log_fn=log)
    assert manual["a"]["notes"] == "birinchi" and manual["b"]["notes"] == "xx" and manual["c"] == {}


@pytest.mark.parametrize("encoding, sep, owner", [
    ("cp1251", ";", "Институт геологии"),          # rus Excel: ';' + cp1251
    ("utf-8-sig", ";", "Geologiya xizmati"),         # ';' + UTF-8 BOM
    ("utf-8", ",", "Oʻzbekgeologiya"),                # oddiy UTF-8 (BOM'siz)
    ("cp1252", ";", "Servicio Geológico"),           # g'arbiy Excel: ';' + cp1252
    ("cp1251", ",", "Институт"),                     # vergul + cp1251
])
def test_manual_metadata_autodetects_delimiter_and_encoding(tmp_path, encoding, sep, owner):
    """Regressiya: Excel mintaqaviy sozlamasi (';' va cp1251/cp1252) metadata.csv'ni jim bo'sh qilib yubormasin."""
    path = tmp_path / "metadata.csv"
    text = sep.join(["band_name", "source_owner", "survey_date"]) + "\n" + sep.join(["layer1", owner, "2020"]) \
        + "\n" + sep.join(["layer2", "", "2021"]) + "\n"
    path.write_bytes(text.encode(encoding))
    before = path.read_bytes()
    msgs, log = _collect_log()
    manual, p = D.load_or_create_manual_metadata(str(tmp_path), ["layer1", "layer2", "layer3"], log_fn=log)
    assert manual["layer1"]["source_owner"] == owner and manual["layer1"]["survey_date"] == "2020"
    assert manual["layer2"]["survey_date"] == "2021" and manual["layer3"] == {}
    assert path.read_bytes() == before and not any("OGOHLANTIRISH" in m and "o'qib" in m for m in msgs)
    if sep == ";" or encoding.startswith("cp"):
        assert any("avtomatik aniqlandi" in m for m in msgs)


# ---------------------------------------------------------------------------
# Mustaqil review regressiyalari
# ---------------------------------------------------------------------------
def test_find_files_skip_hidden_files(tmp_path):
    """macOS '._x.tif' (AppleDouble) va yashirin fayllar ro'yxatga kirmaydi (asl koddagi glob ham olmagan)."""
    for name in ("a.tif", "._a.tif", ".hidden.tiff", "real.shp", ".hidden.shp", "._real.shp"):
        (tmp_path / name).write_bytes(b"")
    assert [os.path.basename(p) for p in D.find_tiff_files(str(tmp_path))] == ["a.tif"]
    assert os.path.basename(D.find_shapefile(str(tmp_path))) == "real.shp"


def test_multiband_file_warns_and_uses_first_band(tmp_path):
    path = tmp_path / "multi.tif"
    arr = np.stack([np.full((30, 30), 1.0), np.full((30, 30), 2.0)]).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=30, width=30, count=2, dtype="float32",
                       transform=from_origin(X0, Y0 + 3000, 100, 100), crs="EPSG:28411") as dst:
        dst.write(arr)
    msgs, log = _collect_log()
    r = D.load_and_align_rasters([str(path)], log_fn=log)
    assert r.stack.shape == (1, 30, 30) and (r.stack[0] == 1.0).all()
    assert any("OGOHLANTIRISH" in m and "2 ta band" in m for m in msgs)


def test_invalid_aoi_geometry_is_fixed_before_union(tmp_path):
    """Noto'g'ri (galstuk) poligonli AOI to'g'ridan-to'g'ri berilganda ham GEOS TopologyException chiqmaydi."""
    bow = Polygon([(X0 + 1000, Y0 + 1000), (X0 + 6000, Y0 + 6000), (X0 + 6000, Y0 + 1000), (X0 + 1000, Y0 + 6000)])
    assert not bow.is_valid
    aoi = gpd.GeoDataFrame(geometry=[bow, box(X0 + 7000, Y0 + 1000, X0 + 9000, Y0 + 3000)], crs="EPSG:28411")
    geom = D._aoi_geometry(aoi)
    assert geom.is_valid and geom.area > 0
    for strategy in ("random", "grid", "distance_weighted"):
        bg = D.generate_background_points(aoi, None, 40, 10.0, 1, strategy)
        xy = D._gdf_xy(bg)
        assert len(bg) == 40 and D._contains_xy(geom, xy[:, 0], xy[:, 1]).all()
    shp = str(tmp_path / "p.shp")
    gpd.GeoDataFrame(geometry=[Point(X0 + 2000, Y0 + 3500), Point(X0 + 8000, Y0 + 2000), Point(X0 + 9500, Y0 + 9500)],
                     crs="EPSG:28411").to_file(shp)
    assert len(D.load_positive_points(shp, aoi)) == 2


def test_background_crs_conversion_and_label():
    """AOI/musbatlar boshqa CRS'da bo'lsa ham koordinatalar 28411 da hosil bo'ladi va shunday belgilanadi."""
    aoi = gpd.GeoDataFrame(geometry=[box(X0 + 200, Y0 + 200, X0 + 6000, Y0 + 6000)], crs="EPSG:28411")
    pos = gpd.GeoDataFrame(geometry=[Point(X0 + 3000, Y0 + 3000)], crs="EPSG:28411")
    bg = D.generate_background_points(aoi.to_crs(4326), pos.to_crs(4326), 40, 500.0, 5)
    xy = D._gdf_xy(bg)
    assert len(bg) == 40 and bg.crs.to_epsg() == TARGET_EPSG and xy[:, 0].min() > 1e6
    assert D._contains_xy(D._aoi_geometry(aoi), xy[:, 0], xy[:, 1]).all()
    assert cKDTree(D._gdf_xy(pos)).query(xy)[0].min() > 499.0
    no_crs = gpd.GeoDataFrame(geometry=list(aoi.geometry), crs=None)
    assert D.generate_background_points(no_crs, None, 5, 0.0, 1).crs is None          # CRS yo'q => 28411 deb olinadi


def test_background_grid_thin_diagonal_aoi():
    """bbox/maydon nisbati juda katta (ingichka diagonal AOI): grid 0 nuqta qaytarmaydi."""
    side = 20000.0
    thin = Polygon([(X0, Y0), (X0 + side, Y0 + side), (X0 + side, Y0 + side - 150), (X0 + 150, Y0)])
    assert box(*thin.bounds).area / thin.area > 130
    aoi = gpd.GeoDataFrame(geometry=[thin], crs="EPSG:28411")
    for strategy in ("grid", "random"):
        bg = D.generate_background_points(aoi, None, 100, 50.0, 3, strategy)
        xy = D._gdf_xy(bg)
        assert len(bg) == 100 and D._contains_xy(thin, xy[:, 0], xy[:, 1]).all()


def test_extract_patches_integer_and_bool_stack():
    """Butun/bool dtype feature_stack: float32 ga o'tkaziladi, chegara NaN (bool'da NaN True bo'lib ketmaydi)."""
    rng = np.random.default_rng(0)
    fs = rng.integers(0, 5, size=(2, 12, 12)).astype(np.int16)
    rows, cols = np.array([0, 5, 11]), np.array([0, 6, 11])
    got = D.extract_patches(fs, rows, cols, 3)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, D.extract_patches(fs.astype(np.float32), rows, cols, 3))
    assert np.isnan(got[0, 0]).all() and np.isfinite(got[1]).all()
    flags = D.extract_patches(fs > 2, rows, cols, 3)
    assert np.isnan(flags[0, 0]).all() and set(np.unique(flags[1])) <= {0.0, 1.0}


def test_diagnostics_small_scale_layers_are_not_constant():
    """Kichik masshtabli (1e-14) lekin o'zgaruvchan qatlam 'o'zgarmas' emas; haqiqiy o'zgarmaslari esa shunday qoladi."""
    rng = np.random.default_rng(4)
    a = rng.normal(size=(40, 40))
    rs = _manual_raster([a * 1e-14, a * 1e-14 + 0.1e-14 * rng.normal(size=(40, 40)), rng.normal(size=(40, 40)),
                         np.full((40, 40), 5.0), np.zeros((40, 40))], ["t1", "t2", "n", "c5", "c0"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = D.data_diagnostics(rs)
    corr, v = d["corr"], d["vif"].set_index("band")
    assert corr.loc["t1", "t2"] > 0.9 and np.isfinite(corr.loc["t1", "t1"])
    assert v.loc["t1", "flag"] == "yuqori" and v.loc["t2", "flag"] == "yuqori" and v.loc["n", "flag"] == "past"
    assert v.loc["c5", "flag"] == "o'zgarmas" and v.loc["c0", "flag"] == "o'zgarmas" and np.isnan(v.loc["c5", "vif"])
    assert [(r.band_a, r.band_b) for r in d["high_corr_pairs"].itertuples()] == [("t1", "t2")]


def test_diagnostics_vif_needs_more_pixels_than_bands():
    """Piksellar soni raqamli bandlardan ko'p bo'lmasa korrelyatsiya singular: VIF hisoblanmaydi."""
    rs = _manual_raster([np.array([[1.0, 2.0]]), np.array([[2.0, 1.0]]), np.array([[3.0, 5.0]]), np.full((1, 2), 7.0)],
                        ["a", "b", "c", "k"])
    d = D.data_diagnostics(rs)
    assert d["n_sample"] == 2
    little = "ma'lumot yetarli emas"
    assert d["vif"].set_index("band")["flag"].to_dict() == {"a": little, "b": little, "c": little, "k": "o'zgarmas"}
    assert d["vif"]["vif"].isna().all()
    rng = np.random.default_rng(0)
    enough = D.data_diagnostics(_manual_raster([rng.normal(size=(1, 5)) for _ in range(3)], ["a", "b", "c"]))
    assert enough["n_sample"] == 5 and np.isfinite(enough["vif"]["vif"]).all()


def test_diagnostics_one_class_dataset_no_warning(raster, pipeline, pos_gdf, bg_gdf):
    """Bir sinfli Dataset (masalan CV qismi): nanmean/nol bo'linish ogohlantirishlarisiz NaN jadval."""
    ds = D.build_dataset(raster, pipeline, pos_gdf, bg_gdf)
    for part in (ds.subset(np.arange(ds.n_pos)), ds.subset(np.arange(ds.n_pos, ds.n))):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            d = D.data_diagnostics(raster, part, max_pixels=100)
        fe = d["feature_effects"]
        assert fe["feature"].tolist() == raster.band_names and fe[["mean_pos", "mean_neg", "cohen_d", "auc"]].isna().all().all()
        assert d["dataset_summary"]["n"] == part.n
