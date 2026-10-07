# -*- coding: utf-8 -*-
"""mpm.persist testlari: save_bundle / load_bundle (manifest, versiya, xatolar) va apply_bundle (band nomi bo'yicha
moslash, tartib, referens grid, yetishmaydigan band). CNN bundle testi - slow."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from importlib import metadata

import joblib
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from mpm import data, persist, predict
from mpm import models as models_mod
from mpm.common import ENSEMBLE_NAME, CancelledError, CancelToken
from mpm.config import default_hyperparams
from mpm.data import FeaturePipeline, RasterStack
from mpm.models import SklearnModel

SK3 = ("RandomForest", "SVM", "XGBoost")


# ---------------------------------------------------------------------------
# Yordamchilar / fixture'lar
# ---------------------------------------------------------------------------
def fast_hp():
    hp = default_hyperparams()
    hp["RandomForest"]["n_estimators"] = 15
    hp["XGBoost"]["n_estimators"] = 15
    return hp


def build_ctx(project, categorical=(), names=SK3, n_draws=2, permute=False, seed=0, need_stack=False):
    """Sintetik loyiha -> raster, pipeline, dataset va o'qitilgan modellar. permute=True: band tartibi teskari
    (alifbo tartibidan farq qiladi, bundle tartibi papka tartibiga mos kelmasligini sinash uchun)."""
    raster = data.load_and_align_rasters(data.find_tiff_files(project["tiff"]), categorical=list(categorical))
    if permute:
        order = list(range(raster.n_bands))[::-1]
        raster = RasterStack(stack=raster.stack[order], band_names=[raster.band_names[i] for i in order],
                             profile=raster.profile, transform=raster.transform, crs_epsg=raster.crs_epsg,
                             categorical=list(raster.categorical),
                             tech_metadata=[raster.tech_metadata[i] for i in order])
    pipe = FeaturePipeline(raster.band_names, raster.categorical).fit(raster)
    aoi = data.load_aoi(data.find_shapefile(project["aoi"]))
    pos = data.load_positive_points(data.find_shapefile(project["points"]), aoi)
    bg = data.generate_background_points(aoi, pos, 90, 500.0, random_state=1, valid_mask=data.valid_pixel_mask(raster),
                                         transform=raster.transform)
    ds = data.build_dataset(raster, pipe, pos, bg, need_feature_stack=need_stack)
    hp = fast_hp()
    fm = {n: [models_mod.make_model(n, hp[n], seed=seed + 10 * d).fit(ds.X, ds.y) for d in range(n_draws)]
          for n in names}
    return {"raster": raster, "pipe": pipe, "ds": ds, "fm": fm, "hp": hp, "project": project,
            "hp_used": {n: [hp[n]] * n_draws for n in names}}


def save(ctx, directory, **over):
    kw = dict(final_models=ctx["fm"], pipeline=ctx["pipe"], hyperparams_used=ctx["hp_used"],
              cfg_dict={"n_splits": 5, "seed": np.int64(42), "tiff_folder": "x"},
              metrics_summary={"RandomForest": {"auc": 0.91, "auc_ci95": (0.8, np.float32(0.95)), "brier": float("nan")}},
              thresholds={"RandomForest": 0.31}, block_size=2500.0, notes="sinov bundle")
    kw.update(over)
    return persist.save_bundle(str(directory), **kw)


@pytest.fixture(scope="module")
def ctx(synth_project_cat):
    """Kategorik qatlamli loyiha: RF + SVM + XGBoost, 2 draw."""
    return build_ctx(synth_project_cat, categorical=["geology_cat"])


@pytest.fixture(scope="module")
def ref(ctx):
    return predict.predict_probability_maps(ctx["raster"], ctx["pipe"], ctx["fm"])


@pytest.fixture(scope="module")
def bundle_dir(ctx, tmp_path_factory):
    d = tmp_path_factory.mktemp("bundle") / "b"
    save(ctx, d)
    return str(d)


@pytest.fixture(scope="module")
def loaded(bundle_dir):
    return persist.load_bundle(bundle_dir)


@pytest.fixture(scope="module")
def ctx_perm(synth_project):
    """6 raqamli qatlam, band tartibi teskari: layer6..layer1; RF + SVM, 1 draw."""
    return build_ctx(synth_project, names=("RandomForest", "SVM"), n_draws=1, permute=True, seed=5)


def make_folder(root, src_dir, rename=None, drop=()):
    os.makedirs(root, exist_ok=True)
    for f in sorted(os.listdir(src_dir)):
        stem, ext = os.path.splitext(f)
        if stem in drop:
            continue
        shutil.copy(os.path.join(src_dir, f), os.path.join(root, (rename(stem) if rename else stem) + ext))
    return str(root)


def read_first(path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile


def same(a, b):
    return np.array_equal(a, b, equal_nan=True)


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


# ---------------------------------------------------------------------------
# save_bundle
# ---------------------------------------------------------------------------
def test_save_bundle_structure(ctx, bundle_dir):
    d = bundle_dir
    assert sorted(os.listdir(d)) == ["README.txt", "manifest.json", "models", "pipeline.json"]
    for name in SK3:
        for draw in ("0", "1"):
            assert os.path.isfile(os.path.join(d, "models", name, draw, "model.joblib"))
    assert sorted(os.listdir(os.path.join(d, "models"))) == sorted(SK3)
    assert not any(f.endswith(".tmp") for f in os.listdir(d))


def test_save_bundle_returns_directory(ctx, tmp_path):
    assert save(ctx, tmp_path / "r") == str(tmp_path / "r")
    assert save(ctx, tmp_path / "p") == str(tmp_path / "p")           # papka o'zi yaratiladi


def test_manifest_contents(ctx, bundle_dir):
    with open(os.path.join(bundle_dir, "manifest.json"), encoding="utf-8") as f:
        text = f.read()

    def no_const(c):
        raise AssertionError(f"qat'iy bo'lmagan JSON konstantasi: {c}")

    m = json.loads(text, parse_constant=no_const)                        # NaN/Infinity yo'q
    assert m["bundle_version"] == persist.BUNDLE_VERSION == 1
    t = datetime.fromisoformat(m["created_utc"])
    assert t.utcoffset() == timedelta(0) and abs(datetime.now(timezone.utc) - t) < timedelta(hours=1)
    assert m["versions"]["scikit-learn"] == metadata.version("scikit-learn")
    assert "python" in m["versions"] and "numpy" in m["versions"] and m["mpm_version"]
    pipe = ctx["pipe"]
    assert m["band_names"] == pipe.band_names and m["feature_names"] == pipe.feature_names
    assert m["categorical"] == ["geology_cat"] and m["n_features"] == pipe.n_features
    assert m["model_names"] == list(SK3)
    for name in SK3:
        assert m["models"][name]["n_draws"] == 2 and m["models"][name]["loader"] == "joblib"
        assert m["models"][name]["input_kind"] == "tabular" and len(m["models"][name]["params"]) == 2
    assert m["hyperparams_used"]["RandomForest"][0]["n_estimators"] == 15
    assert m["cfg"] == {"n_splits": 5, "seed": 42, "tiff_folder": "x"}
    assert m["metrics_summary"]["RandomForest"]["auc"] == 0.91
    assert m["metrics_summary"]["RandomForest"]["auc_ci95"] == [0.8, pytest.approx(0.95)]
    assert m["metrics_summary"]["RandomForest"]["brier"] is None       # NaN -> null
    assert m["thresholds"] == {"RandomForest": 0.31} and m["block_size"] == 2500.0
    assert m["crs_epsg"] == 28411 and m["notes"] == "sinov bundle" and m["ensemble_name"] == ENSEMBLE_NAME
    assert "joblib" in m["security_warning"] and "ishonchli" in m["security_warning"]
    assert jload(os.path.join(bundle_dir, "pipeline.json")) == json.loads(json.dumps(pipe.to_dict()))


def test_readme(ctx, bundle_dir):
    with open(os.path.join(bundle_dir, "README.txt"), encoding="utf-8") as f:
        txt = f.read()
    for token in ("XAVFSIZLIK", "joblib", "pickle", "ishonchli", "load_bundle", "apply_bundle", "manifest.json",
                  "pipeline.json", "prospektivlik indeksi", "geology_cat", "layer1"):
        assert token in txt, token


def test_save_defaults_and_optional_args(ctx, tmp_path):
    d = tmp_path / "min"
    persist.save_bundle(str(d), final_models={"SVM": ctx["fm"]["SVM"][0]}, pipeline=ctx["pipe"],
                        hyperparams_used={}, cfg_dict={})
    m = jload(d / "manifest.json")
    assert m["metrics_summary"] is None and m["thresholds"] is None and m["block_size"] is None
    assert m["crs_epsg"] == 28411 and m["notes"] == "" and m["models"]["SVM"]["n_draws"] == 1
    b = persist.load_bundle(str(d))
    assert list(b["final_models"]) == ["SVM"] and len(b["final_models"]["SVM"]) == 1


def test_save_bundle_validation(ctx, tmp_path):
    with pytest.raises(ValueError, match="final_models"):
        save(ctx, tmp_path / "a", final_models={})
    with pytest.raises(ValueError, match="bo'sh"):
        save(ctx, tmp_path / "b", final_models={"SVM": []})
    unfit = models_mod.make_model("SVM", default_hyperparams()["SVM"])
    with pytest.raises(ValueError, match="o'qitilmagan"):
        save(ctx, tmp_path / "c", final_models={"SVM": [unfit]})
    X = np.random.default_rng(0).normal(size=(40, 3))
    narrow = models_mod.make_model("RandomForest", fast_hp()["RandomForest"]).fit(X, np.r_[np.ones(10), np.zeros(30)])
    with pytest.raises(ValueError, match="feature"):
        save(ctx, tmp_path / "d", final_models={"RandomForest": [narrow]})
    assert not (tmp_path / "a" / "manifest.json").exists()


def test_save_bundle_overwrite_replaces_stale_models(ctx, tmp_path):
    d = tmp_path / "ow"
    save(ctx, d)
    assert (d / "models" / "XGBoost" / "1").is_dir()
    save(ctx, d, final_models={"RandomForest": ctx["fm"]["RandomForest"][:1]}, notes="ikkinchi")
    assert sorted(os.listdir(d / "models")) == ["RandomForest"] and sorted(os.listdir(d / "models" / "RandomForest")) == ["0"]
    b = persist.load_bundle(str(d))
    assert list(b["final_models"]) == ["RandomForest"] and b["notes"] == "ikkinchi"


def test_save_bundle_refuses_foreign_models_dir_and_keeps_other_files(ctx, tmp_path):
    foreign = tmp_path / "foreign"
    (foreign / "models").mkdir(parents=True)
    (foreign / "models" / "mine.txt").write_text("muhim")
    with pytest.raises(ValueError, match="bundle emas"):
        save(ctx, foreign)
    assert (foreign / "models" / "mine.txt").read_text() == "muhim" and not (foreign / "manifest.json").exists()
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("saqlanadi")
    save(ctx, other)
    assert (other / "notes.txt").read_text() == "saqlanadi" and (other / "manifest.json").exists()


def test_incomplete_save_is_not_loadable(ctx, tmp_path, monkeypatch):
    d = tmp_path / "broken"
    save(ctx, d)                                                  # to'liq bundle mavjud
    w = ctx["fm"]["SVM"][1]

    def boom(directory):
        raise OSError("disk to'ldi")

    monkeypatch.setattr(w, "save", boom)
    with pytest.raises(OSError, match="disk"):
        save(ctx, d)
    assert not (d / "manifest.json").exists()                     # yarim bundle yaroqli ko'rinmaydi
    with pytest.raises(FileNotFoundError, match="manifest"):
        persist.load_bundle(str(d))


def test_save_bundle_retry_after_failed_save_succeeds(ctx, tmp_path, monkeypatch):
    """Yarim qolgan saqlashdan keyin xuddi shu papkaga qayta saqlash mumkin (models/ 'begona' deb rad etilmaydi)."""
    d = tmp_path / "retry"
    w = ctx["fm"]["SVM"][1]
    orig = w.save

    def boom(directory):
        raise OSError("disk to'ldi")

    monkeypatch.setattr(w, "save", boom)
    with pytest.raises(OSError, match="disk"):
        save(ctx, d)
    assert (d / "models").exists() and not (d / "manifest.json").exists()
    monkeypatch.setattr(w, "save", orig)
    save(ctx, d)                                                  # qayta urinish xato bermasligi kerak
    assert not (d / persist.INCOMPLETE_MARKER).exists()           # marker yakunda o'chiriladi
    assert sorted(os.listdir(d)) == ["README.txt", "manifest.json", "models", "pipeline.json"]
    assert list(persist.load_bundle(str(d))["final_models"]) == list(SK3)


def test_save_bundle_refuses_foreign_manifest_and_keeps_files(ctx, tmp_path):
    """Boshqa dasturning manifest.json (+ models/) papkasi MPM bundle deb o'chirib yuborilmaydi."""
    d = tmp_path / "foreign_manifest"
    (d / "models").mkdir(parents=True)
    (d / "models" / "mine.txt").write_text("muhim")
    jdump({"name": "boshqa dastur"}, str(d / "manifest.json"))
    with pytest.raises(ValueError, match="manifest"):
        save(ctx, d)
    assert (d / "models" / "mine.txt").read_text() == "muhim"
    assert jload(str(d / "manifest.json")) == {"name": "boshqa dastur"}
    d2 = tmp_path / "foreign_list"
    d2.mkdir()
    jdump([1, 2], str(d2 / "manifest.json"))
    with pytest.raises(ValueError, match="manifest"):
        save(ctx, d2)
    assert jload(str(d2 / "manifest.json")) == [1, 2]


# ---------------------------------------------------------------------------
# load_bundle
# ---------------------------------------------------------------------------
def test_roundtrip_identical_maps(ctx, ref, loaded):
    assert set(loaded) >= {"final_models", "pipeline", "manifest", "cfg", "hyperparams_used"}
    assert list(loaded["final_models"]) == list(SK3)
    for name in SK3:
        assert len(loaded["final_models"][name]) == 2
        for a, b in zip(ctx["fm"][name], loaded["final_models"][name]):
            assert isinstance(b, SklearnModel) and b.is_fitted and b.name == name and b.params == a.params
            np.testing.assert_array_equal(a.predict_proba_pos(ctx["ds"].X), b.predict_proba_pos(ctx["ds"].X))
    out = predict.predict_probability_maps(ctx["raster"], loaded["pipeline"], loaded["final_models"])
    assert list(out["maps"]) == list(ref["maps"])
    for k in ref["maps"]:
        assert same(out["maps"][k], ref["maps"][k]), k                  # bir xil xarita (bit-darajasida)
    assert same(out["uncertainty"], ref["uncertainty"]) and same(out["valid_mask"], ref["valid_mask"])


def test_loaded_metadata(ctx, loaded, bundle_dir):
    pipe = loaded["pipeline"]
    assert isinstance(pipe, FeaturePipeline)
    assert pipe.band_names == ctx["pipe"].band_names and pipe.feature_names == ctx["pipe"].feature_names
    assert pipe.levels == ctx["pipe"].levels and pipe.categorical == ["geology_cat"]
    assert loaded["cfg"] == {"n_splits": 5, "seed": 42, "tiff_folder": "x"}
    assert loaded["hyperparams_used"]["XGBoost"][1]["n_estimators"] == 15
    assert loaded["thresholds"] == {"RandomForest": 0.31} and loaded["block_size"] == 2500.0
    assert loaded["crs_epsg"] == 28411 and loaded["notes"] == "sinov bundle" and loaded["directory"] == bundle_dir
    assert loaded["metrics_summary"]["RandomForest"]["auc"] == 0.91
    assert loaded["manifest"]["bundle_version"] == 1
    # pipeline transform natijasi o'zgarmagan
    np.testing.assert_array_equal(pipe.transform_stack(ctx["raster"]), ctx["pipe"].transform_stack(ctx["raster"]))


def test_load_log_warns_about_trust(bundle_dir):
    log = []
    persist.load_bundle(bundle_dir, log_fn=log.append)
    assert any("ishonchli" in m for m in log) and any("Yuklandi: XGBoost" in m for m in log)


def copy_bundle(bundle_dir, dst):
    shutil.copytree(bundle_dir, dst)
    return str(dst)


def edit_manifest(d, fn):
    p = os.path.join(d, "manifest.json")
    m = jload(p)
    fn(m)
    jdump(m, p)


def test_load_bundle_missing_or_not_a_bundle(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        persist.load_bundle(str(tmp_path / "yo'q"))
    with pytest.raises(FileNotFoundError, match="manifest"):
        persist.load_bundle(str(tmp_path))


def test_load_bundle_version_checks(bundle_dir, tmp_path):
    d = copy_bundle(bundle_dir, tmp_path / "v")
    edit_manifest(d, lambda m: m.update(bundle_version=persist.BUNDLE_VERSION + 1))
    with pytest.raises(ValueError, match="yangi"):
        persist.load_bundle(d)
    for bad in (None, "1", True, 0):
        def setv(m, bad=bad):
            if bad is None:
                m.pop("bundle_version")
            else:
                m["bundle_version"] = bad
        edit_manifest(d, setv)
        with pytest.raises(ValueError, match="bundle_version|versiya"):
            persist.load_bundle(d)
    (tmp_path / "v" / "manifest.json").write_text("{buzilgan json")
    with pytest.raises(ValueError, match="manifest"):
        persist.load_bundle(d)


def test_load_bundle_missing_parts(bundle_dir, tmp_path):
    d = copy_bundle(bundle_dir, tmp_path / "m")
    shutil.rmtree(os.path.join(d, "models", "SVM", "1"))
    with pytest.raises(FileNotFoundError, match="SVM"):
        persist.load_bundle(d)
    d2 = copy_bundle(bundle_dir, tmp_path / "m2")
    os.remove(os.path.join(d2, "pipeline.json"))
    with pytest.raises(FileNotFoundError, match="pipeline"):
        persist.load_bundle(d2)
    d3 = copy_bundle(bundle_dir, tmp_path / "m3")
    edit_manifest(d3, lambda m: m.update(models={}, model_names=[]))
    with pytest.raises(ValueError, match="modellar"):
        persist.load_bundle(d3)
    d4 = copy_bundle(bundle_dir, tmp_path / "m4")
    edit_manifest(d4, lambda m: m["models"]["SVM"].update(n_draws=0))
    with pytest.raises(ValueError, match="SVM"):
        persist.load_bundle(d4)


def test_load_bundle_manifest_not_an_object(bundle_dir, tmp_path):
    d = tmp_path / "list_manifest"
    shutil.copytree(bundle_dir, d)
    jdump([1, 2, 3], str(d / "manifest.json"))
    with pytest.raises(ValueError, match="manifest"):
        persist.load_bundle(str(d))


def test_load_bundle_corrupt_model_file_gives_clear_error(bundle_dir, tmp_path):
    d = tmp_path / "corrupt_model"
    shutil.copytree(bundle_dir, d)
    (d / "models" / "SVM" / "0" / "model.joblib").write_bytes(b"buzilgan")
    with pytest.raises(ValueError, match="SVM"):
        persist.load_bundle(str(d))


def test_load_bundle_detects_pipeline_mismatch(bundle_dir, tmp_path):
    d = copy_bundle(bundle_dir, tmp_path / "pm")
    p = os.path.join(d, "pipeline.json")
    obj = jload(p)
    obj["band_names"] = obj["band_names"][::-1]
    jdump(obj, p)
    with pytest.raises(ValueError, match="mos emas"):
        persist.load_bundle(d)
    d2 = copy_bundle(bundle_dir, tmp_path / "pm2")
    edit_manifest(d2, lambda m: m.update(feature_names=m["feature_names"][:-1]))
    with pytest.raises(ValueError, match="feature"):
        persist.load_bundle(d2)


def test_load_bundle_rejects_non_wrapper_object(bundle_dir, tmp_path):
    d = copy_bundle(bundle_dir, tmp_path / "nw")
    joblib.dump({"zararli": "emas"}, os.path.join(d, "models", "SVM", "0", "model.joblib"))
    with pytest.raises(ValueError, match="ModelWrapper"):
        persist.load_bundle(d)


def test_load_bundle_warns_on_library_version_difference(bundle_dir, tmp_path):
    d = copy_bundle(bundle_dir, tmp_path / "ver")
    edit_manifest(d, lambda m: m["versions"].update({"scikit-learn": "0.0.1"}))
    log = []
    persist.load_bundle(d, log_fn=log.append)
    assert any("versiyalari" in m and "scikit-learn" in m and "0.0.1" in m for m in log)


# ---------------------------------------------------------------------------
# apply_bundle
# ---------------------------------------------------------------------------
def test_apply_bundle_other_folder(ctx, ref, loaded, tmp_path):
    src = ctx["project"]["tiff"]
    new = make_folder(tmp_path / "new", src, rename=lambda s: s.upper())        # LAYER1.tif, GEOLOGY_CAT.tif ...
    # ortiqcha va yashirin fayllar: alifbo bo'yicha birinchi bo'lgan ortiqcha TIFF, yashirin fayl, boshqa kengaytma
    shutil.copy(os.path.join(src, "layer1.tif"), os.path.join(new, "aaa_extra.tif"))
    with open(os.path.join(new, "._LAYER1.tif"), "wb") as f:
        f.write(b"junk")
    with open(os.path.join(new, "readme.txt"), "w") as f:
        f.write("x")
    log, calls = [], []
    out_dir = tmp_path / "out"
    res = persist.apply_bundle(loaded, new, out_dir=str(out_dir), log_fn=log.append,
                               progress_fn=lambda f, m: calls.append(f))
    assert same(res["maps"][ENSEMBLE_NAME], ref["maps"][ENSEMBLE_NAME])
    for name in SK3:
        assert same(res["maps"][name], ref["maps"][name])
    assert same(res["uncertainty"], ref["uncertainty"]) and same(res["valid_mask"], ref["valid_mask"])
    assert set(res) >= {"maps", "uncertainty", "valid_mask", "class_map", "class_breaks", "raster", "saved_paths",
                        "class_stats"}
    r = res["raster"]
    assert isinstance(r, RasterStack) and r.band_names == ctx["pipe"].band_names and r.categorical == ["geology_cat"]
    assert r.stack.shape == ctx["raster"].stack.shape and r.profile["nodata"] == -9999.0
    np.testing.assert_array_equal(r.stack, ctx["raster"].stack)
    assert res["matched_files"]["layer3"].endswith("LAYER3.tif") and len(res["matched_files"]) == 5
    assert res["ignored_files"] == ["aaa_extra.tif"]
    assert any("aaa_extra.tif" in m and "e'tiborsiz" in m for m in log)
    assert not any("junk" in m or "._LAYER1" in m for m in log)
    # sinflash va statistika (konsiz)
    cm, br = res["class_map"], res["class_breaks"]
    assert cm.dtype == np.int8 and len(br) == 4 and set(np.unique(cm)) == {0, 1, 2, 3, 4, 5}
    assert np.array_equal(cm > 0, res["valid_mask"])
    cm_ref, br_ref = predict.classify_map(ref["maps"][ENSEMBLE_NAME])
    np.testing.assert_array_equal(cm, cm_ref)
    assert br == br_ref
    st = res["class_stats"]
    assert st["Piksel"].sum() == int(res["valid_mask"].sum()) and st["Konlar_soni"].isna().all()
    # saqlangan fayllar
    names = sorted(os.path.basename(p) for p in res["saved_paths"])
    assert names == sorted([f"prognoz_{n}.tif" for n in SK3] + ["prognoz_Ensemble_soft_voting.tif",
                                                                  "prognoz_classes.tif", "prognoz_uncertainty.tif"])
    assert all(os.path.isfile(p) for p in res["saved_paths"])
    arr, prof = read_first(str(out_dir / "prognoz_classes.tif"))
    np.testing.assert_array_equal(arr, cm)
    assert prof["dtype"] == "int8" and prof["nodata"] == 0
    arr, prof = read_first(str(out_dir / "prognoz_Ensemble_soft_voting.tif"))
    np.testing.assert_array_equal(arr, np.where(np.isnan(ref["maps"][ENSEMBLE_NAME]), -9999.0,
                                                ref["maps"][ENSEMBLE_NAME]).astype(np.float32))
    assert prof["nodata"] == -9999.0 and not prof["tiled"]
    # progress: monoton, 100% bilan tugaydi
    assert calls == sorted(calls) and calls[-1] == 1.0 and calls[0] < 0.5


def test_apply_bundle_band_order_follows_bundle_not_folder(ctx_perm, tmp_path):
    ctx = ctx_perm
    assert ctx["pipe"].band_names == ["layer6", "layer5", "layer4", "layer3", "layer2", "layer1"]
    d = tmp_path / "pb"
    save(ctx, d)
    ref = predict.predict_probability_maps(ctx["raster"], ctx["pipe"], ctx["fm"])
    new = make_folder(tmp_path / "newperm", ctx["project"]["tiff"],
                      rename=lambda s: s.capitalize() if s.endswith(("1", "3", "5")) else s.upper())
    assert sorted(os.listdir(new)) == ["LAYER2.tif", "LAYER4.tif", "LAYER6.tif", "Layer1.tif", "Layer3.tif", "Layer5.tif"]
    res = persist.apply_bundle(persist.load_bundle(str(d)), new)
    assert same(res["maps"][ENSEMBLE_NAME], ref["maps"][ENSEMBLE_NAME])        # tartib bundle'niki
    for n in ("RandomForest", "SVM"):
        assert same(res["maps"][n], ref["maps"][n])
    assert res["raster"].band_names == ctx["pipe"].band_names
    np.testing.assert_array_equal(res["raster"].stack, ctx["raster"].stack)
    assert res["saved_paths"] == [] and res["uncertainty"] is not None


def test_apply_bundle_accepts_directory_path(ref, bundle_dir, ctx, tmp_path):
    new = make_folder(tmp_path / "pth", ctx["project"]["tiff"])
    res = persist.apply_bundle(bundle_dir, new)
    assert same(res["maps"][ENSEMBLE_NAME], ref["maps"][ENSEMBLE_NAME])


def test_apply_bundle_missing_band_error(loaded, ctx, tmp_path):
    new = make_folder(tmp_path / "miss", ctx["project"]["tiff"], drop=("layer3", "layer4"), rename=str.upper)
    with pytest.raises(ValueError) as e:
        persist.apply_bundle(loaded, new)
    msg = str(e.value)
    assert "['layer3', 'layer4']" in msg                                    # yetishmaydigan bandlar
    assert "LAYER1" in msg and "GEOLOGY_CAT" in msg and "LAYER3" not in msg  # mavjud nomlar ro'yxati
    assert "Yetishmaydigan" in msg and "mavjud" in msg


def test_apply_bundle_folder_errors(loaded, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="topilmadi"):
        persist.apply_bundle(loaded, str(empty))
    with pytest.raises(ValueError, match="topilmadi"):
        persist.apply_bundle(loaded, str(tmp_path / "yo'q"))


def test_apply_bundle_ambiguous_names(loaded, ctx, tmp_path):
    new = make_folder(tmp_path / "amb", ctx["project"]["tiff"], rename=str.upper)
    shutil.copy(os.path.join(new, "LAYER2.tif"), os.path.join(new, "Layer2.tif"))
    with pytest.raises(ValueError, match="layer2"):
        persist.apply_bundle(loaded, new)


def test_apply_bundle_exact_case_preferred(ctx, tmp_path):
    """Nomi harf registi bilan farq qiladigan ikki band bo'lsa, aniq moslik tanlanadi."""
    rng = np.random.default_rng(0)
    root = tmp_path / "case"
    root.mkdir()
    tr = ctx["raster"].transform
    arrs = {n: rng.normal(size=(30, 30)).astype("float32") for n in ("a", "A")}
    for n, a in arrs.items():
        with rasterio.open(root / f"{n}.tif", "w", driver="GTiff", height=30, width=30, count=1, dtype="float32",
                           crs="EPSG:28411", transform=tr, nodata=-9999) as dst:
            dst.write(a, 1)
    pipe = FeaturePipeline(["a", "A"])
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] > 0).astype(int)
    m = models_mod.make_model("RandomForest", fast_hp()["RandomForest"]).fit(X, y)
    d = tmp_path / "bcase"
    persist.save_bundle(str(d), final_models={"RandomForest": [m]}, pipeline=pipe, hyperparams_used={}, cfg_dict={})
    res = persist.apply_bundle(str(d), str(root))
    want = m.predict_proba_pos(np.column_stack([arrs["a"].ravel(), arrs["A"].ravel()]))
    np.testing.assert_allclose(res["maps"]["RandomForest"].ravel(), want, atol=1e-6)


def test_apply_bundle_reference_grid_is_first_selected_file(ctx_perm, tmp_path):
    ctx = ctx_perm
    d = tmp_path / "rg"
    save(ctx, d)
    src = ctx["project"]["tiff"]
    new = make_folder(tmp_path / "rgnew", src)
    # layer1 (alifbo bo'yicha birinchi KERAKLI fayl) 2x yirik gridda; undan oldin turuvchi ortiqcha fayl boshqa gridda
    with rasterio.open(os.path.join(src, "layer1.tif")) as s:
        tr = s.transform
        arr = s.read(1)[::2, ::2]
    coarse = Affine(tr.a * 2, tr.b, tr.c, tr.d, tr.e * 2, tr.f)
    for name, a, t in (("layer1", arr, coarse), ("a_extra", arr[:20, :20], Affine(500.0, 0, tr.c, 0, -500.0, tr.f))):
        with rasterio.open(os.path.join(new, name + ".tif"), "w", driver="GTiff", height=a.shape[0], width=a.shape[1],
                           count=1, dtype="float32", crs="EPSG:28411", transform=t, nodata=-9999) as dst:
            dst.write(a, 1)
    log = []
    res = persist.apply_bundle(str(d), new, out_dir=str(tmp_path / "rgout"), log_fn=log.append)
    assert res["raster"].stack.shape[1:] == (60, 60) and res["raster"].transform == coarse
    assert res["maps"][ENSEMBLE_NAME].shape == (60, 60) and res["valid_mask"].any()
    assert res["ignored_files"] == ["a_extra.tif"]
    assert any("referens" in m and "layer1.tif" in m for m in log)
    assert res["raster"].band_names == ctx["pipe"].band_names                       # tartib baribir bundle'niki
    _, prof = read_first(res["saved_paths"][0])
    assert (prof["height"], prof["width"]) == (60, 60) and prof["transform"] == coarse


def test_apply_bundle_passes_new_categorical_stems(loaded, ctx, tmp_path, monkeypatch):
    new = make_folder(tmp_path / "cs", ctx["project"]["tiff"], rename=str.upper)
    seen = {}
    orig = persist.load_and_align_rasters

    def spy(paths, categorical=(), **kw):
        seen["cat"] = list(categorical)
        seen["paths"] = list(paths)
        return orig(paths, categorical=categorical, **kw)

    monkeypatch.setattr(persist, "load_and_align_rasters", spy)
    persist.apply_bundle(loaded, new)
    assert seen["cat"] == ["GEOLOGY_CAT"]                      # kategorik bandlar bundle'dan, yangi fayl nomi bilan
    assert [os.path.basename(p) for p in seen["paths"]][0] == "GEOLOGY_CAT.tif"      # papka tartibi (referens = birinchi)


def test_apply_bundle_warns_about_unknown_categorical_levels(loaded, ctx, tmp_path):
    new = make_folder(tmp_path / "unk", ctx["project"]["tiff"])
    p = os.path.join(new, "geology_cat.tif")
    with rasterio.open(p, "r+") as dst:
        a = dst.read(1)
        a[:10, :10] = 9                                          # o'qitishda bo'lmagan daraja
        dst.write(a, 1)
    log = []
    res = persist.apply_bundle(loaded, new, log_fn=log.append)
    assert any("geology_cat" in m and "9" in m and "daraja" in m for m in log)
    assert np.isfinite(res["maps"][ENSEMBLE_NAME][res["valid_mask"]]).all()


def test_apply_bundle_class_methods(loaded, ctx, tmp_path):
    new = make_folder(tmp_path / "cm", ctx["project"]["tiff"])
    fixed = persist.apply_bundle(loaded, new, class_method="fixed", class_breaks=[0.1, 0.3])
    assert fixed["class_breaks"] == [0.1, 0.3] and set(np.unique(fixed["class_map"])) <= {0, 1, 2, 3}
    assert len(fixed["class_stats"]) == 3
    eq = persist.apply_bundle(loaded, new, class_method="equal_interval", n_classes=4)
    assert eq["class_breaks"] == [0.25, 0.5, 0.75] and len(eq["class_stats"]) == 4
    q3 = persist.apply_bundle(loaded, new, n_classes=3)
    assert len(q3["class_breaks"]) == 2 and len(q3["class_stats"]) == 3
    with pytest.raises(ValueError, match="usuli"):
        persist.apply_bundle(loaded, new, class_method="jenks")


def test_apply_bundle_invalid_class_args_fail_before_prediction(loaded, ctx, tmp_path, monkeypatch):
    new = make_folder(tmp_path / "early", ctx["project"]["tiff"])
    monkeypatch.setattr(persist, "predict_probability_maps",
                        lambda *a, **k: pytest.fail("bashorat boshlanmasligi kerak edi"))
    with pytest.raises(ValueError):
        persist.apply_bundle(loaded, new, class_method="fixed", class_breaks=[0.5, 0.2])
    with pytest.raises(ValueError):
        persist.apply_bundle(loaded, new, n_classes=1)


def test_apply_bundle_cancel(loaded, ctx, tmp_path):
    new = make_folder(tmp_path / "cn", ctx["project"]["tiff"])
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledError):
        persist.apply_bundle(loaded, new, cancel=tok)
    tok2 = CancelToken()
    n = []

    def prog(f, m):
        n.append(f)
        if len(n) == 3:
            tok2.cancel()

    with pytest.raises(CancelledError):
        persist.apply_bundle(loaded, new, batch_size=1000, progress_fn=prog, cancel=tok2)
    assert not os.path.exists(tmp_path / "cn_out")


def test_apply_bundle_single_model_has_no_uncertainty_file(ctx, tmp_path):
    d = tmp_path / "one"
    save(ctx, d, final_models={"SVM": ctx["fm"]["SVM"]})
    new = make_folder(tmp_path / "onenew", ctx["project"]["tiff"])
    res = persist.apply_bundle(str(d), new, out_dir=str(tmp_path / "oneout"))
    assert res["uncertainty"] is None
    assert sorted(os.path.basename(p) for p in res["saved_paths"]) == [
        "prognoz_Ensemble_soft_voting.tif", "prognoz_SVM.tif", "prognoz_classes.tif"]


def test_apply_bundle_all_nodata_new_area(loaded, ctx, tmp_path):
    new = make_folder(tmp_path / "nod", ctx["project"]["tiff"])
    with rasterio.open(os.path.join(new, "layer2.tif"), "r+") as dst:
        dst.write(np.full((dst.height, dst.width), -9999.0, dtype="float32"), 1)
    with pytest.raises(ValueError, match="yaroqli piksel"):
        persist.apply_bundle(loaded, new)


# ---------------------------------------------------------------------------
# CNN bundle (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_bundle_with_cnn_roundtrip_and_apply(synth_project, tmp_path):
    pytest.importorskip("tensorflow")
    ctx = build_ctx(synth_project, names=("RandomForest",), n_draws=1, need_stack=True, seed=2)
    ds, pipe = ctx["ds"], ctx["pipe"]
    params = {"mode": "patch2d", "window": 5, "filters1": 4, "filters2": 0, "dense_units": 4, "epochs": 2,
              "batch_size": 16, "patience": 2, "augment": False}
    cnn = models_mod.make_model("CNN", params, seed=3, n_features=pipe.n_features)
    cnn.fit(None, ds.y, patches=ds.get_patches(5))
    ctx["fm"]["CNN"] = [cnn]
    ref = predict.predict_probability_maps(ctx["raster"], pipe, ctx["fm"], batch_size=3000)
    d = tmp_path / "cnnb"
    save(ctx, d, hyperparams_used={"CNN": [params]})
    m = jload(d / "manifest.json")
    assert m["models"]["CNN"]["loader"] == "keras" and m["models"]["CNN"]["input_kind"] == "patch"
    assert sorted(os.listdir(d / "models" / "CNN" / "0")) == ["meta.json", "model.keras"]
    assert os.path.isfile(d / "models" / "RandomForest" / "0" / "model.joblib")
    b = persist.load_bundle(str(d))
    from mpm.cnn import CNNModel
    loaded_cnn = b["final_models"]["CNN"][0]
    assert isinstance(loaded_cnn, CNNModel) and loaded_cnn.input_kind == "patch" and loaded_cnn.is_fitted
    out = predict.predict_probability_maps(ctx["raster"], b["pipeline"], b["final_models"], batch_size=3000)
    np.testing.assert_allclose(out["maps"]["CNN"], ref["maps"]["CNN"], atol=1e-5, equal_nan=True)
    assert same(out["maps"]["RandomForest"], ref["maps"]["RandomForest"])
    np.testing.assert_allclose(out["maps"][ENSEMBLE_NAME], ref["maps"][ENSEMBLE_NAME], atol=1e-5, equal_nan=True)
    new = make_folder(tmp_path / "cnnnew", synth_project["tiff"], rename=str.upper)
    res = persist.apply_bundle(b, new, out_dir=str(tmp_path / "cnnout"), batch_size=3000)
    np.testing.assert_allclose(res["maps"]["CNN"], ref["maps"]["CNN"], atol=1e-5, equal_nan=True)
    assert any(p.endswith("prognoz_CNN.tif") for p in res["saved_paths"])
