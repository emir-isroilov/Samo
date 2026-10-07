# -*- coding: utf-8 -*-
"""mpm.models testlari: make_model, SklearnModel (RF / SVM / XGBoost), kalibrlash, parametr-qamrov."""
import sys
import warnings

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted

from mpm import models as models_mod
from mpm.base import ModelWrapper
from mpm.common import CancelledError, CancelToken
from mpm.config import MODEL_NAMES, PARAM_SPECS, default_hyperparams
from mpm.models import SklearnModel, make_model

SK_MODELS = ("RandomForest", "SVM", "XGBoost")
FAST = {"RandomForest": {"n_estimators": 30}, "SVM": {}, "XGBoost": {"n_estimators": 25}}


def hp(name, **over):
    """Tez testlar uchun kichraytirilgan standart giperparametrlar."""
    p = dict(default_hyperparams()[name])
    p.update(FAST[name])
    p.update(over)
    return p


def make_xy(n_pos, n=90, p=5, seed=0):
    """0-feature'da signal bor sintetik ma'lumot; musbatlar BIRINCHI (loyiha konvensiyasi)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = np.r_[np.ones(n_pos), np.zeros(n - n_pos)].astype(int)
    X[:n_pos, 0] += 2.5
    return X, y


@pytest.fixture(scope="module")
def xy():
    return make_xy(15)


def _inner(m):
    """Wrapper ichidagi haqiqiy sklearn/xgboost ob'ekti (SVM uchun SVC)."""
    if m.name == "SVM":
        est = m.model_.calibrated_classifiers_[0].estimator if isinstance(m.model_, CalibratedClassifierCV) \
            else m.model_.pipeline
        return est.named_steps["svc"]
    return m.tree_model()


# ---------------------------------------------------------------------------
# Asosiy: fit / predict / shakl / dtype
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("calibrate", [True, False])
@pytest.mark.parametrize("name", SK_MODELS)
def test_fit_predict_basic(name, calibrate, xy):
    X, y = xy
    m = make_model(name, hp(name), calibrate=calibrate, n_features=X.shape[1])
    assert isinstance(m, ModelWrapper) and m.name == name and m.input_kind == "tabular"
    assert m.fit(X, y) is m and m.is_fitted
    p = m.predict_proba_pos(X)
    assert p.shape == (len(X),) and p.dtype == np.float64
    assert np.all((p >= 0) & (p <= 1)) and np.all(np.isfinite(p))
    assert roc_auc_score(y, p) > 0.9
    pp = m.predict_proba(X)
    assert pp.shape == (len(X), 2) and np.allclose(pp.sum(axis=1), 1.0)
    assert m.fit_info["n_train"] == len(y) and m.fit_info["n_pos"] == int(y.sum())


def test_fit_info_calibration_labels(xy):
    X, y = xy
    assert make_model("RandomForest", hp("RandomForest"), calibrate=False).fit(X, y).fit_info["calibration"] == "none"
    assert make_model("XGBoost", hp("XGBoost"), calibrate=True).fit(X, y).fit_info["calibration"] == "sigmoid"
    iso = make_model("RandomForest", hp("RandomForest"), calibrate=True, calibration_method="isotonic").fit(X, y)
    assert iso.fit_info["calibration"] == "isotonic" and iso.fit_info["calibration_cv"] == 3
    svm = make_model("SVM", hp("SVM"), calibrate=False).fit(X, y)       # SVM da kalibrlash baribir majburiy
    assert svm.fit_info["calibration"] == "svm_platt_only"
    assert isinstance(svm.model_, CalibratedClassifierCV)


def test_float32_input_and_batching(xy):
    X, y = xy
    m = make_model("RandomForest", hp("RandomForest")).fit(X.astype(np.float32), y.astype(np.int8))
    full = m.predict_proba_pos(X.astype(np.float32))
    assert np.array_equal(full, m.predict_proba_pos(X, batch_size=7))
    assert m.predict_proba_pos(X[:0]).shape == (0,)
    with pytest.raises(ValueError, match="batch_size"):
        m.predict_proba_pos(X, batch_size=0)


# ---------------------------------------------------------------------------
# BUG-06: SVM har doim Pipeline(StandardScaler, SVC) + CalibratedClassifierCV(ensemble=False)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("calibrate", [True, False])
def test_svm_structure_no_probability_flag(calibrate, xy):
    X, y = xy
    m = make_model("SVM", hp("SVM"), calibrate=calibrate).fit(X, y)
    cc = m.model_
    assert isinstance(cc, CalibratedClassifierCV) and cc.ensemble is False
    base = cc.calibrated_classifiers_[0].estimator
    assert [n for n, _ in base.steps] == ["scaler", "svc"]
    assert base.named_steps["svc"].get_params()["probability"] in (False, "deprecated")
    assert m.tree_model() is None and m.importance_mdi() is None


def test_svm_scaling_inside_pipeline(xy):
    """Chaqiruvchi xom feature beradi; katta masshtab SVM natijasini buzmasligi kerak."""
    X, y = xy
    scale = np.array([1.0, 1e3, 1e-3, 50.0, 1.0])
    a = make_model("SVM", hp("SVM")).fit(X, y).predict_proba_pos(X)
    b = make_model("SVM", hp("SVM")).fit(X * scale, y).predict_proba_pos(X * scale)
    assert np.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# BUG-12 / BUG-03: ensemble=False, tree_model() va importance_mdi()
# ---------------------------------------------------------------------------
def test_calibrated_base_fitted_on_full_train(xy):
    X, y = xy
    m = make_model("RandomForest", hp("RandomForest", bootstrap=False), calibrate=True).fit(X, y)
    assert len(m.model_.calibrated_classifiers_) == 1
    tm = m.tree_model()
    check_is_fitted(tm)
    # bootstrap=False => har daraxt ildizi butun train ma'lumotini ko'radi
    assert tm.estimators_[0].tree_.n_node_samples[0] == len(y)


@pytest.mark.parametrize("calibrate", [True, False])
@pytest.mark.parametrize("name", ["RandomForest", "XGBoost"])
def test_tree_model_and_importance(name, calibrate, xy):
    X, y = xy
    m = make_model(name, hp(name), calibrate=calibrate)
    assert m.tree_model() is None and m.importance_mdi() is None      # o'qitishdan oldin
    m.fit(X, y)
    tm = m.tree_model()
    check_is_fitted(tm)
    assert hasattr(tm, "feature_importances_")
    if calibrate:
        assert tm is m.model_.calibrated_classifiers_[0].estimator
    else:
        assert tm is m.model_
    imp = m.importance_mdi()
    assert imp.shape == (X.shape[1],) and imp.dtype == np.float64
    assert np.all(imp >= 0) and np.all(np.isfinite(imp))
    assert int(np.argmax(imp)) == 0          # signal 0-feature'da
    if name == "RandomForest":
        assert imp.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BUG-13: kam musbat nuqta - crash yo'q
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("calibrate", [True, False])
@pytest.mark.parametrize("name", SK_MODELS)
def test_three_positives_no_crash(name, calibrate):
    X, y = make_xy(3)
    m = make_model(name, hp(name), calibrate=calibrate).fit(X, y)
    p = m.predict_proba_pos(X)
    assert p.shape == (len(X),) and np.all((p >= 0) & (p <= 1))
    if calibrate or name == "SVM":
        assert m.fit_info["calibration_cv"] == 3
        assert m.fit_info["calibration"] in ("sigmoid", "svm_platt_only")


@pytest.mark.parametrize("name", SK_MODELS)
def test_two_positives_adaptive_cv(name):
    X, y = make_xy(2)
    m = make_model(name, hp(name), calibrate=True, calibration_cv=5).fit(X, y)
    assert m.fit_info["calibration"] == "sigmoid" and m.fit_info["calibration_cv"] == 2
    assert np.all(np.isfinite(m.predict_proba_pos(X)))


@pytest.mark.parametrize("calibrate", [True, False])
@pytest.mark.parametrize("name", SK_MODELS)
def test_single_positive_skips_calibration(name, calibrate):
    X, y = make_xy(1)
    logs = []
    m = make_model(name, hp(name), calibrate=calibrate).fit(X, y, log_fn=logs.append)
    p = m.predict_proba_pos(X)
    assert p.shape == (len(X),) and np.all((p >= 0) & (p <= 1))
    if calibrate or name == "SVM":
        assert m.fit_info["calibration"] == "skipped"
        assert "musbat=1" in m.fit_info["calibration_reason"]
        assert any("kalibrlanmadi" in s for s in logs)
    if name == "SVM":
        assert m.fit_info["svm_fallback"] == "sigmoid_minmax"
        assert not isinstance(m.model_, CalibratedClassifierCV)
        assert p[0] > np.median(p)                      # musbat nuqta yuqori indeks oladi
    else:
        assert not isinstance(m.model_, CalibratedClassifierCV) or not calibrate


def test_svm_fallback_is_monotonic_and_picklable(tmp_path):
    X, y = make_xy(1)
    m = make_model("SVM", hp("SVM")).fit(X, y)
    d = m.model_.pipeline.decision_function(X)
    p = m.predict_proba_pos(X)
    assert np.all(np.diff(p[np.argsort(d)]) >= -1e-12)         # tartib saqlanadi
    assert np.all((p > 0) & (p < 1))
    m.save(str(tmp_path / "svm1"))
    assert np.array_equal(type(m).load(str(tmp_path / "svm1")).predict_proba_pos(X), p)


def test_small_calibration_cv_skips(xy):
    X, y = xy
    m = make_model("RandomForest", hp("RandomForest"), calibrate=True, calibration_cv=1).fit(X, y)
    assert m.fit_info["calibration"] == "skipped" and "calibration_cv=1" in m.fit_info["calibration_reason"]


@pytest.mark.parametrize("name", SK_MODELS)
def test_calibration_failure_falls_back(name, xy, monkeypatch):
    X, y = xy

    def boom(self, est, eff_cv, method):
        raise RuntimeError("sinov xatosi")

    monkeypatch.setattr(SklearnModel, "_calibrator", boom)
    m = make_model(name, hp(name), calibrate=True).fit(X, y)
    assert m.fit_info["calibration"] == "skipped" and "sinov xatosi" in m.fit_info["calibration_reason"]
    assert np.all(np.isfinite(m.predict_proba_pos(X)))


# ---------------------------------------------------------------------------
# BUG-15 / ENH-01: PARAM_SPECS dagi har bir parametr modelga uzatiladi
# ---------------------------------------------------------------------------
ALT = {
    "RandomForest": {"n_estimators": 17, "criterion": "entropy", "max_depth": 6, "min_samples_split": 5,
                     "min_samples_leaf": 3, "max_features": "log2", "bootstrap": False, "max_samples": 0.5,
                     "class_weight": "none", "ccp_alpha": 0.01},
    "SVM": {"kernel": "poly", "C": 2.5, "gamma": 0.05, "degree": 2, "class_weight": "none"},
    "XGBoost": {"n_estimators": 21, "learning_rate": 0.2, "max_depth": 3, "min_child_weight": 2.5,
                "subsample": 0.6, "colsample_bytree": 0.7, "gamma": 0.3, "reg_alpha": 0.4, "reg_lambda": 2.0,
                "scale_pos_weight": 7.5, "tree_method": "exact"},
}


def _expected(name, pname, value):
    """PARAM_SPECS qiymati -> sklearn/xgboost get_params() dagi kutilgan qiymat."""
    if name == "RandomForest":
        if (pname, value) in (("max_features", "all"), ("class_weight", "none")):
            return None
    if name == "SVM":
        if (pname, value) in (("class_weight", "none"), ("gamma", None)):
            return None if pname == "class_weight" else "scale"
    return value


def _check_param(name, pname, value, xy):
    X, y = xy
    over = {pname: value}
    if name == "RandomForest" and pname == "max_samples":
        over["bootstrap"] = True
    m = make_model(name, hp(name, **over), calibrate=True, seed=7, n_jobs=2).fit(X, y)
    got = _inner(m).get_params()[pname]
    exp = _expected(name, pname, value)
    if isinstance(exp, float):
        assert got == pytest.approx(exp), f"{name}.{pname}: {got!r} != {exp!r}"
    else:
        assert got == exp, f"{name}.{pname}: {got!r} != {exp!r}"
    assert m.get_params()[pname] == value


@pytest.mark.parametrize("name", SK_MODELS)
def test_param_specs_fully_covered(name, xy):
    """Har bir PARAM_SPECS parametri boshqa qiymatga o'zgartirilganda estimator'da aks etadi."""
    spec_names = [s.name for s in PARAM_SPECS[name]]
    assert sorted(ALT[name]) == sorted(spec_names), "ALT yangi PARAM_SPECS parametrlarini qamramagan"
    defaults = default_hyperparams()[name]
    for pname in spec_names:
        assert ALT[name][pname] != defaults[pname], f"{name}.{pname}: ALT standartdan farq qilishi kerak"
        _check_param(name, pname, ALT[name][pname], xy)


@pytest.mark.parametrize("name", SK_MODELS)
def test_param_all_choices_and_none_values(name, xy):
    for spec in PARAM_SPECS[name]:
        if spec.kind == "choice":
            for c in spec.choices:
                _check_param(name, spec.name, c, xy)
        elif spec.kind in ("optint", "optfloat") and spec.name != "scale_pos_weight":
            _check_param(name, spec.name, None, xy)        # None -> avtomatik/cheksiz


def test_rf_max_samples_requires_bootstrap(xy):
    X, y = xy
    logs = []
    m = make_model("RandomForest", hp("RandomForest", bootstrap=False, max_samples=0.5)).fit(X, y, log_fn=logs.append)
    assert m.tree_model().get_params()["max_samples"] is None
    assert any("max_samples" in s for s in logs)
    m2 = make_model("RandomForest", hp("RandomForest", bootstrap=True, max_samples=0.5)).fit(X, y)
    assert m2.tree_model().get_params()["max_samples"] == 0.5


@pytest.mark.parametrize("calibrate", [True, False])
def test_xgb_scale_pos_weight(calibrate, xy):
    X, y = xy
    auto = make_model("XGBoost", hp("XGBoost"), calibrate=calibrate).fit(X, y)
    exp = (len(y) - y.sum()) / y.sum()
    assert auto.tree_model().get_params()["scale_pos_weight"] == pytest.approx(exp)
    assert auto.fit_info["scale_pos_weight"] == pytest.approx(exp)
    fixed = make_model("XGBoost", hp("XGBoost", scale_pos_weight=3.0), calibrate=calibrate).fit(X, y)
    assert fixed.tree_model().get_params()["scale_pos_weight"] == 3.0


@pytest.mark.parametrize("name", SK_MODELS)
def test_seed_and_n_jobs_passed(name, xy):
    X, y = xy
    m = make_model(name, hp(name), seed=123, n_jobs=3).fit(X, y)
    inner = _inner(m)
    if name != "SVM":
        assert inner.get_params()["random_state"] == 123 and inner.get_params()["n_jobs"] == 3
    assert m.seed == 123 and m.n_jobs == 3


def test_defaults_and_params_normalized():
    for name in SK_MODELS:
        m = make_model(name, {})
        assert m.get_params() == models_mod.normalize_params(name, default_hyperparams()[name])
        assert set(m.get_params()) == {s.name for s in PARAM_SPECS[name]}
    m = make_model("SVM", {"gamma": "none", "C": "2", "kernel": "rbf"})
    assert m.get_params()["gamma"] is None and m.get_params()["C"] == 2.0
    m = make_model("RandomForest", {"n_estimators": np.int64(12), "bootstrap": np.bool_(False)})
    assert m.get_params()["n_estimators"] == 12 and m.get_params()["bootstrap"] is False


def test_invalid_params_raise():
    with pytest.raises(ValueError, match="noma'lum parametr"):
        make_model("RandomForest", {"n_trees": 5})
    with pytest.raises(ValueError, match="ruxsat etilmagan"):
        make_model("SVM", {"kernel": "tanh"})
    with pytest.raises(ValueError, match="noto'g'ri qiymat"):
        make_model("XGBoost", {"max_depth": "chuqur"})
    with pytest.raises(ValueError, match="noto'g'ri qiymat"):
        make_model("XGBoost", {"learning_rate": float("nan")})
    with pytest.raises(ValueError, match="Kalibrlash usuli"):
        make_model("RandomForest", {}, calibration_method="platt")
    with pytest.raises(ValueError, match="Noma'lum model"):
        make_model("KNN", {})


# ---------------------------------------------------------------------------
# Determinizm, saqlash/yuklash
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("calibrate", [True, False])
@pytest.mark.parametrize("name", SK_MODELS)
def test_determinism(name, calibrate, xy):
    X, y = xy
    a = make_model(name, hp(name), calibrate=calibrate, seed=5, n_jobs=2).fit(X, y).predict_proba_pos(X)
    b = make_model(name, hp(name), calibrate=calibrate, seed=5, n_jobs=2).fit(X, y).predict_proba_pos(X)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", SK_MODELS)
def test_n_jobs_invariant(name, xy):
    """n_jobs o'zgarishi natijaga ta'sir qilmaydi (RF yig'indisi qat'iy tartibda)."""
    X, y = xy
    ps = [make_model(name, hp(name), seed=1, n_jobs=nj).fit(X, y).predict_proba_pos(X) for nj in (1, 2, 3)]
    assert np.array_equal(ps[0], ps[1]) and np.array_equal(ps[0], ps[2])


def test_seed_matters(xy):
    X, y = xy
    p1 = make_model("RandomForest", hp("RandomForest"), seed=1).fit(X, y).predict_proba_pos(X)
    p2 = make_model("RandomForest", hp("RandomForest"), seed=2).fit(X, y).predict_proba_pos(X)
    assert not np.array_equal(p1, p2)


def test_rf_proba_matches_sklearn(xy):
    """_DeterministicRF.predict_proba sklearn'ning o'z natijasi bilan (float aniqligida) mos."""
    X, y = xy
    tm = make_model("RandomForest", hp("RandomForest"), calibrate=False, n_jobs=2).fit(X, y).tree_model()
    assert isinstance(tm, RandomForestClassifier)
    assert np.allclose(tm.predict_proba(X), RandomForestClassifier.predict_proba(tm, X), atol=1e-12)
    assert np.array_equal(tm.predict(X), RandomForestClassifier.predict(tm, X))


@pytest.mark.parametrize("calibrate", [True, False])
@pytest.mark.parametrize("name", SK_MODELS)
def test_save_load_roundtrip(name, calibrate, xy, tmp_path):
    X, y = xy
    m = make_model(name, hp(name), calibrate=calibrate).fit(X, y)
    p = m.predict_proba_pos(X)
    path = m.save(str(tmp_path / name))
    assert path.endswith("model.joblib")
    m2 = type(m).load(str(tmp_path / name))
    assert isinstance(m2, SklearnModel) and m2.is_fitted and m2.name == name
    assert m2.fit_info == m.fit_info and m2.get_params() == m.get_params()
    assert np.array_equal(m2.predict_proba_pos(X), p)
    if name != "SVM":
        assert np.array_equal(m2.importance_mdi(), m.importance_mdi())


# ---------------------------------------------------------------------------
# Ogohlantirishlar (eskirgan API yo'q)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_pos", [15, 3])
@pytest.mark.parametrize("calibrate", [True, False])
@pytest.mark.parametrize("name", SK_MODELS)
def test_no_future_or_deprecation_warnings(name, calibrate, n_pos):
    X, y = make_xy(n_pos)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        m = make_model(name, hp(name), calibrate=calibrate, n_jobs=2).fit(X, y)
        m.predict_proba_pos(X, batch_size=40)
        m.predict_proba(X)
        m.importance_mdi()


# ---------------------------------------------------------------------------
# Kirishni tekshirish, o'qitishdan oldingi holat, bekor qilish
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nan_inf_rejected(bad, xy):
    X, y = xy
    m = make_model("RandomForest", hp("RandomForest"))
    Xb = X.copy()
    Xb[3, 2] = bad
    with pytest.raises(ValueError, match="NaN/inf"):
        m.fit(Xb, y)
    m.fit(X, y)
    with pytest.raises(ValueError, match="NaN/inf"):
        m.predict_proba_pos(Xb)


def test_shape_and_label_validation(xy):
    X, y = xy
    m = make_model("XGBoost", hp("XGBoost"), n_features=X.shape[1])
    with pytest.raises(ValueError, match="2 o'lchamli"):
        m.fit(X[:, 0], y)
    with pytest.raises(ValueError, match="feature'lar soni"):
        m.fit(X[:, :3], y)
    with pytest.raises(ValueError, match="y shakli"):
        m.fit(X, y[:-1])
    with pytest.raises(ValueError, match="0/1"):
        m.fit(X, y * 2)
    with pytest.raises(ValueError, match="ikkala sinf"):
        m.fit(X, np.zeros_like(y))
    with pytest.raises(ValueError, match="X berilmagan"):
        m.fit(None, y)
    m.fit(X, y)
    with pytest.raises(ValueError, match="feature'lar soni"):
        m.predict_proba_pos(X[:, :3])
    with pytest.raises(ValueError, match="X berilmagan"):
        m.predict_proba_pos(None)


def test_predict_before_fit_raises(xy):
    X, _ = xy
    with pytest.raises(RuntimeError, match="o'qitilmagan"):
        make_model("SVM", hp("SVM")).predict_proba_pos(X)


def test_cancel_token(xy):
    X, y = xy
    tok = CancelToken()
    tok.cancel()
    m = make_model("RandomForest", hp("RandomForest"))
    with pytest.raises(CancelledError):
        m.fit(X, y, cancel=tok)
    assert not m.is_fitted


def test_refit_resets_state(xy):
    X, y = xy
    m = make_model("RandomForest", hp("RandomForest")).fit(X, y)
    X2, y2 = make_xy(3, n=60, p=5, seed=4)
    m.fit(X2, y2)
    assert m.fit_info["n_train"] == 60 and m.fit_info["n_pos"] == 3


@pytest.mark.parametrize("name", SK_MODELS)
def test_refit_with_different_n_features(name, xy):
    """Qayta fit() boshqa feature soni bilan mumkin (eski n_features_ qayta fit'ni to'smasin);
    n_features konstruktorda e'lon qilingan bo'lsa fit uni baribir tekshiradi."""
    X, y = xy
    m = make_model(name, hp(name)).fit(X, y)
    m.fit(X[:, :3], y)
    assert m.predict_proba_pos(X[:, :3]).shape == (len(X),)
    with pytest.raises(ValueError, match="feature'lar soni"):
        m.predict_proba_pos(X)                         # endi 3 ta feature kutiladi
    with pytest.raises(ValueError, match="feature'lar soni"):
        make_model(name, hp(name), n_features=4).fit(X, y)


# ---------------------------------------------------------------------------
# Seed diapazoni, katta kirish (xotira), NaN tekshiruvi bo'laklari
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("calibrate", [True, False])
@pytest.mark.parametrize("name", SK_MODELS)
def test_seed_outside_uint32_range(name, calibrate, xy):
    """Manfiy / 2**32 dan katta seed sklearn random_state'ni yiqitmasin; seed % (2**32-1) bilan ishlaydi."""
    X, y = xy
    ref = make_model(name, hp(name), calibrate=calibrate, seed=1).fit(X, y).predict_proba_pos(X)
    for seed in (2 ** 32, 5 * (2 ** 32 - 1) + 1):        # ikkalasi ham 1 (mod 2**32-1)
        m = make_model(name, hp(name), calibrate=calibrate, seed=seed).fit(X, y)
        assert m.seed == seed
        assert np.array_equal(m.predict_proba_pos(X), ref)
    neg = make_model(name, hp(name), calibrate=calibrate, seed=-5).fit(X, y)
    assert neg.seed == -5 and np.all(np.isfinite(neg.predict_proba_pos(X)))
    if name != "SVM":
        assert _inner(neg).get_params()["random_state"] == (-5) % (2 ** 32 - 1)


@pytest.mark.parametrize("name", ["RandomForest", "XGBoost"])
def test_predict_large_float32_no_full_float64_copy(name, xy):
    """float32 kirish bo'lak-bo'lak float64 ga o'tadi: cho'qqi xotira kirish hajmidan oshmaydi."""
    import tracemalloc
    X, y = xy
    Xb = np.tile(X, (150_000 // len(X) + 1, 1))[:150_000].astype(np.float32)
    m = make_model(name, hp(name, n_estimators=10), calibrate=False).fit(X, y)
    m.predict_proba_pos(Xb[:1000])                       # isitish (lazy import/keshlar)
    tracemalloc.start()
    try:
        p = m.predict_proba_pos(Xb)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert p.shape == (len(Xb),)
    assert peak < Xb.nbytes, f"cho'qqi xotira {peak} B >= kirish hajmi {Xb.nbytes} B"


def test_nan_inf_counted_across_scan_chunks(xy):
    """Katta X da NaN/inf bo'laklar bo'yicha yig'iladi (xabarda JAMI son) va oxirgi qatordagisi ham topiladi."""
    X, y = xy
    m = make_model("XGBoost", hp("XGBoost", n_estimators=5), calibrate=False).fit(X, y)
    n = models_mod._SCAN_ROWS * 2 + 7
    Xb = np.zeros((n, X.shape[1]), dtype=np.float32)
    Xb[0, 0] = np.nan
    Xb[models_mod._SCAN_ROWS + 3, 1] = np.inf
    Xb[-1, 2] = -np.inf
    with pytest.raises(ValueError, match=r"3 ta NaN/inf"):
        m.predict_proba_pos(Xb)
    Xb[~np.isfinite(Xb)] = 0.0
    assert m.predict_proba_pos(Xb).shape == (n,)


def test_non_float_input_dtypes(xy):
    """int/bool/object kirishlar qabul qilinadi; raqamga aylanmaydigan object - aniq ValueError."""
    X, y = xy
    Xi = np.round(X * 10).astype(np.int32)
    m = make_model("RandomForest", hp("RandomForest")).fit(Xi, y)
    assert np.array_equal(m.predict_proba_pos(Xi), m.predict_proba_pos(Xi.astype(np.float64)))
    assert m.predict_proba_pos(Xi.astype(object)).shape == (len(X),)
    bad = Xi.astype(object)
    bad[0, 0] = "abc"
    with pytest.raises(ValueError, match="raqamli"):
        m.predict_proba_pos(bad)
    with pytest.raises(ValueError, match="raqamli"):
        m.fit(bad, y)


# ---------------------------------------------------------------------------
# Kutubxona mavjud emas / CNN fabrikasi
# ---------------------------------------------------------------------------
def test_xgboost_missing_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(models_mod, "get_xgboost", lambda: None)
    with pytest.raises(RuntimeError, match="XGBoost o'rnatilmagan"):
        make_model("XGBoost", {})
    make_model("RandomForest", {})          # boshqa modellarga ta'sir qilmaydi


def test_cnn_missing_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(models_mod, "tf_available", lambda: False)
    with pytest.raises(RuntimeError, match="TensorFlow"):
        make_model("CNN", {})
    monkeypatch.setattr(models_mod, "tf_available", lambda: True)
    monkeypatch.setitem(sys.modules, "mpm.cnn", None)       # import ImportError beradi
    with pytest.raises(RuntimeError, match="mpm.cnn"):
        make_model("CNN", {})


class _CNNStub(ModelWrapper):
    """CNNModel o'rnini bosuvchi qo'g'irchoq: faqat konstruktor imzosi muhim."""
    input_kind = "patch"

    def fit(self, X, y, patches=None, cancel=None, log_fn=None):
        return self

    def predict_proba_pos(self, X=None, patches=None, batch_size=8192):
        return np.zeros(0)


class _CNNByParams(_CNNStub):
    def __init__(self, params, seed=0, n_jobs=1, n_features=None):
        super().__init__("CNN", params, seed=seed, n_jobs=n_jobs)
        self.n_features = n_features


class _CNNByName(_CNNStub):
    def __init__(self, name, params, **kw):
        super().__init__(name, params, seed=kw.get("seed", 0), n_jobs=kw.get("n_jobs", 1))
        self.n_features = kw.get("n_features")


@pytest.mark.parametrize("stub", [_CNNByParams, _CNNByName])
def test_make_model_cnn_signature_adaptation(stub, monkeypatch):
    """make_model CNNModel.__init__ imzosiga moslashadi: (params, ...) yoki (name, params, **kw)."""
    import types
    fake = types.ModuleType("mpm.cnn")
    fake.CNNModel = stub
    monkeypatch.setattr(models_mod, "tf_available", lambda: True)
    monkeypatch.setitem(sys.modules, "mpm.cnn", fake)
    m = make_model("CNN", {"mode": "patch2d"}, seed=9, n_jobs=2, n_features=4)
    assert isinstance(m, stub) and m.name == "CNN" and m.seed == 9 and m.n_jobs == 2 and m.n_features == 4
    assert m.get_params() == {"mode": "patch2d"}


@pytest.mark.slow
def test_make_model_cnn_lazy_import():
    cnn = pytest.importorskip("mpm.cnn")        # avval: cnn.py yo'q bo'lsa TF import qilinmaydi (sekin)
    pytest.importorskip("tensorflow")
    for mode, kind in (("patch2d", "patch"), ("tabular1d", "tabular")):
        params = dict(default_hyperparams()["CNN"], mode=mode)
        m = make_model("CNN", params, seed=3, n_jobs=1, n_features=5)
        assert isinstance(m, cnn.CNNModel) and isinstance(m, ModelWrapper)
        assert m.name == "CNN" and m.input_kind == kind


def test_model_names_dispatch():
    for name in MODEL_NAMES:
        if name in SK_MODELS:
            assert isinstance(make_model(name, {}), SklearnModel)
