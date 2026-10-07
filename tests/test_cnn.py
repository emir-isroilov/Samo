# -*- coding: utf-8 -*-
"""mpm.cnn testlari. TensorFlow kerak bo'lganlari @pytest.mark.slow (kichik tarmoq, epochs <= 20).
Har bir fit() ~2 s sobit xarajat (Keras graf izlash) talab qiladi, shuning uchun fit'lar soni kam tutilgan."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mpm import cnn as C
from mpm.cnn import CNNModel
from mpm.common import CancelledError, CancelToken
from mpm.config import PARAM_SPECS, default_hyperparams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# kichik tarmoq: har bir test tez ishlashi uchun
SMALL = dict(default_hyperparams()["CNN"], window=5, filters1=4, filters2=8, kernel_size=3, dense_units=8,
             dropout=0.3, batch_size=16, epochs=6, patience=3, val_fraction=0.2, learning_rate=0.003)


def small(**over):
    return dict(SMALL, **over)


def make_patch_data(n=60, n_pos=12, w=5, p=3, seed=0):
    """(patches, y): musbatlarda markaz atrofida 0-kanalda kuchli signal; ba'zi patchlarda NaN (chegara/nodata)."""
    rng = np.random.default_rng(seed)
    patches = rng.normal(size=(n, w, w, p)).astype(np.float32)
    y = np.zeros(n, dtype=np.int8)
    y[:n_pos] = 1
    c = w // 2
    patches[:n_pos, c - 1:c + 2, c - 1:c + 2, 0] += 2.0
    patches[5, 0, :, :] = np.nan
    patches[n - 1, :, 0, 1] = np.nan
    perm = rng.permutation(n)
    return patches[perm], y[perm]


def center_features(patches):
    c = patches.shape[1] // 2
    return np.nan_to_num(patches[:, c, c, :])


@pytest.fixture(scope="module")
def data():
    return make_patch_data()


@pytest.fixture(scope="module")
def tf_keras():
    tf = pytest.importorskip("tensorflow")
    return tf.keras


@pytest.fixture(scope="module")
def fitted_patch(data, tf_keras):
    """Signalni o'rganadigan patch2d modeli (bir marta o'qitiladi, ko'p testda ishlatiladi)."""
    patches, y = data
    return CNNModel(small(epochs=20, patience=20, learning_rate=0.01, filters1=8), seed=7).fit(None, y, patches=patches)


@pytest.fixture(scope="module")
def fitted_tab(data, tf_keras):
    patches, y = data
    X = center_features(patches)
    X[2, 1] = np.nan                                         # NaN -> 0 (standartlashdan keyin)
    return CNNModel(small(mode="tabular1d", epochs=3), seed=3).fit(X, y), X


# ---------------------------------------------------------------------------
# Tez testlar (TensorFlow'siz)
# ---------------------------------------------------------------------------
def test_tensorflow_not_imported_at_module_level():
    code = ("import sys; import mpm.cnn as c; m = c.CNNModel({}); "
            "assert m.input_kind == 'patch'; assert 'tensorflow' not in sys.modules")
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_constructor_and_input_kind():
    m = CNNModel(default_hyperparams()["CNN"])
    assert m.name == "CNN" and m.input_kind == "patch" and m.seed == 42 and m.n_jobs == 1
    assert not m.is_fitted and m.get_params() == default_hyperparams()["CNN"]
    t = CNNModel({"mode": "tabular1d"}, seed=5, n_jobs=3)
    assert t.input_kind == "tabular" and t.seed == 5 and t.n_jobs == 3
    assert t.params["filters1"] == 16                       # yetishmaganlar standart bilan to'ldirilgan


@pytest.mark.parametrize("bad, msg", [
    ({"nomalum": 1}, "noma'lum parametr"),
    ({"mode": "rnn"}, "ruxsat etilmagan"),
    ({"optimizer": "lbfgs"}, "ruxsat etilmagan"),
    ({"window": 8}, "toq"),
    ({"filters1": 0}, "filters1"),
    ({"filters2": -1}, "filters2"),
    ({"batch_size": 0}, "batch_size"),
    ({"epochs": "ko'p"}, "epochs"),
    ({"dropout": 1.0}, "dropout"),
    ({"l2": -0.1}, "l2"),
    ({"learning_rate": 0.0}, "learning_rate"),
    ({"val_fraction": 1.0}, "val_fraction"),
    ({"augment": "ha"}, "augment"),
])
def test_invalid_params_raise(bad, msg):
    with pytest.raises(ValueError, match=msg):
        CNNModel(bad)


def test_even_window_allowed_in_tabular_mode():
    assert CNNModel({"mode": "tabular1d", "window": 8}).params["window"] == 8


def test_channel_stats_nan_aware():
    rng = np.random.default_rng(1)
    a = rng.normal(5.0, 2.0, size=(30, 5, 5, 4)).astype(np.float32)
    a[0, 0, 0, 0] = np.nan
    a[..., 1] = 3.0                                   # o'zgarmas kanal -> std = 1
    a[..., 2] = np.nan                                # butunlay NaN kanal -> mean 0, std 1
    a[3, 1, 1, 3] = np.inf
    mean, std = C._channel_stats(a)
    assert mean.dtype == np.float32 and std.dtype == np.float32
    ok = np.isfinite(a[..., 0])
    assert mean[0] == pytest.approx(a[..., 0][ok].mean(), rel=1e-5)
    assert std[0] == pytest.approx(a[..., 0][ok].std(), rel=1e-5)
    assert (mean[1], std[1]) == (pytest.approx(3.0), 1.0)
    assert (mean[2], std[2]) == (0.0, 1.0)
    ok3 = np.isfinite(a[..., 3])
    assert mean[3] == pytest.approx(a[..., 3][ok3].mean(), rel=1e-5)
    t = rng.normal(size=(40, 6)).astype(np.float32)
    m2, s2 = C._channel_stats(t)                      # tabular: (n, p)
    np.testing.assert_allclose(m2, t.mean(axis=0), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(s2, t.std(axis=0), rtol=1e-4)


def test_channel_stats_small_units_are_standardized_not_treated_as_constant():
    """Regressiya: std < 1e-8 mutlaq chegarasi Tesla / 1/s^2 kabi kichik birlikdagi qatlamlarni yo'q qilar edi."""
    rng = np.random.default_rng(5)
    base = rng.normal(size=(200, 5, 5, 2)).astype(np.float32)
    ref_mean, ref_std = C._channel_stats(base)
    for scale in (1e-9, 1e-12, 1e-20):
        a = base * np.float32(scale)
        mean, std = C._channel_stats(a)
        np.testing.assert_allclose(std, ref_std * scale, rtol=1e-4)
        np.testing.assert_allclose(mean, ref_mean * scale, rtol=1e-4, atol=1e-6 * scale)
        z = C._standardize(a, mean, std)
        np.testing.assert_allclose(z.reshape(-1, 2).std(axis=0), 1.0, rtol=1e-3)
        np.testing.assert_allclose(z.reshape(-1, 2).mean(axis=0), 0.0, atol=1e-3)
    # haqiqatan o'zgarmas kanallar (shu jumladan float32 yaxlitlash shovqini va juda kichik/katta qiymat) std=1
    n = 3000
    const = np.stack([np.full(n, 0.1, np.float32), np.full(n, 1e-12, np.float32), np.full(n, 4.6e6, np.float32),
                      np.zeros(n, np.float32)], axis=1)
    mean, std = C._channel_stats(const)
    np.testing.assert_array_equal(std, 1.0)
    np.testing.assert_allclose(C._standardize(const, mean, std), 0.0, atol=1e-6)


def test_standardize_sets_nonfinite_to_zero_after_scaling():
    a = np.array([[[[1.0, np.nan], [3.0, 2.0]]]], dtype=np.float32)       # (1, 1, 2, 2)
    out = C._standardize(a, np.array([2.0, 2.0], np.float32), np.array([2.0, 1.0], np.float32))
    np.testing.assert_allclose(out[0, 0], [[-0.5, 0.0], [0.5, 0.0]])
    assert out.dtype == np.float32 and np.isfinite(out).all()


def _dihedral(x):
    """(w, w, p) patchning 8 ta diedral almashtirishi."""
    outs = []
    for k in range(4):
        r = np.rot90(x, k, axes=(0, 1))
        outs += [r, r[:, ::-1]]
    return outs


def test_augment_batch_is_dihedral_and_deterministic():
    rng = np.random.default_rng(0)
    xb = rng.normal(size=(64, 5, 5, 3)).astype(np.float32)
    out = C._augment_batch(xb, np.random.default_rng(3))
    assert out.shape == xb.shape and out.dtype == xb.dtype
    changed = 0
    for a, b in zip(xb, out):
        assert any(np.array_equal(b, d) for d in _dihedral(a))                  # faqat flip/rot90
        np.testing.assert_array_equal(a[2, 2], b[2, 2])                          # markaziy piksel joyida
        changed += not np.array_equal(a, b)
    assert changed > 32                                                          # ko'pchilik o'zgargan
    np.testing.assert_array_equal(out, C._augment_batch(xb, np.random.default_rng(3)))
    assert not np.array_equal(out, C._augment_batch(xb, np.random.default_rng(4)))


def test_batch_source_epoch_shuffle_augment_deterministic():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 5, 5, 2)).astype(np.float32)
    y = (np.arange(50) % 2).astype(np.float32)
    sw = np.ones(50, np.float32)
    src = C._BatchSource(X, y, sw, 8, seed=11, augment=True)
    assert len(src) == 7
    e0 = [src.batch(0, i) for i in range(len(src))]
    e1 = [src.batch(1, i) for i in range(len(src))]
    assert e0[-1][0].shape[0] == 2                                              # oxirgi batch kichik
    assert sorted(np.concatenate([b[1] for b in e0]).tolist()) == sorted(y.tolist())   # har namuna bir marta
    assert not np.array_equal(np.concatenate([b[0] for b in e0]), np.concatenate([b[0] for b in e1]))
    # qayta tuzilish va chaqiruv tartibi natijaga ta'sir qilmaydi
    again = C._BatchSource(X, y, sw, 8, seed=11, augment=True)
    np.testing.assert_array_equal(again.batch(0, 3)[0], e0[3][0])
    np.testing.assert_array_equal(again.batch(1, 0)[0], e1[0][0])
    np.testing.assert_array_equal(again.batch(0, 0)[0], e0[0][0])
    plain = C._BatchSource(X, y, sw, 8, seed=11, augment=False)
    sel = np.random.default_rng([11, 0]).permutation(50)[:8]
    np.testing.assert_array_equal(plain.batch(0, 0)[0], X[sel])                 # augmentatsiyasiz = oddiy aralashtirish


def test_split_is_stratified_and_deterministic():
    y = np.zeros(50, dtype=np.int64)
    y[:10] = 1
    logs = []
    tr, va = CNNModel(small(), seed=3)._split(y, logs.append)
    assert logs == []
    assert len(va) == 10 and len(tr) == 40 and not set(tr) & set(va)
    assert sorted(np.concatenate([tr, va]).tolist()) == list(range(50))
    assert y[va].sum() == 2 and y[tr].sum() == 8                                # sinf nisbati saqlangan
    _, va2 = CNNModel(small(), seed=3)._split(y, logs.append)
    np.testing.assert_array_equal(va, va2)
    assert not np.array_equal(va, CNNModel(small(), seed=4)._split(y, logs.append)[1])


@pytest.mark.parametrize("n_pos", [1, 0])
def test_split_fallback_warns_when_class_too_small(n_pos):
    y = np.zeros(40, dtype=np.int64)
    y[:n_pos] = 1
    logs = []
    tr, va = CNNModel(small(), seed=1)._split(y, logs.append)
    assert va is None and len(tr) == 40
    assert len(logs) == 1 and "OGOHLANTIRISH" in logs[0] and "train loss" in logs[0]


def test_param_coverage_table_is_complete():
    assert BUILD_PARAMS | FIT_PARAMS == {s.name for s in PARAM_SPECS["CNN"]}


# ---------------------------------------------------------------------------
# patch2d (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_patch2d_fit_predict_shape(fitted_patch, data):
    patches, y = data
    m = fitted_patch
    assert m.is_fitted and m.input_kind == "patch" and m.name == "CNN"
    p = m.predict_proba_pos(patches=patches)
    assert p.shape == (len(y),) and p.dtype == np.float64
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    assert roc_auc_score(y, p) > 0.8                                         # signalni o'rgangan
    pp = m.predict_proba(patches=patches)                                    # ModelWrapper.predict_proba
    assert pp.shape == (len(y), 2)
    np.testing.assert_allclose(pp[:, 1], p, atol=1e-6)
    np.testing.assert_allclose(pp.sum(axis=1), 1.0)
    assert m.predict_proba_pos(patches=patches[:0]).shape == (0,)
    assert m.predict_proba_pos(patches=patches[:1]).shape == (1,)
    assert m.fit_info["early_stopping"] == "val_loss"


@pytest.mark.slow
def test_standardization_stats_come_from_fit_data_only(fitted_patch, data):
    patches, _ = data
    ref_mean = np.array([np.nanmean(patches[..., c]) for c in range(3)])
    ref_std = np.array([np.nanstd(patches[..., c]) for c in range(3)])
    np.testing.assert_allclose(fitted_patch.mean_, ref_mean, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(fitted_patch.std_, ref_std, rtol=1e-4)
    fitted_patch.predict_proba_pos(patches=patches * 100 + 50)               # boshqa taqsimot: statistika o'zgarmaydi
    np.testing.assert_allclose(fitted_patch.mean_, ref_mean, rtol=1e-4, atol=1e-5)


@pytest.mark.slow
def test_nan_patches_give_finite_predictions(fitted_patch, data):
    patches, _ = data
    assert np.isnan(patches).any()
    p = fitted_patch.predict_proba_pos(patches=np.full_like(patches[:3], np.nan))
    assert np.isfinite(p).all() and np.ptp(p) < 1e-6                         # NaN -> 0 (standartlashdan keyin)


@pytest.mark.slow
def test_early_stopping_val_loss_and_deterministic(data, tf_keras, monkeypatch):
    patches, y = data
    spied = []
    orig = tf_keras.callbacks.EarlyStopping

    class Spy(orig):
        def __init__(self, *a, **k):
            spied.append(k)
            super().__init__(*a, **k)

    monkeypatch.setattr(tf_keras.callbacks, "EarlyStopping", Spy)
    params = small(epochs=40, patience=2, learning_rate=0.05)
    logs = []
    m1 = CNNModel(params, seed=3).fit(None, y, patches=patches, log_fn=logs.append)
    assert spied == [dict(monitor="val_loss", patience=2, restore_best_weights=True)]
    info = m1.fit_info
    assert set(info) >= {"epochs_run", "best_epoch", "val_loss", "n_train", "n_val", "early_stopping"}
    assert info["early_stopping"] == "val_loss"
    assert info["n_val"] == math.ceil(len(y) * 0.2) and info["n_train"] + info["n_val"] == len(y)
    assert 1 <= info["best_epoch"] <= info["epochs_run"] <= 40
    assert info["epochs_run"] - info["best_epoch"] <= 2                      # patience dan ortiq kutilmagan
    assert info["epochs_run"] < 40                                           # early stopping ishladi
    assert math.isfinite(info["val_loss"]) and info["class_weight_pos"] > 1
    assert logs and not any("OGOHLANTIRISH" in s for s in logs)             # faqat ma'lumot logi
    p1 = m1.predict_proba_pos(patches=patches)
    m2 = CNNModel(params, seed=3).fit(None, y, patches=patches)
    np.testing.assert_allclose(m2.predict_proba_pos(patches=patches), p1, atol=1e-6)    # bir xil seed
    same = {k: v for k, v in m2.fit_info.items() if k not in ("val_loss", "train_loss")}
    assert same == {k: v for k, v in info.items() if k not in ("val_loss", "train_loss")}
    assert m2.fit_info["val_loss"] == pytest.approx(info["val_loss"], rel=1e-5)


@pytest.mark.slow
def test_few_positives_no_crash(tf_keras):
    patches, y = make_patch_data(n=44, n_pos=4, seed=9)
    m = CNNModel(small(epochs=2), seed=1).fit(None, y, patches=patches)
    p = m.predict_proba_pos(patches=patches)
    assert p.shape == (44,) and np.isfinite(p).all()
    assert m.fit_info["n_train"] + m.fit_info["n_val"] == 44
    assert m.fit_info["early_stopping"] == "val_loss" and m.fit_info["n_val"] >= 2


@pytest.mark.slow
def test_single_positive_falls_back_to_train_loss(tf_keras, monkeypatch):
    patches, y = make_patch_data(n=40, n_pos=1, seed=4)
    spied = []
    orig = tf_keras.callbacks.EarlyStopping

    class Spy(orig):
        def __init__(self, *a, **k):
            spied.append(k)
            super().__init__(*a, **k)

    monkeypatch.setattr(tf_keras.callbacks, "EarlyStopping", Spy)
    logs = []
    m = CNNModel(small(epochs=2), seed=1).fit(None, y, patches=patches, log_fn=logs.append)
    assert spied[0]["monitor"] == "loss" and spied[0]["restore_best_weights"] is True
    assert m.fit_info["early_stopping"] == "train_loss"
    assert m.fit_info["n_val"] == 0 and m.fit_info["n_train"] == 40 and m.fit_info["val_loss"] is None
    assert any("OGOHLANTIRISH" in s and "validatsiya" in s for s in logs)
    assert np.isfinite(m.predict_proba_pos(patches=patches)).all()


@pytest.mark.slow
def test_input_validation_errors(fitted_patch, data, tf_keras):
    patches, y = data
    m = CNNModel(small(), seed=1)
    with pytest.raises(RuntimeError, match="(?i)o'qitilmagan"):
        m.predict_proba_pos(patches=patches)
    with pytest.raises(RuntimeError, match="(?i)o'qitilmagan"):
        m.save("unused")
    with pytest.raises(ValueError, match="patches"):
        m.fit(patches[:, 2, 2, :], y)                                          # patches=None
    wide = make_patch_data(w=7)[0]
    with pytest.raises(ValueError, match="window"):
        m.fit(None, y, patches=wide)                                          # 7x7 != window=5
    with pytest.raises(ValueError, match="window"):
        fitted_patch.predict_proba_pos(patches=wide)
    with pytest.raises(ValueError, match="4 o'lchamli"):
        m.fit(None, y, patches=patches[..., 0])
    with pytest.raises(ValueError, match="y uzunligiga"):
        m.fit(None, y[:-1], patches=patches)
    with pytest.raises(ValueError, match="0/1"):
        m.fit(None, y + 1, patches=patches)
    with pytest.raises(ValueError, match="ikkala sinf"):
        m.fit(None, np.zeros(len(y)), patches=patches)
    with pytest.raises(ValueError, match="kanallari"):
        fitted_patch.predict_proba_pos(patches=patches[..., :2])
    with pytest.raises(ValueError, match="batch_size"):
        fitted_patch.predict_proba_pos(patches=patches, batch_size=0)
    assert not m.is_fitted


@pytest.mark.slow
def test_predict_batch_size_does_not_change_result(fitted_patch, data):
    patches, _ = data
    ref = fitted_patch.predict_proba_pos(patches=patches)
    for bs in (1, 7, 32, 8192):
        np.testing.assert_allclose(fitted_patch.predict_proba_pos(patches=patches, batch_size=bs), ref, atol=1e-5)
    old = C._PREDICT_ELEMS
    try:                                                   # kichik chunk (ko'p chunk) bo'lganda ham natija bir xil
        C._PREDICT_ELEMS = 5 * 5 * 3 * 4
        np.testing.assert_allclose(fitted_patch.predict_proba_pos(patches=patches, batch_size=3), ref, atol=1e-5)
    finally:
        C._PREDICT_ELEMS = old


@pytest.mark.slow
def test_fit_info_val_loss_matches_restored_best_weights(fitted_patch, data):
    """restore_best_weights max-epochga yetganda ham ishlaydi: fit_info['val_loss'] qaytarilgan model vazni bilan mos."""
    patches, y = data
    info = fitted_patch.fit_info
    assert info["best_epoch"] <= info["epochs_run"]
    _, va = fitted_patch._split(y.astype(np.int64), lambda msg: None)
    p = np.clip(fitted_patch.predict_proba_pos(patches=patches)[va], 1e-7, 1 - 1e-7)
    yv = y[va].astype(float)
    w = np.where(yv == 1, info["class_weight_pos"], 1.0)
    loss = -(yv * np.log(p) + (1 - yv) * np.log(1 - p))
    assert float((w * loss).sum() / len(va)) == pytest.approx(info["val_loss"], rel=1e-3)


@pytest.mark.slow
def test_concurrent_fits_do_not_crash_and_stay_deterministic(data, tf_keras):
    """Regressiya: parallel oqimlarda (joblib threading) clear_session/seed/qurish qulfsiz bo'lsa qatlam nomlari
    to'qnashib ValueError chiqadi va natija seed'ga bog'liq bo'lmay qoladi."""
    patches, y = data
    params = small(epochs=2)
    seeds = (1, 2)
    ref = {s: CNNModel(params, seed=s).fit(None, y, patches=patches).predict_proba_pos(patches=patches)
           for s in seeds}
    out, errors = {}, []

    def work(s):
        try:
            out[s] = CNNModel(params, seed=s).fit(None, y, patches=patches).predict_proba_pos(patches=patches)
        except Exception as e:                      # noqa: BLE001 - oqim xatosini asosiy oqimga uzatamiz
            errors.append(repr(e))

    threads = [threading.Thread(target=work, args=(s,)) for s in seeds]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for s in seeds:
        np.testing.assert_allclose(out[s], ref[s], atol=1e-5)


@pytest.mark.slow
def test_next_fit_clear_session_keeps_earlier_models_usable(fitted_patch, data, tf_keras):
    patches, y = data
    before = fitted_patch.predict_proba_pos(patches=patches)
    CNNModel(small(epochs=1, filters1=6), seed=2).fit(None, y, patches=patches)         # fit clear_session chaqiradi
    tf_keras.backend.clear_session()
    np.testing.assert_allclose(fitted_patch.predict_proba_pos(patches=patches), before, atol=1e-6)


# ---------------------------------------------------------------------------
# tabular1d (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_tabular1d_fit_predict(fitted_tab, data):
    m, X = fitted_tab
    patches, y = data
    assert m.input_kind == "tabular" and m.is_fitted and m.n_features_ == 3
    assert m.model_.input_shape == (None, 3, 1)
    p = m.predict_proba_pos(X)
    assert p.shape == (len(y),) and p.dtype == np.float64 and np.isfinite(p).all()
    np.testing.assert_allclose(m.predict_proba_pos(X, batch_size=5), p, atol=1e-5)
    np.testing.assert_allclose(m.mean_, np.nanmean(X, axis=0), rtol=1e-4, atol=1e-6)
    assert m.fit_info["early_stopping"] == "val_loss" and m.fit_info["n_val"] == math.ceil(len(y) * 0.2)
    with pytest.raises(ValueError, match="X"):
        m.fit(None, y, patches=patches)
    with pytest.raises(ValueError, match="2 o'lchamli"):
        m.fit(patches, y)
    with pytest.raises(ValueError, match="kanallari"):
        m.predict_proba_pos(X[:, :2])


# ---------------------------------------------------------------------------
# save / load (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("which", ["patch2d", "tabular1d"])
def test_save_load_roundtrip(which, fitted_patch, fitted_tab, data, tmp_path):
    patches, _ = data
    if which == "patch2d":
        m, kw = fitted_patch, {"patches": patches}
    else:
        m, kw = fitted_tab[0], {"X": fitted_tab[1]}
    ref = m.predict_proba_pos(**kw)
    out = m.save(str(tmp_path / "cnn"))
    assert os.path.basename(out) == "model.keras" and os.path.isfile(out)
    assert sorted(os.listdir(tmp_path / "cnn")) == ["meta.json", "model.keras"]
    meta = json.loads((tmp_path / "cnn" / "meta.json").read_text(encoding="utf-8"))
    assert meta["mode"] == which and meta["seed"] == m.seed and meta["n_features"] == 3
    assert meta["params"] == m.params and len(meta["mean"]) == 3 and len(meta["std"]) == 3
    np.testing.assert_allclose(meta["mean"], m.mean_, rtol=1e-6)
    loaded = CNNModel.load(str(tmp_path / "cnn"))
    assert isinstance(loaded, CNNModel) and loaded.is_fitted and loaded.input_kind == m.input_kind
    assert loaded.params == m.params and loaded.seed == m.seed and loaded.fit_info == m.fit_info
    np.testing.assert_allclose(loaded.predict_proba_pos(**kw), ref, atol=1e-6)
    np.testing.assert_array_equal(loaded.mean_, m.mean_)
    np.testing.assert_array_equal(loaded.std_, m.std_)
    with pytest.raises(FileNotFoundError):
        CNNModel.load(str(tmp_path / "yoq"))
    (tmp_path / "cnn" / "meta.json").write_text(json.dumps(dict(meta, format=99)), encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        CNNModel.load(str(tmp_path / "cnn"))


# ---------------------------------------------------------------------------
# cancel (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_cancel_stops_training_and_raises(data, tf_keras, monkeypatch):
    patches, y = data
    token = CancelToken()
    calls = []
    orig = C._BatchSource.batch

    def spy(self, epoch, i):
        calls.append((epoch, i))
        if len(calls) == 6:
            token.cancel()
        return orig(self, epoch, i)

    monkeypatch.setattr(C._BatchSource, "batch", spy)
    m = CNNModel(small(epochs=60, patience=60), seed=1)
    with pytest.raises(CancelledError):
        m.fit(None, y, patches=patches, cancel=token)
    assert len(calls) < 40                       # to'liq o'qitishda 60 epoch x 3 batch = 180 chaqiruv
    assert not m.is_fitted and m.model_ is None


@pytest.mark.slow
def test_precancelled_token_raises_before_training(data, tf_keras, monkeypatch):
    patches, y = data
    token = CancelToken()
    token.cancel()
    calls = []
    monkeypatch.setattr(C._BatchSource, "batch", lambda self, e, i: calls.append(1))
    with pytest.raises(CancelledError):
        CNNModel(small(), seed=1).fit(None, y, patches=patches, cancel=token)
    assert calls == []


# ---------------------------------------------------------------------------
# ENH-01: PARAM_SPECS["CNN"] dagi har bir parametr haqiqatan ishlatiladi
#   * arxitektura/optimizator parametrlari: build_keras_model orqali alohida-alohida (tez) VA fit() ichida;
#   * o'qitish parametrlari (batch_size, epochs, patience, val_fraction, augment): fit() ichida spion bilan.
# ---------------------------------------------------------------------------
def _layers(model, *names):
    return [lay for lay in model.layers if lay.__class__.__name__ in names]


def _check_mode(model):
    assert model.input_shape == (None, 3, 1)
    assert _layers(model, "Conv1D") and not _layers(model, "Conv2D")


def _check_window(model):
    assert model.input_shape == (None, 7, 7, 3)


def _check_filters1(model):
    assert _layers(model, "Conv2D")[0].filters == 6


def _check_filters2_off(model):
    assert len(_layers(model, "Conv2D")) == 1 and len(model.layers) == 5


def _check_filters2_on(model):
    assert [c.filters for c in _layers(model, "Conv2D")] == [4, 12]


def _check_kernel(model):
    assert [c.kernel_size for c in _layers(model, "Conv2D")][0] == (5, 5)


def _check_dense(model):
    assert _layers(model, "Dense")[0].units == 12


def _check_dropout(model):
    assert _layers(model, "Dropout")[0].rate == pytest.approx(0.55)


def _check_l2_on(model):
    regs = [lay.kernel_regularizer for lay in _layers(model, "Conv2D", "Dense")[:-1]]    # oxirgisi: chiqish qatlami
    assert regs and all(r is not None and float(r.l2) == pytest.approx(0.02) for r in regs)


def _check_l2_off(model):
    assert all(lay.kernel_regularizer is None for lay in _layers(model, "Conv2D", "Dense"))


def _check_sgd(model):
    assert model.optimizer.__class__.__name__ == "SGD"


def _check_rmsprop(model):
    assert model.optimizer.__class__.__name__ == "RMSprop"


def _check_adam(model):
    assert model.optimizer.__class__.__name__ == "Adam"


def _check_lr(model):
    assert float(np.asarray(model.optimizer.learning_rate)) == pytest.approx(0.0123, rel=1e-4)


# (parametr, qiymat, tekshiruv(model))
BUILD_CASES = [
    ("mode", "tabular1d", _check_mode), ("window", 7, _check_window), ("filters1", 6, _check_filters1),
    ("filters2", 0, _check_filters2_off), ("filters2", 12, _check_filters2_on), ("kernel_size", 5, _check_kernel),
    ("dense_units", 12, _check_dense), ("dropout", 0.55, _check_dropout), ("l2", 0.02, _check_l2_on),
    ("l2", 0.0, _check_l2_off), ("optimizer", "sgd", _check_sgd), ("optimizer", "rmsprop", _check_rmsprop),
    ("optimizer", "adam", _check_adam), ("learning_rate", 0.0123, _check_lr),
]
BUILD_PARAMS = {name for name, _, _ in BUILD_CASES}
FIT_PARAMS = {"batch_size", "epochs", "patience", "val_fraction", "augment"}


@pytest.mark.slow
@pytest.mark.parametrize("name, value, check", BUILD_CASES, ids=[f"{n}={v}" for n, v, _ in BUILD_CASES])
def test_architecture_param_reaches_keras_model(name, value, check, tf_keras):
    model = C.build_keras_model(small(**{name: value}), n_channels=3)
    check(model)


@pytest.fixture
def spied_fit(tf_keras, monkeypatch):
    """run(params, window) -> kontekst: wrapper, _BatchSource nusxalari, EarlyStopping kwargs, augment chaqiruvlari."""
    state = {"sources": [], "es": [], "aug": 0}

    class Rec(C._BatchSource):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            state["sources"].append(self)

    orig_es = tf_keras.callbacks.EarlyStopping

    class SpyES(orig_es):
        def __init__(self, *a, **k):
            state["es"].append(k)
            super().__init__(*a, **k)

    orig_aug = C._augment_batch

    def count_aug(xb, rng):
        state["aug"] += 1
        return orig_aug(xb, rng)

    monkeypatch.setattr(C, "_BatchSource", Rec)
    monkeypatch.setattr(C, "_augment_batch", count_aug)
    monkeypatch.setattr(tf_keras.callbacks, "EarlyStopping", SpyES)

    def run(params, window=5):
        patches, y = make_patch_data(n=60, n_pos=12, w=window, seed=1)
        m = CNNModel(params, seed=5).fit(None, y, patches=patches)
        return {"m": m, "src": state["sources"][-1], "es": state["es"][-1], "aug_calls": state["aug"]}

    return run


@pytest.mark.slow
def test_fit_uses_training_params_and_architecture(spied_fit):
    """Barcha o'qitish parametrlari + arxitektura parametrlari fit() orqali o'tadi; class_weight = neg/pos."""
    over = dict(filters1=6, filters2=0, kernel_size=5, dense_units=12, dropout=0.55, l2=0.02, optimizer="sgd",
                learning_rate=0.0123, batch_size=5, epochs=3, patience=9, val_fraction=0.35, augment=True)
    ctx = spied_fit(small(**over))
    model, info, src = ctx["m"].model_, ctx["m"].fit_info, ctx["src"]
    for chk in (_check_filters1, _check_filters2_off, _check_kernel, _check_dense, _check_dropout, _check_l2_on,
                _check_sgd, _check_lr):
        chk(model)
    assert info["epochs_run"] == 3                                           # epochs (patience > epochs)
    assert ctx["es"]["patience"] == 9                                        # patience
    assert info["n_val"] == math.ceil(60 * 0.35) and info["n_train"] == 60 - info["n_val"]    # val_fraction
    assert src.batch_size == 5 and len(src) == math.ceil(info["n_train"] / 5)                 # batch_size
    assert src.augment is True and ctx["aug_calls"] > 0                                       # augment
    n_pos, n_neg = int(src.y.sum()), int((src.y == 0).sum())
    assert n_pos + n_neg == info["n_train"] and n_pos < n_neg
    assert src.sw[src.y == 1][0] == pytest.approx(n_neg / n_pos) == pytest.approx(info["class_weight_pos"])
    assert (src.sw[src.y == 0] == 1.0).all()
    assert abs(n_pos / info["n_train"] - 0.2) < 0.06                         # stratified: sinf nisbati saqlangan


@pytest.mark.slow
def test_fit_without_augmentation_and_other_architecture(spied_fit):
    ctx = spied_fit(small(window=7, filters2=12, optimizer="rmsprop", l2=0.0, augment=False, epochs=1), window=7)
    model = ctx["m"].model_
    for chk in (_check_window, _check_filters2_on, _check_rmsprop, _check_l2_off):
        chk(model)
    assert ctx["src"].augment is False and ctx["aug_calls"] == 0             # augment=False


# ---------------------------------------------------------------------------
# data.py bilan integratsiya (slow): haqiqiy Dataset.get_patches -> CNNModel
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_integration_with_dataset_patches(synth_project, tf_keras):
    from mpm import data as D
    raster = D.load_and_align_rasters(D.find_tiff_files(synth_project["tiff"]))
    pipe = D.FeaturePipeline(raster.band_names, raster.categorical).fit(raster)
    aoi = D.load_aoi(D.find_shapefile(synth_project["aoi"]))
    pos = D.load_positive_points(D.find_shapefile(synth_project["points"]), aoi)
    bg = D.generate_background_points(aoi, pos, 40, 500.0, 42, "random", D.valid_pixel_mask(raster), raster.transform)
    ds = D.build_dataset(raster, pipe, pos, bg, need_feature_stack=True)
    patches = ds.get_patches(5)
    m = CNNModel(small(epochs=2), seed=1).fit(ds.X, ds.y, patches=patches)
    p = m.predict_proba_pos(patches=patches)
    assert p.shape == (ds.n,) and np.isfinite(p).all()
    sub = ds.subset(np.arange(0, ds.n, 3))                  # xarita bashorati kabi: bo'lak-bo'lak patch, kichik batch
    np.testing.assert_allclose(m.predict_proba_pos(patches=sub.get_patches(5), batch_size=4), p[::3], atol=1e-5)
