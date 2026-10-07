# -*- coding: utf-8 -*-
"""mpm.tuning testlari: sample_params (tur/diapazon/log) va tune_model (determinizm, base afzal, kam musbat,
bekor qilish, CNN window qidiruvi)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from mpm import tuning as T
from mpm.base import ModelWrapper
from mpm.cnn import normalize_cnn_params
from mpm.common import CancelledError, CancelToken
from mpm.config import PARAM_SPECS, TuningConfig, default_hyperparams, default_search_space, validate_hyperparams
from mpm.data import Dataset
from mpm.models import make_model
from mpm.spatial import assign_spatial_blocks, stratified_group_splits

P = 5
NAMES = [f"f{i}" for i in range(P)]


# ---------------------------------------------------------------------------
# Ma'lumot yordamchilari
# ---------------------------------------------------------------------------
def make_ds(n_pos=24, n_neg=76, seed=0, block=1500.0):
    """Tasodifiy joylashgan nuqtalar; 0-feature'da signal; oxirgi ustun - noyob id (stub testlari uchun)."""
    rng = np.random.default_rng(seed)
    n = n_pos + n_neg
    coords = rng.uniform(0, 10_000, size=(n, 2))
    X = rng.normal(size=(n, P)).astype(np.float32)
    X[:n_pos, 0] += 2.5
    X[:, -1] = np.arange(n)
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(np.int8)
    z = np.zeros(n, dtype=np.int64)
    ds = Dataset(X=X, y=y, coords=coords, rows=z, cols=z, feature_names=NAMES)
    return ds, assign_spatial_blocks(coords, block)


def make_custom_ds(pos_xy, n_neg=60, seed=0, block=1500.0):
    """Musbat nuqtalar koordinatalari aniq berilgan (blok sonini boshqarish uchun)."""
    rng = np.random.default_rng(seed)
    pos_xy = np.asarray(pos_xy, dtype=float).reshape(-1, 2)
    n_pos = len(pos_xy)
    coords = np.vstack([pos_xy, rng.uniform(0, 20_000, size=(n_neg, 2))])
    n = n_pos + n_neg
    X = rng.normal(size=(n, P)).astype(np.float32)
    X[:n_pos, 0] += 2.5
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(np.int8)
    z = np.zeros(n, dtype=np.int64)
    ds = Dataset(X=X, y=y, coords=coords, rows=z, cols=z, feature_names=NAMES)
    return ds, assign_spatial_blocks(coords, block)


def rf_base(**over):
    p = dict(default_hyperparams()["RandomForest"], n_estimators=15)
    p.update(over)
    return p


def fast_tuning(n_iter=5, **kw):
    spaces = {"RandomForest": {"n_estimators": {"min": 10, "max": 25}}}
    spaces.update(kw.pop("spaces", {}))
    return TuningConfig(enabled=True, n_iter=n_iter, inner_splits=kw.pop("inner_splits", 3), spaces=spaces, **kw)


def run(name="RandomForest", ds=None, groups=None, tuning=None, base=None, **kw):
    if ds is None:
        ds, groups = make_ds()
    args = dict(calibrate=True, calibration_method="sigmoid", calibration_cv=3, n_jobs=1, seed=0)
    args.update(kw)
    return T.tune_model(name, base if base is not None else rf_base(), ds, groups,
                        tuning=tuning if tuning is not None else fast_tuning(), **args)


# ---------------------------------------------------------------------------
# Soxta model (make_model o'rniga): ball funksiyasi parametrlarga bog'liq
# ---------------------------------------------------------------------------
class StubModel(ModelWrapper):
    input_kind = "tabular"

    def __init__(self, name, params, score_fn, seed=0, kw=None, trace=None):
        super().__init__(name, params, seed=seed)
        self.score_fn, self.kw, self.trace = score_fn, kw or {}, trace
        self.train_ids = None

    def fit(self, X, y, patches=None, cancel=None, log_fn=None):
        self.train_ids = np.asarray(X)[:, -1].astype(int)
        if log_fn is not None:
            log_fn("  Ogohlantirish: soxta ogohlantirish")
        self.is_fitted = True
        return self

    def predict_proba_pos(self, X=None, patches=None, batch_size=8192):
        if self.trace is not None:
            self.trace.append((self.params, self.seed, self.train_ids, np.asarray(X)[:, -1].astype(int)))
        return np.asarray(self.score_fn(self.params, np.asarray(X)), dtype=np.float64)


def patch_factory(monkeypatch, score_fn, calls=None, trace=None):
    def fake(name, params, **kw):
        if calls is not None:
            calls.append((name, dict(params), kw))
        return StubModel(name, params, score_fn, seed=kw.get("seed", 0), kw=kw, trace=trace)

    monkeypatch.setattr(T, "make_model", fake)


# ---------------------------------------------------------------------------
# sample_params
# ---------------------------------------------------------------------------
def test_sample_params_int_range_type_and_coverage():
    rng = np.random.default_rng(0)
    space = {"max_depth": {"type": "int", "min": 2, "max": 5}}
    vals = [T.sample_params("XGBoost", default_hyperparams()["XGBoost"], space, rng)["max_depth"] for _ in range(300)]
    assert all(type(v) is int for v in vals) and set(vals) == {2, 3, 4, 5}


def test_sample_params_float_linear_and_log():
    rng = np.random.default_rng(1)
    base = default_hyperparams()["SVM"]
    lin = [T.sample_params("SVM", base, {"C": {"type": "float", "min": 0.01, "max": 100.0, "log": False}}, rng)["C"]
           for _ in range(1500)]
    log = [T.sample_params("SVM", base, {"C": {"type": "float", "min": 0.01, "max": 100.0, "log": True}}, rng)["C"]
           for _ in range(1500)]
    assert all(type(v) is float and 0.01 <= v <= 100.0 for v in lin + log)
    assert np.mean(np.array(lin) < 1.0) < 0.05                       # tekis: ~1%
    assert 0.40 < np.mean(np.array(log) < 1.0) < 0.60                # log: 4 dekadaning yarmi
    assert 0.7 < float(np.exp(np.mean(np.log(log)))) < 1.4           # geometrik o'rtacha ~ 1


def test_sample_params_int_log_and_round():
    rng = np.random.default_rng(2)
    space = {"n_estimators": {"type": "int", "min": 10, "max": 1000, "log": True}}
    vals = np.array([T.sample_params("XGBoost", default_hyperparams()["XGBoost"], space, rng)["n_estimators"]
                     for _ in range(1500)])
    assert vals.dtype.kind == "i" and vals.min() >= 10 and vals.max() <= 1000
    assert 0.40 < np.mean(vals < 100) < 0.60                         # log-tekis
    # kasr chegaralar round bilan butunlanadi
    v = T.sample_params("RandomForest", rf_base(), {"min_samples_leaf": {"type": "int", "min": 1.4, "max": 3.6}},
                        rng)["min_samples_leaf"]
    assert type(v) is int and 1 <= v <= 4


def test_sample_params_choice_keeps_original_types():
    rng = np.random.default_rng(3)
    base = default_hyperparams()["CNN"]
    seen = set()
    for _ in range(200):
        r = T.sample_params("CNN", base, {"window": {"type": "choice", "choices": [5, 7, np.int64(9)]}}, rng)
        assert type(r["window"]) is int
        seen.add(r["window"])
    assert seen == {5, 7, 9}
    rf = default_hyperparams()["RandomForest"]
    got = {T.sample_params("RandomForest", rf, {"max_features": {"type": "choice",
                                                                  "choices": ["sqrt", "log2", "all"]}},
                           rng)["max_features"] for _ in range(100)}
    assert got == {"sqrt", "log2", "all"}


def test_sample_params_preserves_other_keys_and_is_valid():
    rng = np.random.default_rng(4)
    base = {"kernel": "linear", "class_weight": "none", "degree": 5, "C": 3.0, "gamma": None}
    r = T.sample_params("SVM", base, {"C": {"type": "float", "min": 0.5, "max": 2.0}}, rng)
    assert r["kernel"] == "linear" and r["class_weight"] == "none" and r["degree"] == 5 and r["gamma"] is None
    assert 0.5 <= r["C"] <= 2.0
    assert set(r) == {s.name for s in PARAM_SPECS["SVM"]}            # to'liq lug'at
    assert base["C"] == 3.0                                          # kirish o'zgarmaydi
    clean, warns = validate_hyperparams({"SVM": r})
    assert clean["SVM"] == r and warns == []                         # validate_hyperparams'dan o'tgan


def test_sample_params_partial_base_filled_with_defaults():
    r = T.sample_params("XGBoost", {"max_depth": 3}, {"learning_rate": {"type": "float", "min": 0.05, "max": 0.1}},
                        np.random.default_rng(0))
    assert r["max_depth"] == 3 and r["n_estimators"] == 300 and set(r) == {s.name for s in PARAM_SPECS["XGBoost"]}
    r0 = T.sample_params("XGBoost", None, {}, np.random.default_rng(0))
    assert r0 == default_hyperparams()["XGBoost"]                    # bo'sh space: o'zgarishsiz


def test_sample_params_clamped_to_spec_range():
    r = T.sample_params("RandomForest", rf_base(), {"n_estimators": {"type": "int", "min": 1, "max": 5}},
                        np.random.default_rng(0))
    assert r["n_estimators"] == 10                                   # spec.min=10 gacha ko'tarilgan


def test_sample_params_deterministic_and_rng_forms():
    space = default_search_space("XGBoost")
    base = default_hyperparams()["XGBoost"]
    a = T.sample_params("XGBoost", base, space, np.random.default_rng(7))
    b = T.sample_params("XGBoost", base, space, np.random.default_rng(7))
    c = T.sample_params("XGBoost", base, space, np.random.default_rng(8))
    assert a == b and a != c
    assert T.sample_params("XGBoost", base, space, 7) == a           # int seed ham qabul qilinadi
    rev = dict(reversed(list(space.items())))
    assert T.sample_params("XGBoost", base, rev, np.random.default_rng(7)) == a   # lug'at tartibiga bog'liq emas


@pytest.mark.parametrize("model", ["RandomForest", "SVM", "XGBoost", "CNN"])
def test_sample_params_default_spaces_valid_for_models(model):
    rng = np.random.default_rng(5)
    space = default_search_space(model)
    base = default_hyperparams()[model]
    for _ in range(60):
        r = T.sample_params(model, base, space, rng)
        assert validate_hyperparams({model: r})[0][model] == r
        for p, cfg in space.items():
            if cfg["type"] == "choice":
                assert r[p] in cfg["choices"]
            else:
                assert cfg["min"] <= r[p] <= cfg["max"] and (cfg["type"] != "int" or type(r[p]) is int)
        if model == "CNN":
            normalize_cnn_params(r)                                  # TF'siz tekshiruv
        else:
            make_model(model, r)                                     # noma'lum/yaroqsiz param bo'lsa ValueError


@pytest.mark.parametrize("model, space", [
    ("RandomForest", {"nomalum": {"type": "int", "min": 1, "max": 2}}),
    ("RandomForest", {"n_estimators": {"type": "int", "min": 50, "max": 10}}),
    ("SVM", {"C": {"type": "float", "min": 0.0, "max": 10.0, "log": True}}),
    ("SVM", {"C": {"type": "float", "min": None, "max": 10.0}}),
    ("SVM", {"C": {"type": "float", "min": "x", "max": 10.0}}),
    ("SVM", {"kernel": {"type": "choice", "choices": []}}),
    ("SVM", {"C": {"type": "zaryad", "min": 1, "max": 2}}),
    ("SVM", {"C": 5}),
    ("YOQ", {}),
])
def test_sample_params_invalid_space(model, space):
    with pytest.raises(ValueError):
        T.sample_params(model, {}, space, np.random.default_rng(0))


def test_sample_params_type_inferred_when_missing():
    rng = np.random.default_rng(0)
    r = T.sample_params("XGBoost", None, {"max_depth": {"min": 2, "max": 4}, "gamma": {"min": 0.0, "max": 1.0},
                                          "tree_method": {"choices": ["hist", "exact"]}}, rng)
    assert type(r["max_depth"]) is int and type(r["gamma"]) is float and r["tree_method"] in ("hist", "exact")


# ---------------------------------------------------------------------------
# tune_model: tuzilma, determinizm, tanlash qoidalari (haqiqiy RF)
# ---------------------------------------------------------------------------
def test_tune_model_structure_real_rf():
    ds, groups = make_ds()
    logs = []
    res = run(ds=ds, groups=groups, tuning=fast_tuning(n_iter=5), log_fn=logs.append)
    assert set(res) >= {"best_params", "best_score", "trials", "n_trials"}
    assert res["n_trials"] == 5 == len(res["trials"])
    base = validate_hyperparams({"RandomForest": rf_base()})[0]["RandomForest"]
    assert res["trials"][0]["params"] == base                        # 1-nomzod = base_params
    for t in res["trials"]:
        assert set(t) >= {"params", "score", "std"} and 0.0 <= t["score"] <= 1.0 and t["std"] >= 0.0
        assert validate_hyperparams({"RandomForest": t["params"]})[0]["RandomForest"] == t["params"]
    assert res["best_score"] == max(t["score"] for t in res["trials"])
    assert res["best_params"] == res["trials"][res["best_index"]]["params"]
    assert set(res["best_params"]) == {s.name for s in PARAM_SPECS["RandomForest"]}      # TO'LIQ lug'at
    assert res["scoring"] == "roc_auc" and res["inner_splits_used"] == 3 and res["trials"][0]["n_folds"] == 3
    assert len({t["params"]["n_estimators"] for t in res["trials"]}) > 1                 # tasodifiy nomzodlar farqli
    assert all(10 <= t["params"]["n_estimators"] <= 25 for t in res["trials"][1:])      # resolved_space (override)
    assert any("Tuning RandomForest" in m for m in logs)


def test_tune_model_deterministic_real_rf():
    ds, groups = make_ds()
    a = run(ds=ds, groups=groups, seed=3)
    b = run(ds=ds, groups=groups, seed=3)
    c = run(ds=ds, groups=groups, seed=4)
    assert a == b
    assert [t["params"] for t in a["trials"][1:]] != [t["params"] for t in c["trials"][1:]]


def test_tune_model_does_not_mutate_inputs():
    ds, groups = make_ds()
    X0, y0, g0 = ds.X.copy(), ds.y.copy(), groups.copy()
    base = rf_base()
    b0 = dict(base)
    tun = fast_tuning()
    t0 = tun.to_dict()
    run(ds=ds, groups=groups, base=base, tuning=tun)
    assert np.array_equal(ds.X, X0) and np.array_equal(ds.y, y0) and np.array_equal(groups, g0)
    assert base == b0 and tun.to_dict() == t0


@pytest.mark.parametrize("name, base_over, space", [
    ("SVM", {}, {}),
    ("XGBoost", {"n_estimators": 20}, {"XGBoost": {"n_estimators": {"min": 10, "max": 25}}}),
])
def test_tune_model_other_models(name, base_over, space):
    base = dict(default_hyperparams()[name], **base_over)
    res = run(name, base=base, tuning=fast_tuning(n_iter=3, spaces=space))
    assert res["n_trials"] == 3 and np.isfinite(res["best_score"])
    make_model(name, res["best_params"])


def test_tune_model_scoring_average_precision_and_invalid(monkeypatch):
    ds, groups = make_ds()
    patch_factory(monkeypatch, lambda p, X: X[:, 0])                  # parametrga bog'liq emas
    splits = list(stratified_group_splits(ds.y, groups, 3, 1, 5))
    exp_ap = float(np.mean([average_precision_score(ds.y[va], ds.X[va, 0]) for _, _, _, va in splits]))
    exp_auc = float(np.mean([roc_auc_score(ds.y[va], ds.X[va, 0]) for _, _, _, va in splits]))
    ap = run(ds=ds, groups=groups, seed=5, tuning=fast_tuning(n_iter=2, scoring="average_precision"))
    auc = run(ds=ds, groups=groups, seed=5, tuning=fast_tuning(n_iter=2, scoring="roc_auc"))
    assert ap["scoring"] == "average_precision" and ap["best_score"] == pytest.approx(exp_ap)
    assert auc["best_score"] == pytest.approx(exp_auc) and exp_ap != pytest.approx(exp_auc)
    logs = []
    tun = fast_tuning(n_iter=2)
    tun.scoring = "bogus"
    bad = run(ds=ds, groups=groups, seed=5, tuning=tun, log_fn=logs.append)
    assert bad["scoring"] == "roc_auc" and bad["best_score"] == pytest.approx(exp_auc)
    assert any("scoring" in m for m in logs)


def test_tune_model_accepts_dict_tuning(monkeypatch):
    patch_factory(monkeypatch, lambda p, X: X[:, 0])
    res = run(tuning=fast_tuning(n_iter=3).to_dict())
    assert res["n_trials"] == 3


# ---------------------------------------------------------------------------
# tanlash qoidalari (soxta model): base afzal, tasodifiy yaxshiroq bo'lsa g'olib
# ---------------------------------------------------------------------------
def test_tune_model_base_preferred_on_tie(monkeypatch):
    patch_factory(monkeypatch, lambda p, X: X[:, 0])                  # hamma nomzod bir xil ball
    res = run("XGBoost", base=default_hyperparams()["XGBoost"], tuning=TuningConfig(enabled=True, n_iter=8),
              seed=2)
    scores = [t["score"] for t in res["trials"]]
    assert len(scores) == 8 and len(set(scores)) == 1
    assert res["best_index"] == 0 and res["best_params"] == res["trials"][0]["params"]
    assert res["best_params"] == default_hyperparams()["XGBoost"]


def test_tune_model_random_candidate_wins_when_better(monkeypatch):
    noise = np.sin(np.arange(1000) * 12.9898)

    def score(p, X):                                                  # max_depth >= 5 bo'lsa signal, aks holda shovqin
        return X[:, 0] if p["max_depth"] >= 5 else noise[X[:, -1].astype(int)]

    patch_factory(monkeypatch, score)
    res = run("XGBoost", base=default_hyperparams()["XGBoost"],      # standart max_depth = 4 (shovqin)
              tuning=TuningConfig(enabled=True, n_iter=12), seed=1)
    assert res["trials"][0]["params"]["max_depth"] == 4 and res["best_index"] != 0
    assert res["best_params"]["max_depth"] >= 5 and res["best_score"] > res["trials"][0]["score"] + 0.1
    assert res["best_params"] == res["trials"][res["best_index"]]["params"]


def test_tune_model_passes_options_and_uses_paired_folds(monkeypatch):
    ds, groups = make_ds()
    calls, trace = [], []
    patch_factory(monkeypatch, lambda p, X: X[:, 0], calls=calls, trace=trace)
    res = run(ds=ds, groups=groups, seed=11, n_jobs=2, calibrate=False, calibration_method="isotonic",
              calibration_cv=4, tuning=fast_tuning(n_iter=4, inner_splits=3))
    assert len(calls) == 4 * 3
    for name, params, kw in calls:
        assert name == "RandomForest"
        assert kw["calibrate"] is False and kw["calibration_method"] == "isotonic" and kw["calibration_cv"] == 4
        assert kw["n_jobs"] == 2 and kw["n_features"] == P
    seeds_by_cand = {}
    for params, seed, tr, va in trace:
        seeds_by_cand.setdefault(tuple(sorted(params.items(), key=str)), []).append(seed)
        assert not (set(groups[tr]) & set(groups[va]))               # blok train/val orasida bo'linmagan
        assert not (set(tr) & set(va))
    assert all(v == [11, 12, 13] for v in seeds_by_cand.values())    # har nomzod uchun bir xil fold seed'lari
    # ichki fold'lar spatial.stratified_group_splits(n_splits=inner_splits, n_repeats=1) bilan bir xil
    splits = list(stratified_group_splits(ds.y, groups, 3, 1, 11))
    assert [sorted(va) for _, _, _, va in splits] == [sorted(t[3]) for t in trace[:3]]
    assert res["n_trials"] == 4


def test_tune_model_fit_warnings_forwarded_once(monkeypatch):
    patch_factory(monkeypatch, lambda p, X: X[:, 0])
    logs = []
    run(tuning=fast_tuning(n_iter=3), log_fn=logs.append)
    assert sum("soxta ogohlantirish" in m for m in logs) == 1


def test_tune_model_failing_candidates_do_not_crash(monkeypatch):
    def score(p, X):
        if p["max_depth"] <= 3:
            raise ValueError("portlash")
        return X[:, 0]

    patch_factory(monkeypatch, score)
    logs = []
    res = run("XGBoost", base=default_hyperparams()["XGBoost"], tuning=TuningConfig(enabled=True, n_iter=12),
              seed=1, log_fn=logs.append)
    bad = [t for t in res["trials"] if t["params"]["max_depth"] <= 3]
    assert bad and all(np.isnan(t["score"]) and t["n_folds"] == 0 for t in bad)
    assert np.isfinite(res["best_score"]) and res["best_params"]["max_depth"] >= 4
    assert any("baholanmadi" in m for m in logs)


def test_tune_model_all_candidates_fail_returns_base(monkeypatch):
    def score(p, X):
        raise RuntimeError("hammasi yiqildi")

    patch_factory(monkeypatch, score)
    logs = []
    base = default_hyperparams()["XGBoost"]
    res = run("XGBoost", base=base, tuning=TuningConfig(enabled=True, n_iter=3), log_fn=logs.append)
    assert res["best_params"] == base and np.isnan(res["best_score"]) and res["n_trials"] == 3
    assert any("baholanmadi" in m for m in logs)


def test_tune_model_single_class_folds_are_skipped(monkeypatch):
    ds, groups = make_ds()
    patch_factory(monkeypatch, lambda p, X: X[:, 0])
    # bir sinfli validation fold'ni majburlash: spatial bo'linishni almashtiramiz
    idx = np.arange(ds.n)
    pos, neg = idx[ds.y == 1], idx[ds.y == 0]
    folds = [(np.r_[pos[:12], neg[:30]], np.r_[pos[12:], neg[30:50]]),
             (np.r_[pos, neg[:30]], neg[30:60]),                      # val'da faqat fon -> tashlanadi
             (np.r_[neg[:30], pos[:5]], pos[5:])]                     # val'da faqat musbat -> tashlanadi
    monkeypatch.setattr(T, "stratified_group_splits",
                        lambda *a, **k: iter([(0, i, tr, va) for i, (tr, va) in enumerate(folds)]))
    res = run(ds=ds, groups=groups, tuning=fast_tuning(n_iter=2, inner_splits=3))
    assert res["trials"][0]["n_folds"] == 1
    assert res["trials"][0]["score"] == pytest.approx(roc_auc_score(ds.y[folds[0][1]], ds.X[folds[0][1], 0]))


def test_score_helper():
    y = np.array([0, 1, 0, 1])
    assert T._score("roc_auc", y, [0.1, 0.9, 0.2, 0.8]) == 1.0
    assert T._score("average_precision", y, [0.1, 0.9, 0.2, 0.8]) == 1.0
    assert np.isnan(T._score("roc_auc", np.zeros(4, dtype=int), [0.1, 0.2, 0.3, 0.4]))     # bir sinf
    assert np.isnan(T._score("roc_auc", y, [0.1, np.nan, 0.2, 0.8]))                        # NaN bashorat


# ---------------------------------------------------------------------------
# Kam musbat / yetarsiz bloklar: hech qachon crash qilmaydi
# ---------------------------------------------------------------------------
def test_few_positives_three_blocks_real_rf():
    ds, groups = make_custom_ds([[500, 500], [5000, 500], [9000, 500]])          # 3 musbat, 3 blok
    res = run(ds=ds, groups=groups, tuning=fast_tuning(n_iter=3, inner_splits=3))
    assert res["n_trials"] == 3 and res["inner_splits_used"] == 3 and np.isfinite(res["best_score"])


def test_inner_splits_reduced_when_positive_blocks_few():
    pts = [[x, 500] for x in range(100, 1400, 130)] + [[x, 500] for x in range(3100, 4400, 130)]   # 2 musbat blok
    ds, groups = make_custom_ds(pts, n_neg=80)
    assert len(np.unique(groups[ds.y == 1])) == 2
    logs = []
    res = run(ds=ds, groups=groups, tuning=fast_tuning(n_iter=3, inner_splits=4), log_fn=logs.append)
    assert res["inner_splits_used"] == 2 and res["n_trials"] == 3 and np.isfinite(res["best_score"])
    assert any("kamaytirildi" in m for m in logs)


@pytest.mark.parametrize("pos_xy", [
    [[500, 500]],                                                    # 1 musbat nuqta
    [[500, 500], [600, 600], [700, 700]],                            # 3 musbat, bitta blok
])
def test_unsplittable_returns_base_with_warning(pos_xy):
    ds, groups = make_custom_ds(pos_xy)
    logs, base = [], rf_base()
    res = run(ds=ds, groups=groups, base=base, tuning=fast_tuning(n_iter=4), log_fn=logs.append)
    assert res["best_params"] == validate_hyperparams({"RandomForest": base})[0]["RandomForest"]
    assert res["trials"] == [] and res["n_trials"] == 0 and np.isnan(res["best_score"])
    assert "skipped_reason" in res and any("OGOHLANTIRISH" in m and "tuning" in m for m in logs)


def test_no_background_returns_base():
    ds, groups = make_ds(n_pos=10, n_neg=0)
    res = run(ds=ds, groups=groups, tuning=fast_tuning(n_iter=2))
    assert res["n_trials"] == 0 and "skipped_reason" in res


def test_skipped_result_has_same_schema_as_normal():
    """Regressiya: o'tkazib yuborilgan (fallback) natijada base_score/best_index/inner_splits_used yo'q edi -
    iste'molchi (plots/pipeline) res["base_score"] da KeyError olardi. Kalitlar to'plami bir xil bo'lishi kerak."""
    normal = run(tuning=fast_tuning(n_iter=2))
    ds, groups = make_custom_ds([[500, 500]])                       # 1 musbat nuqta => split imkonsiz
    skipped = run(ds=ds, groups=groups, tuning=fast_tuning(n_iter=2))
    assert "skipped_reason" in skipped and "skipped_reason" not in normal
    assert set(skipped) == set(normal) | {"skipped_reason"}
    assert skipped["best_index"] == 0 and skipped["inner_splits_used"] == 0 and np.isnan(skipped["base_score"])


def test_invalid_space_returns_base_with_warning():
    logs = []
    tun = fast_tuning(n_iter=3, spaces={"RandomForest": {"n_estimators": {"min": 500, "max": 100}}})
    res = run(tuning=tun, log_fn=logs.append)
    assert res["n_trials"] == 0 and "qidiruv oralig'i yaroqsiz" in res["skipped_reason"]
    assert any("OGOHLANTIRISH" in m for m in logs)


def test_small_search_space_dedupes(monkeypatch):
    patch_factory(monkeypatch, lambda p, X: X[:, 0])
    tun = TuningConfig(enabled=True, n_iter=20, spaces={"XGBoost": {"max_depth": {"min": 3, "max": 4}}})
    # boshqa tunable'larni bitta qiymatga qisqartiramiz
    for p, cfg in default_search_space("XGBoost").items():
        if p != "max_depth":
            lo = cfg["min"]
            tun.spaces["XGBoost"][p] = {"min": lo, "max": lo}
    logs = []
    res = run("XGBoost", base=default_hyperparams()["XGBoost"], tuning=tun, log_fn=logs.append)
    keys = {T._key(t["params"]) for t in res["trials"]}
    assert len(keys) == res["n_trials"] and res["n_trials"] <= 4      # noyob nomzodlar bilan chegaralangan
    assert any("qidiruv fazosi kichik" in m for m in logs)


def test_input_validation_errors():
    ds, groups = make_ds()
    with pytest.raises(ValueError, match="groups"):
        run(ds=ds, groups=groups[:-1])
    with pytest.raises(ValueError, match="Noma'lum model"):
        run("YOQ", ds=ds, groups=groups)


# ---------------------------------------------------------------------------
# cancel / progress
# ---------------------------------------------------------------------------
def test_progress_after_each_candidate(monkeypatch):
    patch_factory(monkeypatch, lambda p, X: X[:, 0])
    fr = []
    run("XGBoost", base=default_hyperparams()["XGBoost"], tuning=TuningConfig(enabled=True, n_iter=5),
        progress_fn=lambda f, msg="": fr.append((f, msg)))
    assert len(fr) == 5 and fr[-1][0] == 1.0 and [f for f, _ in fr] == sorted(f for f, _ in fr)
    assert all("XGBoost" in m for _, m in fr)


def test_cancel_before_and_during(monkeypatch):
    patch_factory(monkeypatch, lambda p, X: X[:, 0])
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledError):
        run(cancel=tok)
    tok2 = CancelToken()
    seen = []

    def prog(f, msg=""):
        seen.append(f)
        tok2.cancel()                                                 # birinchi nomzoddan keyin bekor

    with pytest.raises(CancelledError):
        run("XGBoost", base=default_hyperparams()["XGBoost"], tuning=TuningConfig(enabled=True, n_iter=6),
            cancel=tok2, progress_fn=prog)
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------
def make_cnn_ds(seed=0, H=36, p=3, n_pos=14, n_neg=30, with_stack=True):
    rng = np.random.default_rng(seed)
    stack = rng.normal(size=(p, H, H)).astype(np.float32)
    rows = rng.integers(4, H - 4, size=n_pos + n_neg)
    cols = rng.integers(4, H - 4, size=n_pos + n_neg)
    for r, c in zip(rows[:n_pos], cols[:n_pos]):
        stack[0, r - 1:r + 2, c - 1:c + 2] += 2.5
    X = stack[:, rows, cols].T.copy()
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(np.int8)
    coords = np.column_stack([cols * 100.0, rows * 100.0])
    ds = Dataset(X=X, y=y, coords=coords, rows=rows.astype(np.int64), cols=cols.astype(np.int64),
                 feature_names=[f"b{i}" for i in range(p)], feature_stack=stack if with_stack else None)
    return ds, assign_spatial_blocks(coords, 700.0)


def cnn_base(**over):
    p = dict(default_hyperparams()["CNN"], window=5, filters1=4, filters2=8, dense_units=8, batch_size=16,
             epochs=3, patience=2)
    p.update(over)
    return p


def cnn_tuning(n_iter=4, windows=(3, 5, 7)):
    sp = {"window": {"choices": list(windows)}, "filters1": {"choices": [4]}, "filters2": {"choices": [8]},
          "dense_units": {"choices": [8]}, "batch_size": {"choices": [16]},
          "dropout": {"min": 0.2, "max": 0.3}, "learning_rate": {"min": 0.003, "max": 0.01}}
    return TuningConfig(enabled=True, n_iter=n_iter, inner_splits=2, spaces={"CNN": sp})


def test_cnn_patch2d_without_feature_stack_returns_base():
    ds, groups = make_cnn_ds(with_stack=False)
    logs = []
    res = run("CNN", ds=ds, groups=groups, base=cnn_base(), tuning=cnn_tuning(), log_fn=logs.append)
    assert res["n_trials"] == 0 and "feature_stack" in res["skipped_reason"]
    assert res["best_params"] == validate_hyperparams({"CNN": cnn_base()})[0]["CNN"]
    assert any("OGOHLANTIRISH" in m for m in logs)


@pytest.mark.slow
def test_cnn_window_search_uses_patches_per_window():
    pytest.importorskip("tensorflow")
    ds, groups = make_cnn_ds()
    windows = []
    orig = ds.get_patches

    def spy(w):
        windows.append(int(w))
        return orig(w)

    ds.get_patches = spy
    logs = []
    res = run("CNN", ds=ds, groups=groups, base=cnn_base(), tuning=cnn_tuning(n_iter=4), seed=1, log_fn=logs.append)
    used = [t["params"]["window"] for t in res["trials"]]
    assert res["n_trials"] == 4 and res["trials"][0]["params"]["window"] == 5
    assert set(used) <= {3, 5, 7} and len(set(used)) >= 2             # window haqiqatan qidirilgan
    assert sorted(windows) == sorted(set(used))                       # har window uchun patchlar bir marta olingan
    assert res["best_params"]["window"] in {3, 5, 7} and np.isfinite(res["best_score"])
    assert res["best_params"] == res["trials"][res["best_index"]]["params"]
    assert any("CNN tuning qimmat" in m for m in logs)                # cost-eslatma
    normalize_cnn_params(res["best_params"])


@pytest.mark.slow
def test_cnn_tabular1d_ignores_window():
    pytest.importorskip("tensorflow")
    ds, groups = make_cnn_ds(with_stack=False)
    ds.get_patches = lambda w: pytest.fail("tabular1d da patch olinmasligi kerak")
    tun = cnn_tuning(n_iter=2)
    res = run("CNN", ds=ds, groups=groups, base=cnn_base(mode="tabular1d", window=9), tuning=tun, seed=1)
    assert res["n_trials"] == 2 and {t["params"]["window"] for t in res["trials"]} == {9}
    assert res["best_params"]["mode"] == "tabular1d" and np.isfinite(res["best_score"])
