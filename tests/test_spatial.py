# -*- coding: utf-8 -*-
"""mpm.spatial testlari: bloklar, variogram, stratified group fold'lar, blok-bootstrap, fold_report."""
import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import StratifiedGroupKFold

from mpm import spatial
from mpm.common import RANDOM_STATE
from tests.synth import X0, Y0

RES = 100.0


# ---------------------------------------------------------------------------
# Yordamchilar
# ---------------------------------------------------------------------------
def _points(n_pos=22, n_neg=80, extent=12000.0, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, extent, size=(n_pos + n_neg, 2)) + np.array([X0, Y0])
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(np.int8)
    return coords, y


def _smooth_band(sigma_px, seed=0, size=120):
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.normal(size=(size, size)), sigma=sigma_px)
    return ((f - f.mean()) / f.std()).astype("float32")


def _transform(size=120):
    return from_origin(X0, Y0 + size * RES, RES, RES)


def _val_groups(groups, val):
    return set(np.unique(groups[val]).tolist())


def _rng_range(sigma_px, seed=0, **kw):
    return spatial.estimate_autocorrelation_range(_smooth_band(sigma_px, seed)[None], _transform(),
                                                  random_state=seed, **kw)


# ---------------------------------------------------------------------------
# assign_spatial_blocks
# ---------------------------------------------------------------------------
def test_assign_blocks_cells():
    coords = np.array([[10.0, 10.0], [90.0, 90.0], [110.0, 10.0], [-10.0, -10.0], [10.0, 105.0]])
    g = spatial.assign_spatial_blocks(coords, 100.0)
    assert g.shape == (5,) and g.dtype.kind == "i"
    assert g[0] == g[1]                       # bir xil katak
    assert len({g[0], g[2], g[3], g[4]}) == 4  # qolganlari turli kataklar (manfiy koordinata ham)
    assert set(g.tolist()) == set(range(4))    # zich raqamlash 0..n_blocks-1


def test_assign_blocks_large_coordinates_no_collision():
    # EPSG:28411 kattalikdagi koordinatalar va kichik blok: kalit to'qnashuvi bo'lmasligi kerak
    coords = np.array([[X0 + 1.0, Y0 + 1.0], [X0 + 1.0, Y0 + 999.0], [X0 + 999.0, Y0 + 1.0]])
    g = spatial.assign_spatial_blocks(coords, 10.0)
    assert len(set(g.tolist())) == 3


def test_assign_blocks_validation():
    coords, _ = _points()
    for bad in (0, -5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Blok o'lchami"):
            spatial.assign_spatial_blocks(coords, bad)
    with pytest.raises(ValueError, match="coords"):
        spatial.assign_spatial_blocks(np.zeros((4, 3)), 100.0)
    nan_coords = coords.copy()
    nan_coords[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        spatial.assign_spatial_blocks(nan_coords, 100.0)
    assert spatial.assign_spatial_blocks(np.zeros((0, 2)), 100.0).size == 0


def test_assign_blocks_tiny_block_overflow_raises():
    # blok raqamlari int64 ga sig'maydi: jim-jit axlat guruhlar o'rniga aniq xato
    coords, _ = _points()
    with pytest.raises(ValueError, match="juda kichik"):
        spatial.assign_spatial_blocks(coords, 1e-15)
    assert spatial.assign_spatial_blocks(coords, 1e-3).max() > 0


# ---------------------------------------------------------------------------
# adapt_block_size
# ---------------------------------------------------------------------------
def test_adapt_block_size_unchanged_when_ok():
    coords, y = _points()
    msgs = []
    bs, groups = spatial.adapt_block_size(coords, y, 5, 1000.0, log_fn=msgs.append)
    assert bs == 1000.0 and msgs == []
    assert np.array_equal(groups, spatial.assign_spatial_blocks(coords, 1000.0))


def test_adapt_block_size_shrinks_huge_block():
    coords, y = _points()
    msgs = []
    bs, groups = spatial.adapt_block_size(coords, y, 5, 50000.0, log_fn=msgs.append)
    assert bs < 50000.0
    assert bs == pytest.approx(50000.0 / 2 ** round(np.log2(50000.0 / bs)))   # faqat yarmilash
    assert groups.max() + 1 >= 5
    assert np.unique(groups[y == 1]).size >= 5
    assert np.array_equal(groups, spatial.assign_spatial_blocks(coords, bs))
    assert len(msgs) == 1 and "kamaytirildi" in msgs[0]
    # bir qadam kam yarmilash yetarli bo'lmasligi kerak edi (eng katta mos o'lcham tanlangan)
    g_prev = spatial.assign_spatial_blocks(coords, bs * 2)
    assert g_prev.max() + 1 < 5 or np.unique(g_prev[y == 1]).size < 5


def test_adapt_block_size_requires_positive_blocks():
    # musbatlar bitta zich to'dada, fon keng tarqalgan: bloklar soni yetadi, musbat bloklar yetmaydi
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 400, size=(10, 2))
    neg = rng.uniform(0, 12000, size=(100, 2))
    coords = np.vstack([pos, neg]) + np.array([X0, Y0])
    y = np.r_[np.ones(10), np.zeros(100)].astype(np.int8)
    g0 = spatial.assign_spatial_blocks(coords, 3000.0)
    assert g0.max() + 1 >= 5 and np.unique(g0[y == 1]).size < 5
    bs, groups = spatial.adapt_block_size(coords, y, 5, 3000.0)
    assert bs < 3000.0 and np.unique(groups[y == 1]).size >= 5


def test_adapt_block_size_impossible():
    coords, y = _points(n_pos=3, n_neg=60)
    with pytest.raises(ValueError, match="Musbat nuqtalar soni"):
        spatial.adapt_block_size(coords, y, 5, 1000.0)
    # musbat nuqtalar yetarli, lekin barchasi bir xil joyda
    coords, y = _points(n_pos=8, n_neg=60)
    coords[:8] = coords[0]
    with pytest.raises(ValueError, match="turli joylari"):
        spatial.adapt_block_size(coords, y, 5, 1000.0)
    # min_size cheklovi: blok kichraya olmaydi
    coords, y = _points()
    with pytest.raises(ValueError, match="kamaytirib ham"):
        spatial.adapt_block_size(coords, y, 5, 50000.0, min_size=40000.0)
    with pytest.raises(ValueError, match="n_splits"):
        spatial.adapt_block_size(coords, y, 1, 1000.0)
    with pytest.raises(ValueError, match="y"):
        spatial.adapt_block_size(coords, y[:-1], 5, 1000.0)


def test_adapt_block_size_requires_background():
    coords, _ = _points()
    with pytest.raises(ValueError, match="Fon"):
        spatial.adapt_block_size(coords, np.ones(len(coords), dtype=np.int8), 5, 1000.0)


def test_adapt_block_size_max_halvings():
    # 30 marta yarmilash yetmaydigan holat: ValueError, cheksiz sikl emas
    coords, y = _points()
    with pytest.raises(ValueError, match="kamaytirib ham"):
        spatial.adapt_block_size(coords, y, 5, 1e15, min_size=1e-9)


# ---------------------------------------------------------------------------
# stratified_group_splits
# ---------------------------------------------------------------------------
def _split_data(block_size=1500.0, n_pos=22, n_neg=80, seed=0):
    coords, y = _points(n_pos=n_pos, n_neg=n_neg, seed=seed)
    return coords, y, spatial.assign_spatial_blocks(coords, block_size)


def test_blocks_never_split_and_cover():
    coords, y, groups = _split_data()
    n = len(y)
    splits = list(spatial.stratified_group_splits(y, groups, 5, 4, random_state=1))
    assert len(splits) == 5 * 4
    for repeat in range(4):
        seen = []
        for r, fold, tr, va in splits:
            if r != repeat:
                continue
            assert not (_val_groups(groups, va) & _val_groups(groups, tr))   # blok bo'linmagan
            assert np.intersect1d(tr, va).size == 0
            assert np.array_equal(np.sort(np.r_[tr, va]), np.arange(n))
            seen.append(va)
        assert np.array_equal(np.sort(np.concatenate(seen)), np.arange(n))   # har nuqta 1 marta validation


def test_every_val_fold_has_positive():
    coords, y, groups = _split_data()
    for r, fold, tr, va in spatial.stratified_group_splits(y, groups, 5, 10, random_state=3):
        assert y[va].sum() >= 1 and y[tr].sum() >= 1
        assert (y[va] == 0).sum() >= 1


def test_clustered_positives_still_covered():
    # musbatlar kam blokda to'plangan (5 ta musbat blok = n_splits): har val fold'da baribir musbat bor
    rng = np.random.default_rng(5)
    centers = rng.uniform(1000, 11000, size=(5, 2))
    pos = np.vstack([c + rng.uniform(-150, 150, size=(3, 2)) for c in centers])
    neg = rng.uniform(0, 12000, size=(150, 2))
    coords = np.vstack([pos, neg]) + np.array([X0, Y0])
    y = np.r_[np.ones(15), np.zeros(150)].astype(np.int8)
    groups = spatial.assign_spatial_blocks(coords, 1500.0)
    assert np.unique(groups[y == 1]).size >= 5
    for r, fold, tr, va in spatial.stratified_group_splits(y, groups, 5, 6, random_state=0):
        assert y[va].sum() >= 1
        assert not (_val_groups(groups, va) & _val_groups(groups, tr))


def test_repeats_differ_and_deterministic():
    coords, y, groups = _split_data()
    a = list(spatial.stratified_group_splits(y, groups, 5, 3, random_state=RANDOM_STATE))
    b = list(spatial.stratified_group_splits(y, groups, 5, 3, random_state=RANDOM_STATE))
    assert len(a) == len(b) == 15
    for (r1, f1, t1, v1), (r2, f2, t2, v2) in zip(a, b):
        assert (r1, f1) == (r2, f2)
        assert np.array_equal(t1, t2) and np.array_equal(v1, v2)
    # repeat'lar bir-biridan farq qiladi
    vals = {r: tuple(tuple(v) for rr, f, t, v in a if rr == r) for r in range(3)}
    assert len({vals[0], vals[1], vals[2]}) == 3
    # random_state o'zgarsa natija ham o'zgaradi
    c = list(spatial.stratified_group_splits(y, groups, 5, 1, random_state=RANDOM_STATE + 1))
    assert any(not np.array_equal(v1, v2) for (_, _, _, v1), (_, _, _, v2) in zip(a[:5], c))
    # tartib: repeat 0..n-1, fold 0..k-1
    assert [(r, f) for r, f, _, _ in a] == [(r, f) for r in range(3) for f in range(5)]


def test_is_lazy_generator_but_validates_eagerly():
    coords, y, groups = _split_data()
    gen = spatial.stratified_group_splits(y, groups, 5, 2)
    assert iter(gen) is gen
    with pytest.raises(ValueError):          # xato chaqiruv paytidayoq (next() kerak emas)
        spatial.stratified_group_splits(y[:4], groups[:4], 5, 2)


def test_few_positive_raises_clear_error():
    coords, y, groups = _split_data(n_pos=4, n_neg=80)     # 4 musbat, n_splits=5
    with pytest.raises(ValueError, match="Musbat nuqtali bloklar soni"):
        list(spatial.stratified_group_splits(y, groups, 5, 2))
    coords, y, groups = _split_data(n_pos=3, n_neg=80)
    with pytest.raises(ValueError, match="Musbat nuqtali bloklar soni"):
        list(spatial.stratified_group_splits(y, groups, 5, 2))
    # musbatlar yetarli, lekin bitta blokda (musbat bloklar 1 ta)
    coords, y = _points(n_pos=10, n_neg=80)
    coords[:10] = coords[0] + np.random.default_rng(0).uniform(0, 50, size=(10, 2))
    groups = spatial.assign_spatial_blocks(coords, 2000.0)
    with pytest.raises(ValueError, match="Musbat nuqtali bloklar soni"):
        list(spatial.stratified_group_splits(y, groups, 5, 2))


def test_too_few_blocks_and_bad_args():
    coords, y = _points()
    groups = spatial.assign_spatial_blocks(coords, 1e6)    # bitta blok
    with pytest.raises(ValueError, match="bloklar soni"):
        spatial.stratified_group_splits(y, groups, 5, 1)
    groups = spatial.assign_spatial_blocks(coords, 1500.0)
    with pytest.raises(ValueError, match="n_splits"):
        spatial.stratified_group_splits(y, groups, 1, 1)
    with pytest.raises(ValueError, match="n_repeats"):
        spatial.stratified_group_splits(y, groups, 5, 0)
    with pytest.raises(ValueError, match="0 .*1"):
        spatial.stratified_group_splits(np.full(len(y), 2), groups, 5, 1)


def test_group_splits_single_class_raises():
    coords, y, groups = _split_data()
    with pytest.raises(ValueError, match="Fon"):
        spatial.stratified_group_splits(np.ones(len(y)), groups, 5, 1)       # fon yo'q
    with pytest.raises(ValueError, match="Musbat"):
        spatial.stratified_group_splits(np.zeros(len(y)), groups, 5, 1)      # musbat yo'q


@pytest.mark.parametrize("seed", [-1, -12345, 2 ** 32, 2 ** 32 - 1, 2 ** 40 + 7])
def test_extreme_seeds_are_accepted_and_deterministic(seed):
    # manfiy / 2**32 dan katta seed sklearn'da ValueError berardi
    coords, y, groups = _split_data()
    a = list(spatial.stratified_group_splits(y, groups, 5, 2, random_state=seed))
    b = list(spatial.stratified_group_splits(y, groups, 5, 2, random_state=seed))
    assert len(a) == 10 and all(np.array_equal(x[3], z[3]) for x, z in zip(a, b))
    ra = list(spatial.random_stratified_splits(y, 5, 2, random_state=seed))
    rb = list(spatial.random_stratified_splits(y, 5, 2, random_state=seed))
    assert len(ra) == 10 and all(np.array_equal(x[3], z[3]) for x, z in zip(ra, rb))
    ba = list(spatial.block_bootstrap_indices(groups, 3, random_state=seed))
    bb = list(spatial.block_bootstrap_indices(groups, 3, random_state=seed))
    assert len(ba) == 3 and all(np.array_equal(x, z) for x, z in zip(ba, bb))


def _blocks_from_sizes(pos_sizes, extra_neg, neg_sizes):
    """Blok darajasida sintetik guruhlar: har musbat blokda pos_sizes[i] musbat + extra_neg fon,
    so'ng faqat fonli bloklar."""
    groups, y = [], []
    for gid, size in enumerate(pos_sizes):
        groups += [gid] * (size + extra_neg)
        y += [1] * size + [0] * extra_neg
    for j, size in enumerate(neg_sizes):
        groups += [len(pos_sizes) + j] * size
        y += [0] * size
    return np.array(groups), np.array(y)


@pytest.mark.parametrize("n_splits,pos_sizes,extra_neg,neg_sizes", [
    (5, [8, 4, 1, 1, 1, 1], 2, [10, 10]),
    (10, [12, 6, 5, 3, 2, 2, 1, 1, 1, 1], 1, [5] * 6),
])
def test_val_folds_have_positive_when_positives_clustered_in_few_blocks(n_splits, pos_sizes, extra_neg,
                                                                        neg_sizes):
    # StratifiedGroupKFold bunda 50 urinishdan keyin ham musbatsiz val fold beradi (musbatlar bir necha
    # katta blokda to'plangan, musbat bloklar ~ n_splits); musbat bloklar >= n_splits bo'lgani uchun
    # butun blokni ko'chirish bilan har fold'ga musbat berish mumkin (BUG-05)
    groups, y = _blocks_from_sizes(pos_sizes, extra_neg, neg_sizes)
    assert np.unique(groups[y == 1]).size >= n_splits
    n = len(y)
    splits = list(spatial.stratified_group_splits(y, groups, n_splits, 3, random_state=42))
    assert len(splits) == 3 * n_splits
    for repeat in range(3):
        vals = []
        for r, fold, tr, va in splits:
            if r != repeat:
                continue
            assert y[va].sum() >= 1 and y[tr].sum() >= 1
            assert not (_val_groups(groups, va) & _val_groups(groups, tr))   # blok bo'linmagan
            assert np.array_equal(np.sort(np.r_[tr, va]), np.arange(n))
            vals.append(va)
        assert np.array_equal(np.sort(np.concatenate(vals)), np.arange(n))


def _fake_group_sgkf():
    """Blok bo'yicha izchil, lekin 0-fold'ga faqat musbatsiz bloklarni beradigan soxta SGKF."""
    class Fake:
        calls = []

        def __init__(self, n_splits, shuffle=False, random_state=None):
            self.n_splits = n_splits
            Fake.calls.append(random_state)

        def split(self, X, y, groups):
            _, inv = np.unique(groups, return_inverse=True)
            inv = np.asarray(inv).reshape(-1)
            pos_block = np.zeros(inv.max() + 1, dtype=bool)
            pos_block[inv[y == 1]] = True
            fold_of = np.zeros(pos_block.size, dtype=np.int64)
            fold_of[pos_block] = 1 + np.arange(int(pos_block.sum())) % (self.n_splits - 1)
            label = fold_of[inv]
            idx = np.arange(len(y))
            return [(idx[label != k], idx[label == k]) for k in range(self.n_splits)]
    return Fake


def test_repair_moves_whole_positive_block_after_retries_exhausted(monkeypatch):
    coords, y, groups = _split_data()
    fake = _fake_group_sgkf()
    monkeypatch.setattr(spatial, "StratifiedGroupKFold", fake)
    msgs = []
    splits = list(spatial.stratified_group_splits(y, groups, 5, 2, random_state=0, log_fn=msgs.append,
                                                  max_attempts=4))
    assert len(fake.calls) == 8                                     # har repeat uchun 4 urinish (hammasi yomon)
    assert len(msgs) == 2 and all("qayta taqsimlandi" in m for m in msgs)
    assert not any("Ogohlantirish" in m for m in msgs)              # tuzatildi: ogohlantirish kerak emas
    assert len(splits) == 10
    for r in range(2):
        vals = [va for rr, _, _, va in splits if rr == r]
        assert np.array_equal(np.sort(np.concatenate(vals)), np.arange(len(y)))
    for _, _, tr, va in splits:
        assert y[va].sum() >= 1
        assert not (_val_groups(groups, va) & _val_groups(groups, tr))


def test_repair_positive_folds_unit():
    #            A  A  B  C  D       (A: 2 musbat, B: 1, C: 1, D: fon)
    groups = np.array([0, 0, 1, 2, 3])
    y = np.array([1, 1, 1, 1, 0])
    idx = np.arange(5)
    # 0-fold: A,B,C; 1-fold: D (musbatsiz); 2-fold: bo'sh
    folds = [(idx[[4]], idx[[0, 1, 2, 3]]), (idx[[0, 1, 2, 3]], idx[[4]]), (idx, idx[:0])]
    fixed = spatial._repair_positive_folds(folds, y, groups)
    # donor 0-fold'dan eng kichik musbat bloklar (B, keyin C) butunlay ko'chadi
    assert [va.tolist() for _, va in fixed] == [[0, 1], [2, 4], [3]]
    for tr, va in fixed:
        assert y[va].sum() >= 1 and not (set(groups[tr]) & set(groups[va]))
        assert np.array_equal(np.sort(np.r_[tr, va]), idx)
    # blok bo'yicha izchil bo'lmagan fold'lar (A bloki 0- va 1-fold'ga bo'lingan) o'zgartirilmaydi
    split_block = [(idx[[1, 2, 3, 4]], idx[[0]]), (idx[[0, 2, 3, 4]], idx[[1]]),
                   (idx[[0, 1, 4]], idx[[2, 3]])]
    assert spatial._repair_positive_folds(split_block, y, groups) is split_block
    # tuzatish kerak bo'lmasa fold'lar bir xil qoladi
    ok = [(idx[[2, 3, 4]], idx[[0, 1]]), (idx[[0, 1, 3, 4]], idx[[2]]), (idx[[0, 1, 2]], idx[[3, 4]])]
    assert [va.tolist() for _, va in spatial._repair_positive_folds(ok, y, groups)] == [[0, 1], [2], [3, 4]]


def _fake_sgkf(good_from):
    """Dastlabki urinishlarda (seed % 1000 < good_from) bitta val fold'i musbatsiz chiqaradigan soxta SGKF."""
    class Fake:
        calls = []

        def __init__(self, n_splits, shuffle=False, random_state=None):
            self.n_splits, self.seed = n_splits, random_state
            Fake.calls.append(random_state)

        def split(self, X, y, groups):
            if self.seed % 1000 >= good_from:
                return StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.seed).split(X, y, groups)
            idx = np.arange(len(y))
            neg_first = idx[y == 0][:5]                 # 0-fold: faqat fon nuqtalar
            rest = np.setdiff1d(idx, neg_first)
            parts = [neg_first] + list(np.array_split(rest, self.n_splits - 1))
            return [(np.setdiff1d(idx, v), v) for v in parts]
    return Fake


def test_retry_with_new_seed_until_all_folds_have_positive(monkeypatch):
    coords, y, groups = _split_data()
    fake = _fake_sgkf(good_from=3)
    monkeypatch.setattr(spatial, "StratifiedGroupKFold", fake)
    msgs = []
    splits = list(spatial.stratified_group_splits(y, groups, 5, 2, random_state=0, log_fn=msgs.append))
    assert fake.calls == [0, 1, 2, 3, 1000, 1001, 1002, 1003]   # seed + repeat*1000 + urinish
    assert msgs == []                                            # oxirida muvaffaqiyat: ogohlantirish yo'q
    assert all(y[va].sum() >= 1 for _, _, _, va in splits)


def test_warns_and_uses_best_when_retries_exhausted(monkeypatch):
    coords, y, groups = _split_data()
    fake = _fake_sgkf(good_from=10 ** 6)                        # hech qachon yaxshi bo'lmaydi
    monkeypatch.setattr(spatial, "StratifiedGroupKFold", fake)
    msgs = []
    splits = list(spatial.stratified_group_splits(y, groups, 5, 2, random_state=0, log_fn=msgs.append,
                                                  max_attempts=4))
    assert len(fake.calls) == 8 and len(splits) == 10          # har repeat uchun 4 urinish, baribir davom etadi
    assert len(msgs) == 2 and all("Ogohlantirish" in m and "musbat" in m for m in msgs)


def test_default_max_attempts_is_50(monkeypatch):
    coords, y, groups = _split_data()
    fake = _fake_sgkf(good_from=10 ** 6)
    monkeypatch.setattr(spatial, "StratifiedGroupKFold", fake)
    list(spatial.stratified_group_splits(y, groups, 5, 1, random_state=0))
    assert len(fake.calls) == 50


# ---------------------------------------------------------------------------
# random_stratified_splits
# ---------------------------------------------------------------------------
def test_random_stratified_splits_format_and_determinism():
    _, y = _points()
    a = list(spatial.random_stratified_splits(y, 5, 3, random_state=7))
    b = list(spatial.random_stratified_splits(y, 5, 3, random_state=7))
    assert len(a) == 15
    assert [(r, f) for r, f, _, _ in a] == [(r, f) for r in range(3) for f in range(5)]
    for (_, _, t1, v1), (_, _, t2, v2), in zip(a, b):
        assert np.array_equal(t1, t2) and np.array_equal(v1, v2)
    for r, f, tr, va in a:
        assert np.intersect1d(tr, va).size == 0 and tr.size + va.size == len(y)
        assert y[va].sum() >= 1 and (y[va] == 0).sum() >= 1
    v0 = tuple(tuple(v) for r, f, t, v in a if r == 0)
    v1_ = tuple(tuple(v) for r, f, t, v in a if r == 1)
    assert v0 != v1_
    c = list(spatial.random_stratified_splits(y, 5, 3, random_state=8))
    assert any(not np.array_equal(x[3], z[3]) for x, z in zip(a, c))


def test_random_stratified_splits_errors():
    _, y = _points(n_pos=4, n_neg=50)
    with pytest.raises(ValueError, match="k-fold"):
        spatial.random_stratified_splits(y, 5, 2)
    _, y = _points()
    with pytest.raises(ValueError, match="n_splits"):
        spatial.random_stratified_splits(y, 1, 2)
    with pytest.raises(ValueError, match="n_repeats"):
        spatial.random_stratified_splits(y, 5, 0)


# ---------------------------------------------------------------------------
# block_bootstrap_indices
# ---------------------------------------------------------------------------
def test_block_bootstrap_whole_blocks_with_replacement():
    coords, y, groups = _split_data(block_size=2000.0)
    sizes = np.bincount(groups)
    n_blocks = len(sizes)
    boots = list(spatial.block_bootstrap_indices(groups, 50, random_state=0))
    assert len(boots) == 50
    n_unique_blocks = []
    for idx in boots:
        assert idx.min() >= 0 and idx.max() < len(groups)
        cnt = np.bincount(groups[idx], minlength=n_blocks)
        mult = cnt / sizes
        assert np.allclose(mult, np.round(mult))              # blok BUTUN va butun karra takrorlanadi
        # tanlangan bloklar soni (karralari bilan) = n_blocks
        assert int(np.round(mult).sum()) == n_blocks
        n_unique_blocks.append(int((mult > 0).sum()))
        # nuqta indekslari har bir tanlangan blokning barcha nuqtalarini o'z ichiga oladi
    assert max(n_unique_blocks) < n_blocks                    # almashtirish bilan: takrorlar bor
    assert np.mean(n_unique_blocks) < 0.8 * n_blocks          # ~63% kutiladi


def test_block_bootstrap_deterministic_and_seeded():
    coords, y, groups = _split_data(block_size=2000.0)
    a = list(spatial.block_bootstrap_indices(groups, 10, random_state=3))
    b = list(spatial.block_bootstrap_indices(groups, 10, random_state=3))
    c = list(spatial.block_bootstrap_indices(groups, 10, random_state=4))
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    assert any(x.shape != z.shape or not np.array_equal(x, z) for x, z in zip(a, c))
    assert any(not np.array_equal(a[0], x) for x in a[1:])    # bootstrap'lar o'zaro farq qiladi
    gen = spatial.block_bootstrap_indices(groups, 3)
    assert iter(gen) is gen


def test_block_bootstrap_noncontiguous_labels_and_errors():
    groups = np.array([5, 5, 9, 9, 9, 2, 2, 40])        # zich bo'lmagan yorliqlar, aralash tartib
    for idx in spatial.block_bootstrap_indices(groups, 20, random_state=1):
        cnt = {g: int((groups[idx] == g).sum()) for g in np.unique(groups)}
        for g, c in cnt.items():
            assert c % int((groups == g).sum()) == 0
    with pytest.raises(ValueError, match="n_boot"):
        spatial.block_bootstrap_indices(groups, 0)
    with pytest.raises(ValueError, match="bo'sh"):
        spatial.block_bootstrap_indices(np.array([], dtype=int), 5)


# ---------------------------------------------------------------------------
# fold_report
# ---------------------------------------------------------------------------
def test_fold_report():
    coords, y, groups = _split_data()
    splits = list(spatial.stratified_group_splits(y, groups, 5, 2, random_state=2))
    rep = spatial.fold_report(y, coords, groups, iter(splits))      # iterator ham, ro'yxat ham
    assert list(rep.columns) == ["repeat", "fold", "n_train", "n_val", "pos_train", "pos_val", "n_blocks_val"]
    assert len(rep) == 10
    assert (rep["n_train"] + rep["n_val"] == len(y)).all()
    assert (rep["pos_train"] + rep["pos_val"] == int(y.sum())).all()
    assert (rep["pos_val"] >= 1).all()
    assert rep.groupby("repeat")["n_val"].sum().eq(len(y)).all()
    for (r, f, tr, va), row in zip(splits, rep.itertuples()):
        assert (row.repeat, row.fold, row.n_val) == (r, f, len(va))
        assert row.n_blocks_val == len(_val_groups(groups, va))
    assert spatial.fold_report(y, coords, groups, splits).equals(rep)
    # groups yo'q (random CV): har nuqta alohida blok
    rnd = spatial.fold_report(y, None, None, spatial.random_stratified_splits(y, 5, 1))
    assert (rnd["n_blocks_val"] == rnd["n_val"]).all()
    assert spatial.fold_report(y, coords, groups, []).empty
    assert (spatial.fold_report(y, coords, groups, []).dtypes == "int64").all()     # bo'sh jadval ham int64
    with pytest.raises(ValueError, match="coords"):
        spatial.fold_report(y, coords[:-1], groups, splits)


# ---------------------------------------------------------------------------
# estimate_autocorrelation_range (variogram)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_variogram_reasonable_range_on_smooth_field(seed):
    # sigma=6 piksel (600 m): haqiqiy 95% masofa ~2000 m; sferik model range'i bundan biroz katta
    r = _rng_range(6, seed)
    assert r is not None
    assert 300.0 < r < 6000.0


def test_variogram_range_grows_with_smoothness():
    small = np.mean([_rng_range(3, s) for s in range(3)])
    mid = np.mean([_rng_range(6, s) for s in range(3)])
    large = np.mean([_rng_range(10, s) for s in range(3)])
    assert small < mid < large


def test_variogram_stable_across_random_state_and_n_points():
    band = _smooth_band(6, seed=0)[None]
    tr = _transform()
    vals = [spatial.estimate_autocorrelation_range(band, tr, random_state=s) for s in range(6)]
    assert (max(vals) - min(vals)) / np.median(vals) < 0.25
    few = spatial.estimate_autocorrelation_range(band, tr, n_points=600, random_state=0)
    many = spatial.estimate_autocorrelation_range(band, tr, n_points=3000, random_state=0)
    assert abs(few - many) / many < 0.25


def test_variogram_deterministic():
    band = _smooth_band(6, seed=1)[None]
    tr = _transform()
    a = spatial.estimate_autocorrelation_range(band, tr, random_state=5)
    b = spatial.estimate_autocorrelation_range(band, tr, random_state=5)
    assert a == b and isinstance(a, float)


def test_variogram_all_bands_median_ignores_structureless():
    rng = np.random.default_rng(0)
    fields = [_smooth_band(s, seed=i) for i, s in enumerate((3, 6, 10))]
    noise = rng.normal(size=fields[0].shape).astype("float32")
    const = np.full_like(fields[0], 7.0)
    stack = np.stack(fields + [noise, const])
    tr = _transform()
    msgs = []
    est = spatial.estimate_autocorrelation_range(stack, tr, random_state=11, log_fn=msgs.append)
    singles = [spatial.estimate_autocorrelation_range(stack, tr, band_indices=[i], random_state=11)
               for i in range(3)]
    assert all(s is not None for s in singles)
    assert est == pytest.approx(float(np.median(singles)))
    assert spatial.estimate_autocorrelation_range(stack, tr, band_indices=[3, 4], random_state=11) is None
    assert spatial.estimate_autocorrelation_range(stack, tr, band_indices=2, random_state=11) == singles[2]
    joined = "\n".join(msgs)
    assert "3/5 band" in joined and "e'tiborga olinmadi" in joined


def test_variogram_not_dependent_on_band_order_of_first_band():
    # legacy xatosi: faqat 1-band ishlatilardi. Endi 1-band shovqin bo'lsa ham qolganlari hisobga olinadi
    noise = np.random.default_rng(3).normal(size=(120, 120)).astype("float32")
    smooth = _smooth_band(6, seed=2)
    est = spatial.estimate_autocorrelation_range(np.stack([noise, smooth]), _transform(), random_state=0)
    assert est is not None and 300.0 < est < 6000.0


def test_variogram_none_cases():
    tr = _transform()
    noise = np.random.default_rng(0).normal(size=(1, 120, 120)).astype("float32")
    msgs = []
    assert spatial.estimate_autocorrelation_range(noise, tr, log_fn=msgs.append) is None   # sof shovqin
    assert msgs and "standart blok" in msgs[-1]
    assert spatial.estimate_autocorrelation_range(np.full((1, 120, 120), 2.0, "float32"), tr) is None
    assert spatial.estimate_autocorrelation_range(np.full((1, 120, 120), np.nan, "float32"), tr) is None
    few = np.full((1, 120, 120), np.nan, "float32")
    few[0, :5, :5] = np.random.default_rng(0).normal(size=(5, 5))
    assert spatial.estimate_autocorrelation_range(few, tr) is None                        # <50 valid piksel
    assert spatial.estimate_autocorrelation_range(_smooth_band(6)[None], tr, band_indices=[]) is None


def test_variogram_white_noise_not_reported_as_structure_for_many_seeds():
    # eski kodda sof shovqinning ~25% i 'struktura' deb chiqardi (range = birinchi lag yoki max_lag)
    tr = _transform()
    spurious = [s for s in range(20)
                if spatial.estimate_autocorrelation_range(
                    np.random.default_rng(s).normal(size=(1, 120, 120)).astype("float32"), tr,
                    random_state=s) is not None]
    assert spurious == []


def test_fit_spherical_flat_variogram_has_no_structure():
    # birinchi lagdan qisqa range'da model ustunlari bir xil: nugget/sill ajralmaydi, yassi variogram
    # 'struktura' bo'lib chiqmasligi kerak
    h = np.linspace(100.0, 1500.0, 12)
    a, nugget, psill = spatial._fit_spherical(h, np.ones(12), np.full(12, 500), 1600.0, min_range=300.0)
    assert a >= 300.0 and psill / (nugget + psill) < spatial._STRUCT_MIN


def test_variogram_nan_inf_robust():
    band = _smooth_band(6, seed=0)
    clean = spatial.estimate_autocorrelation_range(band[None], _transform(), random_state=0)
    dirty = band.copy()
    dirty[:12, :12] = np.nan
    dirty[50, 50] = np.inf
    dirty[60, 60] = -np.inf
    dirty[70, 70] = 1e30
    est = spatial.estimate_autocorrelation_range(dirty[None], _transform(), random_state=0)
    assert est is not None and np.isfinite(est)
    assert 300.0 < est < 6000.0
    assert abs(est - clean) / clean < 0.4


def test_variogram_trend_returns_finite_lower_bound():
    yy, xx = np.mgrid[0:120, 0:120]
    est = spatial.estimate_autocorrelation_range((xx + yy).astype("float32")[None], _transform())
    assert est is not None and np.isfinite(est) and est > 0


def test_variogram_argument_validation():
    tr = _transform()
    stack = _smooth_band(6)[None]
    with pytest.raises(ValueError, match="band_indices"):
        spatial.estimate_autocorrelation_range(stack, tr, band_indices=[1])
    with pytest.raises(ValueError, match="band_indices"):
        spatial.estimate_autocorrelation_range(stack, tr, band_indices=[-1])
    with pytest.raises(ValueError, match="stack"):
        spatial.estimate_autocorrelation_range(stack[0], tr)
    with pytest.raises(ValueError, match="n_lags"):
        spatial.estimate_autocorrelation_range(stack, tr, n_lags=2)


# ---------------------------------------------------------------------------
# Sintetik loyiha bilan integratsiya
# ---------------------------------------------------------------------------
def test_pipeline_like_flow_on_synth_project(synth_project):
    tiff_dir = synth_project["tiff"]
    bands = []
    for name in synth_project["layer_names"]:
        with rasterio.open(os.path.join(tiff_dir, f"{name}.tif")) as src:
            bands.append(src.read(1, masked=True).astype("float32").filled(np.nan))
            transform = src.transform
    stack = np.stack(bands)
    est = spatial.estimate_autocorrelation_range(stack, transform, random_state=RANDOM_STATE)
    assert est is not None and 300.0 < est < 6000.0

    pos = gpd.read_file(os.path.join(synth_project["points"], "pts.shp"))
    pos_xy = np.column_stack([pos.geometry.x.to_numpy(), pos.geometry.y.to_numpy()])
    rng = np.random.default_rng(0)
    ext = synth_project["size"] * synth_project["resolution"]
    neg_xy = rng.uniform(0, ext, size=(80, 2)) + np.array([X0, Y0])
    coords = np.vstack([pos_xy, neg_xy])
    y = np.r_[np.ones(len(pos_xy)), np.zeros(len(neg_xy))].astype(np.int8)

    bs, groups = spatial.adapt_block_size(coords, y, 5, est)
    splits = list(spatial.stratified_group_splits(y, groups, 5, 3, random_state=RANDOM_STATE))
    rep = spatial.fold_report(y, coords, groups, splits)
    assert len(rep) == 15 and (rep["pos_val"] >= 1).all()
    for _, _, tr, va in splits:
        assert not (_val_groups(groups, va) & _val_groups(groups, tr))
