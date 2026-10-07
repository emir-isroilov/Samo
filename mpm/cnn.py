# -*- coding: utf-8 -*-
"""
CNN modeli (Keras 3): ikki rejim.

  * mode="patch2d"  (ENH-03, tavsiya): nuqta atrofidagi WxW piksellik oyna (n, w, w, p) ustida 2D-Conv;
    per-kanal standartlashtirish (fit() ga berilgan train patchlaridan), NaN -> 0 (standartlashdan KEYIN);
    augmentatsiya: tasodifiy flip + rot90 (faqat train, har epoch yangi, seed bilan deterministik).
  * mode="tabular1d": eski usul - X (n, p) ni "spektr" deb qarab Conv1D (taqqoslash uchun).

O'qitish (BUG-04): stratified val_fraction ajratmasi, EarlyStopping(monitor="val_loss", restore_best_weights);
val ajratib bo'lmasa (har sinfdan >=2 namuna kerak) - ogohlantirish bilan train loss bo'yicha.
Musbat sinf uchun class_weight (neg/pos) namuna og'irligi sifatida beriladi (val_loss ham shu og'irlik bilan).
CV va final bir xil giperparametr bilan ishlaydi (epochs - maksimum, haqiqiy soni early stopping bilan).
Bashorat (BUG-07): batch_size=8192. Saqlash: model.keras + meta.json (joblib EMAS).

tensorflow modul darajasida import QILINMAYDI: common.get_tf() fit/load ichida chaqiriladi.
"""
from __future__ import annotations

import json
import math
import os
import threading

import numpy as np
from sklearn.model_selection import train_test_split

from .base import ModelWrapper
from .common import RANDOM_STATE, check_cancel, get_tf, noop_log
from .config import PARAM_SPECS

META_FORMAT = 1
MODEL_FILE = "model.keras"
META_FILE = "meta.json"
_STAT_CHUNK = 4096            # statistika hisoblash uchun qatorlar bo'yicha chunk
_CONST_REL = 1e-7             # std <= _CONST_REL * |mean| bo'lsa kanal o'zgarmas (float32 yaxlitlash shovqini)
_PREDICT_ELEMS = 2 ** 25      # predict chunk'ining taxminiy element soni (~128 MB float32)
_BUILD_LOCK = threading.Lock()   # clear_session + seed + qurish: global Keras/random holati oqimlararo bo'lishmasin


# ---------------------------------------------------------------------------
# Parametrlar
# ---------------------------------------------------------------------------
def _to_int(name, v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"CNN.{name}: butun son kerak, berilgan {v!r}") from None
    if not np.isfinite(f):
        raise ValueError(f"CNN.{name}: chekli son kerak, berilgan {v!r}")
    return int(round(f))


def _to_float(name, v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"CNN.{name}: son kerak, berilgan {v!r}") from None
    if not np.isfinite(f):
        raise ValueError(f"CNN.{name}: chekli son kerak, berilgan {v!r}")
    return f


def normalize_cnn_params(params):
    """To'liq (PARAM_SPECS["CNN"] dagi barcha kalitlar) va tekshirilgan lug'at.
    Yetishmaganlari standart bilan to'ldiriladi; noma'lum kalit yoki yaroqsiz qiymat - ValueError."""
    specs = {s.name: s for s in PARAM_SPECS["CNN"]}
    params = dict(params or {})
    unknown = sorted(set(params) - set(specs))
    if unknown:
        raise ValueError(f"CNN: noma'lum parametr(lar) {unknown}; mumkin: {sorted(specs)}")
    out = {}
    for name, spec in specs.items():
        v = params.get(name, spec.default)
        if spec.kind == "int":
            v = _to_int(name, v)
        elif spec.kind == "float":
            v = _to_float(name, v)
        elif spec.kind == "bool":
            if not isinstance(v, (bool, np.bool_, int, np.integer)):
                raise ValueError(f"CNN.{name}: bool kerak, berilgan {v!r}")
            v = bool(v)
        elif spec.kind == "choice":
            v = str(v)
            if v not in spec.choices:
                raise ValueError(f"CNN.{name}: '{v}' ruxsat etilmagan; mumkin: {list(spec.choices)}")
        out[name] = v
    mins = {"filters1": 1, "filters2": 0, "kernel_size": 1, "dense_units": 1, "batch_size": 1,
            "epochs": 1, "patience": 1, "window": 1}
    for name, lo in mins.items():
        if out[name] < lo:
            raise ValueError(f"CNN.{name} kamida {lo} bo'lishi kerak, berilgan {out[name]}")
    if out["mode"] == "patch2d" and out["window"] % 2 == 0:
        raise ValueError(f"CNN.window toq son bo'lishi kerak, berilgan {out['window']}")
    if not 0.0 <= out["dropout"] < 1.0:
        raise ValueError(f"CNN.dropout [0, 1) oralig'ida bo'lishi kerak, berilgan {out['dropout']}")
    if out["l2"] < 0.0:
        raise ValueError(f"CNN.l2 manfiy bo'lmasligi kerak, berilgan {out['l2']}")
    if out["learning_rate"] <= 0.0:
        raise ValueError(f"CNN.learning_rate musbat bo'lishi kerak, berilgan {out['learning_rate']}")
    if not 0.0 < out["val_fraction"] < 1.0:
        raise ValueError(f"CNN.val_fraction (0, 1) oralig'ida bo'lishi kerak, berilgan {out['val_fraction']}")
    return out


# ---------------------------------------------------------------------------
# Keras modelini qurish
# ---------------------------------------------------------------------------
def _make_optimizer(keras, name, lr):
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    raise ValueError(f"CNN.optimizer: noma'lum optimizator {name!r}")


def build_keras_model(params, n_channels, keras=None):
    """Kompilyatsiya qilingan Keras modeli. patch2d: kirish (w, w, n_channels); tabular1d: (n_channels, 1).
    params - normalize_cnn_params natijasi (yoki unga teng to'liq lug'at)."""
    if keras is None:
        tf = get_tf()
        if tf is None:
            raise RuntimeError("TensorFlow o'rnatilmagan: CNN modeli uchun 'pip install tensorflow' kerak.")
        keras = tf.keras
    p = normalize_cnn_params(params)
    L = keras.layers
    reg = keras.regularizers.L2(p["l2"]) if p["l2"] > 0 else None
    k = p["kernel_size"]
    if p["mode"] == "patch2d":
        w = p["window"]
        layer_list = [L.Input(shape=(w, w, int(n_channels))),
                      L.Conv2D(p["filters1"], k, padding="same", activation="relu", kernel_regularizer=reg)]
        if p["filters2"] > 0:
            layer_list.append(L.Conv2D(p["filters2"], k, padding="same", activation="relu",
                                       kernel_regularizer=reg))
        layer_list.append(L.GlobalAveragePooling2D())
    else:
        layer_list = [L.Input(shape=(int(n_channels), 1)),
                      L.Conv1D(p["filters1"], k, padding="same", activation="relu", kernel_regularizer=reg)]
        if p["filters2"] > 0:
            layer_list.append(L.Conv1D(p["filters2"], k, padding="same", activation="relu",
                                       kernel_regularizer=reg))
        layer_list.append(L.GlobalAveragePooling1D())
    layer_list += [L.Dense(p["dense_units"], activation="relu", kernel_regularizer=reg),
                   L.Dropout(p["dropout"]),
                   L.Dense(1, activation="sigmoid")]
    model = keras.Sequential(layer_list)
    # jit_compile=False: XLA har yangi batch shaklida qayta kompilyatsiya qiladi (kichik tarmoqda sekin)
    model.compile(optimizer=_make_optimizer(keras, p["optimizer"], p["learning_rate"]),
                  loss="binary_crossentropy", jit_compile=False)
    return model


# ---------------------------------------------------------------------------
# Standartlashtirish (numpy; TF kerak emas)
# ---------------------------------------------------------------------------
def _channel_stats(arr):
    """Oxirgi o'q (kanal) bo'yicha mean/std, faqat chekli qiymatlardan (nanmean/nanstd, ddof=0).
    Kanal o'zgarmas (std=0 yoki o'rtachaga nisbatan float32 yaxlitlash shovqini darajasida) yoki chekli qiymat
    yo'q bo'lsa std=1 (mean=0 agar qiymat yo'q). Mutlaq chegara YO'Q: kichik birlikdagi qatlamlar (Tesla, 1/s^2)
    ham to'g'ri standartlanadi."""
    n_ch = arr.shape[-1]
    axes = tuple(range(arr.ndim - 1))
    cnt = np.zeros(n_ch)
    tot = np.zeros(n_ch)
    for s in range(0, len(arr), _STAT_CHUNK):
        a = arr[s:s + _STAT_CHUNK].astype(np.float64)
        ok = np.isfinite(a)
        cnt += ok.sum(axis=axes)
        tot += np.where(ok, a, 0.0).sum(axis=axes)
    mean = np.divide(tot, cnt, out=np.zeros(n_ch), where=cnt > 0)
    sq = np.zeros(n_ch)
    for s in range(0, len(arr), _STAT_CHUNK):
        a = arr[s:s + _STAT_CHUNK].astype(np.float64)
        ok = np.isfinite(a)
        sq += np.where(ok, (a - mean) ** 2, 0.0).sum(axis=axes)
    std = np.sqrt(np.divide(sq, cnt, out=np.zeros(n_ch), where=cnt > 0))
    std[~np.isfinite(std) | (std <= _CONST_REL * np.abs(mean))] = 1.0
    mean32, std32 = mean.astype(np.float32), std.astype(np.float32)
    std32[std32 == 0] = 1.0                # float32 ga o'tganda yo'qolib ketgan (denormal) std
    return mean32, std32


def _standardize(arr, mean, std):
    """(arr - mean) / std, keyin chekli bo'lmaganlar (NaN/inf) = 0. float32 qaytaradi."""
    out = (np.asarray(arr, dtype=np.float32) - mean) / std
    return np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Batch manbasi va augmentatsiya (numpy; deterministik)
# ---------------------------------------------------------------------------
def _augment_batch(xb, rng):
    """(b, w, w, p) batchning har namunasiga tasodifiy rot90 (0-3 marta) va gorizontal/vertikal flip.
    Kvadrat toq oyna markaziy pikselni joyida qoldiradi."""
    b = len(xb)
    k = rng.integers(0, 4, size=b)
    flip_h = rng.random(b) < 0.5
    flip_v = rng.random(b) < 0.5
    out = xb.copy()
    for r in (1, 2, 3):
        m = k == r
        if m.any():
            out[m] = np.rot90(out[m], r, axes=(1, 2))
    if flip_h.any():
        out[flip_h] = out[flip_h][:, :, ::-1]
    if flip_v.any():
        out[flip_v] = out[flip_v][:, ::-1]
    return out


class _BatchSource:
    """Train batchlari: har epoch'da namunalar yangidan aralashtiriladi, (seed, epoch, batch) bo'yicha
    deterministik. Chaqiruv tartibiga bog'liq emas (Keras birinchi batchni oldindan ko'rishi mumkin)."""

    def __init__(self, X, y, sw, batch_size, seed, augment):
        self.X, self.y, self.sw = X, y, sw
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.augment = bool(augment)
        self._cache = (-1, None)

    def __len__(self):
        return int(math.ceil(len(self.y) / self.batch_size))

    def _order(self, epoch):
        cached_epoch, order = self._cache
        if cached_epoch != epoch:
            order = np.random.default_rng([self.seed, epoch]).permutation(len(self.y))
            self._cache = (epoch, order)
        return order

    def batch(self, epoch, i):
        sel = self._order(epoch)[i * self.batch_size:(i + 1) * self.batch_size]
        xb = self.X[sel]
        if self.augment:
            xb = _augment_batch(xb, np.random.default_rng([self.seed, epoch, i]))
        return xb, self.y[sel], self.sw[sel]


# ---------------------------------------------------------------------------
# Keras sinflari (TF lazy import tufayli funksiya ichida qurilib, keshlanadi)
# ---------------------------------------------------------------------------
_KERAS_CLASSES = {}


def _keras_classes(keras):
    if not _KERAS_CLASSES:
        class TrainSequence(keras.utils.PyDataset):
            def __init__(self, source):
                super().__init__(workers=0)
                self.source = source
                self.epoch = 0            # EpochSync har epoch boshida o'rnatadi

            def __len__(self):
                return len(self.source)

            def __getitem__(self, i):
                return self.source.batch(self.epoch, i)

        class EpochSync(keras.callbacks.Callback):
            def __init__(self, seq):
                super().__init__()
                self.seq = seq

            def on_epoch_begin(self, epoch, logs=None):
                self.seq.epoch = int(epoch)

        class CancelCallback(keras.callbacks.Callback):
            def __init__(self, cancel):
                super().__init__()
                self.cancel = cancel

            def _check(self):
                if self.cancel is not None and self.cancel.is_cancelled:
                    self.model.stop_training = True

            def on_train_batch_end(self, batch, logs=None):
                self._check()

            def on_epoch_end(self, epoch, logs=None):
                self._check()

        _KERAS_CLASSES.update(TrainSequence=TrainSequence, EpochSync=EpochSync, CancelCallback=CancelCallback)
    return _KERAS_CLASSES


def _clear_session(keras):
    """Keras global holatini (nom hisoblagichlari, grafik keshlari, oldingi modellar chiqindisi) tozalaydi.
    Allaqachon o'qitilgan modellar bashorati ta'sirlanmaydi (tekshirilgan); xatolik o'qitishni to'xtatmasin."""
    try:
        keras.backend.clear_session()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CNNModel
# ---------------------------------------------------------------------------
class CNNModel(ModelWrapper):
    """Patch-CNN (input_kind="patch") yoki 1D-CNN (input_kind="tabular").

    n_jobs faqat saqlanadi (meta.json'ga yoziladi): TensorFlow o'z oqimlarini o'zi boshqaradi, n_jobs ta'sir qilmaydi."""
    name = "CNN"

    def __init__(self, params, seed=RANDOM_STATE, n_jobs=1):
        p = normalize_cnn_params(params)
        super().__init__("CNN", p, seed=seed, n_jobs=n_jobs)
        self.input_kind = "patch" if p["mode"] == "patch2d" else "tabular"
        self.model_ = None
        self.mean_ = None            # (n_features,) float32 - fit() dagi train statistikasi
        self.std_ = None
        self.n_features_ = None

    @property
    def _seed32(self):
        return self.seed % (2 ** 32 - 1)

    # ---------------------------------------------------------------- kirishni tekshirish
    def _inputs(self, X, patches):
        """Kirish massivi (n, w, w, p) yoki (n, p) float32 (nusxasiz, agar allaqachon float32)."""
        p = self.params
        if p["mode"] == "patch2d":
            if patches is None:
                raise ValueError("CNN (patch2d) uchun patches (n, w, w, p) kerak; patches=None berildi.")
            arr = np.asarray(patches, dtype=np.float32)
            if arr.ndim != 4:
                raise ValueError(f"patches 4 o'lchamli (n, w, w, p) bo'lishi kerak, berilgan shakl: {arr.shape}")
            w = p["window"]
            if arr.shape[1] != w or arr.shape[2] != w:
                raise ValueError(f"patches oynasi {arr.shape[1]}x{arr.shape[2]} params['window']={w} ga mos emas "
                                 f"(kutilgan {w}x{w}).")
        else:
            if X is None:
                raise ValueError("CNN (tabular1d) uchun X (n, p) kerak; X=None berildi.")
            arr = np.asarray(X, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"X 2 o'lchamli (n, p) bo'lishi kerak, berilgan shakl: {arr.shape}")
        return arr

    def _prepare(self, arr, mean, std):
        """Standartlashtirilgan batch; tabular1d uchun oxirgi o'q (kanal=1) qo'shiladi."""
        z = _standardize(arr, mean, std)
        return z[..., None] if self.params["mode"] == "tabular1d" else z

    # ---------------------------------------------------------------- val ajratmasi
    def _split(self, y, log):
        """(train_idx, val_idx) stratified; ajratib bo'lmasa (train_idx=hammasi, None) + ogohlantirish."""
        n = len(y)
        idx = np.arange(n)
        n_val = int(math.ceil(n * self.params["val_fraction"]))
        counts = np.bincount(y, minlength=2)
        reason = None
        if counts.min() < 2:
            reason = f"kam sinfda atigi {int(counts.min())} namuna bor (har sinfdan >=2 kerak)"
        elif n_val < 2 or n - n_val < 2:
            reason = f"namunalar juda kam (n={n}, val={n_val})"
        else:
            try:
                tr, va = train_test_split(idx, test_size=n_val, stratify=y, random_state=self._seed32)
            except ValueError as e:
                reason = str(e)
            else:
                if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
                    reason = "validatsiyada yoki train'da bitta sinf qoldi"
                else:
                    return np.sort(tr), np.sort(va)
        log(f"OGOHLANTIRISH: CNN uchun validatsiya ajratib bo'lmadi ({reason}); "
            f"early stopping train loss bo'yicha ishlaydi.")
        return idx, None

    # ---------------------------------------------------------------- fit
    def fit(self, X, y, patches=None, cancel=None, log_fn=None):
        log = log_fn or noop_log
        p = self.params
        tf = get_tf()
        if tf is None:
            raise RuntimeError("TensorFlow o'rnatilmagan: CNN modeli uchun 'pip install tensorflow' kerak.")
        keras = tf.keras
        check_cancel(cancel)

        y = np.asarray(y).ravel()
        if not np.isin(y, (0, 1)).all():
            raise ValueError("CNN: y faqat 0/1 qiymatlardan iborat bo'lishi kerak.")
        y = y.astype(np.int64)
        arr = self._inputs(X, patches)
        if arr.shape[0] != len(y):
            raise ValueError(f"CNN: X/patches qatorlari ({arr.shape[0]}) y uzunligiga ({len(y)}) teng emas.")
        counts = np.bincount(y, minlength=2)
        if counts.min() == 0:
            raise ValueError("CNN: o'qitish uchun ikkala sinf (0 va 1) ham kerak; "
                             f"berilgan: manfiy={int(counts[0])}, musbat={int(counts[1])}.")

        # per-kanal statistika - fit() ga berilgan (train) ma'lumotdan; val ham shu bilan standartlanadi
        mean, std = _channel_stats(arr)
        Z = self._prepare(arr, mean, std)
        yf = y.astype(np.float32)

        tr, va = self._split(y, log)
        n_pos_tr = int(y[tr].sum())
        pos_w = float((len(tr) - n_pos_tr) / max(n_pos_tr, 1))
        sw = np.where(y == 1, pos_w, 1.0).astype(np.float32)
        has_val = va is not None
        monitor = "val_loss" if has_val else "loss"

        # clear_session keyingi modelni qurishdan OLDIN (o'qitilgan modellar bashorati buzilmaydi - tekshirilgan),
        # keyin seed: clear_session global seed generatorini qayta tiklaydi. Qulf: parallel oqimlarda (joblib
        # threading) qatlam nomlari to'qnashmasin va boshlang'ich vaznlar/dropout faqat o'z seed'iga bog'liq bo'lsin.
        n_ch = arr.shape[-1]
        with _BUILD_LOCK:
            _clear_session(keras)
            keras.utils.set_random_seed(self._seed32)
            model = build_keras_model(p, n_ch, keras=keras)
        cls = _keras_classes(keras)
        source = _BatchSource(Z[tr], yf[tr], sw[tr], p["batch_size"], self._seed32,
                              augment=p["augment"] and p["mode"] == "patch2d")
        seq = cls["TrainSequence"](source)
        callbacks = [cls["EpochSync"](seq), cls["CancelCallback"](cancel),
                     keras.callbacks.EarlyStopping(monitor=monitor, patience=p["patience"],
                                                   restore_best_weights=True)]
        val_data = (Z[va], yf[va], sw[va]) if has_val else None
        hist = model.fit(seq, validation_data=val_data, epochs=p["epochs"], shuffle=False, verbose=0,
                         callbacks=callbacks).history
        check_cancel(cancel)

        curve = np.asarray(hist[monitor], dtype=float)
        if not np.isfinite(curve).any():
            raise RuntimeError(f"CNN o'qitishda {monitor} NaN/cheksiz bo'ldi: learning_rate ni kamaytiring.")
        best = int(np.nanargmin(np.where(np.isfinite(curve), curve, np.nan)))
        self.model_ = model
        self.mean_, self.std_ = mean, std
        self.n_features_ = int(n_ch)
        self.is_fitted = True
        self.fit_info = {
            "epochs_run": int(len(curve)), "best_epoch": best + 1,
            "val_loss": float(curve[best]) if has_val else None,
            "train_loss": float(hist["loss"][best]),
            "n_train": int(len(tr)), "n_val": int(len(va)) if has_val else 0,
            "early_stopping": "val_loss" if has_val else "train_loss",
            "class_weight_pos": pos_w,
        }
        log(f"CNN ({p['mode']}): {len(curve)}/{p['epochs']} epoch, eng yaxshi epoch={best + 1}, "
            f"{monitor}={curve[best]:.4f}, n_train={len(tr)}, n_val={self.fit_info['n_val']}.")
        return self

    # ---------------------------------------------------------------- predict
    def predict_proba_pos(self, X=None, patches=None, batch_size=8192):
        if not self.is_fitted or self.model_ is None:
            raise RuntimeError("CNN modeli hali o'qitilmagan (fit() yoki load() kerak).")
        arr = self._inputs(X, patches)
        if arr.shape[-1] != self.n_features_:
            raise ValueError(f"Kirish kanallari/feature'lari soni ({arr.shape[-1]}) o'qitishdagi "
                             f"({self.n_features_}) ga teng emas.")
        n = arr.shape[0]
        bs = int(batch_size)
        if bs < 1:
            raise ValueError(f"batch_size musbat bo'lishi kerak, berilgan {batch_size}")
        out = np.empty(n, dtype=np.float64)
        if n == 0:
            return out
        per_row = max(1, int(np.prod(arr.shape[1:])))
        chunk = max(bs, (_PREDICT_ELEMS // per_row) // bs * bs)       # bs ga karrali: batchlar bir xil
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            z = self._prepare(arr[s:e], self.mean_, self.std_)
            pred = self.model_.predict(z, batch_size=bs, verbose=0)
            out[s:e] = np.asarray(pred, dtype=np.float64).reshape(-1)
        return out

    # ---------------------------------------------------------------- saqlash / yuklash
    def save(self, directory):
        """directory/model.keras + directory/meta.json. model.keras yo'lini qaytaradi."""
        if not self.is_fitted or self.model_ is None:
            raise RuntimeError("O'qitilmagan CNN modelini saqlab bo'lmaydi.")
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, MODEL_FILE)
        import warnings
        with warnings.catch_warnings():
            # keras 3 + numpy 2: ichki __array__(copy=) DeprecationWarning (kutubxona ichida, bizning xato emas)
            warnings.filterwarnings("ignore", message=".*__array__ implementation.*", category=DeprecationWarning)
            self.model_.save(model_path)
        meta = {"format": META_FORMAT, "params": self.params, "seed": self.seed, "n_jobs": self.n_jobs,
                "mode": self.params["mode"], "n_features": self.n_features_,
                "mean": [float(v) for v in self.mean_], "std": [float(v) for v in self.std_],
                "fit_info": self.fit_info}
        with open(os.path.join(directory, META_FILE), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return model_path

    @classmethod
    def load(cls, directory):
        model_path = os.path.join(directory, MODEL_FILE)
        meta_path = os.path.join(directory, META_FILE)
        for path in (model_path, meta_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"CNN fayli topilmadi: {path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format") != META_FORMAT:
            raise ValueError(f"CNN meta.json formati qo'llab-quvvatlanmaydi: {meta.get('format')!r}")
        tf = get_tf()
        if tf is None:
            raise RuntimeError("TensorFlow o'rnatilmagan: CNN modelini yuklab bo'lmaydi.")
        obj = cls(meta["params"], seed=meta.get("seed", RANDOM_STATE), n_jobs=meta.get("n_jobs", 1))
        obj.model_ = tf.keras.models.load_model(model_path, compile=False)
        obj.mean_ = np.asarray(meta["mean"], dtype=np.float32)
        obj.std_ = np.asarray(meta["std"], dtype=np.float32)
        obj.n_features_ = int(meta["n_features"])
        obj.fit_info = dict(meta.get("fit_info") or {})
        obj.is_fitted = True
        return obj
