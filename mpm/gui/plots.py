# -*- coding: utf-8 -*-
"""
Natijalar grafiklari (Qt'siz): faqat `matplotlib.figure.Figure` bilan ishlaydi (Agg'da ham, FigureCanvasQTAgg'da ham).

Har bir `draw_*` funksiyasi:
  * birinchi argument sifatida `Figure` oladi, uni tozalaydi (`fig.clear()`) va o'zi chizadi, `fig`ni qaytaradi;
  * bo'sh/None/yetarli bo'lmagan ma'lumotda xato BERMAYDI - figuraga "Ma'lumot yo'q" yozadi (BUG-01);
  * kutilmagan istisno bo'lsa ham slotni qulatmaydi (xato figuraga yoziladi va logga ketadi);
  * constrained layout ishlatadi: suptitle/legend/colorbar kesilmaydi, legend ma'lumot ustiga tushmaydi (BUG-14).

Terminologiya: chiqish "ehtimollik" emas, "prospektivlik indeksi (0-1)" (musbat:fon nisbati sun'iy).
`pyplot` import QILINMAYDI (global holat yo'q); PyQt5 ham kerak emas.
"""
from __future__ import annotations

import functools
import logging
import math
import os
import textwrap
import warnings

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from ..common import ENSEMBLE_NAME, noop_log, safe_name
from ..explain import normalize_shap_values

__all__ = [
    "INDEX_LABEL", "draw_roc", "draw_pr", "draw_calibration", "draw_confusion", "draw_spatial_diagnostics",
    "draw_importance", "draw_shap_beeswarm", "draw_shap_dependence", "draw_corr_heatmap", "draw_map",
    "draw_success_rate", "draw_tuning_trials", "save_all_figures", "no_data",
]

_log = logging.getLogger("mpm.plots")

INDEX_LABEL = "Prospektivlik indeksi (0-1)"      # colorbar/legend uchun yagona yorliq (ehtimollik EMAS)
NO_DATA = "Ma'lumot yo'q"
TOP_N = 20                                        # feature'lar ko'p bo'lganda ko'rsatiladigan eng muhim soni
TOP_N_THRESHOLD = 30                              # shundan ko'p feature bo'lsa top-N bilan cheklanadi
HIGH_CORR = 0.9

_MODEL_COLORS = {"RandomForest": "#0072B2", "SVM": "#E69F00", "XGBoost": "#009E73", "CNN": "#CC79A7"}
_FALLBACK_COLORS = ("#56B4E9", "#D55E00", "#F0E442", "#8C564B", "#7F7F7F", "#17BECF")
_ENS_COLOR = "#111111"
_RANDOM_COLOR = "#9AA0A6"
_SPATIAL_COLOR = "#B2182B"
_TAB20 = [mcolors.to_hex(c) for c in colormaps["tab20"].colors]
_FOLD_COLORS = _TAB20[0::2] + _TAB20[1::2]        # 20 ta noyob rang (to'qlari birinchi)


class _NoData(Exception):
    """Chizish uchun ma'lumot yetarli emas (xato emas): figuraga "Ma'lumot yo'q" yoziladi."""


# ---------------------------------------------------------------------------
# Umumiy yordamchilar
# ---------------------------------------------------------------------------
def _prepare(fig):
    """Figurani tozalaydi va constrained layout yoqadi (suptitle/legend/colorbar kesilmasligi uchun)."""
    fig.clear()
    try:
        fig.set_layout_engine("constrained")
        fig.get_layout_engine().set(w_pad=0.05, h_pad=0.05, wspace=0.04, hspace=0.06)
    except Exception:                                        # eski matplotlib
        try:
            fig.set_constrained_layout(True)
        except Exception:
            pass
    return fig


def no_data(fig, msg=NO_DATA, detail=None, title=None):
    """Figurani tozalab, markazga "Ma'lumot yo'q" (va ixtiyoriy izoh/sarlavha) yozadi."""
    _prepare(fig)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    if title:
        ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=11, fontweight="bold",
                transform=ax.transAxes)
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=13, color="#555555", transform=ax.transAxes)
    if detail:
        ax.text(0.5, 0.40, textwrap.fill(str(detail), 70), ha="center", va="top", fontsize=8,
                color="#777777", transform=ax.transAxes)
    return fig


def _safe(fn):
    """draw_* dekoratori: _NoData => "Ma'lumot yo'q"; boshqa istisno => logga + figuraga yoziladi (slot qulamaydi)."""
    @functools.wraps(fn)
    def wrapper(fig, *args, **kwargs):
        try:
            fn(fig, *args, **kwargs)
        except _NoData as e:
            try:
                no_data(fig, detail=str(e) or None, title=_titles.get(fn.__name__))
            except Exception:
                pass
        except Exception as e:                                # noqa: BLE001 - GUI slotida abort bo'lmasin (BUG-01)
            _log.warning("%s xato berdi: %s: %s", fn.__name__, type(e).__name__, e, exc_info=True)
            try:
                no_data(fig, detail=f"Chizishda xato: {type(e).__name__}: {e}", title=_titles.get(fn.__name__))
            except Exception:
                pass
        return fig
    return wrapper


_titles = {
    "draw_roc": "ROC egri chiziqlari", "draw_pr": "Precision-Recall", "draw_calibration": "Kalibrlash",
    "draw_confusion": "Chalkashlik matritsasi", "draw_spatial_diagnostics": "Spatial CV diagnostikasi",
    "draw_importance": "Feature importance", "draw_shap_beeswarm": "SHAP beeswarm",
    "draw_shap_dependence": "SHAP dependence", "draw_corr_heatmap": "Korrelyatsiya matritsasi",
    "draw_map": "Xarita", "draw_success_rate": "Success-rate egri chizig'i",
    "draw_tuning_trials": "Giperparametr qidiruvi",
}


def _f(x, default=float("nan")):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v


def _arr(x):
    """1D float massiv (None/yaroqsiz => bo'sh)."""
    if x is None:
        return np.empty(0)
    try:
        a = np.asarray(x, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return np.empty(0)
    return a


def _ci(m, key="auc_ci95"):
    ci = m.get(key)
    try:
        lo, hi = float(ci[0]), float(ci[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return float("nan"), float("nan")
    return lo, hi


def _is_ens(name):
    return str(name) == ENSEMBLE_NAME or "ensemble" in str(name).lower()


def _color(name, i=0):
    if _is_ens(name):
        return _ENS_COLOR
    return _MODEL_COLORS.get(str(name), _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])


def _models_dict(metrics):
    """compute_metrics natijasi ({model: {...}}), (natija, mean_proba) kortej yoki CVBlock/run_cv lug'ati => {model: {...}}."""
    if isinstance(metrics, tuple) and metrics:
        metrics = metrics[0]
    if not isinstance(metrics, dict):
        return {}
    inner = metrics.get("metrics")
    if isinstance(inner, dict) and ("oof" in metrics or "mode" in metrics or "metrics_df" in metrics):
        metrics = inner                                      # CVBlock
    return {str(k): v for k, v in metrics.items() if isinstance(v, dict) and v}


def _wrap_lines(text, width):
    """Har qatorni `width` belgidan oshmasligi uchun bo'ladi (bo'sh qatorlar saqlanadi)."""
    res = []
    for line in str(text).split("\n"):
        res.extend(textwrap.wrap(line, width, break_long_words=True) or [""])
    return "\n".join(res)


def _set_title(ax, text, fontsize=10):
    """Axes sarlavhasi: uzun matn axes kengligiga qarab qatorlarga bo'linadi (tor oynada kesilmaydi)."""
    try:
        gs = ax.get_subplotspec().get_gridspec()
        ratios = list(gs.get_width_ratios() or [1] * gs.ncols)
        share = sum(ratios[c] for c in ax.get_subplotspec().colspan) / sum(ratios)
        w_in = float(ax.figure.get_figwidth()) * share * 0.85
        text = _wrap_lines(text, max(14, int(w_in * 72 / (fontsize * 0.58))))
    except Exception:                                        # noqa: BLE001 - gridspec'siz axes
        pass
    return ax.set_title(text, fontsize=fontsize)


def _suptitle(fig, text, fontsize=11):
    """Figura sarlavhasi: figura kengligiga sig'maydigan matn qatorlarga bo'linadi."""
    try:
        text = _wrap_lines(text, max(20, int(float(fig.get_figwidth()) * 0.92 * 72 / (fontsize * 0.58))))
    except Exception:                                        # noqa: BLE001
        pass
    return fig.suptitle(text, fontsize=fontsize)


def _footnote(fig, text, width=130):
    """Figura pastida kichik izoh (supxlabel constrained layout bilan ishlaydi)."""
    try:
        width = max(30, min(width, int(float(fig.get_figwidth()) * 17)))     # tor figurada izoh kesilmasin
        fig.supxlabel(textwrap.fill(text, width), fontsize=7, color="#666666")
    except Exception:
        pass


def _legend_outside(fig, ax, handles, where="lower", **kw):
    """Legend'ni axes tashqarisida (pastda yoki o'ngda) joylaydi - ma'lumot ustiga tushmaydi."""
    if not handles:
        return None
    loc = {"lower": "outside lower center", "right": "outside center right"}[where]
    kw.setdefault("fontsize", 8)
    kw.setdefault("frameon", False)
    try:
        return fig.legend(handles=handles, loc=loc, **kw)
    except Exception:                                        # eski matplotlib: "outside" yo'q
        if where == "lower":
            return ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), **kw)
        return ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), **kw)


_TRANSPARENT = (0.0, 0.0, 0.0, 0.0)


def _with_bad(cmap, bad):
    """Rang xaritasi nusxasi: NaN/maskalangan piksellar `bad` rangda (set_bad kelajakda eskiradi)."""
    try:
        return cmap.with_extremes(bad=bad)
    except AttributeError:                                   # juda eski matplotlib
        c = cmap.copy()
        c.set_bad(bad)
        return c


def _disp(name, n=18):
    """Qisqa ko'rsatiladigan model nomi (o'q belgilari uchun)."""
    return "Ansambl" if _is_ens(name) else _short(name, n)


def _short(s, n=28):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


def _names_for(p, names):
    if names is not None:
        names = [str(x) for x in names]
        if len(names) == p:
            return names
    return [f"feature_{i + 1}" for i in range(p)]


def _pick_model(ms, model):
    """model nomi bo'yicha lug'at yozuvi; None/topilmasa ansambl yoki birinchi."""
    if model is not None and str(model) in ms:
        return str(model), ms[str(model)]
    if model is not None:
        raise _NoData(f"'{model}' modeli natijalarda yo'q (mavjud: {', '.join(ms) or '-'}).")
    for k in ms:
        if _is_ens(k):
            return k, ms[k]
    k = next(iter(ms))
    return k, ms[k]


# ---------------------------------------------------------------------------
# ENH-04: ROC, PR, kalibrlash, chalkashlik matritsasi
# ---------------------------------------------------------------------------
@_safe
def draw_roc(fig, metrics, title_suffix=""):
    """ROC egri chiziqlari (mean OOF bashorat); legend'da AUC±std (repeat) va blok-bootstrap 95% CI, ansambl qalin.
    Eslatma: nuqta AUC repeat-o'rtacha, CI esa o'rtacha bashorat ustida bootstrap - nuqta qiymat CI chetiga yaqin
    bo'lishi mumkin."""
    ms = {k: m for k, m in _models_dict(metrics).items()
          if _arr(m.get("fpr")).size > 1 and _arr(m.get("fpr")).size == _arr(m.get("tpr")).size}
    if not ms:
        raise _NoData("ROC uchun fpr/tpr yo'q.")
    _prepare(fig)
    ax = fig.add_subplot(111)
    for i, (name, m) in enumerate(ms.items()):
        auc, std = _f(m.get("auc")), _f(m.get("auc_std"))
        lo, hi = _ci(m)
        label = f"{name}: AUC = {auc:.3f}"
        if np.isfinite(std):
            label += f" ± {std:.3f}"
        if np.isfinite(lo) and np.isfinite(hi):
            label += f" (95% CI {lo:.3f}–{hi:.3f})"
        ens = _is_ens(name)
        ax.plot(_arr(m["fpr"]), _arr(m["tpr"]), lw=3.0 if ens else 1.6, color=_color(name, i),
                zorder=3 if ens else 2, label=label)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="Tasodifiy (AUC = 0.5)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.03)
    ax.set_xlabel("False Positive Rate (1 − spetsifiklik)")
    ax.set_ylabel("True Positive Rate (sensitivlik)")
    title = "ROC egri chiziqlari (takroriy out-of-fold CV)"
    if title_suffix:
        title += f" — {title_suffix}"
    _set_title(ax, title, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, title="AUC ± std (takrorlar), 95% CI (blok-bootstrap)",
              title_fontsize=7, framealpha=0.9)
    _footnote(fig, "Egri chiziq - takrorlar bo'yicha o'rtacha bashorat; AUC - takrorlar o'rtachasi, CI - shu o'rtacha "
                   "bashorat ustida blok-bootstrap (nuqta qiymat CI chetiga yaqin bo'lishi mumkin).")


def _prevalence(y, ms):
    if y is not None:
        ya = _arr(y)
        ya = ya[np.isfinite(ya)]
        if ya.size:
            return float(np.mean(ya == 1))
    for m in ms.values():                                    # zaxira: chalkashlik matritsasidan
        try:
            (tn, fp), (fn, tp) = np.asarray(m["confusion"], dtype=float).reshape(2, 2)
            tot = tn + fp + fn + tp
            if tot > 0:
                return float((fn + tp) / tot)
        except (KeyError, TypeError, ValueError):
            continue
    return float("nan")


@_safe
def draw_pr(fig, metrics, y):
    """Precision-Recall egri chiziqlari + baseline (namunadagi musbat ulushi; y kerak)."""
    ms = {k: m for k, m in _models_dict(metrics).items()
          if _arr(m.get("recall")).size > 1 and _arr(m.get("recall")).size == _arr(m.get("precision")).size}
    if not ms:
        raise _NoData("PR uchun precision/recall yo'q.")
    _prepare(fig)
    ax = fig.add_subplot(111)
    for i, (name, m) in enumerate(ms.items()):
        ap, std = _f(m.get("pr_auc")), _f(m.get("pr_auc_std"))
        label = f"{name}: PR-AUC = {ap:.3f}" + (f" ± {std:.3f}" if np.isfinite(std) else "")
        lo, hi = _ci(m, "pr_auc_ci95")
        if np.isfinite(lo) and np.isfinite(hi):
            label += f" (95% CI {lo:.3f}–{hi:.3f})"
        ens = _is_ens(name)
        ax.plot(_arr(m["recall"]), _arr(m["precision"]), lw=3.0 if ens else 1.6, color=_color(name, i),
                zorder=3 if ens else 2, label=label)
    pi = _prevalence(y, ms)
    if np.isfinite(pi):
        ax.axhline(pi, ls="--", color="gray", lw=1, label=f"Baseline (namunadagi musbat ulushi = {pi:.2f})")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.03)
    ax.set_xlabel("Recall (sensitivlik)")
    ax.set_ylabel("Precision")
    _set_title(ax, "Precision-Recall egri chiziqlari (out-of-fold)", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    _footnote(fig, "Namunadagi musbat:fon nisbati sun'iy (fon nuqtalar soni tanlanadi), shuning uchun baseline va "
                   "Precision haqiqiy hudud bo'yicha absolyut ehtimollikni bildirmaydi.")


@_safe
def draw_calibration(fig, metrics):
    """Reliability diagrammasi: bin'dagi o'rtacha bashorat (prob_pred) va kuzatilgan musbat ulushi (prob_true).
    Bin soni o'zgaruvchan (3..10); diagonal = ideal kalibrlash."""
    ms = {}
    for k, m in _models_dict(metrics).items():
        c = m.get("calibration")
        if not isinstance(c, dict):
            continue
        pp, pt = _arr(c.get("prob_pred")), _arr(c.get("prob_true"))
        if pp.size >= 1 and pp.size == pt.size:
            ms[k] = (m, pp, pt)
    if not ms:
        raise _NoData("Kalibrlash (prob_pred/prob_true) yo'q.")
    _prepare(fig)
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="Ideal (y = x)")
    for i, (name, (m, pp, pt)) in enumerate(ms.items()):
        brier = _f(m.get("brier"))
        label = f"{name} ({pp.size} bin" + (f", Brier = {brier:.3f}" if np.isfinite(brier) else "") + ")"
        ens = _is_ens(name)
        ax.plot(pp, pt, marker="o", ms=5 if ens else 4, lw=2.6 if ens else 1.4, color=_color(name, i),
                zorder=3 if ens else 2, label=label)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Bin'dagi o'rtacha bashorat (prospektivlik indeksi, kvantil bin'lar)")
    ax.set_ylabel("Namunada kuzatilgan musbat ulushi")
    _set_title(ax, "Kalibrlash (reliability) diagrammasi", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    _footnote(fig, "Musbat:fon nisbati sun'iy bo'lgani uchun indeks mutlaq ehtimollik emas; diagramma faqat "
                   "namuna ichidagi tartib/shkala mosligini ko'rsatadi.")


@_safe
def draw_confusion(fig, metrics, model=None):
    """Youden bo'sag'idagi 2x2 chalkashlik matritsasi (sonlar + qator %), sarlavhada sensitivlik/spetsifiklik."""
    ms = _models_dict(metrics)
    if not ms:
        raise _NoData("Metrikalar yo'q.")
    name, m = _pick_model(ms, model)
    try:
        cm = np.asarray(m["confusion"], dtype=np.float64).reshape(2, 2)
    except (KeyError, TypeError, ValueError):
        raise _NoData(f"{name} uchun chalkashlik matritsasi yo'q.") from None
    if not np.isfinite(cm).all():
        raise _NoData(f"{name}: chalkashlik matritsasi chekli emas.")
    (tn, fp), (fn, tp) = cm
    sens = _f(m.get("sensitivity"), tp / (tp + fn) if tp + fn > 0 else float("nan"))
    spec = _f(m.get("specificity"), tn / (tn + fp) if tn + fp > 0 else float("nan"))
    thr = _f(m.get("threshold_youden"))
    _prepare(fig)
    ax = fig.add_subplot(111)
    row = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0) * 100.0
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    for i in range(2):
        for j in range(2):
            col = "white" if pct[i, j] > 55 else "#222222"
            ax.text(j, i, f"{int(round(cm[i, j]))}\n({pct[i, j]:.1f}%)", ha="center", va="center", fontsize=13,
                    fontweight="bold", color=col)
    ax.set_xticks([0, 1], ["Bashorat: fon (0)", "Bashorat: musbat (1)"])
    ax.set_yticks([0, 1], ["Haqiqiy: fon (0)", "Haqiqiy: musbat (1)"])
    ax.set_xlabel("Bashorat")
    ax.set_ylabel("Haqiqiy sinf")
    thr_txt = f"Youden bo'sag'i = {thr:.3f}  |  " if np.isfinite(thr) else ""
    _set_title(ax, f"{name}\n{thr_txt}Sensitivlik = {sens:.3f}, Spetsifiklik = {spec:.3f}", fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    _footnote(fig, "Bo'sag' - o'rtacha OOF prospektivlik indeksida Youden J = sens + spec − 1 maksimumi; foizlar "
                   "qator (haqiqiy sinf) bo'yicha.")


# ---------------------------------------------------------------------------
# Spatial CV diagnostikasi
# ---------------------------------------------------------------------------
def _block(result, key):
    b = result.get(key) if isinstance(result, dict) else None
    return b if isinstance(b, dict) else None


def _block_metrics(result, key):
    b = _block(result, key)
    if b is not None:
        return _models_dict(b.get("metrics"))
    legacy = {"spatial": "roc_results", "random": "roc_results_random"}.get(key)
    return _models_dict(result.get(legacy)) if legacy and isinstance(result, dict) else {}


def _dataset_coords(result):
    ds = result.get("dataset") if isinstance(result, dict) else None
    coords = getattr(ds, "coords", None)
    if coords is None and isinstance(result, dict):
        coords = result.get("coords")
    if coords is None:
        return None, None
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2:
        return None, None
    y = getattr(ds, "y", None)
    return c, (None if y is None else np.asarray(y).ravel())


def _bars_panel(ax, spatial_m, random_m):
    names = list(spatial_m) + [k for k in random_m if k not in spatial_m]
    x = np.arange(len(names))
    has_rand = bool(random_m)
    w = 0.38 if has_rand else 0.55
    series = [(random_m, -w / 2, _RANDOM_COLOR, "Random CV"), (spatial_m, +w / 2, _SPATIAL_COLOR, "Spatial blok CV")] \
        if has_rand else [(spatial_m, 0.0, _SPATIAL_COLOR, "Spatial blok CV")]
    allv = [_f(src[n].get("auc")) for src, *_ in series for n in names if n in src]
    allv = [v for v in allv if np.isfinite(v)]
    lower = 0.4
    if allv and min(allv) < 0.45:
        lower = max(0.0, math.floor((min(allv) - 0.1) * 10) / 10)
    seen = set()
    for src, offset, color, label in series:
        for xi, nm in zip(x, names):
            m = src.get(nm)
            auc = _f(m.get("auc")) if m else float("nan")
            if not np.isfinite(auc):
                continue
            lo, hi = _ci(m)
            if np.isfinite(lo) and np.isfinite(hi):
                err = [[max(0.0, auc - lo)], [max(0.0, hi - auc)]]
            else:
                sd = _f(m.get("auc_std"), 0.0)
                err = [[sd], [sd]]
            ax.bar(xi + offset, auc, w, color=color, label=None if label in seen else label,
                   yerr=err, ecolor="#333333", capsize=3, error_kw={"lw": 1}, zorder=2)
            seen.add(label)
            ax.text(xi + offset, (lower + auc) / 2, f"{auc:.2f}", ha="center", va="center", fontsize=7,
                    rotation=90, color="white", fontweight="bold", zorder=3)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Tasodifiy (0.5)", zorder=1)
    ax.set_xticks(x, [_disp(n) for n in names], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(lower, 1.0 + 0.32 * (1.0 - lower))
    ax.set_yticks(np.arange(math.ceil(lower * 10 - 1e-9) / 10, 1.001, 0.1))
    ax.set_ylabel("AUC (95% CI - blok-bootstrap)")
    _set_title(ax, "Random va spatial blok CV: AUC", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)


def _foldmap_panel(fig, ax, coords, fold_map, y, n_pos, block_size):
    """Fold xaritasi: diskret legend (fold raqamlari) tashqarida; musbatlar - bo'sh doira."""
    fm = np.asarray(fold_map).ravel()
    if fm.size != coords.shape[0]:
        raise _NoData(f"fold_map uzunligi ({fm.size}) nuqtalar soniga ({coords.shape[0]}) teng emas.")
    folds = np.unique(fm[fm >= 0]) if fm.dtype.kind in "iu" else np.unique(fm[np.isfinite(fm)])
    folds = list(folds)
    handles = []
    if len(folds) <= len(_FOLD_COLORS):
        lut = {f: _FOLD_COLORS[k] for k, f in enumerate(folds)}
        colors = [lut.get(f, "#cccccc") for f in fm]
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=26, edgecolors="white", linewidths=0.4, zorder=2)
        handles += [Line2D([], [], marker="o", ls="", color=lut[f], markeredgecolor="white", ms=7,
                           label=f"Fold {int(f) + 1}") for f in folds]
    else:                                                    # juda ko'p fold - uzluksiz colorbar
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=fm, cmap="viridis", s=26, zorder=2)
        fig.colorbar(sc, ax=ax, label="Fold raqami (0 dan)", shrink=0.8)
    if y is not None and y.size == coords.shape[0] and (y == 1).any():
        pm = y == 1
    elif n_pos:
        pm = np.zeros(coords.shape[0], dtype=bool)
        pm[:int(n_pos)] = True                               # musbatlar BIRINCHI (Dataset konvensiyasi)
    else:
        pm = None
    if pm is not None and pm.any():
        ax.scatter(coords[pm, 0], coords[pm, 1], facecolors="none", edgecolors="black", s=64, linewidths=1.2,
                   zorder=3)
        handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="black", ms=8,
                              label=f"Ma'lum konlar (n={int(pm.sum())})"))
    ttl = "Spatial fold xaritasi (1-takror)"
    if block_size and np.isfinite(block_size):
        ttl += f"\nblok = {block_size:,.0f} m"
    _set_title(ax, ttl, fontsize=10)
    ax.set_xlabel("X (EPSG:28411), m")
    ax.set_ylabel("Y (EPSG:28411), m")
    ax.set_aspect("equal", adjustable="box")
    _plain_axis(ax)
    return handles


def _plain_axis(ax):
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis="x", labelrotation=25, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)


def _bgsens_panel(ax, bg):
    summ = bg["summary"]
    names = [k for k, v in summ.items() if isinstance(v, dict)]
    for i, nm in enumerate(names):
        s = summ[nm]
        mean, std = _f(s.get("mean")), _f(s.get("std"), 0.0)
        ax.errorbar(i, mean, yerr=std, fmt="o", color=_color(nm, i), capsize=4, ms=6, lw=1.6, zorder=3)
        lo, hi = _f(s.get("min")), _f(s.get("max"))
        if np.isfinite(lo) and np.isfinite(hi):
            ax.vlines(i, lo, hi, color=_color(nm, i), lw=4, alpha=0.3, zorder=2)
    ax.set_xticks(range(len(names)), [_disp(n) for n in names], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUC (o'rtacha ± std; keng chiziq - min..max)")
    nd = bg.get("n_draws")
    _set_title(ax, "Fon nuqtalar sezgirligi" + (f"\n({nd} ta fon tanlovi)" if nd else ""), fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.margins(x=0.15)


@_safe
def draw_spatial_diagnostics(fig, result):
    """1) Random va spatial blok CV AUC (CI bilan) yonma-yon; 2) 1-takror spatial fold xaritasi (legend: fold
    raqamlari); 3) (bor bo'lsa) fon nuqtalar sezgirligi. result - TrainingResult (spatial/random CVBlock, dataset)."""
    if not isinstance(result, dict) or not result:
        raise _NoData("Natija yo'q.")
    sp_m, rn_m = _block_metrics(result, "spatial"), _block_metrics(result, "random")
    coords, y = _dataset_coords(result)
    sp_block = _block(result, "spatial") or {}
    fold_map = sp_block.get("fold_map", result.get("foldmap_spatial"))
    bg = result.get("bg_sensitivity")
    has_bg = isinstance(bg, dict) and isinstance(bg.get("summary"), dict) and bool(bg["summary"])
    has_map = coords is not None and fold_map is not None and np.asarray(fold_map).size > 0
    if not sp_m and not rn_m and not has_map and not has_bg:
        raise _NoData("Spatial CV natijalari yo'q.")
    _prepare(fig)
    n_panels = int(bool(sp_m or rn_m)) + int(has_map) + int(has_bg)
    ratios = ([1.5] if (sp_m or rn_m) else []) + ([1.2] if has_map else []) + ([0.8] if has_bg else [])
    gs = fig.add_gridspec(1, n_panels, width_ratios=ratios)
    k = 0
    handles = []
    first_ax = None
    if sp_m or rn_m:
        ax = fig.add_subplot(gs[0, k])
        _bars_panel(ax, sp_m, rn_m)
        first_ax = ax
        k += 1
    if has_map:
        ax = fig.add_subplot(gs[0, k])
        try:
            handles = _foldmap_panel(fig, ax, coords, fold_map, y, result.get("n_positive"),
                                     _f(result.get("block_size")))
        except _NoData as e:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"Fold xaritasi:\n{NO_DATA}\n({e})", ha="center", va="center", fontsize=9,
                    color="#555555", transform=ax.transAxes)
        first_ax = first_ax or ax
        k += 1
    if has_bg:
        ax = fig.add_subplot(gs[0, k])
        _bgsens_panel(ax, bg)
        first_ax = first_ax or ax
    if handles:
        has_folds = handles[0].get_label().startswith("Fold")
        _legend_outside(fig, first_ax, handles, where="right", title="Validation fold" if has_folds else None,
                        title_fontsize=8)
    bs = _f(result.get("block_size"))
    _suptitle(fig, "Spatial CV diagnostikasi" + (f" (blok o'lchami {bs:,.0f}\u00a0m)" if np.isfinite(bs) and bs > 0 else ""),
                 fontsize=12)
    if sp_m and rn_m:
        diffs = [f"{n}: {_f(rn_m[n].get('auc')) - _f(sp_m[n].get('auc')):+.3f}" for n in sp_m
                 if n in rn_m and np.isfinite(_f(rn_m[n].get('auc'))) and np.isfinite(_f(sp_m[n].get('auc')))]
        if diffs:
            _footnote(fig, "Random − spatial AUC (optimizm): " + "; ".join(diffs), width=170)


# ---------------------------------------------------------------------------
# Feature importance, SHAP
# ---------------------------------------------------------------------------
def _top_idx(values, p):
    """Eng muhim top-N indekslar (o'sish tartibida: eng muhimi oxirida - barh'da tepada)."""
    v = np.where(np.isfinite(values), values, -np.inf)
    order = np.argsort(v, kind="stable")
    if p > TOP_N_THRESHOLD:
        order = order[-TOP_N:]
    return order[np.isfinite(v[order])]


def _bar_panel(ax, values, std, names, color, xlabel, title, zero_line=True):
    p = values.size
    idx = _top_idx(values, p)
    if idx.size == 0:
        raise _NoData("Importance qiymatlari chekli emas.")
    y = np.arange(idx.size)
    xerr = None
    if std is not None and std.size == p:
        xerr = np.where(np.isfinite(std[idx]), np.abs(std[idx]), 0.0)
    ax.barh(y, values[idx], xerr=xerr, color=color, ecolor="#222222", error_kw={"lw": 1}, capsize=2, height=0.7)
    fs = 8 if idx.size <= 20 else 7
    ax.set_yticks(y, [_short(names[i]) for i in idx], fontsize=fs)
    if zero_line:
        ax.axvline(0, color="#444444", lw=0.8)
    ax.set_xlabel(xlabel, fontsize=8)
    t = title + (f"\n(eng muhim {idx.size}/{p})" if idx.size < p else "")
    _set_title(ax, t, fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)


def _shap_entries(result):
    """result['shap'] ({model: entry}) yoki result o'zi ({model: entry}/entry) => {model: entry}."""
    if not isinstance(result, dict):
        return {}
    if "shap" in result:
        s = result["shap"]
        return {str(k): v for k, v in s.items()} if isinstance(s, dict) else {}
    if "shap_values" in result or "mean_abs_shap" in result:
        return {"model": result}
    return {str(k): v for k, v in result.items() if isinstance(v, dict) and ("shap_values" in v or "mean_abs_shap" in v)}


def _shap_arrays(entry, result_names=None):
    """entry => (sv (n,p) float64, X (n,p)|None, names, units, expected). Yaroqsiz bo'lsa _NoData."""
    if not isinstance(entry, dict):
        entry = {"shap_values": entry}
    X = entry.get("X_background")
    Xa = None
    if X is not None:
        try:
            Xa = np.asarray(X, dtype=np.float64)
            if Xa.ndim != 2:
                Xa = None
        except (TypeError, ValueError):
            Xa = None
    raw = entry.get("shap_values")
    sv = None
    if raw is not None:
        n_s, n_f = (Xa.shape if Xa is not None else (None, None))
        try:
            sv = normalize_shap_values(raw, 1, n_samples=n_s, n_features=n_f)
        except (ValueError, TypeError):
            try:
                sv = normalize_shap_values(raw, 1)           # X_background mos kelmasa ham (n, p) ni olamiz
                Xa = None if Xa is not None and Xa.shape != sv.shape else Xa
            except (ValueError, TypeError) as e:
                raise _NoData(f"SHAP qiymatlari yaroqsiz: {e}") from None
    if sv is None or sv.ndim != 2 or sv.shape[0] == 0 or sv.shape[1] == 0:
        raise _NoData("SHAP qiymatlari yo'q yoki bo'sh.")
    if Xa is not None and Xa.shape != sv.shape:
        Xa = None
    names = _names_for(sv.shape[1], entry.get("feature_names") or result_names)
    ev = entry.get("expected_value")
    return sv, Xa, names, entry.get("units"), (None if ev is None else _f(ev))


def _mean_abs_shap(entry, p):
    """Ichki SHAP |mean| (p,): mean_abs_shap mos bo'lsa o'shani, aks holda shap_values'dan; bo'lmasa None."""
    if not isinstance(entry, dict):
        entry = {"shap_values": entry}
    ma = entry.get("mean_abs_shap")
    if ma is not None:
        a = _arr(ma)
        if a.size == p and np.isfinite(a).any():
            return a
    try:
        sv, _, _, _, _ = _shap_arrays(entry)
    except _NoData:
        return None
    if sv.shape[1] != p:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = np.nanmean(np.abs(sv), axis=0)
    return a if np.isfinite(a).any() else None


@_safe
def draw_importance(fig, result):
    """Feature importance: 1-qator - OOF permutation importance (ΔAUC, fold-std error-bar, modellar yonma-yon);
    2-qator - SHAP |mean| (faqat haqiqiy mazmunli bo'lsa; o'q yorlig'ida birlik: modellararo magnituda
    solishtirilmaydi). 30+ feature bo'lsa har modelda top-20 ko'rsatiladi."""
    if not isinstance(result, dict):
        raise _NoData("Natija yo'q.")
    imp = result.get("importance") if isinstance(result.get("importance"), dict) else {}
    models = imp.get("models") if isinstance(imp.get("models"), dict) else None
    if models is None:                                       # zaxira: CVBlock.perm_importance
        sp = _block(result, "spatial") or {}
        models = sp.get("perm_importance") if isinstance(sp.get("perm_importance"), dict) else {}
    names_all = imp.get("feature_names") or result.get("feature_names")
    perm = {}
    for k, v in models.items():
        if not isinstance(v, dict):
            continue
        mean = _arr(v.get("mean"))
        if mean.size == 0 or not np.isfinite(mean).any():
            continue
        std = _arr(v.get("std"))
        perm[str(k)] = (mean, std if std.size == mean.size else None, v)
    shp = {}
    for k, v in _shap_entries(result).items():
        p = None
        if isinstance(v, dict):
            for key in ("mean_abs_shap", "feature_names"):
                a = v.get(key)
                if a is not None and len(a) > 0:
                    p = len(a)
                    break
            if p is None:
                try:
                    p = _shap_arrays(v)[0].shape[1]
                except _NoData:
                    p = None
        ma = _mean_abs_shap(v, p) if p else None
        if ma is not None:
            shp[k] = (ma, v)
    if not perm and not shp:
        raise _NoData("Permutation importance ham, SHAP ham hisoblanmagan.")
    _prepare(fig)
    n_rows = int(bool(perm)) + int(bool(shp))
    n_cols = max(len(perm), len(shp))
    gs = fig.add_gridspec(n_rows, n_cols)
    r = 0
    for c, (name, (mean, std, v)) in enumerate(perm.items()):
        ax = fig.add_subplot(gs[0, c])
        ttl = f"{name}\nPermutation importance"
        nf, nv = v.get("n_folds"), v.get("n_valid_folds")
        if nf:
            ttl += f" ({nv if nv is not None else nf}/{nf} fold)"
        _bar_panel(ax, mean, std, _names_for(mean.size, names_all), _color(name, c),
                   "ΔAUC (pasayish, OOF)", ttl)
    if perm:
        r = 1
    for c, (name, (ma, v)) in enumerate(shp.items()):
        ax = fig.add_subplot(gs[r, c])
        units = (v.get("units") if isinstance(v, dict) else None) or "model chiqishi birligida"
        fn = (v.get("feature_names") if isinstance(v, dict) else None) or names_all
        _bar_panel(ax, ma, None, _names_for(ma.size, fn), _color(name, c),
                   f"o'rtacha |SHAP| ({units})", f"{name}\nSHAP |mean|", zero_line=False)
    method = imp.get("method")
    _suptitle(fig, "Feature importance" + (f" (asosiy usul: {method})" if method else ""), fontsize=11)
    if shp:
        _footnote(fig, "SHAP birliklari modelga bog'liq (RF - ehtimollik, XGBoost - log-odds): modellar orasida "
                       "magnitudani to'g'ridan-to'g'ri solishtirmang. One-hot kategorik ustunlar alohida aralashtiriladi.")


def _beeswarm_offsets(vals, rng, row_height=0.8, nbins=100):
    """shap.summary_plot'dagi kabi: bir xil qiymat atrofida nuqtalarni vertikal tarqatish (-row_height/2..+)."""
    n = vals.size
    if n == 0:
        return np.zeros(0)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    quant = np.round(nbins * (vals - lo) / (hi - lo + 1e-12))
    order = np.argsort(quant + rng.normal(size=n) * 1e-6)
    ys = np.zeros(n)
    layer, last = 0, -1
    for i in order:
        if quant[i] != last:
            layer = 0
        ys[i] = np.ceil(layer / 2.0) * (1 if layer % 2 == 0 else -1)
        layer += 1
        last = quant[i]
    return ys * (row_height / 2.0) / max(1.0, float(np.max(np.abs(ys))))


def _scale01(v):
    """Feature qiymatini 5..95 persentil bo'yicha 0..1 ga (NaN saqlanadi); o'zgarmas => hammasi NaN."""
    v = np.asarray(v, dtype=np.float64)
    fin = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    if not fin.any():
        return out
    lo, hi = np.percentile(v[fin], [5, 95])
    if hi <= lo:
        lo, hi = float(v[fin].min()), float(v[fin].max())
    if hi <= lo:
        return out
    out[fin] = np.clip((v[fin] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _unit_text(units):
    return f" ({units})" if units else ""


@_safe
def draw_shap_beeswarm(fig, result, model=None):
    """SHAP beeswarm (shap.summary_plot'siz): x - SHAP qiymati (musbat sinfga ta'sir), rang - feature qiymati.
    result - TrainingResult ("shap" kaliti) yoki {model: entry} yoki bitta entry. shap_values (n,p) yoki (n,p,2)/list."""
    pool = _shap_entries(result)
    if not pool:
        raise _NoData("SHAP natijasi yo'q (kutubxona o'rnatilmagan yoki hisoblanmagan).")
    if model is None:
        model = next(iter(pool))
    if str(model) not in pool:
        raise _NoData(f"'{model}' uchun SHAP yo'q (mavjud: {', '.join(pool)}).")
    names_hint = None
    if isinstance(result, dict) and isinstance(result.get("importance"), dict):
        names_hint = result["importance"].get("feature_names")
    sv, X, names, units, ev = _shap_arrays(pool[str(model)], names_hint)
    n, p = sv.shape
    if n > 2000:                                             # chizish uchun tasodifiy kichik qism
        sel = np.sort(np.random.default_rng(0).choice(n, 2000, replace=False))
        sv, X = sv[sel], (X[sel] if X is not None else None)
        n = 2000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        imp = np.nanmean(np.abs(sv), axis=0)
    idx = _top_idx(imp, p)
    if idx.size == 0:
        raise _NoData("SHAP qiymatlari chekli emas.")
    _prepare(fig)
    ax = fig.add_subplot(111)
    rng = np.random.default_rng(0)
    cmap = _with_bad(colormaps["coolwarm"], "#BBBBBB")
    for row, j in enumerate(idx):
        v = sv[:, j]
        ok = np.isfinite(v)
        if not ok.any():
            continue
        off = np.zeros(n)
        off[ok] = _beeswarm_offsets(v[ok], rng)
        if X is not None:
            z = _scale01(X[:, j])
            ax.scatter(v[ok], row + off[ok], c=z[ok], cmap=cmap, vmin=0, vmax=1, s=14, alpha=0.85,
                       linewidths=0, zorder=3)
        else:
            ax.scatter(v[ok], row + off[ok], color="#7F7F7F", s=14, alpha=0.8, linewidths=0, zorder=3)
    ax.axvline(0, color="#444444", lw=0.8, zorder=1)
    ax.set_yticks(range(idx.size), [_short(names[j], 30) for j in idx], fontsize=8 if idx.size <= 20 else 7)
    ax.set_ylim(-0.7, idx.size - 0.3)
    for r in range(idx.size):
        ax.axhline(r, color="#DDDDDD", lw=0.5, ls=":", zorder=0)
    ax.set_xlabel(f"SHAP qiymati (musbat sinfga ta'siri){_unit_text(units)}", fontsize=9)
    ttl = f"SHAP beeswarm — {model}"
    if idx.size < p:
        ttl += f" (eng muhim {idx.size}/{p})"
    _set_title(ax, ttl, fontsize=11)
    if X is not None:
        cb = fig.colorbar(ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap), ax=ax, ticks=[0, 1],
                          shrink=0.8, pad=0.02, aspect=25)
        cb.ax.set_yticklabels(["Past", "Yuqori"])
        cb.set_label("Feature qiymati", fontsize=9)
    note = f"n = {n} ta fon nuqta."
    if ev is not None and np.isfinite(ev):
        note += f" Bazaviy qiymat E[f(x)] = {ev:.3f}."
    if X is None:
        note += " Feature qiymatlari (X_background) yo'q - rang berilmadi."
    _footnote(fig, note)


@_safe
def draw_shap_dependence(fig, result, model=None, feature=None, color_feature="auto"):
    """SHAP dependence: x - feature qiymati, y - uning SHAP qiymati; rang - eng bog'liq boshqa feature (color_feature=
    "auto"; None - rangsiz; nom - tanlangan). feature: nom yoki indeks (None => eng muhimi)."""
    pool = _shap_entries(result)
    if not pool:
        raise _NoData("SHAP natijasi yo'q (kutubxona o'rnatilmagan yoki hisoblanmagan).")
    if model is None:
        model = next(iter(pool))
    if str(model) not in pool:
        raise _NoData(f"'{model}' uchun SHAP yo'q (mavjud: {', '.join(pool)}).")
    names_hint = None
    if isinstance(result, dict) and isinstance(result.get("importance"), dict):
        names_hint = result["importance"].get("feature_names")
    sv, X, names, units, _ = _shap_arrays(pool[str(model)], names_hint)
    if X is None:
        raise _NoData("Dependence uchun feature qiymatlari (X_background) yo'q.")
    p = sv.shape[1]
    if feature is None:
        with warnings.catch_warnings():                      # butunlay NaN ustun: "Mean of empty slice"
            warnings.simplefilter("ignore", RuntimeWarning)
            col = np.nanmean(np.abs(sv), axis=0)
        if not np.isfinite(col).any():
            raise _NoData("SHAP qiymatlari chekli emas.")
        j = int(np.nanargmax(col))
    elif isinstance(feature, (int, np.integer)) and not isinstance(feature, bool):
        if not 0 <= int(feature) < p:
            raise _NoData(f"Feature indeksi {feature} oraliqdan tashqarida (0..{p - 1}).")
        j = int(feature)
    else:
        low = [s.lower() for s in names]
        if str(feature) in names:
            j = names.index(str(feature))
        elif str(feature).lower() in low:
            j = low.index(str(feature).lower())
        else:
            raise _NoData(f"Feature topilmadi: {feature!r}.")
    x, s = X[:, j], sv[:, j]
    ok = np.isfinite(x) & np.isfinite(s)
    if ok.sum() == 0:
        raise _NoData("Tanlangan feature uchun chekli nuqtalar yo'q.")
    cj = None
    if color_feature is not None:
        if color_feature == "auto":
            cj = _auto_color_feature(X, x, s, ok, j)
        elif str(color_feature) in names:
            cj = names.index(str(color_feature))
    _prepare(fig)
    ax = fig.add_subplot(111)
    cb_label = None
    if cj is not None:
        cv_ = X[:, cj]
        okc = ok & np.isfinite(cv_)
        sc = ax.scatter(x[okc], s[okc], c=cv_[okc], cmap="coolwarm", s=26, alpha=0.9, linewidths=0, zorder=3)
        if (ok & ~okc).any():
            ax.scatter(x[ok & ~okc], s[ok & ~okc], color="#BBBBBB", s=26, linewidths=0, zorder=3)
        cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cb_label = _short(names[cj], 30)
        cb.set_label(f"Rang: {cb_label}", fontsize=9)
    else:
        ax.scatter(x[ok], s[ok], color=_color(model), s=26, alpha=0.85, linewidths=0, zorder=3)
    _trend_line(ax, x[ok], s[ok])
    ax.axhline(0, color="#444444", lw=0.8, zorder=1)
    ax.set_xlabel(f"{_short(names[j], 40)} (feature qiymati)", fontsize=9)
    ax.set_ylabel(f"SHAP qiymati{_unit_text(units)}", fontsize=9)
    _set_title(ax, f"SHAP dependence — {model}: {_short(names[j], 40)}", fontsize=11)
    ax.grid(alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    _footnote(fig, f"n = {int(ok.sum())} ta fon nuqta. Har nuqta - bir fon namunada feature'ning musbat sinfga hissasi.")


def _auto_color_feature(X, x, s, ok, j):
    """Chiziqli trendni olib tashlagandan keyingi SHAP qoldig'i bilan eng kuchli bog'langan boshqa feature indeksi."""
    try:
        if np.ptp(x[ok]) > 0:
            res = s - np.polyval(np.polyfit(x[ok], s[ok], 1), x)
        else:
            res = s - np.nanmean(s[ok])
    except (np.linalg.LinAlgError, ValueError):
        return None
    best, bj = 0.0, None
    for k in range(X.shape[1]):
        if k == j:
            continue
        xk = X[:, k]
        m = ok & np.isfinite(xk) & np.isfinite(res)
        if m.sum() < 5 or np.ptp(xk[m]) == 0 or np.ptp(res[m]) == 0:
            continue
        c = abs(np.corrcoef(xk[m], res[m])[0, 1])
        if np.isfinite(c) and c > best:
            best, bj = c, k
    return bj


def _trend_line(ax, x, s, n_bins=8):
    """Kvantil bin medianalari chizig'i - umumiy yo'nalishni ko'rsatadi."""
    if np.unique(x).size < 12 or x.size < 20:
        return
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return
    b = np.clip(np.digitize(x, edges[1:-1]), 0, edges.size - 2)
    xs, ys = [], []
    for k in range(edges.size - 1):
        m = b == k
        if m.sum() >= 2:
            xs.append(np.median(x[m]))
            ys.append(np.median(s[m]))
    if len(xs) >= 3:
        ax.plot(xs, ys, color="#222222", lw=1.6, marker="s", ms=3, zorder=4, label="Bin medianalari")


# ---------------------------------------------------------------------------
# ENH-07: korrelyatsiya, xarita, success-rate
# ---------------------------------------------------------------------------
@_safe
def draw_corr_heatmap(fig, corr):
    """Pearson korrelyatsiya heatmap (-1..1); |r| >= 0.9 (diagonaldan tashqari) qora ramka bilan belgilanadi."""
    if corr is None:
        raise _NoData("Korrelyatsiya matritsasi yo'q.")
    try:
        df = corr if isinstance(corr, pd.DataFrame) else pd.DataFrame(corr)
    except (ValueError, TypeError):
        raise _NoData("Korrelyatsiya matritsasi yaroqsiz.") from None
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise _NoData("Korrelyatsiya matritsasi bo'sh.")
    if df.shape[0] != df.shape[1]:
        raise _NoData(f"Korrelyatsiya matritsasi kvadrat emas: {df.shape}.")
    try:
        M = df.to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        raise _NoData("Korrelyatsiya matritsasi raqamli emas.") from None
    if not np.isfinite(M).any():
        raise _NoData("Korrelyatsiya qiymatlari chekli emas.")
    n = M.shape[0]
    names = [str(c) for c in df.columns]
    _prepare(fig)
    ax = fig.add_subplot(111)
    cmap = _with_bad(colormaps["RdBu_r"], "#DDDDDD")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-1, vmax=1, aspect="equal", interpolation="nearest")
    step = max(1, math.ceil(n / 40))
    ticks = list(range(0, n, step))
    fs = 8 if n <= 20 else (7 if n <= 30 else 6)
    ax.set_xticks(ticks, [_short(names[i], 22) for i in ticks], rotation=60, ha="right", fontsize=fs)
    ax.set_yticks(ticks, [_short(names[i], 22) for i in ticks], fontsize=fs)
    hi_mask = (np.abs(M) >= HIGH_CORR) & ~np.eye(n, dtype=bool)
    n_pairs = int(np.triu(hi_mask, 1).sum())
    for i, j in zip(*np.nonzero(hi_mask)):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.6, zorder=4))
    if n <= 15:
        for i in range(n):
            for j in range(n):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(M[i, j]) > 0.6 else "#222222",
                            fontweight="bold" if hi_mask[i, j] else "normal", zorder=5)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Pearson r")
    _set_title(ax, f"Feature korrelyatsiyasi (yuqori korrelyatsiyali juftlar |r| ≥ {HIGH_CORR}: {n_pairs} ta)",
                 fontsize=10)
    ax.tick_params(length=0)
    if n_pairs:
        _legend_outside(fig, ax, [Patch(fill=False, edgecolor="black", lw=1.5, label=f"|r| ≥ {HIGH_CORR}")])


def _affine(transform):
    """Affine yoki 6 elementli (a,b,c,d,e,f) => (a, c, e, f) yoki None."""
    if transform is None:
        return None
    try:
        return float(transform.a), float(transform.c), float(transform.e), float(transform.f)
    except AttributeError:
        pass
    try:
        t = [float(v) for v in transform]
        if len(t) >= 6:
            return t[0], t[2], t[4], t[5]
    except (TypeError, ValueError):
        pass
    return None


def _xy_points(obj):
    """GeoDataFrame/GeoSeries/(n,2) massiv/DataFrame(x,y) => (m,2) float64 (bo'sh bo'lishi mumkin)."""
    if obj is None:
        return np.empty((0, 2))
    try:
        geom = getattr(obj, "geometry", None)
        if geom is not None:
            g = geom[~geom.is_empty & geom.notna()]
            g = g[g.geom_type == "Point"]
            return np.column_stack([g.x.to_numpy(), g.y.to_numpy()]).astype(np.float64) if len(g) else np.empty((0, 2))
        if isinstance(obj, pd.DataFrame) and {"x", "y"} <= set(obj.columns):
            return obj[["x", "y"]].to_numpy(dtype=np.float64)
        a = np.asarray(obj, dtype=np.float64)
        if a.ndim == 2 and a.shape[1] >= 2:
            a = a[:, :2]
            return a[np.isfinite(a).all(axis=1)]
    except Exception:
        pass
    return np.empty((0, 2))


def _geom_segments(geom, out):
    """Shapely geometriyadan (Polygon/Multi*/Collection/Line) chiziq segmentlari ro'yxatini yig'adi."""
    if geom is None or getattr(geom, "is_empty", True):
        return
    t = geom.geom_type
    if t == "Polygon":
        out.append(np.asarray(geom.exterior.coords)[:, :2])
        for r in geom.interiors:
            out.append(np.asarray(r.coords)[:, :2])
    elif t in ("LineString", "LinearRing"):
        out.append(np.asarray(geom.coords)[:, :2])
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            _geom_segments(g, out)


def _aoi_segments(aoi):
    if aoi is None:
        return []
    segs = []
    try:
        if isinstance(getattr(aoi, "geom_type", None), str):         # bitta shapely geometriya
            _geom_segments(aoi, segs)
        else:                                                        # GeoDataFrame / GeoSeries / ro'yxat
            for g in (aoi.geometry if hasattr(aoi, "geometry") else aoi):
                _geom_segments(g, segs)
    except Exception:
        return []
    return [s for s in segs if len(s) >= 2 and np.isfinite(s).all()]


@_safe
def draw_map(fig, array, transform=None, title="", cmap="RdYlGn_r", vmin=0.0, vmax=1.0, aoi_gdf=None,
             positives=None, background=None, class_labels=None, cbar_label=""):
    """Xarita: extent transform'dan (EPSG:28411 koordinatalari o'qlarda), nodata (NaN; sinflarda 0) shaffof.
    class_labels berilsa - diskret colorbar (int sinflar 1..n). Overlay: AOI kontur, musbat (bo'sh doira) va
    fon (x) nuqtalar; legend xaritadan tashqarida (pastda). Pastki colorbar uchun INDEX_LABEL'dan foydalaning."""
    if array is None:
        raise _NoData("Xarita massivi yo'q.")
    a = np.asarray(array)
    if a.ndim != 2 or a.size == 0:
        raise _NoData("Xarita massivi 2 o'lchamli (H, W) va bo'sh bo'lmasligi kerak.")
    a = a.astype(np.float64)
    labels = list(class_labels) if class_labels is not None and len(class_labels) > 0 else None
    if labels is not None:
        mask = ~np.isfinite(a) | (a <= 0)                    # 0 = nodata
    else:
        mask = ~np.isfinite(a)
    if mask.all():
        raise _NoData("Xaritada yaroqli piksel yo'q (barchasi nodata).")
    H, W = a.shape
    aff = _affine(transform)
    if aff is not None:
        pa, pc, pe, pf = aff
        extent = (pc, pc + W * pa, pf + H * pe, pf)          # (left, right, bottom, top); pe < 0
        xlabel, ylabel = "X (EPSG:28411), m", "Y (EPSG:28411), m"
    else:
        extent = (0, W, H, 0)
        xlabel, ylabel = "Ustun (piksel)", "Qator (piksel)"
    _prepare(fig)
    ax = fig.add_subplot(111)
    base = colormaps[cmap] if isinstance(cmap, str) else cmap
    if labels is not None:
        n = len(labels)
        cm = mcolors.ListedColormap(base(np.linspace(0.08, 0.92, n)))
        cm = _with_bad(cm, _TRANSPARENT)
        norm = mcolors.BoundaryNorm(np.arange(0.5, n + 1.5), cm.N)
        im = ax.imshow(np.ma.array(a, mask=mask), cmap=cm, norm=norm, extent=extent, origin="upper",
                       interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, ticks=np.arange(1, n + 1), shrink=0.85, pad=0.02)
        cb.ax.set_yticklabels([_short(s, 18) for s in labels], fontsize=8)
        cb.ax.tick_params(length=0)
        if cbar_label:
            cb.set_label(cbar_label)
    else:
        cm = _with_bad(base, _TRANSPARENT)
        im = ax.imshow(np.ma.array(a, mask=mask), cmap=cm, vmin=vmin, vmax=vmax, extent=extent, origin="upper",
                       interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        if cbar_label:
            cb.set_label(cbar_label)
    handles = []
    segs = _aoi_segments(aoi_gdf)
    if segs:
        ax.add_collection(LineCollection(segs, colors="black", linewidths=1.1, zorder=3))
        handles.append(Line2D([], [], color="black", lw=1.2, label="AOI chegarasi"))
    bg_pts, pos_pts = _xy_points(background), _xy_points(positives)
    if len(bg_pts):
        ax.scatter(bg_pts[:, 0], bg_pts[:, 1], marker="x", s=14, color="#444444", alpha=0.75, linewidths=0.8,
                   zorder=4)
        handles.append(Line2D([], [], marker="x", ls="", color="#444444", ms=6, label=f"Fon nuqtalar (n={len(bg_pts)})"))
    if len(pos_pts):
        ax.scatter(pos_pts[:, 0], pos_pts[:, 1], marker="o", s=34, facecolors="white", edgecolors="black",
                   linewidths=1.1, zorder=5)
        handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="black", ms=7,
                              label=f"Ma'lum konlar (n={len(pos_pts)})"))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if aff is not None:
        _plain_axis(ax)
    if title:
        _set_title(ax, title, fontsize=11)
    _legend_outside(fig, ax, handles, where="lower", ncol=len(handles))


@_safe
def draw_success_rate(fig, curve):
    """Success-rate: x - eng yuqori indeksli maydon ulushi (%), y - ushlangan konlar ulushi (%); tasodifiy diagonal
    va AUC. curve - predict.success_rate_curve natijasi yoki {nom: curve} (bir nechta model)."""
    if not isinstance(curve, dict) or not curve:
        raise _NoData("Success-rate egri chizig'i yo'q.")
    curves = {"": curve} if "area_frac" in curve else {str(k): v for k, v in curve.items() if isinstance(v, dict)}
    valid = {}
    for k, c in curves.items():
        x, y = _arr(c.get("area_frac")), _arr(c.get("capture_frac"))
        if x.size >= 2 and x.size == y.size and np.isfinite(x).all() and np.isfinite(y).all():
            valid[k] = (x, y, c)
    if not valid:
        raise _NoData("Success-rate uchun kon yoki yaroqli piksel yo'q.")
    _prepare(fig)
    ax = fig.add_subplot(111)
    for i, (k, (x, y, c)) in enumerate(valid.items()):
        auc, npos = _f(c.get("auc")), c.get("n_pos")
        label = ((k + ": ") if k else "") + (f"AUC = {auc:.3f}" if np.isfinite(auc) else "AUC = n/a")
        npos = _f(npos)
        if np.isfinite(npos):
            label += f" (n = {int(npos)} kon)"
        ens = _is_ens(k)
        ax.plot(x * 100, y * 100, lw=3.0 if ens else 2.0, color=_color(k, i) if k else "#B2182B", label=label, zorder=3)
    ax.plot([0, 100], [0, 100], ls="--", color="gray", lw=1, label="Tasodifiy (AUC = 0.5)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Eng yuqori prospektivlik indeksli maydon ulushi, %")
    ax.set_ylabel("Ushlangan ma'lum konlar ulushi, %")
    _set_title(ax, "Success-rate (capture) egri chizig'i", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    _footnote(fig, "Piksellar prospektivlik indeksi bo'yicha kamayish tartibida; egri chiziq diagonaldan qancha "
                   "yuqori bo'lsa, xarita shuncha yaxshi (konlar kichik maydonda to'planadi). Model o'qitishda ishlatilgan "
                   "konlar bo'yicha hisoblangan bo'lsa, baho optimistik (OOF emas).")


# ---------------------------------------------------------------------------
# Giperparametr qidiruvi
# ---------------------------------------------------------------------------
def _records(v):
    if isinstance(v, dict):
        v = [v]
    if not isinstance(v, (list, tuple)):
        return []
    return [r for r in v if isinstance(r, dict)]


def _vkey(v):
    return "None" if v is None else str(v)


def _is_num(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) \
        and np.isfinite(float(v))


def _trials(r):
    """Yozuvdagi yaroqli nomzodlar (params lug'ati bor dict'lar)."""
    return [t for t in (r.get("trials") or []) if isinstance(t, dict) and isinstance(t.get("params"), dict)]


def _varied_params(recs):
    """Qidiruvda o'zgargan parametrlar: nomzodlar (trials) yoki tanlangan qiymatlar farq qiladigan kalitlar."""
    keys, seen = [], {}
    for r in recs:
        for t in _trials(r):
            for k, v in (t.get("params") or {}).items():
                seen.setdefault(k, set()).add(_vkey(v))
        for k, v in (r.get("best_params") or {}).items():
            seen.setdefault(k, set()).add(_vkey(v))
    for r in recs:
        for k in (r.get("best_params") or {}):
            if k not in keys and len(seen.get(k, ())) > 1:
                keys.append(k)
    return keys


def _scatter_param(ax, recs, key, color):
    xs_all, vals_all = [], []
    for t_i, r in enumerate(recs):
        for t in _trials(r):
            if key in t["params"]:
                xs_all.append(t_i)
                vals_all.append(t["params"][key])
    chosen = [(t_i, (r.get("best_params") or {}).get(key), bool(r.get("fallback") or r.get("skipped_reason")))
              for t_i, r in enumerate(recs) if key in (r.get("best_params") or {})]
    pool = vals_all + [c[1] for c in chosen]
    nums = [float(v) for v in pool if _is_num(v)]
    has_none = any(v is None for v in pool)
    numeric = bool(nums) and all(_is_num(v) or v is None for v in pool)
    if numeric:
        lo_v, hi_v = min(nums), max(nums)
        is_log = lo_v > 0 and hi_v / lo_v >= 100
        if is_log:
            ax.set_yscale("log")
            none_pos = hi_v * (hi_v / lo_v) ** 0.15
        else:
            none_pos = hi_v + 0.15 * ((hi_v - lo_v) or max(abs(hi_v), 1.0))
        pos = lambda v: none_pos if v is None else float(v)  # noqa: E731
        if has_none:                                         # None (cheksiz/avtomatik) - alohida tick
            ticks = [t for t in MaxNLocator(4).tick_values(lo_v, hi_v) if lo_v <= t <= hi_v] if not is_log else []
            if is_log:
                ticks = [10.0 ** e for e in range(math.ceil(math.log10(lo_v)), math.floor(math.log10(hi_v)) + 1)]
            ax.set_yticks(ticks + [none_pos], [f"{t:g}" for t in ticks] + ["None"], fontsize=7)
            ax.minorticks_off()
            ax.set_ylim(*( (lo_v / (hi_v / lo_v) ** 0.05, none_pos * (hi_v / lo_v) ** 0.05) if is_log
                           else (lo_v - 0.08 * (none_pos - lo_v), none_pos + 0.08 * (none_pos - lo_v))))
    else:
        cats = sorted({_vkey(v) for v in pool})
        lut = {c: i for i, c in enumerate(cats)}
        pos = lambda v: lut[_vkey(v)]                        # noqa: E731
        ax.set_yticks(range(len(cats)), [_short(c, 14) for c in cats], fontsize=7)
        ax.set_ylim(-0.6, len(cats) - 0.4)
    rng = np.random.default_rng(0)
    if xs_all:
        ax.scatter(np.asarray(xs_all) + rng.uniform(-0.15, 0.15, len(xs_all)), [pos(v) for v in vals_all],
                   s=10, color="#BBBBBB", alpha=0.6, linewidths=0, zorder=2)
    ok_x = [c[0] for c in chosen if not c[2]]
    ok_y = [pos(c[1]) for c in chosen if not c[2]]
    fb_x = [c[0] for c in chosen if c[2]]
    fb_y = [pos(c[1]) for c in chosen if c[2]]
    if ok_x:
        ax.scatter(ok_x, ok_y, s=46, color=color, edgecolors="black", linewidths=0.6, zorder=3)
    if fb_x:
        ax.scatter(fb_x, fb_y, s=40, marker="x", color="#D55E00", linewidths=1.6, zorder=3)


TUNING_ROW_H = 1.25                                          # bitta panel qatori uchun minimal balandlik (dyuym)


def _tuning_grid(fig, plan):
    """Panel to'riga mos ustun sonini tanlaydi: qatorlar figura balandligiga (TUNING_ROW_H dyuym/qator) sig'sin.
    Sig'masa har modelda ko'rsatiladigan parametrlar kesiladi. Qaytaradi: (ncols, plan, kesilgan parametrlar soni)."""
    try:
        fw, fh = (float(v) for v in fig.get_size_inches())
    except (TypeError, ValueError, AttributeError):
        fw, fh = 9.0, 6.0
    max_rows = max(len(plan), int(fh // TUNING_ROW_H))
    max_cols = max(2, min(6, int(fw // 1.6)))
    need = max(1 + len(p[2]) for p in plan)
    start = max(2, min(4, need))

    def rows(nc, pl):
        return sum(math.ceil((1 + len(p[2])) / nc) for p in pl)

    for nc in range(min(start, max_cols), max_cols + 1):
        if rows(nc, plan) <= max_rows:
            return nc, plan, 0
    nc = max_cols
    per_model = max(1, max_rows // len(plan))                # har modelga ajratilgan qatorlar
    cap = per_model * nc - 1                                 # ball paneli ham bitta katak
    cut = [(n, r, ps[:cap]) for n, r, ps in plan]
    return nc, cut, sum(len(p[2]) for p in plan) - sum(len(p[2]) for p in cut)


@_safe
def draw_tuning_trials(fig, tuned_params):
    """Nested tuning natijasi: har model uchun (1) tashqi fold'lar bo'yicha tanlangan va bazaviy ball, (2) o'zgargan
    har giperparametr uchun fold'lar bo'yicha tanlangan qiymat (rangli) va sinab ko'rilgan nomzodlar (kulrang).
    Zaxira (fallback/skipped) yozuvlar - 'x' belgi; bo'sh trials hisobga olinadi. tuned_params - cv.run_cv["tuned_params"]."""
    if not isinstance(tuned_params, dict) or not tuned_params:
        raise _NoData("Tuning natijalari yo'q (tuning o'chirilgan).")
    models = {}
    for name, v in tuned_params.items():
        recs = sorted(_records(v), key=lambda r: (r.get("repeat", 0) or 0, r.get("fold", 0) or 0))
        if recs:
            models[str(name)] = recs
    if not models:
        raise _NoData("Tuning natijalari yo'q.")
    plan = []                                                # [(model, recs, [param,...])]
    for name, recs in models.items():
        plan.append((name, recs, _varied_params(recs)[:8]))
    ncols, plan, n_hidden = _tuning_grid(fig, plan)
    maxc = max(14, int(float(fig.get_figwidth()) / ncols * 13))      # sarlavha panel kengligidan chiqmasin
    row_counts = [math.ceil((1 + len(p[2])) / ncols) for p in plan]
    _prepare(fig)
    gs = fig.add_gridspec(sum(row_counts), ncols)
    r0 = 0
    for (name, recs, params), nrows in zip(plan, row_counts):
        color = _color(name)
        multi = len({r.get("repeat", 0) for r in recs}) > 1
        xt = [f"{(r.get('repeat') or 0) + 1}.{(r.get('fold') or 0) + 1}" if multi else f"{(r.get('fold') or 0) + 1}"
              for r in recs]
        n_fb = sum(1 for r in recs if r.get("fallback") or r.get("skipped_reason"))
        panels = ["__score__"] + params
        for k, key in enumerate(panels):
            ax = fig.add_subplot(gs[r0 + k // ncols, k % ncols])
            x = np.arange(len(recs))
            if key == "__score__":
                best = np.array([_f(r.get("best_score")) for r in recs])
                base = np.array([_f(r.get("base_score")) for r in recs])
                if np.isfinite(base).any():
                    ax.plot(x, base, ls="--", marker="o", ms=4, color="gray", label="bazaviy")
                if np.isfinite(best).any():
                    ax.plot(x, best, marker="o", ms=5, color=color, label="tanlangan")
                if not (np.isfinite(best).any() or np.isfinite(base).any()):
                    ax.text(0.5, 0.5, "Ball yo'q\n(zaxira giperparametr)", ha="center", va="center", fontsize=8,
                            color="#777777", transform=ax.transAxes)
                else:
                    ax.legend(fontsize=6, loc="best", framealpha=0.85)
                scoring = next((r.get("scoring") for r in recs if r.get("scoring")), "ball")
                sfx = f", {n_fb}/{len(recs)} fold zaxira" if n_fb else ""        # zaxira izohi kesilmasin
                ax.set_title(_short(f"{name} · {scoring} (ichki CV)", max(12, maxc + 12 - len(sfx))) + sfx,
                             fontsize=8)
                ax.set_ylabel(str(scoring), fontsize=8)
            else:
                _scatter_param(ax, recs, key, color)
                ax.set_title(_short(f"{name} · {key}", maxc), fontsize=8)
            ax.set_xticks(x if len(recs) <= 12 else x[:: math.ceil(len(recs) / 12)])
            ax.set_xticklabels([xt[i] for i in ax.get_xticks().astype(int)], fontsize=7)
            ax.set_xlim(-0.5, len(recs) - 0.5)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(alpha=0.25)
            if k + ncols >= len(panels):                     # x yorlig'i faqat ustun pastidagi panellarda
                ax.set_xlabel("Tashqi fold" if not multi else "Takror.fold", fontsize=7)
        if not params:
            ax = fig.add_subplot(gs[r0, 1 if ncols > 1 else 0])
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{name}: giperparametrlar fold'lar bo'yicha o'zgarmadi\n(qidiruv bajarilmadi yoki "
                              "bazaviy qiymat tanlandi)", ha="center", va="center", fontsize=8, color="#666666",
                    transform=ax.transAxes)
        r0 += nrows
    _suptitle(fig, "Giperparametr qidiruvi (nested spatial CV): fold'lar bo'yicha tanlangan qiymatlar", fontsize=11)
    note = ("Rangli doira - tashqi fold'da tanlangan qiymat; kulrang nuqta - ichki qidiruvda sinalgan nomzod; "
            "'x' - tuning bajarilmagan (zaxira) fold.")
    if n_hidden:
        note += f" Joy yetmaganidan {n_hidden} ta parametr ko'rsatilmadi (rasmni kattalashtiring)."
    _footnote(fig, note)


# ---------------------------------------------------------------------------
# Saqlash
# ---------------------------------------------------------------------------
def save_all_figures(figs, out_dir, formats=("png", "pdf"), dpi=300, log_fn=None):
    """{nom: Figure} ni out_dir ga har format uchun saqlaydi (papka yaratiladi; fayl nomi safe_name). Qaytaradi: yozilgan
    yo'llar ro'yxati. Bitta rasm yiqilsa qolganlari saqlanadi (xato log_fn orqali xabar qilinadi)."""
    log = log_fn or noop_log
    if not figs:
        return []
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(formats, str):
        formats = (formats,)
    fmts = []
    for f in formats or ("png",):
        f = str(f).lower().lstrip(".")
        if f and f not in fmts:
            fmts.append(f)
    paths, used = [], set()
    for name, fig in figs.items():
        if fig is None or not hasattr(fig, "savefig"):
            log(f"  Ogohlantirish: '{name}' Figure emas - o'tkazib yuborildi.")
            continue
        stem = safe_name(name).strip("._") or "figure"
        base, k = stem, 2
        while stem.lower() in used:
            stem = f"{base}_{k}"
            k += 1
        used.add(stem.lower())
        for ext in fmts:
            path = os.path.join(out_dir, f"{stem}.{ext}")
            try:
                fig.savefig(path, dpi=dpi, format=ext, facecolor="white")
                paths.append(path)
            except Exception as e:                           # noqa: BLE001
                log(f"  Ogohlantirish: {path} saqlanmadi ({type(e).__name__}: {e}).")
    return paths
