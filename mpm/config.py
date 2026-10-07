# -*- coding: utf-8 -*-
"""
Konfiguratsiya: giperparametr spetsifikatsiyalari (PARAM_SPECS), RunConfig, TuningConfig,
preset (JSON) saqlash/yuklash va validatsiya.

PARAM_SPECS - YAGONA haqiqat manbai:
  * GUI giperparametr panelini shu ro'yxatdan AVTOMATIK quradi (barcha parametrlar aks etadi);
  * models.py / cnn.py modellarni shu parametrlardan quradi;
  * tuning.py qidiruv oraliqlarini (search_*) shu yerdan oladi;
  * persist.py / export shu lug'atni saqlaydi.

hyperparams lug'ati ko'rinishi: {"RandomForest": {param: qiymat}, "SVM": {...}, "XGBoost": {...}, "CNN": {...}}
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, fields, asdict

from .common import RANDOM_STATE, default_n_jobs, MODEL_ORDER

MODEL_NAMES = tuple(MODEL_ORDER)
PRESET_VERSION = 2


# ---------------------------------------------------------------------------
# ParamSpec
# ---------------------------------------------------------------------------
@dataclass
class ParamSpec:
    """Bitta giperparametr tavsifi.

    kind: "int" | "float" | "bool" | "choice" | "optint" | "optfloat"
      optint/optfloat - qiymat None bo'lishi mumkin (none_label GUI'da None nimani anglatishini ko'rsatadi)
    choice qiymatlari matn; modellar "none" -> None, "all" -> None kabi xaritalashni o'zi bajaradi.
    tunable=True bo'lsa, tuning.py qidiruv oralig'ini (search_min/search_max/search_log yoki
    search_choices) ishlatadi.
    """
    name: str
    label: str
    kind: str
    default: object
    min: float = None
    max: float = None
    step: float = None
    decimals: int = 4
    choices: tuple = ()
    none_label: str = "avtomatik"
    tunable: bool = False
    search_min: float = None
    search_max: float = None
    search_log: bool = False
    search_choices: tuple = None
    group: str = ""
    tooltip: str = ""


def _S(*a, **k):
    return ParamSpec(*a, **k)


PARAM_SPECS = {
    # ------------------------------------------------------------------ RF
    "RandomForest": [
        _S("n_estimators", "Daraxtlar soni (n_estimators)", "int", 500, min=10, max=5000, step=50,
           tunable=True, search_min=100, search_max=1000, group="Daraxtlar",
           tooltip="Ko'proq daraxt - barqarorroq natija, lekin sekinroq."),
        _S("criterion", "Bo'linish mezoni (criterion)", "choice", "gini",
           choices=("gini", "entropy", "log_loss"), group="Daraxtlar"),
        _S("max_depth", "Maksimal chuqurlik (max_depth)", "optint", None, min=1, max=200, step=1,
           none_label="cheksiz", tunable=True, search_min=2, search_max=20, group="Daraxt shakli",
           tooltip="Cheksiz = daraxt to'liq o'sadi (overfitting xavfi kichik namunada)."),
        _S("min_samples_split", "Bo'linish uchun min. namuna (min_samples_split)", "int", 2, min=2, max=100,
           step=1, tunable=True, search_min=2, search_max=20, group="Daraxt shakli"),
        _S("min_samples_leaf", "Bargdagi min. namuna (min_samples_leaf)", "int", 2, min=1, max=100, step=1,
           tunable=True, search_min=1, search_max=10, group="Daraxt shakli"),
        _S("max_features", "Har bo'linishdagi feature'lar (max_features)", "choice", "sqrt",
           choices=("sqrt", "log2", "all"), tunable=True, search_choices=("sqrt", "log2", "all"),
           group="Daraxt shakli", tooltip="'all' = barcha feature'lar (None)."),
        _S("bootstrap", "Bootstrap namuna olish", "bool", True, group="Namuna olish"),
        _S("max_samples", "Bootstrap namuna ulushi (max_samples)", "optfloat", None, min=0.1, max=1.0,
           step=0.05, decimals=2, none_label="to'liq", group="Namuna olish",
           tooltip="Faqat bootstrap yoqilgan bo'lsa ishlaydi."),
        _S("class_weight", "Sinf og'irligi (class_weight)", "choice", "balanced",
           choices=("balanced", "balanced_subsample", "none"), group="Muvozanat",
           tooltip="'none' = og'irliksiz."),
        _S("ccp_alpha", "Kesish (cost-complexity) alpha", "float", 0.0, min=0.0, max=0.5, step=0.001,
           decimals=4, group="Regulyarizatsiya"),
    ],
    # ----------------------------------------------------------------- SVM
    "SVM": [
        _S("kernel", "Yadro (kernel)", "choice", "rbf", choices=("rbf", "linear", "poly", "sigmoid"),
           group="Yadro"),
        _S("C", "Jarima (C)", "float", 1.0, min=0.001, max=10000.0, step=0.5, decimals=3,
           tunable=True, search_min=0.01, search_max=100.0, search_log=True, group="Regulyarizatsiya"),
        _S("gamma", "Gamma", "optfloat", None, min=1e-6, max=100.0, step=0.01, decimals=6,
           none_label="'scale' (avtomatik)", tunable=True, search_min=1e-4, search_max=1.0,
           search_log=True, group="Yadro",
           tooltip="Bo'sh/avtomatik = 'scale' (1 / (n_features * X.var()))."),
        _S("degree", "Daraja (poly yadro uchun)", "int", 3, min=1, max=10, step=1, group="Yadro"),
        _S("class_weight", "Sinf og'irligi (class_weight)", "choice", "balanced",
           choices=("balanced", "none"), group="Muvozanat"),
    ],
    # ------------------------------------------------------------- XGBoost
    "XGBoost": [
        _S("n_estimators", "Daraxtlar soni (n_estimators)", "int", 300, min=10, max=5000, step=50,
           tunable=True, search_min=100, search_max=800, group="Boosting"),
        _S("learning_rate", "O'rganish tezligi (learning_rate)", "float", 0.05, min=0.001, max=1.0,
           step=0.01, decimals=4, tunable=True, search_min=0.01, search_max=0.3, search_log=True,
           group="Boosting"),
        _S("max_depth", "Maksimal chuqurlik (max_depth)", "int", 4, min=1, max=20, step=1,
           tunable=True, search_min=2, search_max=8, group="Daraxt shakli"),
        _S("min_child_weight", "Min. bola og'irligi (min_child_weight)", "float", 1.0, min=0.0, max=100.0,
           step=0.5, decimals=2, tunable=True, search_min=1.0, search_max=10.0, group="Daraxt shakli"),
        _S("subsample", "Qator ulushi (subsample)", "float", 0.8, min=0.1, max=1.0, step=0.05, decimals=2,
           tunable=True, search_min=0.5, search_max=1.0, group="Namuna olish"),
        _S("colsample_bytree", "Feature ulushi (colsample_bytree)", "float", 0.8, min=0.1, max=1.0,
           step=0.05, decimals=2, tunable=True, search_min=0.4, search_max=1.0, group="Namuna olish"),
        _S("gamma", "Min. loss kamayishi (gamma)", "float", 0.0, min=0.0, max=20.0, step=0.1, decimals=3,
           tunable=True, search_min=0.0, search_max=5.0, group="Regulyarizatsiya"),
        _S("reg_alpha", "L1 regulyarizatsiya (reg_alpha)", "float", 0.0, min=0.0, max=100.0, step=0.1,
           decimals=3, tunable=True, search_min=0.0, search_max=5.0, group="Regulyarizatsiya"),
        _S("reg_lambda", "L2 regulyarizatsiya (reg_lambda)", "float", 1.0, min=0.0, max=100.0, step=0.1,
           decimals=3, tunable=True, search_min=0.1, search_max=10.0, search_log=True,
           group="Regulyarizatsiya"),
        _S("scale_pos_weight", "Musbat sinf og'irligi (scale_pos_weight)", "optfloat", None, min=0.01,
           max=1000.0, step=0.5, decimals=3, none_label="avtomatik (neg/pos)", group="Muvozanat"),
        _S("tree_method", "Daraxt usuli (tree_method)", "choice", "hist",
           choices=("hist", "exact", "approx"), group="Boosting"),
    ],
    # ----------------------------------------------------------------- CNN
    "CNN": [
        _S("mode", "CNN turi", "choice", "patch2d", choices=("patch2d", "tabular1d"), group="Arxitektura",
           tooltip="patch2d - nuqta atrofidagi WxW piksellik oyna ustida 2D-CNN (tavsiya). "
                   "tabular1d - eski usul: feature'lar tartibi bo'yicha 1D-Conv (qatlam tartibiga bog'liq)."),
        _S("window", "Oyna o'lchami (piksel, toq son)", "int", 9, min=3, max=31, step=2,
           tunable=True, search_choices=(5, 7, 9, 11, 15), group="Arxitektura",
           tooltip="Faqat patch2d. Spatial blok o'lchami oyna kengligidan (window*piksel) katta bo'lishi kerak."),
        _S("filters1", "1-konv. filtrlar soni", "int", 16, min=2, max=256, step=2, tunable=True,
           search_choices=(8, 16, 32), group="Arxitektura"),
        _S("filters2", "2-konv. filtrlar soni (0 = o'chirilgan)", "int", 32, min=0, max=256, step=2,
           tunable=True, search_choices=(0, 16, 32, 64), group="Arxitektura"),
        _S("kernel_size", "Yadro o'lchami (kernel_size)", "int", 3, min=1, max=7, step=2, group="Arxitektura"),
        _S("dense_units", "Zich qatlam neyronlari", "int", 32, min=2, max=512, step=2, tunable=True,
           search_choices=(16, 32, 64), group="Arxitektura"),
        _S("dropout", "Dropout", "float", 0.3, min=0.0, max=0.9, step=0.05, decimals=2, tunable=True,
           search_min=0.1, search_max=0.6, group="Regulyarizatsiya"),
        _S("l2", "L2 regulyarizatsiya", "float", 0.0, min=0.0, max=0.1, step=0.0001, decimals=5,
           group="Regulyarizatsiya"),
        _S("optimizer", "Optimizator", "choice", "adam", choices=("adam", "sgd", "rmsprop"), group="O'qitish"),
        _S("learning_rate", "O'rganish tezligi", "float", 0.001, min=1e-5, max=0.5, step=0.0005, decimals=5,
           tunable=True, search_min=1e-4, search_max=1e-2, search_log=True, group="O'qitish"),
        _S("batch_size", "Batch o'lchami", "int", 8, min=1, max=512, step=1, tunable=True,
           search_choices=(4, 8, 16, 32), group="O'qitish"),
        _S("epochs", "Maks. epochlar soni", "int", 150, min=1, max=2000, step=10, group="O'qitish"),
        _S("patience", "Erta to'xtatish sabri (patience)", "int", 15, min=1, max=200, step=1, group="O'qitish",
           tooltip="Validatsiya loss'i shuncha epoch yaxshilanmasa, o'qitish to'xtaydi."),
        _S("val_fraction", "Validatsiya ulushi (early stopping uchun)", "float", 0.2, min=0.05, max=0.5,
           step=0.05, decimals=2, group="O'qitish",
           tooltip="O'qitish to'plamining shu ulushi (stratifikatsiya bilan) validatsiyaga ajratiladi."),
        _S("augment", "Augmentatsiya (flip/rot90, patch2d)", "bool", True, group="O'qitish"),
    ],
}


def get_spec(model, name):
    for s in PARAM_SPECS[model]:
        if s.name == name:
            return s
    raise KeyError(f"{model}.{name}")


def default_hyperparams():
    """Barcha modellar uchun standart giperparametrlar (yangi nusxa)."""
    return {m: {s.name: s.default for s in specs} for m, specs in PARAM_SPECS.items()}


def default_search_space(model):
    """{param: {"type": "int"|"float"|"choice", "min", "max", "log", "choices"}} - faqat tunable parametrlar."""
    space = {}
    for s in PARAM_SPECS[model]:
        if not s.tunable:
            continue
        if s.search_choices is not None:
            space[s.name] = {"type": "choice", "choices": list(s.search_choices)}
        else:
            t = "int" if s.kind in ("int", "optint") else "float"
            space[s.name] = {"type": t, "min": s.search_min, "max": s.search_max, "log": bool(s.search_log)}
    return space


# ---------------------------------------------------------------------------
# Giperparametr validatsiyasi
# ---------------------------------------------------------------------------
def _coerce(spec, value):
    """(qiymat, ogohlantirish|None). Tur va diapazonni majburlaydi."""
    msg = None
    if spec.kind in ("optint", "optfloat") and value in (None, "", "none", "None"):
        return None, None
    if spec.kind == "bool":
        return bool(value), None
    if spec.kind == "choice":
        v = str(value)
        if v not in spec.choices:
            return spec.default, f"{spec.name}: '{v}' ruxsat etilmagan, '{spec.default}' ishlatildi"
        return v, None
    try:
        v = int(round(float(value))) if spec.kind in ("int", "optint") else float(value)
    except (TypeError, ValueError):
        return spec.default, f"{spec.name}: '{value}' son emas, standart ishlatildi"
    if spec.min is not None and v < spec.min:
        v, msg = type(v)(spec.min), f"{spec.name}: {spec.min} gacha oshirildi"
    if spec.max is not None and v > spec.max:
        v, msg = type(v)(spec.max), f"{spec.name}: {spec.max} gacha kamaytirildi"
    return v, msg


def validate_hyperparams(hp):
    """hp (qisman bo'lishi mumkin) -> (to'liq va tozalangan hp, [ogohlantirishlar]).
    Noma'lum model/parametrlar tashlab yuboriladi; yetishmaganlari standart bilan to'ldiriladi."""
    clean = default_hyperparams()
    warnings = []
    for model, params in (hp or {}).items():
        if model not in PARAM_SPECS or not isinstance(params, dict):
            warnings.append(f"Noma'lum model e'tiborsiz qoldirildi: {model}")
            continue
        for spec in PARAM_SPECS[model]:
            if spec.name in params:
                v, w = _coerce(spec, params[spec.name])
                clean[model][spec.name] = v
                if w:
                    warnings.append(f"{model}.{w}")
    cnn = clean["CNN"]
    if cnn["window"] % 2 == 0:
        cnn["window"] += 1
        warnings.append("CNN.window toq bo'lishi kerak, +1 qilindi")
    if cnn["kernel_size"] % 2 == 0:
        cnn["kernel_size"] += 1
        warnings.append("CNN.kernel_size toq bo'lishi kerak, +1 qilindi")
    return clean, warnings


def merge_hyperparams(base, override):
    """base ustiga override (qisman) qo'yib, validatsiya qilingan yangi hp qaytaradi."""
    merged = copy.deepcopy(base) if base else default_hyperparams()
    for m, p in (override or {}).items():
        if isinstance(p, dict):
            merged.setdefault(m, {}).update(p)
    return validate_hyperparams(merged)[0]


def hp_summary_text(hp, models=None):
    """Logga yozish uchun ixcham matn."""
    lines = []
    for m in MODEL_NAMES:
        if models is not None and m not in models:
            continue
        p = hp.get(m, {})
        lines.append(f"  {m}: " + ", ".join(f"{k}={v}" for k, v in p.items()))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TuningConfig
# ---------------------------------------------------------------------------
@dataclass
class TuningConfig:
    enabled: bool = False
    mode: str = "nested"            # "nested": har bir tashqi fold ichida; final: butun ma'lumotda
    models: dict = field(default_factory=lambda: {"RandomForest": True, "SVM": True, "XGBoost": True,
                                                  "CNN": False})
    n_iter: int = 20                # tasodifiy qidiruv urinishlari soni
    inner_splits: int = 3           # ichki spatial CV fold'lari
    scoring: str = "roc_auc"        # "roc_auc" | "average_precision"
    spaces: dict = field(default_factory=dict)   # {model: {param: {"min","max","log"|"choices"}}} - qo'lda tahrir

    def resolved_space(self, model):
        """Standart qidiruv oralig'i + foydalanuvchi o'zgartirishlari (faqat tunable parametrlar)."""
        space = default_search_space(model)
        for p, ov in (self.spaces.get(model) or {}).items():
            if p in space and isinstance(ov, dict):
                space[p].update({k: v for k, v in ov.items() if k in ("min", "max", "log", "choices")})
        return space

    def to_dict(self):
        return copy.deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        known = {f.name for f in fields(cls)}
        obj = cls(**{k: copy.deepcopy(v) for k, v in d.items() if k in known})
        obj.inner_splits = int(max(2, obj.inner_splits))
        obj.n_iter = int(max(1, obj.n_iter))
        if obj.scoring not in ("roc_auc", "average_precision"):
            obj.scoring = "roc_auc"
        return obj


# ---------------------------------------------------------------------------
# RunConfig - GUI dan pipeline'ga uzatiladigan BARCHA sozlamalar
# ---------------------------------------------------------------------------
BACKGROUND_STRATEGIES = ("random", "grid", "distance_weighted")
CLASS_METHODS = ("quantile", "equal_interval", "fixed")


@dataclass
class RunConfig:
    # --- kirish
    tiff_folder: str = ""
    points_folder: str = ""
    aoi_folder: str = ""
    output_dir: str = ""
    categorical_layers: list = field(default_factory=list)   # band nomlari (fayl nomlari, kengaytmasiz)
    assume_crs_if_missing: bool = True
    # --- fon (pseudo-absence)
    n_background: int = 80
    min_distance: float = 500.0
    background_strategy: str = "random"
    final_bg_draws: int = 1          # yakuniy modelni K ta turli fon tanlovida o'qitib o'rtachalash
    # --- cross-validation
    n_splits: int = 5
    n_repeats: int = 10
    block_size: float = 0.0          # 0 = avtomatik (variogram)
    variogram_band: str = "auto"     # "auto" = barcha raqamli bandlar medianasi, yoki band nomi
    run_random_cv: bool = True       # benchmark sifatida random CV ham bajarilsinmi
    n_bootstrap: int = 1000          # blok-bootstrap CI uchun
    # --- modellar
    use_models: dict = field(default_factory=lambda: {m: True for m in MODEL_NAMES})
    hyperparams: dict = field(default_factory=default_hyperparams)
    calibrate: bool = True
    calibration_method: str = "sigmoid"   # "sigmoid" | "isotonic"
    calibration_cv: int = 3
    tuning: TuningConfig = field(default_factory=TuningConfig)
    # --- fon sezgirligi
    bg_sensitivity_enabled: bool = False
    bg_sensitivity_draws: int = 5
    bg_sensitivity_repeats: int = 3
    # --- talqin
    perm_importance: bool = True
    perm_importance_repeats: int = 5
    shap_enabled: bool = True
    shap_max_background: int = 50
    # --- umumiy
    seed: int = RANDOM_STATE
    n_jobs: int = field(default_factory=default_n_jobs)
    # --- xarita sinflash
    class_method: str = "quantile"
    n_classes: int = 5
    class_breaks: list = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])

    # ---------------------------------------------------------------- yordamchilar
    def enabled_models(self):
        """Yoqilgan modellar (MODEL_ORDER tartibida). Mavjud bo'lmagan kutubxonalar bu yerda
        tekshirilmaydi - pipeline tekshiradi."""
        return [m for m in MODEL_NAMES if self.use_models.get(m, False)]

    def to_dict(self):
        d = asdict(self)
        d["tuning"] = self.tuning.to_dict()
        d["preset_version"] = PRESET_VERSION
        return copy.deepcopy(d)

    @classmethod
    def from_dict(cls, d):
        """Qisman/eski lug'atdan RunConfig. Noma'lum kalitlar e'tiborsiz; hyperparams validatsiya qilinadi."""
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        kwargs = {k: copy.deepcopy(v) for k, v in d.items() if k in known}
        cfg = cls(**kwargs)
        cfg.hyperparams = validate_hyperparams(cfg.hyperparams)[0]
        cfg.tuning = TuningConfig.from_dict(d.get("tuning") if isinstance(d.get("tuning"), dict)
                                            else (cfg.tuning.to_dict() if isinstance(cfg.tuning, TuningConfig)
                                                  else {}))
        um = {m: bool(cfg.use_models.get(m, True)) for m in MODEL_NAMES} if isinstance(cfg.use_models, dict) \
            else {m: True for m in MODEL_NAMES}
        cfg.use_models = um
        return cfg

    def validate(self):
        """Foydalanuvchiga ko'rsatiladigan muammolar ro'yxati (bo'sh = hammasi joyida)."""
        problems = []
        if not self.enabled_models():
            problems.append("Kamida bitta model yoqilgan bo'lishi kerak.")
        if self.n_splits < 2:
            problems.append("K-fold soni kamida 2 bo'lishi kerak.")
        if self.n_repeats < 1:
            problems.append("CV takrorlash soni kamida 1 bo'lishi kerak.")
        if self.n_background < 5:
            problems.append("Fon nuqtalar soni juda kam.")
        if self.background_strategy not in BACKGROUND_STRATEGIES:
            problems.append(f"Fon strategiyasi noto'g'ri: {self.background_strategy}")
        if self.calibration_method not in ("sigmoid", "isotonic"):
            problems.append(f"Kalibrlash usuli noto'g'ri: {self.calibration_method}")
        if self.class_method not in CLASS_METHODS:
            problems.append(f"Sinflash usuli noto'g'ri: {self.class_method}")
        if self.class_method == "fixed" and len(self.class_breaks) < 1:
            problems.append("'fixed' sinflash uchun chegaralar kerak.")
        if self.tuning.enabled and not any(self.tuning.models.get(m) and self.use_models.get(m)
                                           for m in MODEL_NAMES):
            problems.append("Tuning yoqilgan, lekin yoqilgan modellar orasida tuning uchun belgilangani yo'q.")
        return problems


# ---------------------------------------------------------------------------
# Preset (JSON) saqlash/yuklash
# ---------------------------------------------------------------------------
def save_preset(path, hyperparams=None, tuning=None, cfg=None):
    """Faqat giperparametrlar (+tuning) yoki to'liq RunConfig ni JSON'ga yozadi."""
    if cfg is not None:
        payload = {"kind": "run_config", **cfg.to_dict()}
    else:
        payload = {"kind": "hyperparams", "preset_version": PRESET_VERSION,
                   "hyperparams": validate_hyperparams(hyperparams or default_hyperparams())[0]}
        if tuning is not None:
            payload["tuning"] = tuning.to_dict() if isinstance(tuning, TuningConfig) else dict(tuning)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    return path


def load_preset(path):
    """JSON'ni o'qiydi. Qaytaradi: {"kind": "hyperparams"|"run_config", "hyperparams": hp,
    "tuning": TuningConfig|None, "cfg": RunConfig|None, "warnings": [...]}"""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    kind = payload.get("kind", "hyperparams" if "hyperparams" in payload else "run_config")
    if kind == "run_config":
        cfg = RunConfig.from_dict(payload)
        _, warns = validate_hyperparams(payload.get("hyperparams"))
        return {"kind": kind, "hyperparams": cfg.hyperparams, "tuning": cfg.tuning, "cfg": cfg, "warnings": warns}
    hp, warns = validate_hyperparams(payload.get("hyperparams"))
    tuning = TuningConfig.from_dict(payload["tuning"]) if isinstance(payload.get("tuning"), dict) else None
    return {"kind": kind, "hyperparams": hp, "tuning": tuning, "cfg": None, "warnings": warns}


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def to_jsonable(obj):
    """Ixtiyoriy (numpy/dataclass ichida) ob'ektni JSON'ga yozishga yaroqli ko'rinishga o'tkazadi."""
    return json.loads(json.dumps(obj, default=_json_default))
