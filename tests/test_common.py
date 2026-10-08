# -*- coding: utf-8 -*-
"""mpm.common testlari: ixtiyoriy kutubxonalarni kechiktirib yuklash va import xatosini eslab qolish."""
from __future__ import annotations

import importlib
import sys

import pytest

from mpm import common


@pytest.fixture
def clean_cache(monkeypatch):
    """_lazy_import keshini va xato lug'atini sinov uchun ajratadi (haqiqiy xgboost/shap keshi buzilmasin)."""
    monkeypatch.setattr(common, "_OPTIONAL_CACHE", {})
    monkeypatch.setattr(common, "_OPTIONAL_ERRORS", {})


def test_failed_import_keeps_error_text(clean_cache, monkeypatch):
    """Regressiya: find_spec bor, lekin import yiqilgan bo'lsa None keshlanadi VA xato matni saqlanadi."""
    monkeypatch.setattr(common, "_has_spec", lambda name: True)
    calls = []

    def boom(name, *a, **k):
        calls.append(name)
        raise ImportError("libfoo.so.1: cannot open shared object file")

    monkeypatch.setattr(importlib, "import_module", boom)
    assert common._lazy_import("fakemod_x") is None
    err = common.import_error("fakemod_x")
    assert err == "ImportError: libfoo.so.1: cannot open shared object file"
    assert common._lazy_import("fakemod_x") is None and calls == ["fakemod_x"]      # keshlangan: qayta urinmaydi
    assert common.import_error("fakemod_x") == err


def test_import_error_is_none_for_missing_module_and_success(clean_cache):
    assert common._lazy_import("mpm_yoq_bunday_modul_123") is None       # o'rnatilmagan: find_spec yo'q
    assert common.import_error("mpm_yoq_bunday_modul_123") is None       # xato emas, shunchaki yo'q
    assert common._lazy_import("json") is sys.modules["json"]
    assert common.import_error("json") is None
    assert common.import_error("hech_qachon_so'ralmagan") is None


def test_import_error_other_exception_types_and_truncation(clean_cache, monkeypatch):
    monkeypatch.setattr(common, "_has_spec", lambda name: True)

    def boom(name, *a, **k):
        raise OSError("x" * 1000 + "\nikkinchi qator")

    monkeypatch.setattr(importlib, "import_module", boom)
    assert common._lazy_import("fakemod_y") is None
    err = common.import_error("fakemod_y")
    assert err.startswith("OSError: xxx") and len(err) <= 300 and err.endswith("...") and "\n" not in err


def test_getters_return_none_on_broken_module(clean_cache, monkeypatch):
    monkeypatch.setattr(common, "_has_spec", lambda name: True)

    def boom(name, *a, **k):
        raise RuntimeError(f"{name} buzilgan")

    monkeypatch.setattr(importlib, "import_module", boom)
    assert common.get_xgboost() is None and common.get_shap() is None and common.get_tf() is None
    assert common.import_error("xgboost") == "RuntimeError: xgboost buzilgan"
    assert common.import_error("shap") == "RuntimeError: shap buzilgan"
    assert common.import_error("tensorflow") == "RuntimeError: tensorflow buzilgan"
