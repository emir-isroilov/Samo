# -*- coding: utf-8 -*-
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from tests.synth import make_synthetic_project


@pytest.fixture(scope="session")
def synth_project(tmp_path_factory):
    """Kichik (120x120) sintetik loyiha: 6 raqamli qatlam, ~22 musbat nuqta."""
    return make_synthetic_project(str(tmp_path_factory.mktemp("synth")), size=120, n_layers=6)


@pytest.fixture(scope="session")
def synth_project_cat(tmp_path_factory):
    """Kategorik qatlamli (geology_cat) loyiha."""
    return make_synthetic_project(str(tmp_path_factory.mktemp("synth_cat")), size=100, n_layers=4,
                                  categorical=True, seed=3)
