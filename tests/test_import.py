"""Smoke test: the package and its modules import cleanly."""

import importlib


def test_package_imports() -> None:
    import medal_predictor

    assert medal_predictor.__version__


def test_all_modules_import() -> None:
    for name in ("data", "features", "leakage", "cv", "model", "evaluation", "betting"):
        importlib.import_module(f"medal_predictor.{name}")
