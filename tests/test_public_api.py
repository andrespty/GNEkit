import importlib
import subprocess
import sys

import pytest

import gnekit

PUBLIC = {
    "BaseProblem": "gnekit.core.problem",
    "Player": "gnekit.core.player",
    "BayesianProblem": "gnekit.bayesian.problem",
    "BayesianPlayer": "gnekit.bayesian.player",
    "BaseAlgorithm": "gnekit.algorithms.base",
    "EnergyMethod": "gnekit.algorithms.energy",
    "FBEnergyMethod": "gnekit.algorithms.fb_energy",
    "VectorList": "gnekit.core.types",
}


def test_all_lists_the_public_names():
    assert sorted(gnekit.__all__) == sorted(PUBLIC)


@pytest.mark.parametrize("name, module", sorted(PUBLIC.items()))
def test_public_name_is_the_library_object(name, module):
    assert getattr(gnekit, name) is getattr(importlib.import_module(module), name)


MODULES = [
    "gnekit",
    "gnekit.core", "gnekit.core.problem", "gnekit.core.player", "gnekit.core.validation",
    "gnekit.bayesian", "gnekit.bayesian.problem",
    "gnekit.algorithms", "gnekit.algorithms.base", "gnekit.algorithms.energy",
    "gnekit.algorithms.fb_energy", "gnekit.algorithms.vector_energy",
    "gnekit.problems.gnep", "gnekit.problems.bayesian",
]


@pytest.mark.parametrize("module", MODULES)
def test_each_module_imports_on_its_own(module):
    """A fresh interpreter per module, so a circular import cannot hide behind import order."""
    src = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src")
    code = f"import sys; sys.path.insert(0, {src!r}); import {module}"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
