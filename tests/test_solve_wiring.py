"""The solve loop is shared by all algorithms. These tests check what it hands to
SciPy without running a real optimization: basinhopping is replaced by a stub
that returns the initial point."""
import types

import numpy as np
import pytest

import gnekit.algorithms.base as base_module
from _compat import ALGORITHMS, PROBLEMS


@pytest.fixture
def fake_basinhopping(monkeypatch):
    calls = []

    def fake(func, x0, **kwargs):
        calls.append({"func": func, "x0": np.asarray(x0), **kwargs})
        return types.SimpleNamespace(x=np.asarray(x0))

    monkeypatch.setattr(base_module, "basinhopping", fake)
    return calls


@pytest.mark.parametrize("alg_name", ["EnergyMethod", "FBEnergyMethod"])
def test_default_settings_match_previous_behavior(fake_basinhopping, alg_name):
    problem = PROBLEMS["QuadraticGame"]()
    problem.set_initial_point(0.5, 0.1)
    problem.solve(ALGORITHMS[alg_name])

    (call,) = fake_basinhopping
    assert call["stepsize"] == 0.01
    assert call["niter"] == 1000
    assert call["interval"] == 1
    assert call["niter_success"] == 100
    assert call["disp"] is True
    assert call["minimizer_kwargs"]["method"] == "SLSQP"
    assert call["minimizer_kwargs"]["options"] == base_module.DEFAULT_MINIMIZER_OPTIONS["SLSQP"]
    assert callable(call["minimizer_kwargs"]["jac"])

    n_primal = sum(p.size for p in problem.players)
    n_constraints = len(problem.constraints())
    expected = [0.5] * n_primal + [0.1] * n_constraints
    if alg_name == "FBEnergyMethod":
        expected += [0.1] * (2 * n_primal)   # bound multipliers appended
    np.testing.assert_array_equal(call["x0"], expected)
    assert len(call["minimizer_kwargs"]["bounds"]) == len(expected)


def test_settings_are_forwarded(fake_basinhopping):
    problem = PROBLEMS["QuadraticGame"]()
    problem.set_initial_point(0.5, 0.1)
    problem.solve(ALGORITHMS["EnergyMethod"], niter=3, method="L-BFGS-B",
                  minimizer_options={"ftol": 1e-20})

    (call,) = fake_basinhopping
    assert call["niter"] == 3
    assert call["minimizer_kwargs"]["method"] == "L-BFGS-B"
    assert call["minimizer_kwargs"]["options"] == {"ftol": 1e-20}   # no defaults for L-BFGS-B


def test_user_options_override_the_defaults(fake_basinhopping):
    problem = PROBLEMS["QuadraticGame"]()
    problem.set_initial_point(0.5, 0.1)
    problem.solve(ALGORITHMS["EnergyMethod"], minimizer_options={"ftol": 1e-20})

    (call,) = fake_basinhopping
    expected = {**base_module.DEFAULT_MINIMIZER_OPTIONS["SLSQP"], "ftol": 1e-20}
    assert call["minimizer_kwargs"]["options"] == expected


def test_jac_can_be_turned_off(fake_basinhopping):
    problem = PROBLEMS["QuadraticGame"]()
    problem.set_initial_point(0.5, 0.1)
    problem.solve(ALGORITHMS["EnergyMethod"], jac=False)

    (call,) = fake_basinhopping
    assert "jac" not in call["minimizer_kwargs"]


@pytest.mark.parametrize("alg_name", ["EnergyMethod", "FBEnergyMethod"])
def test_dual_bound_sets_the_multiplier_bounds(fake_basinhopping, alg_name):
    problem = PROBLEMS["QuadraticGame"]()
    problem.set_initial_point(0.5, 0.1)
    problem.solve(ALGORITHMS[alg_name], dual_bound=7.0)

    (call,) = fake_basinhopping
    n_primal = sum(p.size for p in problem.players)
    dual_bounds = call["minimizer_kwargs"]["bounds"][n_primal:]
    assert dual_bounds and all(b == (0, 7.0) for b in dual_bounds)


def test_summary_prints_solution_blocks(fake_basinhopping, capsys):
    problem = PROBLEMS["QuadraticGame"]()
    problem.set_initial_point(0.5, 0.1)
    problem.solve(ALGORITHMS["FBEnergyMethod"])
    out = capsys.readouterr().out
    for label in ["Primal Actions", "Dual Actions", "Lower Bound Actions", "Upper Bound Actions",
                  "KKT VALIDATION"]:
        assert label in out
