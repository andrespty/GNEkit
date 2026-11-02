"""check_kkt handles variable bounds by projection when there are no bound multipliers."""
import jax.numpy as jnp
import pytest

from gnekit import EnergyMethod, FBEnergyMethod, Player


def players(bounds):
    return [
        Player(name="P1", size=1, f_index=0, constraints=[None], bounds=bounds),
        Player(name="P2", size=1, f_index=1, constraints=[None], bounds=bounds),
    ]


def f1(x):
    return (x[0][0, 0] - 2.0) ** 2


def f2(x):
    return (x[1][0, 0] - 0.5) ** 2


def report(algorithm_cls, x, lambdas, bounds=(0.0, 1.0)):
    alg = algorithm_cls([f1, f2], [], players(bounds), disp=False)
    return alg.check_kkt(jnp.array(x), jnp.array(lambdas))


def test_solution_at_upper_bound_is_kkt():
    # P1 wants 2 but is capped at 1, P2 is interior at 0.5
    r = report(EnergyMethod, [1.0, 0.5], [])
    assert r["P1"]["stationarity"] == pytest.approx(0.0)
    assert r["P1"]["is_kkt"] and r["P2"]["is_kkt"]


def test_interior_point_with_nonzero_gradient_is_not_kkt():
    r = report(EnergyMethod, [0.9, 0.5], [])
    assert r["P1"]["stationarity"] == pytest.approx(0.1)
    assert not r["P1"]["is_kkt"]


def test_bound_violation_is_reported_as_infeasible():
    r = report(EnergyMethod, [1.2, 0.5], [])
    assert r["P1"]["primal_feas"] == pytest.approx(0.2)
    assert not r["P1"]["is_kkt"]


def test_infinite_bounds_reduce_to_gradient_norm():
    r = report(EnergyMethod, [1.0, 0.5], [], bounds=(-jnp.inf, jnp.inf))
    assert r["P1"]["stationarity"] == pytest.approx(2.0)
    assert r["P1"]["primal_feas"] == 0.0


def test_bound_multipliers_are_used_when_present():
    # FB carries mu_lb and mu_ub. mu_ub = 2 balances grad f1 = -2 at x1 = 1
    lambdas = [0.0, 0.0, 2.0, 0.0]  # [mu_lb (P1, P2), mu_ub (P1, P2)]
    r = report(FBEnergyMethod, [1.0, 0.5], lambdas)
    assert r["P1"]["is_kkt"] and r["P2"]["is_kkt"]
