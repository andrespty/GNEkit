"""The exact merit gradient passed to the local minimizer."""
import numpy as np
import jax.numpy as jnp
import pytest
from scipy.optimize import approx_fprime

from _compat import PROBLEMS
from gnekit import BaseAlgorithm, EnergyMethod, FBEnergyMethod, Player


@pytest.mark.parametrize("algorithm", [EnergyMethod, FBEnergyMethod])
@pytest.mark.parametrize("name", ["A1", "A3", "A12", "QuadraticGame"])
def test_jacobian_matches_finite_differences(algorithm, name):
    problem = PROBLEMS[name]()
    alg = algorithm(problem.objectives(), problem.constraints(), problem.players, disp=False)
    bounds = np.array(alg.bounds_all, dtype=float)
    lo = np.where(np.isfinite(bounds[:, 0]), bounds[:, 0], -5.0)
    hi = np.where(np.isfinite(bounds[:, 1]), bounds[:, 1], 5.0)
    x = lo + (hi - lo) * np.random.default_rng(0).uniform(0.1, 0.9, len(lo))
    fd = approx_fprime(x, alg.min_func, 1e-7)
    np.testing.assert_allclose(alg.merit_jacobian(x), fd, rtol=1e-4, atol=1e-4 * (1 + np.abs(fd).max()))


def test_nonfinite_entries_fall_back_to_finite_differences():
    # f1 = x^1.5 has an infinite second derivative at x = 0, so the exact
    # merit gradient is not finite there
    players = [Player(name="P1", size=1, f_index=0, constraints=[None], bounds=(0.0, 1.0))]
    alg = EnergyMethod([lambda x: x[0][0, 0] ** 1.5], [], players, disp=False)
    x = np.array([0.0])
    assert not np.all(np.isfinite(alg.grad_min_func(jnp.asarray(x))))
    assert np.all(np.isfinite(alg.merit_jacobian(x)))


def test_custom_min_func_disables_jac():
    class Custom(BaseAlgorithm):
        def min_func(self, x):
            return float(np.sum(np.asarray(x) ** 2))

    players = [Player(name="P1", size=1, f_index=0, constraints=[None], bounds=(0.0, 1.0))]
    alg = Custom([lambda x: x[0][0, 0] ** 2], [], players, disp=False)
    assert alg.jac is False
