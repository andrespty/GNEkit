"""The multipliers setting: one multiplier per shared constraint, or one per player.

Game: min (x_i - 1)^2 over x_i in [0, 5], subject to the shared constraint
x_1 + x_2 <= 1. Every point with x_1 + x_2 = 1 and x in [0, 1]^2 is a GNE,
with multipliers lambda_i = 2 (1 - x_i). The variational equilibrium has
equal multipliers, so it is x = (0.5, 0.5).
"""
import io
import contextlib

import jax.numpy as jnp
import numpy as np
import pytest

from gnekit import BaseProblem, EnergyMethod, FBEnergyMethod, Player


def f1(x):
    return (x[0][0, 0] - 1.0) ** 2


def f2(x):
    return (x[1][0, 0] - 1.0) ** 2


def g(x):
    return x[0][0, 0] + x[1][0, 0] - 1.0


PLAYERS = [
    Player(name="P1", size=1, f_index=0, constraints=[0], bounds=(0.0, 5.0)),
    Player(name="P2", size=1, f_index=1, constraints=[0], bounds=(0.0, 5.0)),
]


def build(algorithm=EnergyMethod, **settings):
    return algorithm([f1, f2], [g], PLAYERS, disp=False, **settings)


def kkt(alg, x, lambdas):
    with contextlib.redirect_stdout(io.StringIO()):
        report = alg.check_kkt(jnp.array(x), jnp.array(lambdas))
    return all(r["is_kkt"] for r in report.values())


def test_shared_is_the_default():
    alg = build()
    assert alg.multipliers == "shared"
    assert alg.multiplier_pairs == [(None, 0)]
    assert len(alg.bounds_dual) == 1


def test_per_player_has_one_multiplier_per_player_and_constraint():
    alg = build(multipliers="per_player")
    assert alg.multiplier_pairs == [(0, 0), (1, 0)]
    assert len(alg.bounds_dual) == 2
    assert alg.dual_index(1, 0) == 1


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="multipliers"):
        build(multipliers="both")


def test_gne_that_is_not_a_variational_equilibrium():
    x = [0.25, 0.75]
    assert kkt(build(multipliers="per_player"), x, [1.5, 0.5])
    for lam in [0.5, 1.0, 1.5]:
        assert not kkt(build(), x, [lam])


def test_variational_equilibrium_passes_in_both_modes():
    x = [0.5, 0.5]
    assert kkt(build(), x, [1.0])
    assert kkt(build(multipliers="per_player"), x, [1.0, 1.0])


def test_equal_multipliers_give_the_same_primal_energy():
    actions = jnp.array([[0.3], [0.4]])
    shared = build().lagrange_gradient(actions, jnp.array([[0.7]]))
    per_player = build(multipliers="per_player").lagrange_gradient(actions, jnp.array([[0.7], [0.7]]))
    np.testing.assert_allclose(shared, per_player)


def test_initial_point_is_expanded():
    alg = build(multipliers="per_player")
    np.testing.assert_allclose(alg.expand_initial_point(jnp.array([0.1, 0.2, 0.3])), [0.1, 0.2, 0.3, 0.3])
    assert build().expand_initial_point(jnp.array([0.1, 0.2, 0.3])).shape == (3,)


@pytest.mark.parametrize("algorithm", [EnergyMethod, FBEnergyMethod])
def test_merit_function_is_finite_in_per_player_mode(algorithm):
    alg = build(algorithm, multipliers="per_player")
    ip = alg.extend_initial_point(alg.expand_initial_point(jnp.array([0.3, 0.4, 0.1])))
    assert ip.shape == (len(alg.bounds_all),)
    assert np.isfinite(alg.min_func(ip))


def test_problem_solve_forwards_the_setting():
    class Game(BaseProblem):
        def define_players(self):
            return PLAYERS

        def objectives(self):
            return [f1, f2]

        def constraints(self):
            return [g]

    problem = Game()
    problem.set_initial_point(0.3, 0.1)
    with contextlib.redirect_stdout(io.StringIO()):
        primal, dual = problem.solve(EnergyMethod, multipliers="per_player", niter=1, disp=False)
    assert primal.shape == (2,)
    assert dual.shape == (2,)
