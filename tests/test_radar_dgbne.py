"""RadarGameDGBNE: the paper's radar example."""
import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from gnekit.problems.bayesian import RadarGameDGBNE
from gnekit.problems.bayesian import RadarGame_DGBNE as radar
from gnekit.core.utils import construct_vectors

N, M = radar.N, radar.M


def random_profile(seed=0):
    return np.random.default_rng(seed).uniform(0, 1, (N, M))


def as_vectors(A):
    return construct_vectors(jnp.asarray(A.reshape(-1, 1)), [M] * N)


def brute_force_ex_ante(i, A):
    """Upsilon_i by an explicit loop over the M^N type profiles."""
    total = 0.0
    for js in itertools.product(range(M), repeat=N):
        prob = np.prod([radar.ETA[l, js[l]] for l in range(N)])
        Y = sum(radar.ALPHA * radar.THETA[l, js[l]] * A[l, js[l]] for l in range(N) if l != i)
        a = A[i, js[i]]
        u = radar.THETA[i, js[i]] * a / (radar.BETA[i] * a + Y + radar.SIGMA2) - radar.Q_COST * a
        total += prob * u
    return total


def test_exact_objective_matches_enumeration():
    A = random_profile()
    objectives = RadarGameDGBNE().objectives()
    for i in range(N):
        assert float(objectives[i](as_vectors(A))) == pytest.approx(-brute_force_ex_ante(i, A), rel=1e-12)


@pytest.mark.parametrize("R", [40, 60])
def test_approx_objective_is_within_the_approximation_error(R):
    A = random_profile(1)
    exact = RadarGameDGBNE().objectives()
    approx = RadarGameDGBNE(mode="approx", R=R).objectives()
    eps = radar.paper_quantities(R)["eps"]
    for i in range(N):
        gap = abs(float(approx[i](as_vectors(A))) - float(exact[i](as_vectors(A))))
        assert gap <= eps


def test_constraints_are_interim_caps_per_player_and_type():
    problem = RadarGameDGBNE()
    A = random_profile(2)
    constraints = problem.constraints()
    assert len(constraints) == N * M
    for i, player in enumerate(problem.players):
        for j, c_idx in enumerate(player.constraints):
            expected = radar.interim_constraint(i, j, list(A), xp=np)
            assert float(constraints[c_idx](as_vectors(A))) == pytest.approx(expected, rel=1e-12)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        RadarGameDGBNE(mode="fast")
