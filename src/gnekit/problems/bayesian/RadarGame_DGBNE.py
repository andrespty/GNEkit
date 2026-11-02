"""Multistatic radar power game with an interim interference cap (D-GBNE paper example).

Five radars, three types each. Radar i with type theta_i^j chooses power
a_i^j in [0, 1] and maximizes

    u_i = theta_i a_i / (beta_i a_i + Y_i + sigma^2) - q a_i,
    Y_i = alpha sum_{l != i} theta_l a_l,

in expectation over independent types, subject to the interim cap

    h_i(a_i^j, x_{-i} | theta_i^j) = theta_i^j a_i^j
        + alpha sum_{l != i} sum_j' eta_l^j' theta_l^j' a_l^j' - b <= 0.

The game can be solved exactly (``mode="exact"``) or with the separable
exponential sum approximation of 1/y (``mode="approx"``), which costs
O(R N M) per player instead of O(M^N). The module also provides the
quantities reported in the paper (error constants, uniqueness test, error
bound, binding pattern).
"""
from gnekit.core.types import VectorList
from gnekit.bayesian.player import BayesianPlayer
from gnekit.bayesian.problem import BayesianProblem
import jax.numpy as jnp
import numpy as np

THETA = np.array([[0.25, 0.5, 0.75],
                  [0.5, 1.0, 1.5],
                  [0.75, 1.25, 2.0],
                  [1.0, 1.75, 2.5],
                  [1.5, 2.25, 3.0]])
ETA = np.array([[0.5, 0.3, 0.2],
                [0.2, 0.5, 0.3],
                [0.3, 0.4, 0.3],
                [0.4, 0.4, 0.2],
                [0.2, 0.3, 0.5]])
BETA = np.array([0.40, 0.45, 0.50, 0.55, 0.60])   # self-interference beta_i
SIGMA2 = 1.0                                      # noise sigma^2
Q_COST = 0.4                                      # power cost q
ALPHA = 0.015                                     # cross gain ratio alpha
B_CAP = 1.0                                       # interference cap b
N, M = THETA.shape                                # 5 radars, 3 types each


def Y_max():
    """Y_i^max = alpha sum_{l != i} max_j theta_l^j."""
    return np.array([ALPHA * sum(THETA[l].max() for l in range(N) if l != i)
                     for i in range(N)])


def y_max():
    """y_max = max_i (beta_i + Y_i^max + sigma^2), the largest denominator."""
    return float(np.max(BETA + Y_max() + SIGMA2))


def exp_sum_nodes(R, y_min=SIGMA2, y_hi=None, tol=1e-14):
    """Trapezoidal rule for 1/y = int exp(s - e^s y) ds (Beylkin and Monzon).

    The truncation [s_lo, s_hi] makes both tails smaller than tol on [y_min, y_hi],
    so the error is controlled by the step h, which shrinks as R grows.
    Returns (t_nu, omega_nu), each of shape (R,).
    """
    s_lo = np.log(tol)                                  # left tail ~ e^{s_lo}
    s_hi = np.log(np.log(1.0 / tol) / y_min) + 1.0      # right tail ~ exp(-e^{s_hi} y_min)
    s = np.linspace(s_lo, s_hi, R)
    h = s[1] - s[0]
    w = np.full(R, h)
    w[0] *= 0.5
    w[-1] *= 0.5
    return np.exp(s), w * np.exp(s)


def E_errors(t, om, npts=4001):
    """E_0, E_1, E_2: sup errors of f_R, f_R', f_R'' against 1/y, -1/y^2, 2/y^3."""
    y = np.linspace(SIGMA2, y_max(), npts)[:, None]
    ex = np.exp(-t[None, :] * y)
    f0, f1, f2 = ex @ om, -(ex * t) @ om, (ex * t ** 2) @ om
    y = y[:, 0]
    return (float(np.max(np.abs(f0 - 1 / y))),
            float(np.max(np.abs(f1 + 1 / y ** 2))),
            float(np.max(np.abs(f2 - 2 / y ** 3))))


# ---------------------------------------------------------------------------
# Ex-ante utilities, written for any array module xp (jnp in the solver, np for checks)
# A has shape (N, M): A[l, j] = a_l^j
# ---------------------------------------------------------------------------
def _grid(vec, k, xp):
    shape = [1] * N
    shape[k] = M
    return xp.reshape(vec, shape)


def ex_ante_exact(i, A, xp=jnp):
    """Upsilon_i(x) by direct enumeration over the M^N type profiles."""
    a_i = _grid(A[i], i, xp)
    th_i = _grid(xp.asarray(THETA[i]), i, xp)
    Y = 0.0
    prob = 1.0
    for l in range(N):
        prob = prob * _grid(xp.asarray(ETA[l]), l, xp)
        if l != i:
            Y = Y + ALPHA * _grid(xp.asarray(THETA[l]), l, xp) * _grid(A[l], l, xp)
    u = th_i * a_i / (BETA[i] * a_i + Y + SIGMA2) - Q_COST * a_i
    return xp.sum(prob * u)


def ex_ante_approx(i, A, t, om, xp=jnp):
    """Upsilon_hat_i(x) from u_hat_i in the separable form (Proposition prop:evaluation_cost).

    Upsilon_hat_i = sum_j eta_i^j theta_i^j a_i^j sum_nu omega_nu exp(-t_nu (beta_i a_i^j + sigma^2))
                        prod_{l != i} phibar_l[nu]  -  q sum_j eta_i^j a_i^j,
    phibar_l[nu] = sum_j' eta_l^j' exp(-t_nu alpha theta_l^j' a_l^j').
    Cost O(R N M) per player.
    """
    t = xp.asarray(t)
    om = xp.asarray(om)
    rivals = xp.ones_like(t)
    for l in range(N):
        if l == i:
            continue
        ex = xp.exp(-t[:, None] * (ALPHA * xp.asarray(THETA[l]) * A[l])[None, :])   # (R, M)
        rivals = rivals * (ex @ xp.asarray(ETA[l]))                                 # (R,)
    own = xp.exp(-t[:, None] * (BETA[i] * A[i][None, :] + SIGMA2))                # (R, M)
    inner = (om * rivals) @ own                                                     # (M,)
    eta_i = xp.asarray(ETA[i])
    return xp.sum(eta_i * xp.asarray(THETA[i]) * A[i] * inner) - Q_COST * xp.sum(eta_i * A[i])


def interim_constraint(i, j, A, xp=jnp):
    """h_i(a_i^j, x_{-i} | theta_i^j) <= 0."""
    rival_use = 0.0
    for l in range(N):
        if l != i:
            rival_use = rival_use + ALPHA * xp.sum(xp.asarray(ETA[l] * THETA[l]) * A[l])
    return THETA[i, j] * A[i][j] + rival_use - B_CAP


# ---------------------------------------------------------------------------
# Problem class for the solver library
# ---------------------------------------------------------------------------
class RadarGameDGBNE(BayesianProblem):
    """
    Radar power game of the D-GBNE paper.

    Parameters
    ----------
    players : list of BayesianPlayer, optional
        Overrides the default players.
    mode : {"exact", "approx"}, optional
        ``"exact"`` enumerates the M^N type profiles. ``"approx"`` uses the
        exponential sum approximation with R nodes. Defaults to ``"exact"``.
    R : int, optional
        Number of exponential sum nodes for ``mode="approx"``. Defaults to 60.
    """
    def __init__(self, players=None, mode="exact", R=60):
        if mode not in ("exact", "approx"):
            raise ValueError(f"mode must be 'exact' or 'approx', got {mode!r}")
        self.mode = mode
        self.R = R
        t, om = exp_sum_nodes(R)
        self.t_nodes = jnp.asarray(t)
        self.omega = jnp.asarray(om)
        super().__init__(players)

    def define_players(self):
        return BayesianPlayer.batch_create(
            [M] * N,                                        # vector sizes
            [THETA[i].tolist() for i in range(N)],          # type values
            list(range(N)),                                 # objective indices
            [[i + j * N for j in range(M)] for i in range(N)],  # constraint indices
            [(0, 1) for _ in range(N)],                     # A_i(theta_i^j) = [0, 1]
            type_probs=[ETA[i].tolist() for i in range(N)],
        )

    def _A(self, x: VectorList):
        actions = self.split_profiles(x)
        return [jnp.reshape(actions[l], (-1,)) for l in range(N)]

    def objectives(self):
        def obj(x: VectorList, i: int):
            A = self._A(x)
            if self.mode == "exact":
                return -ex_ante_exact(i, A)
            return -ex_ante_approx(i, A, self.t_nodes, self.omega)

        return [lambda x, i=i: obj(x, i) for i in range(N)]

    def constraints(self):
        def g(x: VectorList, i: int, j: int):
            return jnp.reshape(interim_constraint(i, j, self._A(x)), ())

        # index i + j * N, matching define_players
        return [lambda x, i=i, j=j: g(x, i, j) for j in range(M) for i in range(N)]


# ---------------------------------------------------------------------------
# Paper quantities
# ---------------------------------------------------------------------------
def uniqueness_constants(ngrid=401):
    """Constants of rem:weak_coupling_constants and the test of cor:weak_coupling."""
    a = np.linspace(0, 1, ngrid)[:, None]
    mu = np.zeros((N, M))
    ell = np.zeros((N, M))
    for i in range(N):
        Y = np.linspace(0, Y_max()[i], ngrid)[None, :]
        y = BETA[i] * a + Y + SIGMA2
        rival_norm = np.sqrt(sum(np.sum((ETA[l] * THETA[l]) ** 2) for l in range(N) if l != i))
        for j in range(M):
            th = THETA[i, j]
            mu[i, j] = np.min(2 * th * BETA[i] * (Y + SIGMA2) / y ** 3)
            ell[i, j] = ALPHA * rival_norm * np.max(th * np.abs(BETA[i] * a - Y - SIGMA2) / y ** 3)
    Lx = np.sqrt(np.sum((ell / mu) ** 2, axis=1))
    Lw = np.sum(ETA * (ALPHA * THETA) * ell / mu, axis=1)       # ||Q~_i(theta_i^j)|| = alpha theta_i^j
    Lc = np.sqrt(np.sum(THETA ** -2.0, axis=1))
    kappa = np.full(N, 1 / ALPHA - 1)                           # gamma_i = alpha
    O = np.ones((N, N)) - np.eye(N)
    G = np.diag(Lx + Lc * Lw) @ O
    IG = np.linalg.inv(np.eye(N) - G)
    V = np.hstack([np.ones((N, 1)), np.eye(N)])
    Dk = np.diag(np.concatenate([[1.0], kappa]))
    Nm = V.T @ np.diag(Lw) @ O @ IG @ np.diag(Lc) @ V
    return dict(mu=mu, ell=ell, Lc=Lc, Lx=Lx, Lw=Lw, kappa=kappa, O=O, G=G, IG=IG, V=V,
                rhoG=float(np.max(np.abs(np.linalg.eigvals(G)))),
                lam_min=float(np.min(np.linalg.eigvalsh(Dk - 0.5 * (Nm + Nm.T)))))


def concavity_ok(E1, E2, ngrid=401):
    """Condition eq:radar_concavity for every i."""
    a = np.linspace(0, 1, ngrid)[:, None]
    for i in range(N):
        Y = np.linspace(0, Y_max()[i], ngrid)[None, :]
        rhs = np.min(2 * (Y + SIGMA2) / (BETA[i] * a + Y + SIGMA2) ** 3)
        if not 2 * E1 + BETA[i] * E2 < rhs:
            return False
    return True


def error_bound(eps, C):
    """Bound of thm:approx_error_packing."""
    eps_x = eps * np.sqrt(np.sum(C["mu"] ** -2.0, axis=1))
    eps_w = eps * np.sum(ETA * (ALPHA * THETA) / C["mu"], axis=1)
    eps_tot = C["Lc"] * eps_w + eps_x
    eps_V = C["V"].T @ (np.diag(C["Lw"]) @ C["O"] @ C["IG"] @ eps_tot + eps_w)
    A = C["IG"] @ np.diag(C["Lc"]) @ C["V"]
    return float(np.linalg.norm(A, 2) * np.linalg.norm(eps_V) / C["lam_min"]
                 + np.linalg.norm(C["IG"] @ eps_tot))


def paper_quantities(R, C=None):
    C = uniqueness_constants() if C is None else C
    t, om = exp_sum_nodes(R)
    E0, E1, E2 = E_errors(t, om)
    eps = float(np.max(THETA * (E0 + BETA[:, None] * E1)))
    return dict(R=R, E0=E0, E1=E1, E2=E2, eps=eps, concave=concavity_ok(E1, E2),
                bound=error_bound(eps, C), rhoG=C["rhoG"], lam_min=C["lam_min"])


def binding_pattern(A, tol=1e-6):
    """Boolean (N, M) array: True where h_i(a_i^j, x_{-i} | theta_i^j) = 0 (asterisks in tab:ex_radar_actions)."""
    A = [np.asarray(A[l]).reshape(-1) for l in range(N)]
    return np.array([[abs(interim_constraint(i, j, A, xp=np)) < tol for j in range(M)]
                     for i in range(N)])


def table_row(R, x_star, x_hat, C=None):
    """One row of tab:ex_radar_error: R, epsilon, bound, observed ||x_hat - x_star||.
    x_star, x_hat: (N, M) arrays of equilibrium actions of the exact and approximate games."""
    P = paper_quantities(R, C)
    err = float(np.linalg.norm(np.asarray(x_hat) - np.asarray(x_star)))
    return dict(R=R, eps=P["eps"], bound=P["bound"], observed=err, ratio=err / P["bound"],
                concave=P["concave"])
