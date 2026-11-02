"""Deterministic snapshot of a problem: structure and function values at fixed points.

No solver is run. Every number depends only on the problem definition and on
the algorithm's merit function, so the snapshot must not change while the
code is reorganized.
"""
import contextlib
import io

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _points(problem, seed=0):
    """Two primal points inside the bounds: the box midpoint and a seeded random point."""
    rng = np.random.default_rng(seed)
    mid, rand = [], []
    for p in problem.players:
        for lb, ub in p.get_full_bounds():
            lo = lb if np.isfinite(lb) else 0.0
            hi = ub if np.isfinite(ub) else lo + 2.0
            mid.append(0.5 * (lo + hi))
            rand.append(lo + (hi - lo) * (0.1 + 0.8 * rng.random()))
    return {"mid": np.array(mid), "rand": np.array(rand)}


def _split(problem, flat):
    out, k = [], 0
    for p in problem.players:
        out.append(jnp.asarray(flat[k:k + p.size]).reshape(-1, 1))
        k += p.size
    return out


def _num(x):
    return [float(v) for v in np.ravel(np.asarray(x, dtype=float))]


def problem_snapshot(problem):
    snap = {
        "players": [
            {
                "name": p.name, "size": p.size, "f_index": p.f_index,
                "constraints": [c for c in p.constraints],
                "bounds": [list(b) for b in p.get_full_bounds()],
            }
            for p in problem.players
        ],
        "n_objectives": len(problem.objectives()),
        "n_constraints": len(problem.constraints()),
        "values": {},
    }
    for tag, flat in _points(problem).items():
        x = _split(problem, flat)
        objs = problem.objectives()
        cons = problem.constraints()
        snap["values"][tag] = {
            "point": _num(flat),
            "objectives": [float(f(x)) for f in objs],
            "constraints": [float(g(x)) for g in cons],
            "objective_grads": [
                _num(jnp.concatenate([gi.ravel() for gi in jax.grad(f)(x)])) for f in objs
            ],
        }
    return snap


def algorithm_snapshot(problem, Algorithm):
    """Merit function at fixed primal-dual points (dual = 0.1 for every multiplier)."""
    alg = Algorithm(problem.objectives(), problem.constraints(), problem.players)
    n_dual = len(alg.bounds_all) - alg.total_actions
    out = {}
    for tag, flat in _points(problem).items():
        x = np.concatenate([flat, np.full(n_dual, 0.1)])
        out[tag] = float(alg.min_func(x))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report = alg.check_kkt(jnp.asarray(flat), jnp.full(n_dual, 0.1))
        # Some algorithms only print the report, so the printed text is always kept.
        out[f"kkt_{tag}"] = {
            "printed": buf.getvalue(),
            "report": None if report is None else {
                name: {k: (bool(v) if k == "is_kkt" else float(v)) for k, v in res.items()}
                for name, res in report.items()
            },
        }
    return out
