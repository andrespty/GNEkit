from abc import ABC
from functools import partial
from scipy.optimize import approx_fprime, basinhopping
import timeit
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from gnekit.core.player import players_to_lists, Player
from gnekit.core.types import ConsFunction, ObjFunction
from gnekit.core.utils import construct_vectors, one_hot_encoding
from gnekit.core.validation import (
    validate_constraint_funcs,
    validate_obj_funcs,
    validate_player_functions,
    validate_player_list,
)

DEFAULT_MINIMIZER_OPTIONS = {
    "SLSQP": {"ftol": 1e-16, "maxiter": 1000},
}


class BaseAlgorithm(ABC):
    """
    Abstract base class for GNE-seeking algorithms.

    Handles all shared setup: derivative compilation, bound extraction,
    constraint encoding, the solve loop, and KKT validation.

    A merit-function algorithm implements two methods, `lagrange_gradient`
    (the primal part of the merit function) and `gradient_dual` (the dual
    part). The merit function minimized by the solver is the sum of both.
    Alternatively, a subclass may override `min_func` directly.

    Parameters
    ----------
    obj_funcs : list of ObjFunction
        Objective functions, one per objective in the problem.
    constraints : list of ConsFunction
        Shared constraint functions ``g(x) <= 0``.
    player_list : list of Player
        Players defining variable sizes, objective assignments, and
        constraint participation.
    validate : bool, optional
        Whether to validate objective and constraint functions on
        construction. Defaults to ``True``.
    method : str, optional
        Local minimizer passed to ``scipy.optimize.minimize`` inside
        basin-hopping. Defaults to ``"SLSQP"``.
    minimizer_options : dict, optional
        ``options`` passed to the local minimizer, for example
        ``{"ftol": 1e-20, "maxiter": 5000}``. They are merged over the
        defaults in `DEFAULT_MINIMIZER_OPTIONS` for the chosen method.
    jac : bool, optional
        Pass the exact gradient of the merit function, computed with JAX,
        to the local minimizer. Entries where it is not finite, for example
        where an objective has no second derivative on a bound, fall back
        to finite differences. ``False`` uses finite differences
        everywhere. Ignored when a subclass overrides `min_func`.
        Defaults to ``True``.
    dual_bound : float, optional
        Upper bound on every multiplier. Defaults to ``100``.
    niter, stepsize, interval, niter_success, disp : optional
        Basin-hopping settings. Defaults: ``1000``, ``0.01``, ``1``,
        ``100``, ``True``.
    multipliers : {"shared", "per_player"}, optional
        How shared constraints get multipliers. ``"shared"`` (default)
        gives each constraint one multiplier used by every player in it,
        which yields a variational equilibrium. ``"per_player"`` gives
        each player its own multiplier for each constraint it is in, which
        is the KKT system of the full GNEP. The duals are ordered as in
        `multiplier_pairs`: one per constraint in ``"shared"`` mode, and one
        per ``(player, constraint)`` pair, player by player, in
        ``"per_player"`` mode. The initial point always holds one dual per
        constraint and is expanded by `expand_initial_point`.

    Notes
    -----
    All objective and constraint derivatives are JIT-compiled via
    ``jax.jit(jax.grad(...))`` at construction time to avoid recompilation
    during solving.
    """
    def __init__(self,
                 obj_funcs: List[ObjFunction],
                 constraints: List[ConsFunction],
                 player_list: List[Player],
                 validate = True,
                 *,
                 method: str = "SLSQP",
                 minimizer_options: dict = None,
                 niter: int = 1000,
                 stepsize: float = 0.01,
                 interval: int = 1,
                 niter_success: int = 100,
                 disp: bool = True,
                 multipliers: str = "shared",
                 jac: bool = True,
                 dual_bound: float = 100.0,
                 ):
        if multipliers not in ("shared", "per_player"):
            raise ValueError(
                f"multipliers must be 'shared' or 'per_player', got {multipliers!r}"
            )
        self.multipliers = multipliers
        self.dual_bound = dual_bound

        validate_player_list(player_list)
        player_info = players_to_lists(player_list)
        self.action_sizes = player_info["sizes"]
        self.bounds = player_info["bounds"]

        # Validate objective and constraint functions
        self.obj_functions = validate_obj_funcs(obj_funcs, self.action_sizes) if validate else obj_funcs
        self.const = validate_constraint_funcs(constraints) if validate else constraints

        # Player function indices validation
        validate_player_functions(player_list, self.obj_functions, self.const)

        # Player Objective and Constraint indices
        self.player_obj_idx = player_info["objectives"]
        self.player_const_idx = player_info["constraints"]
        self.players = player_list

        # Pre Compile Derivatives for speed
        self.obj_derivatives = [jax.jit(jax.grad(obj)) for obj in obj_funcs]
        self.const_derivatives = [jax.jit(jax.grad(const)) for const in constraints]

        self.player_const_idx_matrix = one_hot_encoding(
            self.player_const_idx,
            self.action_sizes,
            len(constraints)
        )

        # Pre calculations
        action_splits = jnp.cumsum(jnp.array([0] + self.action_sizes))
        self.action_splits = [int(x) for x in action_splits]
        self.total_actions = sum(self.action_sizes)

        self._build_multiplier_map()
        self.bounds_dual = [(0, self.dual_bound) for _ in range(self.num_constraint_duals)]
        self.bounds_all = self.bounds + self.bounds_dual

        # Bounds
        lb_list = [b[0] for b in self.bounds]
        ub_list = [b[1] for b in self.bounds]

        # Store these as JAX arrays for the energy_handler to use
        self.lb_vector = jnp.array(lb_list).reshape(-1, 1)
        self.ub_vector = jnp.array(ub_list).reshape(-1, 1)

        # Solver settings
        self.method = method
        self.minimizer_options = minimizer_options
        self.niter = niter
        self.stepsize = stepsize
        self.interval = interval
        self.niter_success = niter_success
        self.disp = disp
        # The exact gradient needs the lagrange_gradient/gradient_dual pair
        self.jac = jac and type(self).min_func is BaseAlgorithm.min_func

        self._grad_func_jit = jax.jit(jax.grad(self._jit_min_func))

    # ------------------------------------------------------------------
    # Multipliers
    # ------------------------------------------------------------------
    def _build_multiplier_map(self):
        """
        Map each constraint multiplier to the constraint and players it belongs to.

        Sets
        ----
        multiplier_pairs : list of tuple
            ``(player, constraint)`` for each multiplier. In ``"shared"``
            mode the player is ``None``.
        num_constraint_duals : int
            Number of constraint multipliers.
        dual_constraint_idx : jax.Array of int
            Constraint index of each multiplier.
        multiplier_matrix : jax.Array of shape (total_actions, num_constraint_duals)
            1 where a multiplier enters the stationarity of an action variable.
        """
        num_const = len(self.const)
        if self.multipliers == "shared":
            self.multiplier_pairs = [(None, j) for j in range(num_const)]
            self.multiplier_matrix = self.player_const_idx_matrix
            self._dual_lookup = {
                (i, j): j
                for i, player in enumerate(self.players)
                for j in player.constraints if j is not None
            }
        else:
            self.multiplier_pairs = [
                (i, j)
                for i, player in enumerate(self.players)
                for j in player.constraints if j is not None
            ]
            matrix = np.zeros((self.total_actions, len(self.multiplier_pairs)))
            for k, (i, _) in enumerate(self.multiplier_pairs):
                matrix[self.action_splits[i]:self.action_splits[i + 1], k] = 1.0
            self.multiplier_matrix = jnp.array(matrix)
            self._dual_lookup = {pair: k for k, pair in enumerate(self.multiplier_pairs)}

        self.num_constraint_duals = len(self.multiplier_pairs)
        self.dual_constraint_idx = jnp.array(
            [j for _, j in self.multiplier_pairs], dtype=int
        )

    def dual_index(self, player: int, constraint: int) -> int:
        """Position of the multiplier of ``constraint`` used by ``player``."""
        return self._dual_lookup[(player, constraint)]

    def constraint_values(self, vector_actions) -> jnp.ndarray:
        """
        Constraint value ``g_j(x)`` for each multiplier.

        Returns
        -------
        jax.Array of shape (num_constraint_duals,)
        """
        if not self.const:
            return jnp.zeros(0)
        g = jnp.array([c(vector_actions) for c in self.const])
        return g[self.dual_constraint_idx]

    # ------------------------------------------------------------------
    # Merit function
    # ------------------------------------------------------------------
    def lagrange_gradient(self, actions: jnp.ndarray, dual_actions: jnp.ndarray) -> jnp.ndarray:
        """
        Primal part of the merit function, one entry per action variable.

        Parameters
        ----------
        actions : vector of shape (sum(action_sizes), 1)
            Player action vectors stacked vertically.
        dual_actions : vector of shape (N_d, 1)
            Dual variables.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement lagrange_gradient and gradient_dual, "
            "or override min_func."
        )

    def gradient_dual(self, actions: jnp.ndarray, dual_actions: jnp.ndarray) -> jnp.ndarray:
        """
        Dual part of the merit function, one entry per dual variable.

        Parameters
        ----------
        actions : vector of shape (sum(action_sizes), 1)
            Player action vectors stacked vertically.
        dual_actions : vector of shape (N_d, 1)
            Dual variables.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement lagrange_gradient and gradient_dual, "
            "or override min_func."
        )

    @partial(jax.jit, static_argnums=(0,))
    def _jit_min_func(self, x):
        actions = x[:self.total_actions].reshape(-1, 1)
        dual_actions = x[self.total_actions:].reshape(-1, 1)

        primal = self.lagrange_gradient(actions, dual_actions)
        dual = self.gradient_dual(actions, dual_actions)
        return jnp.sum(primal) + jnp.sum(dual)

    def min_func(self, x: List[float]) -> float:
        """
        Merit function minimized by the solver.

        Parameters
        ----------
        x : list of float
            Flattened vector of primal and dual variables.

        Returns
        -------
        float
            Scalar value to minimize.
        """
        x_jax = jnp.asarray(x)
        return np.float64(self._jit_min_func(x_jax))

    def grad_min_func(self, x: jnp.ndarray):
        """Gradient of the merit function with respect to all primal and dual variables."""
        return np.array(self._grad_func_jit(x))

    def merit_jacobian(self, x) -> np.ndarray:
        """
        Gradient of the merit function passed to the local minimizer.

        Uses the exact JAX gradient, and finite differences for the entries
        where it is not finite.
        """
        x = np.asarray(x, dtype=float)
        grad = self.grad_min_func(jnp.asarray(x))
        bad = ~np.isfinite(grad)
        if bad.any():
            grad[bad] = approx_fprime(x, self.min_func)[bad]
        return grad

    # ------------------------------------------------------------------
    # Building blocks shared by merit functions
    # ------------------------------------------------------------------
    def objective_gradient(self, vector_actions) -> jnp.ndarray:
        """
        Stack each player's gradient of its own objective.

        Parameters
        ----------
        vector_actions : list of jax.Array
            Player action vectors, as returned by `construct_vectors`.

        Returns
        -------
        jax.Array of shape (sum(action_sizes), 1)
        """
        obj_grads_by_type = [der(vector_actions) for der in self.obj_derivatives]
        primal_grads = []
        for p_idx, obj_type_idx in enumerate(self.player_obj_idx):
            # Pull the specific gradient for player 'p_idx' from the correct objective logic
            g = obj_grads_by_type[obj_type_idx][p_idx]
            primal_grads.append(g.reshape(-1, 1))

        # Combine primal gradients into one vector
        return jnp.concatenate(primal_grads)

    def constraint_gradient(self, vector_actions, constraint_duals) -> jnp.ndarray:
        """
        Multiplier-weighted constraint gradients, restricted to the players in each constraint.

        Parameters
        ----------
        vector_actions : list of jax.Array
            Player action vectors, as returned by `construct_vectors`.
        constraint_duals : vector of shape (num_constraint_duals, 1)
            Constraint multipliers, ordered as in `multiplier_pairs`.

        Returns
        -------
        jax.Array of shape (sum(action_sizes), 1)
        """
        all_c_grads = []
        for const_der in self.const_derivatives:
            c_grad_list = const_der(vector_actions)
            # Flatten to a single column for this specific constraint
            all_c_grads.append(jnp.concatenate([g.ravel() for g in c_grad_list]))

        # This creates a (Total_Actions, Num_Constraints) matrix
        jacobian_matrix = jnp.stack(all_c_grads, axis=1)[:, self.dual_constraint_idx]  # (Total_actions, Num_duals)
        weighted_multipliers = self.multiplier_matrix * constraint_duals.ravel() # (Total_actions, Num_duals)
        constraint_contribution = jnp.sum(jacobian_matrix * weighted_multipliers, axis=1)
        return constraint_contribution.reshape(-1, 1)

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------
    def extend_initial_point(self, ip: jnp.ndarray) -> jnp.ndarray:
        """
        Add any extra variables the algorithm needs to the user's initial point.

        The default returns ``ip`` unchanged. Algorithms with additional
        variables, such as multipliers for the bounds, append them here.
        """
        return ip

    def expand_initial_point(self, ip: jnp.ndarray) -> jnp.ndarray:
        """
        Turn one dual per constraint into one dual per multiplier.

        In ``"per_player"`` mode each player's multiplier for constraint
        ``j`` starts at the value given for ``j``. In ``"shared"`` mode
        ``ip`` is returned unchanged.
        """
        if self.multipliers == "shared":
            return ip
        ip = jnp.asarray(ip)
        primal, duals = ip[:self.total_actions], ip[self.total_actions:]
        return jnp.concatenate([primal, duals[self.dual_constraint_idx]])

    def solve(self, ip: jnp.ndarray):
        """
        Solve the GNE problem from an initial point using basin-hopping.

        Wraps ``scipy.optimize.basinhopping`` with the local minimizer and
        settings given at construction, and the primal/dual bounds from the
        player definitions.

        Parameters
        ----------
        ip : jnp.ndarray
            Flat initial point containing stacked primal variables and one
            dual per constraint. Use `BaseProblem.set_initial_point` to
            construct this.

        Returns
        -------
        result : OptimizeResult
            The SciPy optimization result object.
        elapsed_time : float
            Wall-clock time in seconds.

        See Also
        --------
        [`BaseProblem.solve`](../problems/base_problem.md) : Higher-level entry point that calls this method.
        """
        ip = self.extend_initial_point(self.expand_initial_point(ip))
        minimizer_kwargs = dict(
            method=self.method,
            bounds=self.bounds_all
        )
        options = {**DEFAULT_MINIMIZER_OPTIONS.get(self.method, {}),
                   **(self.minimizer_options or {})}
        if options:
            minimizer_kwargs["options"] = options
        if self.jac:
            minimizer_kwargs["jac"] = self.merit_jacobian
        start = timeit.default_timer()
        result = basinhopping(
            self.min_func,
            ip,
            stepsize=self.stepsize,
            niter=self.niter,
            minimizer_kwargs=minimizer_kwargs,
            interval=self.interval,
            niter_success=self.niter_success,
            disp=self.disp,
        )
        stop = timeit.default_timer()
        elapsed_time = stop - start
        self.result_summary(result.x, elapsed_time)
        return result, elapsed_time

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def solution_blocks(self, x):
        """
        Named blocks of a solution vector, in print order.

        The default splits ``x`` into the primal actions and the constraint
        multipliers. Algorithms with additional variables extend this.
        """
        return [
            ("Primal Actions", x[:self.total_actions]),
            ("Dual Actions", x[self.total_actions:]),
        ]

    def split_multipliers(self, lambdas):
        """
        Split the dual part of a solution into constraint and bound multipliers.

        Returns
        -------
        tuple
            ``(constraint_duals, mu_lb, mu_ub)``. The default has no bound
            multipliers and returns ``(lambdas, None, None)``.
        """
        return lambdas, None, None

    def result_summary(self, x, time):
        """
        Print a solution summary and run KKT validation.

        Parameters
        ----------
        x : jnp.ndarray
            Flat solution vector of primal and dual variables.
        time : float
            Elapsed solve time in seconds.
        """
        print("\nRESULT SUMMARY")
        print("Elapsed time: ", time, " seconds")
        print("Final function value: ", self.min_func(x))
        for label, block in self.solution_blocks(x):
            print(f"{label}: ", block)
        self.check_kkt(jnp.array(x[:self.total_actions]), jnp.array(x[self.total_actions:]))

    def check_kkt(self, actions: jnp.ndarray, lambdas: jnp.ndarray, tol: float = 1e-6):
        """
        Compute KKT residuals for all players at a candidate solution.

        For each player, evaluates the four KKT conditions:

        - **Stationarity**
        - **Primal feasibility** — maximum violation of the constraints
          ``g_j <= 0`` and of the variable bounds.
        - **Dual feasibility** — maximum multiplier violation ``max(0, -lambda_j)``.
        - **Complementary slackness** — norm of ``lambda_j * g_j``.

        How bounds enter stationarity depends on the algorithm:

        - Without bound multipliers (the default, e.g. `EnergyMethod`), the
          bounds are handled by projection. Stationarity is the projected
          residual ``|| x_i - clip(x_i - grad L_i, lb_i, ub_i) ||``, which is
          zero exactly when ``grad L_i`` is zero at interior components,
          nonnegative at the lower bound, and nonpositive at the upper bound.
        - With bound multipliers (see `split_multipliers`, e.g.
          `FBEnergyMethod`), the bounds are constraints with their own
          multipliers and stationarity is ``|| grad L_i ||``.

        Parameters
        ----------
        actions : jnp.ndarray
            Flat array of primal variables for all players.
        lambdas : jnp.ndarray
            Flat array of dual variables: the constraint multipliers,
            followed by any bound multipliers of the algorithm.
        tol : float, optional
            Tolerance below which a residual is considered satisfied.
            Defaults to ``1e-6``.

        Returns
        -------
        dict
            A dictionary keyed by player name. Each value is a dict with:

            - ``"stationarity"`` — float.
            - ``"primal_feas"`` — float.
            - ``"dual_feas"`` — float.
            - ``"comp_slack"`` — float.
            - ``"is_kkt"`` — bool, ``True`` if all residuals are below ``tol``.
        """
        x_flat = jnp.array(actions)
        x_structured = construct_vectors(x_flat, self.action_sizes)
        shared_lambdas, mu_lb, mu_ub = self.split_multipliers(lambdas)

        kkt_report = {}

        for i, player in enumerate(self.players):
            # --- 1. Objective Gradient ---
            # Grad of f_i w.r.t player i's actions (x_i)
            grad_fi = self.obj_derivatives[player.f_index](x_structured)[i]

            # --- 2. Lagrangian Stationarity ---
            # We sum: grad_fi + sum(lambda_j * grad_gj) for all j in player.constraints
            l_grad_sum = jnp.zeros_like(grad_fi)
            g_vals = []
            p_lambdas = []

            for c_idx in player.constraints:
                if c_idx is not None:
                    # Get the multiplier player i uses for this constraint
                    l_val = shared_lambdas[self.dual_index(i, c_idx)]
                    p_lambdas.append(l_val)

                    # Get the constraint value
                    g_val = self.const[c_idx](x_structured)
                    g_vals.append(g_val)

                    # Get the gradient of constraint j w.r.t player i's actions
                    grad_gj = self.const_derivatives[c_idx](x_structured)[i]
                    l_grad_sum += l_val * grad_gj

            start, end = self.action_splits[i], self.action_splits[i + 1]
            p_actions = x_flat[start:end].reshape(-1, 1)
            p_lb = self.lb_vector[start:end]
            p_ub = self.ub_vector[start:end]

            if mu_lb is None:
                g_all = jnp.array(g_vals).flatten()
                l_all = jnp.array(p_lambdas).flatten()
                # Bounds handled by projection: natural residual of the box constrained problem
                grad_L = grad_fi + l_grad_sum
                stat_res = jnp.linalg.norm(p_actions - jnp.clip(p_actions - grad_L, p_lb, p_ub))
                bound_violation = jnp.concatenate([(p_lb - p_actions).flatten(), (p_actions - p_ub).flatten()])
            else:
                # Bounds as constraints lb - x <= 0 and x - ub <= 0 with their own multipliers
                p_mu_lb = mu_lb[start:end].reshape(-1, 1)
                p_mu_ub = mu_ub[start:end].reshape(-1, 1)

                # Add bound gradients: -1 for lower bound, +1 for upper bound
                l_grad_sum += (-1.0 * p_mu_lb) + (1.0 * p_mu_ub)

                g_lb_vals = p_lb - p_actions
                g_ub_vals = p_actions - p_ub

                g_all = jnp.concatenate([jnp.array(g_vals).flatten(), g_lb_vals.flatten(), g_ub_vals.flatten()])
                l_all = jnp.concatenate([jnp.array(p_lambdas).flatten(), p_mu_lb.flatten(), p_mu_ub.flatten()])
                # Stationarity: || grad_L_i ||, bound multipliers included
                stat_res = jnp.linalg.norm(grad_fi + l_grad_sum)
                bound_violation = jnp.zeros(0)

            # Primal Feasibility: max(0, g_j) over constraints and bounds
            # If 0, constraint is satisfied
            feas_all = jnp.concatenate([g_all, bound_violation])
            primal_res = jnp.max(jnp.maximum(0, feas_all)) if feas_all.size > 0 else 0.0

            # Dual Feasibility: max(0, -lambda_j)
            # If 0, constraint is active, lambda is compensating
            dual_res = jnp.max(jnp.maximum(0, -l_all)) if l_all.size > 0 else 0.0

            # Complementary Slackness: || lambda_j * g_j ||
            # Must be 0 for KKT to be satisfied
            slack_res = jnp.linalg.norm(l_all * g_all) if g_all.size > 0 else 0.0

            kkt_report[player.name] = {
                "stationarity": float(stat_res),
                "primal_feas": float(primal_res),
                "dual_feas": float(dual_res),
                "comp_slack": float(slack_res),
                "is_kkt": all(r < tol for r in [stat_res, primal_res, dual_res, slack_res])
            }

        self._print_kkt_report(kkt_report)
        return kkt_report

    @staticmethod
    def _print_kkt_report(report):
        """Print a formatted KKT residual table to stdout."""
        print(f"\n{'=' * 20} KKT VALIDATION {'=' * 20}")
        for name, metrics in report.items():
            print(f"PLAYER: {name}")
            print(f"  Stationarity: {metrics['stationarity']:.2e}")
            print(f"  Primal Feasibility:  {metrics['primal_feas']:.2e}")
            print(f"  Dual Feasibility:   {metrics['dual_feas']:.2e}")
            print(f"  Complementary Slackness:   {metrics['comp_slack']:.2e}")
        print("=" * 56)

    def summary(self):
        """Print a structured overview of the game and algorithm configuration."""
        print("=" * 60)
        print("GENERALIZED GAME SUMMARY")
        print("=" * 60)

        print(f"Number of objective functions: {len(self.obj_functions)}")
        print(f"Number of constraint functions: {len(self.const)}")

        print(f"\nNumber of players: {len(self.action_sizes)}")
        print(f"Total action dimension: {sum(self.action_sizes)}")

        print("\nPlayers:")
        print("-" * 60)
        for i, size in enumerate(self.action_sizes):
            print(f"Player {i}:")
            print(f"  Action size: {size}")
            print(f"  Objective index: {int(self.player_obj_idx[i])}")

            if hasattr(self, "bounds") and self.bounds is not None:
                print(f"  Bounds: {self.bounds[i]}")

            if hasattr(self, "player_const_idx_matrix"):
                active_constraints = self.player_const_idx_matrix[i]
                active_indices = [
                    idx for idx, val in enumerate(active_constraints)
                    if val == 1
                ]
                print(f"  Constraint indices: {active_indices}")

            print("-" * 60)

        print("Derivative compilation:")
        print(f"  Objective derivatives compiled: {len(self.obj_derivatives)}")
        print(f"  Constraint derivatives compiled: {len(self.const_derivatives)}")

        print("=" * 60)