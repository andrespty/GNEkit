from abc import abstractmethod, ABC
from scipy.optimize import basinhopping
import timeit
import jax
from solvers.validation import *
from solvers.schema import *
from solvers.gnep_solver import *
from solvers.gnep_solver.BasePlayer import players_to_lists, Player

class BaseAlgorithm(ABC):
    """
    Abstract base class for GNE-seeking algorithms.

    Handles all shared setup: derivative compilation, bound extraction,
    constraint encoding, and KKT validation. Subclasses only need to
    implement `min_func`.

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
                 validate = True
                 ):
        validate_player_list(player_list)
        player_info = players_to_lists(player_list)
        self.action_sizes = player_info["sizes"]
        self.bounds = player_info["bounds"]
        self.bounds_dual = [(0, 100) for _ in constraints]
        self.bounds_all = self.bounds + self.bounds_dual

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

        # Bounds
        lb_list = [b[0] for b in self.bounds]
        ub_list = [b[1] for b in self.bounds]

        # Store these as JAX arrays for the energy_handler to use
        self.lb_vector = jnp.array(lb_list).reshape(-1, 1)
        self.ub_vector = jnp.array(ub_list).reshape(-1, 1)

    @abstractmethod
    def min_func(self, x: List[float]) -> float:
        """
        Objective function minimized by the solver.

        Parameters
        ----------
        x : list of float
            Flattened vector of primal and dual variables.

        Returns
        -------
        float
            Scalar value to minimize.
        """
        pass

    def check_kkt(self, actions: jnp.ndarray, lambdas: jnp.ndarray, tol: float = 1e-6):
        """
        Compute KKT residuals for all players at a candidate solution.

        For each player, evaluates the four KKT conditions:

        - **Stationarity** — norm of the Lagrangian gradient.
        - **Primal feasibility** — maximum constraint violation ``max(0, g_j)``.
        - **Dual feasibility** — maximum multiplier violation ``max(0, -lambda_j)``.
        - **Complementary slackness** — norm of ``lambda_j * g_j``.

        Parameters
        ----------
        actions : jnp.ndarray
            Flat array of primal variables for all players.
        lambdas : jnp.ndarray
            Flat array of dual multipliers, indexed by constraint.
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
                    # Get the specific multiplier for this constraint
                    l_val = lambdas[c_idx]
                    p_lambdas.append(l_val)

                    # Get the constraint value
                    g_val = self.const[c_idx](x_structured)
                    g_vals.append(g_val)

                    # Get the gradient of constraint j w.r.t player i's actions
                    grad_gj = self.const_derivatives[c_idx](x_structured)[i]
                    l_grad_sum += l_val * grad_gj

            # Convert to arrays for vectorized residual math
            g_vals = jnp.array(g_vals)
            p_lambdas = jnp.array(p_lambdas)

            # Residuals
            # Stationarity: || grad_L_i ||
            stat_res = jnp.linalg.norm(grad_fi + l_grad_sum)

            # Primal Feasibility: max(0, g_j)
            # If 0, constraint is satisfied
            primal_res = jnp.max(jnp.maximum(0, g_vals)) if g_vals.size > 0 else 0.0

            # Dual Feasibility: max(0, -lambda_j)
            # If 0, constraint is active, lambda is compensating
            dual_res = jnp.max(jnp.maximum(0, -p_lambdas)) if p_lambdas.size > 0 else 0.0

            # Complementary Slackness: || lambda_j * g_j ||
            # Must be 0 for KKT to be satisfied
            slack_res = jnp.linalg.norm(p_lambdas * g_vals) if g_vals.size > 0 else 0.0

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

    def solve(self, ip: jnp.ndarray):
        """
        Solve the GNE problem from an initial point using basin-hopping.

        Wraps ``scipy.optimize.basinhopping`` with SLSQP as the local
        minimizer and the primal/dual bounds from the player definitions.

        Parameters
        ----------
        ip : jnp.ndarray
            Flat initial point containing stacked primal and dual variables.
            Use `BaseProblem.set_initial_point` to construct this.

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
        minimizer_kwargs = dict(
            method="SLSQP",
            # options={"eps": 1e-6}
            # jac=self.solver.grad_min_func,
            bounds=self.bounds_all
        )
        start = timeit.default_timer()
        result = basinhopping(
            self.min_func,
            ip,
            stepsize=0.01,
            niter=1000,
            minimizer_kwargs=minimizer_kwargs,
            interval=1,
            niter_success=100,
            disp=True,
            # callback=stopping_criterion
        )
        stop = timeit.default_timer()
        elapsed_time = stop - start
        self.result_summary(result.x, elapsed_time)
        return result, elapsed_time

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
        primal = x[:self.total_actions]
        dual = x[self.total_actions:]
        print("Primal Actions: ", primal)
        print("Dual Actions: ", dual)
        self.check_kkt(jnp.array(primal), jnp.array(dual))

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