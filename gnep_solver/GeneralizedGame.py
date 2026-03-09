from gnep_solver import *
from gnep_solver.utils import one_hot_encoding
from typing import List
from scipy.optimize import basinhopping
import timeit
from gnep_solver.Player import Player, players_to_lists
import jax
import jax.numpy as jnp
from jax import eval_shape
from .validation import *
from gnep_solver.EnergyMethod import EnergyMethod
from .utils import *
jax.config.update("jax_enable_x64", True)

class GeneralizedGame:
    def __init__(
            self,
            obj_funcs: List[ObjFunction],
            constraints: List[ConsFunction],
            player_list: List[Player]
        ):
        # Extract structured info
        validate_player_list(player_list)
        player_info = players_to_lists(player_list)
        self.action_sizes = player_info["sizes"]
        self.bounds = player_info["bounds"]

        # Validate objective and constraint functions
        self.obj_functions = validate_obj_funcs(obj_funcs, self.action_sizes)
        self.const = validate_constraint_funcs(constraints)

        # Player function indices validation
        validate_player_functions(player_list, self.obj_functions, self.const)

        # Player Objective and Constraint indices
        self.player_obj_idx = player_info["objectives"]
        self.player_const_idx = player_info["constraints"]
        self.players = player_list

        # Pre Compile Derivatives for speed
        self.obj_derivatives = [jax.jit(jax.grad(obj)) for obj in obj_funcs]
        self.const_derivatives = [jax.jit(jax.grad(const)) for const in constraints]

        # Pre-instantiate the solver once to avoid overhead in loops
        self.solver = EnergyMethod(
            self.action_sizes,
            self.obj_derivatives,
            self.const,
            self.const_derivatives,
            self.player_obj_idx,
            self.player_const_idx,
            self.bounds
        )

    def check_kkt(self, actions: jnp.ndarray, lambdas: jnp.ndarray, tol: float = 1e-6):
        """
        Computes KKT residuals where 'lambdas' is a global array of multipliers.
        Each player.constraints contains the indices mapping into 'lambdas'.
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
        print(f"\n{'=' * 20} KKT VALIDATION {'=' * 20}")
        for name, metrics in report.items():
            print(f"PLAYER: {name}")
            print(f"  Stationarity: {metrics['stationarity']:.2e}")
            print(f"  Primal Feasibility:  {metrics['primal_feas']:.2e}")
            print(f"  Dual Feasibility:   {metrics['dual_feas']:.2e}")
            print(f"  Complementary Slackness:   {metrics['comp_slack']:.2e}")
        print("=" * 56)

    def summary(self):
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

    def grad_val(self, actions: jnp.ndarray) -> List[jnp.ndarray]:
        x = construct_vectors(jnp.array(actions), self.action_sizes)
        return [df(x) for df in self.obj_derivatives]

    def energy_val(self, actions: jnp.ndarray) -> float:
        return self.solver.min_func(actions)

    def solve_game(self, ip: jnp.ndarray):
        minimizer_kwargs = dict(
            method="SLSQP",
            tol=1e-9,
            # options={"eps": 1e-6}
            jac=self.solver.grad_min_func,
        )

        start = timeit.default_timer()
        result = basinhopping(
            self.solver.min_func,
            ip,
            stepsize=0.01,
            niter=1000,
            minimizer_kwargs=minimizer_kwargs,
            interval=1,
            niter_success=100,
            disp=True,
            # callback=stopping_criterion
        )
        print(self.energy_val(jnp.array(result.x)))
        stop = timeit.default_timer()
        elapsed_time = stop - start
        return result, elapsed_time