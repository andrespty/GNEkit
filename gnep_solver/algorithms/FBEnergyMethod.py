import jax
import numpy as np
from typing import List, Optional, Callable
import jax.numpy as jnp
from jax import Array
jax.config.update("jax_enable_x64", True)
from gnep_solver.utils import construct_vectors, one_hot_encoding
from gnep_solver.schema import Vector
from functools import partial
from .BaseAlgorithm import BaseAlgorithm
from scipy.optimize import basinhopping
import timeit

class FBEnergyMethod(BaseAlgorithm):
    def __init__(self, obj_funcs, constraints, player_list):
        super().__init__(obj_funcs, constraints, player_list)

        # Dual actions for the bounds
        self.total_dual_size = len(self.const) + 2 * self.total_actions
        self.bounds_dual = [(0, 100) for _ in range(self.total_dual_size)]
        self.bounds_all = self.bounds + self.bounds_dual

    @staticmethod
    def fb_function(a, b):
        """Standard Fischer-Burmeister function."""
        return jnp.sqrt(a ** 2 + b ** 2 + 1e-8) - (a + b)

    @partial(jax.jit, static_argnums=(0,))
    def _jit_min_func(self, x):
        actions = x[:self.total_actions].reshape(-1, 1)
        dual_actions = x[self.total_actions:].reshape(-1, 1)

        primal = self.lagrange_gradient(actions, dual_actions)
        dual = self.gradient_dual(actions, dual_actions)
        return jnp.sum(primal) + jnp.sum(dual)

    def min_func(self, x: List[float]) -> float:
        """
        Compute the total energy of the system.

        Parameters
        ----------
        x : Vector
            Player action vectors stacked vertically.

        Returns
        -------
        float
            Total energy of the system.
        """
        x_jax = jnp.asarray(x)
        return np.float64(self._jit_min_func(x_jax))

    def grad_min_func(self, x: jnp.ndarray):
        return np.array(self._grad_func_jit(x))

    @staticmethod
    def energy_handler(gradient: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized energy calculation using per-element lower and upper bounds.
        """
        return gradient ** 2

    @staticmethod
    def energy_handler2(gradient: jnp.ndarray,
                       actions: jnp.ndarray,
                       lb: jnp.ndarray,
                       ub: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized energy calculation using per-element lower and upper bounds.
        """
        denom_neg = 1.0 - gradient
        denom_pos = 1.0 + gradient

        return jnp.where(
            gradient <= 0,
            jnp.abs(ub - actions) * (gradient ** 2) / denom_neg,
            jnp.abs(actions - lb) * (gradient ** 2) / denom_pos
        )

    def lagrange_gradient(self, actions: Vector, dual_actions: Vector) -> Vector:
        """
        Compute the gradient of the Lagrangian Function.

        Parameters
        ----------
        actions : vector of shape (sum(action_sizes), 1)
            Player action vectors.
        dual_actions : vector of shape (N_d, 1)
            Dual variables.

        Returns
        -------
        vector of shape (sum(action_sizes), 1)
            Gradient of primal energy with respect to actions.
        """
        vector_actions = construct_vectors(actions, self.action_sizes) # List of jnp.array shape (action_size[i], 1)

        obj_grads_by_type = [der(vector_actions) for der in self.obj_derivatives]
        primal_grads = []
        for p_idx, obj_type_idx in enumerate(self.player_obj_idx):
            # Pull the specific gradient for player 'p_idx' from the correct objective logic
            g = obj_grads_by_type[obj_type_idx][p_idx]
            primal_grads.append(g.reshape(-1, 1))

        # Combine primal gradients into one vector
        result = jnp.concatenate(primal_grads)

        # 2. Slice Dual Actions
        # [0 : N_shared] -> Shared Constraints
        # [N_shared : N_shared + Total_Actions] -> Lower Bounds
        # [N_shared + Total_Actions : ] -> Upper Bounds
        num_const = len(self.const)
        const_duals = dual_actions[:num_const].reshape(-1, 1)
        mu_lb = dual_actions[num_const: num_const + self.total_actions].reshape(-1, 1)
        mu_ub = dual_actions[num_const + self.total_actions:].reshape(-1, 1)

        if self.const_derivatives:
            all_c_grads = []
            for const_der in self.const_derivatives:
                c_grad_list = const_der(vector_actions)
                # Flatten to a single column for this specific constraint
                all_c_grads.append(jnp.concatenate([g.ravel() for g in c_grad_list]))

            # This creates a (Total_Actions, Num_Constraints) matrix
            jacobian_matrix = jnp.stack(all_c_grads, axis=1)
            weighted_multipliers = self.player_const_idx_matrix * const_duals.ravel() # (Total_actions, Num_constraints)
            constraint_contribution = jnp.sum(jacobian_matrix * weighted_multipliers, axis=1)
            result += constraint_contribution.reshape(-1, 1)

        result += (-1.0 * mu_lb) + (1.0 * mu_ub)
        return self.energy_handler(result)

    def gradient_dual(self, actions: Vector, dual_actions: Vector) -> Vector:
        """
        Compute the gradient of the dual energy.

        Parameters
        ----------
        actions : vector of shape (sum(action_sizes), 1)
            Player action vectors.
        dual_actions : vector of shape (N_d, 1)
            Dual variables.

        Returns
        -------
        vector of shape (N_d, 1)
            Gradient of dual energy with respect to constraints.
        """
        if not self.const:
            return jnp.zeros((1, 1))

        actions_vectors = construct_vectors(actions, self.action_sizes)
        g_values = jnp.array([-c(actions_vectors) for c in self.const]).reshape(-1, 1)

        g_lb = self.lb_vector - actions
        g_ub = actions - self.ub_vector

        all_g = jnp.concatenate([g_values, g_lb, g_ub]).reshape(-1, 1)

        fb = self.fb_function(dual_actions, -all_g)
        grad_dual_vec = self.energy_handler(fb)
        return grad_dual_vec.reshape(-1, 1)

    def solve(self, ip: jnp.ndarray):
        print('FB Energy Method being used')
        num_bound_duals = 2 * self.total_actions
        bound_duals_ip = jnp.full((num_bound_duals,), 0.1)
        full_ip = jnp.concatenate([ip, bound_duals_ip])
        # print("Primal Actions: ",full_ip[:self.total_actions])
        # print("Dual Actions: ",full_ip[self.total_actions:self.total_actions + len(self.const)])
        # print("Bound Actions: ",full_ip[self.total_actions + len(self.const):])
        # print("Bounds All: ", self.bounds_all)
        minimizer_kwargs = dict(
            method="SLSQP",
            # options={"eps": 1e-6}
            # jac=self.solver.grad_min_func,
            bounds=self.bounds_all
        )
        start = timeit.default_timer()
        result = basinhopping(
            self.min_func,
            full_ip,
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
        print("\nRESULT SUMMARY")
        print("Elapsed time: ", time, " seconds")
        print("Final function value: ", self.min_func(x))
        primal = x[:self.total_actions]
        dual = x[self.total_actions:self.total_actions + len(self.const)]
        dual_bounds = x[self.total_actions + len(self.const):]
        dual_lb = dual_bounds[:self.total_actions]
        dual_ub = dual_bounds[self.total_actions:]
        print("Primal Actions: ", primal)
        print("Dual Actions: ", dual)
        print("Lower Bound Actions: ", dual_lb)
        print("Upper Bound Actions: ", dual_ub)
        self.check_kkt(jnp.array(primal), jnp.array(x[self.total_actions:]))

    def check_kkt(self, actions: jnp.ndarray, lambdas: jnp.ndarray, tol: float = 1e-6):
        x_flat = jnp.array(actions)
        x_structured = construct_vectors(x_flat, self.action_sizes)

        # 1. Slice lambdas to get the boundary duals
        num_shared = len(self.const)
        shared_lambdas = lambdas[:num_shared]
        mu_lb = lambdas[num_shared: num_shared + self.total_actions]
        mu_ub = lambdas[num_shared + self.total_actions:]

        kkt_report = {}

        for i, player in enumerate(self.players):
            # --- Objective Gradient ---
            grad_fi = self.obj_derivatives[player.f_index](x_structured)[i]

            # --- Shared Lagrangian Stationarity ---
            l_grad_sum = jnp.zeros_like(grad_fi)
            g_vals_list = []
            p_lambdas_list = []

            for c_idx in player.constraints:
                if c_idx is not None:
                    l_val = shared_lambdas[c_idx]
                    p_lambdas_list.append(l_val)

                    g_val = self.const[c_idx](x_structured)
                    g_vals_list.append(g_val)

                    grad_gj = self.const_derivatives[c_idx](x_structured)[i]
                    l_grad_sum += l_val * grad_gj

            # --- Bound Stationarity Contribution ---
            # Get this player's specific bound duals using the action_splits
            start, end = self.action_splits[i], self.action_splits[i + 1]
            p_mu_lb = mu_lb[start:end].reshape(-1, 1)
            p_mu_ub = mu_ub[start:end].reshape(-1, 1)

            # Add bound gradients: -1 for lower bound, +1 for upper bound
            l_grad_sum += (-1.0 * p_mu_lb) + (1.0 * p_mu_ub)

            # --- Feasibility for Bounds ---
            p_actions = x_flat[start:end].reshape(-1, 1)
            p_lb = self.lb_vector[start:end]
            p_ub = self.ub_vector[start:end]

            g_lb_vals = p_lb - p_actions
            g_ub_vals = p_actions - p_ub

            # Combine all constraint values and multipliers for this player
            g_all = jnp.concatenate([jnp.array(g_vals_list).flatten(), g_lb_vals.flatten(), g_ub_vals.flatten()])
            l_all = jnp.concatenate([jnp.array(p_lambdas_list).flatten(), p_mu_lb.flatten(), p_mu_ub.flatten()])

            # --- Residuals ---
            stat_res = jnp.linalg.norm(grad_fi + l_grad_sum)
            primal_res = jnp.max(jnp.maximum(0, g_all)) if g_all.size > 0 else 0.0
            dual_res = jnp.max(jnp.maximum(0, -l_all)) if l_all.size > 0 else 0.0
            slack_res = jnp.linalg.norm(l_all * g_all) if g_all.size > 0 else 0.0

            kkt_report[player.name] = {
                "stationarity": float(stat_res),
                "primal_feas": float(primal_res),
                "dual_feas": float(dual_res),
                "comp_slack": float(slack_res),
                "is_kkt": all(r < tol for r in [stat_res, primal_res, dual_res, slack_res])
            }

        self._print_kkt_report(kkt_report)