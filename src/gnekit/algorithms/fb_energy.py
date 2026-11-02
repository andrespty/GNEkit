import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from gnekit.core.utils import construct_vectors
from gnekit.core.types import Vector
from gnekit.algorithms.base import BaseAlgorithm


class FBEnergyMethod(BaseAlgorithm):
    """
    Fischer-Burmeister variant of the energy method.

    Bounds on the actions are treated as constraints with their own
    multipliers, and complementarity is measured with the Fischer-Burmeister
    function. The dual variables are, in order: the constraint multipliers
    (see `BaseAlgorithm.multiplier_pairs`), one per lower bound, and one
    per upper bound.
    """
    def __init__(self, obj_funcs, constraints, player_list, *args, **kwargs):
        super().__init__(obj_funcs, constraints, player_list, *args, **kwargs)

        # Dual actions for the bounds
        self.total_dual_size = self.num_constraint_duals + 2 * self.total_actions
        self.bounds_dual = [(0, 100) for _ in range(self.total_dual_size)]
        self.bounds_all = self.bounds + self.bounds_dual

    @staticmethod
    def fb_function(a, b):
        """Standard Fischer-Burmeister function."""
        return jnp.sqrt(a ** 2 + b ** 2 + 1e-8) - (a + b)

    @staticmethod
    def energy_handler(gradient: jnp.ndarray) -> jnp.ndarray:
        """
        Squared residual, applied elementwise.
        """
        return gradient ** 2

    def lagrange_gradient(self, actions: Vector, dual_actions: Vector) -> Vector:
        """
        Compute the primal energy from the gradient of the Lagrangian.

        Parameters
        ----------
        actions : vector of shape (sum(action_sizes), 1)
            Player action vectors.
        dual_actions : vector of shape (N_d, 1)
            Dual variables.

        Returns
        -------
        vector of shape (sum(action_sizes), 1)
            Primal energy, one entry per action variable.
        """
        vector_actions = construct_vectors(actions, self.action_sizes) # List of jnp.array shape (action_size[i], 1)
        result = self.objective_gradient(vector_actions)

        # Slice Dual Actions
        # [0 : N_shared] -> Shared Constraints
        # [N_shared : N_shared + Total_Actions] -> Lower Bounds
        # [N_shared + Total_Actions : ] -> Upper Bounds
        num_const = self.num_constraint_duals
        const_duals = dual_actions[:num_const].reshape(-1, 1)
        mu_lb = dual_actions[num_const: num_const + self.total_actions].reshape(-1, 1)
        mu_ub = dual_actions[num_const + self.total_actions:].reshape(-1, 1)

        if self.const_derivatives:
            result += self.constraint_gradient(vector_actions, const_duals)

        result += (-1.0 * mu_lb) + (1.0 * mu_ub)
        return self.energy_handler(result)

    def gradient_dual(self, actions: Vector, dual_actions: Vector) -> Vector:
        """
        Compute the dual energy.

        Parameters
        ----------
        actions : vector of shape (sum(action_sizes), 1)
            Player action vectors.
        dual_actions : vector of shape (N_d, 1)
            Dual variables.

        Returns
        -------
        vector of shape (N_d, 1)
            Dual energy, one entry per constraint and per bound.
        """
        if not self.const:
            return jnp.zeros((1, 1))

        actions_vectors = construct_vectors(actions, self.action_sizes)
        # All entries in the form g <= 0: shared constraints, lb - x, x - ub
        g_values = self.constraint_values(actions_vectors).reshape(-1, 1)

        g_lb = self.lb_vector - actions
        g_ub = actions - self.ub_vector

        all_g = jnp.concatenate([g_values, g_lb, g_ub]).reshape(-1, 1)

        fb = self.fb_function(dual_actions, -all_g)
        grad_dual_vec = self.energy_handler(fb)
        return grad_dual_vec.reshape(-1, 1)

    def extend_initial_point(self, ip: jnp.ndarray) -> jnp.ndarray:
        """Append the bound multipliers, all set to 0.1, to the initial point."""
        print('FB Energy Method being used')
        num_bound_duals = 2 * self.total_actions
        bound_duals_ip = jnp.full((num_bound_duals,), 0.1)
        return jnp.concatenate([ip, bound_duals_ip])

    def split_multipliers(self, lambdas):
        num_shared = self.num_constraint_duals
        shared_lambdas = lambdas[:num_shared]
        mu_lb = lambdas[num_shared: num_shared + self.total_actions]
        mu_ub = lambdas[num_shared + self.total_actions:]
        return shared_lambdas, mu_lb, mu_ub

    def solution_blocks(self, x):
        dual = x[self.total_actions:self.total_actions + self.num_constraint_duals]
        dual_bounds = x[self.total_actions + self.num_constraint_duals:]
        return [
            ("Primal Actions", x[:self.total_actions]),
            ("Dual Actions", dual),
            ("Lower Bound Actions", dual_bounds[:self.total_actions]),
            ("Upper Bound Actions", dual_bounds[self.total_actions:]),
        ]
