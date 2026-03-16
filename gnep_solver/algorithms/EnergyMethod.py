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

class EnergyMethod(BaseAlgorithm):
    def __init__(self, obj, const, players):
        super().__init__(obj, const, players)
        self._grad_func_jit = jax.jit(jax.grad(self._jit_min_func))

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

    @partial(jax.jit, static_argnums=(0,))
    def _jit_min_func(self, x):
        actions = x[:self.total_actions].reshape(-1, 1)
        dual_actions = x[self.total_actions:].reshape(-1, 1)

        primal = self.lagrange_gradient(actions, dual_actions)
        dual = self.gradient_dual(actions, dual_actions)
        return jnp.sum(primal) + jnp.sum(dual)

    def grad_min_func(self, x: jnp.ndarray):
        return np.array(self._grad_func_jit(x))

    @staticmethod
    def energy_handler(gradient: jnp.ndarray,
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

        if self.const_derivatives:
            all_c_grads = []
            for const_der in self.const_derivatives:
                c_grad_list = const_der(vector_actions)
                # Flatten to a single column for this specific constraint
                all_c_grads.append(jnp.concatenate([g.ravel() for g in c_grad_list]))

            # This creates a (Total_Actions, Num_Constraints) matrix
            jacobian_matrix = jnp.stack(all_c_grads, axis=1)
            weighted_multipliers = self.player_const_idx_matrix * dual_actions.ravel() # (Total_actions, Num_constraints)
            constraint_contribution = jnp.sum(jacobian_matrix * weighted_multipliers, axis=1)
            result += constraint_contribution.reshape(-1, 1)
        return self.energy_handler(result, actions, self.lb_vector, self.ub_vector)

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
        g_values = jnp.array([-c(actions_vectors) for c in self.const])

        # Ensure dual_actions is flat for the handler
        lambdas = dual_actions.ravel()

        # 3. Create vectorized bounds for the dual variables
        # These match the size of the lambdas
        dual_lb = jnp.zeros_like(lambdas)
        dual_ub = jnp.full_like(lambdas, 100)

        # 4. Run the energy_handler once on the entire vector
        # JAX will parallelize the 'where' logic across all constraints
        grad_dual_vec = self.energy_handler(g_values, lambdas, dual_lb, dual_ub)

        # 5. Return as the expected column vector
        return grad_dual_vec.reshape(-1, 1)

