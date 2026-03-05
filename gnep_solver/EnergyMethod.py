import jax
import numpy as np
from typing import List, Optional
import jax.numpy as jnp
from jax import Array
jax.config.update("jax_enable_x64", True)
from .utils import construct_vectors, one_hot_encoding
from .schema import Vector
from functools import partial

class EnergyMethod:
    def __init__(self,
                 action_sizes: List[int],
                 obj_derivatives,
                 constraints,
                 constraint_derivatives,
                 player_obj_idx: List[int],
                 player_const_idx: List[List[int]],
                 bounds: List[tuple[float, float]]
                 ):
        self.action_sizes = action_sizes
        self.obj_derivatives = obj_derivatives
        self.constraints = constraints
        self.constraint_derivatives = constraint_derivatives

        self.player_obj_idx = player_obj_idx
        self.player_const_idx_matrix = one_hot_encoding(
            player_const_idx,
            self.action_sizes,
            len(constraints)
        )

        # Pre calculations
        action_splits = jnp.cumsum(jnp.array([0] + action_sizes))
        self.action_splits = [int(x) for x in action_splits]
        self.total_actions = sum(action_sizes)

        # Bounds
        lb_list = []
        ub_list = []
        for i, (lb, ub) in enumerate(bounds):
            # Repeat the bounds for each decision variable the player owns
            lb_list.append(jnp.full((action_sizes[i],), lb))
            ub_list.append(jnp.full((action_sizes[i],), ub))

        # Store these as JAX arrays for the energy_handler to use
        self.lb_vector = jnp.concatenate(lb_list).reshape(-1, 1)
        self.ub_vector = jnp.concatenate(ub_list).reshape(-1, 1)

    @partial(jax.jit, static_argnums=(0,))
    def _jit_min_func(self, x):
        # Move ALL the math from your current min_func into here
        # No np.array, no list conversions, no print statements
        actions = x[:self.total_actions].reshape(-1, 1)
        dual_actions = x[self.total_actions:].reshape(-1, 1)

        primal = self.primal_energy_function(actions, dual_actions)
        dual = self.dual_energy_function(actions, dual_actions)
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
        return np.float32(self._jit_min_func(x_jax))

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


    def primal_energy_function(self, actions: Vector, dual_actions: Vector) -> tuple[Array, ...]:
        """
        Compute the energy contribution of primal players.

        Parameters
        ----------
        actions : vector of shape (sum(action_sizes), 1)
            Player action vectors.
        dual_actions : vector of shape (N_d, 1)
            Dual variables.

        Returns
        -------
        vector of shape (sum(action_sizes), 1)
            Energy contribution per primal player action.
        """
        gradient = self.lagrange_gradient(actions, dual_actions)
        return self.energy_handler(gradient, actions, self.lb_vector, self.ub_vector)

    def dual_energy_function(self, actions: Vector, dual_actions: Vector) -> Vector:
        """
        Compute the energy contribution of dual players.

        Parameters
        ----------
        actions : vector of shape (sum(action_sizes), 1)
            Player action vectors.
        dual_actions : vector of shape (N_d, 1)
            Dual variables.

        Returns
        -------
        vector of shape (N_d, 1)
            Energy contribution per dual variable.
        """
        eng_dual = self.gradient_dual(actions, dual_actions)
        return eng_dual

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
        vector_actions = construct_vectors(actions, self.action_sizes)
        result = jnp.zeros_like(actions) # shape (-1,1)

        # Primal Gradients from objectives
        for obj_idx, obj_der in enumerate(self.obj_derivatives):
            grads = obj_der(vector_actions)

            for p_idx, assigned_obj_idx in enumerate(self.player_obj_idx):
                if assigned_obj_idx == obj_idx:
                    start = self.action_splits[p_idx]
                    end = self.action_splits[p_idx + 1]
                    result = result.at[start:end].set(grads[p_idx].reshape(-1, 1))

        # Adding constraints, should be vectors with same size of result
        for c_idx, const_der in enumerate(self.constraint_derivatives):
            p_vector = self.player_const_idx_matrix[:, c_idx].reshape(-1,1) # one hot encoding
            const_grad_list = const_der(vector_actions) # shape ([size of action of player 1],[size of action of player 2],[])
            full_const_grad = jnp.concatenate([g.ravel() for g in const_grad_list]).reshape(-1, 1)

            dual_encoding = p_vector * dual_actions[c_idx]
            result += full_const_grad * dual_encoding
        return result

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
        grad_dual = []
        actions_vectors = construct_vectors(actions, self.action_sizes)
        for jdx, constraint in enumerate(self.constraints):
            g = -constraint(actions_vectors)
            dual_lb = jnp.zeros_like(dual_actions[jdx])
            dual_ub = jnp.full_like(dual_actions[jdx], 1e6)
            g = self.energy_handler(g, dual_actions[jdx], dual_lb, dual_ub)
            grad_dual.append(g.flatten())
        g_dual = jnp.concatenate(grad_dual).reshape(-1, 1)
        return g_dual