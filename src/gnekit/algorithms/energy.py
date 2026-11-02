import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from gnekit.core.utils import construct_vectors
from gnekit.core.types import Vector
from gnekit.algorithms.base import BaseAlgorithm


class EnergyMethod(BaseAlgorithm):
    """
    Energy method for generalized Nash equilibrium problems.

    The key idea is to define a single scalar "energy" function that combines
    the primal and dual gradients, and then minimize this energy with respect to
    both the actions and the dual variables. The energy is zero exactly at a KKT
    point. Bounds on the actions are handled inside the energy, so no
    multipliers are introduced for them.
    """

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
        if self.const_derivatives:
            result += self.constraint_gradient(vector_actions, dual_actions)
        return self.energy_handler(result, actions, self.lb_vector, self.ub_vector)

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
            Dual energy, one entry per constraint.
        """
        if not self.const:
            return jnp.zeros((1, 1))

        actions_vectors = construct_vectors(actions, self.action_sizes)
        g_values = jnp.array([-c(actions_vectors) for c in self.const])

        # Ensure dual_actions is flat for the handler
        lambdas = dual_actions.ravel()

        # Bounds of the dual variables, matching the size of the lambdas
        dual_lb = jnp.zeros_like(lambdas)
        dual_ub = jnp.full_like(lambdas, 100)

        # Run the energy_handler once on the entire vector
        grad_dual_vec = self.energy_handler(g_values, lambdas, dual_lb, dual_ub)

        return grad_dual_vec.reshape(-1, 1)
