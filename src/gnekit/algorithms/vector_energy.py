import numpy as np

from gnekit.core.utils import construct_vectors, one_hot_encoding
from gnekit.core.types import Vector
from gnekit.algorithms.energy import EnergyMethod


class VectorEnergyMethod(EnergyMethod):
    """
    Energy method that accepts vector gradients of multiple players together.

    Same energy as `EnergyMethod`. The only difference is how the primal
    gradient is assembled: an objective may return the gradient blocks of
    several players at once.
    """
    def __init__(self, obj, const, players, *args, **kwargs):
        super().__init__(obj, const, players, *args, **kwargs)

        self.player_obj_idx_matrix = one_hot_encoding(self.player_obj_idx, self.action_sizes, len(self.obj_derivatives))

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

        result = np.zeros_like(actions)
        # Track action indices per player
        action_splits = np.cumsum(np.insert(self.action_sizes, 0, 0))  # Start and end indices for each player
        for obj_idx, mask in enumerate(self.player_obj_idx_matrix.T):
            player_indices = np.where(mask)[0]
            o = self.obj_derivatives[obj_idx](vector_actions)
            if o.shape == result.shape:
                # Case 1: objective returns full (a,1) vector
                result += mask.reshape(-1, 1) * o
            else:
                offset = 0
                mask = (self.player_obj_idx == obj_idx)
                player_indices = np.where(mask)[0]
                for player in player_indices:
                    start_idx, end_idx = action_splits[player], action_splits[player + 1]
                    size = end_idx - start_idx
                    result[start_idx:end_idx] = o  # [offset:offset + size]
                    offset += size

        if self.const_derivatives:
            result += self.constraint_gradient(vector_actions, dual_actions)
        return self.energy_handler(result, actions, self.lb_vector, self.ub_vector)
