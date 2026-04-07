from abc import abstractmethod
from typing import List, Type, Sequence
from solvers.algorithms.BaseAlgorithm import BaseAlgorithm
from solvers.dgbne_solver.BayesianPlayer import BayesianPlayer
from solvers.validation import validate_problem_functions
from solvers.gnep_solver import *
from solvers.gnep_solver.BaseProblem import BaseProblem
import jax.numpy as jnp

class BayesianProblem(BaseProblem):
    @property
    def players(self) -> List[BayesianPlayer]:
        return self._players

    @players.setter
    def players(self, value):
        if value is None:
            self._players = None
            return

        if not isinstance(value, list):
            raise TypeError("Players must be provided as a list.")

        self._validate_player_type(value)
        self._players = value

    def _validate_player_type(self, player_list):
        for i, p in enumerate(player_list):
            if not isinstance(p, BayesianPlayer):
                raise TypeError(f"Item at index {i} must be a BayesianPlayer.")
            
    
    def reshape_player_actions(self, x_i, player_idx: int) -> jnp.ndarray:
        """
        Reshape a flat player vector into:
        (n_types, action_size_per_type)
        """
        player = self.players[player_idx]
        x_i = jnp.asarray(x_i, dtype=jnp.float64)
        return x_i.reshape(player.n_types, player.action_size_per_type)
    
    def split_profiles(self, x_structured: Sequence[jnp.ndarray]) -> List[jnp.ndarray]:
        """
        Convert each player's flat vector into per-type action rows.
        """
        return [
            self.reshape_player_actions(x_i, i)
            for i, x_i in enumerate(x_structured)
        ]


    def expected_action(self, x_i, player_idx: int) -> jnp.ndarray:
        player = self.players[player_idx]
        if player.type_probs is None:
            raise ValueError(
                f"Player {player.name} has no type_probabilities."
            )

        actions = self.reshape_player_actions(x_i, player_idx)
        probs = jnp.asarray(player.type_probs, dtype=jnp.float64).reshape(-1, 1)
        return jnp.sum(probs * actions, axis=0)
    
    
    def expected_other_actions(self, x_structured, player_idx: int):
        """
        Return the expected actions of all players except player_idx.
        """
        return [
            self.expected_action(x_i, i)
            for i, x_i in enumerate(x_structured)
            if i != player_idx
        ]
    
    def get_others_idx(self, player_idx: int):
        return [i for i in range(len(self.players)) if i != player_idx]

    def type_weighted_sum(self, values, player_idx: int):
        """
        values should have first dimension = n_types for that player.
        """
        player = self.players[player_idx]
        if player.type_probs is None:
            raise ValueError(
                f"Player {player.name} has no type_probabilities."
            )

        probs = jnp.asarray(player.type_probs, dtype=jnp.float64)
        values = jnp.asarray(values, dtype=jnp.float64)
        return jnp.sum(probs * values)
    


    @validate_problem_functions(derivative=False)
    def objectives(self):
        """Return a list of the objectives of the problem."""
        pass

    @validate_problem_functions(derivative=False)
    def constraints(self):
        """Return a list of the constraints of the problem."""
        pass

    @abstractmethod
    def define_players(self) -> List[BayesianPlayer]:
        """
        Override this to define players inside the subclass.
        Return None or an empty list if players must be added externally.
        """
        pass


    def solve(self, Algorithm: Type[BaseAlgorithm]):
        return super().solve(Algorithm)

