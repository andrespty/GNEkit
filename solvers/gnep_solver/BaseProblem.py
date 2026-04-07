from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import List, Type
from solvers.validation import validate_problem_functions
from ..algorithms.BaseAlgorithm import BaseAlgorithm
from solvers.gnep_solver import *
from solvers.algorithms import EnergyMethod

class BaseProblem(ABC):
    def __init__(self, players: List[Player] = None):
        self.players = players if players is not None else self.define_players()
        self.primal_ip = None
        self.dual_ip = None
        self.algorithm = None

    @property
    def players(self):
        return self._players

    @players.setter
    def players(self, value):
        # 1. Allow None (if the user hasn't provided them yet)
        if value is None:
            self._players = None
            return

        # 2. Ensure it is a list
        if not isinstance(value, list):
            raise TypeError("Players must be provided as a list.")

        self._validate_player_type(value)
        self._players = value

    def _validate_player_type(self, player_list):
        """Default validation for BaseProblem."""
        for i, p in enumerate(player_list):
            if not isinstance(p, Player):
                raise TypeError(f"Item at index {i} must be a Player object, got {type(p)}")

    @property
    def primal_ip(self):
        return self._primal_ip

    @primal_ip.setter
    def primal_ip(self, value):
        if value is None:
            self._primal_ip = None
            return

        if self.players is None:
            raise ValueError("Cannot set primal_ip before players are defined.")

        value_arr = jnp.asarray(value)
        total_vars = sum(p.size for p in self.players)

        if value_arr.size != total_vars:
            raise ValueError(
                f"Primal IP size mismatch. Provided: {value_arr.size}, "
                f"Expected (sum of player sizes): {total_vars}"
            )

        self._primal_ip = value

    @property
    def dual_ip(self):
        return self._dual_ip

    @dual_ip.setter
    def dual_ip(self, value):
        if value is None:
            self._dual_ip = None
            return

        value_arr = jnp.asarray(value)
        num_constraints = len(self.constraints())

        if value_arr.size != num_constraints:
            raise ValueError(
                f"Dual IP size mismatch. Provided: {value_arr.size}, "
                f"Expected (number of constraints): {num_constraints}"
            )

        self._dual_ip = value



    @abstractmethod
    @validate_problem_functions(derivative=False)
    def objectives(self):
        """Return a list of the objectives of the problem."""
        pass

    @abstractmethod
    @validate_problem_functions(derivative=False)
    def constraints(self):
        """Return a list of the constraints of the problem."""
        pass

    @abstractmethod
    def define_players(self) -> List[Player]:
        """
        Override this to define players inside the subclass.
        Return None or an empty list if players must be added externally.
        """
        pass

    def known_solution(self):
        """Return a list of the known solutions of the problem."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide a known solution."
        )

    def set_initial_point(self, primal_x, dual_x):
        if isinstance(primal_x, list):
            self.primal_ip = primal_x
        if isinstance(primal_x, float):
            n_vars = sum(p.size for p in self.players)
            self.primal_ip = [primal_x for _ in range(n_vars)]
        else:
            raise TypeError("Primal IP must be a list or a float.")

        if isinstance(dual_x, list):
            self.dual_ip = dual_x
        elif isinstance(dual_x, float):
            self.dual_ip = [dual_x for _ in range(len(self.constraints()))]
        else:
            raise TypeError("Dual IP must be a list or a float.")

        return self.primal_ip, self.dual_ip

    def solve(self, Algorithm:Type[BaseAlgorithm] = None):
        """Solve the problem."""
        alg = Algorithm if Algorithm else EnergyMethod
        algorithm = alg(self.objectives(), self.constraints(), self.players)

        ip = jnp.array(self.primal_ip + self.dual_ip)
        res, time = algorithm.solve(ip)

        primal_vars = sum(algorithm.action_sizes)
        primal_x = res.x[:primal_vars]
        dual_x = res.x[primal_vars:]
        return primal_x, dual_x