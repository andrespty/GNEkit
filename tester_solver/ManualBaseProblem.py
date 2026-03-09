from abc import ABC, abstractmethod
import jax.numpy as jnp
from gnep_solver.schema import *
from typing import List
from gnep_solver.Player import Player
from gnep_solver.GeneralizedGame import GeneralizedGame
from .ManualGeneralizedGame import ManualGeneralizedGame
from gnep_solver.validation import validate_derivative_functions
from gnep_solver.BaseProblem import BaseProblem

class ManualBaseProblem(BaseProblem):
    def __init__(self, players: List[Player] = None):
        super().__init__(players)

    @abstractmethod
    @validate_derivative_functions
    def objectives_der(self):
        """Return a list of the objectives of the problem."""
        pass

    @abstractmethod
    @validate_derivative_functions
    def constraints_der(self):
        """Return a list of the constraints of the problem."""
        pass

    def solver(self) -> ManualGeneralizedGame:
        """Solve the problem."""
        if self.players is None:
            raise ValueError(
            f"Cannot solve {self.__class__.__name__}: No players defined. "
            "Pass them to __init__ or implement define_players()."
        )
        return ManualGeneralizedGame(
            self.objectives(),
            self.objectives_der(),
            self.constraints(),
            self.constraints_der(),
            self.players
        )