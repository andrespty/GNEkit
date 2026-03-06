from abc import ABC, abstractmethod
import jax.numpy as jnp
from gnep_solver.schema import *
from typing import List
from gnep_solver.Player import Player
from gnep_solver.GeneralizedGame import GeneralizedGame
from .ManualGeneralizedGame import ManualGeneralizedGame
from gnep_solver.validation import validate_math_functions

class BaseProblem(ABC):
    def __init__(self, players: List[Player] = None):
        self.players = players if players is not None else self.define_players()

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

        # 3. Validate every item in the list
        for i, p in enumerate(value):
            if not isinstance(p, Player):
                raise TypeError(
                    f"Item at index {i} is a {type(p).__name__}, "
                    f"but it must be a Player object."
                )
        self._players = value

    @abstractmethod
    @validate_math_functions
    def objectives(self):
        """Return a list of the objectives of the problem."""
        pass

    @abstractmethod
    def objectives_der(self):
        """Return a list of the objectives of the problem."""
        pass

    @abstractmethod
    @validate_math_functions
    def constraints(self):
        """Return a list of the constraints of the problem."""
        pass

    @abstractmethod
    def constraints_der(self):
        """Return a list of the constraints of the problem."""
        pass

    @abstractmethod
    def define_players(self) -> List[Player]:
        """
        Override this to define players inside the subclass.
        Return None or an empty list if players must be added externally.
        """
        pass

    def add_players(self, players: List[Player]):
        """Add players to the problem."""
        if self.players is not None:
            print(f"Warning: Overwriting existing players in {self.__class__.__name__}")
        self.players = players

    def known_solution(self):
        """Return a list of the known solutions of the problem."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide a known solution."
        )

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