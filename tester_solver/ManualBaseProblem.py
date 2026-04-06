from abc import abstractmethod
from typing import List
from solvers.gnep_solver.BasePlayer import Player
from .ManualGeneralizedGame import ManualGeneralizedGame
from solvers.gnep_solver import validate_problem_functions
from solvers.gnep_solver.BaseProblem import BaseProblem

class ManualBaseProblem(BaseProblem):
    def __init__(self, players: List[Player] = None):
        super().__init__(players)
        self.engine = ManualGeneralizedGame(
            self.objectives(),
            self.objectives_der(),
            self.constraints(),
            self.constraints_der(),
            self.players
        )

    @abstractmethod
    @validate_problem_functions(derivative=True)
    def objectives_der(self):
        """Return a list of the objectives of the problem."""
        pass

    @abstractmethod
    @validate_problem_functions(derivative=True)
    def constraints_der(self):
        """Return a list of the constraints of the problem."""
        pass