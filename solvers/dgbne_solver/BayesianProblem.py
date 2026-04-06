from typing import List, Type
from solvers.dgbne_solver.BayesianPlayer import BayesianPlayer
from solvers.validation import validate_problem_functions
from .algorithms import BayesianAlgorithm
from .algorithms.EnergyMethodBayesian import EnergyMethodBayesian
from solvers.gnep_solver import *
from solvers.gnep_solver.BaseProblem import BaseProblem

class BayesianProblem(BaseProblem):
    def _validate_player_type(self, player_list):
        for i, p in enumerate(player_list):
            if not isinstance(p, BayesianPlayer):
                raise TypeError(f"Item at index {i} must be a BayesianPlayer.")

    @validate_problem_functions(derivative=False)
    def objectives(self):
        """Return a list of the objectives of the problem."""
        pass

    @validate_problem_functions(derivative=False)
    def constraints(self):
        """Return a list of the constraints of the problem."""
        pass

    def define_players(self) -> List[Player]:
        """
        Override this to define players inside the subclass.
        Return None or an empty list if players must be added externally.
        """
        pass


    def solve(self, Algorithm: Type['BayesianAlgorithm'] = EnergyMethodBayesian):
        return super().solve(Algorithm)

