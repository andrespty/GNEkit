from abc import ABC, abstractmethod
import numpy as np
from typing import List
from .GNEPlayer import GNEPlayer
from .GNESolver import GNESolver
from .validation import validate_problem_functions

class GNEProblem(ABC):
    def __init__(self, players: List[GNEPlayer] = None):
        self.players = players if players is not None else self.define_players()
        self.primal_ip = None
        self.dual_ip = None
        self.engine = GNESolver(
            self.objectives(),
            self.objectives_der(),
            self.constraints(),
            self.constraints_der(),
            self.players
        )

    @property
    def players(self):
        return self._players

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

        value_arr = np.asarray(value)
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

        value_arr = np.asarray(value)
        num_constraints = len(self.constraints())

        if value_arr.size != num_constraints:
            raise ValueError(
                f"Dual IP size mismatch. Provided: {value_arr.size}, "
                f"Expected (number of constraints): {num_constraints}"
            )

        self._dual_ip = value

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
            if not isinstance(p, GNEPlayer):
                raise TypeError(
                    f"Item at index {i} is a {type(p).__name__}, "
                    f"but it must be a Player object."
                )
        self._players = value

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
    @validate_problem_functions(derivative=True)
    def objectives_der(self):
        """Return a list of the objectives of the problem."""
        pass

    @abstractmethod
    @validate_problem_functions(derivative=True)
    def constraints_der(self):
        """Return a list of the constraints of the problem."""
        pass

    @abstractmethod
    def define_players(self) -> List[GNEPlayer]:
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

    def summary(self):
        """Return a string summary of the problem."""
        return self.engine.summary()

    def check_gradient(self, actions: np.ndarray) -> List[np.ndarray]:
        return self.engine.grad_val(actions)

    def check_energy(self, actions: np.ndarray) -> float:
        return self.engine.energy_val(actions)

    def check_kkt(self, primal, dual, tol=1e-6):
        return self.engine.check_kkt(np.array(primal), np.array(dual), tol)

    def solve(self):
        """Solve the problem."""
        ip = np.array(self.primal_ip + self.dual_ip)
        res, time = self.engine.solve_game(ip)
        primal_vars = sum(self.engine.action_sizes)
        primal_x = res.x[:primal_vars]
        dual_x = res.x[primal_vars:]
        return primal_x, dual_x

    def get_solver(self):
        return self.engine