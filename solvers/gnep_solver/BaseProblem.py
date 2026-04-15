from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import List, Type, Callable
from solvers.validation import validate_problem_functions
from ..algorithms.BaseAlgorithm import BaseAlgorithm
from solvers.gnep_solver import *
from solvers.algorithms import EnergyMethod

class BaseProblem(ABC):
    """
    Abstract base class for standard generalized Nash equilibrium problems.

    A `BaseProblem` defines the mathematical structure of a GNE
    through three core components:

    - a list of players
    - a list of objective functions
    - a list of constraint functions

    Subclasses must implement `define_players()`, `objectives()`, and
    `constraints()`.

    The usual workflow is:

    1. instantiate the problem
    2. set the initial primal and dual points with `set_initial_point(...)`
    3. solve the problem with `solve(...)`

    Notes
    -----
    Objective and constraint functions must accept the full action profile as a
    list of player vectors and must return scalar values.
    """
    def __init__(self, players: List[Player] = None):
        """
        Initialize the problem.

        Parameters
        ----------
        players : list of Player, optional
            Explicit list of players for the problem. If omitted, the player list is
            obtained from `define_players()`.

        Notes
        -----
        The primal and dual initial points are initialized to `None`. They must be
        set before calling `solve()`.
        """
        self.players = players if players is not None else self.define_players()
        self.primal_ip = None
        self.dual_ip = None
        self.algorithm = None

    @property
    def players(self):
        """
        List of players in the problem.

        Returns
        -------
        list of Player
            The players defining the variable structure, objective ownership, and
            constraint assignment of the problem.
        """
        return self._players

    @players.setter
    def players(self, value):
        """
        Set the player list for the problem.

        Parameters
        ----------
        value : list of Player or None
            Player definitions for the problem. If `None`, players are left unset.

        Raises
        ------
        TypeError
            If `value` is not a list or if any element is not a `Player`.
        """
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
        """
        Validate that all entries are standard `Player` instances.

        Parameters
        ----------
        player_list : list
            Sequence of player objects to validate.

        Raises
        ------
        TypeError
            If any entry is not an instance of `Player`.
        """
        for i, p in enumerate(player_list):
            if not isinstance(p, Player):
                raise TypeError(f"Item at index {i} must be a Player object, got {type(p)}")

    @property
    def primal_ip(self):
        """
        Initial primal point.

        Returns
        -------
        list or None
            Initial values for the stacked primal decision variables.
        """
        return self._primal_ip

    @primal_ip.setter
    def primal_ip(self, value):
        """
        Set the initial primal point.

        Parameters
        ----------
        value : array-like or None
            Initial values for the stacked primal variables.

        Raises
        ------
        ValueError
            If players are not yet defined or if the size of the provided vector
            does not match the total number of player variables.
        """
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
        """
        Initial dual point.

        Returns
        -------
        list or None
            Initial values for the dual variables associated with the constraints.
        """
        return self._dual_ip

    @dual_ip.setter
    def dual_ip(self, value):
        """
        Set the initial dual point.

        Parameters
        ----------
        value : array-like or None
            Initial values for the dual variables.

        Raises
        ------
        ValueError
            If the size of the provided vector does not match the number of
            constraints in the problem.
        """
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
    def objectives(self) -> List[Callable]:
        """
        Return the objective functions of the problem.

        Returns
        -------
        list of callable
            List of scalar-valued objective functions. Each function must accept the
            full action profile as a list of player vectors.

        Notes
        -----
        The objective associated with each player is selected through that player's
        `f_index`.
        """
        pass

    @abstractmethod
    @validate_problem_functions(derivative=False)
    def constraints(self) -> List[Callable]:
        """
        Return the constraint functions of the problem.

        Returns
        -------
        list of callable
            List of scalar-valued constraint functions. Each function must accept the
            full action profile as a list of player vectors.

        Notes
        -----
        Constraint participation is assigned through each player's `constraints`
        field.
        """
        pass

    @abstractmethod
    def define_players(self) -> List[Player]:
        """
        Define the players in the problem.

        Returns
        -------
        list of Player
            Player objects describing the dimension, objective mapping, constraint
            mapping, and optional bounds for each player.

        Notes
        -----
        Subclasses should override this method to provide the default player
        definitions for the problem.
        """
        pass

    def known_solution(self):
        """
        Return a known solution for the problem, if available.

        Returns
        -------
        Any
            A known reference solution for the problem.

        Raises
        ------
        NotImplementedError
            If the problem class does not provide a known solution.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide a known solution."
        )

    def set_initial_point(self, primal_x, dual_x):
        """
        Set the initial primal and dual points.

        Parameters
        ----------
        primal_x : list or float
            Initial primal values. If a float is provided, it is broadcast to all
            primal variables.
        dual_x : list or float
            Initial dual values. If a float is provided, it is broadcast to all
            constraint multipliers.

        Returns
        -------
        tuple
            A pair `(primal_ip, dual_ip)` containing the initialized primal and dual
            vectors.

        Raises
        ------
        TypeError
            If `primal_x` or `dual_x` is neither a list nor a float.

        Notes
        -----
        The primal dimension must match the total number of player variables, and
        the dual dimension must match the number of constraints.
        """
        if isinstance(primal_x, list):
            self.primal_ip = primal_x
        elif isinstance(primal_x, float):
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
        """
        Solve the generalized Nash equilibrium problem.

        Parameters
        ----------
        Algorithm : type[BaseAlgorithm], optional
            Algorithm class used to solve the problem. If omitted, `EnergyMethod` is
            used by default.

        Returns
        -------
        tuple
            A pair `(primal_x, dual_x)` where `primal_x` contains the stacked primal
            variables and `dual_x` contains the associated dual variables.

        Notes
        -----
        This method builds the selected algorithm from the current objectives,
        constraints, and players, then solves the problem starting from the stored
        initial point.
        """
        alg = Algorithm if Algorithm else EnergyMethod
        algorithm = alg(self.objectives(), self.constraints(), self.players)

        ip = jnp.array(self.primal_ip + self.dual_ip)
        res, time = algorithm.solve(ip)

        primal_vars = sum(algorithm.action_sizes)
        primal_x = res.x[:primal_vars]
        dual_x = res.x[primal_vars:]
        return primal_x, dual_x