from abc import abstractmethod
from typing import List, Type, Sequence, Callable
from solvers.algorithms.BaseAlgorithm import BaseAlgorithm
from solvers.dgbne_solver.BayesianPlayer import BayesianPlayer
from solvers.validation import validate_problem_functions
from solvers.gnep_solver import *
from solvers.gnep_solver.BaseProblem import BaseProblem
import jax.numpy as jnp

class BayesianProblem(BaseProblem):
    """
    Abstract base class for Bayesian Generalized Nash Equilibrium problems.

    `BayesianProblem` extends `BaseProblem` to support players with type-dependent
    actions and probabilities. Each player is represented by a `BayesianPlayer`,
    whose decision vector is interpreted as a stack of per-type action blocks.

    In addition to the standard problem interface, this class provides helper
    methods for:

    - reshaping a player's flat action vector into type-action form
    - computing expected actions under type probabilities
    - extracting the actions of other players
    - forming type-weighted sums

    Subclasses must implement `define_players()`, `objectives()`, and
    `constraints()`.

    Notes
    -----
    Objective and constraint functions must accept the full action profile as a
    list of player vectors and must return scalar values.
    """
    @property
    def players(self) -> List[BayesianPlayer]:
        """
        List of Bayesian players in the problem.

        Returns
        -------
        list of BayesianPlayer
            The players defining the type structure, action dimensions, objective
            ownership, and constraint assignment of the Bayesian problem.
        """
        return self._players

    @players.setter
    def players(self, value):
        """
        Set the player list for the Bayesian problem.

        Parameters
        ----------
        value : list of BayesianPlayer or None
            Player definitions for the problem. If `None`, players are left unset.

        Raises
        ------
        TypeError
            If `value` is not a list or if any element is not a `BayesianPlayer`.
        """
        if value is None:
            self._players = None
            return

        if not isinstance(value, list):
            raise TypeError("Players must be provided as a list.")

        self._validate_player_type(value)
        self._players = value

    def _validate_player_type(self, player_list):
        """
        Validate that all entries are `BayesianPlayer` instances.

        Parameters
        ----------
        player_list : list
            Sequence of player objects to validate.

        Raises
        ------
        TypeError
            If any entry is not an instance of `BayesianPlayer`.
        """
        for i, p in enumerate(player_list):
            if not isinstance(p, BayesianPlayer):
                raise TypeError(f"Item at index {i} must be a BayesianPlayer.")
            
    
    def reshape_player_actions(self, x_i, player_idx: int) -> jnp.ndarray:
        """
        Reshape a player's flat decision vector into type-structured form.

        Parameters
        ----------
        x_i : array-like
            Flat action vector for a single player.
        player_idx : int
            Index of the player in `self.players`.

        Returns
        -------
        jax.Array
            Array of shape `(n_types, action_size_per_type)` containing the player's
            actions separated by type.

        Notes
        -----
        This is a convenience method for writing objective and constraint functions
        in a more natural type-dependent form.
        """
        player = self.players[player_idx]
        x_i = jnp.asarray(x_i, dtype=jnp.float64)
        return x_i.reshape(player.n_types, player.action_size_per_type)
    
    def split_profiles(self, x_structured: Sequence[jnp.ndarray]) -> List[jnp.ndarray]:
        """
        Reshape every player's flat action vector into type-structured form.

        Parameters
        ----------
        x_structured : sequence of array-like
            Full action profile as a sequence of flat player vectors.

        Returns
        -------
        list of jax.Array
            List in which the i-th entry has shape
            `(n_types_i, action_size_per_type_i)` for player `i`.
        """
        return [
            self.reshape_player_actions(x_i, i)
            for i, x_i in enumerate(x_structured)
        ]


    def expected_action(self, x_i, player_idx: int) -> jnp.ndarray:
        """
        Compute the expected action of a player across its types.

        Parameters
        ----------
        x_i : array-like
            Flat action vector for a single player.
        player_idx : int
            Index of the player in `self.players`.

        Returns
        -------
        jax.Array
            Expected action vector of shape `(action_size_per_type,)`.

        Raises
        ------
        ValueError
            If the player does not define `type_probs`.

        Notes
        -----
        The expectation is computed using the player's type probabilities.
        """
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
        Compute the expected actions of all players except one.

        Parameters
        ----------
        x_structured : sequence of array-like
            Full action profile as a sequence of flat player vectors.
        player_idx : int
            Index of the excluded player.

        Returns
        -------
        list of jax.Array
            Expected action vectors for all players other than `player_idx`.
        """
        return [
            self.expected_action(x_i, i)
            for i, x_i in enumerate(x_structured)
            if i != player_idx
        ]
    
    def get_others_idx(self, player_idx: int):
        """
        Return the indices of all players other than the selected player.

        Parameters
        ----------
        player_idx : int
            Index of the reference player.

        Returns
        -------
        list of int
            Indices of all players except `player_idx`.
        """
        return [i for i in range(len(self.players)) if i != player_idx]

    def type_weighted_sum(self, values, player_idx: int):
        """
        Compute a probability-weighted sum over a player's types.

        Parameters
        ----------
        values : array-like
            Values indexed by the player's types. The first dimension must match the
            number of types for the selected player.
        player_idx : int
            Index of the player in `self.players`.

        Returns
        -------
        jax.Array
            Scalar weighted sum of the provided values.

        Raises
        ------
        ValueError
            If the player does not define `type_probs`.

        Notes
        -----
        This helper is useful when an objective or constraint is first computed
        per type and then aggregated into a scalar quantity.
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
    def objectives(self) -> List[Callable]:
        """
        Return the objective functions of the Bayesian problem.

        Returns
        -------
        list of callable
            List of scalar-valued objective functions. Each function must accept the
            full action profile as a list of player vectors.

        Notes
        -----
        In Bayesian problems, objective functions often use helpers such as
        `reshape_player_actions()`, `expected_action()`, and `type_weighted_sum()`
        to work with type-dependent actions.
        """
        pass

    @validate_problem_functions(derivative=False)
    def constraints(self) -> List[Callable]:
        """
        Return the constraint functions of the Bayesian problem.

        Returns
        -------
        list of callable
            List of scalar-valued constraint functions. Each function must accept the
            full action profile as a list of player vectors.

        Notes
        -----
        Constraints may be written directly in terms of flat player vectors or in
        terms of expected/type-structured actions using the helper methods provided
        by this class.
        """
        pass

    @abstractmethod
    def define_players(self) -> List[BayesianPlayer]:
        """
        Define the Bayesian players in the problem.

        Returns
        -------
        list of BayesianPlayer
            Player objects describing the total decision dimension, type values,
            type probabilities, action size per type, objective mapping, constraint
            mapping, and optional bounds for each player.

        Notes
        -----
        Subclasses should override this method to provide the default Bayesian
        player definitions for the problem.
        """
        pass


    def solve(self, Algorithm: Type[BaseAlgorithm]):
        """
        Solve the Bayesian generalized Nash equilibrium problem.

        Parameters
        ----------
        Algorithm : type[BaseAlgorithm]
            Algorithm class used to solve the problem.

        Returns
        -------
        tuple
            A pair `(primal_x, dual_x)` where `primal_x` contains the stacked primal
            variables and `dual_x` contains the associated dual variables.

        Notes
        -----
        This method delegates to `BaseProblem.solve(...)` and preserves the same
        solver workflow for Bayesian problems.
        """
        return super().solve(Algorithm)

