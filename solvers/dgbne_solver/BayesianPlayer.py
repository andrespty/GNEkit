from typing import List, Optional, Tuple
from dataclasses import dataclass
from solvers.CorePlayer import PlayerValidator
from solvers.gnep_solver import Player
import numpy as np

from solvers.gnep_solver import Player

@dataclass(frozen=True)
class BayesianPlayer(Player):
    """
    A player in a Bayesian generalized Nash equilibrium problem.

    Extends `Player` with private type information. Each player has a
    finite set of possible types, and maintains a separate action variable for
    each type. The total decision variable size must satisfy:
        ``size == len(type_values) * action_size_per_type``.
    Parameters
    ----------
    type_values : tuple of float
        The possible private type realizations for this player. Must be
        non-empty.
    type_probs : tuple of float or None, optional
        Prior probabilities over types. Must sum to 1 and match the length
        of ``type_values``. If ``None``, no prior is assigned.
    action_size_per_type : int, optional
        Dimension of the action variable for each type. Defaults to 1.

    Raises
    ------
    ValueError
        If ``type_values`` is empty.
    ValueError
        If ``action_size_per_type`` is not positive.
    ValueError
        If ``size != len(type_values) * action_size_per_type``.
    ValueError
        If ``type_probs`` does not match the length of ``type_values`` or
        does not sum to 1.

    Examples
    --------
    >>> p = BayesianPlayer(
    ...     name="P1",
    ...     size=2,
    ...     f_index=0,
    ...     constraints=(0,),
    ...     type_values=(0.5, 1.0),
    ...     type_probs=(0.4, 0.6),
    ...     action_size_per_type=1,
    ... )
    >>> p.n_types
    2

    See Also
    --------
    `Player` : Base class for standard (non-Bayesian) players.
    """
    type_values: Tuple[float, ...] = ()
    type_probs: Optional[Tuple[float, ...]] = None
    action_size_per_type: int = 1

    def __post_init__(self):
        super().__post_init__()  # Validate base Player attributes first

        if len(self.type_values) == 0:
            raise ValueError("type_values must be non-empty.")

        if self.action_size_per_type <= 0:
            raise ValueError("action_size_per_type must be positive.")

        expected_size = len(self.type_values) * self.action_size_per_type
        if self.size != expected_size:
            raise ValueError(
                f"size must equal len(type_values) * action_size_per_type. "
                f"Got size={self.size}, expected={expected_size}."
            )

        if self.type_probs is not None:
            if len(self.type_probs) != len(self.type_values):
                raise ValueError(
                    "type_probs must have the same length as type_values."
                )
            if not np.isclose(sum(self.type_probs), 1.0):
                raise ValueError("type_probs must sum to 1.0.")
            
    @property
    def n_types(self) -> int:
        """Number of private types for this player."""
        return len(self.type_values)
    


    @classmethod
    def batch_create(cls, sizes, type_values, objectives,constraints,bounds=None,names=None, type_probs=None,
        action_size_per_type=None,):
        """
        Construct multiple Bayesian players from parallel attribute lists.

        Parameters
        ----------
        sizes : list of int
            Decision variable dimension for each player. Must equal
            ``len(type_values[i]) * action_size_per_type[i]`` for each ``i``.
        type_values : list of tuple of float
            Type realizations for each player.
        objectives : list of int
            Objective index for each player.
        constraints : list of tuple
            Constraint participation tuples, one per player.
        bounds : list of (float, float) or None, optional
            Per-player scalar bounds. Defaults to ``None`` for all players.
        names : list of str, optional
            Player names. Defaults to ``"P0"``, ``"P1"``, ... if omitted.
        type_probs : list of tuple of float or None, optional
            Prior probabilities over types per player. Defaults to ``None``
            for all players.
        action_size_per_type : list of int, optional
            Action dimension per type per player. Defaults to 1 for all
            players.

        Returns
        -------
        list of BayesianPlayer
            Players constructed from the provided attributes.

        Raises
        ------
        ValueError
            If any of the provided lists differ in length.

        Examples
        --------
        >>> players = BayesianPlayer.batch_create(
        ...     sizes=[2, 2],
        ...     type_values=[[0.5, 1.0], [0.2, 0.8]],
        ...     objectives=[0, 1],
        ...     constraints=[(0,), (0,)],
        ...     type_probs=[[0.4, 0.6], [0.5, 0.5]],
        ... )
        >>> len(players)
        2
        """
        n_var = len(sizes)

        if bounds is None:
            bounds = [None] * n_var

        if names is None:
            names = [f"P{i}" for i in range(n_var)]

        if type_probs is None:
            type_probs = [None] * n_var

        if action_size_per_type is None:
            action_size_per_type = [1] * n_var

        if not (len(sizes)
                == len(type_values) 
                == len(objectives) 
                == len(constraints) 
                == len(bounds) 
                == len(names)
                == len(type_probs)
                == len(action_size_per_type)
            ):
            raise ValueError("All player attribute lists must have the same length.")
        
        return [
            cls(
                name=name,
                size=size,
                f_index=obj,
                constraints=const,
                bounds=bound,
                type_values=tuple(tvals),
                type_probs=None if probs is None else tuple(probs),
                action_size_per_type=a_per_type,
            )
            for name, size, tvals, obj, const, bound, probs, a_per_type in zip(
                names,
                sizes,
                type_values,
                objectives,
                constraints,
                bounds,
                type_probs,
                action_size_per_type,
            )
        ]

def bayesian_players_to_lists(players):
    """
    Decompose a list of Bayesian players into parallel attribute lists.

    Extends `players_to_lists` with Bayesian-specific fields:
    ``type_values``, ``type_probs``, and ``action_size_per_type``.

    Parameters
    ----------
    players : list of BayesianPlayer
        The players to decompose.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``"sizes"`` — list of int, one per player.
        - ``"type_values"`` — list of tuple, type realizations per player.
        - ``"type_probs"`` — list of tuple or None, priors per player.
        - ``"action_size_per_type"`` — list of int, action dimension per type.
        - ``"objectives"`` — list of int, objective index per player.
        - ``"constraints"`` — list of tuple, constraint indices per player.
        - ``"bounds"`` — flat list of ``(lower, upper)`` pairs expanded
          across all player variables.
        - ``"names"`` — list of str or None, one per player.

    See Also
    --------
    [`players_to_lists`](./players_to_lists.md) : The equivalent function for standard players.
    [`BayesianPlayer.batch_create`](../players/bayesian_player.md) : Inverse operation.
    """
    return {
        "sizes": [p.size for p in players],
        "type_values": [p.type_values for p in players],
        "type_probs": [p.type_probs for p in players],
        "action_size_per_type": [p.action_size_per_type for p in players],
        "objectives": [p.f_index for p in players],
        "constraints": [p.constraints for p in players],
        "bounds": [bound for p in players for bound in p.get_full_bounds()],
        "names": [p.name for p in players],
    }