from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import jax.numpy as jnp
from solvers.CorePlayer import PlayerValidator

@dataclass(frozen=True)
class Player:
    """
    Immutable description of a single player in a generalized Nash equilibrium problem.

    A ``Player`` specifies the dimension of a player's decision variable, which
    objective function they minimize, which constraints they participate in, and
    optional variable bounds.

    Parameters
    ----------
    name : str or None
        Human-readable identifier for the player.
    size : int
        Dimension of the player's decision variable.
    f_index : int
        Index into the problem's objective list that this player minimizes.
    constraints : tuple of int or None
        Indices of the constraints this player participates in. An empty tuple
        means the player has no shared constraints.
    bounds : tuple of (float, float) or None
        A ``(lower, upper)`` scalar bound applied uniformly to all of the
        player's variables. If ``None``, variables are unbounded.

    Examples
    --------
    >>> p = Player(name="P1", size=2, f_index=0, constraints=(0, 1), bounds=(0.0, 5.0))
    >>> p.size
    2
    """

    name: Optional[str]
    size: int
    f_index: int
    constraints: Tuple[Optional[int], ...] = ()
    bounds: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        validator = PlayerValidator()
        validator.validate(self.name, self.size, self.f_index, self.constraints, self.bounds)

    def get_full_bounds(self):
        """
        Return per-variable bounds as a list of ``(lower, upper)`` pairs.

        Expands the player's scalar ``bounds`` attribute into one pair per
        variable. If ``bounds`` is ``None``, each variable is assigned
        ``(-inf, inf)``.

        Returns
        -------
        list of (float, float)
            A list of length ``size`` where each entry is a
            ``(lower, upper)`` pair for the corresponding variable.

        Examples
        --------
        >>> Player(name="P1", size=2, f_index=0, bounds=(0.0, 1.0)).get_full_bounds()
        [(0.0, 1.0), (0.0, 1.0)]

        >>> Player(name="P1", size=2, f_index=0).get_full_bounds()
        [(-inf, inf), (-inf, inf)]
        """
        if self.bounds is None:
            # Default to large values if no bounds provided
            lb_arr = np.full((self.size,), -np.inf)
            ub_arr = np.full((self.size,), np.inf)

        elif isinstance(self.bounds, tuple):
            lb, ub = self.bounds
            # Ensure lb/ub are scalars or compatible with np.full
            lb_arr = np.full((self.size,), float(lb))
            ub_arr = np.full((self.size,), float(ub))

        else:
            # Case where self.bounds is already an array/list of bounds
            # Ensure it is a NumPy array to avoid JAX leakage
            bounds_np = np.asarray(self.bounds)
            lb_arr, ub_arr = bounds_np[:, 0], bounds_np[:, 1]

        return list(zip(lb_arr.tolist(), ub_arr.tolist()))

    @classmethod
    def batch_create(cls,sizes,objectives,constraints,bounds=None,names=None):
        """
        Construct multiple players from parallel attribute lists.

        A convenience factory for problems with many players, avoiding
        repeated ``Player(...)`` calls.

        Parameters
        ----------
        sizes : list of int
            Decision variable dimension for each player.
        objectives : list of int
            Objective index for each player.
        constraints : list of tuple
            Constraint participation tuples, one per player.
        bounds : list of (float, float) or None, optional
            Per-player scalar bounds. Defaults to ``None`` (unbounded) for
            all players if omitted.
        names : list of str, optional
            Player names. Defaults to ``"P0"``, ``"P1"``, ... if omitted.

        Returns
        -------
        list of Player
            Players constructed from the provided attributes.

        Raises
        ------
        ValueError
            If the provided lists differ in length.

        Examples
        --------
        >>> players = Player.batch_create(
        ...     sizes=[1, 1],
        ...     objectives=[0, 1],
        ...     constraints=[(0,), (0,)],
        ...     bounds=[(0.0, 5.0), (0.0, 5.0)],
        ... )
        >>> len(players)
        2
        """
        if bounds is None:
            bounds = [None] * len(sizes)

        if names is None:
            names = [f"P{i}" for i in range(len(sizes))]

        if not (len(sizes) == len(objectives) == len(constraints) == len(bounds) == len(names)):
            raise ValueError("All player attribute lists must have the same length.")

        return [
            cls(name, size, obj, const, bound)
            for name, size, obj, const, bound
            in zip(names, sizes, objectives, constraints, bounds)
        ]

def players_to_lists(players: List[Player]):
    """
    Decompose a list of players into parallel attribute lists.

    Converts a structured list of `Player` objects into a dictionary
    of flat lists, suitable for passing to solver internals that expect
    separated attribute arrays.

    Parameters
    ----------
    players : list of Player
        The players to decompose.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``"sizes"`` — list of int, one per player.
        - ``"objectives"`` — list of int, objective index per player.
        - ``"constraints"`` — list of tuple, constraint indices per player.
        - ``"bounds"`` — flat list of ``(lower, upper)`` pairs, expanded
          across all player variables via :meth:`Player.get_full_bounds`.
        - ``"names"`` — list of str or None, one per player.

    See Also
    --------
    [`Player.batch_create`](../players/base_player.md) : The inverse operation; constructs players from
        parallel attribute lists.

    Examples
    --------
    >>> players = [Player(name="P1", size=1, f_index=0, constraints=(0,), bounds=(0.0, 1.0))]
    >>> players_to_lists(players)
    {'sizes': [1], 'objectives': [0], 'constraints': [(0,)], 'bounds': [(0.0, 1.0)], 'names': ['P1']}
    """
    return {
        "sizes": [p.size for p in players],
        "objectives": [p.f_index for p in players],
        "constraints": [p.constraints for p in players], # nested list of int
        "bounds": [bound for p in players for bound in p.get_full_bounds()], # list of tuples
        "names": [p.name for p in players],
    }
