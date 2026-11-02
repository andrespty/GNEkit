import inspect
from functools import wraps
from typing import Any, Callable, List, Type

import jax.numpy as jnp
from jax import eval_shape

from gnekit.core.player import Player
from gnekit.core.utils import construct_vectors

def validate_scalar_output(func: Callable, action_sizes: List[int]):
    dims = sum(action_sizes)
    dummy = construct_vectors(jnp.zeros((dims,)), action_sizes)
    out_shape = eval_shape(func, dummy)

    if out_shape.shape != ():
        raise ValueError(
            f"{func.__name__} must return scalar. "
            f"Got shape {out_shape.shape}"
        )

def validate_obj_funcs(obj_funcs: List[Callable], action_dims: List[int]) -> List[Callable]:
    """
    obj_funcs must be a list with functions that return a scalar output
    Parameters
    ----------
    obj_funcs
    action_dims

    Returns
    -------

    """
    if not isinstance(obj_funcs, list) or len(obj_funcs) == 0:
        raise ValueError("obj_funcs must be a non-empty list.")
    for f in obj_funcs:
        if not callable(f):
            raise TypeError("All objective functions must be callable.")
        validate_scalar_output(f, action_dims)
    return obj_funcs

def validate_constraint_funcs(constraint_funcs: List[Callable]) -> List[Callable]:
    if not isinstance(constraint_funcs, list):
        raise TypeError("constraints must be a list.")
    for c in constraint_funcs:
        if not callable(c):
            raise TypeError("All constraint functions must be callable.")
    return constraint_funcs

def validate_player_list(player_list: List[Any], expected_class: Type = Player):
    if not isinstance(player_list, list) or len(player_list) == 0:
        raise ValueError("player_list must be a non-empty list of Player.")
    for p in player_list:
        if not isinstance(p, expected_class):
            raise TypeError("player_list must contain Player objects.")

def validate_player_functions(player_list: List[Any], obj_funcs: List[Callable], constraints: List[Callable], expected_class: Type = Player):
    validate_player_list(player_list, expected_class)
    for p in player_list:
        # Validate objective index
        if p.f_index >= len(obj_funcs):
            raise ValueError(
                f"Player {p.name} references objective {p.f_index}, "
                f"but only {len(obj_funcs)} objectives exist."
            )

        # Validate constraint indices
        for c_idx in p.constraints:
            if c_idx is not None and c_idx >= len(constraints):
                raise ValueError(
                    f"Player {p.name} references constraint {c_idx}, "
                    f"but only {len(constraints)} constraints exist."
                )


def _takes_one_argument(func: Callable) -> bool:
    """True if ``func`` can be called with exactly one positional argument."""
    try:
        inspect.signature(func).bind(None)
    except TypeError:
        return False
    except ValueError:  # no signature available (some builtins); assume callable
        return True
    return True


def _validate_function_list(problem, kind: str, funcs, action_sizes: List[int], allow_empty: bool):
    owner = type(problem).__name__
    if not isinstance(funcs, list):
        raise TypeError(f"{owner}.{kind}() must return a list of callables, got {type(funcs).__name__}.")
    if not funcs and not allow_empty:
        raise TypeError(f"{owner}.{kind}() must return a non-empty list of callables.")
    for i, f in enumerate(funcs):
        if not callable(f):
            raise TypeError(f"{owner}.{kind}()[{i}] is not callable.")
        if not _takes_one_argument(f):
            raise ValueError(
                f"{owner}.{kind}()[{i}] must take exactly one argument, "
                f"the list of player action vectors."
            )
        try:
            validate_scalar_output(f, action_sizes)
        except ValueError as exc:
            raise ValueError(f"{owner}.{kind}()[{i}]: {exc}") from None


def validate_problem(problem) -> None:
    """
    Check a problem definition once, when the problem is created.

    Checks that ``objectives()`` returns a non-empty list and ``constraints()``
    a list (possibly empty) of callables, that every function takes exactly
    one argument (the list of player action vectors) and returns a scalar,
    and that every player's objective and constraint indices exist.

    Raises
    ------
    TypeError, ValueError
        With a message naming the problem class and the offending function or
        player.
    """
    players = problem.players
    action_sizes = [p.size for p in players]
    objectives = problem.objectives()
    constraints = problem.constraints()
    _validate_function_list(problem, "objectives", objectives, action_sizes, allow_empty=False)
    _validate_function_list(problem, "constraints", constraints, action_sizes, allow_empty=True)
    validate_player_functions(players, objectives, constraints)
