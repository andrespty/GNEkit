import inspect
from functools import wraps
from typing import List, Callable
import numpy as jnp
from .GNEPlayer import GNEPlayer
from .utils import *

def validate_scalar_output(func: Callable, action_sizes: List[int]):
    dims = sum(action_sizes)
    dummy = construct_vectors(jnp.zeros((dims,)), action_sizes)
    out = func(dummy)

    if not np.isscalar(out) and getattr(out, 'shape', None) != ():
        raise ValueError(
            f"{func.__name__} must return a scalar. "
            f"Got type {type(out)} with shape {getattr(out, 'shape', 'N/A')}"
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

def validate_player_list(player_list: List[GNEPlayer]):
    if not isinstance(player_list, list) or len(player_list) == 0:
        raise ValueError("player_list must be a non-empty list of Player.")
    for p in player_list:
        if not isinstance(p, GNEPlayer):
            raise TypeError("player_list must contain Player objects.")

def validate_player_functions(player_list: List[GNEPlayer], obj_funcs: List[Callable], constraints: List[Callable]):
    validate_player_list(player_list)
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


# BaseProblem validation wrappers
def validate_problem_functions(derivative=False):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func_list = func(*args, **kwargs)

            # 1. Basic List Check
            if not isinstance(func_list, list) or len(func_list) == 0:
                raise TypeError(f"{func.__name__} must return a non-empty list of callables.")

            if not self.players:
                return func_list  # Can't validate shapes without players

            num_players = len(self.players)
            # We need the action sizes to check the inner shapes
            action_sizes = [p.size for p in self.players]

            for i, f in enumerate(func_list):
                if not callable(f):
                    raise TypeError(f"Item {i} in {func.__name__} is not callable.")

                # 2. Signature Check: Must take exactly 1 argument (x_structured)
                sig = inspect.signature(f)
                if len(sig.parameters) != 1:
                    raise ValueError(f"Function {i} must take exactly 1 argument.")

                # 3. Structural "Dry Run" Check
                # Create a dummy input matching the expected structure
                dummy_x = [jnp.zeros((size,)) for size in action_sizes]

                if derivative:
                    try:
                        out = f(dummy_x)
                        if not isinstance(out, (list, tuple)):
                            raise TypeError(f"Derivative {i} must return a list/tuple of gradients.")

                        if len(out) != num_players:
                            raise ValueError(
                                f"Derivative {i} returned {len(out)} gradient components, "
                                f"but there are {num_players} players."
                            )
                    except Exception as e:
                        raise ValueError(f"Derivative {i} failed structural validation: {e}")
                else:
                    validate_scalar_output(f, action_sizes)
            return func_list
        return wrapper
    return decorator

