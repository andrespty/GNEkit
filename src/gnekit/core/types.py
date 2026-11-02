"""
Type aliases used throughout the solver.

All function types assume the **action profile convention**: functions receive
a :class:`VectorList` (one array per player) and return either a scalar or a
vector.
"""

from typing import List, Tuple, Dict, Optional, Callable, Union, Any
import jax
import jax.numpy as jnp
from jax import Array # The JAX equivalent of NDArray
from typing_extensions import TypeAlias

# --- Array types ---

#: A 1D or 2D JAX array representing a single player's decision variable.
Vector: TypeAlias = Array # JAX Array, shape (n,1)

#: A 2D JAX array, typically a Jacobian or weight matrix.
Matrix: TypeAlias = Array # JAX Array, shape (m,n)

#: A list of player action vectors, one per player.
VectorList: TypeAlias = List[Vector]

#: Constraint participation spec for a player. ``None`` means the player
#: has no assigned constraints.
PlayerConstraint: TypeAlias = Union[List[int], None, List[None]]

# --- Function types ---

#: Objective function: maps the full action profile to a scalar cost.
ObjFunction: TypeAlias = Callable[[VectorList], float]

#: Gradient of an objective w.r.t. a single player's variables.
ObjFunctionGrad: TypeAlias = Callable[[VectorList], Vector]

#: Constraint function: maps the full action profile to a vector of
#: violations ``g(x) <= 0``.
ConsFunction: TypeAlias = Callable[[VectorList], Vector]

#: Jacobian of the constraints w.r.t. a player's variables.
ConsFunctionGrad: TypeAlias = Callable[[VectorList], Matrix]

#: Wrapped function used internally by the solver with flattened inputs.
WrappedFunction: TypeAlias = Callable[[Array], Array]

__all__ = [
    "Vector",
    "Matrix",
    "VectorList",
    "PlayerConstraint",
    "ObjFunction",
    "ObjFunctionGrad",
    "ConsFunction",
    "ConsFunctionGrad",
    "WrappedFunction",
]