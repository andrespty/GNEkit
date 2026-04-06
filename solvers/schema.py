from typing import List, Tuple, Dict, Optional, Callable, Union, Any
import jax
import jax.numpy as jnp
from jax import Array # The JAX equivalent of NDArray


# We use Array for both Vector and Matrix in JAX
Vector = Array
Matrix = Array
VectorList = List[Vector]
PlayerConstraint = Union[List[int], None, List[None]]

# --- Function Types ---
# ObjFunction: Takes the strategies of all players, returns the scalar payoff/cost for one player.
# Note: In GNEPs, the objective usually returns a float (the cost to be minimized).
ObjFunction = Callable[[VectorList], float]

# ObjFunctionGrad: Returns the gradient of the objective with respect to a player's own variables.
# It should return a Vector (the same shape as the player's decision variable).
ObjFunctionGrad = Callable[[VectorList], Vector]

# ConsFunction: Returns a Vector of constraint violations (g(x) <= 0).
ConsFunction = Callable[[VectorList], Vector]

# ConsFunctionGrad: Usually represents the Jacobian of the constraints.
ConsFunctionGrad = Callable[[VectorList], Matrix]

# WrappedFunction: Often used for internal solver loops where inputs are flattened.
WrappedFunction = Callable[[Array], Array]

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