import numpy as np
import jax.numpy as jnp
from typing import List, Union, Tuple, Callable, Optional
from .schema import VectorList, Vector, Matrix, PlayerConstraint, ObjFunction

def flatten_variables(vectors: VectorList, scalars: List[float]) -> jnp.ndarray:
    """
    Flatten a collection of vectors and scalars into a single JAX array.

    This function concatenates a list of player action vectors and a list of
    additional scalars (such as Lagrange multipliers or slack variables) into
    a single 1D array. It is optimized for use within JIT-compiled functions.

    Parameters
    ----------
    vectors : VectorList
        List of JAX arrays (usually column vectors) representing player strategies.
    scalars : List[float]
    	A list of scalar values to append after the flattened vectors.

    Returns
    -------
    jax.Array
        A 1D JAX array containing all elements from the input vectors,
        followed by the scalar values.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> vectors = [jnp.array([[1.0], [2.0]]), jnp.array([[3.0]])]
    >>> scalars = [4.0, 5.0]
    >>> flatten_variables(vectors, scalars)
    Array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float32)
    """
    # 1. Flatten all vectors into 1D and combine with the scalars
    # jnp.concatenate is faster than hstack for a mix of arrays and lists
    # We convert scalars to an array first to ensure a single JAX operation
    flat_vectors = [v.ravel() for v in vectors]
    flat_scalars = jnp.atleast_1d(jnp.array(scalars))
    return jnp.concatenate([*flat_vectors, flat_scalars])

def construct_vectors(actions: Vector, action_sizes: List[int]) -> VectorList:
    """
    Split a concatenated action array into separate action vectors for each player.

    This function validates the input types and shapes, ensuring that the total
    number of rows in ``actions`` matches the sum of ``action_sizes``. It then
    partitions the flattened array into a list of per-player subarrays, each
    reshaped back to a column vector.

    Parameters
    ----------
    actions : jax.Array
        A JAX array of shape (sum(action_sizes), 1) containing all players'
        actions stacked vertically.
    action_sizes : list of int
        A list specifying the number of decision variables for each player.
        The sum of these sizes must match the total number of elements in ``actions``.

    Returns
    -------
    list of jax.Array
        A list of 2D JAX arrays (column vectors), where the i-th element has
        shape (action_sizes[i], 1).

    Raises
    ------
    TypeError
        If ``actions`` is not a JAX array or if ``action_sizes`` is not a
        list of integers.
    ValueError
        If the total size of ``actions`` does not equal the sum of
        ``action_sizes``.

    Notes
    -----
    To maintain JIT compatibility, ``action_sizes`` should be treated as a
    static argument, as it defines the structure of the output list.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> actions = jnp.array([[1.0], [2.0], [3.0], [4.0]])
    >>> action_sizes = [2, 2]
    >>> construct_vectors(actions, action_sizes)
    [Array([[1.], [2.]], dtype=float32), Array([[3.], [4.]], dtype=float32)]
    """
    total_size = sum(action_sizes)
    actions = jnp.asarray(actions)
    # 1. Type Validation
    # In JAX, we check for jax.Array (which covers jnp.ndarray)
    if not isinstance(actions, jnp.ndarray):
        raise TypeError(f"actions must be a JAX array, got {type(actions)}")

    if not isinstance(action_sizes, list) or not all(isinstance(x, int) for x in action_sizes):
        raise TypeError("action_sizes must be a List[int]")

    # 2. Shape Validation
    if actions.shape[0] != total_size:
        raise ValueError(
            f"Actions shape {actions.shape[0]} does not match "
            f"sum of action_sizes {total_size}."
        )

    # 1. Calculate split points (Static - happens at compile time)
    indices = []
    current = 0
    for size in action_sizes[:-1]:
        current += size
        indices.append(current)

    # 2. Split the flat array
    list_of_arrays = jnp.split(actions.ravel(), indices)

    # 3. Use vmap or a list comprehension that JAX can optimize
    # Note: Keeping them (n,) is faster than (n,1) in JAX due to broadcasting
    return [arr.reshape(-1, 1) for arr in list_of_arrays]


def one_hot_encoding(
        funcs_idx: List[PlayerConstraint],
        sizes: List[int],
        num_functions: int
) -> jnp.ndarray:
    """
    Build a matrix mapping player action variables to their assigned functions.

    This matrix (M) has shape (sum(sizes), num_functions). If variable 'i'
    is associated with function 'j', M[i, j] = 1, otherwise 0.

    Parameters
    ----------
    funcs_idx : List[PlayerConstraint]
        A list where each entry corresponds to a player and contains the
        indices of the functions they are associated with.
    sizes : List[int]
        A list of integers specifying the number of variables each player controls.
    num_functions : int
        The total number of unique functions (columns).

    Returns
    -------
    jax.Array
        A binary matrix of shape (sum(sizes), num_functions) representing
        the mapping.

    Examples
    --------
    >>> funcs_idx = [[0, 2], None, [1]]
    >>> sizes = [2, 1, 1]
    >>> num_functions = 3
    >>> one_hot_encoding(funcs_idx, sizes, num_functions)
    Array([[1, 0, 1],
           [1, 0, 1],
           [0, 0, 0],
           [0, 1, 0]], dtype=int32)
    """
    if len(funcs_idx) != len(sizes):
        raise ValueError("funcs_idx and sizes must match in length")

    total_vars = sum(sizes)
    # We build using NumPy first because JAX arrays are immutable.
    # Since this is a structural setup function, NumPy is actually more efficient here.
    M = np.zeros((total_vars, num_functions), dtype=np.int32)

    # Calculate row offsets
    current_row = 0
    for idx, player_funcs in enumerate(funcs_idx):
        size = sizes[idx]

        # Skip if player has no functions assigned
        if player_funcs is not None and player_funcs != [None]:
            # Ensure player_funcs is iterable (e.g., [0, 2])
            if isinstance(player_funcs, int):
                player_funcs = [player_funcs]

            # Use NumPy advanced indexing to set blocks of rows/columns to 1
            # This sets all variables of this player to the specified functions
            M[current_row: current_row + size, player_funcs] = 1

        current_row += size

    # Convert to JAX array for use in the solver
    return jnp.array(M)

def create_wrapped_function(
        original_func: ObjFunction,
        actions: VectorList,
        player_idx: int
) -> Callable[[jnp.ndarray], float]:
    """
    Create a JAX-compatible wrapped objective function for a single player.

    Fixes the actions of all opponents and returns a function of only the
    specified player's variables. This is optimized for JAX's automatic
    differentiation engine.

    Parameters
    ----------
    original_func : ObjFunction
        The original payoff/cost function: f(VectorList) -> float.
    actions : VectorList
        List of JAX arrays representing the current state of the game.
    player_idx : int
        The index of the player whose actions are being varied.

    Returns
    -------
    Callable[[jnp.ndarray], float]
        A function that takes a 1D or 2D JAX array for player i and returns
         the scalar objective value.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> def cost_func(xs): return jnp.sum(xs[0])**2 + jnp.sum(xs[1])
    >>> actions = [jnp.array([[1.0]]), jnp.array([[2.0]])]
    >>> wrapped = create_wrapped_function(cost_func, actions, 0)
    >>> wrapped(jnp.array([5.0]))
    27.0
    """
    # Pre-copy the list to avoid modifying the external state
    # We keep this as a list of tracers for JAX
    current_actions = list(actions)

    def wrap_func(player_var_opt: jnp.ndarray) -> float:
        # 1. Ensure the input is the correct shape (n, 1)
        # We use .reshape to handle both flat and column vector inputs
        p_var = player_var_opt.reshape(-1, 1)

        # 2. Update only the relevant player's slot
        # In a closure, JAX will treat current_actions as a 'constant'
        # except for the index we replace.
        new_actions = current_actions[:player_idx] + [p_var] + current_actions[player_idx + 1:]

        return original_func(new_actions)

    return wrap_func