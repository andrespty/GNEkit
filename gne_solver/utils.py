from .misc import *
from .types import *
from typing import List, Union, Tuple
import numpy as np

def construct_vectors(actions: Vector, action_sizes: List[int]) -> VectorList:
    """
        Split a concatenated action array into separate action vectors for each player.

        This function validates the input types and shapes, ensuring that the total
        number of rows in "actions" matches the sum of "action_sizes". It then splits
        the stacked column vector into per-player subarrays in the same order as specified
        in "action_sizes".

        Parameters
        ----------
        actions : numpy.ndarray of shape (sum(action_sizes), 1)
            A 2D NumPy array containing all players' actions stacked vertically.
            The number of rows must equal the sum of all entries in ``action_sizes``.
        action_sizes : list of int
            A list specifying the length of each player's action vector.
            The sum of these sizes must match the number of rows in ``actions``.

        Raises
        -------
        Type Error
            if "actions" is not a NumPy array or if "action_sizes" is not a list
            of integers
        Value Error
            If the number of rows in "actions" does not equal the sum of all
            entries in ``action_sizes``.

        Returns
        -------
        list of numpy.ndarray
            A list of 2D NumPy arrays, each corresponding to one player's action vector.
            The arrays are in the same order as the players in ``action_sizes``.

        Examples
        --------
        >>> actions = np.array([[1.0], [2.0], [3.0], [4.0]])
        >>> action_sizes = [2, 2]
        >>> construct_vectors(actions, action_sizes)
        [array([[1.],
                [2.]]),
         array([[3.],
                [4.]])]
    """
    total_size = sum(action_sizes)
    if not isinstance(actions, np.ndarray):
        raise TypeError("actions must be a 2D NumPy array")

    if not isinstance(action_sizes, list):
        raise TypeError("action_sizes must be a list")

    if not all(isinstance(x, int) for x in action_sizes):
        bad = [type(x).__name__ for x in action_sizes if not isinstance(x, int)]
        raise TypeError("action_sizes must be a list with only integers")

    if actions.shape[0] != total_size:
        raise ValueError(f"Number of rows in 'actions' ({actions.shape[0]})"
                         f"must equal the sum of 'action_sizes' ({total_size})"
        )
    value_array = np.array(actions).reshape(-1,1)
    indices = np.cumsum(action_sizes)
    return np.split(value_array, indices[:-1])

def one_hot_encoding(funcs_idx: List[Union[int, PlayerConstraint]], sizes: List[int], num_functions: int) -> Matrix:
    """
        This function builds a matrix mapping each player’s action variables to the functions they are assigned
         Creates a zeros NumPy matrix and then iterates through functions mapping them to correct player's action variables


        and assigning them to the matrix.
        Parameters
        ----------
        funcs_idx : List
            A list of either integers or PlayerConstraint (PlayerConstraint=Union[int], None, list[None])
        sizes: List[int]
            A list of integers thats length needs to equal the length of funcs_idx
        num_functions: Int
            An integer value that indicates the number of possible functions
        Returns
        -------
        Matrix
            Returns a matrix of shape (sum(sizes), num_functions), where
            each row represents a player's variables and each column represents
            a function.

        Examples
        --------
        >>> import numpy as np
        >>> funcs_idx = [[0, 2], None, [1]]
        >>> sizes = [2, 3, 1]
        >>> num_functions = 3
        >>> M = one_hot_encoding(funcs_idx, sizes, num_functions)
        >>> M
        3.14
    """
    assert len(funcs_idx) == len(sizes), "funcs_idx and sizes must match in length"

    total_vars = sum(sizes)
    M = np.zeros((total_vars, num_functions), dtype=int)

    # Row offsets per variable set
    offsets = np.cumsum([0] + sizes[:-1])

    for var_idx, funcs in enumerate(funcs_idx):
        # Treat [None] as "uses no functions"
        if funcs is None or funcs == [None]:
            continue
        start = offsets[var_idx]
        end = start + sizes[var_idx]
        M[start:end, funcs] = 1

    return M

def create_wrapped_function(original_func: ObjFunction, actions: VectorList, player_idx: int) -> WrappedFunction:
    """
        Create a wrapped objective function for a single player's optimization.

        Fix all players' action vectors except the one at ``player_idx``, and
        return a new function that accepts only this player's decision variables
        as input. The wrapped function automatically reconstructs the full list
        of action vectors and evaluates the original objective function.

        Parameters
        ----------
        original_func : ObjFunction
            The original objective function that accepts a list of action vectors.
        actions : VectorList
            List of 2D NumPy arrays with shape (n, 1), representing all players'
            current actions.
        player_idx : int
            Index of the player whose action vector should remain variable in
            the wrapped function.

        Returns
        -------
        WrappedFunction
            A function that takes the chosen player's action vector (as a list of floats),
            reshapes it into a 2D column vector with shape (n, 1), and evaluates
            ``original_func`` with the updated set of actions.

        Examples
        --------
        >>> import numpy as np
        >>> def original_func(var_list):
        ...     return sum(v.sum() for v in var_list)
        >>> actions = [np.array([[1.0], [2.0]]), np.array([[3.0]])]
        >>> wrapped = create_wrapped_function(original_func, actions, player_idx=0)
        >>> wrapped([10.0, 20.0])
        33.0
    """
    fixed_vars = actions[:player_idx] + actions[player_idx + 1:]

    def wrap_func(player_var_opt: List[float]) -> Vector:
        player_var_opt = np.array(player_var_opt).reshape(-1, 1)
        new_vars = fixed_vars[:player_idx] + [player_var_opt] + fixed_vars[player_idx:]
        return original_func(new_vars)

    return wrap_func

def create_wrapped_function_single(original_func: ObjFunction,actions: VectorList,player_idx: int) -> WrappedFunction:
    """
        Create a wrapped function returning only a single player's output.

        Fix all players' action vectors except the one at ``player_idx``, and
        return a new function that accepts only this player's decision variables
        as input. The wrapped function reconstructs the full list of action
        vectors, evaluates the original function, and returns only the output
        corresponding to the chosen player.

        Parameters
         -------
        original_func : ObjFunction
            The original objective function that accepts a list of action vectors.
        actions : VectorList
            List of 2D NumPy arrays with shape (n, 1), representing all players'
            current actions.
        player_idx : int
            Index of the player whose output should be returned by the wrapped function.

        Returns
        -------
        WrappedFunction
            A function that takes the chosen player's action vector, reshapes it into a 2D column vector with shape (n, 1), evaluates
            ``original_func`` with the updated set of actions, and returns only
            the output for that player.

        Examples
         -------
        >>> import numpy as np
        >>> def original_func(var_list):
        ...     return [v.sum() for v in var_list]
        >>> actions = [np.array([[1.0], [2.0]]), np.array([[3.0]])]
        >>> wrapped_single = create_wrapped_function_single(original_func, actions, player_idx=0)
        >>> wrapped_single([10.0, 20.0])
        30.0
    """
    fixed_vars = actions[:player_idx] + actions[player_idx + 1:]  # list of np vectors

    def wrap_func(player_var_opt):
        player_var_opt = np.array(player_var_opt).reshape(-1, 1)
        new_vars = fixed_vars[:player_idx] + [player_var_opt] + fixed_vars[player_idx:]
        return original_func(new_vars)[player_idx]
    
    return wrap_func

def objective_check(objective_functions: List[ObjFunction], actions: VectorList) -> Vector:
    """
        Evaluate and aggregate the objective function values for all players in a Generalized Nash Equilibrium (GNE) system.

        This function iterates through a list of objective functions—each corresponding to a player—and
        computes their objective values given the current set of player action vectors.
        It then concatenates all results into a single column vector for downstream optimization or equilibrium checks.

        Parameters
        ----------
        objective_functions : list[ObjFunction]
            A list of callable objective functions, one for each player.
        actions : VectorList
            A list or array of player action vectors representing the current state of all players' decisions.

        Returns
        -------
        numpy.ndarray
            A column vector (2D array of shape `(n, 1)`) containing all players’ evaluated objective function values,
            concatenated in order of the provided `objective_functions`.

        Examples
        --------
        >>> import numpy as np
        >>> def player1(vars): return np.array([vars[0]**2 + vars[1]])
        >>> def player2(vars): return np.array([vars[1]**2 + vars[0]])
        >>> objective_functions = [player1, player2]
        >>> actions = [np.array([1]), np.array([2])]
        >>> objective_check(objective_functions, actions)
        array([[3.],
               [5.]])
        """
    objective_values = []
    for objective in objective_functions:
        o = objective(actions)
        objective_values.append(o)
    return np.concatenate(objective_values).reshape(-1, 1)
    objective_values = []
    for objective in objective_functions:
        o = objective(actions)
        objective_values.append(o)
    return np.concatenate(objective_values).reshape(-1, 1)

def constraint_check(constraints: List[ConsFunction], actions: VectorList, epsilon: float = 1e-3) -> Tuple[VectorList, List[bool]]:
    """
    Evaluate constraint satisfaction for each player or system constraint in a GNE problem.
    Checks whether each constraint function is satisfied within a specified numerical tolerance.
    It returns both the evaluated constraint values and a Boolean list indicating whether each constraint is satisfied.

    Parameters
    ----------
    constraints : list[ConsFunction]
        A list of callable constraint functions. Each function should accept a list of player action vectors
        and return a NumPy array representing constraint residuals (e.g., values ≤ 0 indicate satisfaction).
    actions : VectorList
        A list or array of player action vectors representing the current decisions for all players.
    epsilon : float, optional
        Numerical tolerance used to determine constraint satisfaction.
        Defaults to `1e-3`. A constraint is considered satisfied if all its values are ≤ `epsilon`.

    Returns
    -------
    tuple
        A tuple containing:

        - **constraint_values** (`VectorList`): A list of NumPy arrays representing the evaluated constraint values for each function.
        - **constraint_satisfaction** (`list[bool]`): A list of Boolean values where `True` indicates that the corresponding constraint is satisfied.

    Examples
    --------
    >>> import numpy as np
    >>> def cons1(vars): return np.array([vars[0] + vars[1] - 3])   # x + y <= 3
    >>> def cons2(vars): return np.array([vars[0] - 2])             # x <= 2
    >>> constraints = [cons1, cons2]
    >>> actions = [np.array([1]), np.array([1.5])]
    >>> constraint_check(constraints, actions)
    ([array([-0.5]), array([-1.])], [True, True])

    >>> # Example with a violated constraint
    >>> actions = [np.array([2.5]), np.array([1.0])]
    >>> constraint_check(constraints, actions)
    ([array([0.5]), array([0.5])], [False, False])
    """
    constraint_values = []
    constraint_satisfaction = []
    for c_idx, constraint in enumerate(constraints):
        c = constraint(actions)
        if not np.all(np.ravel(c) <= epsilon):
            constraint_values.append(c)
            constraint_satisfaction.append(False)
        else:
            constraint_values.append(c)
            constraint_satisfaction.append(True)
    return constraint_values, constraint_satisfaction

def compare_solutions(
    computed_solution: List[float],
    paper_solution: List[float],
    action_sizes: List[int],
    objective_functions: List[ObjFunction],
    solution_name: List[str] = None
) -> Vector:
    if solution_name is None:
        solution_name=['Computed', 'Paper']
    computed_res = np.array(computed_solution).reshape(-1, 1)
    paper_res = np.array(paper_solution).reshape(-1, 1)

    computed_res_vectors = construct_vectors(computed_res, action_sizes)
    paper_res_vectors = construct_vectors(paper_res, action_sizes)

    computed_res_obj_func = objective_check(objective_functions, computed_res_vectors)
    paper_res_obj_func = objective_check(objective_functions, paper_res_vectors)

    difference = np.array(computed_res_obj_func) - np.array(paper_res_obj_func)
    print("Objective Functions")
    print_table(computed_res_obj_func, paper_res_obj_func, solution_name[0], solution_name[1])
    return difference.reshape(-1, 1)


"______________Recheck_____________"
def calculate_main_objective(self, actions):
    objective_values_matrix = [
        self.objective_functions[idx](actions) for idx in self.player_objective_function
    ]
    return np.array(deconstruct_vectors(objective_values_matrix))

def summary(result, time, wrapper, action_sizes, paper_res=None):
    print(result.x)
    print('Time: ', time)
    print('Iterations: ', result.nit)
    if paper_res:
        print('Paper Result: \n', paper_res)
    print('Solution: \n', result.x)
    print('Total Energy: ', wrapper(result.x))
    if paper_res:
        paper = np.array(paper_res).reshape(-1,1)
        computed_actions = np.array(result.x[:sum(action_sizes)]).reshape(-1,1)
        calculated_obj = calculate_main_objective(construct_vectors(computed_actions, action_sizes))
        paper_obj = calculate_main_objective(construct_vectors(paper, action_sizes))
        print('Difference: ', sum(deconstruct_vectors(calculated_obj)) - sum(deconstruct_vectors(paper_obj)))
