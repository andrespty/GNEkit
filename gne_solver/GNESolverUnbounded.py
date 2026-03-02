import numpy as np
from numpy.typing import NDArray
from typing import List
from .utils import *
from .types import *
from scipy.optimize import basinhopping
import timeit

class GNEP_Solver_Unbounded:
    """
       <span style="background-color:#7B68EE; color:white; padding:2px 6px; border-radius:4px;">Class</span>

       Solve an unbounded Generalized Nash Equilibrium Problem (GNEP).

       Implements an energy-based method to compute Nash equilibria
       in multi-player optimization problems with objectives and constraints.
       Each player has an objective function and a set of constraints, and
       gradients are used to build energy functions for both primal and dual
       players.

       Parameters
       ----------
       obj_funcs : list of ObjFunction
           List of objective functions, one for each player or shared across
           multiple players. Each function accepts a list of player action
           vectors and returns a scalar.
       derivative_obj_funcs : list of ObjFunctionGrad
           List of gradients of the objective functions. Each gradient function
           returns either the full gradient (shape = (sum(action_sizes), 1))
           or a partial gradient for its own variables (shape = (n_i, 1)).
       constraints : list of ConsFunction
           List of constraint functions. Each function takes all player
           action vectors and returns a constraint value or vector.
       derivative_constraints : list of ConsFunctionGrad
           List of gradients of the constraints with respect to the actions.
       player_obj_func : list of int
           Indices specifying which objective function corresponds to each player.
       player_constraints : list of PlayerConstraint
           Indices specifying which constraints apply to each player.
       player_vector_sizes : list of int
           Sizes of each player's action vector. The sum determines the dimension
           of the action space.

       Attributes
       ----------
       objective_functions : list of ObjFunction
           Original objective functions.
       objective_function_derivatives : list of ObjFunctionGrad
           Gradients of the objective functions.
       constraints : list of ConsFunction
           Constraint functions.
       constraint_derivatives : list of ConsFunctionGrad
           Gradients of the constraints.
       player_objective_function : vector
           Encodes which objective function is used for each player.
       player_constraints : vector
           Encodes which constraints are used for each player.
       action_sizes : list of int
           Sizes of each player's action vector.
       N : int
           Number of players.
       result : scipy.optimize.OptimizeResult
           Result object returned by the solver after calling `solve_game`.
       time : float
           Elapsed computation time (seconds).

       Examples
       -------
       >>> import numpy as np
       >>> # Define two simple player objectives
       >>> def f1(actions): return actions[0]**2
       >>> def f2(actions): return actions[1]**2
       >>> def df1(actions): return np.array([2*actions[0]]).reshape(-1,1)
       >>> def df2(actions): return np.array([2*actions[1]]).reshape(-1,1)
       >>> # Initialize solver
       >>> solver = GNEP_Solver_Unbounded(
       ...     obj_funcs=[f1, f2],
       ...     derivative_obj_funcs=[df1, df2],
       ...     constraints=[],
       ...     derivative_constraints=[],
       ...     player_obj_func=[0,1],
       ...     player_constraints=[],
       ...     player_vector_sizes=[1,1]
       ... )
       >>> # Solve game starting from initial guess
       >>> initial_guess = [5.0, -3.0]
       >>> result, elapsed_time = solver.solve_game(initial_guess, disp=False)
       >>> print(result.x)
       >>> print(elapsed_time)
    """

    def __init__(self,
                 obj_funcs:                     List[ObjFunction],
                 derivative_obj_funcs:          List[ObjFunctionGrad],
                 constraints:                   List[ConsFunction],
                 derivative_constraints:        List[ConsFunctionGrad],
                 player_obj_func:               List[int],
                 player_constraints:            List[PlayerConstraint],
                 player_vector_sizes:           List[int]
                 ):
        self.objective_functions =              obj_funcs                        # list of functions
        self.player_obj_func =                  one_hot_encoding(player_obj_func, player_vector_sizes, len(derivative_obj_funcs))
        self.objective_function_derivatives =   derivative_obj_funcs             # list of functions
        self.constraints =                      constraints                      # list of functions
        self.constraint_derivatives =           derivative_constraints           # list of functions
        self.player_objective_function =        np.array(player_obj_func, dtype=int)        # which obj function is used for each player
        self.player_constraints =               one_hot_encoding(player_constraints, player_vector_sizes, len(derivative_constraints))     # which constraints are used for each player
        self.action_sizes =                     player_vector_sizes    # size of each player's action vector
        self.N =                                len(player_obj_func)

    def wrapper(self, initial_actions: List[float]) -> float:
        """
        Compute the total energy from a flat list of player and dual actions.

        Parameters
        ----------
        initial_actions : list of float
            Flattened list containing all players' actions followed by dual variables.

        Returns
        -------
        float
            Total energy (sum of primal and dual contributions).
        """
        actions_count = sum(self.action_sizes)
        actions = np.array(initial_actions[:actions_count], dtype=np.float64).reshape(-1,1)
        dual_actions = np.array(initial_actions[actions_count:], dtype=np.float64).reshape(-1,1)
        return self.energy_function(actions, dual_actions)

    def energy_function(self, actions: Vector, dual_actions: Vector) -> float:
        """
        Compute the total energy of the system.

        Parameters
        ----------
        actions : vector of shape (sum(action_sizes), 1)
            Player action vectors stacked vertically.
        dual_actions : vector of shape (N_d, 1)
            Dual variables corresponding to constraints.

        Returns
        -------
        float
            Total energy of the system.
        """
        primal_players_energy = self.primal_energy_function(actions, dual_actions)
        dual_players_energy = self.dual_energy_function(actions, dual_actions)
        return float(np.sum(primal_players_energy, axis=0) + np.sum(dual_players_energy, axis=0))

    @staticmethod
    def energy_handler(gradient: Vector, actions: Vector, isDual=False) -> Vector:
        """
        Convert gradients into energy contributions.

        Parameters
        ----------
        gradient : vector of shape (n, 1)
            Gradient vector to transform into energy.
        actions : vector of shape (n, 1)
            Corresponding action vector.
        isDual : bool, optional
            Whether the energy comes from dual players. Default is False.

        Returns
        -------
        vector of shape (n, 1)
            Energy contributions per element.
        """
        if isDual:
            # dual_eng = np.square((actions ** 2 / (1 + actions ** 2)) * (gradient ** 2 / (1 + gradient ** 2)) + np.exp(-actions ** 2) * (np.maximum(0, -gradient) ** 2 / (1 + np.maximum(0, -gradient) ** 2)))
            # dual_eng = np.square( np.sqrt(actions**2 + gradient**2) - (actions + gradient) )
            eps = 1e-5
            fb = np.sqrt(actions ** 2 + gradient ** 2 + eps**2) - (actions + gradient)
            return fb**2
        else:
            mu = 1e-3
            return (mu**2) * (np.sqrt(1 + (gradient **2)/(mu ** 2) ) - 1)

    def primal_energy_function(self, actions: Vector, dual_actions: Vector) -> Vector:
        """
        Compute the energy contribution of primal players.

        Parameters
        ----------
        actions : vector of shape (sum(action_sizes), 1)
            Player action vectors.
        dual_actions : vector of shape (N_d, 1)
            Dual variables.

        Returns
        -------
        vector of shape (sum(action_sizes), 1)
            Energy contribution per primal player action.
        """
        gradient = self.calculate_gradient(actions, dual_actions)
        return self.energy_handler(gradient, actions)

    def dual_energy_function(self, actions: Vector, dual_actions: Vector) -> Vector:
        """
        Compute the energy contribution of dual players.

        Parameters
        ----------
        actions : vector of shape (sum(action_sizes), 1)
            Player action vectors.
        dual_actions : vector of shape (N_d, 1)
            Dual variables.

        Returns
        -------
        vector of shape (N_d, 1)
            Energy contribution per dual variable.
        """
        eng_dual = self.calculate_gradient_dual(actions, dual_actions)
        return eng_dual

    # Gradient of primal player
    def calculate_gradient(self, actions: Vector, dual_actions: Vector) -> Vector:
        """
        Compute the gradient of the primal energy.

        Parameters
        ----------
        actions : vector of shape (sum(action_sizes), 1)
            Player action vectors.
        dual_actions : vector of shape (N_d, 1)
            Dual variables.

        Returns
        -------
        vector of shape (sum(action_sizes), 1)
            Gradient of primal energy with respect to actions.
        """
        result = np.zeros_like(actions) # shape (-1,1)
        # Track action indices per player
        action_splits = np.cumsum(np.insert(self.action_sizes, 0, 0) )  # Start and end indices for each player
        vector_actions = construct_vectors(actions, self.action_sizes)
        for obj_idx, mask in enumerate(self.player_obj_func.T):
            mask: NDArray[bool]
            o = self.objective_function_derivatives[obj_idx](vector_actions)
            # Case 1: objective gradient returns full (sum(number of actions),1) vector
            if o.shape == result.shape:
                result += mask.reshape(-1,1) * o.reshape(-1,1)

            # Case 2: objective returns gradient only for its actions (actions,1) vector
            else:
                offset = 0
                # mask: NDArray[np.bool_] = (self.player_objective_function == obj_idx)
                mask = np.equal(self.player_objective_function, obj_idx)
                player_indices = np.where(mask)[0]
                for player in player_indices:
                    start_idx, end_idx = action_splits[player], action_splits[player + 1]
                    size = end_idx - start_idx
                    result[start_idx:end_idx] = o#[offset:offset + size]
                    offset += size

        # Adding constraints, should be vectors with same size of result
        for c_idx, p_vector in enumerate(self.player_constraints.T):
            p_vector: NDArray[int]
            result += p_vector.reshape(-1,1) * dual_actions[c_idx] * self.constraint_derivatives[c_idx](vector_actions)
        return result

    # Gradient of dual player
    def calculate_gradient_dual(self, actions: Vector, dual_actions: Vector) -> Vector:
        """
        Compute the gradient of the dual energy.

        Parameters
        ----------
        actions : vector of shape (sum(action_sizes), 1)
            Player action vectors.
        dual_actions : vector of shape (N_d, 1)
            Dual variables.

        Returns
        -------
        vector of shape (N_d, 1)
            Gradient of dual energy with respect to constraints.
        """
        grad_dual = []
        actions_vectors = construct_vectors(actions, self.action_sizes)
        for jdx, constraint in enumerate(self.constraints):
            g = -constraint(actions_vectors)
            g = self.energy_handler(g, dual_actions[jdx], isDual=True).astype(np.float64)
            grad_dual.append(g.flatten())
        g_dual = np.concatenate(grad_dual).reshape(-1, 1)
        return g_dual

    def solve_game(self, initial_guess: List[float], disp: bool=True):
        """
        Solve the GNEP optimization problem.

        Parameters
        ----------
        initial_guess : list of float
            Initial guess for all player actions and dual variables.
        disp : bool, optional
            Whether to display solver progress. Default is True.

        Returns
        -------
        result : OptimizeResult
            Result object from the optimization routine.
        time : float
            Time taken to compute the solution.

        Examples
        --------
        >>> solver = GNEP_Solver_Unbounded(
        ...     obj_funcs,
        ...     grad_obj_funcs,
        ...     constraints,
        ...     grad_constraints,
        ...     player_obj_func,
        ...     player_constraints,
        ...     player_vector_sizes
        ... )
        >>> # Initial guess for [x1, x2, λ]
        >>> initial_guess = [0.5, 0.5, 0.1]
        >>>
        >>> result, elapsed_time = solver.solve_game(initial_guess)
        >>> print("Optimal actions and duals:", result.x)
        >>> print("Computation time:", elapsed_time)
       """
        minimizer_kwargs = dict(method="SLSQP")
        start = timeit.default_timer()
        result = basinhopping(
            self.wrapper,
            initial_guess,
            stepsize=0.01,
            niter=1000,
            minimizer_kwargs=minimizer_kwargs,
            interval=1,
            niter_success=100,
            disp=disp,
            # callback=stopping_criterion
        )
        stop = timeit.default_timer()
        elapsed_time = stop - start
        self.result = result
        self.time = elapsed_time
        return result, elapsed_time
