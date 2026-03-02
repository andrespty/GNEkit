import numpy as np
from griffe import Parameters
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt
from .misc import *
from .utils import *

class GNEP_Solver_Bounded:
    """
        <span style="background-color:#7B68EE; color:white; padding:2px 6px; border-radius:4px;">Class</span>

        Solve a bounded Generalized Nash Equilibrium Problem (GNEP).

        Parameters
        ----------
        enter parameters.

        Attributes
        ----------
        enter attributes.

        Examples
        ----------
        usages here
    """
    def __init__(self,
                 obj_funcs:                     List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
                 derivative_obj_funcs:          List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
                 constraints:                   List[Callable[[npt.NDArray[np.float64]], np.float64]],
                 derivative_constraints:        List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
                 player_obj_func:               List[int],
                 player_constraints:            List[List[int]],
                 bounds:                        List[Tuple[float, float]],
                 player_vector_sizes:           List[int] = None,
                 ):
        self.objective_functions =              obj_funcs                        # list of functions
        self.player_obj_func =                  one_hot_encoding(player_obj_func, player_vector_sizes, len(derivative_obj_funcs))
        self.objective_function_derivatives =   derivative_obj_funcs             # list of functions
        self.constraints =                      constraints                      # list of functions
        self.constraint_derivatives =           derivative_constraints           # list of functions
        self.player_objective_function =        np.array(player_obj_func)        # which obj function is used for each player
        self.player_constraints =               one_hot_encoding(player_constraints, player_vector_sizes, len(derivative_constraints))     # which constraints are used for each player
        self.action_sizes =                     player_vector_sizes    # size of each player's action vector
        self.N =                                len(player_obj_func)
        self.bounds =                           np.array(bounds)


    def wrapper(self, initial_actions: List[float]) -> float:
        """
        Input:
          initial_actions: python list of all players' actions
        Output:
          total energy: float value
        """
        actions_count = sum(self.action_sizes)
        actions = np.array(initial_actions[:actions_count]).reshape(-1,1)
        dual_actions = np.array(initial_actions[actions_count:]).reshape(-1,1)
        return self.energy_function(actions, dual_actions)

    def energy_function(self, actions: npt.NDArray[np.float64], dual_actions: npt.NDArray[np.float64]) -> float:
        """
        Input:
          actions: 2d np.array shape (sum(number of actions),1)   i.e. [[1], [2], [3], ..., [number of actions]]
          dual_actions: 2d np.array shape (N_d,1)                 i.e. [[1], [2], [3], ..., [N_d]]
        Output:
          total energy: float value
        """
        primal_players_energy = self.primal_energy_function(actions, dual_actions)
        dual_players_energy = self.calculate_energy_dual(actions, dual_actions)
        return np.sum(primal_players_energy) + np.sum(dual_players_energy)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def energy_handler(self, gradient: npt.NDArray[np.float64], actions: npt.NDArray[np.float64], isDual=False):
        """
        Input:
          gradient: 2d np.array shape (sum(number of actions),1)
        Output:
          total energy: float value
        """
        N_d = len(self.constraints)
        if isDual:
            bounds = self.bounds[-N_d:]
            actions = np.tile(actions, (N_d, 1))
            gradient = np.tile(gradient, (N_d, 1))
        else:
            if N_d == 0:
                bounds = self.bounds
            else:
                bounds = self.bounds[:-N_d]
        lb = bounds[:, 0].reshape(-1, 1)
        ub = bounds[:, 1].reshape(-1, 1)
        # print(bounds)
        # print(gradient)
        # Original
        engval = np.where(
            gradient <= 0,
            (ub - actions) * np.log(1 - gradient),
            (actions - lb) * np.log(1 + gradient)
        )
        # Experiments
        ## Better than original
        engval = np.where(
            gradient <= 0,
            np.abs(ub - actions) * (gradient**2)/(1-gradient),
            np.abs(actions - lb) * (gradient**2)/(1 + gradient)
        )
        return engval

    def primal_energy_function(self, actions: npt.NDArray[np.float64], dual_actions: npt.NDArray[np.float64]):
        """
        Input:
          actions: 2d np.array shape (sum(number of actions),1)
          dual_actions: 2d np.array shape (N_d,1)
        Output:
          (sum(number of actions),1) vector with the energy of each players' action
        """
        gradient = self.calculate_gradient(actions, dual_actions)
        return self.energy_handler(gradient, actions)

    # Gradient of primal player
    def calculate_gradient(self, actions: npt.NDArray[np.float64], dual_actions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Input:
          actions: 2d np.array shape (sum(number of actions),1)
          dual_actions: 2d np.array shape (N_d,1)
        Output:
          (sum(number of actions),1) vector
        """
        result = np.zeros_like(actions)
        # Track action indices per player
        action_splits = np.cumsum(np.insert(self.action_sizes, 0, 0) )  # Start and end indices for each player
        for obj_idx, mask in enumerate(self.player_obj_func.T):
            player_indices = np.where(mask)[0]
            o = self.objective_function_derivatives[obj_idx](construct_vectors(actions, self.action_sizes))
            if o.shape == result.shape:
                # Case 1: objective returns full (a,1) vector
                result += mask.reshape(-1,1) * o
            else:
                offset = 0
                mask = (self.player_objective_function == obj_idx)
                player_indices = np.where(mask)[0]
                for player in player_indices:
                    start_idx, end_idx = action_splits[player], action_splits[player + 1]
                    size = end_idx - start_idx
                    result[start_idx:end_idx] = o#[offset:offset + size]
                    offset += size
        # print(result)
        # Add constraints
        # constraints should be vectors with same size of result
        for c_idx, p_vector in enumerate(self.player_constraints.T):
            result += p_vector.reshape(-1,1) * dual_actions[c_idx] * self.constraint_derivatives[c_idx](actions)
        return result
    # Gradient of dual player
    def calculate_energy_dual(self, actions: npt.NDArray[np.float64], dual_actions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Input:
          actions: 2d np.array shape (sum(number of actions),1)
          dual_actions: 2d np.array shape (N_d,1)
        Output:
          (N_d,1) vector
        """
        if len(self.constraints) == 0:
            return 0
        grad_dual = []
        actions_vectors = construct_vectors(actions, self.action_sizes)
        for jdx, constraint in enumerate(self.constraints):
            g = -constraint(actions_vectors)
            g = self.energy_handler(g, dual_actions[jdx], isDual=True)
            grad_dual.append(g.flatten())
        g_dual = np.concatenate(grad_dual).reshape(-1, 1)
        return g_dual

    def solve_game(self, initial_guess: List[float],bounds: List[Tuple[float, float]], disp=True):
        """
        Input:
          initial_guess: python list of all players' actions
          bounds: python list of bounds for each player
        Output:
          result: scipy.optimize.optimize.OptimizeResult object
          time: float
        """
        minimizer_kwargs = dict(method="SLSQP", bounds=bounds)
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

    def calculate_main_objective(self, actions):
        objective_values_matrix = [
            self.objective_functions[idx](actions) for idx in self.player_objective_function
        ]
        return np.array(deconstruct_vectors(objective_values_matrix))

    def summary(self, paper_res=None):
        if self.result:
            print(self.result.x)
            print('Time: ', self.time)
            print('Iterations: ', self.result.nit)
            if paper_res:
                print('Paper Result: \n', paper_res)
            print('Solution: \n', self.result.x)
            print('Total Energy: ', self.wrapper(self.result.x))
            if paper_res:
                paper = np.array(paper_res).reshape(-1,1)
                computed_actions = np.array(self.result.x[:sum(self.action_sizes)]).reshape(-1,1)
                calculated_obj = self.calculate_main_objective(construct_vectors(computed_actions, self.action_sizes))
                paper_obj = self.calculate_main_objective(construct_vectors(paper, self.action_sizes))
                print('Difference: ', sum(deconstruct_vectors(calculated_obj)) - sum(deconstruct_vectors(paper_obj)))