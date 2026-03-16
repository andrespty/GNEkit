import numpy as np
from griffe import Parameters
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

from gnep_solver import Player
from .GNEPlayer import GNEPlayer
from .misc import *
from .utils import *
from .GNEPlayer import players_to_lists
from .validation import validate_player_list

class GNESolver:
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
                 obj_funcs_der:                 List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
                 constraints:                   List[Callable[[npt.NDArray[np.float64]], np.float64]],
                 constraints_der:               List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
                 player_list:                   List[GNEPlayer]
                 ):

        validate_player_list(player_list)
        player_info = players_to_lists(player_list)
        self.action_sizes = player_info["sizes"]
        self.bounds_primal = player_info["bounds"]
        self.bounds_dual = [(0,100) for _ in constraints_der]
        self.bounds_all = self.bounds_primal + self.bounds_dual

        # Functions
        self.objective_functions =              obj_funcs
        self.constraints =                      constraints
        self.obj_derivatives =                  obj_funcs_der
        self.const_derivatives =                constraints_der

        # Indices
        self.player_obj_idx = player_info["objectives"]
        self.player_const_idx = player_info["constraints"]
        self.players = player_list

        # Pre calculations
        action_splits = np.cumsum(np.array([0] + self.action_sizes))
        self.action_splits = [int(x) for x in action_splits]
        self.total_actions = sum(self.action_sizes)
        self.player_const_idx_matrix =           one_hot_encoding(self.player_const_idx, self.action_sizes, len(self.const_derivatives))     # which constraints are used for each player
        self.player_obj_idx_matrix =             one_hot_encoding(self.player_obj_idx, self.action_sizes, len(self.obj_derivatives))

        lb_list = []
        ub_list = []
        lb_list_d = []
        ub_list_d = []

        for i, (lb, ub) in enumerate(self.bounds_primal):
            # Repeat the bounds for each decision variable the player owns
            lb_list.append(lb)
            ub_list.append(ub)

        for i, (lb, ub) in enumerate(self.bounds_dual):
            # Repeat the bounds for each decision variable the player owns
            lb_list_d.append(lb)
            ub_list_d.append(ub)

        self.lb_vector = np.concatenate(lb_list).reshape(-1, 1)
        self.ub_vector = np.concatenate(ub_list).reshape(-1, 1)
        self.lb_vector_d = np.array(lb_list_d).reshape(-1,1)
        self.ub_vector_d = np.array(ub_list_d).reshape(-1,1)


    def min_func(self, x: List[float]) -> float:
        """
        Input:
          x: python list of all players' actions
        Output:
          total energy: float value
        """
        actions = np.array(x[:self.total_actions]).reshape(-1,1)
        dual_actions = np.array(x[self.total_actions:]).reshape(-1,1)

        primal_players_energy = self.primal_energy_function(actions, dual_actions)
        dual_players_energy = self.calculate_energy_dual(actions, dual_actions)

        return np.sum(primal_players_energy) + np.sum(dual_players_energy)

    def energy_handler(self, gradient: npt.NDArray[np.float64], actions: npt.NDArray[np.float64], lb, ub):
        """
        Input:
          gradient: 2d np.array shape (sum(number of actions),1)
        Output:
          total energy: float value
        """
        # engval = np.where(
        #     gradient <= 0,
        #     (ub - actions) * np.log(1 - gradient),
        #     (actions - lb) * np.log(1 + gradient)
        # )
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
        return self.energy_handler(gradient, actions, self.lb_vector, self.ub_vector)

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
        for obj_idx, mask in enumerate(self.player_obj_idx_matrix.T):
            player_indices = np.where(mask)[0]
            o = self.obj_derivatives[obj_idx](construct_vectors(actions, self.action_sizes))
            if o.shape == result.shape:
                # Case 1: objective returns full (a,1) vector
                result += mask.reshape(-1,1) * o
            else:
                offset = 0
                mask = (self.player_obj_idx == obj_idx)
                player_indices = np.where(mask)[0]
                for player in player_indices:
                    start_idx, end_idx = action_splits[player], action_splits[player + 1]
                    size = end_idx - start_idx
                    result[start_idx:end_idx] = o#[offset:offset + size]
                    offset += size
        # Add constraints
        # constraints should be vectors with same size of result
        for c_idx, p_vector in enumerate(self.player_const_idx_matrix.T):
            result += p_vector.reshape(-1,1) * dual_actions[c_idx] * self.const_derivatives[c_idx](actions)
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
        g_values = np.array([-c(actions_vectors) for c in self.constraints]).reshape(-1,1)
        lambdas = dual_actions.ravel().reshape(-1, 1)
        g = self.energy_handler(g_values, lambdas, self.lb_vector_d, self.ub_vector_d)
        return g.reshape(-1, 1)

    def solve_game(self, initial_guess: List[float], disp=True):
        """
        Input:
          initial_guess: python list of all players' actions
          bounds: python list of bounds for each player
        Output:
          result: scipy.optimize.optimize.OptimizeResult object
          time: float
        """
        print("BOUNDS ALL: ",self.bounds_all)
        minimizer_kwargs = dict(method="SLSQP", bounds=self.bounds_all)
        start = timeit.default_timer()
        result = basinhopping(
            self.min_func,
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