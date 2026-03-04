import numpy as np
from numpy.typing import NDArray
from typing import List
from .utils import *
from .types import *
from scipy.optimize import basinhopping
import timeit

class GNESolver:
    def __init__(self,
                 obj_funcs:                     List[ObjFunction],
                 constraints:                   List[ConsFunction],
                 player_obj_func:               List[int],
                 player_constraints:            List[PlayerConstraint],
                 player_vector_sizes:           List[int]
                 ):
        self.objective_functions =              obj_funcs                        # list of functions
        self.player_obj_func =                  one_hot_encoding(player_obj_func, player_vector_sizes, len(obj_funcs))
        self.constraints =                      constraints                      # list of functions
        self.player_objective_function =        np.array(player_obj_func, dtype=int)        # which obj function is used for each player
        self.player_constraints =               one_hot_encoding(player_constraints, player_vector_sizes, len(constraints))     # which constraints are used for each player
        self.action_sizes =                     player_vector_sizes    # size of each player's action vector
        self.N =                                len(player_obj_func)

