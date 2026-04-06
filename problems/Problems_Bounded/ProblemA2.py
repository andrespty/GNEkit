import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


class A2:
    @staticmethod
    def paper_solution():
        value_1 = [0.29962894677774, 0.00997828224734, 0.00997828224734,
                   0.00997828224734, 0.59852469355630, 0.02187270661760,
                   0.00999093169361, 0.00999093169361, 0.00999093169361,
                   0.00999093169361]

        value_2 = [0.29962898846513, 0.00997828313762, 0.00997828313762,
                   0.00997828313762, 0.59745624992082, 0.02220301920403,
                   0.01013441012117, 0.01013441012117, 0.01013441012117,
                   0.01013441012117]

        return [value_1, value_2]

    @staticmethod
    def define_players():
        B = 1
        player_vector_sizes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        player_objective_functions = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        player_constraints = [[None], [0], [0], [0], [0, 1], [0, 1], [0], [0], [0], [0]]
        bounds = [(0.3, 0.5), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, 0.06),
                  (0.01, 0.05), (0, 10), (0, 10)]
        bounds_training = [(0.3, 0.5), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B),
                           (0.01, 0.06), (0.01, 0.05), (0, 10), (0, 10)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A2.obj_func_1, A2.obj_func_2]

    @staticmethod
    def objective_function_derivatives():
        return [A2.obj_func_der_1, A2.obj_func_der_2]

    @staticmethod
    def constraints():
        return [A2.g0, A2.g1]

    @staticmethod
    def constraint_derivatives():
        return [A2.g0_der, A2.g1_der]

    @staticmethod
    def obj_func_1(x):
        # x: numpy array (N, 1)
        B = 1
        S = sum(x)
        obj = (-x / S) * (1 - S / B)
        return obj

    @staticmethod
    def obj_func_2(x):
        # x: numpy array (N,1)
        # B: constant
        B = 1
        S = sum(x)
        obj = (-x / S) * (1 - S / B) ** 2
        return obj

    @staticmethod
    def obj_func_der_1(x):
        # x: numpy array (N,1)
        B = 1
        x = np.concatenate(x).reshape(-1, 1)
        S = sum(x)
        obj = (x - S) / (S ** 2) + 1
        return obj

    @staticmethod
    def obj_func_der_2(x):
        # x: numpy array (N,1)
        B = 1
        x = np.concatenate(x).reshape(-1, 1)
        S = sum(x) + 1e-3
        obj = (2 * B * (S ** 2) - (B ** 2) * S - S ** 3 - x * (S ** 2) + x * (B ** 2)) / (S ** 2)
        return obj

    @staticmethod
    def g0(x):
        # x: numpy array (N,1)
        B = 1
        return sum(x) - B

    @staticmethod
    def g1(x):
        # x: numpy array (N,1)
        B = 1
        return 0.99 - sum(x)

    @staticmethod
    def g0_der(x):
        return 1
    @staticmethod
    def g1_der(x):
        return -1
