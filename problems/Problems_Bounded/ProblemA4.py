import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


class A4:
    @staticmethod
    def paper_solution():
        value_1 = [0.99982626069210, 0.99996267821636, 0.99987070414176,
                   0.99985869062731, 0.99983447394048, 0.99991824127925,
                   0.99991381820076]
        value_2 = [0.99982626069210, 0.99996267821636, 0.99987070414176,
                   0.99985869062731, 0.99983447394048, 0.99991824127925,
                   0.99991381820076]
        value_3 = [0.99991935573409, 0.99998267343165, 0.99993998316707,
                   0.99993440803353, 0.99992316858615, 0.99987984742324,
                   0.99987334759435]
        return [value_1, value_2, value_3]

    @staticmethod
    def define_players():
        player_vector_sizes = [3, 2, 2]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0, 1], [2], [3]]
        bounds = [(1, 10), (1, 10), (1, 10), (1, 10), (1, 10), (1, 10), (1, 10), (0, 100), (0, 100), (0, 100),
                           (0, 100)]
        bounds_training = [(1, 10), (1, 10), (1, 10), (1, 10), (1, 10), (1, 10), (1, 10), (0, 100), (0, 100), (0, 100),
                           (0, 100)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A4.obj_func_1, A4.obj_func_2, A4.obj_func_3]

    @staticmethod
    def objective_function_derivatives():
        return [A4.obj_func_der_1, A4.obj_func_der_2, A4.obj_func_der_3]

    @staticmethod
    def constraints():
        return [A4.g0, A4.g1, A4.g2, A4.g3]

    @staticmethod
    def constraint_derivatives():
        return [A4.g0_der, A4.g1_der, A4.g2_der, A4.g3_der]

    # Define Functions below

    @staticmethod
    def obj_func(
            x: npt.NDArray[np.float64],
            x_ni: npt.NDArray[np.float64],
            A: npt.NDArray[np.float64],
            B: npt.NDArray[np.float64],
            b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # x: actions vector (d_i, 1)
        # A: constant matrix (m, n)
        # B: constant matrix (m, sum(d_{-i}))
        # b: constant vector (m, 1)
        obj = 0.5 * x.T @ A @ x + x.T @ (B @ x_ni + b)
        return obj

    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        A1 = np.array([
            [20 + (x2[0, 0] ** 2), 5, 3],
            [5, 5 + (x2[1, 0] ** 2), -5],
            [3, -5, 15]
        ])

        B1 = np.array([[-6, 10, 11, 20], [10, -4, -17, 9], [15, 8, -22, 21]])
        b1 = np.array([[1], [-1], [1]])
        x1 = x[0]
        x_n1 = np.vstack((x[1], x[2]))
        return A4.obj_func(x1, x_n1, A1, B1, b1)

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        A2 = np.array([
            [11 + (x3[0, 0] ** 2), -1],
            [-1, 9]
        ])

        A2 = np.array([[11, -1], [-1, 9]])
        B2 = np.array([[20, 1, -3, 12, 1], [10, -4, 8, 16, 21]])
        b2 = np.array([[1], [0]])
        x_n2 = np.vstack((x1, x3))
        return A4.obj_func(x2, x_n2, A2, B2, b2)

    @staticmethod
    def obj_func_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        A3 = np.array([
            [48, 39],
            [39, 53 + (x1[0, 0] ** 2)]
        ])

        B3 = np.array([[10, -2, 22, 12, 16], [9, 19, 21, -4, 20]])
        b3 = np.array([[-1], [2]])
        x_n3 = np.vstack((x1, x2))
        return A4.obj_func(x3, x_n3, A3, B3, b3)

    @staticmethod
    def obj_func_der(
            x: npt.NDArray[np.float64],
            x_ni: npt.NDArray[np.float64],
            A: npt.NDArray[np.float64],
            B: npt.NDArray[np.float64],
            b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # x: actions vector (d_i, 1)
        # A: constant matrix (m, n)
        # B: constant matrix (m, sum(d_{-i}))
        # b: constant vector (m, 1)
        obj = A @ x + B @ x_ni + b
        return obj

    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        A1 = np.array([
            [20 + (x2[0, 0] ** 2), 5, 3],
            [5, 5 + (x2[1, 0] ** 2), -5],
            [3, -5, 15]
        ])
        B1 = np.array([[-6, 10, 11, 20], [10, -4, -17, 9], [15, 8, -22, 21]])
        b1 = np.array([[1], [-1], [1]])
        x_n1 = np.vstack((x2, x3))
        return A4.obj_func_der(x1, x_n1, A1, B1, b1)

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        A2 = np.array([
            [11 + (x3[0, 0] ** 2), -1],
            [-1, 9]
        ])
        B2 = np.array([[20, 1, -3, 12, 1], [10, -4, 8, 16, 21]])
        b2 = np.array([[1], [0]])
        x_n2 = np.vstack((x1, x3))
        return A4.obj_func_der(x2, x_n2, A2, B2, b2)

    @staticmethod
    def obj_func_der_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        A3 = np.array([
            [48, 39],
            [39, 53 + (x1[0, 0] ** 2)]
        ])
        B3 = np.array([[10, -2, 22, 12, 16], [9, 19, 21, -4, 20]])
        b3 = np.array([[-1], [2]])
        x_n3 = np.vstack((x1, x2))
        return A4.obj_func_der(x3, x_n3, A3, B3, b3)

    @staticmethod
    def g0(x):
        x1, x2, x3 = x
        return np.sum(x1) - 20

    @staticmethod
    def g1(x):
        x1, x2, x3 = x
        return (x1[0] + x1[1] - x1[2] - x2[0] + x3[1] - 5)[0]

    @staticmethod
    def g2(x):
        x1, x2, x3 = x
        return (x2[0] + x2[1] - x1[1] - x1[2] + x3[0] - 7)[0]

    @staticmethod
    def g3(x):
        x1, x2, x3 = x
        return (x3[1] - x1[0] - x1[2] + x2[0] - 4)[0]

    # partial g0 / partial x1
    @staticmethod
    def g0_der(x1):
        return np.array([[1, 1, 1, 1, 1, 1, 1]]).reshape(-1, 1)

    # partial g1 / partial x1
    @staticmethod
    def g1_der(x1):
        return np.array([[1, 1, -1, -1, 0, 0, 1]]).reshape(-1, 1)

    # partial g2 / partial x2
    @staticmethod
    def g2_der(x1):
        return np.array([[0, -1, -1, 1, 1, 1, 0]]).reshape(-1, 1)

    # partial g3 / partial x3
    @staticmethod
    def g3_der(x1):
        return np.array([[-1, 0, -1, 1, 0, 
                          0, 1]]).reshape(-1, 1)