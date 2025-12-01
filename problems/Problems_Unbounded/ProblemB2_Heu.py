import numpy as np
from numpy._typing import NDArray
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt
from gne_solver.misc import *
# Variable DEFS

# xA = np.array [xA1, xA2, xA3, xA4, yA
# xB = np.array[xB1, xB2, xB3, xB4, yB]
# x = [xA, xB] 2 arrays of 5 elements


class B2dev:
    e = (5, 2, 2, 2)
    k = (300, 300, 300, 300)
    Ci = (30000, 50000, 40000, 30000)
    c = (28, 26)
    cA = c[0]
    cB = c[1]
    gamma = 1 / 1.1

    @staticmethod
    def define_players():
        player_vector_sizes = [5, 5]
        player_objective_functions = [0, 1]

        player_constraints = [[0, 1, 2, 3, 4, 5, 7], [0, 1, 2, 3, 4, 6, 8]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [B2dev.obj_func_1, B2dev.obj_func_2]

    @staticmethod
    def objective_function_derivatives():
        return [B2dev.obj_func_1_der, B2dev.obj_func_2_der]
        # return [A3U.obj_func_der_1, A3U.obj_func_der_2, A3U.obj_func_der_3]

    def cost_a(self, xA):
        # xA is np.array shape (5,1)
        # xA1 = xA[0]
        xA1, xA2, xA3, xA4, yA = xA
        e1, e2, e3, e4 = self.e
        cost = (self.cA * np.sum(xA[:4]) + e1 * xA1
                + e2 * (xA4 + yA)
                + e3 * (xA3 - yA)
                + e4 * yA)
        return cost

    def cost_a_der(self, xA):
        xA1 = xA[0]
        xA2 = xA[1]
        xA3 = xA[2]
        xA4 = xA[3]
        yA = xA[4]
        e1 = self.e[0]
        e2 = self.e[1]
        e3 = self.e[2]
        e4 = self.e[3]
        return np.array([self.cA + e1, self.cA, self.cA + e3, self.cA + e2, e2 -e3 + e4])

    def cost_b(self, xB):
        xB1, xB2, xB3, xB4, yB = xB
        e1, e2, e3, e4 = self.e
        cost = (self.cB * np.sum(xB[:4]) + e1 * xB1
                + e2 * (xB1 + xB2 + xB3 - yB)
                + e3 * (xB3 - yB)
                + e4 * yB)
        return cost

    def cost_b_der(self, xB):
        xB1 = xB[0]
        xB2 = xB[1]
        xB3 = xB[2]
        xB4 = xB[3]
        yB = xB[4]
        e = self.e
        e1 = e[0]
        e2 = e[1]
        e3 = e[2]
        e4 = e[3]
        cA = self.c[0]
        cB = self.c[1]
        return np.array([cB +e1 + e2, cB + e2, cB + e2 + e3, cB, -e2 -e3 +e4])

    def price_vector(self, xA, xB):

        Ci = self.Ci
        C1 = Ci[0]
        C2 = Ci[1]
        C3 = Ci[2]
        C4 = Ci[3]

        xB1, xB2, xB3, xB4, yB = xB
        xA1, xA2, xA3, xA4, yA = xA
        return (self.Ci[:4] ** self.gamma) / ((xA[:4] + xB[:4]) ** self.gamma)

    def a_price_vector_der(self, xA, xB):
        Ci = self.Ci
        gamma = self.gamma
        C1 = Ci[0]
        C2 = Ci[1]
        C3 = Ci[2]
        C4 = Ci[3]
        der1 = -gamma * (C1**gamma) * (xA[0] + xB[0])**(-gamma -1)
        der2 = -gamma * (C2**gamma) * (xA[1] + xB[1])**(-gamma -1)
        der3 = -gamma * (C3**gamma) * (xA[2] + xB[2])**(-gamma -1)
        der4 = -gamma * (C4**gamma) * (xA[3] + xB[3])**(-gamma -1)
        price_gradient = np.array([der1, der2, der3, der4])
        return price_gradient

    def b_price_vector_der(self, x):
        xA, xB = x
        Ci = self.Ci
        gamma = self.gamma
        C1 = Ci[0]
        C2 = Ci[1]
        C3 = Ci[2]
        C4 = Ci[3]

        der1 = -gamma * (C1 ** gamma) * (xA[0] + xB[0]) ** (-gamma - 1)
        der2 = -gamma * (C2 ** gamma) * (xA[1] + xB[1]) ** (-gamma - 1)
        der3 = -gamma * (C3 ** gamma) * (xA[2] + xB[2]) ** (-gamma - 1)
        der4 = -gamma * (C4 ** gamma) * (xA[3] + xB[3]) ** (-gamma - 1)

        return np.array([der1, der2, der3, der4])

    def objective_function_player_1(self, x):
        p1 = x[0]
        p2 = x[1]
        Ci = self.Ci
        gamma = self.gamma
        p = B2dev.price_vector(x)
        revenue = float(np.dot(p, p1[:4]))
        return revenue - B2dev.cost_A(p1)

    @staticmethod
    def obj_func_1( xA, xB, Ci, gamma, cA, e):
        p = B2dev.price_vector(xA, xB, Ci, gamma)
        r = float(np.dot(p, xA[:4]))
        return r - xA.cost_A(xA, e, cA)

    def obj_func_1_der(self, x):
        xA, xB =x[0], x[1]
        xA1 = xA[0]
        xA2 = xA[1]
        xA3 = xA[2]
        xA4 = xA[3]
        yA = xA[4]

        e1 = B2dev.e[0]
        e2 = B2dev.e[1]
        e3 = B2dev.e[2]
        e4 = B2dev.e[3]

        C1 = self.Ci[0]
        C2 = self.Ci[1]
        C3 = self.Ci[2]
        C4 = self.Ci[3]

        p = B2dev.price_vector(xA, xB, Ci, gamma)
        dp_A = B2dev.a_price_vector_der(xA, xB, Ci, gamma)
        grad_cost_A =B2dev.cost_a_der(xA, e, cA)
        grad = np.zeros(5, dtype = float)
        for i in range(4):
            grad[i] = p[i] + xA[i] * dp_A[i] - grad_cost_A[i]
        grad[4] = -grad_cost_A[4]

        return grad

    @staticmethod
    def obj_func_2(xA, xB, Ci, gamma, cB, e):
        p = B2dev.price_vector(xB, xA, Ci, gamma)
        r = float(np.dot(p, xB[:4]))
        return r - xB.cost_B(xB, e, cB)


    def obj_func_2_der(self, x):
        xA, xB = x[0], x[1]
        xB1 = xB[0]
        xB2 = xB[1]
        xB3 = xB[2]
        xB4 = xB[3]
        yB = xB[4]
        e= self.e
        e1 = e[0]
        e2 = e[1]
        e3 = e[2]
        e4 = e[3]

        Ci = self.Ci
        C1 = Ci[0]
        C2 = Ci[1]
        C3 = Ci[2]
        C4 = Ci[3]

        gamma = self.gamma

        p = B2dev.price_vector(xA, xB, Ci, gamma)
        dp_B = B2dev.b_price_vector_der(xA, xB, Ci, gamma)
        grad_cost_B = B2dev.cost_b_der(xB, e, cB)

        grad = np.zeros(5, dtype=float)

        for i in range(4):
            grad[i] = p[i] + xB[i] * dp_B[i] - grad_cost_B[i]

        # yB component (revenue does not depend on yB)
        grad[4] = -grad_cost_B[4]

        return grad

    @staticmethod
    def constraints():
        return [B2dev.g0, B2dev.g1, B2dev.g2, B2dev.g3, B2dev.g4, B2dev.g5, B2dev.g6, B2dev.g7, B2dev.g8]

    @staticmethod
    def constraint_der():
        return [B2dev.g0_der, B2dev.g1_der, B2dev.g2_der, B2dev.g3_der, B2dev.g4_der,B2dev.g5_der(), B2dev.g6_der(), B2dev.g7_der(), B2dev.g8_der()]



    @staticmethod
    def g0_der():
        return np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    @staticmethod
    def g1_der():
        return np.array([0, 0, 0, 1, 1, -1, -1, -1, 0, 1])
    @staticmethod
    def g2_der():
        return np.array([0, 0, 0, -1, -1, 1, 1, 1, 0, -1])
    @staticmethod
    def g3_der():
        return np.array([0, 0, 1, 0, -1, 0, 0, 1, 0, -1])
    @staticmethod
    def g4_der():
        return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    @staticmethod
    def g5_der():
        return np.array([-1, -1, -1, -1, -1,  0, 0, 0, 0, 0])
    @staticmethod
    def g6_der():
        return np.array([0, 0, 0, 0, 0, -1, -1, -1, -1, -1])
    @staticmethod
    def g7_der():
        return np.array([0, 0, -1, 0, 1, 0, 0, 0, 0, 0])
    @staticmethod
    def g8_der():
        return np.array([0, 0, 0, 0, 0, 0, 0, -1, 0, 1])



    @staticmethod
    def g0(x: tuple[NDArray[np.float64], NDArray[np.float64]], k: tuple[float, float, float, float]) -> float:
        """xA1 + xB1 ≤ k1  ->  g≤0"""
        xA, xB = x
        k1, k2, k3, k4 = k
        return (xA[0] + xB[0]) - k1  # ≤ 0

    @staticmethod
    def g1(x: tuple[NDArray[np.float64], NDArray[np.float64]], k: tuple[float, float, float, float]) -> float:
        xA, xB = x
        k2, k3, k4 = k
        k2 = k[1]
        return float((xA[3] + xA[4] - xB[0] - xB[1] - xB[2] + xB[4] - k2))

    @staticmethod
    def g2(x: tuple[NDArray[np.float64], NDArray[np.float64]], k: tuple[float, float, float, float]) -> float:
        xA, xB = x
        k2 = k[1]
        return float((-xA[3] - xA[4] +xB[0] + xB[1] + xB[2] -xB[4] -k2))

    @staticmethod
    def g3(x: tuple[NDArray[np.float64], NDArray[np.float64]],k: tuple[float, float, float, float]) -> float:
        xA, xB = x
        k1, k2, k3, k4 = k
        k3 = k[2]
        return float((xA[2] + xB[2] - xA[4] - xB[4]) - k3)

    @staticmethod
    def g4(x: tuple[NDArray[np.float64], NDArray[np.float64]], k: tuple[float, float, float, float]) -> float:
        xA, xB = x
        k1, k2, k3, k4 = k
        k4 = k[3]
        return float((xA[4] + xB[4]) - k4)

    @staticmethod
    def g5(x) -> NDArray[np.float64]:
        xA, xB = x
        xA = x[0]
        return .1 -xA[:4]

    @staticmethod
    def g6(x) -> NDArray[np.float64]:
        xA, xB = x
        xB = x[1]
        return .1 -xB[:4]

    @staticmethod
    def g7(x) -> float:
        xA, xB = x
        xA = x[0]
        return float(xA[4] - xA[2])

    @staticmethod
    def g8(x) -> float:
        xA, xB = x
        xB = x[1]
        return float(xB[4] - xB[2])



   
