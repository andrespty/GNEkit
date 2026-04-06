import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

class A10a:
    F= 2
    C= 5
    P= 3
    N = F + C + 1

    @staticmethod
    def define_players():
        player_vector_sizes = [A10a.P for _ in range(A10a.N)]
        player_objective_functions = [0,0,1,2,3,4,5,6]  # change to all 0s
        player_constraints = [[0], [0], [1],[1],[1],[1],[1], [2,3]]
        bounds = [(0, 100) for _ in range(A10a.N * A10a.P)] + [(0, 100) for _ in range(len(A10a.constraints()))]
        bounds_training = [(0, 100) for _ in range(A10a.N * A10a.P)] + [(0, 100) for _ in range(len(A10a.constraints()))]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A10a.obj_func_firms,
                A10a.obj_func_consumers_1,
                A10a.obj_func_consumers_2,
                A10a.obj_func_consumers_3,
                A10a.obj_func_consumers_4,
                A10a.obj_func_consumers_5,
                A10a.obj_func_market
        ]

    @staticmethod
    def objective_function_derivatives():
        return [A10a.obj_func_firms_der,
                A10a.obj_func_consumers_1_der,
                A10a.obj_func_consumers_2_der,
                A10a.obj_func_consumers_3_der,
                A10a.obj_func_consumers_4_der,
                A10a.obj_func_consumers_5_der,
                A10a.obj_func_market_der
        ]

    @staticmethod
    def constraints():
        return [A10a.g0, A10a.g1, A10a.g2, A10a.g3]

    @staticmethod
    def constraint_derivatives():
        return [A10a.g0_der, A10a.g1_der, A10a.g2_der, A10a.g3_der]

    @staticmethod
    def obj_func_firms(x):
        p = x[-1].reshape(-1,1)
        y = np.hstack(x[:A10a.F]).reshape(A10a.P, -1)
        return p.T @ y # returns (1,F) vector

    @staticmethod
    def obj_func_consumers_1(x):
        x = np.hstack(x[A10a.F: A10a.F+A10a.C][0]).reshape(A10a.P, -1)
        Q_i = A10a.get_constants(i=1).get('Q_i')
        b_i = A10a.get_constants(i=1).get('b_i')
        return A10a.utility_function(x, Q_i, b_i)

    @staticmethod
    def obj_func_consumers_2(x):
        x = np.hstack(x[A10a.F: A10a.F + A10a.C][1]).reshape(A10a.P, -1)
        Q_i = A10a.get_constants(i=2).get('Q_i')
        b_i = A10a.get_constants(i=2).get('b_i')
        return A10a.utility_function(x, Q_i, b_i)

    @staticmethod
    def obj_func_consumers_3(x):
        x = np.hstack(x[A10a.F: A10a.F + A10a.C][2]).reshape(A10a.P, -1)
        Q_i = A10a.get_constants(i=3).get('Q_i')
        b_i = A10a.get_constants(i=3).get('b_i')
        return A10a.utility_function(x, Q_i, b_i)

    @staticmethod
    def obj_func_consumers_4(x):
        x = np.hstack(x[A10a.F: A10a.F + A10a.C][3]).reshape(A10a.P, -1)
        Q_i = A10a.get_constants(i=4).get('Q_i')
        b_i = A10a.get_constants(i=4).get('b_i')
        return A10a.utility_function(x, Q_i, b_i)

    @staticmethod
    def obj_func_consumers_5(x):
        x = np.hstack(x[A10a.F: A10a.F + A10a.C][4]).reshape(A10a.P, -1)
        Q_i = A10a.get_constants(i=5).get('Q_i')
        b_i = A10a.get_constants(i=5).get('b_i')
        return A10a.utility_function(x, Q_i, b_i)

    @staticmethod
    def obj_func_market(x):
        p = x[-1].reshape(-1,1)
        y = np.hstack(x[:A10a.F]).reshape(A10a.P, -1)
        x = np.hstack(x[A10a.F:A10a.F+A10a.C]).reshape(A10a.P, -1)
        xi = A10a.get_xi()
        return p.T @ ( np.sum(x, axis=1, keepdims=True) - np.sum(y, axis=1, keepdims=True) - np.sum(xi, axis=1, keepdims=True) )

    @staticmethod
    def utility_function(x_i, Q_i, b_i):
        return -0.5 * x_i.T @ Q_i @ x_i + b_i.T @ x_i

    @staticmethod
    def utility_function_der(x_i, Q_i, b_i):
        return - Q_i @ x_i + b_i

    @staticmethod
    def obj_func_consumers_1_der(x):
        x = np.hstack(x[A10a.F: A10a.F + A10a.C][0]).reshape(A10a.P, -1)
        Q_i = A10a.get_constants(i=1).get('Q_i')
        b_i = A10a.get_constants(i=1).get('b_i')
        return A10a.utility_function_der(x, Q_i, b_i)

    @staticmethod
    def obj_func_consumers_2_der(x):
        x = np.hstack(x[A10a.F: A10a.F + A10a.C][1]).reshape(A10a.P, -1)
        Q_i = A10a.get_constants(i=2).get('Q_i')
        b_i = A10a.get_constants(i=2).get('b_i')
        return A10a.utility_function_der(x, Q_i, b_i)

    @staticmethod
    def obj_func_consumers_3_der(x):
        x = np.hstack(x[A10a.F: A10a.F + A10a.C][2]).reshape(A10a.P, -1)
        Q_i = A10a.get_constants(i=3).get('Q_i')
        b_i = A10a.get_constants(i=3).get('b_i')
        return A10a.utility_function_der(x, Q_i, b_i)

    @staticmethod
    def obj_func_consumers_4_der(x):
        x = np.hstack(x[A10a.F: A10a.F + A10a.C][3]).reshape(A10a.P, -1)
        Q_i = A10a.get_constants(i=4).get('Q_i')
        b_i = A10a.get_constants(i=4).get('b_i')
        return A10a.utility_function_der(x, Q_i, b_i)

    @staticmethod
    def obj_func_consumers_5_der(x):
        x = np.hstack(x[A10a.F: A10a.F + A10a.C][4]).reshape(A10a.P, -1)
        Q_i = A10a.get_constants(i=5).get('Q_i')
        b_i = A10a.get_constants(i=5).get('b_i')
        return A10a.utility_function_der(x, Q_i, b_i)

    @staticmethod
    def obj_func_firms_der(x):
        p = x[-1].reshape(-1,1)
        return p # returns (1,F) vector

    @staticmethod
    def obj_func_market_der(x):
        y = np.hstack(x[:A10a.F]).reshape(A10a.P, -1)
        x = np.hstack(x[A10a.F:A10a.F + A10a.C]).reshape(A10a.P, -1)
        xi = A10a.get_xi()
        return np.sum(x, axis=1, keepdims=True) - np.sum(y, axis=1, keepdims=True) - np.sum(xi, axis=1, keepdims=True)

    @staticmethod
    def g0(x):
        idx = 10*np.array([i+1 for i in range(A10a.F)]).reshape(-1,1)
        y = np.hstack(x[:A10a.F]).reshape(A10a.P, -1)
        sum_y = np.sum(y,axis=0).reshape(-1,1)**2
        return sum_y - idx

    @staticmethod
    def g1(x):
        p = x[-1].reshape(-1, 1)
        x = np.hstack(x[A10a.F: A10a.F + A10a.C]).reshape(A10a.P, -1)
        xi = A10a.get_xi()
        return (p.T @ x).reshape(-1,1) - (p.T @ xi).reshape(-1,1)

    @staticmethod
    def g2(x):
        p = x[-1].reshape(-1, 1)
        return np.sum(p, axis=0) - 1

    @staticmethod
    def g3(x):
        p = x[-1].reshape(-1,   1)
        return -np.sum(p, axis=0) + 1

    @staticmethod
    def g0_der(x):
        y = 2 * x[:A10a.F*A10a.P].reshape(-1, 1)
        pad = np.array([0 for i in range(A10a.C * A10a.P + A10a.P)]).reshape(-1, 1)
        return np.vstack((y, pad))

    @staticmethod
    def g1_der(x):
        zeros = np.zeros_like(x).reshape(-1,1)
        p = x[-A10a.P:].reshape(-1, 1)
        p_stack = np.vstack([p for _ in range(A10a.C)])
        zeros[A10a.F*A10a.P : A10a.F*A10a.P + A10a.P * A10a.C] = p_stack
        return zeros

    @staticmethod
    def g2_der(x):
        return 1

    @staticmethod
    def g3_der(x):
        return -1



    @staticmethod
    def get_xi():
        x1 = np.array([2, 3, 4]).reshape(-1, 1)
        x2 = np.array([2, 3, 4]).reshape(-1, 1)
        x3 = np.array([6, 5, 4]).reshape(-1, 1)
        x4 = np.array([6, 5, 4]).reshape(-1, 1)
        x5 = np.array([6, 5, 4]).reshape(-1, 1)
        xi = np.hstack([x1,x2,x3,x4,x5]).reshape(A10a.P, -1 )
        return xi

    @staticmethod
    def get_constants(i=1):
        if i in [1,2]:
            Q_i = np.array([
                [6, -2, 5],
                [-2, 6, -7],
                [5, -7, 20]
            ])
            b_i = np.array([30+i+A10a.F,30+i+A10a.F,30+i+A10a.F]).reshape(-1,1)
        elif i in [3,4,5]:
            Q_i = np.array([
                [6, 1, 0],
                [1, 7, -5],
                [0, -5, 7]
            ])
            b_i = np.array([30 + 2*(i + A10a.F), 30 + 2*(i + A10a.F), 30 + 2*(i + A10a.F)]).reshape(-1, 1)
        return dict(Q_i=Q_i, b_i=b_i)