import numpy as np
from gne_solver.GNEProblem import GNEProblem
from gne_solver.GNEPlayer import GNEPlayer

class A1(GNEProblem):
    def known_solution(self):
        value_1 = [0.29923815223336,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805]
        return [value_1]

    def define_players(self):
        B = 1
        player_vector_sizes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        player_objective_functions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        player_constraints = [[None], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        bounds = [(0.3, 0.5), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B)]
        return GNEPlayer.batch_create(player_vector_sizes, player_objective_functions, player_constraints, bounds)

    def objectives(self):
        def obj_func(x):
            # x: numpy array (N, 1)
            # B: constant
            S = sum(x)
            B = 1
            obj = (-x / S) * (1 - S / B)
            return obj
        return [obj_func]

    def objectives_der(self):
        def obj_func_der(x):
            # x: numpy array (N,1)
            # B: constant
            x = np.concatenate(x).reshape(-1, 1)
            B = 1
            S = sum(x)
            # print(S)
            obj = 1 + (x - S) / (S ** 2)
            return obj
        return [obj_func_der]

    def constraints(self):
        def g0(x):
            # x: numpy array (N,1)
            # B: constant
            B = 1
            return sum(x) - B
        return [g0]

    def constraints_der(self):
        def g0_der(x):
            return 1
        return [g0_der]
