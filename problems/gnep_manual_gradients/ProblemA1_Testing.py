import jax.numpy as jnp
from solvers.gnep_solver import VectorList
from tester_solver.ManualBaseProblem import ManualBaseProblem
from solvers.gnep_solver.BasePlayer import Player


class ProblemA1_Manual(ManualBaseProblem):
    def known_solution(self):
        value = [0.29923815223336,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805]
        return value

    def define_players(self):
        player_vector_sizes = [1 for _ in range(10)]
        player_objective_functions = [i for i in range(10)]
        player_constraints = [[None], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        B=1
        bounds = [(0.3, 0.5), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B)]
        return Player.batch_create(
            player_vector_sizes,
            player_objective_functions,
            player_constraints,
            bounds
        )

    def objectives(self):
        def obj_func(x):
            actions = jnp.concatenate(x)
            S_total = jnp.sum(actions)
            return jnp.reshape((-actions / S_total) * (1 - S_total), (-1,))

        return [lambda x, i=i: obj_func(x)[i] for i in range(len(self.players))]

    def objectives_der(self):
        def obj_func_der(x):
            # x: numpy array (N,1)
            # B: constant
            x = jnp.concatenate(x).reshape(-1, 1)
            B = 1
            S = jnp.sum(x)
            # print(S)
            obj = 1 + (x - S)/(S**2)
            return obj
        return [lambda x, i=i: {i:obj_func_der(x)[i]} for i in range(len(self.players))]

    def constraints(self):
        def g0(x: VectorList):
            return jnp.sum(jnp.concatenate(x)) - 1

        return [g0]

    def constraints_der(self):
        def g0_der(x):
            dx1 = jnp.array([1.])
            return [dx1 for _ in range(len(self.players))]
        return [g0_der]