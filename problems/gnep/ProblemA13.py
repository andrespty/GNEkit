import jax.numpy as jnp
from solvers.schema import VectorList
from solvers.gnep_solver.BaseProblem import BaseProblem
from solvers.gnep_solver.BasePlayer import Player

class ProblemA13(BaseProblem):
    def known_solution(self):
        value_1 = [21.14480155732168,
                   16.02785326538717,
                   2.7259709656438]
        return value_1

    def define_players(self):
        player_vector_sizes = [1, 1, 1]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0,1] for _ in range(3)]
        bounds = [(0.0, 100.0) for _ in range(3)]
        return Player.batch_create(
            player_vector_sizes,
            player_objective_functions,
            player_constraints,
            bounds
        )

    def objectives(self):
        def obj_func(x,c1,c2,S):
            d1 = 3
            d2 = 0.01
            obj = x * (c1 + c2 * x - d1 + d2 * S)
            return jnp.reshape(obj, ())

        def obj_func_1(x: VectorList) -> jnp.array:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            S = x1 + x2 + x3
            c1 = 0.10
            c2 = 0.01
            return obj_func(x1, c1, c2, S)

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            S = x1 + x2 + x3
            c1 = 0.12
            c2 = 0.05
            return obj_func(x2, c1, c2, S)

        def obj_func_3(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            S = x1 + x2 + x3
            c1 = 0.15
            c2 = 0.01
            return obj_func(x3, c1, c2, S)

        return [obj_func_1, obj_func_2, obj_func_3]

    def constraints(self):
        def g0(x: VectorList):
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            return jnp.reshape(3.25 * x1 + 1.25 * x2 + 4.125 * x3 - 100, ())

        def g1(x: VectorList):
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            return jnp.reshape(2.2915 * x1 + 1.5625 * x2 + 2.814 * x3 - 100, ())
        return [g0, g1]
