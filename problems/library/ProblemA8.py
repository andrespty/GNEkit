import jax.numpy as jnp
from solvers.gnep_solver import VectorList
from solvers.gnep_solver.BaseProblem import BaseProblem
from solvers.gnep_solver.BasePlayer import Player

class ProblemA8(BaseProblem):
    def known_solution(self):
        value_1 = [
            0.62503131162143,
            0.37500031253875,
            0.93754579549990
        ]
        return value_1

    def define_players(self):
        player_vector_sizes = [1, 1, 1]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0, 1], [0, 1], []]
        bounds = [(0, 100), (0, 100), (0, 2)]
        return Player.batch_create(
            player_vector_sizes,
            player_objective_functions,
            player_constraints,
            bounds
        )

    def objectives(self):
        def obj_func_1(x: VectorList) -> jnp.array:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            return jnp.reshape(-x1, ())

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            return jnp.reshape((x2 - 0.5) ** 2, ())

        def obj_func_3(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            return jnp.reshape((x3 - 1.5 * x1) ** 2, ())

        return [obj_func_1, obj_func_2, obj_func_3]

    def constraints(self):
        def g0(x: VectorList):
            x1, x2, x3 = x
            return jnp.reshape(x1 + x2 - 1, ())

        def g1(x: VectorList):
            x1, x2, x3 = x
            return jnp.reshape(x3 - x1 - x2,())

        return [g0, g1]
