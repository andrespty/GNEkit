import jax.numpy as jnp
from solvers.gnep_solver import VectorList
from solvers.gnep_solver.BaseProblem import BaseProblem
from solvers.gnep_solver.BasePlayer import Player


class ProblemA17(BaseProblem):
    def known_solution(self):
        value_1 = [0.00000737,
                   11.00002440,
                   7.99997560]
        return value_1

    def define_players(self):
        player_vector_sizes = [2, 1]
        player_objective_functions = [0, 1]
        player_constraints = [[0, 1] for _ in range(2)]
        bounds = [(0, 100) for _ in range(2)]
        return Player.batch_create(
            player_vector_sizes,
            player_objective_functions,
            player_constraints,
            bounds
        )

    def objectives(self):
        def obj_func_1(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            return jnp.reshape(x1[0]**2 + x1[0] * x1[1] + x1[1]**2 + jnp.sum(x1) * x2[0] - 25*x1[0] - 38*x1[1], ())

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            return jnp.reshape(x2[0]**2 + jnp.sum(x1) * x2[0] - 25 * x2[0], ())

        return [obj_func_1, obj_func_2]

    def constraints(self):
        def g0(x: VectorList):
            x1 = x[0]
            x2 = x[1]
            return jnp.reshape(x1[0] + 2*x1[1] - x2[0] -14, ())

        def g1(x: VectorList):
            x1 = x[0]
            x2 = x[1]
            return jnp.reshape(3*x1[0] + 2*x1[1] + x2[0] - 30, ())
        return [g0, g1]
