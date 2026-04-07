import jax.numpy as jnp
from solvers.schema import VectorList
from solvers.gnep_solver.BaseProblem import BaseProblem
from solvers.gnep_solver.BasePlayer import Player

class ProblemA12(BaseProblem):
    def known_solution(self):
        value_1 = [5.33331555561568, 5.33331555561]
        return value_1

    def define_players(self):
        player_vector_sizes = [1, 1]
        player_objective_functions = [0, 1]
        player_constraints = [[None], [None]]
        bounds = [(-10, 10), (-10, 10)]
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
            return jnp.reshape(x1 * (x1 + x2 - 16), ())

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            return jnp.reshape(x2 * (x1 + x2 - 16), ())

        return [obj_func_1, obj_func_2]

    def constraints(self):
        return []
