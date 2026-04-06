import jax.numpy as jnp
from solvers.gnep_solver import VectorList
from solvers.gnep_solver.BaseProblem import BaseProblem
from solvers.gnep_solver.BasePlayer import Player

class ProblemA11(BaseProblem):
    def known_solution(self):
        value_1 = [0.75002417863494, 0.2500241786374]
        return value_1

    def define_players(self):
        player_vector_sizes = [1, 1]
        player_objective_functions = [0, 1]
        player_constraints = [[0], [0]]
        bounds = [(0, 1), (0, 1)]
        return Player.batch_create(
            player_vector_sizes,
            player_objective_functions,
            player_constraints,
            bounds
        )

    def objectives(self):
        def obj_func_1(x: VectorList) -> jnp.array:
            x1 = x[0].reshape(-1, 1)
            x2 = x[1]
            return jnp.reshape((x1 - 1)**2,())

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            return jnp.reshape((x2 - 0.5) ** 2, ())

        return [obj_func_1, obj_func_2]

    def constraints(self):
        def g0(x: VectorList):
            x1, x2 = x
            return jnp.reshape(x1 + x2 - 1, ())
        return [g0]
