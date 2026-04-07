import jax.numpy as jnp
from solvers.schema import VectorList
from solvers.gnep_solver.BaseProblem import BaseProblem
from solvers.gnep_solver.BasePlayer import Player

class ProblemA15(BaseProblem):
    def known_solution(self):
        value_1 =  [46.66150692423980, 32.15293850189938, 15.00419467998705, 22.10485810522063, 12.34076570922471, 12.34076570922471]
        return value_1

    def define_players(self):
        player_vector_sizes = [1, 2, 3]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[None] for _ in range(3)]
        P1_bounds = [(0,80)]
        P2_bounds = [(0,80), (0,50)]
        P3_bounds = [(0,55), (0,30), (0,40)]
        bounds = [P1_bounds, P2_bounds, P3_bounds]
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
            x3 = x[2]

            S = 2 * (jnp.sum(x1) + jnp.sum(x2) + jnp.sum(x3)) - 378.4
            c1 = 0.04
            d1 = 2.0
            e1 = 0.0
            obj = S * x1.ravel() + (0.5 * c1 * x1.ravel() ** 2 + d1 * x1.ravel() + e1)
            return jnp.reshape(obj, ())

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1].reshape(-1,1)
            x3 = x[2]

            S = 2 * (jnp.sum(x1) + jnp.sum(x2) + jnp.sum(x3)) - 378.4
            v2 = jnp.sum(x2)
            c2 = jnp.array([0.035, 0.125]).reshape(-1,1)
            d2 = jnp.array([1.75, 1]).reshape(-1,1)
            e2 = jnp.array([0.0, 0.0]).reshape(-1,1)
            obj = S * v2 + jnp.sum(0.5 * c2 * x2 ** 2 + d2 * x2 + e2)
            return jnp.reshape(obj, ())

        def obj_func_3(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2].reshape(-1,1)

            S = 2 * (jnp.sum(x1) + jnp.sum(x2) + jnp.sum(x3)) - 378.4
            v3 = jnp.sum(x3)
            c3 = jnp.array([0.0166, 0.05, 0.05]).reshape(-1,1)
            d3 = jnp.array([3.25, 3.0, 3.0]).reshape(-1,1)
            e3 = jnp.array([0.0, 0.0, 0.0]).reshape(-1,1)
            obj = S * v3 + jnp.sum(0.5 * c3 * x3 ** 2 + d3 * x3 + e3)
            return jnp.reshape(obj, ())

        return [obj_func_1, obj_func_2, obj_func_3]

    def constraints(self):
        return []
