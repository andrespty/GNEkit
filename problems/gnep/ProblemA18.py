import jax.numpy as jnp
from solvers.schema import VectorList
from solvers.gnep_solver.BaseProblem import BaseProblem
from solvers.gnep_solver.BasePlayer import Player


class ProblemA18(BaseProblem):
    def define_players(self):
        player_vector_sizes = [6, 6]
        player_objective_functions = [0, 1]
        player_constraints = [[0, 1, 4, 5, 6], [2, 3, 4, 5, 6]]
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
            obj1 = (15-self.S1(x)) * (x1[0] + x1[3]) + (15-self.S2(x)) * (x1[1] + x1[4]) + (15-self.S3(x)) * (x1[2] + x1[5])
            return jnp.reshape(obj1, ())

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            obj2 = (15-self.S1(x)) * (x2[0] + x2[3]) + (15-self.S2(x)) * (x2[1] + x2[4]) + (15-self.S3(x)) * (x2[2] + x2[5])
            return jnp.reshape(obj2, ())

        return [obj_func_1, obj_func_2]

    def constraints(self):
        def g0(x: VectorList):
            x1 = x[0]
            x2 = x[1]
            return jnp.reshape(x1[0] + x1[1] + x1[2] - 100, ())

        def g1(x: VectorList):
            x1 = x[0]
            x2 = x[1]
            return jnp.reshape(x1[3] + x1[4] + x1[5] - 50, ())

        def g2(x: VectorList):
            x1 = x[0]
            x2 = x[1]
            return jnp.reshape(x2[0] + x2[1] + x2[2] - 100, ())

        def g3(x: VectorList):
            x1 = x[0]
            x2 = x[1]
            return jnp.reshape(x2[3] + x2[4] + x2[5] - 50, ())

        def g4(x: VectorList):
            return jnp.reshape(jnp.abs(self.S1(x) - self.S2(x)) - 1,())

        def g5(x: VectorList):
            return jnp.reshape(jnp.abs(self.S1(x) - self.S3(x)) - 1, ())

        def g6(x: VectorList):
            return jnp.reshape(jnp.abs(self.S2(x) - self.S3(x)) - 1, ())

        return [g0, g1, g2, g3, g4, g5, g6]

    def S1(self,x: VectorList):
        x1 = x[0]
        x2 = x[1]
        return 40 - (40 / 500) * (x1[0] + x1[3] + x2[0] + x2[3])

    def S2(self,x: VectorList):
        x1 = x[0]
        x2 = x[1]
        return 35 - (35 / 400) * (x1[1] + x1[4] + x2[1] + x2[4])

    def S3(self,x: VectorList):
        x1 = x[0]
        x2 = x[1]
        return 32 - (32 / 600) * (x1[2] + x1[5] + x2[2] + x2[5])