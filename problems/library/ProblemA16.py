import jax.numpy as jnp
from gnep_solver import Vector, VectorList
from gnep_solver.BaseProblem import BaseProblem
from gnep_solver.Player import Player


class ProblemA16(BaseProblem):
    def __init__(self):
        super().__init__()
        self.P = 75 # [75, 100, 150, 200]
        self.gamma = 1.1

    def known_solution(self):
        value_1 = jnp.array([10.403965, 13.035817, 15.407354, 17.381556, 18.771308])  # 75
        value_2 = jnp.array([14.050088, 17.798379, 20.907187, 23.111429, 24.132916])  # 100
        value_3 = jnp.array([23.588799, 28.684248, 32.021533, 33.287258, 32.418182])  # 150
        value_4 = jnp.array([35.785329, 40.748959, 42.802485, 41.966381, 38.696846])  # 200
        value_5 = jnp.array([36.912, 41.842, 43.705, 42.665, 39.182])
        return value_1

    def define_players(self):
        player_vector_sizes = [1, 1, 1, 1, 1]
        player_objective_functions = [0, 1, 2, 3, 4]
        player_constraints = [[0] for _ in range(5)]
        bounds = [(0, 100) for _ in range(5)]
        return Player.batch_create(
            player_vector_sizes,
            player_objective_functions,
            player_constraints,
            bounds
        )

    def objectives(self):
        def f_func(x_i, c_i, delta_i, K_i):
            return c_i * x_i + (delta_i/(1+delta_i)) * K_i**(-1/delta_i) * x_i ** ((1+delta_i)/delta_i)

        def obj_func(x_i, c_i, delta_i, K_i, S):
            return jnp.reshape(f_func(x_i, c_i, delta_i, K_i) - 5000 ** (1/self.gamma) * x_i * S ** (-1/self.gamma), ())

        def obj_func_1(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]
            x5 = x[4]

            c1 = 10
            K1 = 5
            delta1 = 1.2
            S = x1 + x2 + x3 + x4 + x5
            return obj_func(x1, c1, delta1, K1, S)

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]
            x5 = x[4]

            c2 = 8
            K2 = 5
            delta2 = 1.1
            S = x1 + x2 + x3 + x4 + x5
            return obj_func(x2, c2, delta2, K2, S)

        def obj_func_3(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]
            x5 = x[4]

            c3 = 6
            K3 = 5
            delta3 = 1.0
            S = x1 + x2 + x3 + x4 + x5
            return obj_func(x3, c3, delta3, K3, S)

        def obj_func_4(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]
            x5 = x[4]

            c4 = 4
            K4 = 5
            delta4 = 0.9
            S = x1 + x2 + x3 + x4 + x5
            return obj_func(x4, c4, delta4, K4, S)

        def obj_func_5(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]
            x5 = x[4]

            c5 = 2
            K5 = 5
            delta5 = 0.8
            S = x1 + x2 + x3 + x4 + x5
            return obj_func(x5, c5, delta5, K5, S)

        return [obj_func_1, obj_func_2, obj_func_3, obj_func_4, obj_func_5]

    def constraints(self):
        def g0(x: VectorList):
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]
            x5 = x[4]

            return jnp.reshape(x1 + x2 + x3 + x4 + x5 - self.P, ())
        return [g0]
