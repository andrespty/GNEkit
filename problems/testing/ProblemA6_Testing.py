import jax.numpy as jnp
from gnep_solver import Vector, VectorList
from tester_solver.ManualBaseProblem import ManualBaseProblem
from gnep_solver.Player import Player
from typing import List

class ProblemA6(ManualBaseProblem):
    def __init__(self, players: List[Player] = None):
        super().__init__(players)
        # --- Matrix Constants (B and b are constant; A matrices are x-dependent)
        self.B1 = jnp.array([[-2, 0, 1, 2], [1, -4, -7, 9], [3, 8, 22, 21]])
        self.B2 = jnp.array([[-2, 1, -3, 12, -1], [0, -4, 8, 16, 21]])
        self.B3 = jnp.array([[1, -7, 22, -12, 16], [2, -9, 21, 1, 21]])

        self.b1 = jnp.array([[1], [-2], [-3]])
        self.b2 = jnp.array([[1], [2]])
        self.b3 = jnp.array([[1], [-2]])

    def known_solution(self):
        value = [0.99987722673822, 2.31570964703584, 0.99989251930167,
                 1.31499923583926, 0.99989852480755,
                 0.99992298465841, 1.09709158271764]
        return value

    def define_players(self):
        player_vector_sizes = [3, 2, 2]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0, 1, 2], [3, 4], [5, 6]]
        bounds = [(1, 10) for _ in range(3)]
        return Player.batch_create(
            player_vector_sizes,
            player_objective_functions,
            player_constraints,
            bounds
        )

    def objectives(self):
        def obj_func(x, x_ni, A, B, b):
            return jnp.reshape(0.5 * x.T @ A @ x + x.T @ (B @ x_ni + b), ())

        def obj_func_1(x: VectorList) -> jnp.ndarray:
            x1 = x[0].reshape(-1, 1)
            x2 = x[1].reshape(-1, 1)
            x_n1 = jnp.concatenate([x[1].ravel(), x[2].ravel()]).reshape(-1, 1)
            A1 = jnp.array([
                [20 + x2[0, 0] ** 2, 5, 3],
                [5, 5 + x2[1, 0] ** 2, -5],
                [3, -5, 15]
            ])
            return obj_func(x1, x_n1, A1, self.B1, self.b1)

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x2 = x[1].reshape(-1, 1)
            x3 = x[2].reshape(-1, 1)
            x_n2 = jnp.concatenate([x[0].ravel(), x[2].ravel()]).reshape(-1, 1)
            A2 = jnp.array([
                [11 + x3[0, 0] ** 2, -1],
                [-1, 9]
            ])
            return obj_func(x2, x_n2, A2, self.B2, self.b2)

        def obj_func_3(x: VectorList) -> jnp.ndarray:
            x1 = x[0].reshape(-1, 1)
            x3 = x[2].reshape(-1, 1)
            x_n3 = jnp.concatenate([x[0].ravel(), x[1].ravel()]).reshape(-1, 1)
            A3 = jnp.array([
                [48, 39],
                [39, 53 + x1[0, 0] ** 2]
            ])
            return obj_func(x3, x_n3, A3, self.B3, self.b3)

        return [obj_func_1, obj_func_2, obj_func_3]

    def objectives_der(self):
        def obj_func_der(x, x_ni, A, B, b):
            obj = A @ x + B @ x_ni + b
            return jnp.reshape(obj, (-1,))

        def obj_func_der_1(x):
            x1 = x[0].reshape(-1, 1)
            x2 = x[1].reshape(-1, 1)
            x_n1 = jnp.concatenate([x[1].ravel(), x[2].ravel()]).reshape(-1, 1)
            A1 = jnp.array([
                [20 + x2[0, 0] ** 2, 5, 3],
                [5, 5 + x2[1, 0] ** 2, -5],
                [3, -5, 15]
            ])
            return {0: obj_func_der(x1, x_n1, A1, self.B1, self.b1)}

        def obj_func_der_2(x):
            x2 = x[1].reshape(-1, 1)
            x3 = x[2].reshape(-1, 1)
            x_n2 = jnp.concatenate([x[0].ravel(), x[2].ravel()]).reshape(-1, 1)
            A2 = jnp.array([
                [11 + x3[0, 0] ** 2, -1],
                [-1, 9]
            ])
            return {1: obj_func_der(x2, x_n2, A2, self.B2, self.b2)}

        def obj_func_der_3(x):
            x1 = x[0].reshape(-1, 1)
            x3 = x[2].reshape(-1, 1)
            x_n3 = jnp.concatenate([x[0].ravel(), x[1].ravel()]).reshape(-1, 1)
            A3 = jnp.array([
                [48, 39],
                [39, 53 + x1[0, 0] ** 2]
            ])
            return {2: obj_func_der(x3, x_n3, A3, self.B3, self.b3)}

        return [obj_func_der_1, obj_func_der_2, obj_func_der_3]

    def constraints(self):
        def g0(x: VectorList):
            # sum(x1) - 20 <= 0
            return jnp.sum(x[0]) - 20.0

        def g1(x: VectorList):
            x1, x2, x3 = x
            val = x1[0] + x1[1] - x1[2] - x2[0] + x3[1] - 3.7
            return jnp.reshape(val, ())

        def g2(x: VectorList):
            x1, x2, x3 = x
            val = x1[0] ** 4 + x3[0] * x1[1] - x2[0] - 2.0
            return jnp.reshape(val, ())

        def g3(x: VectorList):
            x1, x2, x3 = x
            val = x2[0] - x2[1] - x1[1] - x1[2] + x3[0] - 7.0
            return jnp.reshape(val, ())

        def g4(x: VectorList):
            x1, x2, x3 = x
            val = (x2[0] - 2) ** 2 + x2[1] ** 2 - x1[0] ** 2 - 0.75
            return jnp.reshape(val, ())

        def g5(x: VectorList):
            x1, x2, x3 = x
            val = x3[1] - x1[0] - x1[2] + x2[0] - 4.0
            return jnp.reshape(val, ())

        def g6(x: VectorList):
            x1, x2, x3 = x
            val = 2 * x3[0] ** 2 - (x3[1] - 2) ** 2 - x2[0] * x3[0] - 1.5
            return jnp.reshape(val, ())

        return [g0, g1, g2, g3, g4, g5, g6]

    def constraints_der(self):
        def g0_der(x):
            dx1 = jnp.array([1., 1., 1.])
            dx2 = jnp.array([0., 0.])
            dx3 = jnp.array([0., 0.])
            return [dx1, dx2, dx3]

        def g1_der(x):
            dx1 = jnp.array([1., 1., -1.])
            dx2 = jnp.array([-1., 0.])
            dx3 = jnp.array([0., 1.])
            return [dx1, dx2, dx3]

        def g2_der(x):
            x1, x2, x3 = x
            dx1 = jnp.array([4 * x1[0] ** 3, x3[0], 0.])
            dx2 = jnp.array([-1., 0.])
            dx3 = jnp.array([x1[1], 0.])
            return [dx1, dx2, dx3]

        def g3_der(x):
            dx1 = jnp.array([0., -1., -1.])
            dx2 = jnp.array([1., -1.])
            dx3 = jnp.array([1., 0.])
            return [dx1, dx2, dx3]

        def g4_der(x):
            x1, x2, x3 = x
            dx1 = jnp.array([-2 * x1[0], 0., 0.])
            dx2 = jnp.array([2 * (x2[0] - 2), 2 * x2[1]])
            dx3 = jnp.array([0., 0.])
            return [dx1, dx2, dx3]

        def g5_der(x):
            dx1 = jnp.array([-1., 0., -1.])
            dx2 = jnp.array([1., 0.])
            dx3 = jnp.array([0., 1.])
            return [dx1, dx2, dx3]

        def g6_der(x):
            x1, x2, x3 = x
            dx1 = jnp.array([0., 0., 0.])
            dx2 = jnp.array([-x3[0], 0.])
            dx3 = jnp.array([4 * x3[0] - x2[0], -2 * (x3[1] - 2)])
            return [dx1, dx2, dx3]

        return [g0_der, g1_der, g2_der, g3_der, g4_der, g5_der, g6_der]