import jax.numpy as jnp
from gnep_solver import Vector, VectorList
from tester_solver.ManualBaseProblem import ManualBaseProblem
from gnep_solver.Player import Player
from typing import List

class ProblemA3(ManualBaseProblem):
    def __init__(self, players: List[Player] = None):
        super().__init__(players)
        # --- Matrix Constants
        self.A1 = jnp.array([[20, 5, 3], [5, 5, -5], [3, -5, 15]])
        self.A2 = jnp.array([[11, -1], [-1, 9]])
        self.A3 = jnp.array([[48, 39], [39, 53]])

        self.B1 = jnp.array([[-6, 10, 11, 20], [10, -4, -17, 9], [15, 8, -22, 21]])
        self.B2 = jnp.array([[20, 1, -3, 12, 1], [10, -4, 8, 16, 21]])
        self.B3 = jnp.array([[10, -2, 22, 12, 16], [9, 19, 21, -4, 20]])

        self.b1 = jnp.array([[1], [-1], [1]])
        self.b2 = jnp.array([[1], [0]])
        self.b3 = jnp.array([[-1], [2]])

    def known_solution(self):
        value = [-0.38046562696258, -0.12266997083581, -0.99322817120517,
                 0.39034789080544, 1.16385412687962, 0.05039533464000,
                 0.01757740533460]
        return value

    def define_players(self):
        player_vector_sizes = [3, 2, 2]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0, 1], [2], [3]]
        bounds = [(-10, 10) for _ in range(3)]
        return Player.batch_create(
            player_vector_sizes,
            player_objective_functions,
            player_constraints,
            bounds
        )

    def objectives(self):
        def obj_func(x, x_ni, A, B, b):
            # Result of quadratic form is (1, 1). JAX needs a scalar for grad.
            return jnp.reshape(0.5 * x.T @ A @ x + x.T @ (B @ x_ni + b), ())

        def obj_func_1(x: VectorList) -> jnp.ndarray:
            x1 = x[0].reshape(-1, 1)
            x_n1 = jnp.concatenate([x[1].ravel(), x[2].ravel()]).reshape(-1, 1)
            # Using jnp.reshape to ensure we return a 0-dim array (scalar)
            return obj_func(x1, x_n1, self.A1, self.B1, self.b1)

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x2 = x[1].reshape(-1, 1)
            x_n2 = jnp.concatenate([x[0].ravel(), x[2].ravel()]).reshape(-1, 1)
            return obj_func(x2, x_n2, self.A2, self.B2, self.b2)

        def obj_func_3(x: VectorList) -> jnp.ndarray:
            x3 = x[2].reshape(-1, 1)
            x_n3 = jnp.concatenate([x[0].ravel(), x[1].ravel()]).reshape(-1, 1)
            return obj_func(x3, x_n3, self.A3, self.B3, self.b3)
        # df/dx1, df/dx2
        return [obj_func_1, obj_func_2, obj_func_3]

    def objectives_der(self):
        def obj_func_der(x, x_ni, A, B, b):
            obj = A @ x + B @ x_ni + b
            return jnp.reshape(obj, (-1,))
        def obj_func_der_1(x):
            x1 = x[0].reshape(-1, 1)
            x_n1 = jnp.concatenate([x[1].ravel(), x[2].ravel()]).reshape(-1, 1)
            return {0: obj_func_der(x1, x_n1, self.A1, self.B1, self.b1)}
        def obj_func_der_2(x):
            x2 = x[1].reshape(-1, 1)
            x_n2 = jnp.concatenate([x[0].ravel(), x[2].ravel()]).reshape(-1, 1)
            return {1: obj_func_der(x2, x_n2, self.A2, self.B2, self.b2)}
        def obj_func_der_3(x):
            x3 = x[2].reshape(-1, 1)
            x_n3 = jnp.concatenate([x[0].ravel(), x[1].ravel()]).reshape(-1, 1)
            return {2: obj_func_der(x3, x_n3, self.A3, self.B3, self.b3)}
        return [obj_func_der_1, obj_func_der_2, obj_func_der_3]

    def constraints(self):
        def g0(x: VectorList):
            # sum(x1) - 20 <= 0
            return jnp.sum(x[0]) - 20.0

        def g1(x: VectorList):
            x1, x2, x3 = x
            # Ensure we use JAX indices and no .item() or [0]
            val = x1[0] + x1[1] - x1[2] - x2[0] + x3[1] - 5.0
            return jnp.reshape(val, ())  #

        def g2(x: VectorList):
            x1, x2, x3 = x
            val = x2[0] - x2[1] - x1[1] - x1[2] + x3[0] - 7.0
            return jnp.reshape(val, ())

        def g3(x: VectorList):
            x1, x2, x3 = x
            val = x3[1] - x1[0] - x1[2] + x2[0] - 4.0
            return jnp.reshape(val, ())

        return [g0, g1, g2, g3]

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
            dx1 = jnp.array([0., -1., -1.])
            dx2 = jnp.array([1., -1.])
            dx3 = jnp.array([1., 0.])
            return [dx1, dx2, dx3]

        def g3_der(x):
            dx1 = jnp.array([-1., 0., -1.])
            dx2 = jnp.array([1., 0.])
            dx3 = jnp.array([0., 1.])
            return [dx1, dx2, dx3]
        return [g0_der, g1_der, g2_der, g3_der]