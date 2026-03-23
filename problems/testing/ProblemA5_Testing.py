import jax.numpy as jnp
from gnep_solver import Vector, VectorList
from tester_solver.ManualBaseProblem import ManualBaseProblem
from gnep_solver.Player import Player
from typing import List

class ProblemA5(ManualBaseProblem):
    def __init__(self, players: List[Player] = None):
        super().__init__(players)
        # --- Matrix Constants
        self.A1 = jnp.array([[20, 6, 0], [6, 6, -1], [0, -1, 8]])
        self.A2 = jnp.array([[11, 1], [1, 7]])
        self.A3 = jnp.array([[28, 14], [14, 29]])

        self.B1 = jnp.array([[-1, -2, -4, -3], [0, -3, 0, -4], [0, 1, 9, 6]])
        self.B2 = jnp.array([[-1, 0, 0, -7, 4], [-2, -3, 1, 4, 11]])
        self.B3 = jnp.array([[-4, 0, 9, -7, 4], [-3, -4, 6, 4, 11]])

        self.b1 = jnp.array([[1], [-1], [1]])
        self.b2 = jnp.array([[1], [0]])
        self.b3 = jnp.array([[-1], [2]])

    def known_solution(self):
        value = [-0.00006229891126, 0.20279012064850, -0.00003469558295,
                 -0.00028322020054, 0.07258934064261,
                 0.02531280162415, -0.00007396699835]
        return value

    def define_players(self):
        player_vector_sizes = [3, 2, 2]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0, 1], [2], [3]]
        bounds = [(0, 10) for _ in range(3)]
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
            return obj_func(x1, x_n1, self.A1, self.B1, self.b1)

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x2 = x[1].reshape(-1, 1)
            x_n2 = jnp.concatenate([x[0].ravel(), x[2].ravel()]).reshape(-1, 1)
            return obj_func(x2, x_n2, self.A2, self.B2, self.b2)

        def obj_func_3(x: VectorList) -> jnp.ndarray:
            x3 = x[2].reshape(-1, 1)
            x_n3 = jnp.concatenate([x[0].ravel(), x[1].ravel()]).reshape(-1, 1)
            return obj_func(x3, x_n3, self.A3, self.B3, self.b3)

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
            # sum(x1) + sum(x2) + sum(x3) - 20 <= 0
            return jnp.sum(x[0]) + jnp.sum(x[1]) + jnp.sum(x[2]) - 20.0

        def g1(x: VectorList):
            x1, x2, x3 = x
            val = x1[0] + x1[1] - x1[2] - x2[0] + x3[1] - 5.0
            return jnp.reshape(val, ())

        def g2(x: VectorList):
            x1, x2, x3 = x
            val = x2[0] + x2[1] - x1[1] - x1[2] + x3[0] - 7.0
            return jnp.reshape(val, ())

        def g3(x: VectorList):
            x1, x2, x3 = x
            val = x3[1] - x1[0] - x1[2] + x2[0] - 4.0
            return jnp.reshape(val, ())

        return [g0, g1, g2, g3]

    def constraints_der(self):
        def g0_der(x):
            dx1 = jnp.array([1., 1., 1.])
            dx2 = jnp.array([1., 1.])
            dx3 = jnp.array([1., 1.])
            return [dx1, dx2, dx3]

        def g1_der(x):
            dx1 = jnp.array([1., 1., -1.])
            dx2 = jnp.array([-1., 0.])
            dx3 = jnp.array([0., 1.])
            return [dx1, dx2, dx3]

        def g2_der(x):
            dx1 = jnp.array([0., -1., -1.])
            dx2 = jnp.array([1., 1.])
            dx3 = jnp.array([1., 0.])
            return [dx1, dx2, dx3]

        def g3_der(x):
            dx1 = jnp.array([-1., 0., -1.])
            dx2 = jnp.array([1., 0.])
            dx3 = jnp.array([0., 1.])
            return [dx1, dx2, dx3]

        return [g0_der, g1_der, g2_der, g3_der]