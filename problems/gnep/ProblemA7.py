import jax.numpy as jnp
from solvers.schema import VectorList
from solvers.gnep_solver.BaseProblem import BaseProblem
from solvers.gnep_solver.BasePlayer import Player


class ProblemA7(BaseProblem):
    def __init__(self):
        self.matrix = jnp.array([
            [110, -3, 22, -14, -27, 1, 9, 19, -2, 23, -7, -20, -4, 22, -19, 22, 3, 13, -12, 18],
            [-3, 79, -9, -21, 18, 61, 0, 14, 58, -11, 4, -16, 20, -19, 13, -17, -1, 24, 22, 5],
            [22, -9, 90, 28, 22, -9, -21, -1, -5, 29, 15, -7, 4, 30, 2, 9, -1, -19, -60, 4],
            [-14, -21, 28, 106, 11, -33, -42, 14, 28, -10, 3, 6, 13, 22, -8, 6, -3, 15, -3, 0],
            [-27, 18, 22, 11, 134, 4, -4, -29, 39, -62, 74, 2, 4, -34, -1, 13, 8, 18, 12, 35],
            [1, 61, -9, -33, 4, 119, -14, 12, 12, -6, -23, -14, 16, -4, 15, -2, 8, 16, 9, -9],
            [9, 0, -21, -42, -4, -14, 72, -14, 6, -9, 12, 2, -24, 13, 29, 17, 13, -1, 19, 21],
            [19, 14, -1, 14, -29, 12, -14, 92, -10, 5, 8, 0, -4, 23, 8, -50, -11, 48, -8, 3],
            [-2, 58, -5, 28, 39, 12, 6, -10, 124, -39, -4, -16, 24, -18, 26, 4, 13, 29, 43, 23],
            [23, -11, 29, -10, -62, -6, -9, 5, -39, 130, -42, -21, 21, 68, -24, -21, -30, -54, -23, 9],
            [-7, 4, 15, 3, 74, -23, 12, 8, -4, -42, 138, -4, -24, -12, -27, 24, 21, 2, -10, 18],
            [-20, -16, -7, 6, 2, -14, 2, 0, -16, -21, -4, 89, -11, -14, -16, -32, -7, -5, 13, -4],
            [-4, 20, 4, 13, 4, 16, -24, -4, 24, 21, -24, -11, 107, 31, -3, -2, -22, 17, 4, 22],
            [22, -19, 30, 22, -34, -4, 13, 23, -18, 68, -12, -14, 31, 116, -1, 5, -18, -16, -43, 27],
            [-19, 13, 2, -8, -1, 15, 29, 8, 26, -24, -27, -16, -3, -1, 98, -4, -2, 50, 23, 8],
            [22, -17, 9, 6, 13, -2, 17, -50, 4, -21, 24, -32, -2, 5, -4, 102, 46, -29, -17, -1],
            [3, -1, -1, -3, 8, 8, 13, -11, 13, -30, 21, -7, -22, -18, -2, 46, 110, -16, 24, 12],
            [13, 24, -19, 15, 18, 16, -1, 48, 29, -54, 2, -5, 17, -16, 50, -29, -16, 102, 45, 14],
            [-12, 22, -60, -3, 12, 9, 19, -8, 43, -23, -10, 13, 4, -43, 23, -17, 24, 45, 119, 21],
            [18, 5, 4, 0, 35, -9, 21, 3, 23, 9, 18, -4, 22, 27, 8, -1, 12, 14, 21, 59]
        ])
        super().__init__()
    def known_solution(self):
        value_1 = [
            0.99988245735506, 0.99985542095046, 0.99989138444537, 0.99988866261891, 0.99984494662577, 0.99986703246906, 0.99986897052169, 0.99992059068103, 0.99981225576918, 1.00013812006334, 0.99987211313045, 1.84253230021096, 0.99986555230493, 0.99987070302597, 0.99987574778109, 0.99993185140789, 0.99988068741824, 0.99984157413000, 0.99986193178624, 0.99983143496263
        ]
        return value_1

    def define_players(self):
        player_vector_sizes = [5,5,5,5]
        player_objective_functions = [0, 1, 2, 3]
        player_constraints = [[0], [1], [2], [3]]
        bounds = [(1, 5) for _ in range(4)]
        return Player.batch_create(
            player_vector_sizes,
            player_objective_functions,
            player_constraints,
            bounds
        )

    def objectives(self):
        def obj_func(x, x_ni, A, B):
            # Result of quadratic form is (1, 1). JAX needs a scalar for grad.
            return jnp.reshape(0.5 * x.T @ A @ x + x.T @ (B @ x_ni), ())

        def obj_func_1(x: VectorList) -> jnp.array:
            x1 = x[0].reshape(-1, 1)
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]
            A1 = self.matrix[0:5, 0:5]
            B1 = self.matrix[0:5, 5:20]
            x_n1 = jnp.concatenate([x2.ravel(), x3.ravel(), x4.ravel()]).reshape(-1, 1)
            return obj_func(x1, x_n1, A1, B1)

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1].reshape(-1, 1)
            x3 = x[2]
            x4 = x[3]
            A2 = self.matrix[5:10, 5:10]
            B2 = jnp.hstack((self.matrix[5:10, 0:5], self.matrix[5:10, 10:20]))
            x_n2 = jnp.concatenate([x1.ravel(), x3.ravel(), x4.ravel()]).reshape(-1, 1)
            return obj_func(x2, x_n2, A2, B2)

        def obj_func_3(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2].reshape(-1, 1)
            x4 = x[3]
            A3 = self.matrix[10:15, 10:15]
            B3 = jnp.hstack((self.matrix[10:15, 0:10], self.matrix[10:15, 15:20]))
            x_n3 = jnp.concatenate([x1.ravel(), x2.ravel(), x4.ravel()]).reshape(-1, 1)
            return obj_func(x3, x_n3, A3, B3)

        def obj_func_4(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]
            A4 = self.matrix[15:20, 15:20]
            B4 = self.matrix[15:20, 0:15]
            x_n4 = jnp.concatenate([x1.ravel(), x2.ravel(), x3.ravel()]).reshape(-1, 1)
            return obj_func(x4, x_n4, A4, B4)

        return [obj_func_1, obj_func_2, obj_func_3, obj_func_4]

    def constraints(self):
        def g0(x: VectorList):
            x1, x2, x3, x4 = x
            return jnp.reshape(x1[0] + 2*x1[1] - x1[2] + 3*x1[3] - 4*x1[4] - 2 + x2[1] - 3*x2[2],())

        def g1(x: VectorList):
            x1, x2, x3, x4 = x
            return jnp.reshape(-x2[0] + 3*x2[1] - 2*x2[2] + x2[3] +3*x2[4] - 4 + x3[0] - 3*x3[4] + 2*x4[2], ())

        def g2(x: VectorList):
            x1, x2, x3, x4 = x
            return jnp.reshape(-2*x3[0] + 3*x3[1] + x3[2] - x3[3] - 2*x3[4] - 4 + x1[0] - 4*x4[4], ())

        def g3(x: VectorList):
            x1, x2, x3, x4 = x
            return jnp.reshape(4*x4[0] - 2*x4[1] - 3*x4[2] - 6*x4[3] + 5*x4[4] - 3 + x1[0] + x1[1] - x2[0] - x2[1], ())

        return [g0, g1, g2, g3]
