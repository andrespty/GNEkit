import jax.numpy as jnp
from solvers.schema import VectorList
from solvers.gnep_solver.BaseProblem import BaseProblem
from solvers.gnep_solver.BasePlayer import Player


class ProblemA6(BaseProblem):
    def known_solution(self):
        value_1 = [0.99987722673822, 2.31570964703584, 0.99989251930167, 1.31499923583926, 0.99989852480755,
                   0.99992298465841, 1.09709158271764]
        return value_1

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
            # Result of quadratic form is (1, 1). JAX needs a scalar for grad.
            return jnp.reshape(0.5 * x.T @ A @ x + x.T @ (B @ x_ni + b), ())

        def obj_func_1(x: VectorList) -> jnp.array:
            x1 = x[0].reshape(-1,1)
            x2 = x[1].reshape(-1,)
            x3 = x[2].reshape(-1,)
            A1 = jnp.array([
                [20 + (x2[0] ** 2), 5, 3],
                [5, 5 + (x2[1] ** 2), -5],
                [3, -5, 15]
            ])

            B1 = jnp.array([[-2, 0, 1, 2], [1, -4, -7, 9], [3, 8, 22, 21]])
            b1 = jnp.array([[1], [-2], [-3]])
            x_n1 = jnp.concatenate([x2.ravel(), x3.ravel()]).reshape(-1, 1)
            return obj_func(x1, x_n1, A1, B1, b1)

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x1 = x[0].reshape(-1,)
            x2 = x[1].reshape(-1, 1)
            x3 = x[2].reshape(-1, )
            A2 = jnp.array([
                [11 + (x3[0] ** 2), -1],
                [-1, 9]
            ])
            B2 = jnp.array([[-2, 1, -3, 12, -1], [0, -4, 8, 16, 21]])
            b2 = jnp.array([[1], [2]])
            x_n2 = jnp.concatenate([x1.ravel(), x3.ravel()]).reshape(-1, 1)
            return obj_func(x2, x_n2, A2, B2, b2)

        def obj_func_3(x: VectorList) -> jnp.ndarray:
            x1 = x[0].reshape(-1,)
            x2 = x[1].reshape(-1, )
            x3 = x[2].reshape(-1, 1)
            A3 = jnp.array([
                [48, 39],
                [39, 53 + (x1[0] ** 2)]
            ])
            B3 = jnp.array([[1, -7, 22, -12, 16], [2, -9, 21, 1, 21]])
            b3 = jnp.array([[1], [-2]])

            x_n3 = jnp.concatenate([x1.ravel(), x2.ravel()]).reshape(-1, 1)
            return obj_func(x3, x_n3,A3, B3, b3)

        return [obj_func_1, obj_func_2, obj_func_3]

    def constraints(self):
        def g0(x: VectorList):
            x1 = x[0].reshape(-1, )
            x2 = x[1].reshape(-1, )
            x3 = x[2].reshape(-1,)
            return jnp.sum(x1) - 20

        def g1(x: VectorList):
            x1 = x[0].reshape(-1, )
            x2 = x[1].reshape(-1, )
            x3 = x[2].reshape(-1, )
            return jnp.reshape(x1[0] + x1[1] - x1[2] - x2[0] + x3[1] - 3.7, ())

        def g2(x: VectorList):
            x1 = x[0].reshape(-1, )
            x2 = x[1].reshape(-1, )
            x3 = x[2].reshape(-1, )
            return jnp.reshape((x1[0] ** 4) + x3[0] * x1[1] - x2[0] - 2, ())

        def g3(x: VectorList):
            x1 = x[0].reshape(-1, )
            x2 = x[1].reshape(-1, )
            x3 = x[2].reshape(-1, )
            return jnp.reshape(x2[0] - x2[1] - x1[1] - x1[2] + x3[0] - 7.0, ())

        def g4(x: VectorList):
            x1 = x[0].reshape(-1, )
            x2 = x[1].reshape(-1, )
            x3 = x[2].reshape(-1, )
            return jnp.reshape((x2[0] - 2) ** 2 + x2[1] ** 2 - x1[0] ** 2 - 0.75, ())

        def g5(x: VectorList):
            x1 = x[0].reshape(-1, )
            x2 = x[1].reshape(-1, )
            x3 = x[2].reshape(-1, )
            return jnp.reshape(x3[1] - x1[0] - x1[2] + x2[0] - 4.0, ())

        def g6(x: VectorList):
            x1 = x[0].reshape(-1, )
            x2 = x[1].reshape(-1, )
            x3 = x[2].reshape(-1, )
            return jnp.reshape(2*(x3[0] ** 2) - (x3[1] - 2) ** 2 - x2[0] * x3[0] - 1.5, ())

        return [g0, g1, g2, g3, g4, g5, g6]
