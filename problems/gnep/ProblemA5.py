import jax.numpy as jnp
from solvers.schema import VectorList
from solvers.gnep_solver.BaseProblem import BaseProblem
from solvers.gnep_solver.BasePlayer import Player


class ProblemA5(BaseProblem):
    def known_solution(self):
        value_1 = [-0.00006229891126, 0.20279012064850, -0.00003469558295,
                       -0.00028322020054, 0.07258934064261,
                       0.02531280162415, -0.00007396699835]
        return value_1

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

        def obj_func_1(x: VectorList) -> jnp.array:
            x1 = x[0].reshape(-1,1)
            x2 = x[1]
            x3 = x[2]
            A1 = jnp.array([
                [20, 6, 0],
                [6, 6, -1],
                [0, -1, 8]
            ])

            B1 = jnp.array([[-1, -2, -4, -3], [0, -3, 0, -4], [0, 1, 9, 6]])
            b1 = jnp.array([[1], [-1], [1]])
            x_n1 = jnp.concatenate([x2.ravel(), x3.ravel()]).reshape(-1, 1)
            return obj_func(x1, x_n1, A1, B1, b1)

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1].reshape(-1, 1)
            x3 = x[2]
            A2 = jnp.array([
                [11, 1],
                [1, 7]
            ])
            B2 = jnp.array([[-1, 0, 0, -7, 4], [-2, -3, 1, 4, 11]])
            b2 = jnp.array([[1], [0]])
            x_n2 = jnp.concatenate([x1.ravel(), x3.ravel()]).reshape(-1, 1)
            return obj_func(x2, x_n2, A2, B2, b2)

        def obj_func_3(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2].reshape(-1, 1)
            A3 = jnp.array([
                [28, 14],
                [14, 29]
            ])
            B3 = jnp.array([[-4, 0, 9, -7, 4], [-3, -4, 6, 4, 11]])
            b3 = jnp.array([[-1], [2]])

            x_n3 = jnp.concatenate([x1.ravel(), x2.ravel()]).reshape(-1, 1)
            return obj_func(x3, x_n3,A3, B3, b3)

        return [obj_func_1, obj_func_2, obj_func_3]

    def constraints(self):
        def g0(x: VectorList):
            x1, x2, x3 = x
            return jnp.sum(x1) - 20

        def g1(x: VectorList):
            x1, x2, x3 = x
            return jnp.reshape(x1[0] + x1[1] - x1[2] - x2[0] + x3[1] - 5.0, ())

        def g2(x: VectorList):
            x1, x2, x3 = x
            return jnp.reshape(x2[0] - x2[1] - x1[1] - x1[2] + x3[0] - 7.0, ())

        def g3(x: VectorList):
            x1, x2, x3 = x
            return jnp.reshape(x3[1] - x1[0] - x1[2] + x2[0] - 4.0, ())

        return [g0, g1, g2, g3]
