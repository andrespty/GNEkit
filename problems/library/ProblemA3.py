import jax.numpy as jnp
from gnep_solver import Vector, VectorList

class A3:
    # --- Matrix Constants
    A1 = jnp.array([[20, 5, 3], [5, 5, -5], [3, -5, 15]])
    A2 = jnp.array([[11, -1], [-1, 9]])
    A3 = jnp.array([[48, 39], [39, 53]])

    B1 = jnp.array([[-6, 10, 11, 20], [10, -4, -17, 9], [15, 8, -22, 21]])
    B2 = jnp.array([[20, 1, -3, 12, 1], [10, -4, 8, 16, 21]])
    B3 = jnp.array([[10, -2, 22, 12, 16], [9, 19, 21, -4, 20]])

    b1 = jnp.array([[1], [-1], [1]])
    b2 = jnp.array([[1], [0]])
    b3 = jnp.array([[-1], [2]])

    @staticmethod
    def paper_solution():
        # Cleaned up to return standard lists of floats
        value = [-0.38046562696258, -0.12266997083581, -0.99322817120517,
                 0.39034789080544, 1.16385412687962, 0.05039533464000,
                 0.01757740533460]
        return [value, value, value]

    @staticmethod
    def define_players():
        player_vector_sizes = [3, 2, 2]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0, 1], [2], [3]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A3.obj_func_1, A3.obj_func_2, A3.obj_func_3]

    @staticmethod
    def constraints():
        return [A3.g0, A3.g1, A3.g2, A3.g3]

    # Define Functions below
    @staticmethod
    def obj_func(x, x_ni, A, B, b):
        # Result of quadratic form is (1, 1). JAX needs a scalar for grad.
        return 0.5 * x.T @ A @ x + x.T @ (B @ x_ni + b)

    @staticmethod
    def obj_func_1(x: VectorList) -> jnp.ndarray:
        x1 = x[0].reshape(-1, 1)
        x_n1 = jnp.concatenate([x[1].ravel(), x[2].ravel()]).reshape(-1, 1)
        # Using jnp.reshape to ensure we return a 0-dim array (scalar)
        return jnp.squeeze(A3.obj_func(x1, x_n1, A3.A1, A3.B1, A3.b1))

    @staticmethod
    def obj_func_2(x: VectorList) -> jnp.ndarray:
        x2 = x[1].reshape(-1, 1)
        x_n2 = jnp.concatenate([x[0].ravel(), x[2].ravel()]).reshape(-1, 1)
        return jnp.squeeze(A3.obj_func(x2, x_n2, A3.A2, A3.B2, A3.b2))

    @staticmethod
    def obj_func_3(x: VectorList) -> jnp.ndarray:
        x3 = x[2].reshape(-1, 1)
        x_n3 = jnp.concatenate([x[0].ravel(), x[1].ravel()]).reshape(-1, 1)
        return jnp.squeeze(A3.obj_func(x3, x_n3, A3.A3, A3.B3, A3.b3))

    # --- JAX-Native Constraints ---
    @staticmethod
    def g0(x: VectorList):
        # sum(x1) - 20 <= 0
        return jnp.sum(x[0]) - 20.0

    @staticmethod
    def g1(x: VectorList):
        x1, x2, x3 = x
        # Ensure we use JAX indices and no .item() or [0]
        val = x1[0] + x1[1] - x1[2] - x2[0] + x3[1] - 5.0
        return jnp.reshape(val, ())  # Forces into a scalar

    @staticmethod
    def g2(x: VectorList):
        x1, x2, x3 = x
        val = x2[0] - x2[1] - x1[1] - x1[2] + x3[0] - 7.0
        return jnp.reshape(val, ())

    @staticmethod
    def g3(x: VectorList):
        x1, x2, x3 = x
        val = x3[1] - x1[0] - x1[2] + x2[0] - 4.0
        return jnp.reshape(val, ())