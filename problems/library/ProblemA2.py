import jax.numpy as jnp
from gnep_solver import Vector, VectorList, Player
from gnep_solver.BaseProblem import BaseProblem

class ProblemA2(BaseProblem):
    def known_solution(self):
        # Cleaned up to return standard lists of floats
        value_1 = [0.29962894677774, 0.00997828224734, 0.00997828224734,
                   0.00997828224734, 0.59852469355630, 0.02187270661760,
                   0.00999093169361, 0.00999093169361, 0.00999093169361,
                   0.00999093169361]
        return value_1

    def define_players(self):
        B=1
        player_vector_sizes = [1 for _ in range(10)]
        player_objective_functions = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        player_constraints = [[None], [0], [0], [0], [0, 1], [0, 1], [0], [0], [0], [0]]
        bounds_training = [(0.3, 0.5), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, 0.06), (0.01, 0.05)]
        return Player.batch_create(player_vector_sizes, player_objective_functions, player_constraints, bounds_training)

    def objectives(self):
        def obj_func(xi, x_n):
            # Result of quadratic form is (1, 1). JAX needs a scalar for grad.
            S = xi + jnp.sum(jnp.concatenate(x_n))
            return jnp.reshape((-xi/S) * (1 - S), ())

        def obj_func2(xi, x_n):
            # Result of quadratic form is (1, 1). JAX needs a scalar for grad.
            S = xi + jnp.sum(jnp.concatenate(x_n))
            return jnp.reshape((-xi / S) * (1 - S) ** 2, ())

        return [obj_func, obj_func2]

    def constraints(self):
        def g0(x: VectorList):
            return jnp.sum(jnp.concatenate(x)) - 1

        def g1(x: VectorList):
            return 0.99 - jnp.sum(jnp.concatenate(x))

        return [g0, g1]