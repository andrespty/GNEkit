import jax.numpy as jnp
from solvers.gnep_solver import VectorList, Player
from solvers.gnep_solver.BaseProblem import BaseProblem


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
        player_objective_functions = [i for i in range(10)]
        player_constraints = [[None], [0], [0], [0], [0, 1], [0, 1], [0], [0], [0], [0]]
        bounds_training = [(0.3, 0.5), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, 0.06), (0.01, 0.05)]
        return Player.batch_create(player_vector_sizes, player_objective_functions, player_constraints, bounds_training)

    def objectives(self):
        def logic_type_0(xi, S_total):
            return jnp.reshape((-xi / (S_total + 1e-9)) * (1 - S_total), ())

        def logic_type_1(xi, S_total):
            return jnp.reshape((-xi / (S_total + 1e-9)) * (1 - S_total)**2, ())

        def get_player_objective(x_list, p_idx):
            S_total = jnp.sum(jnp.concatenate(x_list))
            xi = x_list[p_idx]

            # Use your mapping to decide which logic to run
            mapping = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
            obj_type = mapping[p_idx]

            if obj_type == 0:
                return logic_type_0(xi, S_total)
            else:
                return logic_type_1(xi, S_total)

        return [lambda x, i=i: get_player_objective(x, i) for i in range(len(self.players))]

    def constraints(self):
        def g0(x: VectorList):
            return jnp.sum(jnp.concatenate(x)) - 1

        def g1(x: VectorList):
            return 0.99 - jnp.sum(jnp.concatenate(x))

        return [g0, g1]