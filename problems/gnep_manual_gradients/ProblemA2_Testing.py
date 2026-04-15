import jax.numpy as jnp
from solvers.gnep_solver import VectorList
from tester_solver.ManualBaseProblem import ManualBaseProblem
from solvers.gnep_solver.BasePlayer import Player


class ProblemA2_Manual(ManualBaseProblem):
    def known_solution(self):
        value = [0.29962894677774, 0.00997828224734, 0.00997828224734,
                   0.00997828224734, 0.59852469355630, 0.02187270661760,
                   0.00999093169361, 0.00999093169361, 0.00999093169361,
                   0.00999093169361]
        return value

    def define_players(self):
        player_vector_sizes = [1 for _ in range(10)]
        player_objective_functions = [i for i in range(10)]
        player_constraints = [[None], [0], [0], [0], [0, 1], [0, 1], [0], [0], [0], [0]]
        B=1
        bounds = [(0.3, 0.5), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, 0.06), (0.01, 0.05)]
        return Player.batch_create(
            player_vector_sizes,
            player_objective_functions,
            player_constraints,
            bounds
        )

    def objectives(self):
        def logic_type_0(xi, S_total):
            return jnp.reshape((-xi / S_total) * (1 - S_total), ())

        def logic_type_1(xi, S_total):
            return jnp.reshape((-xi / S_total) * (1 - S_total) ** 2, ())

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

    def objectives_der(self):
        def logic_type_0(x, S):
            return jnp.reshape(1 + (x - S)/(S**2), ())

        def logic_type_1(xi, S):
            return jnp.reshape(xi * ((1-S) * (S+1)/(S**2)) - ((1-S)**2)/S, ())

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

        return [lambda x, i=i: {i:get_player_objective(x, i)} for i in range(len(self.players))]

    def constraints(self):
        def g0(x: VectorList):
            return jnp.sum(jnp.concatenate(x)) - 1

        def g1(x: VectorList):
            return 0.99 - jnp.sum(jnp.concatenate(x))

        return [g0, g1]

    def constraints_der(self):
        def g0_der(x):
            dx1 = jnp.array([1.])
            return [dx1 for _ in range(len(self.players))]

        def g1_der(x):
            dx1 = jnp.array([-1.])
            return [dx1 for _ in range(len(self.players))]

        return [g0_der, g1_der]