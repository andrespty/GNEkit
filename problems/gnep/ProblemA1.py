import jax.numpy as jnp
from solvers.schema import VectorList
from solvers.gnep_solver import Player
from solvers.gnep_solver.BaseProblem import BaseProblem


class ProblemA1(BaseProblem):
    def known_solution(self):
        # Cleaned up to return standard lists of floats
        value_1 = [0.29923815223336,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805]
        return value_1

    def define_players(self):
        B=1
        player_vector_sizes = [1 for _ in range(10)]
        player_objective_functions = [i for i in range(10)]
        player_constraints = [[None], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        bounds_training = [(0.3, 0.5), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B)]
        return Player.batch_create(player_vector_sizes, player_objective_functions, player_constraints, bounds_training)

    def objectives(self):
        def obj_func(x):
            actions = jnp.concatenate(x)
            S_total = jnp.sum(actions)
            return jnp.reshape((-actions / S_total) * (1 - S_total), (-1,))

        return [lambda x, i=i: obj_func(x)[i] for i in range(len(self.players))]

    def constraints(self):
        def g0(x: VectorList):
            return jnp.sum(jnp.concatenate(x)) - 1

        return [g0]