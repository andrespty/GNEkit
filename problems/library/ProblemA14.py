import jax.numpy as jnp
from solvers.gnep_solver import VectorList
from solvers.gnep_solver.BaseProblem import BaseProblem
from solvers.gnep_solver.BasePlayer import Player

class ProblemA14(BaseProblem):
    def known_solution(self):
        value_1 = [ 0.08999991899425, 0.08999991899426, 0.08999991899425,
                    0.08999991899425, 0.08999991899425, 0.08999991899425,
                    0.08999991899425, 0.08999991899426, 0.08999991899425,
                    0.08999991899425]
        return value_1

    def define_players(self):
        B = 1
        player_vector_sizes = [1 for _ in range(10)]
        player_objective_functions = [i for i in range(10)]
        player_constraints = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        bounds_training = [(0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B),
                           (0.01, B), (0.01, B)]
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