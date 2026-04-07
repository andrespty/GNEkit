from solvers.schema import VectorList
from solvers.dgbne_solver.BayesianPlayer import BayesianPlayer
from solvers.dgbne_solver.BayesianProblem import BayesianProblem
import jax.numpy as jnp


class QuadraticGame(BayesianProblem):
    def __init__(self, players = None):
        self.C = 1.0
        self.alpha = 9.0
        self.gamma = 5.0
        self.delta = 11.0
        super().__init__(players)


    def define_players(self):
        player_vector_sizes = [2, 2]
        player_type_values = [
            [3.0, 7.0],  
            [2.0, 8.0], 
        ]
        player_type_probs = [
            [0.8, 0.2],  
            [0.4, 0.6],  
        ]
        player_objective_functions = [0, 1]
        player_constraints = [[0, 1], [2,3]]
        bounds = [(0,1) for _ in range(2)]
        return BayesianPlayer.batch_create(
            player_vector_sizes,
            player_type_values,
            player_objective_functions,
            player_constraints,
            bounds,
            type_probs=player_type_probs
        )

    def objectives(self):
        def ex_ante_utility_1(actions):
            probs = jnp.asarray(self.players[0].type_probs).reshape(-1, 1)
            type_vals =jnp.asarray(self.players[0].type_values).reshape(-1, 1)
            s_i = actions[0].reshape(-1, 1)
            interim_ui = - s_i * type_vals + (s_i - 1) ** 2
            return jnp.reshape(probs.T @ interim_ui, ())

        def ex_ante_utility_2(actions):
            probs = jnp.asarray(self.players[1].type_probs).reshape(-1,1)
            type_vals = jnp.asarray(self.players[1].type_values).reshape(-1, 1)
            s_i = actions[1].reshape(-1, 1)
            interim_ui = - s_i * type_vals + (s_i - 0.5) ** 2
            return jnp.reshape(probs.T @ interim_ui, ())

        return [ex_ante_utility_1, ex_ante_utility_2]

    def constraints(self):
        def g1_const(actions, t):
            s_1 = actions[0].reshape(-1, 1)
            type_vals =jnp.asarray(self.players[0].type_values).reshape(-1, 1)
            expected_s2 = self.expected_other_actions(actions, 0)[0].reshape(-1, 1)
            total_exp = jnp.sum(jnp.stack(expected_s2), axis=0)
            return jnp.reshape((s_1*type_vals + total_exp - 1)[t], ())

        def g2_const(actions, t):
            s_2 = actions[1].reshape(-1, 1)
            type_vals =jnp.asarray(self.players[1].type_values).reshape(-1, 1)
            expected_s1 = self.expected_other_actions(actions, 1)[0].reshape(-1, 1)
            total_exp = jnp.sum(jnp.stack(expected_s1), axis=0)
            return jnp.reshape((s_2*type_vals + total_exp - 1)[t], ())

        return [lambda x: g1_const(x, 0), lambda x: g1_const(x, 1), lambda x: g2_const(x, 0), lambda x: g2_const(x, 1)]
