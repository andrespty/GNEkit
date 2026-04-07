from solvers.schema import VectorList
from solvers.dgbne_solver.BayesianPlayer import BayesianPlayer
from solvers.dgbne_solver.BayesianProblem import BayesianProblem
import jax.numpy as jnp


class AllocationGame(BayesianProblem):
    def __init__(self, players = None):
        self.C = 1.0
        self.alpha = 9.0
        self.gamma = 5.0
        self.delta = 11.0
        super().__init__(players)


    def define_players(self):
        player_vector_sizes = [2, 2, 2]
        player_type_values = [
            [5.73, 10.64],  # vehicle 1: [5.73, 10.64]
            [3.82, 7.09],  # vehicle 2: [3.82, 7.09]
            [4.77, 8.86],  # vehicle 3: [4.77, 8.86]
        ]
        player_type_probs = [
            [0.3, 0.7],  # vehicle 1: [5.73, 10.64]
            [0.7, 0.3],  # vehicle 2: [3.82, 7.09]
            [0.7, 0.3],  # vehicle 3: [4.77, 8.86]
        ]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0,3], [1,4], [2,5]]
        bounds = [(0,1) for _ in range(3)]
        return BayesianPlayer.batch_create(
            player_vector_sizes,
            player_type_values,
            player_objective_functions,
            player_constraints,
            bounds,
            type_probs=player_type_probs
        )

    def objectives(self):
        def U(a, t):
            a = jnp.reshape(a, (-1,1))
            t = jnp.reshape(t, (-1,1))
            return t * jnp.log(1.0 + self.gamma * a) - self.delta * a
        
        def ex_ante_payoff(i, actions):
            probs = jnp.asarray(self.players[i].type_probs).reshape(-1, 1)
            type_vals =jnp.asarray(self.players[i].type_values).reshape(-1, 1)
            payoff = probs.T @ U(actions[i], type_vals)
            return jnp.reshape(payoff, ())
            
        def obj_func(x: VectorList,p_idx: int):
            action_matrices = self.split_profiles(x)
            return -ex_ante_payoff(p_idx, action_matrices)

        return [lambda x, i=i: obj_func(x, i) for i in range(len(self.players))]

    def constraints(self):
        def interim_g0(x: VectorList, i: int, t: int):
            action_matrices = self.split_profiles(x)
            actions = action_matrices[i]
            expected_others = self.expected_other_actions(x, i) 
            expected_total = jnp.sum(jnp.stack(expected_others), axis=0)
            return jnp.reshape(actions[t] + expected_total - self.C, ())

        return [lambda x, i=i: interim_g0(x, i, 0) for i in range(len(self.players))] + [lambda x, i=i: interim_g0(x, i, 1) for i in range(len(self.players))]
