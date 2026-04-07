from solvers.schema import VectorList
from solvers.dgbne_solver.BayesianPlayer import BayesianPlayer
from solvers.dgbne_solver.BayesianProblem import BayesianProblem
import jax.numpy as jnp


class RadarPowerGame(BayesianProblem):
    def __init__(self, players = None):
        self.N = 2
        self.T = 2

        self.sigma = 0.1
        self.I = 2.5
        self.P_max = 1.0

        self.C = jnp.array([0.5, 0.3])

        g_cross = jnp.zeros((self.N, self.N, self.T))
        g_cross = g_cross.at[0, 1, 0].set(1.0)
        g_cross = g_cross.at[0, 1, 1].set(4.0)
        g_cross = g_cross.at[1, 0, 0].set(1.0)
        g_cross = g_cross.at[1, 0, 1].set(4.0)
        self.G_cross = g_cross

        self.G_interf = jnp.array([[1.0,3.0], [2.0, 5.0]])
        super().__init__(players)


    def define_players(self):
        player_vector_sizes = [2, 2]
        player_type_values = [
            [1.5, 4.0], 
            [1.0, 3.0],  
        ]
        player_type_probs = [
            [0.4, 0.6], 
            [0.7, 0.3], 
        ]
        player_objective_functions = [0, 1]
        player_constraints = [[0,2], [1,3]]
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
        def ex_ante_payoff(i: int, actions):
            probs_i = jnp.asarray(self.players[i].type_probs).reshape(-1, 1)
            type_values = jnp.asarray(self.players[i].type_values).reshape(-1, 1)              # (T,1)
            a_i = actions[i].reshape(-1, 1)                     # (T,1)
            c_i = self.C[i]

            other_idx = self.get_others_idx(i)[0]
            probs_j = jnp.asarray(self.players[other_idx].type_probs).reshape(1, -1)   # (1,T)
            a_j = actions[other_idx].reshape(1, -1)                                     # (1,T)
            g_ji = self.G_cross[other_idx, i].reshape(1, -1)                            # (1,T)

            interference = probs_j * g_ji * a_j                                         # (1,T)
            denom = c_i * a_i + interference + self.sigma                              # (T,T)
            numer = type_values * a_i                                                            # (T,1)
            interim_by_type = jnp.sum(numer / (denom + 1e-15) * probs_j, axis=1, keepdims=True)
            return jnp.reshape(jnp.sum(probs_i * interim_by_type), ())

        def obj_func(x: VectorList, p_idx: int):
            action_matrices = self.split_profiles(x)
            return -ex_ante_payoff(p_idx, action_matrices)

        return [lambda x, i=i: obj_func(x, i) for i in range(len(self.players))]
    

    def constraints(self):
        def interim_g0(x: VectorList, i: int, t: int):
            action_matrices = self.split_profiles(x)
            actions = action_matrices[i].reshape(-1, 1)
            other_idx = self.get_others_idx(i)[0]
            expected_total = jnp.sum(jnp.asarray(self.players[other_idx].type_probs).reshape(-1,1) * self.G_interf[other_idx].reshape(-1,1) * action_matrices[other_idx])
            return jnp.reshape((actions*self.G_interf[i].reshape(-1,1) + expected_total - self.I)[t], ())

        return [lambda x, i=i: interim_g0(x, i, 0) for i in range(len(self.players))] + [lambda x, i=i: interim_g0(x, i, 1) for i in range(len(self.players))]
