from solvers.schema import VectorList
from solvers.dgbne_solver.BayesianPlayer import BayesianPlayer
from solvers.dgbne_solver.BayesianProblem import BayesianProblem
import jax.numpy as jnp


class PayoffMatrixGame(BayesianProblem):
    def __init__(self, players = None):
        self.eps_s = 0.9
        self.eps_u = 0.7
        self.delta_u = 0.3
        self.delta_s = 0.2
        self.e_s1 = 0.3
        self.e_u1 = 0.2
        self.e_u2 = 0.2
        self.e_s2 = 0.1
        
        self.phi = 1
        self.r = 1

        self.big = 10000
        super().__init__(players)


    def define_players(self):
        player_vector_sizes = [4, 4]
        player_action_size_per_type = [2,2]
        player_type_values = [
            [0, 1],  # Player 1: malicious (0) or regular (1)
            [0, 1],  # Player 2: unsophisticated (0) or sophisticated (1)
        ]
        player_type_probs = [
            [0.5, 0.5],  
            [0.5, 0.5],  
        ]
        player_objective_functions = [0, 1]
        player_constraints = [[0,1, 4, 5], [2,3,6,7]]
        bounds = [(0,1) for _ in range(2)]
        return BayesianPlayer.batch_create(
            player_vector_sizes,
            player_type_values,
            player_objective_functions,
            player_constraints,
            bounds,
            type_probs=player_type_probs,
            action_size_per_type=player_action_size_per_type
        )

    def objectives(self):
        def matrix_1(p_idx): # 0,0
            p1 = jnp.array([
                [self.phi * ( 1- 2*self.eps_u) - self.e_u1, self.phi - self.e_u1], 
                [-self.phi, -self.phi]
            ])
            p2 = jnp.array([
                [self.phi * (2 * self.eps_u - 1) - self.e_u2, -self.phi], 
                [self.phi * (1 - self.delta_u)- self.e_u2, self.phi]
            ])
            return [p1, p2][p_idx]
        
        def matrix_2(p_idx): # 0,1
            p1 = jnp.array([
                [(self.phi + self.r) * ( 1- 2*self.eps_s) - self.e_s1, self.phi - self.e_s1], 
                [-self.phi, -self.phi]
            ])
            p2 = jnp.array([
                [(self.phi + self.r) * (2 * self.eps_s - 1) - self.e_s2, -self.phi], 
                [self.phi * (1 - self.delta_s)- self.e_s1, self.phi]
            ])
            return [p1, p2][p_idx]
        
        def matrix_3(p_idx): # 1,0
            p1 = jnp.array([
                [-self.big, -self.big],
                [-self.phi, -self.phi]
            ])
            p2 = jnp.array([
                [self.big, self.big],
                [self.phi * (1 - self.delta_u)- self.e_u2, self.phi]
            ])
            return [p1, p2][p_idx]
        
        def matrix_4(p_idx): # 1,1
            p1 = jnp.array([
                [-self.big, -self.big],
                [-self.phi, -self.phi]
            ])
            p2 = jnp.array([
                [self.big, self.big],
                [self.phi * (1 - self.delta_s)- self.e_s2, self.phi]
            ])
            return [p1, p2][p_idx]

        def ex_ante_utility(actions, p_idx):
            a_1 = jnp.repeat(actions[0].reshape(-1, 2), 2, axis=0)
            a_2 = jnp.tile(actions[1].reshape(-1, 2), (2, 1))
            m1 = matrix_1(p_idx) # 0,0
            m2 = matrix_2(p_idx) # 0,1
            m3 = matrix_3(p_idx) # 1,0
            m4 = matrix_4(p_idx) # 1,1

            M = jnp.array([m1, m2, m3, m4])
            
            result = (M @ a_2[..., None]).squeeze(-1)
            exp = jnp.sum(a_1 * result, axis=1).reshape(-1, 1)

            probs_p1 = jnp.asarray(self.players[0].type_probs).reshape(-1, 1)
            probs_p2 = jnp.asarray(self.players[1].type_probs).reshape(-1, 1)
            P = (probs_p1[:, None] * probs_p2[None, :]).reshape(-1, 1)
            return -jnp.sum(P * exp)

        return [lambda x: ex_ante_utility(x, 0), lambda x: ex_ante_utility(x, 1)]

    def constraints(self):
        def g1_const(actions, t, p_idx):
            s = actions[p_idx].reshape(-1, 2)
            probs_actions = s.sum(axis=1).reshape(-1, 1)
            return jnp.reshape(probs_actions[t] - 1, ())

        def g2_const(actions, t, p_idx):
            s = actions[p_idx].reshape(-1, 2)
            probs_actions = s.sum(axis=1).reshape(-1, 1)
            return jnp.reshape(1-probs_actions[t], ())

        return [
            lambda x: g1_const(x, 0, 0), 
            lambda x: g1_const(x, 1, 0),  
            lambda x: g1_const(x, 0, 1), 
            lambda x: g1_const(x, 1, 1),

            lambda x: g2_const(x, 0, 0), 
            lambda x: g2_const(x, 1, 0),  
            lambda x: g2_const(x, 0, 1), 
            lambda x: g2_const(x, 1, 1),
        ]
