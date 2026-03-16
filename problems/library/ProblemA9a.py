import jax.numpy as jnp
from gnep_solver import Vector, VectorList
from gnep_solver.BaseProblem import BaseProblem
from gnep_solver.Player import Player
from typing import List
from jax import debug

class ProblemA9a(BaseProblem):
    def __init__(self):
        self.K = 8
        self.N = 7
        self.h_matrix = jnp.array([
    [0.0362, 0.0008, 0.0018, 0.0022, 0.0085, 0.0008, 0.0060],
    [0.2211, 0.0003, 0.0014, 0.0074, 0.0005, 0.0037, 0.0006],
    [0.3356, 0.0032, 0.0027, 0.0043, 0.0007, 0.0012, 0.0003],
    [0.0077, 0.0039, 0.0032, 0.0073, 0.0089, 0.0006, 0.0002],
    [0.0036, 0.0019, 0.0032, 0.0014, 0.0017, 0.0038, 0.0046],
    [0.1811, 0.0013, 0.0014, 0.0024, 0.0010, 0.0000, 0.0121],
    [0.0442, 0.0093, 0.0002, 0.0036, 0.0096, 0.0031, 0.0042],
    [0.3507, 0.0017, 0.0004, 0.0042, 0.0034, 0.0061, 0.0129],
    [0.0145, 0.1487, 0.0001, 0.0013, 0.0001, 0.0005, 0.0008],
    [0.0035, 0.1341, 0.0021, 0.0001, 0.0000, 0.0001, 0.0000],
    [0.0042, 0.3074, 0.0024, 0.0002, 0.0001, 0.0004, 0.0004],
    [0.0006, 0.3358, 0.0017, 0.0008, 0.0002, 0.0004, 0.0022],
    [0.0007, 0.0083, 0.0045, 0.0003, 0.0002, 0.0000, 0.0019],
    [0.0024, 0.2214, 0.0011, 0.0000, 0.0001, 0.0006, 0.0016],
    [0.0034, 0.1261, 0.0079, 0.0002, 0.0002, 0.0007, 0.0001],
    [0.0032, 0.0288, 0.0059, 0.0004, 0.0001, 0.0001, 0.0014],
    [0.0040, 0.0051, 0.1475, 0.0042, 0.0002, 0.0000, 0.0005],
    [0.0030, 0.0002, 0.0300, 0.0003, 0.0000, 0.0001, 0.0001],
    [0.0046, 0.0014, 0.2207, 0.0089, 0.0000, 0.0001, 0.0003],
    [0.0059, 0.0005, 0.3028, 0.0024, 0.0002, 0.0001, 0.0002],
    [0.0037, 0.0002, 0.0828, 0.0008, 0.0003, 0.0001, 0.0001],
    [0.0023, 0.0003, 0.0167, 0.0009, 0.0014, 0.0001, 0.0002],
    [0.0009, 0.0001, 0.0956, 0.0001, 0.0005, 0.0000, 0.0003],
    [0.0006, 0.0028, 0.0566, 0.0018, 0.0000, 0.0001, 0.0002],
    [0.0005, 0.0002, 0.0004, 0.0991, 0.0057, 0.0000, 0.0003],
    [0.0000, 0.0000, 0.0016, 0.0377, 0.0003, 0.0002, 0.0003],
    [0.0014, 0.0001, 0.0002, 0.8100, 0.0035, 0.0001, 0.0000],
    [0.0019, 0.0002, 0.0016, 0.3608, 0.0141, 0.0000, 0.0000],
    [0.0013, 0.0005, 0.0014, 0.0617, 0.0014, 0.0006, 0.0000],
    [0.0016, 0.0003, 0.0004, 0.0723, 0.0016, 0.0005, 0.0001],
    [0.0007, 0.0000, 0.0028, 0.3314, 0.0126, 0.0002, 0.0000],
    [0.0011, 0.0001, 0.0002, 0.0953, 0.0026, 0.0000, 0.0001],
    [0.0001, 0.0000, 0.0001, 0.0009, 0.0993, 0.0015, 0.0001],
    [0.0002, 0.0003, 0.0003, 0.0013, 0.1702, 0.0210, 0.0000],
    [0.0007, 0.0000, 0.0002, 0.0024, 1.1062, 0.0033, 0.0004],
    [0.0012, 0.0002, 0.0006, 0.0014, 0.2817, 0.0091, 0.0004],
    [0.0003, 0.0003, 0.0004, 0.0008, 0.5208, 0.0024, 0.0001],
    [0.0001, 0.0001, 0.0001, 0.0006, 0.3959, 0.0010, 0.0001],
    [0.0004, 0.0001, 0.0011, 0.0009, 0.1057, 0.0029, 0.0004],
    [0.0000, 0.0001, 0.0000, 0.0020, 0.0932, 0.0058, 0.0006],
    [0.0002, 0.0010, 0.0006, 0.0001, 0.0001, 0.3905, 0.0028],
    [0.0004, 0.0001, 0.0000, 0.0001, 0.0009, 0.1322, 0.0032],
    [0.0001, 0.0009, 0.0001, 0.0003, 0.0018, 0.3691, 0.0130],
    [0.0004, 0.0007, 0.0003, 0.0001, 0.0018, 0.1142, 0.0022],
    [0.0009, 0.0010, 0.0002, 0.0001, 0.0019, 0.1127, 0.0046],
    [0.0017, 0.0014, 0.0000, 0.0001, 0.0022, 0.1665, 0.0015],
    [0.0009, 0.0001, 0.0000, 0.0002, 0.0015, 0.0066, 0.0042],
    [0.0004, 0.0006, 0.0005, 0.0001, 0.0022, 0.2389, 0.0026],
    [0.0008, 0.0054, 0.0004, 0.0001, 0.0001, 0.0040, 0.1110],
    [0.0005, 0.0013, 0.0008, 0.0004, 0.0000, 0.0006, 0.0834],
    [0.0033, 0.0063, 0.0001, 0.0004, 0.0000, 0.0013, 0.4948],
    [0.0012, 0.0079, 0.0000, 0.0000, 0.0001, 0.0029, 0.0355],
    [0.0002, 0.0095, 0.0002, 0.0002, 0.0001, 0.0003, 0.3045],
    [0.0010, 0.0018, 0.0000, 0.0002, 0.0003, 0.0009, 0.5359],
    [0.0003, 0.0045, 0.0002, 0.0001, 0.0002, 0.0007, 0.0328],
    [0.0005, 0.0035, 0.0006, 0.0000, 0.0008, 0.0008, 0.2950]
])
        super().__init__()

    def define_players(self):
        player_vector_sizes = [self.K for _ in range(self.N)]
        player_objective_functions = [i for i in range(self.N)]  # change to all 0s
        player_constraints = [[i] for i in range(self.N)]
        bounds = [(0, 100) for _ in range(self.N)]
        return Player.batch_create(
            player_vector_sizes,
            player_objective_functions,
            player_constraints,
            bounds
        )

    def objectives(self):
        def obj_func(x, i):
            x1, x2, x3, x4, x5, x6, x7 = x
            return jnp.sum(x[i])

        return [lambda x: obj_func(x, i) for i in range(self.N)]

    def constraints(self):
        def get_h_v(vu):
            if vu > self.N - 1:
                print('Cant be done')
                return 1
            H = self.h_matrix
            # padding = mu * A9a.K
            return H[:, vu].reshape(-1, 1)

        def g0(x):
            K, N = self.K, self.N
            sigma = 0.3162
            L = 8

            # Flatten input

            X = jnp.concatenate(x).reshape(N * K, 1)  # (K*N, 1)
            H_all = jnp.stack([get_h_v(vu).reshape(N, K) for vu in range(N)], axis=0)  # shape: (N, N, K)

            # Reshape X to (N, K, 1) for broadcasting
            X_split = jnp.concatenate(x).reshape(N, K, 1)  # shape: (N, K, 1)

            # Compute elementwise product: shape (N, N, K, 1)
            HX_all = H_all[..., :, jnp.newaxis] * X_split[jnp.newaxis, ...]

            # Split signal vs interference:
            # Signal for vu is at H_all[vu, vu] * x[vu] → shape (N, K, 1)
            signal = jnp.array([HX_all[i, i] for i in range(N)])  # shape (N, K, 1)

            # Sum all players' contributions (axis=1), then subtract self
            total_hx = jnp.sum(HX_all, axis=1)  # shape: (N, K, 1)
            interference = total_hx - signal  # shape: (N, K, 1)

            # Compute constraint
            constraint = jnp.log2(1 + (signal / (sigma ** 2 + interference)))   # shape: (N, K, 1)

            # Sum over K dimensions per player
            return jnp.sum(constraint, axis=1).reshape(-1, 1) - L

        unrolled_constraints = []
        for i in range(self.N):
            # Each lambda returns exactly ONE scalar for ONE player
            unrolled_constraints.append(
                lambda x, idx=i: jnp.reshape(g0(x)[idx], ())
            )
        return unrolled_constraints