import numpy as np

# from tests.Problems_Unbounded.ProblemA16 import player_vector_sizes
from gne_solver.utils import construct_vectors


class A9a:
    K=8
    N=7

    @staticmethod
    def define_players():
        player_vector_sizes = [A9a.K for _ in range(A9a.N)]
        player_objective_functions = [0 for _ in range(A9a.N)]  # change to all 0s
        player_constraints = [[0] for _ in range(A9a.N)]
        bounds = [(0, 100) for _ in range(A9a.N * A9a.K)] + [(0,100)]
        bounds_training =  [(0, 100) for _ in range(A9a.N * A9a.K)] + [(0,100)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A9a.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [A9a.obj_func_der]

    @staticmethod
    def constraints():
        return [A9a.g0]

    @staticmethod
    def constraint_derivatives():
        return [A9a.g0_der]

    @staticmethod
    def obj_func(x):
        # x: numpy array (N, 1)
        # B: constant
        return sum(x)

    @staticmethod
    def obj_func_der(x):
        return np.array([1 for _ in range(A9a.N * A9a.K)]).reshape(-1,1)

    @staticmethod
    def g0_manual(x):
        """
        here for checks only
        """
        X = np.concatenate(x).reshape(-1,1)
        sigma = 0.3162
        values = []
        for vu in range(A9a.N):
            L = 8
            H = A9a.get_h_v(vu).reshape(A9a.N, A9a.K)
            hx = H[vu].reshape(-1, 1) * x[vu]

            H_ni = np.delete(H, vu, axis=0).reshape(-1,1)
            X_ni = np.delete(X, slice(vu * A9a.K, (vu + 1) * A9a.K), axis=0)

            hx_ni = np.sum((H_ni * X_ni).reshape(A9a.N-1, A9a.K), axis=0).reshape(-1,1)
            # print(hx_ni)
            constraint = np.log2( 1 + (hx/(sigma**2 + hx_ni)) ) - L
            values.append(np.sum(constraint).flatten())
        return np.concatenate(values).reshape(-1,1)

    @staticmethod
    def g0(x):
        K, N = A9a.K, A9a.N
        sigma = 0.3162
        L = 8

        # Flatten input

        X = np.concatenate(x).reshape(N * K, 1)  # (K*N, 1)
        H_all = np.stack([A9a.get_h_v(vu).reshape(N, K) for vu in range(N)], axis=0)  # shape: (N, N, K)

        # Reshape X to (N, K, 1) for broadcasting
        X_split = np.concatenate(x).reshape(N, K, 1)  # shape: (N, K, 1)

        # Compute elementwise product: shape (N, N, K, 1)
        HX_all = H_all[..., :, np.newaxis] * X_split[np.newaxis, ...]

        # Split signal vs interference:
        # Signal for vu is at H_all[vu, vu] * x[vu] → shape (N, K, 1)
        signal = np.array([HX_all[i, i] for i in range(N)])  # shape (N, K, 1)

        # Sum all players' contributions (axis=1), then subtract self
        total_hx = np.sum(HX_all, axis=1)  # shape: (N, K, 1)
        interference = total_hx - signal  # shape: (N, K, 1)

        # Compute constraint
        constraint = np.log2(1 + (signal / (sigma ** 2 + interference))) - L  # shape: (N, K, 1)

        # Sum over K dimensions per player
        return np.sum(constraint, axis=1).reshape(-1,1)  # shape: (N,)

    @staticmethod
    def g0_der(x):
        sigma = 0.3162
        result = []
        for vu in range(A9a.N):
            H_v = A9a.get_h_v(vu).reshape(A9a.N, A9a.K)
            players = x.reshape(A9a.N, A9a.K)
            H_vv = H_v[vu].reshape(-1, 1)
            D = (sigma ** 2) + np.sum(players * H_v, axis=0).reshape(-1,1)
            grad = (1/np.log(2))*(H_vv / D)
            result.append(grad.ravel())
        result = np.concatenate(result).reshape(-1, 1)
        return result

    @staticmethod
    def get_h_v(vu):
        if vu > A9a.N - 1:
            print('Cant be done')
            return 1
        H = A9a.h_matrix()
        # padding = mu * A9a.K
        return H[:, vu].reshape(-1,1)

    @staticmethod
    def h_matrix():
        return np.array([
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
#
primal_ip = [1. for _ in range(7*8)]

x = np.vstack(primal_ip).reshape(-1,1)
player_vector_sizes = [A9a.K for _ in range(A9a.N)]
print(A9a.g0_manual(construct_vectors(x,player_vector_sizes)))
print(A9a.g0(construct_vectors(x,player_vector_sizes)))
print(A9a.g0_manual(x).shape)
