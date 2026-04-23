from solvers.schema import VectorList
from solvers.dgbne_solver.BayesianPlayer import BayesianPlayer
from solvers.dgbne_solver.BayesianProblem import BayesianProblem
import jax.numpy as jnp
import numpy as np

def make_exp_sum_nodes(R: int, x_min: float = 0.2, x_max: float = 30.0):
    """
    Construct nodes t_l and weights c_l such that
        1/x ≈ sum_l c_l * exp(-t_l * x)  for x in [x_min, x_max]
    via trapezoidal rule on a log-scaled grid of the Laplace integral.
    """
    # Choose s range so that exp(-t*x) is small at both endpoints
    # Need t_min * x_max to be large (say ≥ 20) → t_min ≤ 20/x_max
    # Need t_max * x_min to be large → t_max ≥ 20/x_min
    s_min = np.log(1.0 / x_max) - 10.0   # extra margin
    s_max = np.log(20.0 / x_min)

    s = np.linspace(s_min, s_max, R)
    h = (s_max - s_min) / (R - 1)
    # Trapezoidal weights (endpoints get h/2, interior gets h)
    w = np.full(R, h)
    w[0] *= 0.5
    w[-1] *= 0.5
    
    t = np.exp(s)
    c = w * np.exp(s)  # Jacobian of t = e^s
    return jnp.asarray(t), jnp.asarray(c)


class RadarPowerGameTT(BayesianProblem):
    def __init__(self, players=None, R: int = 40):
        self.N = 15
        self.T = 3
        self.R = R

        self.sigma = 0.1
        self.I = 30.0
        self.P_max = 1.0

        # Three groups of 5 players each.
        c_by_group = jnp.array([0.3, 0.5, 0.8])
        self.C = jnp.concatenate([
            jnp.full((5,), c_by_group[0]),
            jnp.full((5,), c_by_group[1]),
            jnp.full((5,), c_by_group[2]),
        ])  # (15,)

        types_by_group = jnp.array([
            [0.5, 1.0, 1.5],
            [1.5, 2.5, 3.5],
            [3.0, 4.5, 6.0],
        ])
        self.type_values_array = jnp.concatenate([
            jnp.tile(types_by_group[0], (5, 1)),
            jnp.tile(types_by_group[1], (5, 1)),
            jnp.tile(types_by_group[2], (5, 1)),
        ], axis=0)  # (15, 3)

        probs_by_group = jnp.array([
            [0.5, 0.3, 0.2],
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5],
        ])
        self.type_probs_array = jnp.concatenate([
            jnp.tile(probs_by_group[0], (5, 1)),
            jnp.tile(probs_by_group[1], (5, 1)),
            jnp.tile(probs_by_group[2], (5, 1)),
        ], axis=0)  # (15, 3)

        # Cross-interference rule: g_{ji}^{(theta_j)} = theta_j (independent of i)
        # G_cross[j, i, t] = type value of player j at type index t
        self.G_cross = jnp.broadcast_to(
            self.type_values_array[:, None, :],
            (self.N, self.N, self.T)
        )  # (15, 15, 3)

        # Constraint coefficient rule: tilde_g_k^{(theta_k)} = theta_k
        self.G_interf = self.type_values_array  # (15, 3)

        # Exponential-sum nodes and weights for 1/x approximation
        self.t_nodes, self.c_weights = make_exp_sum_nodes(R)  # (R,), (R,)

        super().__init__(players)

    def define_players(self):
        player_vector_sizes = [self.T] * self.N
        player_type_values = [self.type_values_array[i].tolist() for i in range(self.N)]
        player_type_probs = [self.type_probs_array[i].tolist() for i in range(self.N)]
        player_objective_functions = list(range(self.N))
        player_constraints = [
            [i + t * self.N for t in range(self.T)]
            for i in range(self.N)
        ]
        bounds = [(0, 1) for _ in range(self.N)]
        return BayesianPlayer.batch_create(
            player_vector_sizes,
            player_type_values,
            player_objective_functions,
            player_constraints,
            bounds,
            type_probs=player_type_probs,
        )

    def _gamma_other(self, k: int, i: int, a_k: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Γ_k^{(i)} for k != i: an R-vector with entries
            Γ_k^{(i)}[ℓ] = sum_{θ_k} η_k(θ_k) * exp(-t_ℓ * g_{ki}^{(θ_k)} * a_k^{(θ_k)})

        Args:
            k: index of the other player
            i: index of the player whose utility we are computing
            a_k: shape (T,), player k's strategy

        Returns:
            shape (R,)
        """
        probs_k = self.type_probs_array[k]                # (T,)
        g_ki = self.G_cross[k, i]                         # (T,)
        # exponent matrix: shape (R, T) with entries -t_ℓ * g_{ki}^{(θ_k)} * a_k^{(θ_k)}
        exponent = -self.t_nodes[:, None] * (g_ki * a_k)[None, :]  # (R, T)
        # weighted sum over θ_k
        return jnp.sum(probs_k[None, :] * jnp.exp(exponent), axis=1)  # (R,)

    def _gamma_self(self, i: int, a_i: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Γ_i^{(i)}: an R-vector with entries
            Γ_i^{(i)}[ℓ] = sum_{θ_i} η_i(θ_i) * θ_i * a_i^{(θ_i)} * exp(-t_ℓ * c_i * a_i^{(θ_i)})

        Args:
            i: player index
            a_i: shape (T,)

        Returns:
            shape (R,)
        """
        probs_i = self.type_probs_array[i]                # (T,)
        types_i = self.type_values_array[i]               # (T,)
        c_i = self.C[i]
        # exponent: shape (R, T)
        exponent = -self.t_nodes[:, None] * (c_i * a_i)[None, :]  # (R, T)
        # numerator factor: shape (T,)
        num_factor = types_i * a_i                                # (T,)
        # weighted sum over θ_i
        return jnp.sum(probs_i[None, :] * num_factor[None, :] * jnp.exp(exponent), axis=1)  # (R,)

    def ex_ante_TT(self, i: int, actions) -> jnp.ndarray:
        """
        TT-approximated ex-ante utility for player i.

        Since all cores are diagonal R×R, the matrix product reduces to
        elementwise multiplication of R-vectors, contracted with the
        boundary prefactor c_l * exp(-t_l * sigma^2).
        """
        # Boundary prefactor: shape (R,)
        prefactor = self.c_weights * jnp.exp(-self.t_nodes * self.sigma)

        # Self core: shape (R,)
        a_i = actions[i].reshape(-1)
        gamma_self = self._gamma_self(i, a_i)

        # Other cores: elementwise product of R-vectors over all j != i
        gamma_others = jnp.ones((self.R,))
        for j in range(self.N):
            if j == i:
                continue
            a_j = actions[j].reshape(-1)
            gamma_others = gamma_others * self._gamma_other(j, i, a_j)

        # Final scalar: sum_l prefactor[l] * gamma_self[l] * gamma_others[l]
        return jnp.sum(prefactor * gamma_self * gamma_others)

    def objectives(self):
        def obj_func(x: VectorList, p_idx: int):
            action_matrices = self.split_profiles(x)
            return -self.ex_ante_TT(p_idx, action_matrices)

        return [lambda x, i=i: obj_func(x, i) for i in range(len(self.players))]

    def constraints(self):
        def interim_g(x: VectorList, i: int, t: int):
            action_matrices = self.split_profiles(x)
            a_i = action_matrices[i].reshape(-1)
            g_tilde_i = self.G_interf[i].reshape(-1)

            other_indices = self.get_others_idx(i)
            expected_total = 0.0
            for j in other_indices:
                probs_j = jnp.asarray(self.players[j].type_probs).reshape(-1)
                g_tilde_j = self.G_interf[j].reshape(-1)
                a_j = action_matrices[j].reshape(-1)
                expected_total = expected_total + jnp.sum(probs_j * g_tilde_j * a_j)

            return jnp.reshape(
                (a_i * g_tilde_i + expected_total - self.I)[t], ()
            )

        return [
            lambda x, i=i, t=t: interim_g(x, i, t)
            for t in range(self.T)
            for i in range(self.N)
        ]

    # =========================================================================
    # Verification utilities (exact computation, used after equilibrium found)
    # =========================================================================
    

    def ex_ante_exact(self, i: int, actions) -> jnp.ndarray:
        """
        Exact ex-ante utility for player i, fully vectorized.
        Cost: O(T^N) memory and flops, but done in a single JAX operation.
        """
        actions = self.split_profiles(actions)
        
        # For each player k, form the vector (a_k^{θ_k=0}, a_k^{θ_k=1}, a_k^{θ_k=2})
        a = jnp.stack([actions[k].reshape(-1) for k in range(self.N)], axis=0)  # (N, T)
        
        # Reshape each player's action vector to have shape (1,...,1,T,1,...,1)
        # with T in position k. Broadcasting then produces the full (T,T,...,T) grid.
        def broadcast_axis(vec, k):
            shape = [1] * self.N
            shape[k] = self.T
            return vec.reshape(shape)
        
        # Player i's action across the grid
        a_i_grid = broadcast_axis(a[i], i)  # shape (1,...,T,...,1)
        theta_i_grid = broadcast_axis(self.type_values_array[i], i)
        
        # Sum interference from all other players across the grid
        interference_grid = jnp.zeros([self.T] * self.N)
        for j in range(self.N):
            if j == i:
                continue
            a_j_grid = broadcast_axis(a[j], j)
            g_ji_grid = broadcast_axis(self.G_cross[j, i], j)
            interference_grid = interference_grid + g_ji_grid * a_j_grid
        
        # Full utility tensor, shape (T, T, ..., T)
        denom = self.C[i] * a_i_grid + interference_grid + self.sigma
        u_tensor = (theta_i_grid * a_i_grid) / denom
        
        # Joint probability tensor via outer product of marginals
        prob_tensor = jnp.ones([self.T] * self.N)
        for k in range(self.N):
            prob_tensor = prob_tensor * broadcast_axis(self.type_probs_array[k], k)
        
        # Contract
        return jnp.sum(prob_tensor * u_tensor)


    def stationarity_residual_exact(self, i: int, actions, lambdas):
        """
        Compute the gradient of the TRUE Lagrangian L_i with respect to a_i
        at the given (actions, lambdas) — the stationarity residual.

        Returns a vector of shape (T,).
        """
        from jax import grad

        def lagrangian(a_i_flat):
            # Build the full action profile with player i's action replaced
            full_actions = list(actions)
            full_actions[i] = a_i_flat
            # Negative of utility (since we minimize -Upsilon)
            U = -self.ex_ante_exact(i, full_actions)
            # Add constraint terms: sum over types of lambda_{i,t} * h_i(...,t)
            constr = 0.0
            g_tilde_i = self.G_interf[i].reshape(-1)
            other_indices = [j for j in range(self.N) if j != i]
            expected_total = 0.0
            for j in other_indices:
                probs_j = self.type_probs_array[j]
                g_tilde_j = self.G_interf[j].reshape(-1)
                a_j = full_actions[j].reshape(-1)
                expected_total = expected_total + jnp.sum(probs_j * g_tilde_j * a_j)
            for t in range(self.T):
                h_t = a_i_flat[t] * g_tilde_i[t] + expected_total - self.I
                constr = constr + lambdas[t] * h_t
            return U + constr

        return grad(lagrangian)(actions[i].reshape(-1))
    
    def tensor_error(self, i: int, actions) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute ||U_i - Û_i^ε||_F and ||∂_i U_i - ∂_i Û_i^ε||_F at a given
        strategy profile, where:
            U_i(θ) = true per-realization utility
            Û_i^ε(θ) = exponential-sum approximation
        
        Returns:
            (frobenius error in values, frobenius error in derivatives w.r.t. a_i^θ_i)
        """
        actions = self.split_profiles(actions)
        a = jnp.stack([actions[k].reshape(-1) for k in range(self.N)], axis=0)  # (N, T)

        def broadcast_axis(vec, k):
            shape = [1] * self.N
            shape[k] = self.T
            return vec.reshape(shape)

        # Build the full utility tensors (shape (T,)*N) — 3^15 entries
        a_i_grid = broadcast_axis(a[i], i)
        theta_i_grid = broadcast_axis(self.type_values_array[i], i)

        interference_grid = jnp.zeros([self.T] * self.N)
        for j in range(self.N):
            if j == i:
                continue
            a_j_grid = broadcast_axis(a[j], j)
            g_ji_grid = broadcast_axis(self.G_cross[j, i], j)
            interference_grid = interference_grid + g_ji_grid * a_j_grid

        denom = self.C[i] * a_i_grid + interference_grid + self.sigma
        U_true = (theta_i_grid * a_i_grid) / denom

        # Approximate utility via exponential sum:
        # Û_i(θ) = θ_i a_i^θ_i * Σ_ℓ c_ℓ exp(-t_ℓ * denom(θ))
        # Compute by summing R rank-1 tensors
        U_approx = jnp.zeros([self.T] * self.N)
        for ell in range(self.R):
            t_l = self.t_nodes[ell]
            c_l = self.c_weights[ell]
            # exp(-t_l * denom) factors across players
            term = c_l * jnp.exp(-t_l * self.sigma) * jnp.exp(-t_l * self.C[i] * a_i_grid)
            for j in range(self.N):
                if j == i:
                    continue
                a_j_grid = broadcast_axis(a[j], j)
                g_ji_grid = broadcast_axis(self.G_cross[j, i], j)
                term = term * jnp.exp(-t_l * g_ji_grid * a_j_grid)
            U_approx = U_approx + theta_i_grid * a_i_grid * term

        # Frobenius error for values
        val_err = jnp.sqrt(jnp.sum((U_true - U_approx) ** 2))

        # Derivative tensors: ∂u_i / ∂a_i^θ_i
        # True: d/da_i [θ_i a_i / denom] = θ_i / denom - θ_i a_i c_i / denom^2
        #     = (θ_i * (interference + σ)) / denom^2
        dU_true = (theta_i_grid * (interference_grid + self.sigma)) / (denom ** 2)

        # Approximate: derivative of the exp-sum approximation w.r.t. a_i^θ_i
        # θ_i a_i * Σ_ℓ c_ℓ e^(-t_ℓ σ) e^(-t_ℓ c_i a_i) * ∏_{j≠i} e^(-t_ℓ g_ji a_j)
        # d/da_i = θ_i * Σ_ℓ [c_ℓ e^(-t_ℓ σ) * (1 - t_ℓ c_i a_i) * e^(-t_ℓ c_i a_i) * ∏_{j≠i} ...]
        dU_approx = jnp.zeros([self.T] * self.N)
        for ell in range(self.R):
            t_l = self.t_nodes[ell]
            c_l = self.c_weights[ell]
            factor_self = (1.0 - t_l * self.C[i] * a_i_grid) * jnp.exp(-t_l * self.C[i] * a_i_grid)
            term = c_l * jnp.exp(-t_l * self.sigma) * factor_self
            for j in range(self.N):
                if j == i:
                    continue
                a_j_grid = broadcast_axis(a[j], j)
                g_ji_grid = broadcast_axis(self.G_cross[j, i], j)
                term = term * jnp.exp(-t_l * g_ji_grid * a_j_grid)
            dU_approx = dU_approx + theta_i_grid * term

        deriv_err = jnp.sqrt(jnp.sum((dU_true - dU_approx) ** 2))

        return val_err, deriv_err