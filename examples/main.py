from problems.bayesian import *
from problems.gnep import *
from solvers.algorithms import *
from solvers.utils import construct_vectors 
import timeit
import jax.numpy as jnp

if __name__ == '__main__':
    # print("Problem A1")
    # PA1 = ProblemA1()
    # PA1.set_initial_point(2.0, 1.0)
    # p1,d1 = PA1.solve()

    # print("Allocation Game - Akkarajitsakul")
    # PAG = AllocationGame()
    # PAG.set_initial_point(0.5, 0.1)
    # PAG.solve(EnergyMethod)

    print("Radar Power Game - TT")
    PAG = RadarPowerGameTT(R=80)
    PAG.set_initial_point(0.5, 0.1)

    print("Primal: ", PAG.primal_ip)
    print("TT: ", PAG.ex_ante_TT(0, construct_vectors(PAG.primal_ip, [3 for _ in range(PAG.N)])))
    tt_time = timeit.timeit(
        lambda: PAG.ex_ante_TT(0, construct_vectors(PAG.primal_ip, [3 for _ in range(PAG.N)])),
        number=1
    )
    print("Time taken:", tt_time, "seconds")
    print("Exact: ", PAG.ex_ante_exact(0, construct_vectors(PAG.primal_ip, [3 for _ in range(PAG.N)])))
    tt_time = timeit.timeit(
        lambda: PAG.ex_ante_exact(0, construct_vectors(PAG.primal_ip, [3 for _ in range(PAG.N)])),
        number=1
    )
    print("Time taken:", tt_time, "seconds")


    # a_eq = construct_vectors(PAG.primal_ip, [3 for _ in range(PAG.N)])

    # # Per-player epsilons (one per player since each player has their own utility tensor)
    # for i in range(PAG.N):
    #     val_err, deriv_err = PAG.tensor_error(i, a_eq)
    #     print(f"Player {i}: ||U-Û||_F = {float(val_err):.3e}, ||∂U-∂Û||_F = {float(deriv_err):.3e}")

    # # ε to use in Theorem 2 is the max
    # eps_vals = [PAG.tensor_error(i, a_eq)[0] for i in range(PAG.N)]
    # eps_derivs = [PAG.tensor_error(i, a_eq)[1] for i in range(PAG.N)]
    # epsilon = float(max(max(eps_vals), max(eps_derivs)))
    # print(f"\nε = {epsilon:.3e}")

    # # Also compute ||η||_F
    # eta_F = jnp.sqrt(jnp.prod(jnp.sum(PAG.type_probs_array ** 2, axis=1)))
    # print(f"||η||_F = {float(eta_F):.3e}")
    # print(f"Theorem 2 bound: ||η||_F · ε = {float(eta_F) * epsilon:.3e}")


    PAG.solve(EnergyMethod)


    # # Sample the quadrature at 100 points in the expected range
    # x_grid = jnp.linspace(0.2, 30.0, 20)
    # approx = jnp.sum(PAG.c_weights[:, None] * jnp.exp(-PAG.t_nodes[:, None] * x_grid[None, :]), axis=0)
    # true = 1.0 / x_grid
    # rel_err = jnp.abs(approx - true) / true
    # for x, r in zip(x_grid, rel_err):
    #     print(f"  x={float(x):6.2f}  rel_err={float(r):.2e}")


    # a = construct_vectors(PAG.primal_ip, [3 for _ in range(PAG.N)])
    # a_stacked = jnp.stack([a[k].reshape(-1) for k in range(PAG.N)], axis=0)

    # i = 0
    # min_x = PAG.C[i] * jnp.min(a_stacked[i]) + PAG.sigma
    # max_x = PAG.C[i] * jnp.max(a_stacked[i]) + PAG.sigma
    # for j in range(PAG.N):
    #     if j == i: continue
    #     max_x = max_x + jnp.max(PAG.G_cross[j, i] * a_stacked[j])
    # print(f"x range approx: [{float(min_x):.3f}, {float(max_x):.3f}]")