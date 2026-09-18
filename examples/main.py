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

    PAG = QuadraticGame()
    PAG.set_initial_point(0.5, 0.1)
    PAG.solve(EnergyMethod)

    # print("Radar Power Game - TT")
    # PAG = RadarPowerGameTT(R=80)
    # PAG.set_initial_point(0.9, 0.1)

    # print("Primal: ", PAG.primal_ip)
    # print("TT Evaluation: ", PAG.ex_ante_TT(0, construct_vectors(PAG.primal_ip, [3 for _ in range(PAG.N)])))
    # tt_time = timeit.timeit(
    #     lambda: PAG.ex_ante_TT(0, construct_vectors(PAG.primal_ip, [3 for _ in range(PAG.N)])),
    #     number=1
    # )
    # print("Time taken:", tt_time, "seconds")
    # print("Exact Evaluation: ", PAG.ex_ante_exact(0, construct_vectors(PAG.primal_ip, [3 for _ in range(PAG.N)])))
    # tt_time = timeit.timeit(
    #     lambda: PAG.ex_ante_exact(0, construct_vectors(PAG.primal_ip, [3 for _ in range(PAG.N)])),
    #     number=1
    # )
    # print("Time taken:", tt_time, "seconds")


    # a_eq = construct_vectors(PAG.primal_ip, [3 for _ in range(PAG.N)])

    # # Per-player epsilons (one per player since each player has their own utility tensor)

    # print("\nComputing error ||U-Û||_F and ||∂U-∂Û||_F for each player...")
    # errors = [PAG.tensor_error(i, a_eq) for i in range(PAG.N)]

    # for i in range(PAG.N):
    #     val_err, deriv_err = errors[i]
    #     print(f"Player {i}: ||U-Û||_F = {float(val_err):.3e}, ||∂U-∂Û||_F = {float(deriv_err):.3e}")

    # eps_vals   = [errors[i][0] for i in range(PAG.N)]
    # eps_derivs = [errors[i][1] for i in range(PAG.N)]
    # epsilon    = float(max(max(eps_vals), max(eps_derivs)))
    # print(f"ε = {epsilon:.3e}")

    # # # # Also compute ||η||_F
    # eta_F = jnp.sqrt(jnp.prod(jnp.sum(PAG.type_probs_array ** 2, axis=1)))
    # print(f"||η||_F = {float(eta_F):.3e}")
    # print(f"Theorem 2 bound: ||η||_F · ε = {float(eta_F) * epsilon:.3e}")


    # print("\nSolving the game using the Energy Method...")
    # primalx, dualx = PAG.solve(EnergyMethod)

    # x_struct = construct_vectors(primalx.reshape(-1, 1), [3] * PAG.N)

    # grad_f0 = PAG.algorithm.obj_derivatives[0](x_struct)[0]

    # player0_constraints = PAG.players[0].constraints
    # grad_g0 = sum(
    #     dualx[c] * PAG.algorithm.const_derivatives[c](x_struct)[0]
    #     for c in player0_constraints
    # )

    # print("player 0 raw Lagrangian grad:", (grad_f0 + grad_g0).ravel())

    # x_final = jnp.concatenate([primalx.ravel(), dualx.ravel()])
    # grad_at_final = PAG.algorithm.grad_min_func(x_final)
    # print("Energy gradient norm at final point:", jnp.linalg.norm(grad_at_final))
    # print("Energy gradient (primal part):", grad_at_final)
    # print("Energy gradient (primal part):", grad_at_final[:PAG.algorithm.total_actions])


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