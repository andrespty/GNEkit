from gnep_solver import *
from gnep_solver.utils import one_hot_encoding
from typing import List
from scipy.optimize import basinhopping
import timeit
from gnep_solver.Player import Player
import jax
import jax.numpy as jnp
from gnep_solver.EnergyMethod import EnergyMethod
jax.config.update("jax_enable_x64", True)

class GeneralizedGame:
    def __init__(self,
                 obj_funcs: List[ObjFunction],
                 constraints: List[ConsFunction],
                 player_list: List[Player]
        ):
        self.N = len(player_list)
        self.obj_functions = obj_funcs                                    # list of functions
        self.const = constraints                                          # list of functions
        self.action_sizes = [player.size for player in player_list]
        self.bounds = [player.bounds for player in player_list]

        # Indices
        p_obj_func_idx = [player.f_index for player in player_list]
        p_constraint_idx = [player.constraints for player in player_list]

        # JAX Compatible indices
        self.player_obj_idx = jnp.array(p_obj_func_idx, dtype=jnp.int32)

        # Pre Compile Derivatives for speed
        self.obj_derivatives = [jax.jit(jax.grad(obj)) for obj in obj_funcs]
        self.const_derivatives = [jax.jit(jax.grad(const)) for const in constraints]

        # Constraint Matrix
        self.player_const_idx_matrix = one_hot_encoding(
            p_constraint_idx,
            self.action_sizes,
            len(constraints)
        )

        # Pre-instantiate the solver once to avoid overhead in loops
        self.solver = EnergyMethod(
            self.action_sizes,
            self.obj_derivatives,
            self.const,
            self.const_derivatives,
            self.player_obj_idx,
            self.player_const_idx_matrix,
            self.bounds
        )

    def grad_val(self, actions: jnp.ndarray) -> List[jnp.ndarray]:
        x = construct_vectors(jnp.array(actions).reshape(-1,1), self.action_sizes)
        return [df(x) for df in self.obj_derivatives]

    def energy_val(self, actions: jnp.ndarray) -> float:
        return self.solver.min_func(actions)

    def solve_game(self, ip: jnp.ndarray):
        minimizer_kwargs = dict(method="SLSQP")
        start = timeit.default_timer()
        result = basinhopping(
            self.solver.min_func,
            ip,
            stepsize=0.01,
            niter=1000,
            minimizer_kwargs=minimizer_kwargs,
            interval=1,
            niter_success=100,
            disp=True,
            # callback=stopping_criterion
        )
        print(self.energy_val(jnp.array(result.x)))
        stop = timeit.default_timer()
        elapsed_time = stop - start
        return result, elapsed_time