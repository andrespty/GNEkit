import jax.numpy as jnp
from gnep_solver import Vector, VectorList
from tester_solver.ManualBaseProblem import ManualBaseProblem
from gnep_solver.Player import Player
from typing import List

class ProblemA8(ManualBaseProblem):
    def __init__(self, players: List[Player] = None):
        super().__init__(players)

    def known_solution(self):
        value = [0.62503131162143, 0.37500031253875, 0.93754579549990]
        return value

    def define_players(self):
        player_vector_sizes = [1, 1, 1]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0, 1], [0, 1], [None]]
        bounds = [(0, 100) for _ in range(3)] # player 3 bounds may need to be tighter (0,2)
        return Player.batch_create(
            player_vector_sizes,
            player_objective_functions,
            player_constraints,
            bounds
        )

    def objectives(self):
        def obj_func_1(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            return jnp.reshape(-x1[0], ())

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x2 = x[1]
            return jnp.reshape((x2[0] - 0.5) ** 2, ())

        def obj_func_3(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x3 = x[2]
            return jnp.reshape((x3[0] - 1.5 * x1[0]) ** 2, ())

        return [obj_func_1, obj_func_2, obj_func_3]

    def objectives_der(self):
        def obj_func_der_1(x):
            return {0: jnp.array([-1.])}

        def obj_func_der_2(x):
            x2 = x[1]
            return {1: jnp.array([2 * x2[0] - 1.])}

        def obj_func_der_3(x):
            x1 = x[0]
            x3 = x[2]
            return {2: jnp.array([2 * x3[0] - 3 * x1[0]])}

        return [obj_func_der_1, obj_func_der_2, obj_func_der_3]

    def constraints(self):
        def g0(x: VectorList):
            x1, x2, x3 = x
            val = x1[0] + x2[0] - 1.0
            return jnp.reshape(val, ())

        def g1(x: VectorList):
            x1, x2, x3 = x
            val = x3[0] - x1[0] - x2[0]
            return jnp.reshape(val, ())

        return [g0, g1]

    def constraints_der(self):
        def g0_der(x):
            dx1 = jnp.array([1.])
            dx2 = jnp.array([1.])
            dx3 = jnp.array([0.])
            return [dx1, dx2, dx3]

        def g1_der(x):
            dx1 = jnp.array([-1.])
            dx2 = jnp.array([-1.])
            dx3 = jnp.array([1.])
            return [dx1, dx2, dx3]

        return [g0_der, g1_der]