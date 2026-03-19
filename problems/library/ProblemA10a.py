import jax.numpy as jnp
from gnep_solver import Vector, VectorList
from gnep_solver.BaseProblem import BaseProblem
from gnep_solver.Player import Player

class ProblemA10a(BaseProblem):
    def __init__(self):
        self.F = 2
        self.C = 5
        self.P = 3
        super().__init__()

    def define_players(self):
        F, C, P = self.F, self.C, self.P
        N = F + C + 1
        player_vector_sizes = [P for _ in range(N)]
        player_objective_functions = [0, 1, 2, 3, 4, 5, 6, 7]  # change to all 0s
        player_constraints = [[0], [1], [2], [3], [4], [5], [6], [7, 8]]
        bounds = [(0, 100) for _ in range(F + C)] + [(0, 1)]
        return Player.batch_create(player_vector_sizes, player_objective_functions, player_constraints, bounds)

    def get_xi(self):
        x1 = jnp.array([2, 3, 4]).reshape(-1, 1)
        x2 = jnp.array([2, 3, 4]).reshape(-1, 1)
        x3 = jnp.array([6, 5, 4]).reshape(-1, 1)
        x4 = jnp.array([6, 5, 4]).reshape(-1, 1)
        x5 = jnp.array([6, 5, 4]).reshape(-1, 1)
        return [x1,x2,x3,x4,x5]

    def get_constants(self, i):
        if i in [0,1]:
            Q_i = jnp.array([
                [6, -2, 5],
                [-2, 6, -7],
                [5, -7, 20]
            ])
            b_i = jnp.array([30+i+self.F,30+i+self.F,30+i+self.F]).reshape(-1,1)
        elif i in [2,3,4]:
            Q_i = jnp.array([
                [6, 1, 0],
                [1, 7, -5],
                [0, -5, 7]
            ])
            b_i = jnp.array([30 + 2*(i + self.F), 30 + 2*(i + self.F), 30 + 2*(i + self.F)]).reshape(-1, 1)
        return dict(Q_i=Q_i, b_i=b_i)

    def objectives(self):
        def obj_func_firms(x, i):
            p = x[-1].reshape(-1, 1)
            x_firms = x[:self.F]
            y = x_firms[i].reshape(-1, 1)
            return jnp.reshape(p.T @ y, ())  # returns (1,F) vector
        obj_firms = [lambda x, i=i: obj_func_firms(x,i) for i in range(self.F)]

        def utility_function(x_i, Q_i, b_i):
            return jnp.reshape(-0.5 * x_i.T @ Q_i @ x_i + b_i.T @ x_i, ())

        def obj_func_consumers(x, i):
            x = jnp.hstack(x[self.F: self.F + self.C][i]).reshape(self.P, -1)
            Q_i = self.get_constants(i=i).get('Q_i')
            b_i = self.get_constants(i=i).get('b_i')
            return utility_function(x, Q_i, b_i)

        obj_consumers = [lambda x, i=i: obj_func_consumers(x,i) for i in range(self.C)]

        def obj_func_market(x):
            p = x[-1].reshape(-1, 1)
            y = jnp.hstack(x[:self.F]).reshape(self.P, -1)
            x = jnp.hstack(x[self.F:self.F + self.C]).reshape(self.P, -1)
            xi = jnp.hstack(self.get_xi()).reshape(self.P, -1)
            return jnp.reshape(p.T @ (jnp.sum(x, axis=1, keepdims=True) - jnp.sum(y, axis=1, keepdims=True) - jnp.sum(xi, axis=1,keepdims=True)), ())

        return obj_firms + obj_consumers + [obj_func_market]

    def constraints(self):
        def g0(x, i):
            idx = 10 * jnp.array([i + 1 for i in range(self.F)]).reshape(-1, 1)
            y = jnp.hstack(x[:self.F]).reshape(self.P, -1)
            sum_y = jnp.sum(y, axis=0).reshape(-1, 1) ** 2
            return jnp.reshape((sum_y - idx)[i], ())

        g0s = [lambda x, i=i: g0(x,i) for i in range(self.F)]

        def g1(x,i):
            p = x[-1].reshape(-1, 1)
            x = jnp.hstack(x[self.F: self.F + self.C][i]).reshape(self.P, -1)
            xi = self.get_xi()[i]
            return jnp.reshape((p.T @ x).reshape(-1, 1) - (p.T @ xi).reshape(-1, 1), ())

        g1s = [lambda x, i=i: g1(x,i) for i in range(self.C)]

        def g2(x):
            p = x[-1].reshape(-1, 1)
            return jnp.reshape(jnp.sum(p, axis=0) - 1, ())

        def g3(x):
            p = x[-1].reshape(-1, 1)
            return jnp.reshape(-jnp.sum(p, axis=0) + 1, ())

        return g0s + g1s + [g2, g3]