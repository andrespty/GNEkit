from solvers.gnep_solver import VectorList
from solvers.dgbne_solver.BayesianPlayer import BayesianPlayer
from solvers.dgbne_solver.BayesianProblem import BayesianProblem


class RoutingProblem(BayesianProblem):
    def define_players(self):
        player_vector_sizes = [2, 2, 2]
        player_types = [[1,2],[3,4],[5,6]]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[None], [None], [None]]
        bounds = [(0,100) for _ in range(3)]
        return BayesianPlayer.batch_create(
            player_vector_sizes,
            player_types,
            player_objective_functions,
            player_constraints,
            bounds
        )

    def objectives(self):
        def obj_func_1(x: VectorList):
            return

        def obj_func_2(x: VectorList):
            return

        def obj_func_3(x: VectorList):
            return

        return [obj_func_1, obj_func_2, obj_func_3]

    def constraints(self):
        return []
