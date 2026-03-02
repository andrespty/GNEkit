from gne_solver import *
from gne_solver.NashCheck import *
from problems import *

if __name__ == '__main__':
    # Testing: Change the next line to test a problem
    problem_n = A4U
    bounded = False
    single_obj_vector = False
    # Github

    if bounded:
        problem = get_problem(problem_n)
        (player_vector_sizes,
         player_objective_functions,
         player_constraints, bounds, bounds_training) = problem['players']
        print('Here')
        solver1 = GNEP_Solver_Bounded(
            problem['obj_funcs'],
            problem['obj_ders'],
            problem['constraints'],
            problem['constraint_ders'],
            player_objective_functions,
            player_constraints,
            bounds_training,
            player_vector_sizes,
        )
        print(player_vector_sizes)

        # Set Initial Point
        primal, dual = get_initial_point(
            player_vector_sizes,
            problem["constraints"],
            primal_ip=1,
            dual_ip=1
        )
        print("Initial Points")
        print(flatten_variables(primal, dual))
        print(bounds_training)
        # # Solve Problem
        sol = solver1.solve_game(flatten_variables(primal, dual),bounds=bounds_training )
        print('\n\n')
        # solver1.summary(problem.paper_solution()[0])
        solver1.summary()
        # print(sol)
        print('\n\n')

        # solver1.nash_check()
    else:
        problem = get_problem(problem_n)
        (player_vector_sizes,
         player_objective_functions,
         player_constraints) = problem['players']

        solver1 = GNEP_Solver_Unbounded(
            problem['obj_funcs'],
            problem['obj_ders'],
            problem['constraints'],
            problem['constraint_ders'],
            player_objective_functions,
            player_constraints,
            player_vector_sizes,
        )
        # Set Initial Point
        primal, dual = get_initial_point(player_vector_sizes, problem['constraints'], primal_ip=1, dual_ip=1)
        print(flatten_variables(primal, dual))
        # # Solve Problem
        ip1 = flatten_variables(primal, dual)
        res, elapsed_time = solver1.solve_game(ip1)
        # print('\n\n')
        summary(
            res,
            elapsed_time,
            solver1.wrapper,
            player_vector_sizes
        )
        nash_check(problem_n, res.x[:sum(player_vector_sizes)], single_obj_vector=single_obj_vector)



