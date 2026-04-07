from problems.bayesian import *
from problems.gnep import *
from solvers.algorithms import * 

if __name__ == '__main__':
    # print("Problem A1")
    # PA1 = ProblemA1()
    # PA1.set_initial_point(2.0, 1.0)
    # p1,d1 = PA1.solve()

    print("Allocation Game - Akkarajitsakul")
    PAG = AllocationGame()
    PAG.set_initial_point(0.5, 0.1)
    PAG.solve(EnergyMethod)
