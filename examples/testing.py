from problems import *
from solvers.utils import *

if __name__ == '__main__':
    p_obj_func = [0,1,2,3]
    p_constraints = [[1,2], None, [1,3], [None]]
    player_vector_sizes =[1,2,1,3]
    num_funcs = 4

    one = one_hot_encoding(p_obj_func, player_vector_sizes, num_funcs)
    two = one_hot_encoding(p_constraints, player_vector_sizes, num_funcs)
    print(one)
    print(two)
