"""Problem definitions are checked when the problem is created."""
import jax.numpy as jnp
import pytest

from gnekit.core import BaseProblem, Player
from gnekit.bayesian import BayesianPlayer, BayesianProblem


def two_players(constraints=((0,), (0,)), f_indices=(0, 1)):
    return [
        Player(name="P1", size=1, f_index=f_indices[0], constraints=constraints[0], bounds=(0.0, 5.0)),
        Player(name="P2", size=1, f_index=f_indices[1], constraints=constraints[1], bounds=(0.0, 5.0)),
    ]


def f1(x):
    return (x[0][0, 0] - 1.0) ** 2 + x[1][0, 0]


def f2(x):
    return (x[1][0, 0] - 2.0) ** 2 + x[0][0, 0]


def g1(x):
    return x[0][0, 0] + x[1][0, 0] - 3.0


def make_problem(objectives=(f1, f2), constraints=(g1,), players=None):
    class Problem(BaseProblem):
        def define_players(self):
            return players if players is not None else two_players()

        def objectives(self):
            return list(objectives)

        def constraints(self):
            return list(constraints)

    return Problem


def test_valid_problem_is_created():
    problem = make_problem()()
    assert len(problem.players) == 2


def test_problem_without_constraints_is_allowed():
    make_problem(constraints=(), players=two_players(constraints=((), ())))()


def test_objective_must_return_a_scalar():
    def vector_objective(x):
        return jnp.concatenate(x).ravel()

    with pytest.raises(ValueError, match=r"objectives\(\)\[1\].*scalar"):
        make_problem(objectives=(f1, vector_objective))()


def test_constraint_must_return_a_scalar():
    def vector_constraint(x):
        return jnp.concatenate(x)

    with pytest.raises(ValueError, match=r"constraints\(\)\[0\].*scalar"):
        make_problem(constraints=(vector_constraint,))()


def test_function_must_take_one_argument():
    def two_arguments(x, y):
        return f1(x)

    with pytest.raises(ValueError, match=r"objectives\(\)\[0\].*exactly one argument"):
        make_problem(objectives=(two_arguments, f2))()


def test_extra_arguments_with_defaults_are_allowed():
    objectives = [lambda x, i=i: (f1, f2)[i](x) for i in range(2)]
    make_problem(objectives=objectives)()


def test_objectives_must_be_callable():
    with pytest.raises(TypeError, match=r"objectives\(\)\[1\] is not callable"):
        make_problem(objectives=(f1, 3.0))()


def test_objectives_must_not_be_empty():
    with pytest.raises(TypeError, match="non-empty"):
        make_problem(objectives=())()


def test_player_objective_index_must_exist():
    with pytest.raises(ValueError, match="references objective 2"):
        make_problem(players=two_players(f_indices=(0, 2)))()


def test_player_constraint_index_must_exist():
    with pytest.raises(ValueError, match="references constraint 1"):
        make_problem(players=two_players(constraints=((0,), (1,))))()


def test_parameters_set_after_super_init_are_seen():
    class Problem(BaseProblem):
        def __init__(self):
            super().__init__()
            self.target = 1.0   # used by objectives(), set after the base constructor

        def define_players(self):
            return two_players()

        def objectives(self):
            return [lambda x: (x[0][0, 0] - self.target) ** 2, f2]

        def constraints(self):
            return [g1]

    Problem()


def test_bayesian_problem_must_define_objectives():
    class Incomplete(BayesianProblem):
        def define_players(self):
            return [BayesianPlayer(name="P1", size=2, f_index=0, constraints=(),
                                   type_values=(1.0, 2.0), type_probs=(0.5, 0.5))]

        def constraints(self):
            return []

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
