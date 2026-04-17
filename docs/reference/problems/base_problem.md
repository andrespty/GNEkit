# BaseProblem

::: solvers.gnep_solver.BaseProblem.BaseProblem
    options:
        show_root_heading: false
        show_root_toc_entry: false
        inherited_members: false
        show_source: false
        docstring_style: numpy
        members:
        - __init__
        - players
        - primal_ip
        - dual_ip
        - known_solution
        - define_players
        - objectives
        - constraints
        - set_initial_point
        - solve

## Example

```python
from solvers.gnep_solver import BaseProblem, Player
from solvers.algorithms import EnergyMethod

class SimpleProblem(BaseProblem):
    def define_players(self):
        return [
            Player(name="P1", size=1, f_index=0, constraints=(0,), bounds=(0.0, 5.0)),
            Player(name="P2", size=1, f_index=1, constraints=(0,), bounds=(0.0, 5.0)),
        ]

    def objectives(self):
        def f1(x):
            x1, x2 = x
            return (x1[0, 0] - 1.0) ** 2 + x2[0, 0]

        def f2(x):
            x1, x2 = x
            return (x2[0, 0] - 2.0) ** 2 + x1[0, 0]

        return [f1, f2]

    def constraints(self):
        def g1(x):
            x1, x2 = x
            return x1[0, 0] + x2[0, 0] - 3.0

        return [g1]

problem = SimpleProblem()
problem.set_initial_point(1.0, 0.1)
primal_x, dual_x = problem.solve(EnergyMethod)
```