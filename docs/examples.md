# Examples

## Minimal Standard Example

```python
from problems.gnep import ProblemA1
from solvers.algorithms import EnergyMethod

problem = ProblemA1()
problem.set_initial_point(1.0, 0.1)
primal_x, dual_x = problem.solve(EnergyMethod)
```

## Minimal Bayesian Example

```python
from problems.bayesian import AllocationGame
from solvers.algorithms import EnergyMethod

problem = AllocationGame()
problem.set_initial_point(0.5, 0.1)
primal_x, dual_x = problem.solve(EnergyMethod)
```

## Intended Workflow

The current intended workflow is:

1. import a reusable problem class from `problems.gnep` or `problems.bayesian`
2. instantiate the problem
3. set initial primal and dual values
4. solve with an algorithm from `solvers.algorithms`

## Reusable Problem Families

### Standard GNEP problems

Examples in `problems/gnep/` include:

- `ProblemA1`
- `ProblemA2`
- `ProblemA3`
- ...
- `ProblemA18`

### Bayesian problems

Examples in `problems/bayesian/` include:

- `AllocationGame`
- `QuadraticGame`
- `RadarPowerGame`

## Defining Your Own Problem

### Standard problem

```python
class MyProblem(BaseProblem):
    def define_players(self):
        ...

    def objectives(self):
        ...

    def constraints(self):
        ...
```

### Bayesian problem

```python
class MyBayesianProblem(BayesianProblem):
    def define_players(self):
        ...

    def objectives(self):
        ...

    def constraints(self):
        ...
```

The canonical runnable demonstration should remain `examples/main.py`, kept small and focused on the current public API.
