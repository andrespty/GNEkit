# GNE Solver

`GNE Solver` is a Python project for modeling and solving generalized Nash equilibrium problems, including Bayesian games with shared constraints.

The current codebase is organized around two main packages:

- `problems/` for reusable game definitions
- `solvers/` for reusable abstractions and optimization algorithms

The main workflow is:

1. choose or define a problem class
2. set primal and dual initial points
3. choose an algorithm
4. solve for primal and dual variables

Minimal example:

```python
from problems.bayesian import AllocationGame
from solvers.algorithms import EnergyMethod

problem = AllocationGame()
problem.set_initial_point(0.5, 0.1)
primal_x, dual_x = problem.solve(EnergyMethod)
```

The current architecture centers on:

- `Player` and `BayesianPlayer`
- `BaseProblem` and `BayesianProblem`
- `EnergyMethod` and `FBEnergyMethod`

Use the guides in this documentation for installation, the current package layout, and the intended problem-definition workflow.
