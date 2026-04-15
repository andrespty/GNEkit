# Quickstart

This page walks through the basic workflow for solving a problem with `GNE Solver`.

The standard pattern is:

1. Import a problem class
2. Instantiate the problem
3. Set the initial primal and dual values
4. Choose an algorithm
5. Solve the problem

## Minimal Example

```python
from problems.bayesian import AllocationGame
from solvers.algorithms import EnergyMethod

problem = AllocationGame()
problem.set_initial_point(0.5, 0.1)
primal_x, dual_x = problem.solve(EnergyMethod)

print("Primal solution:", primal_x)
print("Dual solution:", dual_x)
```

## What Happens Here
- `AllocationGame()` creates a predefined problem instance
- `set_initial_point(0.5,0.1)` initializes the primal and dual variables
- `solve(EnergyMethod)` runs the solver with the selected algorithm
- The solver returns:
  - `primal_x`: the primal decision variables in a list
  - `dual_x`: the dual variables associated with constraints
  

## Choosing Initial Values
Before calling `solve(...)`, you must provide an initial point. 

You can pass:
- A scalar float, which will be broadcast across all variables
- A list with one entry per variable
  
Example with scalar initialization:
```python
problem.set_initial_point(1.0, 0.1)
```

Example with explicit vectors:
```python
problem.set_initial_point(
    [1.0, 1.0 ,2.0],   # This game has three primal variables
    [0.1, 0.5]         # and two constraints     
)
```
The primal vector length must match the total number of player variables, and the dual vector length must match the number of constraints in the problem.

## Choosing an Algorithm
Algorithms are passed as classes to `solve(...)`.

Example:
```python
from solvers.algorithms import EnergyMethod
...
primal_x, dual_x = problem.solve(EnergyMethod)
```

## Next Steps
After this page, a good path is:

- Read [`Core Concepts`](../guides/core_concepts.md) to understand the main abstractions
- Read [`User Guide`](../guides/user_guide.md) for the full workflow
- Browse [`Examples`](../examples/examples.md) for more complete problem setups
- Use [`API Reference`](../reference/index.md) for class-level details


