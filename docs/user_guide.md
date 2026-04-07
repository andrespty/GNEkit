# User Guide

## Core Idea

The project is designed around reusable problem classes.

A problem class defines:

- its players
- its objective functions
- its constraint functions

An algorithm then solves the resulting equilibrium problem.

## Package Layout

### `solvers/`

Contains the reusable abstractions and algorithms.

Important modules:

- `solvers.gnep_solver.BasePlayer`
- `solvers.gnep_solver.BaseProblem`
- `solvers.dgbne_solver.BayesianPlayer`
- `solvers.dgbne_solver.BayesianProblem`
- `solvers.algorithms.EnergyMethod`
- `solvers.algorithms.FBEnergyMethod`

### `problems/gnep/`

Contains standard GNEP benchmark-style problem classes.

### `problems/bayesian/`

Contains Bayesian game problem classes built on the same solver workflow.

## Solving a Standard Problem

```python
from problems.gnep import ProblemA1
from solvers.algorithms import EnergyMethod

problem = ProblemA1()
problem.set_initial_point(1.0, 0.1)
primal_x, dual_x = problem.solve(EnergyMethod)
```

## Solving a Bayesian Problem

```python
from problems.bayesian import AllocationGame
from solvers.algorithms import EnergyMethod

problem = AllocationGame()
problem.set_initial_point(0.5, 0.1)
primal_x, dual_x = problem.solve(EnergyMethod)
```

## Defining a New Problem

For a standard problem, subclass `BaseProblem` and implement:

- `define_players()`
- `objectives()`
- `constraints()`

For a Bayesian problem, subclass `BayesianProblem` and implement the same methods while using the Bayesian helpers for:

- reshaping type-contingent strategy vectors
- expected actions
- type-weighted quantities

## Initial Points

`set_initial_point(primal_x, dual_x)` is the standard entry point before solving.

- `primal_x` sets the primal variables
- `dual_x` sets the dual variables associated with the problem constraints

The total primal dimension must match the sum of all player action sizes, and the dual dimension must match the number of constraints.

## Algorithms

The main algorithms currently exposed are:

- `EnergyMethod`
- `FBEnergyMethod`

`EnergyMethod` is the default general-purpose choice for most examples.
