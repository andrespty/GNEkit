# Generalized Nash Kit (GNEkit)

`GNEkit` is a Python library for modeling and solving generalized Nash equilibrium problems in a reusable, research-friendly way.

The library is organized around a simple workflow:

1. define or import a problem
2. define the players and constraints
3. choose an algorithm
4. solve for the primal and dual variables

It supports both standard generalized Nash equilibrium problems and Bayesian variants built on the same core abstractions.

## What This Library Provides

- Reusable abstractions for players and problems
- Algorithms for solving equilibrium problems
- Utilities for vector construction and solver internals
- Built-in benchmark-style problem definitions
- Support for Bayesian games with typed players

## Core Workflow

A typical session looks like this:

```python
from problems.bayesian import AllocationGame
from solvers.algorithms import EnergyMethod

problem = AllocationGame()
problem.set_initial_point(0.5, 0.1)
primal_x, dual_x = problem.solve(EnergyMethod)
```

At a high level:

- `Player` and `BayesianPlayer` describe each participant
- `BaseProblem` and `BayesianProblem` define the game
- `EnergyMethod` and related algorithms solve the resulting system

## Who This is For
This project is especially useful for:

- Researchers working on generalized Nash equilibrium problems
- Students learning equilibrium problem formulations
- Developers building reusable game-theoretic problem classes
- Anyone experimenting with constrained multi-player optimization

## Documentation Roadmap
If you are new to the library, start here:

- Go to [`Getting Started`](./getting_started/overview.md) for installation and a first runnable example
- Read the [`Guides`](./guides/core_concepts.md) section to understand the problem-definition workflow
- Use [`API Reference`](./reference/index.md) for class and function details
- Browse [`Examples`](./examples/examples.md) for concrete problem setups

## Project Structure
The codebase is centered on two main packages:

- `problems/` contains reusable problem definitions
- `solvers/` contains core abstractions, algorithms, and utilities.