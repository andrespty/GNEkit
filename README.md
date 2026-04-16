# GNEkit — Generalized Nash Kit

A Python library for modeling and solving **Generalized Nash Equilibrium (GNE)** problems and **Discrete Generalized Bayesian Nash Equilibrium (D-GBNE)** problems.

GNEkit provides a clean class-based interface for defining multi-player games with shared constraints, a collection of benchmark problems from the literature, and a set of algorithms built on JAX automatic differentiation.

📖 **[`Full Documentation`](https://andrespty.github.io/GNEkit/)** 📖

---

## Features

- Define standard GNE problems by subclassing `BaseProblem` with three methods
- Define Bayesian (D-GBNE) problems with discrete private types via `BayesianProblem`
- Plug in algorithms as classes — swap solvers without changing the problem definition
- Pre-compiled JAX gradients for all objectives and constraints
- Built-in KKT verification after every solve
- 18 standard GNE benchmark problems from Facchinei & Kanzow (2009)
- 3 D-GBNE example problems (quadratic, wireless allocation, radar power allocation)
- Extend with custom algorithms by subclassing `BaseAlgorithm`

---

## Installation

**Requirements:** Python 3.8+, `numpy`, `scipy`, `jax`

Clone the repository and install in editable mode:

```bash
git clone https://github.com/andrespty/GNEkit.git
cd gnekit
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

---

## Quickstart

The standard workflow is: import a problem, set an initial point, choose an algorithm, solve.

```python
from problems.gnep import ProblemA1
from solvers.algorithms import EnergyMethod

problem = ProblemA1()
problem.set_initial_point(0.5, 0.1)
primal_x, dual_x = problem.solve(EnergyMethod)

print("Primal solution:", primal_x)
print("Dual solution:", dual_x)
```

The same workflow applies to Bayesian problems:

```python
from problems.bayesian import AllocationGame
from solvers.algorithms import EnergyMethod

problem = AllocationGame()
problem.set_initial_point(0.5, 0.1)
primal_x, dual_x = problem.solve(EnergyMethod)
```

`set_initial_point` accepts either a scalar (broadcast across all variables) or explicit vectors for the primal and dual parts. `solve` returns the primal actions and dual variables at equilibrium and automatically runs KKT verification.

---

## Defining a New Problem

Subclass `BaseProblem` and implement three methods:

```python
from solvers.gnep_solver import BaseProblem, Player

class MyProblem(BaseProblem):
    def define_players(self):
        return [
            Player(name="P1", size=1, f_index=0, constraints=[0], bounds=(0.0, 5.0)),
            Player(name="P2", size=1, f_index=1, constraints=[0], bounds=(0.0, 5.0)),
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
```

For Bayesian problems with discrete private types, subclass `BayesianProblem` instead. See the [guides](https://andrespty.github.io/GNEkit/guides/core_concepts/) for both workflows.

---

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| `EnergyMethod` | Energy function minimization with primal-dual structure |
| `FBEnergyMethod` | Forward-backward variant of the energy method |

All algorithms inherit from `BaseAlgorithm`, which handles player validation, JAX derivative compilation, and the `basinhopping` + SLSQP optimization loop. To implement a custom algorithm, subclass `BaseAlgorithm` and define `min_func`. See the [creating an algorithm guide](https://andrespty.github.io/GNEkit/guides/creating_algorithm/).

---

## Project Structure

```
gnekit/
├── solvers/
│   ├── algorithms/        # BaseAlgorithm, EnergyMethod, FBEnergyMethod, VectorEnergyMethod
│   ├── gnep_solver/       # BaseProblem, Player
│   ├── dgbne_solver/      # BayesianProblem, BayesianPlayer
│   ├── utils.py           # construct_vectors, flatten_variables, one_hot_encoding
│   ├── schema.py          # Type definitions (Vector, ObjFunction, ConsFunction)
│   └── validation.py      # Input validators
├── problems/
│   ├── gnep/              # 18 standard GNE benchmark problems (A1–A18)
│   └── bayesian/          # 3 D-GBNE example problems
├── docs/                  # MkDocs documentation source
└── examples/              # Runnable example scripts
```

---

## Benchmark Problems

GNEkit includes all 18 test problems from:

> Facchinei, F. & Kanzow, C. (2009). *Penalty Methods for the Solution of Generalized Nash Equilibrium Problems (With Complete Test Problems)*. University of Würzburg.

And three D-GBNE problems from:

> Ho, A. (2026). *Discrete Generalized Bayesian Nash Equilibrium.*

The Bayesian problems are adapted from Akkarajitsakul et al. (2011) and Deligiannis & Lambotharan (2017). All problems are documented with their mathematical formulations and known solutions at [GNEkit](https://andrespty.github.io/GNEkit/examples/).

---
## License

MIT License. See `LICENSE` for details.
