# GNE Problems Solver

GNE Problems Solver is a Python package for modeling and solving Generalized Nash Equilibrium (GNE) problems, supporting both bounded and unbounded cases. It provides a flexible framework for defining multi-player optimization problems with shared and individual constraints, and includes utilities for solution analysis and Nash equilibrium verification.

## Features
- Solve bounded and unbounded GNE problems
- Modular problem definitions for easy extension
- Utilities for Nash equilibrium checking and solution analysis
- Example problems and usage scripts
- Extensible for research and teaching

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Example usage (see `examples/main.py`):

```python
from gne_solver import *
from problems import *

problem_n = A10eU  # Select a problem
bounded = False

if bounded:
    problem = get_problem(problem_n)
    solver = GNEP_Solver_Bounded(...)
    sol = solver.solve_game(...)
    solver.summary()
else:
    problem = get_problem(problem_n)
    solver = GNEP_Solver_Unbounded(...)
    sol = solver.solve_game(...)
    solver.summary()
```

See the `examples/` directory for more scripts and usage patterns.

## Project Structure
- `solvers/` — Core solver logic, Nash checking, utilities
- `problems/` — Problem definitions (bounded and unbounded)
- `examples/` — Example scripts for running and testing problems
- `docs/` — Documentation (installation, user guide, reference)
- `development/` — Development scripts and standards

## Documentation

Full documentation is available in the `docs/` directory. To build and view locally:

```bash
mkdocs serve
```

## Contributing

Contributions are welcome! Please see `docs/development/standards.md` for guidelines.

## License

This project is licensed under the MIT License.
