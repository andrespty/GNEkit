# Installation

## Requirements

The current project depends on:

- Python 3.8+
- `numpy`
- `scipy`
- `jax`

## Install From the Repository

From the repository root:

```bash
pip install -r requirements.txt
```

For editable local development:

```bash
pip install -e .
```

## Notes

- The active solver stack in `solvers/` imports `jax` directly, so `jax` must be installed.
- The reusable problem definitions live in `problems/`.
- Using `pip install -e .` is the cleanest way to make both `solvers` and `problems` importable while developing.
