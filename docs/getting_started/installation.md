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

- The algorithms import `jax` directly, so `jax` must be installed.
- The source lives in `src/gnekit/`, and the benchmark problems in `src/gnekit/problems/`.
- Using `pip install -e .` makes `gnekit` importable while developing.
- Run the test suite with `pytest` from the repository root.
