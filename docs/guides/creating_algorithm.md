# Creating a Custom Algorithm

This guide explains how to implement a custom algorithm that works with `GNEkit`'s problem and solver infrastructure.

Every algorithm in this library is a subclass of `BaseAlgorithm`. The base class handles all the bookkeeping — loading players, compiling derivatives, managing bounds — so you only need to define one thing: the **merit function** that the optimizer minimizes.

## The Required Structure

A new algorithm must inherit from `BaseAlgorithm` and implement the `min_func` method:

```python
from solvers.algorithms import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    def __init__(self, obj, const, players):
        super().__init__(obj, const, players)

    def min_func(self, x):
        ...
        return scalar_value
```

The `__init__` signature must match exactly: `obj`, `const`, `players`. Calling `super().__init__()` triggers validation and pre-compilation, and populates all the attributes you'll use inside `min_func`.

## What `BaseAlgorithm` Provides

After calling `super().__init__()`, the following attributes are available:

**Problem structure:**

- `self.total_actions` — total number of primal variables across all players
- `self.action_sizes` — list of action vector sizes per player, e.g. `[2, 3, 2]`
- `self.action_splits` — cumulative split indices for slicing the flat primal vector
- `self.player_obj_idx` — which objective function index belongs to each player
- `self.player_const_idx` — which constraint indices apply to each player
- `self.player_const_idx_matrix` — a `(total_actions, num_constraints)` one-hot matrix encoding which player is responsible for each constraint

**Functions and derivatives:**

- `self.obj_functions` — validated list of objective functions
- `self.const` — validated list of constraint functions
- `self.obj_derivatives` — JIT-compiled gradients of each objective (via `jax.grad`)
- `self.const_derivatives` — JIT-compiled gradients of each constraint (via `jax.grad`)

**Bounds:**

- `self.bounds` — list of `(lb, ub)` tuples for primal variables
- `self.bounds_dual` — list of `(0, 100)` tuples for dual variables, one per constraint
- `self.bounds_all` — `self.bounds + self.bounds_dual`, passed directly to the optimizer
- `self.lb_vector`, `self.ub_vector` — primal bounds as JAX column vectors

## What `min_func` Receives

`min_func` is called by the optimizer with a **single flat array** `x`:

```
x = [ primal variables ... | dual variables ... ]
```

The first `self.total_actions` entries are the stacked primal action variables (all players concatenated). The remaining entries are the dual variables (Lagrange multipliers), one per constraint.

To work with them:

```python
def min_func(self, x):
    x_jax = jnp.asarray(x)
    actions = x_jax[:self.total_actions]       # primal part
    duals   = x_jax[self.total_actions:]       # dual part
    ...
```

To split the primal vector into per-player vectors, use `construct_vectors`:

```python
from solvers.utils import construct_vectors

vector_actions = construct_vectors(actions.reshape(-1, 1), self.action_sizes)
# vector_actions[i] is player i's action as a column vector
```

## What `min_func` Must Return

`min_func` must return a single Python `float`. The optimizer minimizes this value, so a return of zero (or near zero) should correspond to equilibrium.

```python
def min_func(self, x) -> float:
    ...
    return float(some_scalar)
```

!!! warning
    The optimizer (`basinhopping` + `SLSQP`) expects a Python `float`, not a JAX array. Use `np.float64(...)` or `float(...)` to convert.

## Computing Gradients

The base class pre-compiles a JAX gradient for each objective and each constraint. You can call them directly inside `min_func`:

```python
# Gradient of objective k with respect to all players' actions
# Returns a list of arrays, one per player
grad = self.obj_derivatives[k](vector_actions)

# Gradient for player i specifically
grad_i = grad[i]

# Similarly for constraints
c_grad = self.const_derivatives[k](vector_actions)
```

If your merit function depends on the Lagrangian, you can assemble it from these pieces manually, or follow the pattern used in `EnergyMethod`.

## How `solve` Works

`BaseAlgorithm.solve(ip)` takes an initial point `ip` (a flat array over all primal and dual variables) and runs `basinhopping` with `SLSQP` as the local minimizer, using `self.bounds_all` to enforce variable bounds. It calls `self.min_func` at each iteration.

When it finishes, it prints a result summary and runs `check_kkt` automatically to verify the equilibrium conditions.

You do not need to override `solve`. Implementing `min_func` is sufficient.

!!! note
    In the standard workflow, `solve` is the function called on a problem instance — `problem.solve(MyAlgorithm)`. This means `solve` is the entry point for the entire optimization run. If you want to use a different optimizer altogether (e.g. a gradient-based method, a Newton step, or a custom iterative scheme), you can override `solve` in your subclass instead of — or in addition to — `min_func`.

## A Complete Example

The following is a minimal algorithm that uses the KKT stationarity residual as its merit function. At equilibrium, the gradient of the Lagrangian with respect to each player's actions should be zero, so minimizing the squared norm of these residuals drives the solver toward a Nash equilibrium.

```python
import numpy as np
import jax.numpy as jnp
from solvers.algorithms import BaseAlgorithm
from solvers.utils import construct_vectors


class StationarityMethod(BaseAlgorithm):
    def __init__(self, obj, const, players):
        super().__init__(obj, const, players)

    def min_func(self, x):
        x_jax = jnp.asarray(x)
        actions = x_jax[:self.total_actions].reshape(-1, 1)
        duals   = x_jax[self.total_actions:].reshape(-1, 1)

        vector_actions = construct_vectors(actions, self.action_sizes)

        # Precompute all objective and constraint gradients
        obj_grads  = [d(vector_actions) for d in self.obj_derivatives]
        cons_grads = [d(vector_actions) for d in self.const_derivatives]

        total = 0.0
        for p_idx, obj_idx in enumerate(self.player_obj_idx):
            # Gradient of player's objective w.r.t. their own actions
            grad = obj_grads[obj_idx][p_idx].ravel()

            # Add weighted constraint gradients
            for c_idx in self.players[p_idx].constraints:
                lam = float(duals[c_idx])
                grad = grad + lam * cons_grads[c_idx][p_idx].ravel()

            total += float(jnp.sum(grad ** 2))

        return total
```

To use it, pass the class name to `problem.solve`:

```python
from problems.gnep import MyProblem

problem = MyProblem()
problem.set_initial_point(1.0, 0.1)
result, elapsed = problem.solve(StationarityMethod)
```

## A Good Checklist

Before using a custom algorithm, check that:

- `__init__` accepts exactly `(self, obj, const, players)` and calls `super().__init__()`
- `min_func` accepts a single flat array and returns a Python `float`
- the primal slice is `x[:self.total_actions]` and the dual slice is `x[self.total_actions:]`
- JAX arrays are converted to Python floats before returning
- the merit function returns values close to zero at equilibrium

## Next Step

After implementing an algorithm, the `Choosing an Algorithm` guide describes when to prefer one approach over another and what convergence behaviour to expect from each.
