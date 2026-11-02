# Creating a Custom Algorithm

This guide explains how to implement a custom algorithm that works with `GNEkit`'s problem and solver infrastructure.

Every algorithm in this library is a subclass of `BaseAlgorithm`. The base class handles the bookkeeping (loading players, compiling derivatives, managing bounds, running the optimizer, checking the KKT conditions), so you only need to define the **merit function** that the optimizer minimizes.

## The Required Structure

A merit function in `GNEkit` has a primal part and a dual part. A new algorithm implements one method for each:

```python
from gnekit import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    def lagrange_gradient(self, actions, dual_actions):
        ...
        return primal_terms   # shape (total_actions, 1)

    def gradient_dual(self, actions, dual_actions):
        ...
        return dual_terms     # shape (num_constraints, 1)
```

The merit function minimized by the solver is

```
sum(lagrange_gradient(actions, dual_actions)) + sum(gradient_dual(actions, dual_actions))
```

and it is JIT-compiled with JAX, so both methods must use `jax.numpy` and avoid Python side effects.

You do not need to write `__init__`. If you do, forward all arguments to the base class so that solver settings keep working:

```python
class MyAlgorithm(BaseAlgorithm):
    def __init__(self, obj, const, players, *args, my_param=1.0, **kwargs):
        super().__init__(obj, const, players, *args, **kwargs)
        self.my_param = my_param
```

!!! note
    If your merit function does not split into a primal and a dual part, override `min_func(self, x)` instead. It receives the flat vector `x = [primal | dual]` and must return a Python `float`.

## What `BaseAlgorithm` Provides

After construction, the following attributes are available:

**Problem structure:**

- `self.total_actions`: total number of primal variables across all players
- `self.action_sizes`: list of action vector sizes per player, e.g. `[2, 3, 2]`
- `self.action_splits`: cumulative split indices for slicing the flat primal vector
- `self.player_obj_idx`: which objective function index belongs to each player
- `self.player_const_idx`: which constraint indices apply to each player
- `self.player_const_idx_matrix`: a `(total_actions, num_constraints)` one-hot matrix encoding which player is responsible for each constraint

**Functions and derivatives:**

- `self.obj_functions`: validated list of objective functions
- `self.const`: validated list of constraint functions
- `self.obj_derivatives`: JIT-compiled gradients of each objective (via `jax.grad`)
- `self.const_derivatives`: JIT-compiled gradients of each constraint (via `jax.grad`)

**Building blocks:**

- `self.objective_gradient(vector_actions)`: each player's gradient of its own objective, stacked into a `(total_actions, 1)` vector
- `self.constraint_gradient(vector_actions, dual_actions)`: multiplier-weighted constraint gradients, restricted to the players in each constraint, as a `(total_actions, 1)` vector
- `self.constraint_values(vector_actions)`: the value `g_j(x)` for each multiplier, as a `(num_constraint_duals,)` vector

**Multipliers:**

- `self.num_constraint_duals`: number of constraint multipliers
- `self.multiplier_pairs`: the `(player, constraint)` pair of each multiplier. In `"shared"` mode the player is `None`
- `self.dual_index(player, constraint)`: position of the multiplier a player uses for a constraint

Use `constraint_values` and `num_constraint_duals` instead of `self.const` and `len(self.const)` so your algorithm works in both multiplier modes.

Their sum is the gradient of the Lagrangian.

**Bounds:**

- `self.bounds`: list of `(lb, ub)` tuples for primal variables
- `self.bounds_dual`: list of `(0, 100)` tuples for dual variables, one per constraint multiplier
- `self.bounds_all`: `self.bounds + self.bounds_dual`, passed directly to the optimizer
- `self.lb_vector`, `self.ub_vector`: primal bounds as JAX column vectors

## What the Methods Receive

Both `lagrange_gradient` and `gradient_dual` receive:

- `actions`: the stacked primal variables of all players, shape `(total_actions, 1)`
- `dual_actions`: the constraint multipliers, shape `(num_constraint_duals, 1)`

To split the primal vector into per-player vectors, use `construct_vectors`:

```python
from gnekit.core.utils import construct_vectors

vector_actions = construct_vectors(actions, self.action_sizes)
# vector_actions[i] is player i's action as a column vector
```

## How `solve` Works

`problem.solve(MyAlgorithm)` builds the algorithm from the problem's objectives, constraints and players, then calls `BaseAlgorithm.solve(ip)` with the stored initial point. This runs `scipy.optimize.basinhopping` with a local minimizer (SLSQP by default) and `self.bounds_all` as the variable bounds.

When it finishes, it prints a result summary and runs `check_kkt` to verify the equilibrium conditions.

Solver settings are passed as keyword arguments to `problem.solve`:

```python
primal_x, dual_x = problem.solve(
    MyAlgorithm,
    minimizer_options={"ftol": 1e-20, "maxiter": 5000},
    niter=200,
)
```

| Setting | Default | Meaning |
|---------|---------|---------|
| `method` | `"SLSQP"` | Local minimizer passed to `scipy.optimize.minimize` |
| `minimizer_options` | `None` | `options` dict for the local minimizer |
| `niter` | `1000` | Basin-hopping iterations |
| `stepsize` | `0.01` | Basin-hopping step size |
| `interval` | `1` | Iterations between step size updates |
| `niter_success` | `100` | Stop after this many iterations without improvement |
| `disp` | `True` | Print basin-hopping progress |
| `multipliers` | `"shared"` | `"shared"`: one multiplier per constraint (variational equilibrium). `"per_player"`: one per player and constraint (full GNEP) |

!!! tip
    Merit functions that are quadratic in the KKT residual reach small values long before the residual is small. Tighten `ftol` in `minimizer_options` when you need an accurate equilibrium.

You do not need to override `solve`. If you want a different optimizer altogether, override `solve(self, ip)` and return `(result, elapsed_time)`, where `result.x` is the flat solution vector.

## A Complete Example

The following algorithm uses the squared KKT residual as its merit function. The primal part is the squared gradient of the Lagrangian, and the dual part is the squared complementarity `(λ g(x))²`. Both vanish at a KKT point of an interior solution.

```python
import jax.numpy as jnp
from gnekit import BaseAlgorithm
from gnekit.core.utils import construct_vectors


class StationarityMethod(BaseAlgorithm):
    def lagrange_gradient(self, actions, dual_actions):
        vector_actions = construct_vectors(actions, self.action_sizes)
        grad = self.objective_gradient(vector_actions)
        if self.const_derivatives:
            grad += self.constraint_gradient(vector_actions, dual_actions)
        return grad ** 2

    def gradient_dual(self, actions, dual_actions):
        if not self.const:
            return jnp.zeros((1, 1))
        vector_actions = construct_vectors(actions, self.action_sizes)
        g = self.constraint_values(vector_actions).reshape(-1, 1)
        return (dual_actions * g) ** 2
```

To use it, pass the class to `problem.solve`:

```python
from gnekit.problems.gnep import ProblemA1

problem = ProblemA1()
problem.set_initial_point(1.0, 0.1)
primal_x, dual_x = problem.solve(StationarityMethod, niter=50)
```

This example ignores the variable bounds and the sign of `g`, so it is only a starting point. See `EnergyMethod` for a merit function that handles both.

## A Good Checklist

Before using a custom algorithm, check that:

- it subclasses `BaseAlgorithm` and implements `lagrange_gradient` and `gradient_dual` (or `min_func`)
- both methods use `jax.numpy` and return column vectors
- any custom `__init__` forwards `*args, **kwargs` to `super().__init__()`
- the merit function is zero exactly at equilibrium
