# Type Reference

Types used throughout the solver.

## Array types

| Alias | Definition | Description |
|-------|-----------|-------------|
| `Vector` | `Array` | A JAX array with shape (x, 1) |
| `Matrix` | `Array` | A JAX array with shape (x, y) |
| `VectorList` | `List[Vector]` | A python list of `Vector` |
| `PlayerConstraint` | `List[int] | None` | List of constraint indices for a player. `None` means no constraints assigned. Must match with `constraints()` in the [`Problem`](../problems/base_problem.md) |

## Function types

| Alias | Signature | Description |
|-------|-----------|-------------|
| `ObjFunction` | `(VectorList) -> float` | Maps the full action profile to a scalar cost. |
| `ObjFunctionGrad` | `(VectorList) -> Vector` | Gradient of an objective w.r.t. a single player's variables. |
| `ConsFunction` | `(VectorList) -> Vector` | Constraint function, returns violations `g(x) <= 0`. |
| `ConsFunctionGrad` | `(VectorList) -> Matrix` | Jacobian of the constraints. |
| `WrappedFunction` | `(Array) -> Array` | Flattened input/output function used internally by the solver. |