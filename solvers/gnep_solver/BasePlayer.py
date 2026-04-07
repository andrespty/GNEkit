from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import jax.numpy as jnp
from solvers.CorePlayer import PlayerValidator

@dataclass(frozen=True)
class Player:
    name: Optional[str]
    size: int
    f_index: int
    constraints: Tuple[Optional[int], ...] = ()
    bounds: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        validator = PlayerValidator()
        validator.validate(self.name, self.size, self.f_index, self.constraints, self.bounds)

    def get_full_bounds(self):
        """Returns (lower_bounds, upper_bounds) as arrays of length self.size."""
        if self.bounds is None:
            # Default to large values if no bounds provided
            lb_arr = np.full((self.size,), -np.inf)
            ub_arr = np.full((self.size,), np.inf)

        elif isinstance(self.bounds, tuple):
            lb, ub = self.bounds
            # Ensure lb/ub are scalars or compatible with np.full
            lb_arr = np.full((self.size,), float(lb))
            ub_arr = np.full((self.size,), float(ub))

        else:
            # Case where self.bounds is already an array/list of bounds
            # Ensure it is a NumPy array to avoid JAX leakage
            bounds_np = np.asarray(self.bounds)
            lb_arr, ub_arr = bounds_np[:, 0], bounds_np[:, 1]

        return list(zip(lb_arr.tolist(), ub_arr.tolist()))

    @classmethod
    def batch_create(cls,sizes,objectives,constraints,bounds=None,names=None):
        if bounds is None:
            bounds = [None] * len(sizes)

        if names is None:
            names = [f"P{i}" for i in range(len(sizes))]

        if not (len(sizes) == len(objectives) == len(constraints) == len(bounds) == len(names)):
            raise ValueError("All player attribute lists must have the same length.")

        return [
            cls(name, size, obj, const, bound)
            for name, size, obj, const, bound
            in zip(names, sizes, objectives, constraints, bounds)
        ]

def players_to_lists(players: List[Player]):
    return {
        "sizes": [p.size for p in players],
        "objectives": [p.f_index for p in players],
        "constraints": [p.constraints for p in players], # nested list of int
        "bounds": [bound for p in players for bound in p.get_full_bounds()], # list of tuples
        "names": [p.name for p in players],
    }
