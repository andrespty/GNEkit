from typing import List, Optional, Tuple

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

@dataclass(frozen=True)
class GNEPlayer:
    name: Optional[str]
    size: int
    f_index: int
    constraints: Tuple[Optional[int], ...] = ()
    bounds: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        # -------------------
        # Name validation
        # -------------------
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("Player name must be a string or None.")

        # -------------------
        # Action size validation
        # -------------------
        if not isinstance(self.size, int):
            raise TypeError("action_size must be an integer.")
        if self.size <= 0:
            raise ValueError("action_size must be greater than 0.")

        # -------------------
        # Objective index validation
        # -------------------
        if not isinstance(self.f_index, int):
            raise TypeError("obj_func_idx must be an integer.")
        if self.f_index < 0:
            raise ValueError("obj_func_idx must be >= 0.")

        # -------------------
        # Constraints validation
        # -------------------
        for c in self.constraints:
            if c is not None and (not isinstance(c, int) or c < 0):
                raise ValueError("constraints must contain integers >= 0 or None.")

        # -------------------
        # Bounds validation
        # -------------------
        if self.bounds is not None:
            # Handle single tuple case: (lower, upper)
            if isinstance(self.bounds, tuple) and len(self.bounds) == 2:
                self._validate_bound_pair(self.bounds)

            # Handle list of tuples case: [(l1, u1), (l2, u2), ...]
            elif isinstance(self.bounds, list):
                if len(self.bounds) != self.size:
                    raise ValueError(f"Player {self.name} has {self.size} variables but {len(self.bounds)} bounds.")
                for b in self.bounds:
                    self._validate_bound_pair(b)
            else:
                raise TypeError("bounds must be a tuple (lb, ub) or a list of tuples.")


    @staticmethod
    def _validate_bound_pair(b):
        if not isinstance(b, tuple) or len(b) != 2:
            raise TypeError("Each bound must be a tuple (lower, upper).")
        lower, upper = b
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            raise TypeError("Bounds must be numeric.")
        if lower >= upper:
            raise ValueError(f"Lower bound {lower} must be less than upper bound {upper}.")

    def get_full_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (lower_bounds, upper_bounds) as arrays of length self.size."""
        if self.bounds is None:
            # Default to large values if no bounds provided
            return np.full((self.size,), -np.inf), np.full((self.size,), np.inf)

        if isinstance(self.bounds, tuple):
            lb, ub = self.bounds
            return np.full((self.size,), lb), np.full((self.size,), ub)

        # List case
        lbs = np.array([b[0] for b in self.bounds]).reshape(-1, )
        ubs = np.array([b[1] for b in self.bounds]).reshape(-1, )
        return lbs, ubs

    # -------------------
    # Helper constructor
    # -------------------
    @classmethod
    def batch_create(
        cls,
        sizes,
        objectives,
        constraints,
        bounds=None,
        names=None,
    ):
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

def players_to_lists(players: List[GNEPlayer]):
    return {
        "sizes": [p.size for p in players],
        "objectives": [p.f_index for p in players],
        "constraints": [p.constraints for p in players],
        "bounds": [p.get_full_bounds() for p in players],
        "names": [p.name for p in players],
    }

