from typing import List, Optional, Tuple
from dataclasses import dataclass
from solvers.CorePlayer import PlayerValidator
from solvers.gnep_solver import Player
import numpy as np

@dataclass(frozen=True)
class BayesianPlayer:
    name: Optional[str]
    size: int
    types: List[int]
    f_index: int
    constraints: Tuple[Optional[int], ...] = ()
    bounds: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        if not isinstance(self.types, list):
            raise TypeError("Types must be a list.")
        PlayerValidator().validate(self.name, self.size, self.f_index, self.constraints, self.bounds)

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
    def batch_create(cls, sizes, types, objectives,constraints,bounds=None,names=None):
        if bounds is None:
            bounds = [None] * len(sizes)

        if names is None:
            names = [f"P{i}" for i in range(len(sizes))]

        if not (len(sizes)== len(types) == len(objectives) == len(constraints) == len(bounds) == len(names)):
            raise ValueError("All player attribute lists must have the same length.")

        return [
            cls(name, size, player_type, obj, const, bound)
            for name, size, player_type, obj, const, bound
            in zip(names, sizes, types, objectives, constraints, bounds)
        ]

def bayesian_players_to_list(player_list:List[BayesianPlayer]):
    return {
        "sizes": [p.size for p in player_list],
        "types": [p.types for p in player_list],
        "objectives": [p.f_index for p in player_list],
        "constraints": [p.constraints for p in player_list], # nested list of int
        "bounds": [bound for p in player_list for bound in p.get_full_bounds()], # list of tuples
        "names": [p.name for p in player_list],
    }

@dataclass(frozen=True)
class JointTypeDistribution:
    type_profiles: List[Tuple[int, ...]]
    probabilities: List[float]

    def __post_init__(self):
        if len(self.type_profiles) != len(self.probabilities):
            raise ValueError("Mismatch in profiles and probabilities.")

        if not np.isclose(sum(self.probabilities), 1.0):
            raise ValueError("Probabilities must sum to 1.")