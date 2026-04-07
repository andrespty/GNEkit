from typing import List, Optional, Tuple
from dataclasses import dataclass
from solvers.CorePlayer import PlayerValidator
from solvers.gnep_solver import Player
import numpy as np

from solvers.gnep_solver import Player

@dataclass(frozen=True)
class BayesianPlayer(Player):
    type_values: Tuple[float, ...] = ()
    type_probs: Optional[Tuple[float, ...]] = None
    action_size_per_type: int = 1

    def __post_init__(self):
        super().__post_init__()  # Validate base Player attributes first

        if len(self.type_values) == 0:
            raise ValueError("type_values must be non-empty.")

        if self.action_size_per_type <= 0:
            raise ValueError("action_size_per_type must be positive.")

        expected_size = len(self.type_values) * self.action_size_per_type
        if self.size != expected_size:
            raise ValueError(
                f"size must equal len(type_values) * action_size_per_type. "
                f"Got size={self.size}, expected={expected_size}."
            )

        if self.type_probs is not None:
            if len(self.type_probs) != len(self.type_values):
                raise ValueError(
                    "type_probs must have the same length as type_values."
                )
            if not np.isclose(sum(self.type_probs), 1.0):
                raise ValueError("type_probs must sum to 1.0.")
            
    @property
    def n_types(self) -> int:
        return len(self.type_values)
    


    @classmethod
    def batch_create(cls, sizes, type_values, objectives,constraints,bounds=None,names=None, type_probs=None,
        action_size_per_type=None,):
        n_var = len(sizes)

        if bounds is None:
            bounds = [None] * n_var

        if names is None:
            names = [f"P{i}" for i in range(n_var)]

        if type_probs is None:
            type_probs = [None] * n_var

        if action_size_per_type is None:
            action_size_per_type = [1] * n_var

        if not (len(sizes)
                == len(type_values) 
                == len(objectives) 
                == len(constraints) 
                == len(bounds) 
                == len(names)
                == len(type_probs)
                == len(action_size_per_type)
            ):
            raise ValueError("All player attribute lists must have the same length.")
        
        return [
            cls(
                name=name,
                size=size,
                f_index=obj,
                constraints=const,
                bounds=bound,
                type_values=tuple(tvals),
                type_probs=None if probs is None else tuple(probs),
                action_size_per_type=a_per_type,
            )
            for name, size, tvals, obj, const, bound, probs, a_per_type in zip(
                names,
                sizes,
                type_values,
                objectives,
                constraints,
                bounds,
                type_probs,
                action_size_per_type,
            )
        ]

def bayesian_players_to_lists(players):
    return {
        "sizes": [p.size for p in players],
        "type_values": [p.type_values for p in players],
        "type_probs": [p.type_probs for p in players],
        "action_size_per_type": [p.action_size_per_type for p in players],
        "objectives": [p.f_index for p in players],
        "constraints": [p.constraints for p in players],
        "bounds": [bound for p in players for bound in p.get_full_bounds()],
        "names": [p.name for p in players],
    }