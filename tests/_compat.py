"""Single place that knows where library objects live.

The baseline tests import everything through this module, so moving files
during the restructure only requires updating the imports below.
"""
from gnekit.problems.gnep import (
    ProblemA1, ProblemA2, ProblemA3, ProblemA4, ProblemA5, ProblemA6, ProblemA7,
    ProblemA8, ProblemA9a, ProblemA9b, ProblemA10a, ProblemA11, ProblemA12,
    ProblemA13, ProblemA14, ProblemA15, ProblemA16, ProblemA17, ProblemA18,
)
from gnekit.problems.bayesian import AllocationGame, QuadraticGame, RadarPowerGame, RadarGameDGBNE
from gnekit.problems.bayesian.PayoffMatrixGame import PayoffMatrixGame
from gnekit.algorithms import EnergyMethod, FBEnergyMethod

PROBLEMS = {
    "A1": ProblemA1, "A2": ProblemA2, "A3": ProblemA3, "A4": ProblemA4,
    "A5": ProblemA5, "A6": ProblemA6, "A7": ProblemA7, "A8": ProblemA8,
    "A9a": ProblemA9a, "A9b": ProblemA9b, "A10a": ProblemA10a, "A11": ProblemA11,
    "A12": ProblemA12, "A13": ProblemA13, "A14": ProblemA14, "A15": ProblemA15,
    "A16": ProblemA16, "A17": ProblemA17, "A18": ProblemA18,
    "AllocationGame": AllocationGame, "QuadraticGame": QuadraticGame,
    "RadarPowerGame": RadarPowerGame, "PayoffMatrixGame": PayoffMatrixGame,
    "RadarGameDGBNE": RadarGameDGBNE,
}

ALGORITHMS = {
    "EnergyMethod": EnergyMethod,
    "FBEnergyMethod": FBEnergyMethod,
}
