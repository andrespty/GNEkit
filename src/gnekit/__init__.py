"""
GNEkit: modeling and solving generalized Nash equilibrium problems.

The names most problems need are available directly from ``gnekit``::

    from gnekit import BaseProblem, Player, EnergyMethod

Benchmark problems live in ``gnekit.problems.gnep`` and
``gnekit.problems.bayesian``.
"""
from gnekit.core import BaseProblem, Player
from gnekit.core.types import VectorList
from gnekit.bayesian import BayesianPlayer, BayesianProblem
from gnekit.algorithms import BaseAlgorithm, EnergyMethod, FBEnergyMethod

__version__ = "0.1.0"
__author__ = "Andres Ho"

__all__ = [
    # Problems and players
    "BaseProblem",
    "Player",
    "BayesianProblem",
    "BayesianPlayer",
    # Algorithms
    "BaseAlgorithm",
    "EnergyMethod",
    "FBEnergyMethod",
    # Types
    "VectorList",
]
