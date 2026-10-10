"""chargestates: charge states of a Coulomb impurity in gapped 2D Dirac materials.

Many-body (no-pair, Furry-picture) treatment of the electrons bound to the impurity,
built on the single-particle machinery of collapse2d (paper 1).
"""
from .deps import collapse2d  # noqa: F401
from .furry import Grid, FurryBasis  # noqa: F401

__version__ = "1.0.1"
