"""Environment registry for the Section 3.2 classification MDPs."""

from .easy_mdp import EasyMDP
from .hard_mdp import HardMDP
from .sparse_mdp import SparseMDP

__all__ = [
    "EasyMDP",
    "HardMDP",
    "SparseMDP",
]
