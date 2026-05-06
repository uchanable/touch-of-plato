"""Dataset loaders for the Platonic Touch paper.

Currently only TVL (Touch-Vision-Language, Fu et al. ICML 2024) is
implemented. See the paper §3.3.
"""
from .tvl import TVLDataset, TVLItem

__all__ = ["TVLDataset", "TVLItem"]
