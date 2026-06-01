# Dynamic ART-DRL Neural Network Models Package
"""
models
======

PyTorch modules for the Dynamic ART-DRL trading system.

Components
----------
- :class:`GatedResidualNetwork`       — GRN building block (TFT)
- :class:`VariableSelectionNetwork`   — Adaptive feature weighting
- :class:`StateEncoder`               — VSN + Transformer → state vector
"""

from models.grn import GatedResidualNetwork
from models.vsn import VariableSelectionNetwork
from models.state_encoder import StateEncoder

__all__ = [
    "GatedResidualNetwork",
    "VariableSelectionNetwork",
    "StateEncoder",
]
