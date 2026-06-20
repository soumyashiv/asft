"""
ASFT — Adaptive Sparse Fine-Tuning Framework
=============================================

A next-generation, hardware-adaptive AI learning framework that dramatically
reduces training cost and time while improving accuracy, reliability, and
reasoning quality.

Learning Priority Hierarchy:
    Memory → Workflow Optimization → Tool Learning →
    Skill Packs → Sparse Fine-Tuning → Full Fine-Tuning

Full retraining is always the last resort.
"""

__version__ = "0.1.0"
__author__ = "ASFT Contributors"
__license__ = "MIT"

from asft.core.config import ASFTConfig
from asft.core.hardware_profiler import HardwareProfile, detect_hardware
from asft.core.registry import Registry

__all__ = [
    "__version__",
    "ASFTConfig",
    "detect_hardware",
    "HardwareProfile",
    "Registry",
]
