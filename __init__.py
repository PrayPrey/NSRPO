"""
NSRPO Project - Null-Space Regularized Policy Optimization

A framework for reinforcement learning with null-space decoder regularization.
Combines GRPO (Generalized Reward Preference Optimization) with null-space
projection techniques for better representation learning and training stability.
"""

__version__ = "0.1.0"
__author__ = "NSRPO Development Team"

from . import models
from . import utils

__all__ = ["models", "utils"]