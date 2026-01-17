"""Control module for counterfactual generation.

Provides gradient-based and Jacobian-based controllers as alternatives
to diffusion-based counterfactual generation.
"""

from .gradient_optimizer import GradientBasedController, GradientOptimizerConfig
from .jacobian_controller import JacobianController, JacobianConfig
from .controller import CausalController

__all__ = [
    "GradientBasedController",
    "GradientOptimizerConfig",
    "JacobianController",
    "JacobianConfig",
    "CausalController",
]
