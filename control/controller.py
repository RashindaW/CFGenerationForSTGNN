"""Unified controller interface for counterfactual generation.

Provides a single entry point for different control strategies.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .gradient_optimizer import GradientBasedController, GradientOptimizerConfig
from .jacobian_controller import JacobianController, JacobianConfig
from .perturbation_controller import PerturbationController, PerturbationConfig


class CausalController:
    """Unified interface supporting multiple control strategies.

    Supports:
        - "gradient": Gradient-based optimization
        - "jacobian": Jacobian-based linear approximation
    """

    def __init__(
        self,
        forecaster: nn.Module,
        control_indices: torch.Tensor,
        target_indices: torch.Tensor,
        adjacency: torch.Tensor,
        method: str = "gradient",
        x_bounds: Optional[Tuple[float, float]] = None,
        model_type: str = "stgcn",
        **method_kwargs,
    ):
        """Initialize the unified controller.

        Args:
            forecaster: Trained forecaster model (horizon=1).
            control_indices: Indices of control nodes in the graph.
            target_indices: Indices of target nodes to match.
            adjacency: Graph adjacency matrix.
            method: Control method - "gradient" or "jacobian".
            x_bounds: (min, max) bounds for control values.
            model_type: Type of forecaster model.
            **method_kwargs: Method-specific configuration parameters.
        """
        self.method = method
        self.control_indices = control_indices
        self.target_indices = target_indices

        if method == "gradient":
            config = GradientOptimizerConfig(
                n_steps=method_kwargs.get("n_steps", 100),
                lr=method_kwargs.get("lr", 0.01),
                lambda_reg=method_kwargs.get("lambda_reg", 0.1),
                optimizer_type=method_kwargs.get("optimizer_type", "adam"),
                convergence_tol=method_kwargs.get("convergence_tol", 1e-6),
                grad_clip=method_kwargs.get("grad_clip", 1.0),
                target_weight=method_kwargs.get("target_weight", 1.0),
            )
            self._controller = GradientBasedController(
                forecaster=forecaster,
                control_indices=control_indices,
                target_indices=target_indices,
                adjacency=adjacency,
                config=config,
                x_bounds=x_bounds,
                model_type=model_type,
            )
        elif method == "jacobian":
            config = JacobianConfig(
                regularization=method_kwargs.get("regularization", 1e-4),
                max_delta=method_kwargs.get("max_delta", None),
                max_iterations=method_kwargs.get("max_iterations", 10),
                convergence_tol=method_kwargs.get("convergence_tol", 1e-4),
            )
            self._controller = JacobianController(
                forecaster=forecaster,
                control_indices=control_indices,
                target_indices=target_indices,
                adjacency=adjacency,
                config=config,
                x_bounds=x_bounds,
                model_type=model_type,
            )
        elif method == "perturbation":
            config = PerturbationConfig(
                n_random_samples=method_kwargs.get("n_random_samples", 200),
                perturbation_scale=method_kwargs.get("perturbation_scale", 0.1),
                n_refine_steps=method_kwargs.get("n_refine_steps", 50),
                refine_lr=method_kwargs.get("refine_lr", 0.01),
                finite_diff_eps=method_kwargs.get("finite_diff_eps", 1e-4),
                convergence_tol=method_kwargs.get("convergence_tol", 1e-6),
            )
            # For causal_forecaster, get manipulated_indices from the model
            manipulated_indices = None
            if model_type == "causal_forecaster" and hasattr(forecaster, "manipulated_indices"):
                manipulated_indices = forecaster.manipulated_indices
            self._controller = PerturbationController(
                forecaster=forecaster,
                control_indices=control_indices,
                target_indices=target_indices,
                adjacency=adjacency,
                config=config,
                x_bounds=x_bounds,
                model_type=model_type,
                manipulated_indices=manipulated_indices,
            )
        else:
            raise ValueError(f"Unknown control method: {method}. Use 'gradient', 'jacobian', or 'perturbation'.")

    def control(
        self,
        current_window: torch.Tensor,
        y_desired: torch.Tensor,
        target_channel: int = 0,
    ) -> Dict[str, Any]:
        """Find optimal control using the configured method.

        Args:
            current_window: Input window of shape (lag, nodes, features).
            y_desired: Desired target values of shape (num_target, 1) or (num_target,).
            target_channel: Which feature channel to optimize for.

        Returns:
            Dictionary with:
                x_optimal: Optimized control values (num_control, features).
                y_predicted: Predicted output with optimal controls (num_target, 1).
                method_metadata: Method-specific info (loss_history, jacobian, etc.).
        """
        if self.method == "gradient":
            x_optimal, y_predicted, loss_history = self._controller.find_intervention(
                current_window, y_desired, target_channel
            )
            return {
                "x_optimal": x_optimal,
                "y_predicted": y_predicted,
                "method_metadata": {
                    "loss_history": loss_history,
                    "method": "gradient",
                },
            }
        elif self.method == "jacobian":
            x_new, delta_x, y_predicted, jacobian_info = self._controller.find_intervention(
                current_window, y_desired, target_channel
            )
            # Extract jacobian from info dict for sensitivity analysis
            jacobian = jacobian_info.get("jacobian")
            sensitivities = (
                self._controller.analyze_sensitivities(jacobian)
                if jacobian is not None
                else {}
            )
            return {
                "x_optimal": x_new,
                "y_predicted": y_predicted,
                "method_metadata": {
                    "delta_x": delta_x,
                    "jacobian": jacobian,
                    "sensitivities": sensitivities,
                    "method": "jacobian",
                    # Include iteration diagnostics
                    "iterations": jacobian_info.get("iterations", 1),
                    "initial_error": jacobian_info.get("initial_error"),
                    "final_error": jacobian_info.get("final_error"),
                    "improvement_percent": jacobian_info.get("improvement_percent"),
                    "converged": jacobian_info.get("converged", False),
                },
            }
        elif self.method == "perturbation":
            x_optimal, y_predicted, loss_history = self._controller.find_intervention(
                current_window, y_desired, target_channel
            )
            return {
                "x_optimal": x_optimal,
                "y_predicted": y_predicted,
                "method_metadata": {
                    "loss_history": loss_history,
                    "method": "perturbation",
                },
            }
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def apply_control(
        self,
        current_window: torch.Tensor,
        x_optimal: torch.Tensor,
    ) -> torch.Tensor:
        """Apply optimized control values to window.

        Args:
            current_window: Original window (lag, nodes, features).
            x_optimal: Optimized control values (num_control, features).

        Returns:
            Modified window with controls applied at last timestep.
        """
        return self._controller.apply_control(current_window, x_optimal)

    @property
    def device(self) -> torch.device:
        """Get the device of the underlying controller."""
        return self._controller.device
