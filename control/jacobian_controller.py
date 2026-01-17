"""Jacobian-based controller for counterfactual generation.

Uses linear approximation via Jacobian for small control changes.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from train import forward_pass


def prepare_forecaster_input(x: torch.Tensor) -> torch.Tensor:
    """Reorder samples into the (B, C, N, T) layout expected by ST-GNNs."""
    return x.permute(0, 3, 2, 1).contiguous()


@dataclass
class JacobianConfig:
    """Configuration for Jacobian-based control."""

    regularization: float = 1e-4  # Tikhonov regularization for pseudo-inverse
    max_delta: Optional[float] = None  # Max allowed change magnitude per control


class JacobianController:
    """Jacobian-based controller using linear approximation.

    Mathematical Basis:
        For small changes: ΔY ≈ J · ΔX
        Therefore: ΔX ≈ J⁺ · ΔY (pseudo-inverse solution)

    Where J = ∂Y/∂X is the Jacobian matrix of the forecaster
    with respect to control node inputs.
    """

    def __init__(
        self,
        forecaster: nn.Module,
        control_indices: torch.Tensor,
        target_indices: torch.Tensor,
        adjacency: torch.Tensor,
        config: JacobianConfig,
        x_bounds: Optional[Tuple[float, float]] = None,
        model_type: str = "stgcn",
        lag_weights: Optional[torch.Tensor] = None,
        neighbor_only_inputs: bool = False,
    ):
        """Initialize the Jacobian-based controller.

        Args:
            forecaster: Trained forecaster model (horizon=1).
            control_indices: Indices of control nodes in the graph.
            target_indices: Indices of target nodes to match.
            adjacency: Graph adjacency matrix.
            config: Controller configuration.
            x_bounds: (min, max) bounds for control values.
            model_type: Type of forecaster model.
            lag_weights: Optional weights for lag dimensions.
            neighbor_only_inputs: Whether to use neighbor-only inputs.
        """
        self.forecaster = forecaster
        self.control_indices = control_indices
        self.target_indices = target_indices
        self.adjacency = adjacency
        self.config = config
        self.x_bounds = x_bounds
        self.model_type = model_type
        self.lag_weights = lag_weights
        self.neighbor_only_inputs = neighbor_only_inputs

        self.device = next(forecaster.parameters()).device
        self.forecaster.eval()

    def compute_jacobian(
        self,
        current_window: torch.Tensor,
        target_channel: int = 0,
    ) -> torch.Tensor:
        """Compute Jacobian ∂Y/∂X using automatic differentiation.

        Args:
            current_window: Input window of shape (lag, nodes, features).
            target_channel: Which feature channel targets are for.

        Returns:
            jacobian: Matrix of shape (num_target, num_control * features).
        """
        current_window = current_window.to(self.device)

        num_control = len(self.control_indices)
        num_features = current_window.shape[-1]
        num_target = len(self.target_indices)

        # Create a copy of the window with requires_grad on control nodes at last timestep
        window = current_window.clone()

        # Extract control values as learnable parameters
        x_control = window[-1, self.control_indices, :].clone().requires_grad_(True)

        # Substitute back into window
        window_modified = window.clone()
        window_modified[-1, self.control_indices, :] = x_control

        # Forward pass
        forecaster_input = prepare_forecaster_input(window_modified.unsqueeze(0))

        y_pred_full = forward_pass(
            self.forecaster,
            forecaster_input,
            self.model_type,
            lag_weights=self.lag_weights,
            adjacency=self.adjacency,
            neighbor_only_inputs=self.neighbor_only_inputs,
        )

        # Extract target predictions: (1, nodes, horizon) -> (num_target,)
        y_pred = y_pred_full[0, self.target_indices, 0]  # horizon=1

        # Compute Jacobian row by row using backward passes
        jacobian_rows = []
        for i in range(num_target):
            if x_control.grad is not None:
                x_control.grad.zero_()

            # Backprop from single target output
            y_pred[i].backward(retain_graph=True)

            # Get gradient w.r.t. control inputs
            grad = x_control.grad.clone()  # (num_control, features)
            jacobian_rows.append(grad.view(-1))  # Flatten to (num_control * features,)

        jacobian = torch.stack(jacobian_rows, dim=0)  # (num_target, num_control * features)

        return jacobian

    def find_intervention(
        self,
        current_window: torch.Tensor,
        y_desired: torch.Tensor,
        target_channel: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find intervention using Jacobian pseudo-inverse.

        Args:
            current_window: Input window of shape (lag, nodes, features).
            y_desired: Desired target values of shape (num_target, 1) or (num_target,).
            target_channel: Which feature channel to optimize for.

        Returns:
            x_new: New control values (num_control, features).
            delta_x: Change from current (num_control, features).
            y_predicted: Predicted output with new controls (num_target, 1).
            jacobian: The computed Jacobian matrix.
        """
        current_window = current_window.to(self.device)
        y_desired = y_desired.to(self.device)

        if y_desired.dim() == 2:
            y_desired = y_desired.squeeze(-1)  # (num_target,)

        num_control = len(self.control_indices)
        num_features = current_window.shape[-1]

        # Get current control values and prediction
        x_current = current_window[-1, self.control_indices, :].clone()

        with torch.no_grad():
            forecaster_input = prepare_forecaster_input(current_window.unsqueeze(0))
            y_current_full = forward_pass(
                self.forecaster,
                forecaster_input,
                self.model_type,
                lag_weights=self.lag_weights,
                adjacency=self.adjacency,
                neighbor_only_inputs=self.neighbor_only_inputs,
            )
            y_current = y_current_full[0, self.target_indices, 0]  # (num_target,)

        # Compute desired change
        delta_y = y_desired - y_current  # (num_target,)

        # Compute Jacobian
        jacobian = self.compute_jacobian(current_window, target_channel)

        # Solve for delta_x using regularized pseudo-inverse
        # J^+ = (J^T J + λI)^{-1} J^T  (Tikhonov regularization)
        JtJ = jacobian.T @ jacobian
        reg_term = self.config.regularization * torch.eye(
            JtJ.shape[0], device=self.device, dtype=JtJ.dtype
        )
        Jt_dy = jacobian.T @ delta_y

        # Solve linear system
        delta_x_flat = torch.linalg.solve(JtJ + reg_term, Jt_dy)

        # Reshape to (num_control, features)
        delta_x = delta_x_flat.view(num_control, num_features)

        # Apply max_delta constraint if specified
        if self.config.max_delta is not None:
            delta_norm = delta_x.abs().max()
            if delta_norm > self.config.max_delta:
                delta_x = delta_x * (self.config.max_delta / delta_norm)

        # Compute new control values
        x_new = x_current + delta_x

        # Apply bounds
        x_new = self._project_constraints(x_new)

        # Get prediction with new controls
        with torch.no_grad():
            window_modified = current_window.clone()
            window_modified[-1, self.control_indices, :] = x_new

            forecaster_input = prepare_forecaster_input(window_modified.unsqueeze(0))
            y_pred_full = forward_pass(
                self.forecaster,
                forecaster_input,
                self.model_type,
                lag_weights=self.lag_weights,
                adjacency=self.adjacency,
                neighbor_only_inputs=self.neighbor_only_inputs,
            )
            y_predicted = y_pred_full[0, self.target_indices, :1]  # (num_target, 1)

        return x_new, delta_x, y_predicted, jacobian

    def analyze_sensitivities(
        self,
        jacobian: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Analyze which controls most affect which targets.

        Args:
            jacobian: Jacobian matrix of shape (num_target, num_control * features).

        Returns:
            Dictionary with:
                sensitivity_matrix: Absolute Jacobian values (num_target, num_control, features).
                control_importance: Total influence per control node (num_control,).
                target_sensitivity: Total sensitivity per target node (num_target,).
        """
        num_control = len(self.control_indices)
        num_features = jacobian.shape[1] // num_control
        num_target = jacobian.shape[0]

        # Reshape to (num_target, num_control, features)
        jacobian_3d = jacobian.view(num_target, num_control, num_features)

        # Absolute sensitivities
        sensitivity_matrix = jacobian_3d.abs()

        # Total influence per control node (sum over targets and features)
        control_importance = sensitivity_matrix.sum(dim=(0, 2))  # (num_control,)

        # Total sensitivity per target (sum over controls and features)
        target_sensitivity = sensitivity_matrix.sum(dim=(1, 2))  # (num_target,)

        return {
            "sensitivity_matrix": sensitivity_matrix,
            "control_importance": control_importance,
            "target_sensitivity": target_sensitivity,
        }

    def _project_constraints(self, x: torch.Tensor) -> torch.Tensor:
        """Project control values onto feasible bounds."""
        if self.x_bounds is not None:
            x_min, x_max = self.x_bounds
            x = torch.clamp(x, min=x_min, max=x_max)
        return x

    def apply_control(
        self,
        current_window: torch.Tensor,
        x_new: torch.Tensor,
    ) -> torch.Tensor:
        """Apply new control values to the window.

        Args:
            current_window: Original window (lag, nodes, features).
            x_new: New control values (num_control, features).

        Returns:
            Modified window with control values applied at last timestep.
        """
        modified_window = current_window.clone()
        modified_window[-1, self.control_indices, :] = x_new
        return modified_window
