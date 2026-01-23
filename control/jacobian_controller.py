"""Jacobian-based controller for counterfactual generation.

Uses linear approximation via Jacobian for small control changes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

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
    # Iterative refinement parameters (Gauss-Newton)
    max_iterations: int = 10  # Max iterations for Gauss-Newton refinement
    convergence_tol: float = 1e-4  # Convergence tolerance for residual norm
    line_search_alphas: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.25, 0.1, 0.05]
    )  # Step sizes to try in line search


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
        # Note: We keep the model in its current mode. For RNNs with cuDNN,
        # backward passes require training mode, so we'll handle this in compute_jacobian.

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

        # Save original training mode and set to train for backward pass (required for cuDNN RNNs)
        was_training = self.forecaster.training
        self.forecaster.train()

        # Freeze model parameters - we only want gradients w.r.t. x_control
        original_requires_grad = {}
        for name, param in self.forecaster.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False

        try:
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

        finally:
            # Restore original model state
            for name, param in self.forecaster.named_parameters():
                param.requires_grad = original_requires_grad[name]
            if not was_training:
                self.forecaster.eval()

    def find_intervention(
        self,
        current_window: torch.Tensor,
        y_desired: torch.Tensor,
        target_channel: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Find intervention using iterative Jacobian refinement (Gauss-Newton).

        Uses iterative refinement to handle nonlinearity in the neural network,
        instead of a single linearization step.

        Args:
            current_window: Input window of shape (lag, nodes, features).
            y_desired: Desired target values of shape (num_target, 1) or (num_target,).
            target_channel: Which feature channel to optimize for.

        Returns:
            x_new: New control values (num_control, features).
            delta_x: Change from baseline (num_control, features).
            y_predicted: Predicted output with new controls (num_target, 1).
            jacobian_info: Dictionary with Jacobian and iteration diagnostics.
        """
        current_window = current_window.to(self.device)
        y_desired = y_desired.to(self.device)

        if y_desired.dim() == 2:
            y_desired = y_desired.squeeze(-1)  # (num_target,)

        num_control = len(self.control_indices)
        num_features = current_window.shape[-1]

        # Get baseline control values
        x_baseline = current_window[-1, self.control_indices, :].clone()
        x_current = x_baseline.clone()

        # Get initial prediction for diagnostics
        with torch.no_grad():
            forecaster_input = prepare_forecaster_input(current_window.unsqueeze(0))
            y_initial_full = forward_pass(
                self.forecaster,
                forecaster_input,
                self.model_type,
                lag_weights=self.lag_weights,
                adjacency=self.adjacency,
                neighbor_only_inputs=self.neighbor_only_inputs,
            )
            y_initial = y_initial_full[0, self.target_indices, 0]  # (num_target,)

        initial_error = (y_initial - y_desired).norm().item()

        # Iterative Gauss-Newton refinement
        final_jacobian = None
        iteration = 0
        residual_history = [initial_error]

        print(f"\n[Jacobian] Starting iterative refinement:")
        print(f"  Initial error: {initial_error:.6f}")
        print(f"  y_desired: {y_desired.cpu().numpy()}")
        print(f"  y_initial: {y_initial.cpu().numpy()}")
        print(f"  x_baseline range: [{x_baseline.min().item():.4f}, {x_baseline.max().item():.4f}]")

        for iteration in range(self.config.max_iterations):
            # 1. Create modified window with current control values
            window_modified = current_window.clone()
            window_modified[-1, self.control_indices, :] = x_current

            # 2. Get current prediction
            with torch.no_grad():
                forecaster_input = prepare_forecaster_input(window_modified.unsqueeze(0))
                y_pred_full = forward_pass(
                    self.forecaster,
                    forecaster_input,
                    self.model_type,
                    lag_weights=self.lag_weights,
                    adjacency=self.adjacency,
                    neighbor_only_inputs=self.neighbor_only_inputs,
                )
                y_current_pred = y_pred_full[0, self.target_indices, 0]  # (num_target,)

            # 3. Compute residual (error to target)
            residual = y_desired - y_current_pred  # (num_target,)
            residual_norm = residual.norm().item()
            residual_history.append(residual_norm)

            # Check convergence
            if residual_norm < self.config.convergence_tol:
                print(f"  Iter {iteration}: Converged! residual={residual_norm:.6f}")
                break

            # 4. Compute Jacobian at current point
            jacobian = self.compute_jacobian(window_modified, target_channel)
            final_jacobian = jacobian  # Keep last computed Jacobian for output

            jacobian_norm = jacobian.norm().item()
            jacobian_max = jacobian.abs().max().item()

            # 5. Solve for step direction using regularized pseudo-inverse
            # J^+ = (J^T J + λI)^{-1} J^T  (Tikhonov regularization)
            JtJ = jacobian.T @ jacobian
            reg_term = self.config.regularization * torch.eye(
                JtJ.shape[0], device=self.device, dtype=JtJ.dtype
            )
            Jt_r = jacobian.T @ residual

            # Solve linear system for step direction
            delta_x_flat = torch.linalg.solve(JtJ + reg_term, Jt_r)

            # Reshape to (num_control, features)
            delta_x_step = delta_x_flat.view(num_control, num_features)
            step_norm_before = delta_x_step.norm().item()

            # Apply max_delta constraint to step if specified
            if self.config.max_delta is not None:
                delta_norm = delta_x_step.abs().max()
                if delta_norm > self.config.max_delta:
                    delta_x_step = delta_x_step * (self.config.max_delta / delta_norm)

            # 6. Line search for step size
            alpha = self._line_search(
                current_window, x_current, delta_x_step, y_desired
            )

            step_norm_after = (alpha * delta_x_step).norm().item()
            x_before = x_current.clone()

            # 7. Update control values
            x_current = x_current + alpha * delta_x_step
            x_current = self._project_constraints(x_current)

            actual_change = (x_current - x_before).norm().item()

            print(f"  Iter {iteration}: residual={residual_norm:.6f}, "
                  f"J_norm={jacobian_norm:.4f}, J_max={jacobian_max:.6f}, "
                  f"step_norm={step_norm_before:.6f}, alpha={alpha:.2f}, "
                  f"actual_change={actual_change:.6f}")

        # Compute final delta from baseline
        delta_x = x_current - x_baseline

        # Get final prediction with optimized controls
        with torch.no_grad():
            window_modified = current_window.clone()
            window_modified[-1, self.control_indices, :] = x_current

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

        # Compute diagnostics
        final_error = (y_predicted.squeeze(-1) - y_desired).norm().item()
        improvement_percent = (
            (initial_error - final_error) / initial_error * 100
            if initial_error > 1e-8
            else 0.0
        )

        # Warning if improvement is poor
        if improvement_percent < 10 and initial_error > self.config.convergence_tol:
            print(
                f"[Warning] Jacobian controller achieved only {improvement_percent:.1f}% "
                f"improvement (initial_error={initial_error:.4f}, final_error={final_error:.4f})"
            )

        # Build jacobian_info dictionary
        jacobian_info = {
            "jacobian": final_jacobian,
            "iterations": iteration + 1,
            "initial_error": initial_error,
            "final_error": final_error,
            "improvement_percent": improvement_percent,
            "residual_history": residual_history,
            "converged": residual_history[-1] < self.config.convergence_tol,
        }

        return x_current, delta_x, y_predicted, jacobian_info

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

    def _line_search(
        self,
        current_window: torch.Tensor,
        x_current: torch.Tensor,
        delta_x: torch.Tensor,
        y_desired: torch.Tensor,
        verbose: bool = False,
    ) -> float:
        """Find best step size using backtracking line search.

        Args:
            current_window: Input window of shape (lag, nodes, features).
            x_current: Current control values (num_control, features).
            delta_x: Proposed step direction (num_control, features).
            y_desired: Desired target values (num_target,).
            verbose: Whether to print debug info.

        Returns:
            Best step size alpha.
        """
        best_alpha = 0.0
        best_loss = float("inf")
        losses = []

        for alpha in self.config.line_search_alphas:
            x_trial = x_current + alpha * delta_x
            x_trial = self._project_constraints(x_trial)

            window_trial = current_window.clone()
            window_trial[-1, self.control_indices, :] = x_trial

            with torch.no_grad():
                forecaster_input = prepare_forecaster_input(window_trial.unsqueeze(0))
                y_pred_full = forward_pass(
                    self.forecaster,
                    forecaster_input,
                    self.model_type,
                    lag_weights=self.lag_weights,
                    adjacency=self.adjacency,
                    neighbor_only_inputs=self.neighbor_only_inputs,
                )
                y_pred_target = y_pred_full[0, self.target_indices, 0]  # (num_target,)
                loss = ((y_pred_target - y_desired) ** 2).mean().item()

            losses.append((alpha, loss))

            if loss < best_loss:
                best_loss = loss
                best_alpha = alpha

        if verbose:
            print(f"    Line search: {[(a, f'{l:.6f}') for a, l in losses]}, best_alpha={best_alpha}")

        return best_alpha

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
