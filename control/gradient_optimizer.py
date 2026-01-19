"""Gradient-based controller for counterfactual generation.

Finds optimal control inputs X* by optimizing through the differentiable
forward model using gradient descent.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from train import forward_pass


def prepare_forecaster_input(x: torch.Tensor) -> torch.Tensor:
    """Reorder samples into the (B, C, N, T) layout expected by ST-GNNs."""
    return x.permute(0, 3, 2, 1).contiguous()


@dataclass
class GradientOptimizerConfig:
    """Configuration for gradient-based optimization."""

    n_steps: int = 100
    lr: float = 0.01
    lambda_reg: float = 0.1
    optimizer_type: str = "adam"  # "adam" or "lbfgs"
    convergence_tol: float = 1e-6
    grad_clip: float = 1.0
    target_weight: float = 1.0  # Weight for target matching loss


class GradientBasedController:
    """Gradient-based controller that optimizes control node values.

    Optimization Problem:
        X* = argmin_X ||f(X) - Y*||² + λ||X - X_baseline||²
        subject to: X_min ≤ X ≤ X_max

    Where:
        X = control node values at last timestep
        f(X) = forecaster output for target nodes
        Y* = desired target values
        X_baseline = original control values (for minimal intervention)
    """

    def __init__(
        self,
        forecaster: nn.Module,
        control_indices: torch.Tensor,
        target_indices: torch.Tensor,
        adjacency: torch.Tensor,
        config: GradientOptimizerConfig,
        x_bounds: Optional[Tuple[float, float]] = None,
        model_type: str = "stgcn",
        lag_weights: Optional[torch.Tensor] = None,
        neighbor_only_inputs: bool = False,
    ):
        """Initialize the gradient-based controller.

        Args:
            forecaster: Trained forecaster model (horizon=1).
            control_indices: Indices of control nodes in the graph.
            target_indices: Indices of target nodes to match.
            adjacency: Graph adjacency matrix.
            config: Optimization configuration.
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
        # backward passes require training mode, so we'll handle this in find_intervention.

    def find_intervention(
        self,
        current_window: torch.Tensor,
        y_desired: torch.Tensor,
        target_channel: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """Find optimal control intervention.

        Args:
            current_window: Input window of shape (lag, nodes, features).
            y_desired: Desired target values of shape (num_target, 1) or (num_target,).
            target_channel: Which feature channel to optimize for.

        Returns:
            x_optimal: Optimized control values (num_control, features).
            y_predicted: Predicted output with optimal controls (num_target, 1).
            loss_history: List of loss values per iteration.
        """
        current_window = current_window.to(self.device)
        y_desired = y_desired.to(self.device)

        if y_desired.dim() == 1:
            y_desired = y_desired.unsqueeze(-1)

        # Save original training mode and set to train for backward pass (required for cuDNN RNNs)
        was_training = self.forecaster.training
        self.forecaster.train()

        # Freeze model parameters - we only want to optimize x_opt
        original_requires_grad = {}
        for name, param in self.forecaster.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False

        try:
            # Extract baseline control values (last timestep)
            x_baseline = current_window[-1, self.control_indices, :].clone()

            # Initialize learnable control values
            x_opt = x_baseline.clone().requires_grad_(True)

            # Setup optimizer
            if self.config.optimizer_type == "adam":
                optimizer = torch.optim.Adam([x_opt], lr=self.config.lr)
            elif self.config.optimizer_type == "lbfgs":
                optimizer = torch.optim.LBFGS(
                    [x_opt], lr=self.config.lr, max_iter=20, line_search_fn="strong_wolfe"
                )
            else:
                raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

            loss_history = []
            prev_loss = float("inf")

            # Debug: print initial state
            with torch.no_grad():
                init_loss, init_pred = self._compute_loss(
                    current_window, x_opt, y_desired, x_baseline, target_channel
                )
                print(f"[GradientController] Initial - target: {y_desired.flatten()[:3].tolist()}, "
                      f"pred: {init_pred.flatten()[:3].tolist()}, loss: {init_loss.item():.6f}")
                print(f"[GradientController] Control baseline (first 3): {x_baseline[:3, 0].tolist()}")

            for step in range(self.config.n_steps):
                if self.config.optimizer_type == "lbfgs":

                    def closure():
                        optimizer.zero_grad()
                        loss, _ = self._compute_loss(
                            current_window, x_opt, y_desired, x_baseline, target_channel
                        )
                        loss.backward()
                        if self.config.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_([x_opt], self.config.grad_clip)
                        return loss

                    loss = optimizer.step(closure)
                    loss_val = loss.item()
                else:
                    optimizer.zero_grad()
                    loss, _ = self._compute_loss(
                        current_window, x_opt, y_desired, x_baseline, target_channel
                    )
                    loss.backward()

                    # Debug: print gradient info on first step
                    if step == 0:
                        if x_opt.grad is not None:
                            grad_mag = x_opt.grad.abs().mean().item()
                            grad_max = x_opt.grad.abs().max().item()
                            print(f"[GradientController] Step 0 gradient - mean: {grad_mag:.10f}, max: {grad_max:.10f}")
                        else:
                            print("[GradientController] WARNING: x_opt.grad is None!")

                    if self.config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_([x_opt], self.config.grad_clip)

                    optimizer.step()
                    loss_val = loss.item()

                # Project onto constraints
                with torch.no_grad():
                    x_opt.data = self._project_constraints(x_opt.data)

                loss_history.append(loss_val)

                # Check convergence
                if abs(prev_loss - loss_val) < self.config.convergence_tol:
                    break
                prev_loss = loss_val

            # Get final prediction
            with torch.no_grad():
                final_loss, y_pred = self._compute_loss(
                    current_window, x_opt, y_desired, x_baseline, target_channel
                )
                control_change = (x_opt - x_baseline).abs().mean().item()
                print(f"[GradientController] Final - pred: {y_pred.flatten()[:3].tolist()}, "
                      f"loss: {final_loss.item():.6f}, mean |delta_control|: {control_change:.6f}")
                print(f"[GradientController] Converged in {len(loss_history)} steps")

            return x_opt.detach(), y_pred.detach(), loss_history

        finally:
            # Restore original model state
            for name, param in self.forecaster.named_parameters():
                param.requires_grad = original_requires_grad[name]
            if not was_training:
                self.forecaster.eval()

    def _compute_loss(
        self,
        current_window: torch.Tensor,
        x_control: torch.Tensor,
        y_desired: torch.Tensor,
        x_baseline: torch.Tensor,
        target_channel: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the optimization loss.

        Returns:
            loss: Scalar loss value.
            y_pred: Predicted target values.
        """
        # Create modified window with optimized control values
        # Use differentiable operations to maintain gradient flow
        lag_len, num_nodes, num_features = current_window.shape

        # Keep all timesteps except last unchanged
        window_except_last = current_window[:-1]  # (lag-1, nodes, features)

        # For the last timestep, we need to replace control node values with x_control
        # using differentiable operations (no in-place indexing)
        last_original = current_window[-1]  # (num_nodes, features)

        # Create a float mask: 1.0 for control nodes, 0.0 for others
        mask = torch.zeros(num_nodes, 1, device=current_window.device, dtype=current_window.dtype)
        mask[self.control_indices] = 1.0

        # Expand x_control to full node size using scatter (differentiable)
        indices = self.control_indices.unsqueeze(-1).expand(-1, num_features)  # (num_control, features)
        x_control_full = torch.zeros(num_nodes, num_features, device=current_window.device, dtype=current_window.dtype)
        x_control_full = x_control_full.scatter(0, indices, x_control)

        # Combine: control nodes get x_control, others get original values
        # last_new = mask * x_control_full + (1 - mask) * last_original
        last_timestep_new = mask * x_control_full + (1.0 - mask) * last_original

        # Concatenate to form full window
        window_modified = torch.cat([window_except_last, last_timestep_new.unsqueeze(0)], dim=0)

        # Forward pass through forecaster
        # Input shape: (1, lag, nodes, features) -> prepare -> (1, features, nodes, lag)
        forecaster_input = prepare_forecaster_input(window_modified.unsqueeze(0))

        y_pred_full = forward_pass(
            self.forecaster,
            forecaster_input,
            self.model_type,
            lag_weights=self.lag_weights,
            adjacency=self.adjacency,
            neighbor_only_inputs=self.neighbor_only_inputs,
        )

        # Extract target node predictions
        # y_pred_full shape: (1, nodes, horizon) where horizon=1
        y_pred = y_pred_full[0, self.target_indices, :]  # (num_target, 1)

        # Target matching loss
        target_loss = self.config.target_weight * torch.mean((y_pred - y_desired) ** 2)

        # Regularization loss (minimal intervention)
        reg_loss = self.config.lambda_reg * torch.mean((x_control - x_baseline) ** 2)

        total_loss = target_loss + reg_loss

        return total_loss, y_pred

    def _project_constraints(self, x: torch.Tensor) -> torch.Tensor:
        """Project control values onto feasible bounds."""
        if self.x_bounds is not None:
            x_min, x_max = self.x_bounds
            x = torch.clamp(x, min=x_min, max=x_max)
        return x

    def apply_control(
        self,
        current_window: torch.Tensor,
        x_optimal: torch.Tensor,
    ) -> torch.Tensor:
        """Apply optimized control values to the window.

        Args:
            current_window: Original window (lag, nodes, features).
            x_optimal: Optimized control values (num_control, features).

        Returns:
            Modified window with control values applied at last timestep.
        """
        modified_window = current_window.clone()
        modified_window[-1, self.control_indices, :] = x_optimal
        return modified_window
