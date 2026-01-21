"""Perturbation-based controller for counterfactual generation.

Uses random search + finite differences gradient descent to find optimal
control node perturbations without requiring backpropagation through the model.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from train import forward_pass


def prepare_forecaster_input(x: torch.Tensor) -> torch.Tensor:
    """Reorder samples into the (B, C, N, T) layout expected by ST-GNNs."""
    return x.permute(0, 3, 2, 1).contiguous()


def _convert_global_to_local_indices(
    global_indices: torch.Tensor,
    manipulated_indices: torch.Tensor,
) -> torch.Tensor:
    """Convert global node indices to local indices within manipulated nodes.

    For causal_forecaster, the model only outputs predictions for manipulated nodes.
    This function finds the position of each global target index within the
    manipulated_indices list.

    Args:
        global_indices: Global node indices to convert.
        manipulated_indices: Global indices of manipulated nodes.

    Returns:
        Local indices corresponding to positions in manipulated_indices.
    """
    manip_list = manipulated_indices.tolist()
    local_indices = []
    for gi in global_indices.tolist():
        if gi in manip_list:
            local_indices.append(manip_list.index(gi))
        else:
            raise ValueError(
                f"Target index {gi} is not in manipulated_indices. "
                "Cannot use non-manipulated nodes as targets for causal_forecaster."
            )
    return torch.tensor(local_indices, device=global_indices.device, dtype=torch.long)


@dataclass
class PerturbationConfig:
    """Configuration for perturbation-based optimization."""

    # Phase 1: Random search
    n_random_samples: int = 200  # Number of random perturbations to try
    perturbation_scale: float = 0.1  # Scale for uniform sampling: U(-scale, scale)

    # Phase 2: Refinement
    n_refine_steps: int = 50  # Gradient descent iterations
    refine_lr: float = 0.01  # Learning rate for refinement
    finite_diff_eps: float = 1e-4  # Epsilon for numerical gradient
    convergence_tol: float = 1e-6  # Early stopping tolerance


class PerturbationController:
    """Perturbation-based controller using random search and finite differences.

    Two-phase Approach:
        Phase 1 (Random Search): Sample random perturbations uniformly within bounds
        for all control nodes, evaluate each through the forecaster, keep the best.

        Phase 2 (Refinement): Starting from the best perturbation, use finite
        differences to approximate gradients and perform gradient descent.
    """

    def __init__(
        self,
        forecaster: nn.Module,
        control_indices: torch.Tensor,
        target_indices: torch.Tensor,
        adjacency: torch.Tensor,
        config: PerturbationConfig,
        x_bounds: Optional[Tuple[float, float]] = None,
        model_type: str = "stgcn",
        lag_weights: Optional[torch.Tensor] = None,
        neighbor_only_inputs: bool = False,
        manipulated_indices: Optional[torch.Tensor] = None,
    ):
        """Initialize the perturbation-based controller.

        Args:
            forecaster: Trained forecaster model (horizon=1).
            control_indices: Indices of control nodes in the graph.
            target_indices: Indices of target nodes to match (global indices).
            adjacency: Graph adjacency matrix.
            config: Optimization configuration.
            x_bounds: (min, max) bounds for control values (optional global bounds).
            model_type: Type of forecaster model.
            lag_weights: Optional weights for lag dimensions.
            neighbor_only_inputs: Whether to use neighbor-only inputs.
            manipulated_indices: Global indices of manipulated nodes. Required
                for causal_forecaster model type to convert between global and
                local indices.
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
        self.manipulated_indices = manipulated_indices

        self.device = next(forecaster.parameters()).device

        # For causal_forecaster, compute local target indices
        self._is_causal_forecaster = model_type == "causal_forecaster"
        if self._is_causal_forecaster:
            if manipulated_indices is None:
                # Try to get from model
                if hasattr(forecaster, "manipulated_indices"):
                    self.manipulated_indices = forecaster.manipulated_indices
                else:
                    raise ValueError(
                        "manipulated_indices must be provided for causal_forecaster model type"
                    )
            self._local_target_indices = _convert_global_to_local_indices(
                target_indices, self.manipulated_indices
            )
        else:
            self._local_target_indices = None

    def find_intervention(
        self,
        current_window: torch.Tensor,
        y_desired: torch.Tensor,
        target_channel: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """Find optimal control intervention using perturbation-based optimization.

        Args:
            current_window: Input window of shape (lag, nodes, features).
            y_desired: Desired target values of shape (num_target, 1) or (num_target,).
            target_channel: Which feature channel to optimize for.

        Returns:
            x_optimal: Optimized control values (num_control, features).
            y_predicted: Predicted output with optimal controls (num_target, 1).
            loss_history: List of loss values during optimization.
        """
        current_window = current_window.to(self.device)
        y_desired = y_desired.to(self.device)

        if y_desired.dim() == 1:
            y_desired = y_desired.unsqueeze(-1)

        # Extract baseline control values (last timestep)
        x_baseline = current_window[-1, self.control_indices, :].clone()
        num_control = len(self.control_indices)
        num_features = current_window.shape[-1]

        # Compute data-driven bounds for each control node
        lower_bounds, upper_bounds = self._compute_data_driven_bounds(
            current_window, target_channel
        )

        # Debug: print initial state
        with torch.no_grad():
            init_loss, init_pred = self._evaluate_perturbation(
                current_window,
                y_desired,
                target_channel,
                torch.zeros_like(x_baseline),
                x_baseline,
                lower_bounds,
                upper_bounds,
            )
            print(
                f"[PerturbationController] Initial - target: {y_desired.flatten()[:3].tolist()}, "
                f"pred: {init_pred.flatten()[:3].tolist()}, loss: {init_loss.item():.6f}"
            )
            print(
                f"[PerturbationController] Control baseline (first 3): {x_baseline[:3, 0].tolist()}"
            )

        # Phase 1: Random search
        best_perturbation, best_loss = self._random_search(
            current_window,
            y_desired,
            target_channel,
            x_baseline,
            lower_bounds,
            upper_bounds,
        )
        print(
            f"[PerturbationController] Phase 1 (random search) - best loss: {best_loss:.6f}"
        )

        # Phase 2: Finite differences refinement
        x_optimal, y_predicted, loss_history = self._refine_with_finite_diff(
            current_window,
            y_desired,
            target_channel,
            best_perturbation,
            x_baseline,
            lower_bounds,
            upper_bounds,
        )

        # Compute final stats
        with torch.no_grad():
            control_change = (x_optimal - x_baseline).abs().mean().item()
            final_loss = loss_history[-1] if loss_history else best_loss
            print(
                f"[PerturbationController] Final - pred: {y_predicted.flatten()[:3].tolist()}, "
                f"loss: {final_loss:.6f}, mean |delta_control|: {control_change:.6f}"
            )
            print(
                f"[PerturbationController] Converged in {len(loss_history)} refinement steps"
            )

        return x_optimal, y_predicted, loss_history

    def _compute_data_driven_bounds(
        self,
        current_window: torch.Tensor,
        target_channel: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute data-driven bounds for each control node.

        For each control node, computes bounds from the input window values,
        expanded by 25% in both directions.

        Args:
            current_window: Input window of shape (lag, nodes, features).
            target_channel: Which feature channel to use for bounds.

        Returns:
            lower_bounds: Lower bound for each control node (num_control,).
            upper_bounds: Upper bound for each control node (num_control,).
        """
        num_control = len(self.control_indices)
        lower_bounds = torch.zeros(num_control, device=self.device)
        upper_bounds = torch.zeros(num_control, device=self.device)

        for i, ctrl_idx in enumerate(self.control_indices):
            node_values = current_window[:, ctrl_idx, target_channel]
            max_val = node_values.max()
            min_val = node_values.min()

            # Expand range by 25% in both directions
            range_expansion = 0.25 * (max_val - min_val).abs()
            if range_expansion < 1e-6:  # Prevent zero range
                range_expansion = 0.25 * max_val.abs() if max_val.abs() > 1e-6 else 0.1

            upper_bounds[i] = max_val + range_expansion
            lower_bounds[i] = min_val - range_expansion

        return lower_bounds, upper_bounds

    def _random_search(
        self,
        current_window: torch.Tensor,
        y_desired: torch.Tensor,
        target_channel: int,
        x_baseline: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Phase 1: Random search over perturbations.

        Args:
            current_window: Input window of shape (lag, nodes, features).
            y_desired: Desired target values.
            target_channel: Which feature channel to optimize for.
            x_baseline: Baseline control values.
            lower_bounds: Lower bounds for control values.
            upper_bounds: Upper bounds for control values.

        Returns:
            best_perturbation: Best perturbation found (num_control, features).
            best_loss: Loss value for the best perturbation.
        """
        num_control = len(self.control_indices)
        num_features = current_window.shape[-1]
        scale = self.config.perturbation_scale

        best_perturbation = torch.zeros_like(x_baseline)
        best_loss = float("inf")

        with torch.no_grad():
            for _ in range(self.config.n_random_samples):
                # Sample perturbation uniformly in [-scale, scale]
                perturbation = (
                    torch.rand(num_control, num_features, device=self.device) * 2 - 1
                ) * scale

                # Evaluate this perturbation
                loss, _ = self._evaluate_perturbation(
                    current_window,
                    y_desired,
                    target_channel,
                    perturbation,
                    x_baseline,
                    lower_bounds,
                    upper_bounds,
                )

                if loss < best_loss:
                    best_loss = loss
                    best_perturbation = perturbation.clone()

        return best_perturbation, best_loss

    def _refine_with_finite_diff(
        self,
        current_window: torch.Tensor,
        y_desired: torch.Tensor,
        target_channel: int,
        initial_perturbation: torch.Tensor,
        x_baseline: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """Phase 2: Refine perturbation using finite differences gradient descent.

        Args:
            current_window: Input window of shape (lag, nodes, features).
            y_desired: Desired target values.
            target_channel: Which feature channel to optimize for.
            initial_perturbation: Starting perturbation from Phase 1.
            x_baseline: Baseline control values.
            lower_bounds: Lower bounds for control values.
            upper_bounds: Upper bounds for control values.

        Returns:
            x_optimal: Optimized control values (num_control, features).
            y_predicted: Predicted output with optimal controls (num_target, 1).
            loss_history: List of loss values during refinement.
        """
        perturbation = initial_perturbation.clone()
        loss_history = []
        prev_loss = float("inf")

        with torch.no_grad():
            for step in range(self.config.n_refine_steps):
                # Compute numerical gradient via finite differences
                grad = self._compute_numerical_gradient(
                    current_window,
                    y_desired,
                    target_channel,
                    perturbation,
                    x_baseline,
                    lower_bounds,
                    upper_bounds,
                )

                # Gradient descent update
                perturbation = perturbation - self.config.refine_lr * grad

                # Apply bounds: compute x_perturbed, clamp, then recompute perturbation
                x_perturbed = x_baseline + perturbation
                x_perturbed = self._apply_bounds(
                    x_perturbed, lower_bounds, upper_bounds, target_channel
                )
                perturbation = x_perturbed - x_baseline

                # Evaluate current loss
                current_loss, y_pred = self._evaluate_perturbation(
                    current_window,
                    y_desired,
                    target_channel,
                    perturbation,
                    x_baseline,
                    lower_bounds,
                    upper_bounds,
                )
                loss_history.append(current_loss.item())

                # Check convergence
                if abs(prev_loss - current_loss.item()) < self.config.convergence_tol:
                    break
                prev_loss = current_loss.item()

            # Get final prediction
            x_optimal = x_baseline + perturbation
            x_optimal = self._apply_bounds(
                x_optimal, lower_bounds, upper_bounds, target_channel
            )
            _, y_predicted = self._evaluate_perturbation(
                current_window,
                y_desired,
                target_channel,
                x_optimal - x_baseline,
                x_baseline,
                lower_bounds,
                upper_bounds,
            )

        return x_optimal, y_predicted, loss_history

    def _compute_numerical_gradient(
        self,
        current_window: torch.Tensor,
        y_desired: torch.Tensor,
        target_channel: int,
        perturbation: torch.Tensor,
        x_baseline: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute numerical gradient using central finite differences.

        For each control dimension i:
            grad[i] = (loss(p + eps*e_i) - loss(p - eps*e_i)) / (2*eps)

        Args:
            current_window: Input window of shape (lag, nodes, features).
            y_desired: Desired target values.
            target_channel: Which feature channel to optimize for.
            perturbation: Current perturbation values.
            x_baseline: Baseline control values.
            lower_bounds: Lower bounds for control values.
            upper_bounds: Upper bounds for control values.

        Returns:
            grad: Numerical gradient with same shape as perturbation.
        """
        eps = self.config.finite_diff_eps
        grad = torch.zeros_like(perturbation)

        num_control, num_features = perturbation.shape

        for i in range(num_control):
            for j in range(num_features):
                # Forward perturbation
                p_plus = perturbation.clone()
                p_plus[i, j] += eps

                # Backward perturbation
                p_minus = perturbation.clone()
                p_minus[i, j] -= eps

                # Evaluate both
                loss_plus, _ = self._evaluate_perturbation(
                    current_window,
                    y_desired,
                    target_channel,
                    p_plus,
                    x_baseline,
                    lower_bounds,
                    upper_bounds,
                )
                loss_minus, _ = self._evaluate_perturbation(
                    current_window,
                    y_desired,
                    target_channel,
                    p_minus,
                    x_baseline,
                    lower_bounds,
                    upper_bounds,
                )

                # Central difference
                grad[i, j] = (loss_plus - loss_minus) / (2 * eps)

        return grad

    def _evaluate_perturbation(
        self,
        current_window: torch.Tensor,
        y_desired: torch.Tensor,
        target_channel: int,
        perturbation: torch.Tensor,
        x_baseline: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a perturbation by computing MSE loss.

        Args:
            current_window: Input window of shape (lag, nodes, features).
            y_desired: Desired target values.
            target_channel: Which feature channel to optimize for.
            perturbation: Perturbation to evaluate.
            x_baseline: Baseline control values.
            lower_bounds: Lower bounds for control values.
            upper_bounds: Upper bounds for control values.

        Returns:
            loss: MSE loss between predicted and desired target values.
            y_pred: Predicted target values (local indices for causal_forecaster).
        """
        # Apply perturbation to control nodes
        x_perturbed = x_baseline + perturbation
        x_perturbed = self._apply_bounds(
            x_perturbed, lower_bounds, upper_bounds, target_channel
        )

        # Create modified window
        window_modified = current_window.clone()
        window_modified[-1, self.control_indices, :] = x_perturbed

        # Forward pass through forecaster
        forecaster_input = prepare_forecaster_input(window_modified.unsqueeze(0))

        if self._is_causal_forecaster:
            # For causal_forecaster, use forward_manipulated_only to get (B, 41, H)
            y_pred_full = self.forecaster.forward_manipulated_only(forecaster_input)
            # Shape: (1, num_manip, 1)
            # Extract target predictions using local indices
            y_pred = y_pred_full[0, self._local_target_indices, :]  # (num_target, 1)
        else:
            y_pred_full = forward_pass(
                self.forecaster,
                forecaster_input,
                self.model_type,
                lag_weights=self.lag_weights,
                adjacency=self.adjacency,
                neighbor_only_inputs=self.neighbor_only_inputs,
            )
            # Shape: (1, num_nodes, horizon) where horizon=1
            # Extract target node predictions using global indices
            y_pred = y_pred_full[0, self.target_indices, :]  # (num_target, 1)

        # Compute MSE loss
        loss = torch.mean((y_pred - y_desired) ** 2)

        return loss, y_pred

    def _apply_bounds(
        self,
        x: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        target_channel: int,
    ) -> torch.Tensor:
        """Apply bounds to control values.

        Args:
            x: Control values (num_control, features).
            lower_bounds: Lower bound per control node (num_control,).
            upper_bounds: Upper bound per control node (num_control,).
            target_channel: Which feature channel the bounds apply to.

        Returns:
            Bounded control values.
        """
        x_bounded = x.clone()

        # Apply per-node bounds for the target channel
        x_bounded[:, target_channel] = torch.clamp(
            x_bounded[:, target_channel],
            min=lower_bounds,
            max=upper_bounds,
        )

        # Also apply global bounds if specified
        if self.x_bounds is not None:
            x_min, x_max = self.x_bounds
            x_bounded = torch.clamp(x_bounded, min=x_min, max=x_max)

        return x_bounded

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

    @property
    def device(self) -> torch.device:
        """Get the device of the controller."""
        return self._device

    @device.setter
    def device(self, value: torch.device) -> None:
        """Set the device of the controller."""
        self._device = value
