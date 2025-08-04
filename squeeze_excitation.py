"""
Squeeze and Excitation (SE) Module for Neural Networks

This module implements the Squeeze and Excitation mechanism introduced in:
"Squeeze-and-Excitation Networks" by Hu et al. (2018)

The SE module adaptively recalibrates channel-wise feature responses by explicitly
modeling interdependencies between channels. It consists of two operations:

1. Squeeze: Global information embedding via global average pooling
2. Excitation: Adaptive recalibration via a gating mechanism

Educational Implementation with detailed explanations and visualization helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """
    Squeeze and Excitation Module

    This module performs channel-wise attention by:
    1. Squeezing spatial dimensions to capture global context
    2. Learning channel interdependencies through FC layers
    3. Exciting (scaling) channels based on learned importance

    Args:
        in_channels (int): Number of input channels
        reduction_ratio (int): Reduction ratio for the bottleneck layer (default: 16)
        activation (str): Activation function for excitation ('relu', 'gelu', 'swish')
        gate_activation (str): Final gating activation ('sigmoid', 'hardsigmoid')

    Shape:
        - Input: (B, C, H, W) for 2D or (B, C, T) for 1D or (B, T, C) for sequence data
        - Output: Same shape as input, with channel-wise scaling applied
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        activation: str = "relu",
        gate_activation: str = "sigmoid",
    ):
        super().__init__()

        # Validate inputs
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if reduction_ratio <= 0:
            raise ValueError(f"reduction_ratio must be positive, got {reduction_ratio}")

        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        # Calculate bottleneck dimension (minimum of 1)
        self.bottleneck_channels = max(1, in_channels // reduction_ratio)

        # Squeeze operation: Global Average Pooling (handled in forward)
        # No parameters needed - just spatial dimension reduction

        # Excitation operation: Two-layer MLP
        self.fc1 = nn.Linear(in_channels, self.bottleneck_channels, bias=False)
        self.fc2 = nn.Linear(self.bottleneck_channels, in_channels, bias=False)

        # Activation functions
        self.activation = self._get_activation(activation)
        self.gate_activation = self._get_activation(gate_activation)

        # Initialize weights for better convergence
        self._initialize_weights()

    def _get_activation(self, activation: str) -> nn.Module:
        """Helper to get activation function by name"""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),  # SiLU is Swish
            "sigmoid": nn.Sigmoid(),
            "hardsigmoid": nn.Hardsigmoid(),
        }

        if activation not in activations:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                f"Choose from {list(activations.keys())}"
            )

        return activations[activation]

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in [self.fc1, self.fc2]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Squeeze and Excitation module

        Args:
            x: Input tensor of shape (B, C, H, W), (B, C, T), or (B, T, C)

        Returns:
            Output tensor of same shape as input, with SE scaling applied
        """
        # Store original shape and determine tensor format
        original_shape = x.shape
        batch_size = x.size(0)

        # Handle different input formats and convert to (B, C, ...)
        if len(x.shape) == 3:
            # Could be (B, C, T) or (B, T, C) - check which makes sense
            if x.size(-1) == self.in_channels:
                # (B, T, C) format - transpose to (B, C, T)
                x = x.transpose(1, 2)
                sequence_format = True
            else:
                # (B, C, T) format - already correct
                sequence_format = False
        elif len(x.shape) == 4:
            # (B, C, H, W) format - already correct
            sequence_format = False
        else:
            raise ValueError(
                f"Unsupported input shape: {original_shape}. "
                f"Expected 3D (B,C,T) or (B,T,C) or 4D (B,C,H,W)"
            )

        # === SQUEEZE OPERATION ===
        # Global Average Pooling to squeeze spatial/temporal dimensions
        # This creates a channel descriptor that captures global information

        if len(x.shape) == 4:  # 2D case: (B, C, H, W) -> (B, C)
            squeezed = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        else:  # 1D case: (B, C, T) -> (B, C)
            squeezed = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        # === EXCITATION OPERATION ===
        # Two-layer MLP to learn channel interdependencies

        # First FC layer: dimensionality reduction
        # This creates a bottleneck that forces the network to learn
        # compact representations of channel relationships
        excited = self.fc1(squeezed)  # (B, C) -> (B, C/r)
        excited = self.activation(excited)  # Non-linearity

        # Second FC layer: dimensionality restoration
        # This maps back to original channel count and learns
        # the final channel importance weights
        excited = self.fc2(excited)  # (B, C/r) -> (B, C)
        channel_weights = self.gate_activation(excited)  # Final gating

        # === SCALE OPERATION ===
        # Apply learned channel weights to original features

        # Reshape channel weights for broadcasting
        if len(x.shape) == 4:  # 2D case
            channel_weights = channel_weights.view(batch_size, self.in_channels, 1, 1)
        else:  # 1D case
            channel_weights = channel_weights.view(batch_size, self.in_channels, 1)

        # Element-wise multiplication (channel-wise scaling)
        scaled_output = x * channel_weights

        # Restore original format if needed
        if sequence_format:
            # Convert back from (B, C, T) to (B, T, C)
            scaled_output = scaled_output.transpose(1, 2)

        return scaled_output

    def get_channel_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract channel importance weights without applying them

        Useful for visualization and analysis of which channels
        the network considers most important for a given input.

        Args:
            x: Input tensor

        Returns:
            Channel importance weights of shape (B, C)
        """
        # Handle input format conversion
        if len(x.shape) == 3 and x.size(-1) == self.in_channels:
            x = x.transpose(1, 2)

        # Squeeze operation
        if len(x.shape) == 4:
            squeezed = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        else:
            squeezed = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        # Excitation operation
        excited = self.fc1(squeezed)
        excited = self.activation(excited)
        excited = self.fc2(excited)
        channel_weights = self.gate_activation(excited)

        return channel_weights

    def extra_repr(self) -> str:
        """String representation for debugging"""
        return (
            f"in_channels={self.in_channels}, "
            f"reduction_ratio={self.reduction_ratio}, "
            f"bottleneck_channels={self.bottleneck_channels}"
        )


class SEBlock(nn.Module):
    """
    Complete SE Block with optional residual connection

    This is a convenient wrapper that combines any base layer
    (e.g., Conv2d, Linear) with Squeeze and Excitation.

    Args:
        base_layer: The main computation layer (Conv2d, Linear, etc.)
        se_channels: Number of channels for SE module
        reduction_ratio: SE reduction ratio
        use_residual: Whether to add residual connection
    """

    def __init__(
        self,
        base_layer: nn.Module,
        se_channels: int,
        reduction_ratio: int = 16,
        use_residual: bool = True,
    ):
        super().__init__()

        self.base_layer = base_layer
        self.se = SqueezeExcitation(se_channels, reduction_ratio)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection"""
        # Apply base layer
        out = self.base_layer(x)

        # Apply SE attention
        out = self.se(out)

        # Add residual connection if enabled and shapes match
        if self.use_residual and x.shape == out.shape:
            out = out + x

        return out


def create_se_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    reduction_ratio: int = 16,
) -> SEBlock:
    """
    Factory function to create SE-enhanced convolution block

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        reduction_ratio: SE reduction ratio

    Returns:
        SEBlock with Conv2d + BatchNorm2d + ReLU + SE
    """
    conv_block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

    return SEBlock(
        base_layer=conv_block,
        se_channels=out_channels,
        reduction_ratio=reduction_ratio,
        use_residual=(in_channels == out_channels and stride == 1),
    )


# Educational helper functions for understanding SE mechanics


def visualize_se_effect(
    se_module: SqueezeExcitation,
    input_tensor: torch.Tensor,
    channel_names: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Visualize the effect of SE module on input channels

    Args:
        se_module: Trained SE module
        input_tensor: Input to analyze
        channel_names: Optional names for channels

    Returns:
        Tuple of (output_tensor, channel_weights, analysis_dict)
    """
    se_module.eval()

    with torch.no_grad():
        # Get channel importance
        weights = se_module.get_channel_importance(input_tensor)

        # Apply SE module
        output = se_module(input_tensor)

        # Compute statistics
        input_channel_means = input_tensor.mean(
            dim=[0, 2, 3] if len(input_tensor.shape) == 4 else [0, 2]
        )
        output_channel_means = output.mean(
            dim=[0, 2, 3] if len(output.shape) == 4 else [0, 2]
        )

        analysis = {
            "channel_weights": weights.cpu(),
            "weight_statistics": {
                "mean": weights.mean().item(),
                "std": weights.std().item(),
                "min": weights.min().item(),
                "max": weights.max().item(),
            },
            "most_important_channels": weights.mean(0).argsort(descending=True)[:5],
            "least_important_channels": weights.mean(0).argsort(descending=False)[:5],
            "input_channel_means": input_channel_means,
            "output_channel_means": output_channel_means,
            "scaling_effect": (
                output_channel_means / (input_channel_means + 1e-8)
            ).cpu(),
        }

        if channel_names:
            analysis["channel_names"] = channel_names

    return output, weights, analysis


if __name__ == "__main__":
    # Educational demonstration
    print("=== Squeeze and Excitation Module Demo ===\n")

    # Create SE module
    se = SqueezeExcitation(in_channels=64, reduction_ratio=16)
    print(f"SE Module: {se}")
    print(f"Parameters: {sum(p.numel() for p in se.parameters())}")

    # Test with different input formats
    print("\n=== Testing Different Input Formats ===")

    # 2D CNN format: (B, C, H, W)
    x_2d = torch.randn(2, 64, 32, 32)
    out_2d = se(x_2d)
    print(f"2D Input: {x_2d.shape} -> Output: {out_2d.shape}")

    # 1D CNN format: (B, C, T)
    x_1d = torch.randn(2, 64, 128)
    out_1d = se(x_1d)
    print(f"1D Input: {x_1d.shape} -> Output: {out_1d.shape}")

    # Sequence format: (B, T, C)
    x_seq = torch.randn(2, 128, 64)
    out_seq = se(x_seq)
    print(f"Sequence Input: {x_seq.shape} -> Output: {out_seq.shape}")

    # Analyze channel importance
    weights = se.get_channel_importance(x_2d)
    print(f"\nChannel importance weights shape: {weights.shape}")
    print(f"Weight statistics - Mean: {weights.mean():.3f}, Std: {weights.std():.3f}")
    print(f"Most important channels: {weights.mean(0).argsort(descending=True)[:5]}")
