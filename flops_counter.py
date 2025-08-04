"""FLOPs counter for PyTorch modules.

This module provides comprehensive FLOPs (Floating Point Operations) counting for neural network modules.
It uses a hybrid approach combining runtime measurement via PyTorch hooks with theoretical analysis.

Key Features:
- Runtime measurement using forward hooks
- Theoretical analysis with mathematical formulas
- Module-specific counting for different layer types
- Detailed breakdown of FLOPs by component
- Complexity analysis (O(T) vs O(T²))
- Parameter counting and efficiency metrics

Mathematical Background:
- FLOPs = Floating Point Operations per forward pass
- Linear layers: B × D_in × D_out (matrix multiplication)
- Layer Norm: 4 × B × T × D (mean, variance, normalization, scale/shift)
- Attention: B × T × D × (3D + 2H×T×d + H×T) (quadratic complexity)
- TokenMixer: B × T × D × (1 + 2H/d + 4) (linear complexity)
"""

from typing import Any

import torch
import torch.nn as nn


class FLOPsCounter:
    """Count floating point operations for PyTorch modules.

    This class uses PyTorch's forward hooks to intercept module forward passes
    and count FLOPs in real-time. It combines runtime measurement with theoretical
    analysis to provide accurate FLOPs estimates.

    The counting process:
    1. Register forward hooks for all submodules
    2. Run forward pass with dummy input
    3. Intercept each module's computation via hooks
    4. Apply module-specific counting formulas
    5. Aggregate results and provide detailed breakdown
    """

    def __init__(self):
        """Initialize the FLOPs counter.

        Attributes:
            flops (int): Total FLOPs counted
            params (int): Total parameters counted
            module_flops (dict): FLOPs per module name
            module_params (dict): Parameters per module name
        """
        self.flops = 0
        self.params = 0
        self.module_flops = {}
        self.module_params = {}

    def reset(self):
        """Reset the FLOPs counter to initial state."""
        self.flops = 0
        self.params = 0
        self.module_flops = {}
        self.module_params = {}

    def count_flops(
        self,
        module: nn.Module,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        """Count FLOPs for a given module using forward hooks.

        This method performs the main FLOPs counting process:
        1. Registers forward hooks for all submodules
        2. Runs a forward pass with dummy input
        3. Intercepts computations via hooks
        4. Applies module-specific counting formulas
        5. Returns comprehensive results

        Args:
            module: PyTorch module to analyze
            input_shape: Shape of input tensor (batch_size, seq_len, hidden_dim)
            output_shape: Shape of output tensor (optional, auto-detected if None)

        Returns:
            Dictionary containing:
            - flops: Total FLOPs counted
            - params: Total parameters
            - module_flops: FLOPs per module name
            - module_params: Parameters per module name
            - input_shape: Input tensor shape
            - output_shape: Output tensor shape
            - flops_per_param: Efficiency metric
        """
        self.reset()

        # Register hooks for all modules in the network
        # Each hook will intercept the forward pass and count FLOPs
        hooks = []
        for name, submodule in module.named_modules():
            hook = submodule.register_forward_hook(
                lambda m, i, o, name=name: self._hook_fn(m, i, o, name)
            )
            hooks.append(hook)

        # Create dummy input tensor for forward pass
        # This allows us to trace through the network without affecting real data
        dummy_input = torch.randn(input_shape)

        # Run forward pass with gradient computation disabled
        # This is more efficient and we only need the computation trace
        with torch.no_grad():
            output = module(dummy_input)

        # Clean up hooks to prevent memory leaks
        for hook in hooks:
            hook.remove()

        # Auto-detect output shape if not provided
        if output_shape is None:
            if isinstance(output, tuple):
                output_shape = output[0].shape
            else:
                output_shape = output.shape

        # Count total parameters in the module
        self.params = sum(p.numel() for p in module.parameters())

        return {
            "flops": self.flops,
            "params": self.params,
            "module_flops": self.module_flops.copy(),
            "module_params": self.module_params.copy(),
            "input_shape": input_shape,
            "output_shape": output_shape,
            "flops_per_param": self.flops / max(self.params, 1),
        }

    def _hook_fn(
        self,
        module: nn.Module,
        input_tensors: tuple[torch.Tensor, ...],
        output_tensors: tuple[torch.Tensor, ...],
        name: str,
    ):
        """Hook function to count FLOPs for each module during forward pass.

        This function is called by PyTorch's forward hook system for each module.
        It analyzes the input/output tensors and applies the appropriate FLOPs
        counting formula based on the module type.

        Args:
            module: The module being processed
            input_tensors: Tuple of input tensors to the module
            output_tensors: Tuple of output tensors from the module
            name: Name of the module in the network
        """
        # Handle tuple outputs (common in attention modules)
        if isinstance(output_tensors, tuple):
            actual_outputs = output_tensors
        else:
            actual_outputs = (output_tensors,)

        # Apply module-specific FLOPs counting
        module_flops = self._count_module_flops(module, input_tensors, actual_outputs)
        self.flops += module_flops
        self.module_flops[name] = module_flops

        # Count parameters for this specific module
        module_params = sum(p.numel() for p in module.parameters())
        self.module_params[name] = module_params

    def _count_module_flops(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
    ) -> int:
        """Count FLOPs for a specific module type using appropriate formulas.

        This method dispatches to the correct counting function based on module type.
        Each module type has its own mathematical formula for FLOPs computation.

        Args:
            module: The module to analyze
            inputs: Input tensors to the module
            outputs: Output tensors from the module

        Returns:
            Number of FLOPs for this module
        """
        module_type = type(module).__name__

        if module_type == "Linear":
            return self._count_linear_flops(module, inputs[0], outputs[0])
        elif module_type == "LayerNorm":
            return self._count_layernorm_flops(module, inputs[0], outputs[0])
        elif module_type == "Dropout":
            return 0  # Dropout doesn't add FLOPs, just masks
        elif module_type in ["TokenMixer", "SelfAttention", "MultiHeadAttention"]:
            return self._count_attention_flops(module, inputs[0], outputs[0])
        else:
            # For unknown modules, estimate based on input/output shapes
            return self._estimate_flops(inputs[0], outputs[0])

    def _count_linear_flops(
        self, module: nn.Linear, input_tensor: torch.Tensor, output_tensor: torch.Tensor
    ) -> int:
        """Count FLOPs for linear (fully connected) layer.

        Mathematical Formula:
        - Matrix multiplication: B × D_in × D_out FLOPs
        - Bias addition: B × D_out FLOPs (if bias exists)
        - Total: B × D_in × D_out + B × D_out (if bias)

        Where:
        - B = batch size
        - D_in = input features
        - D_out = output features

        Args:
            module: Linear layer module
            input_tensor: Input tensor of shape (B, ..., D_in)
            output_tensor: Output tensor of shape (B, ..., D_out)

        Returns:
            Number of FLOPs for this linear layer
        """
        batch_size = input_tensor.shape[0]
        input_features = module.in_features
        output_features = module.out_features

        # Matrix multiplication: B × D_in × D_out
        # Each output element requires D_in multiplications and (D_in-1) additions
        flops = batch_size * input_features * output_features

        # Add bias if present: B × D_out additions
        if module.bias is not None:
            flops += batch_size * output_features

        return flops

    def _count_layernorm_flops(
        self,
        module: nn.LayerNorm,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
    ) -> int:
        """Count FLOPs for layer normalization.

        Mathematical Formula:
        - Mean computation: B × T × D FLOPs
        - Variance computation: B × T × D FLOPs
        - Normalization: B × T × D FLOPs
        - Scale and shift: 2 × B × T × D FLOPs
        - Total: 4 × B × T × D FLOPs

        Where:
        - B = batch size
        - T = sequence length (or 1 for non-sequential data)
        - D = hidden dimension

        LayerNorm computes: y = γ * (x - μ) / √(σ² + ε) + β
        Where μ is mean, σ² is variance, γ and β are learnable parameters.

        Args:
            module: LayerNorm module
            input_tensor: Input tensor
            output_tensor: Output tensor

        Returns:
            Number of FLOPs for this layer normalization
        """
        batch_size = input_tensor.shape[0]
        seq_len = input_tensor.shape[1] if len(input_tensor.shape) > 2 else 1
        hidden_dim = input_tensor.shape[-1]

        # Mean computation: B × T × D operations
        # Each element contributes to the mean calculation
        flops = batch_size * seq_len * hidden_dim

        # Variance computation: B × T × D operations
        # Each element contributes to the variance calculation
        flops += batch_size * seq_len * hidden_dim

        # Normalization: B × T × D operations
        # Each element is normalized using the computed mean and variance
        flops += batch_size * seq_len * hidden_dim

        # Scale and shift: 2 × B × T × D operations
        # γ (scale) and β (shift) are applied to each element
        flops += batch_size * seq_len * hidden_dim * 2

        return flops

    def _count_attention_flops(
        self, module: nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor
    ) -> int:
        """Count FLOPs for attention mechanisms (Self-Attention, Multi-Head Attention).

        Mathematical Formula (Quadratic Complexity O(T²)):
        - Linear projections (Q,K,V): 3 × B × T × D × D FLOPs
        - Attention scores: B × H × T × T × d FLOPs
        - Softmax: B × H × T × T FLOPs
        - Attention output: B × H × T × T × d FLOPs
        - Output projection: B × T × D × D FLOPs
        - Total: B × T × D × (3D + 2H×T×d + H×T)

        Where:
        - B = batch size
        - T = sequence length
        - D = hidden dimension
        - H = number of heads
        - d = D/H = head dimension

        The quadratic complexity comes from the T×T attention matrix.

        Args:
            module: Attention module
            input_tensor: Input tensor of shape (B, T, D)
            output_tensor: Output tensor of shape (B, T, D)

        Returns:
            Number of FLOPs for this attention mechanism
        """
        batch_size, seq_len, hidden_dim = input_tensor.shape

        # Get number of heads and head dimension
        if hasattr(module, "num_heads"):
            num_heads = module.num_heads
            head_dim = hidden_dim // num_heads
        else:
            # Default values for unknown attention modules
            num_heads = 8
            head_dim = hidden_dim // num_heads

        # Linear projections for Q, K, V: 3 × B × T × D × D
        # Each projection transforms the input to query, key, value
        flops = 3 * batch_size * seq_len * hidden_dim * hidden_dim

        # Attention scores: B × H × T × T × d
        # Computing attention weights between all token pairs
        flops += batch_size * num_heads * seq_len * seq_len * head_dim

        # Softmax: B × H × T × T
        # Normalizing attention weights across sequence dimension
        flops += batch_size * num_heads * seq_len * seq_len

        # Attention output: B × H × T × T × d
        # Applying attention weights to values
        flops += batch_size * num_heads * seq_len * seq_len * head_dim

        # Output projection: B × T × D × D
        # Final linear transformation to combine heads
        flops += batch_size * seq_len * hidden_dim * hidden_dim

        return flops

    def _estimate_flops(
        self, input_tensor: torch.Tensor, output_tensor: torch.Tensor
    ) -> int:
        """Estimate FLOPs for unknown modules based on input/output shapes.

        This is a fallback method for modules without specific counting formulas.
        It assumes each output element requires some computation.

        Args:
            input_tensor: Input tensor
            output_tensor: Output tensor

        Returns:
            Estimated number of FLOPs
        """
        output_elements = output_tensor.numel()

        # Rough estimate: assume each output element requires computation
        return output_elements


def count_token_mixer_flops(
    num_tokens: int, hidden_dim: int, num_heads: int, batch_size: int = 1
) -> dict[str, Any]:
    """Count FLOPs for TokenMixer module with detailed analysis.

    TokenMixer achieves linear complexity O(T) through tensor reshaping operations
    instead of the quadratic attention matrix used in self-attention.

    Mathematical Formula (Linear Complexity O(T)):
    - Head partitioning: B × T × D FLOPs
    - Permutation operations: B × T × H × d FLOPs
    - Flattening and reshaping: B × T × H × d FLOPs
    - Layer normalization: 4 × B × T × D FLOPs
    - Total: B × T × D × (1 + 2H/d + 4)

    Where:
    - B = batch size
    - T = number of tokens
    - D = hidden dimension
    - H = number of heads
    - d = D/H = head dimension

    Args:
        num_tokens: Number of tokens (sequence length)
        hidden_dim: Hidden dimension
        num_heads: Number of heads
        batch_size: Batch size

    Returns:
        Dictionary with FLOPs breakdown including theoretical analysis
    """
    from token_mixer import TokenMixer

    counter = FLOPsCounter()
    module = TokenMixer(num_tokens, hidden_dim, num_heads)
    input_shape = (batch_size, num_tokens, hidden_dim)

    result = counter.count_flops(module, input_shape)

    # Add theoretical analysis for comparison
    theoretical_flops = _analyze_token_mixer_flops(
        num_tokens, hidden_dim, num_heads, batch_size
    )
    result["theoretical_flops"] = theoretical_flops
    result["theoretical_breakdown"] = _get_token_mixer_breakdown(
        num_tokens, hidden_dim, num_heads, batch_size
    )

    return result


def count_attention_flops(
    embed_dim: int, num_heads: int, seq_len: int, batch_size: int = 1
) -> dict[str, Any]:
    """Count FLOPs for attention module with detailed analysis.

    Self-attention has quadratic complexity O(T²) due to the attention matrix
    that computes relationships between all token pairs.

    Mathematical Formula (Quadratic Complexity O(T²)):
    - Linear projections (Q,K,V): 3 × B × T × D × D FLOPs
    - Attention scores: B × H × T × T × d FLOPs
    - Softmax: B × H × T × T FLOPs
    - Attention output: B × H × T × T × d FLOPs
    - Output projection: B × T × D × D FLOPs
    - Total: B × T × D × (3D + 2H×T×d + H×T)

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        seq_len: Sequence length
        batch_size: Batch size

    Returns:
        Dictionary with FLOPs breakdown including theoretical analysis
    """
    from attention import SelfAttention

    counter = FLOPsCounter()
    module = SelfAttention(embed_dim, num_heads)
    input_shape = (batch_size, seq_len, embed_dim)

    result = counter.count_flops(module, input_shape)

    # Add theoretical analysis for comparison
    theoretical_flops = _analyze_attention_flops(
        embed_dim, num_heads, seq_len, batch_size
    )
    result["theoretical_flops"] = theoretical_flops
    result["theoretical_breakdown"] = _get_attention_breakdown(
        embed_dim, num_heads, seq_len, batch_size
    )

    return result


def count_rank_mixer_flops(
    num_tokens: int,
    hidden_dim: int,
    num_heads: int,
    ff_dim: int,
    num_layers: int = 1,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Count FLOPs for RankMixer module (TokenMixer + Feed Forward).

    RankMixer combines TokenMixer (linear complexity) with feed-forward networks.
    The total complexity remains linear O(T) for the TokenMixer component.

    Mathematical Formula:
    - TokenMixer: B × T × D × (1 + 2H/d + 4) FLOPs
    - Feed Forward: B × T × D × ff_dim + B × T × ff_dim + B × T × ff_dim × D FLOPs
    - Total: TokenMixer + Feed Forward

    Args:
        num_tokens: Number of tokens
        hidden_dim: Hidden dimension
        num_heads: Number of heads
        ff_dim: Feed forward dimension
        num_layers: Number of layers
        batch_size: Batch size

    Returns:
        Dictionary with FLOPs breakdown including theoretical analysis
    """
    try:
        from rank_mixer_architecture import RankMixerBlock

        counter = FLOPsCounter()
        module = RankMixerBlock(num_tokens, hidden_dim, num_heads, ff_dim)
        input_shape = (batch_size, num_tokens, hidden_dim)

        result = counter.count_flops(module, input_shape)

        # Add theoretical analysis
        theoretical_flops = _analyze_rank_mixer_flops(
            num_tokens, hidden_dim, num_heads, ff_dim, batch_size
        )
        result["theoretical_flops"] = theoretical_flops
        result["theoretical_breakdown"] = _get_rank_mixer_breakdown(
            num_tokens, hidden_dim, num_heads, ff_dim, batch_size
        )

        # Multiply by number of layers for full model
        result["total_flops"] = result["flops"] * num_layers
        result["total_theoretical_flops"] = theoretical_flops * num_layers
        result["total_params"] = result["params"] * num_layers

        return result

    except ImportError:
        # Fallback to estimation if RankMixer not available
        token_mixer_result = count_token_mixer_flops(
            num_tokens, hidden_dim, num_heads, batch_size
        )

        # Estimate feed forward FLOPs
        ff_flops = _analyze_feed_forward_flops(
            num_tokens, hidden_dim, ff_dim, batch_size
        )

        total_flops = token_mixer_result["flops"] + ff_flops
        total_params = token_mixer_result["params"] + _estimate_feed_forward_params(
            hidden_dim, ff_dim
        )

        return {
            "flops": total_flops,
            "params": total_params,
            "flops_per_param": total_flops / max(total_params, 1),
            "theoretical_flops": total_flops,
            "total_flops": total_flops * num_layers,
            "total_params": total_params * num_layers,
            "estimated": True,
        }


def _analyze_token_mixer_flops(
    num_tokens: int, hidden_dim: int, num_heads: int, batch_size: int
) -> int:
    """Theoretical FLOPs analysis for TokenMixer (Linear Complexity O(T)).

    TokenMixer achieves linear complexity through tensor reshaping operations:
    1. Head partitioning: Split tokens into multiple heads
    2. Permutation: Rearrange tensor dimensions for mixing
    3. Reshape: Flatten and unflatten for information redistribution
    4. Layer normalization: Standard normalization

    Mathematical Formula:
    - Head partitioning: B × T × D FLOPs
    - Permutation operations: B × T × H × d FLOPs
    - Flattening and reshaping: B × T × H × d FLOPs
    - Layer normalization: 4 × B × T × D FLOPs
    - Total: B × T × D × (1 + 2H/d + 4)

    Args:
        num_tokens: Number of tokens
        hidden_dim: Hidden dimension
        num_heads: Number of heads
        batch_size: Batch size

    Returns:
        Theoretical FLOPs count
    """
    head_dim = hidden_dim // num_heads

    # Head partitioning: B × T × D
    # Split hidden dimension into multiple heads
    flops = batch_size * num_tokens * hidden_dim

    # Permutation operations: B × T × H × d
    # Rearrange tensor dimensions for mixing
    flops += batch_size * num_tokens * num_heads * head_dim

    # Flattening and reshaping: B × T × H × d
    # Reshape operations for information redistribution
    flops += batch_size * num_tokens * num_heads * head_dim

    # Layer normalization: 4 × B × T × D
    # Standard layer normalization (mean, variance, normalize, scale/shift)
    flops += batch_size * num_tokens * hidden_dim * 4

    return flops


def _analyze_attention_flops(
    embed_dim: int, num_heads: int, seq_len: int, batch_size: int
) -> int:
    """Theoretical FLOPs analysis for attention (Quadratic Complexity O(T²)).

    Self-attention computes relationships between all token pairs, leading to
    quadratic complexity in sequence length.

    Mathematical Formula:
    - Linear projections (Q,K,V): 3 × B × T × D × D FLOPs
    - Attention scores: B × H × T × T × d FLOPs
    - Softmax: B × H × T × T FLOPs
    - Attention output: B × H × T × T × d FLOPs
    - Output projection: B × T × D × D FLOPs
    - Total: B × T × D × (3D + 2H×T×d + H×T)

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        seq_len: Sequence length
        batch_size: Batch size

    Returns:
        Theoretical FLOPs count
    """
    head_dim = embed_dim // num_heads

    # Linear projections for Q, K, V: 3 × B × T × D × D
    # Transform input to query, key, and value representations
    flops = 3 * batch_size * seq_len * embed_dim * embed_dim

    # Attention scores: B × H × T × T × d
    # Compute attention weights between all token pairs
    flops += batch_size * num_heads * seq_len * seq_len * head_dim

    # Softmax: B × H × T × T
    # Normalize attention weights across sequence dimension
    flops += batch_size * num_heads * seq_len * seq_len

    # Attention output: B × H × T × T × d
    # Apply attention weights to values
    flops += batch_size * num_heads * seq_len * seq_len * head_dim

    # Output projection: B × T × D × D
    # Final linear transformation to combine heads
    flops += batch_size * seq_len * embed_dim * embed_dim

    return flops


def _analyze_rank_mixer_flops(
    num_tokens: int, hidden_dim: int, num_heads: int, ff_dim: int, batch_size: int
) -> int:
    """Theoretical FLOPs analysis for RankMixer (TokenMixer + Feed Forward).

    RankMixer combines the linear complexity TokenMixer with feed-forward networks.
    The overall complexity is dominated by the linear TokenMixer component.

    Mathematical Formula:
    - TokenMixer: B × T × D × (1 + 2H/d + 4) FLOPs
    - Feed Forward: B × T × D × ff_dim + B × T × ff_dim + B × T × ff_dim × D FLOPs
    - Total: TokenMixer + Feed Forward

    Args:
        num_tokens: Number of tokens
        hidden_dim: Hidden dimension
        num_heads: Number of heads
        ff_dim: Feed forward dimension
        batch_size: Batch size

    Returns:
        Theoretical FLOPs count
    """
    # TokenMixer FLOPs (linear complexity)
    token_mixer_flops = _analyze_token_mixer_flops(
        num_tokens, hidden_dim, num_heads, batch_size
    )

    # Feed Forward FLOPs (linear complexity)
    ff_flops = _analyze_feed_forward_flops(num_tokens, hidden_dim, ff_dim, batch_size)

    return token_mixer_flops + ff_flops


def _analyze_feed_forward_flops(
    num_tokens: int, hidden_dim: int, ff_dim: int, batch_size: int
) -> int:
    """Theoretical FLOPs analysis for feed forward network (Linear Complexity O(T)).

    Feed-forward networks consist of two linear layers with an activation function.
    They have linear complexity in sequence length.

    Mathematical Formula:
    - First linear layer: B × T × D × ff_dim FLOPs
    - Activation function: B × T × ff_dim FLOPs
    - Second linear layer: B × T × ff_dim × D FLOPs
    - Total: B × T × D × ff_dim + B × T × ff_dim + B × T × ff_dim × D

    Args:
        num_tokens: Number of tokens
        hidden_dim: Hidden dimension
        ff_dim: Feed forward dimension
        batch_size: Batch size

    Returns:
        Theoretical FLOPs count
    """
    # First linear layer: B × T × D × ff_dim
    # Expand hidden dimension to feed-forward dimension
    flops = batch_size * num_tokens * hidden_dim * ff_dim

    # Activation function: B × T × ff_dim
    # Apply activation (GELU, ReLU, etc.) to each element
    flops += batch_size * num_tokens * ff_dim

    # Second linear layer: B × T × ff_dim × D
    # Contract feed-forward dimension back to hidden dimension
    flops += batch_size * num_tokens * ff_dim * hidden_dim

    return flops


def _estimate_feed_forward_params(hidden_dim: int, ff_dim: int) -> int:
    """Estimate parameters for feed forward network.

    Mathematical Formula:
    - First linear layer: D × ff_dim + ff_dim (weights + bias)
    - Second linear layer: ff_dim × D + D (weights + bias)
    - Total: D × ff_dim + ff_dim + ff_dim × D + D

    Args:
        hidden_dim: Hidden dimension
        ff_dim: Feed forward dimension

    Returns:
        Number of parameters
    """
    # First linear layer: weights + bias
    params = hidden_dim * ff_dim + ff_dim

    # Second linear layer: weights + bias
    params += ff_dim * hidden_dim + hidden_dim

    return params


def _get_token_mixer_breakdown(
    num_tokens: int, hidden_dim: int, num_heads: int, batch_size: int
) -> dict[str, int]:
    """Get detailed FLOPs breakdown for TokenMixer.

    Provides component-wise FLOPs analysis for TokenMixer operations.

    Args:
        num_tokens: Number of tokens
        hidden_dim: Hidden dimension
        num_heads: Number of heads
        batch_size: Batch size

    Returns:
        Dictionary with FLOPs breakdown by component
    """
    head_dim = hidden_dim // num_heads

    return {
        "head_partitioning": batch_size * num_tokens * hidden_dim,
        "permutation_ops": batch_size * num_tokens * num_heads * head_dim,
        "reshape_ops": batch_size * num_tokens * num_heads * head_dim,
        "layer_norm": batch_size * num_tokens * hidden_dim * 4,
        "total": _analyze_token_mixer_flops(
            num_tokens, hidden_dim, num_heads, batch_size
        ),
    }


def _get_attention_breakdown(
    embed_dim: int, num_heads: int, seq_len: int, batch_size: int
) -> dict[str, int]:
    """Get detailed FLOPs breakdown for attention.

    Provides component-wise FLOPs analysis for self-attention operations.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        seq_len: Sequence length
        batch_size: Batch size

    Returns:
        Dictionary with FLOPs breakdown by component
    """
    head_dim = embed_dim // num_heads

    return {
        "linear_projections": 3 * batch_size * seq_len * embed_dim * embed_dim,
        "attention_scores": batch_size * num_heads * seq_len * seq_len * head_dim,
        "softmax": batch_size * num_heads * seq_len * seq_len,
        "attention_output": batch_size * num_heads * seq_len * seq_len * head_dim,
        "output_projection": batch_size * seq_len * embed_dim * embed_dim,
        "total": _analyze_attention_flops(embed_dim, num_heads, seq_len, batch_size),
    }


def _get_rank_mixer_breakdown(
    num_tokens: int, hidden_dim: int, num_heads: int, ff_dim: int, batch_size: int
) -> dict[str, int]:
    """Get detailed FLOPs breakdown for RankMixer.

    Provides component-wise FLOPs analysis for RankMixer (TokenMixer + Feed Forward).

    Args:
        num_tokens: Number of tokens
        hidden_dim: Hidden dimension
        num_heads: Number of heads
        ff_dim: Feed forward dimension
        batch_size: Batch size

    Returns:
        Dictionary with FLOPs breakdown by component
    """
    token_mixer_breakdown = _get_token_mixer_breakdown(
        num_tokens, hidden_dim, num_heads, batch_size
    )
    ff_breakdown = _get_feed_forward_breakdown(
        num_tokens, hidden_dim, ff_dim, batch_size
    )

    return {
        "token_mixer": token_mixer_breakdown["total"],
        "feed_forward": ff_breakdown["total"],
        "total": token_mixer_breakdown["total"] + ff_breakdown["total"],
        "token_mixer_breakdown": token_mixer_breakdown,
        "feed_forward_breakdown": ff_breakdown,
    }


def _get_feed_forward_breakdown(
    num_tokens: int, hidden_dim: int, ff_dim: int, batch_size: int
) -> dict[str, int]:
    """Get detailed FLOPs breakdown for feed forward network.

    Provides component-wise FLOPs analysis for feed-forward operations.

    Args:
        num_tokens: Number of tokens
        hidden_dim: Hidden dimension
        ff_dim: Feed forward dimension
        batch_size: Batch size

    Returns:
        Dictionary with FLOPs breakdown by component
    """
    return {
        "first_linear": batch_size * num_tokens * hidden_dim * ff_dim,
        "activation": batch_size * num_tokens * ff_dim,
        "second_linear": batch_size * num_tokens * ff_dim * hidden_dim,
        "total": _analyze_feed_forward_flops(
            num_tokens, hidden_dim, ff_dim, batch_size
        ),
    }


def format_flops(flops: int) -> str:
    """Format FLOPs count in human-readable format.

    Converts raw FLOPs count to appropriate units (K, M, G, T).

    Args:
        flops: Raw FLOPs count

    Returns:
        Formatted string with appropriate unit
    """
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPS"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPS"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPS"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f} KFLOPS"
    else:
        return f"{flops} FLOPS"


def compare_flops_efficiency(
    module1_flops: int, module1_params: int, module2_flops: int, module2_params: int
) -> dict[str, float]:
    """Compare FLOPs efficiency between two modules.

    Provides efficiency metrics for comparing different architectures:
    - flops_ratio: How many times more FLOPs module1 uses vs module2
    - params_ratio: How many times more parameters module1 uses vs module2
    - flops_per_param_ratio: Efficiency comparison accounting for both FLOPs and params

    Args:
        module1_flops: FLOPs for first module
        module1_params: Parameters for first module
        module2_flops: FLOPs for second module
        module2_params: Parameters for second module

    Returns:
        Dictionary with efficiency comparison metrics
    """
    return {
        "flops_ratio": module1_flops / max(module2_flops, 1),
        "params_ratio": module1_params / max(module2_params, 1),
        "flops_per_param_ratio": (module1_flops / max(module1_params, 1))
        / (module2_flops / max(module2_params, 1)),
    }
