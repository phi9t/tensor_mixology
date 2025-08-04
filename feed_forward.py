"""
Tensor Mixology: Educational Feed Forward Networks

This module demonstrates different feed forward network architectures commonly
used in transformer models. Each variant is educational and shows different
design patterns and trade-offs.

Educational Focus:
- Understanding the role of FFNs in transformer architectures
- Learning about different activation functions and their properties
- Exploring gating mechanisms (GLU, SwiGLU)
- Understanding residual connections and normalization placement
- Comparing computational costs of different FFN variants

Key Concepts:
1. Feed forward networks process each token independently
2. They typically expand to a larger intermediate dimension
3. Different activation functions have different properties
4. Gating mechanisms can improve model capacity
"""

import torch
import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    """
    Educational Standard Feed Forward Network

    ARCHITECTURE OVERVIEW:
    This is the classic FFN used in transformer models like BERT and GPT.
    It follows the pattern: LayerNorm -> Linear -> Activation -> Dropout -> Linear -> Dropout -> Residual

    WHY THIS DESIGN:
    1. LayerNorm stabilizes training and normalizes input distributions
    2. First Linear layer expands to intermediate dimension (usually 4x hidden_dim)
    3. Activation function introduces non-linearity (GELU is common in modern models)
    4. Dropout prevents overfitting during training
    5. Second Linear layer projects back to original dimension
    6. Residual connection helps with gradient flow and training stability

    COMPUTATIONAL COST:
    - Forward pass FLOPs: 2 × hidden_dim × ff_dim (two matrix multiplications)
    - Parameters: 2 × hidden_dim × ff_dim + biases
    - Memory: O(batch_size × seq_len × ff_dim) for intermediate activations

    ACTIVATION FUNCTION CHOICES:
    - GELU: Smooth, differentiable everywhere, used in BERT/GPT
    - ReLU: Simple, fast, but can cause dead neurons
    - SiLU/Swish: Smooth like GELU, used in modern models

    LEARNING OUTCOMES:
    - Understand why FFNs use expansion and contraction
    - Learn about activation function choices and their trade-offs
    - Practice with residual connections and normalization
    - Compare parameter counts vs performance trade-offs

    Args:
        hidden_dim (int): The input/output dimension (e.g., 768 for BERT-base)
        ff_dim (int): The intermediate dimension (typically 4x hidden_dim = 3072)
        dropout_rate (float): Dropout probability (default: 0.1)
        activation (str): Activation function ('gelu', 'relu', 'swish') (default: 'gelu')
        bias (bool): Whether to use bias in linear layers (default: True)
    """

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Feed forward layers
        self.linear1 = nn.Linear(hidden_dim, ff_dim, bias=bias)
        self.linear2 = nn.Linear(ff_dim, hidden_dim, bias=bias)

        # Activation function
        self.activation = self._get_activation(activation)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "swish": nn.SiLU(),
            "silu": nn.SiLU(),
        }

        if activation.lower() not in activation_map:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                f"Supported: {list(activation_map.keys())}"
            )

        return activation_map[activation.lower()]

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Educational Forward Pass: Understanding FFN Operations Step-by-Step

        This method processes each token independently through the feed forward network.
        The key insight is that FFNs operate on the feature dimension, not the sequence dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D) where:
                             B = batch size, T = sequence length, D = hidden dimension
            verbose (bool): Whether to print educational information about each step

        Returns:
            torch.Tensor: Output tensor of same shape (B, T, D) with processed features

        EDUCATIONAL WALKTHROUGH:
        1. Store residual for skip connection (helps gradient flow)
        2. Normalize input (stabilizes training)
        3. Expand to larger dimension (creates capacity for complex transformations)
        4. Apply non-linear activation (enables complex feature learning)
        5. Apply dropout (prevents overfitting during training)
        6. Contract back to original dimension (output projection)
        7. Apply dropout again (additional regularization)
        8. Add residual connection (preserves information and helps training)
        """
        if verbose:
            B, T, D = x.shape
            print(f"🍽️  FFN Forward Pass: Processing {T} tokens with {D} features each")
            print(f"📊 Input shape: {x.shape}")
            print(
                f"🔧 FFN Config: {D} -> {self.ff_dim} -> {D} (expansion ratio: {self.ff_dim/D:.1f}x)"
            )

        # =================== STEP 1: RESIDUAL STORAGE ===================
        # CONCEPT: Store original input for residual connection
        # This helps with gradient flow and preserves input information
        residual = x
        if verbose:
            print("💾 Step 1: Stored residual connection")

        # =================== STEP 2: LAYER NORMALIZATION ===================
        # CONCEPT: Normalize features to have mean=0, std=1
        # This stabilizes training and helps with gradient flow
        # LayerNorm normalizes across the feature dimension (last dim)
        x = self.norm(x)
        if verbose:
            print("🔄 Step 2: Applied LayerNorm - normalized features")
            print(f"   Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")

        # =================== STEP 3: EXPANSION TRANSFORMATION ===================
        # CONCEPT: Linear transformation to larger dimension
        # This creates more capacity for the network to learn complex patterns
        # Typical expansion ratio is 4x (e.g., 768 -> 3072)
        x = self.linear1(x)
        if verbose:
            print(f"📈 Step 3: Expanded to intermediate dimension {x.shape}")
            print(f"   Parameters used: {self.hidden_dim * self.ff_dim:,}")

        # =================== STEP 4: NON-LINEAR ACTIVATION ===================
        # CONCEPT: Apply activation function to introduce non-linearity
        # Without activation, the FFN would just be a linear transformation
        # Different activations have different properties:
        # - GELU: smooth, used in BERT/GPT
        # - ReLU: simple but can cause dead neurons
        # - SiLU: smooth, used in modern models
        x = self.activation(x)
        if verbose:
            activation_name = self.activation.__class__.__name__
            print(f"⚡ Step 4: Applied {activation_name} activation")
            print(f"   Non-zero activations: {(x > 0).float().mean().item():.2%}")

        # =================== STEP 5: DROPOUT (TRAINING ONLY) ===================
        # CONCEPT: Randomly set some activations to zero during training
        # This prevents overfitting and improves generalization
        x = self.dropout1(x)
        if verbose and self.training:
            print(f"🎲 Step 5: Applied dropout (rate: {self.dropout_rate})")

        # =================== STEP 6: CONTRACTION TRANSFORMATION ===================
        # CONCEPT: Linear transformation back to original dimension
        # This is the output projection that combines all the intermediate features
        x = self.linear2(x)
        if verbose:
            print(f"📉 Step 6: Contracted back to original dimension {x.shape}")
            print(f"   Parameters used: {self.ff_dim * self.hidden_dim:,}")

        # =================== STEP 7: FINAL DROPOUT ===================
        # CONCEPT: Additional regularization on the output
        x = self.dropout2(x)
        if verbose and self.training:
            print("🎲 Step 7: Applied final dropout")

        # =================== STEP 8: RESIDUAL CONNECTION ===================
        # CONCEPT: Add the original input to the transformed output
        # This helps with:
        # 1. Gradient flow (gradients can flow directly through the residual)
        # 2. Training stability (prevents vanishing gradients)
        # 3. Information preservation (original features are preserved)
        output = residual + x
        if verbose:
            print("🔗 Step 8: Added residual connection")
            print(f"📤 Final output shape: {output.shape}")
            print("✅ FFN processing complete!")

        return output

    def get_parameter_count(self) -> dict:
        """
        Educational method to understand parameter usage in FFN.

        Returns:
            dict: Breakdown of parameters by component
        """
        params = {}

        # Linear layer parameters
        linear1_params = self.hidden_dim * self.ff_dim
        linear2_params = self.ff_dim * self.hidden_dim
        if hasattr(self.linear1, "bias") and self.linear1.bias is not None:
            linear1_params += self.ff_dim
            linear2_params += self.hidden_dim

        params["linear1_weights"] = self.hidden_dim * self.ff_dim
        params["linear1_bias"] = self.ff_dim if self.linear1.bias is not None else 0
        params["linear2_weights"] = self.ff_dim * self.hidden_dim
        params["linear2_bias"] = self.hidden_dim if self.linear2.bias is not None else 0
        params["layernorm"] = 2 * self.hidden_dim  # gamma and beta
        params["total"] = sum(params.values())

        return params


class GLUFeedForward(nn.Module):
    """
    Gated Linear Unit (GLU) variant of feed forward network.

    Uses GLU activation which splits the intermediate dimension in half,
    applying sigmoid gating to one half and multiplying with the other half.

    Args:
        hidden_dim (int): The input/output dimension
        ff_dim (int): The intermediate feed-forward dimension (should be even)
        dropout_rate (float): Dropout probability (default: 0.1)
        bias (bool): Whether to use bias in linear layers (default: True)
    """

    def __init__(
        self, hidden_dim: int, ff_dim: int, dropout_rate: float = 0.1, bias: bool = True
    ):
        super().__init__()

        if ff_dim % 2 != 0:
            raise ValueError("ff_dim must be even for GLU variant")

        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Feed forward layers
        self.linear1 = nn.Linear(hidden_dim, ff_dim, bias=bias)
        self.linear2 = nn.Linear(ff_dim // 2, hidden_dim, bias=bias)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with GLU activation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Store residual
        residual = x

        # Layer normalization
        x = self.norm(x)

        # First linear transformation
        x = self.linear1(x)

        # Split for GLU
        gate, value = x.chunk(2, dim=-1)

        # Apply GLU: sigmoid(gate) * value
        x = torch.sigmoid(gate) * value
        x = self.dropout1(x)

        # Second linear transformation
        x = self.linear2(x)
        x = self.dropout2(x)

        # Residual connection
        return residual + x


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU variant of feed forward network.

    Uses SwiGLU activation (Swish/SiLU with GLU) which has shown improved
    performance in large language models.

    Args:
        hidden_dim (int): The input/output dimension
        ff_dim (int): The intermediate feed-forward dimension
        dropout_rate (float): Dropout probability (default: 0.1)
        bias (bool): Whether to use bias in linear layers (default: False)
    """

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # SwiGLU requires three linear layers
        self.w1 = nn.Linear(hidden_dim, ff_dim, bias=bias)  # Gate
        self.w2 = nn.Linear(ff_dim, hidden_dim, bias=bias)  # Down projection
        self.w3 = nn.Linear(hidden_dim, ff_dim, bias=bias)  # Up projection

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Store residual
        residual = x

        # Layer normalization
        x = self.norm(x)

        # SwiGLU: SiLU(W1(x)) * W3(x)
        gate = F.silu(self.w1(x))
        value = self.w3(x)
        x = gate * value

        # Down projection with dropout
        x = self.w2(x)
        x = self.dropout(x)

        # Residual connection
        return residual + x
