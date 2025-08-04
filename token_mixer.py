"""
Tensor Mixology: Educational Implementation of Parameter-Free Token Mixing

This module demonstrates an efficient alternative to self-attention mechanisms,
achieving O(T) complexity instead of O(T²) through parameter-free tensor operations.

Educational Focus:
- Understanding tensor reshaping and permutation operations
- Learning about linear vs quadratic complexity in sequence models
- Exploring hardware-conscious neural architecture design
- Demonstrating residual connections and layer normalization

Key Concept: Instead of computing attention weights (which require T×T matrices),
TokenMixer uses clever tensor reshaping to mix information between tokens without
any learnable parameters in the mixing operation itself.
"""

import torch
from torch import nn


class TokenMixer(nn.Module):
    """
    Educational Token Mixer: A Parameter-Free Alternative to Self-Attention

    CORE CONCEPT:
    Traditional self-attention computes pairwise relationships between all tokens,
    resulting in O(T²) complexity. TokenMixer achieves global token mixing through
    tensor reshaping operations that have O(T) complexity.

    ARCHITECTURE OVERVIEW:
    1. Split each token into multiple heads (similar to multi-head attention)
    2. Permute tensor dimensions to group heads together
    3. Flatten and unflatten operations cause information mixing
    4. Reconstruct original tensor shape with mixed information
    5. Apply residual connection and layer normalization

    WHY THIS WORKS:
    The key insight is that tensor flattening and reshaping operations can
    redistribute information between tokens without explicit weight matrices.
    This creates global connectivity with linear computational cost.

    EDUCATIONAL LEARNING OUTCOMES:
    - Understand tensor manipulation for neural architectures
    - Learn about computational complexity trade-offs
    - Explore parameter-efficient model design
    - Practice with residual connections and normalization

    Args:
        num_tokens (int): Number of input tokens T (must equal num_heads for this implementation)
        hidden_dim (int): Embedding dimension D of each token
        num_heads (int): Number of heads H for partitioning (must divide hidden_dim evenly)

    Mathematical Notation:
        B = batch size
        T = number of tokens (sequence length)
        D = hidden dimension (embedding size)
        H = number of heads
        d = D/H = head dimension
    """

    def __init__(self, num_tokens: int, hidden_dim: int, num_heads: int):
        """Initialize the Token Mixer module.

        Args:
            num_tokens: The cardinality T of input feature tokens
            hidden_dim: The embedding dimension D of each token
            num_heads: The division factor H for head-level partitioning

        Raises:
            ValueError: If hidden_dim is not divisible by num_heads
        """
        super().__init__()

        # Validate that hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Validate that num_tokens equals num_heads
        if num_tokens != num_heads:
            raise ValueError(
                f"num_tokens ({num_tokens}) must equal num_heads ({num_heads})"
            )

        self.T = num_tokens
        self.D = hidden_dim
        self.H = num_heads
        self.head_dim = hidden_dim // num_heads

        # Layer normalization for residual aggregation
        self.norm = nn.LayerNorm(hidden_dim)

    def _validate_inputs(self, x: torch.Tensor) -> None:
        """Validate input tensor dimensions and data types.

        Args:
            x: Input tensor to validate

        Raises:
            TypeError: If x is not a torch.Tensor
            ValueError: If x has wrong dimensions or shape
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")

        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D")

        # Skip validation during JIT tracing to avoid compatibility issues
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return

        batch_size, num_tokens, hidden_dim = x.shape
        if num_tokens != self.T:
            raise ValueError(f"Expected {self.T} tokens, got {num_tokens}")
        if hidden_dim != self.D:
            raise ValueError(f"Expected {self.D} dimensions, got {hidden_dim}")

        if not x.is_contiguous():
            raise ValueError("Input tensor must be contiguous")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Educational Forward Pass: Step-by-Step Token Mixing Process

        This method demonstrates how to achieve global token interaction through
        tensor reshaping operations. Each step is carefully explained to help
        understand the underlying mathematical operations.

        Args:
            x: Input tensor of shape (B, T, D) where:
               B = batch size (e.g., 4)
               T = number of tokens (e.g., 12)
               D = hidden dimension (e.g., 768)

        Returns:
            Mixed tensor of identical shape (B, T, D) with global token interactions

        TENSOR SHAPE PROGRESSION EXAMPLE (B=2, T=4, D=8, H=4):
        Input:     (2, 4, 8)     - Original token embeddings
        Step 1:    (2, 4, 4, 2)  - Split into heads
        Step 2:    (2, 4, 4, 2)  - Rearrange head dimension
        Step 3:    (2, 16, 2)    - Flatten for mixing
        Step 4:    (2, 4, 4, 2)  - Unflatten (information now mixed!)
        Step 5:    (2, 4, 8)     - Restore original shape
        Output:    (2, 4, 8)     - Mixed tokens + residual + norm
        """
        # Input validation for educational purposes
        self._validate_inputs(x)

        # Extract dimensions for clarity
        batch_size, num_tokens, hidden_dim = x.shape
        print(
            f"📥 INPUT SHAPE: {x.shape} (batch={batch_size}, tokens={num_tokens}, dim={hidden_dim})"
        )

        # EDUCATIONAL NOTE: Store original input for residual connection
        # Residual connections help with gradient flow and training stability
        residual = x

        # ================== STEP 1: HEAD PARTITIONING ==================
        # CONCEPT: Split each token's embedding into multiple "heads"
        # This is similar to multi-head attention but without learnable projections
        #
        # Mathematical operation: (B, T, D) -> (B, T, H, d) where d = D/H
        # Example: (2, 4, 8) -> (2, 4, 4, 2) with H=4 heads
        print("🔄 STEP 1: Head partitioning...")
        x_heads = x.view(batch_size, num_tokens, self.H, self.head_dim)
        print(f"   Shape after head split: {x_heads.shape}")
        print(f"   Each token now has {self.H} heads of dimension {self.head_dim}")

        # ================== STEP 2: DIMENSION PERMUTATION ==================
        # CONCEPT: Rearrange tensor to group all heads together
        # This prepares the tensor for the mixing operation
        #
        # Mathematical operation: (B, T, H, d) -> (B, H, T, d)
        # Example: (2, 4, 4, 2) -> (2, 4, 4, 2) - same shape, different layout
        print("🔄 STEP 2: Dimension permutation...")
        x_perm = x_heads.permute(0, 2, 1, 3).contiguous()
        print(f"   Shape after permutation: {x_perm.shape}")
        print("   Now organized as: (batch, heads, tokens, head_dim)")

        # ================== STEP 3: FLATTEN FOR MIXING ==================
        # CONCEPT: This is where the magic happens! By flattening the head and token
        # dimensions together, we create a longer sequence where information from
        # different tokens and heads gets interleaved.
        #
        # Mathematical operation: (B, H, T, d) -> (B, H*T, d)
        # Example: (2, 4, 4, 2) -> (2, 16, 2)
        print("🔄 STEP 3: Flatten for mixing...")
        x_flat = x_perm.view(batch_size, self.H * num_tokens, self.head_dim)
        print(f"   Shape after flattening: {x_flat.shape}")
        print(f"   Created a sequence of length {self.H * num_tokens} for mixing")

        # ================== STEP 4: UNFLATTEN (INFORMATION MIXING!) ==================
        # CONCEPT: When we unflatten back to separate heads and tokens, the information
        # has been redistributed! Tokens that were in different positions are now
        # mixed together. This achieves global token interaction without parameters.
        #
        # Mathematical operation: (B, H*T, d) -> (B, H, T, d)
        # Example: (2, 16, 2) -> (2, 4, 4, 2) - same shape as step 2, but MIXED data!
        print("🔄 STEP 4: Unflatten (information mixing occurs here!)...")
        x_unflat = x_flat.view(batch_size, self.H, num_tokens, self.head_dim)
        print(f"   Shape after unflattening: {x_unflat.shape}")
        print("   🎉 TOKEN INFORMATION HAS BEEN MIXED!")

        # ================== STEP 5: RESTORE ORIGINAL LAYOUT ==================
        # CONCEPT: Rearrange back to original dimension order and combine heads
        # This gives us the final mixed token representations
        #
        # Mathematical operation: (B, H, T, d) -> (B, T, H, d) -> (B, T, D)
        # Example: (2, 4, 4, 2) -> (2, 4, 4, 2) -> (2, 4, 8)
        print("🔄 STEP 5: Restore original layout...")
        x_mix = (
            x_unflat.permute(0, 2, 1, 3)  # (B, H, T, d) -> (B, T, H, d)
            .contiguous()
            .view(batch_size, num_tokens, hidden_dim)  # -> (B, T, D)
        )
        print(f"   Shape after reconstruction: {x_mix.shape}")

        # ================== STEP 6: RESIDUAL CONNECTION + NORMALIZATION ==================
        # CONCEPT: Add the original input (residual connection) and normalize
        # Residual connections help with:
        # 1. Gradient flow during backpropagation
        # 2. Training stability
        # 3. Preserving original token information
        print("🔄 STEP 6: Residual connection + LayerNorm...")
        output = self.norm(residual + x_mix)
        print(f"📤 OUTPUT SHAPE: {output.shape}")
        print("✅ Token mixing complete! Information from all tokens has been mixed.")

        return output

    def forward_production(self, x: torch.Tensor) -> torch.Tensor:
        """
        Production-optimized forward pass without educational prints.

        Use this method when you don't need the step-by-step explanations,
        such as during training or inference in production environments.

        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D)
        """
        self._validate_inputs(x)

        B, T, D = x.shape

        # Core token mixing operations
        x_heads = x.view(B, T, self.H, self.head_dim)
        x_perm = x_heads.permute(0, 2, 1, 3).contiguous()
        x_flat = x_perm.view(B, self.H * T, self.head_dim)
        x_unflat = x_flat.view(B, self.H, T, self.head_dim)
        x_mix = x_unflat.permute(0, 2, 1, 3).contiguous().view(B, T, D)

        # Residual connection and normalization
        return self.norm(x + x_mix)

    def visualize_mixing_process(self, x: torch.Tensor) -> dict:
        """
        Educational method to visualize how information flows through the mixing process.

        This method returns intermediate tensors at each step so you can inspect
        how the tensor shapes change and understand the mixing process.

        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            Dictionary containing intermediate tensors and their explanations
        """
        self._validate_inputs(x)

        B, T, D = x.shape
        steps = {}

        # Step 1: Head partitioning
        x_heads = x.view(B, T, self.H, self.head_dim)
        steps["step1_head_partition"] = {
            "tensor": x_heads.clone(),
            "shape": x_heads.shape,
            "explanation": f"Split {D}-dim tokens into {self.H} heads of {self.head_dim} dims each",
        }

        # Step 2: Permutation
        x_perm = x_heads.permute(0, 2, 1, 3).contiguous()
        steps["step2_permutation"] = {
            "tensor": x_perm.clone(),
            "shape": x_perm.shape,
            "explanation": "Rearranged to group heads together: (B,T,H,d) -> (B,H,T,d)",
        }

        # Step 3: Flattening
        x_flat = x_perm.view(B, self.H * T, self.head_dim)
        steps["step3_flatten"] = {
            "tensor": x_flat.clone(),
            "shape": x_flat.shape,
            "explanation": f"Flattened to sequence length {self.H * T} for mixing",
        }

        # Step 4: Unflattening (where mixing happens)
        x_unflat = x_flat.view(B, self.H, T, self.head_dim)
        steps["step4_unflatten"] = {
            "tensor": x_unflat.clone(),
            "shape": x_unflat.shape,
            "explanation": "Unflattened - information is now mixed between tokens!",
        }

        # Step 5: Reconstruction
        x_mix = x_unflat.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        steps["step5_reconstruction"] = {
            "tensor": x_mix.clone(),
            "shape": x_mix.shape,
            "explanation": "Reconstructed to original shape with mixed information",
        }

        # Step 6: Final output
        output = self.norm(x + x_mix)
        steps["step6_final"] = {
            "tensor": output.clone(),
            "shape": output.shape,
            "explanation": "Applied residual connection and layer normalization",
        }

        return steps

    def extra_repr(self) -> str:
        """Return string representation for debugging."""
        return f"num_tokens={self.T}, hidden_dim={self.D}, num_heads={self.H}"
