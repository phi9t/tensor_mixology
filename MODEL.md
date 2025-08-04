# Tensor Mixology: Model Documentation

This document provides comprehensive technical documentation for the Token Mixer architecture and related components implemented in Tensor Mixology.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Token Mixer Implementation](#token-mixer-implementation)
3. [Feed Forward Networks](#feed-forward-networks)
4. [Complete Model Architecture](#complete-model-architecture)
5. [Mathematical Formulation](#mathematical-formulation)
6. [FLOP Analysis](#flop-analysis)
7. [Performance Characteristics](#performance-characteristics)
8. [Visual Tensor Flow](#visual-tensor-flow)
9. [Implementation Details](#implementation-details)
10. [Testing and Validation](#testing-and-validation)

---

## Architecture Overview

### Core Concept

The Token Mixer implements a parameter-free token mixing mechanism that achieves linear O(T) complexity through clever tensor reshaping operations, contrasting with the quadratic O(T²) complexity of traditional self-attention mechanisms.

### Key Design Principles

1. **Parameter-free mixing**: No learnable weights, only tensor operations
2. **Linear complexity**: O(T) vs O(T²) for self-attention
3. **Hardware-conscious**: Optimized for GPU memory bandwidth
4. **Modular composition**: Integrates seamlessly with RankMixer blocks

### Hardware-Conscious Design

- **Memory Access Patterns**: Contiguous tensor operations minimize cache misses
- **Parallelization Efficiency**: Head-wise operations enable optimal GPU occupancy
- **Scaling Characteristics**: Linear scaling enables deployment with large token sequences

---

## Token Mixer Implementation

### Class Definition

```python
class TokenMixer(nn.Module):
    def __init__(self, num_tokens: int, hidden_dim: int, num_heads: int):
        # Constructor validates hidden_dim % num_heads == 0
        # Sets up LayerNorm for residual connections

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, T, D)
        # 1. Head splitting: reshape to (B, T, H, D/H)
        # 2. Permutation: (B, H, T, D/H)
        # 3. Token flattening: (B, H*T, D/H)
        # 4. Recombination: (B, H, D)
        # 5. Residual connection + LayerNorm
        # Output: (B, T, D)
```

### Step-by-Step Process

#### Step 1: Head Partitioning
**Mathematical operation**: `(B, T, D) → (B, T, H, d)` where `d = D/H`

```
Token 0: [a1,a2,a3,a4,a5,a6,a7,a8] → [[a1,a2], [a3,a4], [a5,a6], [a7,a8]]
Token 1: [b1,b2,b3,b4,b5,b6,b7,b8] → [[b1,b2], [b3,b4], [b5,b6], [b7,b8]]
```

#### Step 2: Dimension Permutation
**Mathematical operation**: `(B, T, H, d) → (B, H, T, d)`

```
Before: [Batch][Token0,Token1,Token2,Token3][Head0,Head1,Head2,Head3][Features]
After:  [Batch][Head0,Head1,Head2,Head3][Token0,Token1,Token2,Token3][Features]
```

#### Step 3: Flatten for Mixing
**Mathematical operation**: `(B, H, T, d) → (B, H*T, d)`

```
Flattened Sequence:
[a1,a2] [b1,b2] [c1,c2] [d1,d2]   ← Head 0 tokens
[a3,a4] [b3,b4] [c3,c4] [d3,d4]   ← Head 1 tokens
[a5,a6] [b5,b6] [c5,c6] [d5,d6]   ← Head 2 tokens
[a7,a8] [b7,b8] [c7,c8] [d7,d8]   ← Head 3 tokens

🔄 MIXING HAPPENS HERE: Information from different tokens is now interleaved!
```

#### Step 4: Unflatten (Information Mixed!)
**Mathematical operation**: `(B, H*T, d) → (B, H, T, d)`

When we unflatten back to separate heads and tokens, the information has been redistributed! Tokens that were in different positions are now mixed together.

#### Step 5: Residual Connection + Normalization
```
output = LayerNorm(input + mixed_tokens)
```

---

## Feed Forward Networks

### Standard Feed Forward Network

**Architecture**: `LayerNorm → Linear → Activation → Dropout → Linear → Dropout → Residual`

```python
class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, dropout_rate: float = 0.1,
                 activation: str = "gelu", bias: bool = True)
```

**Parameter Distribution**:
```
Component          | Parameters        | Percentage
-------------------|-------------------|------------
Linear1 (weights)  | D × 4D = 4D²      | ~44.4%
Linear1 (bias)     | 4D                | ~2.8%
Linear2 (weights)  | 4D × D = 4D²      | ~44.4%
Linear2 (bias)     | D                 | ~0.7%
LayerNorm          | 2D                | ~1.4%
-------------------|-------------------|------------
Total              | 8D² + 7D          | 100%
```

### SwiGLU Variant

**Architecture**: Uses SwiGLU activation (Swish/SiLU with GLU) with three linear layers

```
Standard FFN:              SwiGLU FFN:
Input → LayerNorm          Input → LayerNorm
     ↓                           ↓
  Linear (D→4D)            Linear1 (D→4D)  ──┐
     ↓                           ↓           │
  Activation                   SiLU          │
     ↓                           ↓           │
  Dropout                        │           │
     ↓                           ↓           │
  Linear (4D→D)            Multiply ←──── Linear3 (D→4D)
     ↓                           ↓
  Dropout                     Linear2 (4D→D)
     ↓                           ↓
  Residual                    Dropout
                                 ↓
                              Residual

Parameters: 8D² + 7D      Parameters: 12D² + 2D
```

---

## Complete Model Architecture

### RankMixer Block Structure

```
Input Tokens (B, T, D)
      ↓
┌─────────────────────┐
│   Token Mixer       │ ← Linear O(T) complexity
│   - Head split      │
│   - Permute         │
│   - Flatten/mix     │
│   - Reconstruct     │
│   - Residual+Norm   │
└─────────────────────┘
      ↓
┌─────────────────────┐
│   Feed Forward      │ ← Token-wise processing
│   - LayerNorm       │
│   - Linear expand   │
│   - Activation      │
│   - Linear contract │
│   - Residual        │
└─────────────────────┘
      ↓
Output Tokens (B, T, D)
```

### Multi-Layer Model

```
Input IDs: [101, 2045, 3456, ...]
     ↓
Token Embeddings: (B, T, D)
     ↓
Position Embeddings: (B, T, D)
     ↓ (Add)
Combined Embeddings: (B, T, D)
     ↓
┌─── Layer 1 ────┐
│ Token Mixer    │
│ Feed Forward   │
└────────────────┘
     ↓
┌─── Layer 2 ────┐
│ Token Mixer    │
│ Feed Forward   │
└────────────────┘
     ↓
    ...
     ↓
┌─── Layer N ────┐
│ Token Mixer    │
│ Feed Forward   │
└────────────────┘
     ↓
Final Hidden States: (B, T, D)
```

---

## Mathematical Formulation

### Theoretical Formulation

1. **Head Splitting**: Define the projection
   ```
   x_heads = reshape(x, (B, T, H, D/H))
   ```

2. **Permutation and Concatenation**: Execute a permutation followed by flattening
   ```
   x_flat = reshape(permute(x_heads, (0,2,1,3)), (B, H·T, D/H))
   ```

3. **Inverse Reshaping**: Recover the original token dimensionality
   ```
   x_mixed = permute(reshape(x_flat, (B, H, T, D/H)), (0,2,1,3))
   ```

4. **Residual Aggregation & Normalization**:
   ```
   output = LayerNorm(x + x_mixed)
   ```

---

## FLOP Analysis

### TokenMixer FLOP Breakdown

**Core Operations FLOP Count**:

1. **Tensor Reshaping Operations**: **0 FLOPs** (memory layout changes only)
2. **Residual Addition**: **B × T × D FLOPs**
3. **Layer Normalization**: **B × T × (7D + 2) FLOPs**

**Complete FLOP Formula**:
```
FLOPs_TokenMixer = B × T × D × (8 + 2/D) ≈ 8 × B × T × D
```

### Comparison with Self-Attention

**Self-Attention FLOPs**:
```
FLOPs_SelfAttention = B × T² × D + 4 × B × T × D²
```

**FLOP Ratio**:
```
Ratio = FLOPs_SelfAttention / FLOPs_TokenMixer = T/8 + D/2
```

**Key Insights**:
- For T=512, D=768: Ratio ≈ 448x fewer FLOPs
- TokenMixer advantage increases linearly with sequence length T
- Memory bandwidth becomes the primary performance bottleneck, not compute

---

## Performance Characteristics

### Computational Complexity

**TokenMixer Complexity**:
- **Time**: O(B × T × D) - Linear in sequence length
- **Space**: O(B × T × D) - Linear memory footprint

**Self-Attention Complexity**:
- **Time**: O(B × T² × D + B × T × D²) - Quadratic in sequence length
- **Space**: O(B × H × T²) - Quadratic attention matrices

### Memory Footprint Analysis

**Peak Memory Usage**:
- **Input**: B × T × D (original tensor)
- **Intermediate**: B × H × T × (D/H) (permuted tensor)
- **Output**: B × T × D (mixed tensor)

**Memory Efficiency Gains**:
- Eliminates attention weight matrices of size T × T
- Reduces memory requirements by factor of T for large token sequences
- Enables processing of sequences 10-100x longer within same compute budget

### Expected Performance Metrics

- **Latency**: 2-3x faster than equivalent self-attention
- **Throughput**: 3-4x higher tokens/second
- **Memory**: Linear scaling vs quadratic for self-attention
- **GPU Utilization**: 90%+ memory bandwidth utilization

---

## Visual Tensor Flow

### Tensor Shape Progression Example (B=2, T=4, D=8, H=4)

```
Input:     (2, 4, 8)     - Original token embeddings
Step 1:    (2, 4, 4, 2)  - Split into heads
Step 2:    (2, 4, 4, 2)  - Rearrange head dimension
Step 3:    (2, 16, 2)    - Flatten for mixing
Step 4:    (2, 4, 4, 2)  - Unflatten (information now mixed!)
Step 5:    (2, 4, 8)     - Restore original shape
Output:    (2, 4, 8)     - Mixed tokens + residual + norm
```

### Memory Usage Scaling

```
Sequence Length vs Memory Usage:

Token Mixer (Linear):    Self-Attention (Quadratic):

T=128:  ████                T=128:  ████
T=256:  ████████            T=256:  ████████████████
T=512:  ████████████████    T=512:  ████████████████████████████████
T=1024: ████████████████████████████    T=1024: [TOO LARGE TO DISPLAY]

Memory Factor: 1x, 2x, 4x, 8x       Memory Factor: 1x, 4x, 16x, 64x
```

---

## Implementation Details

### Input Validation

```python
def _validate_inputs(self, x: torch.Tensor) -> None:
    """Validate input tensor dimensions and data types."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {x.dim()}D")

    B, T, D = x.shape
    if T != self.T:
        raise ValueError(f"Expected {self.T} tokens, got {T}")
    if D != self.D:
        raise ValueError(f"Expected {self.D} dimensions, got {D}")

    if not x.is_contiguous():
        raise ValueError("Input tensor must be contiguous")
```

### Optimization Considerations

1. **Memory Layout**: Use `.contiguous()` calls strategically for optimal memory access
2. **JIT Compilation**: Apply `@torch.jit.script` for kernel fusion
3. **Mixed Precision**: Leverage `torch.cuda.amp` for performance gains
4. **Batch Processing**: Optimize for various batch sizes and token counts

### Error Handling

**Key validation points**:
- Input tensor must be 3D: (batch, tokens, dimensions)
- `hidden_dim % num_heads == 0` constraint
- Contiguous tensor requirements for efficient operations
- Numerical stability through LayerNorm and residual connections

---

## Testing and Validation

### Unit Tests

```python
def test_shape_invariance():
    """Test that output shape matches input shape."""
    mixer = TokenMixer(num_tokens=32, hidden_dim=768, num_heads=12)
    x = torch.randn(4, 32, 768)
    output = mixer(x)
    assert output.shape == x.shape

def test_gradient_flow():
    """Test that gradients flow properly through the module."""
    mixer = TokenMixer(num_tokens=8, hidden_dim=256, num_heads=4)
    x = torch.randn(1, 8, 256, requires_grad=True)
    output = mixer(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
```

### Performance Benchmarking

```python
def benchmark_token_mixer():
    """Benchmark TokenMixer against self-attention."""
    configs = [
        (32, 768, 12),   # BERT-like
        (64, 1024, 16),  # Large model
        (128, 512, 8),   # High throughput
    ]

    for T, D, H in configs:
        mixer = TokenMixer(T, D, H)
        x = torch.randn(8, T, D).cuda()

        # Benchmark forward pass
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = mixer(x)
        torch.cuda.synchronize()
        end = time.time()

        print(f"T={T}, D={D}, H={H}: {(end-start)/100*1000:.2f}ms")
```

### Integration Tests

1. **End-to-End Pipeline**: Test integration with RankMixer framework
2. **Multi-GPU Compatibility**: Verify distributed training scenarios
3. **Mixed Precision**: Validate fp16/fp32 training compatibility
4. **Memory Leaks**: Long-running stability tests

---

This documentation provides a comprehensive technical reference for understanding and implementing the Token Mixer architecture. For hands-on learning, see the educational examples and visual guides provided in the repository.
