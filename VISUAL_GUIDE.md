# Visual Guide: Understanding Tensor Mixology Architecture

This guide provides visual representations of tensor shape transformations and architectural concepts to help engineers and researchers understand the Token Mixer implementation in Tensor Mixology.

## Table of Contents

1. [Tensor Shape Basics](#tensor-shape-basics)
2. [Token Mixer Step-by-Step](#token-mixer-step-by-step)
3. [Feed Forward Network Flow](#feed-forward-network-flow)
4. [Complete Architecture Overview](#complete-architecture-overview)
5. [Complexity Comparison](#complexity-comparison)

---

## Tensor Shape Basics

### Understanding 3D Tensors in Sequence Models

```
Input Tensor Shape: (Batch, Tokens, Features)
                     (B,     T,      D)

Example: (4, 8, 768)
         │  │   │
         │  │   └── 768 features per token (hidden dimension)
         │  └────── 8 tokens per sequence (sequence length)
         └───────── 4 sequences per batch (batch size)
```

### Visual Representation

```
Batch Dimension (B=4):
┌─────────────────────────────────────────────┐
│ Sequence 1: [Token1] [Token2] ... [Token8]  │ ← Shape: (8, 768)
│ Sequence 2: [Token1] [Token2] ... [Token8]  │
│ Sequence 3: [Token1] [Token2] ... [Token8]  │
│ Sequence 4: [Token1] [Token2] ... [Token8]  │
└─────────────────────────────────────────────┘

Each Token: [f1, f2, f3, ..., f768] ← 768 features
```

---

## Token Mixer Step-by-Step

### Input Configuration
- **Batch Size (B)**: 2
- **Tokens (T)**: 4
- **Hidden Dim (D)**: 8
- **Heads (H)**: 4
- **Head Dim (d)**: 2

### Step 1: Head Partitioning
```
Input:     (2, 4, 8)     Original token embeddings
            ↓
Reshape:   (2, 4, 4, 2)  Split into heads

Visual:
Token 0: [a1,a2,a3,a4,a5,a6,a7,a8] → [[a1,a2], [a3,a4], [a5,a6], [a7,a8]]
Token 1: [b1,b2,b3,b4,b5,b6,b7,b8] → [[b1,b2], [b3,b4], [b5,b6], [b7,b8]]
Token 2: [c1,c2,c3,c4,c5,c6,c7,c8] → [[c1,c2], [c3,c4], [c5,c6], [c7,c8]]
Token 3: [d1,d2,d3,d4,d5,d6,d7,d8] → [[d1,d2], [d3,d4], [d5,d6], [d7,d8]]
         │                            │
         └── Original                 └── 4 heads of dimension 2 each
```

### Step 2: Permutation
```
Before:    (2, 4, 4, 2)  (Batch, Tokens, Heads, HeadDim)
           ↓ permute(0,2,1,3)
After:     (2, 4, 4, 2)  (Batch, Heads, Tokens, HeadDim)

Visual Layout Change:
Before: [Batch][Token0,Token1,Token2,Token3][Head0,Head1,Head2,Head3][Features]
After:  [Batch][Head0,Head1,Head2,Head3][Token0,Token1,Token2,Token3][Features]

Head 0: [[a1,a2], [b1,b2], [c1,c2], [d1,d2]]  ← All tokens, head 0
Head 1: [[a3,a4], [b3,b4], [c3,c4], [d3,d4]]  ← All tokens, head 1
Head 2: [[a5,a6], [b5,b6], [c5,c6], [d5,d6]]  ← All tokens, head 2
Head 3: [[a7,a8], [b7,b8], [c7,c8], [d7,d8]]  ← All tokens, head 3
```

### Step 3: Flatten for Mixing
```
Before:    (2, 4, 4, 2)   Separate heads and tokens
           ↓ flatten middle dimensions
After:     (2, 16, 2)     Combined sequence of length 16

Flattened Sequence:
[a1,a2] [b1,b2] [c1,c2] [d1,d2]   ← Head 0 tokens
[a3,a4] [b3,b4] [c3,c4] [d3,d4]   ← Head 1 tokens
[a5,a6] [b5,b6] [c5,c6] [d5,d6]   ← Head 2 tokens
[a7,a8] [b7,b8] [c7,c8] [d7,d8]   ← Head 3 tokens

🔄 MIXING HAPPENS HERE: Information from different tokens is now interleaved!
```

### Step 4: Unflatten (Information Mixed!)
```
Before:    (2, 16, 2)     Mixed sequence
           ↓ unflatten back to heads
After:     (2, 4, 4, 2)   Heads and tokens separated again

BUT: Information has been redistributed!

Head 0: [[a1,a2], [b1,b2], [c1,c2], [d1,d2]]  ← Same data positions
Head 1: [[a3,a4], [b3,b4], [c3,c4], [d3,d4]]  ← But tokens have been
Head 2: [[a5,a6], [b5,b6], [c5,c6], [d5,d6]]  ← processed together
Head 3: [[a7,a8], [b7,b8], [c7,c8], [d7,d8]]  ← in the flattened form
```

### Step 5: Residual + Normalization
```
Mixed:     (2, 4, 8)  ← Output from Step 5
Original:  (2, 4, 8)  ← Stored input
           ↓ Add them together
Sum:       (2, 4, 8)  ← Element-wise addition
           ↓ LayerNorm
Final:     (2, 4, 8)  ← Normalized output
```

---

## Feed Forward Network Flow

### Standard Feed Forward Architecture

```
Input:     (B, T, D)    Example: (4, 8, 768)
           ↓
LayerNorm: (B, T, D)    Normalize features
           ↓
Linear1:   (B, T, 4D)   Expand: (4, 8, 3072)
           ↓
Activation:(B, T, 4D)   Apply GELU/ReLU/SiLU
           ↓
Dropout:   (B, T, 4D)   Regularization (training only)
           ↓
Linear2:   (B, T, D)    Contract: (4, 8, 768)
           ↓
Dropout:   (B, T, D)    Final regularization
           ↓
Residual:  (B, T, D)    Add original input
           ↓
Output:    (B, T, D)    Same shape as input
```

### Parameter Distribution

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

For D=768: ~4.7M parameters per FFN layer
```

### SwiGLU Variant Comparison

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

## Complete Architecture Overview

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

## Complexity Comparison

### Computational Complexity

```
Token Mixer:
┌─────────────────────────────────┐
│ Operations         │ FLOPs      │
├─────────────────────────────────┤
│ Reshape ops        │ 0          │
│ Residual add       │ B×T×D      │
│ LayerNorm          │ ~7×B×T×D   │
├─────────────────────────────────┤
│ Total              │ ~8×B×T×D   │
│ Complexity         │ O(T)       │
└─────────────────────────────────┘

Self-Attention:
┌─────────────────────────────────┐
│ Operations         │ FLOPs      │
├─────────────────────────────────┤
│ QKV projections    │ 3×B×T×D²   │
│ Attention scores   │ B×H×T²×d   │
│ Attention weights  │ B×H×T²×d   │
│ Output projection  │ B×T×D²     │
├─────────────────────────────────┤
│ Total              │ ~B×T²×D    │
│ Complexity         │ O(T²)      │
└─────────────────────────────────┘
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

### Performance Scaling Chart

```
FLOP Ratio (Self-Attention / Token Mixer) vs Sequence Length:

    1000x │                                              ╭─
          │                                        ╭─────╯
     100x │                                  ╭─────╯
          │                            ╭─────╯
      10x │                      ╭─────╯
          │                ╭─────╯
       1x │──────────╭─────╯
          └─────────────────────────────────────────────
           128   256   512   1024  2048  4096  8192
                        Sequence Length

Key Insight: Token Mixer advantage grows linearly with sequence length!
At T=512, D=768: Token Mixer is ~448x more efficient than self-attention
```

---

## Implementation Tips

### Memory Optimization
```python
# ✅ Good: Use .contiguous() after permutations
x = x.permute(0, 2, 1, 3).contiguous()

# ❌ Bad: Multiple non-contiguous operations
x = x.permute(0, 2, 1, 3).view(B, H*T, d).permute(0, 2, 1)
```

### Debugging Tensor Shapes
```python
def debug_shapes(x, step_name):
    print(f"{step_name}: {x.shape}, contiguous: {x.is_contiguous()}")
    return x

# Use in forward pass:
x = debug_shapes(x.view(B, T, H, d), "after_head_split")
```

### Educational Mode
```python
# Enable educational prints
mixer = TokenMixer(num_tokens=8, hidden_dim=256, num_heads=8)
output = mixer(input_tensor)  # Prints step-by-step progress

# Disable for production
output = mixer.forward_production(input_tensor)  # No prints
```

---

This visual guide should help you understand the Token Mixer architecture from both conceptual and implementation perspectives. The key insight is that clever tensor reshaping can achieve global token mixing with linear complexity, making it a powerful alternative to quadratic self-attention mechanisms.
