# CLAUDE.md - Tensor Mixology

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Tensor Mixology** is a PyTorch-based implementation of a Token Mixer module for the RankMixer framework. The project focuses on creating an efficient, parameter-free token mixing mechanism that provides linear time complexity O(T) as an alternative to quadratic O(T²) self-attention mechanisms.

The Token Mixer performs head-wise feature subspace projections and global token reassembly through tensor reshaping and permutation operations without learnable parameters, maximizing GPU FLOP utilization and memory bandwidth efficiency.

## Environment Setup

This project uses `uv` for Python package management. The virtual environment is located at `.venv/`.

### Package Management Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install packages (use uv instead of pip)
uv pip install <package_name>

# List installed packages
uv pip list

# Install from requirements (if available)
uv pip install -r requirements.txt
```

### Core Dependencies

The project uses the following key dependencies:
- **torch** (2.7.1): Core PyTorch framework for neural network implementation
- **torchvision** (0.22.1): Computer vision utilities
- **numpy** (2.3.2): Numerical computing
- **plotly** (6.2.0): Data visualization for performance analysis
- **polars** (1.32.0): High-performance data manipulation
- **rich** (14.1.0): Terminal formatting and progress bars
- **tqdm** (4.67.1): Progress bars for training loops

## Architecture and Implementation

### Core Module: TokenMixer

The main implementation follows this architectural pattern:

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
        # 4. Inverse operations to restore (B, T, D)
        # 5. Residual connection + LayerNorm
        # Output: (B, T, D)
```

### Key Design Principles

1. **Parameter-free mixing**: No learnable weights, only tensor operations
2. **Linear complexity**: O(T) vs O(T²) for self-attention
3. **Hardware-conscious**: Optimized for GPU memory bandwidth
4. **Modular composition**: Integrates seamlessly with RankMixer blocks

## Development Workflow

### Testing Strategy

The project uses comprehensive testing including:

```python
# Unit tests for shape invariance
def test_shape_invariance():
    mixer = TokenMixer(num_tokens=12, hidden_dim=768, num_heads=12)
    x = torch.randn(4, 12, 768)
    output = mixer(x)
    assert output.shape == x.shape

# Gradient flow validation
def test_gradient_flow():
    # Ensure gradients propagate properly through residual connections

# Performance benchmarking
def benchmark_token_mixer():
    # Compare against self-attention mechanisms
```

### Performance Analysis Commands

```python
# Memory profiling
torch.profiler.profile() # Use for kernel-level analysis

# Benchmark comparisons
torch.utils.benchmark # Compare with self-attention

# Mixed precision testing
torch.cuda.amp # Validate fp16 compatibility
```

### Optimization Considerations

1. **Memory Layout**: Use `.contiguous()` calls strategically for optimal memory access
2. **JIT Compilation**: Apply `@torch.jit.script` for kernel fusion
3. **Mixed Precision**: Leverage `torch.cuda.amp` for performance gains
4. **Batch Processing**: Optimize for various batch sizes and token counts

## Expected Performance Characteristics

- **Latency**: 2-3x faster than equivalent self-attention
- **Throughput**: 3-4x higher tokens/second
- **Memory**: Linear scaling vs quadratic for self-attention
- **GPU Utilization**: 90%+ memory bandwidth utilization

## Integration Patterns

The TokenMixer is designed to integrate within:
1. **RankMixer blocks**: Two-stage token mixing + feed-forward
2. **Transformer variants**: Drop-in replacement for attention layers
3. **Multi-modal architectures**: Cross-modal token interactions

## Common Development Tasks

When implementing or modifying the TokenMixer:

1. **Validate tensor shapes** at each transformation step
2. **Test gradient flow** through residual connections
3. **Benchmark performance** against baseline attention mechanisms
4. **Profile memory usage** to ensure linear scaling
5. **Test edge cases** (single tokens, zero inputs, extreme dimensions)

## Error Handling

Key validation points:
- Input tensor must be 3D: (batch, tokens, dimensions)
- `hidden_dim % num_heads == 0` constraint
- Contiguous tensor requirements for efficient operations
- Numerical stability through LayerNorm and residual connections