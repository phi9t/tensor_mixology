"""Tests for feed forward modules."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from feed_forward import FeedForward, GLUFeedForward, SwiGLUFeedForward


class TestFeedForward:
    """Test cases for FeedForward module."""

    def test_initialization(self):
        """Test FeedForward initialization."""
        ff = FeedForward(hidden_dim=512, ff_dim=2048)

        assert ff.hidden_dim == 512
        assert ff.ff_dim == 2048
        assert ff.dropout_rate == 0.1
        assert isinstance(ff.norm, nn.LayerNorm)
        assert isinstance(ff.linear1, nn.Linear)
        assert isinstance(ff.linear2, nn.Linear)
        assert isinstance(ff.dropout1, nn.Dropout)
        assert isinstance(ff.dropout2, nn.Dropout)

    def test_initialization_custom_params(self):
        """Test FeedForward initialization with custom parameters."""
        ff = FeedForward(
            hidden_dim=768, ff_dim=3072, dropout_rate=0.2, activation="relu", bias=False
        )

        assert ff.hidden_dim == 768
        assert ff.ff_dim == 3072
        assert ff.dropout_rate == 0.2
        assert isinstance(ff.activation, nn.ReLU)
        assert ff.linear1.bias is None
        assert ff.linear2.bias is None

    def test_activation_functions(self):
        """Test different activation functions."""
        activations = ["gelu", "relu", "swish", "silu"]

        for activation in activations:
            ff = FeedForward(hidden_dim=256, ff_dim=1024, activation=activation)
            x = torch.randn(2, 10, 256)
            output = ff(x)
            assert output.shape == x.shape

    def test_invalid_activation(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError, match="Unsupported activation"):
            FeedForward(hidden_dim=256, ff_dim=1024, activation="invalid")

    def test_forward_pass(self):
        """Test forward pass with residual connection."""
        ff = FeedForward(hidden_dim=512, ff_dim=2048)
        x = torch.randn(4, 16, 512)

        output = ff(x)

        assert output.shape == x.shape
        assert not torch.allclose(
            output, x
        )  # Should be different due to transformation

    def test_forward_pass_with_dropout(self):
        """Test forward pass with dropout enabled."""
        ff = FeedForward(hidden_dim=256, ff_dim=1024, dropout_rate=0.5)
        x = torch.randn(2, 8, 256)

        # Set to eval mode to disable dropout for deterministic testing
        ff.eval()
        output1 = ff(x)
        output2 = ff(x)

        # In eval mode, outputs should be identical
        assert torch.allclose(output1, output2)

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        ff = FeedForward(hidden_dim=128, ff_dim=512)
        x = torch.randn(1, 5, 128, requires_grad=True)

        output = ff(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_parameter_count(self):
        """Test parameter count calculation."""
        hidden_dim = 256
        ff_dim = 1024

        ff = FeedForward(hidden_dim=hidden_dim, ff_dim=ff_dim)

        # LayerNorm: hidden_dim * 2 (weight + bias)
        layernorm_params = hidden_dim * 2
        # Linear1: hidden_dim * ff_dim + ff_dim (weight + bias)
        linear1_params = hidden_dim * ff_dim + ff_dim
        # Linear2: ff_dim * hidden_dim + hidden_dim (weight + bias)
        linear2_params = ff_dim * hidden_dim + hidden_dim

        expected_params = layernorm_params + linear1_params + linear2_params
        actual_params = sum(p.numel() for p in ff.parameters())

        assert actual_params == expected_params

    def test_zero_input(self):
        """Test behavior with zero input."""
        ff = FeedForward(hidden_dim=128, ff_dim=512)
        x = torch.zeros(2, 4, 128)

        output = ff(x)

        assert output.shape == x.shape
        # Output should not be zero due to residual connection
        assert not torch.allclose(output, torch.zeros_like(output))


class TestGLUFeedForward:
    """Test cases for GLUFeedForward module."""

    def test_initialization(self):
        """Test GLUFeedForward initialization."""
        ff = GLUFeedForward(hidden_dim=512, ff_dim=2048)

        assert ff.hidden_dim == 512
        assert ff.ff_dim == 2048
        assert ff.dropout_rate == 0.1
        assert isinstance(ff.norm, nn.LayerNorm)
        assert isinstance(ff.linear1, nn.Linear)
        assert isinstance(ff.linear2, nn.Linear)
        assert ff.linear1.out_features == 2048
        assert ff.linear2.in_features == 1024  # Half of ff_dim

    def test_initialization_odd_ff_dim(self):
        """Test that odd ff_dim raises error."""
        with pytest.raises(ValueError, match="ff_dim must be even"):
            GLUFeedForward(hidden_dim=256, ff_dim=1023)

    def test_forward_pass(self):
        """Test forward pass with GLU activation."""
        ff = GLUFeedForward(hidden_dim=256, ff_dim=1024)
        x = torch.randn(3, 8, 256)

        output = ff(x)

        assert output.shape == x.shape
        assert not torch.allclose(output, x)

    def test_glu_activation_mechanism(self):
        """Test that GLU activation works correctly."""
        ff = GLUFeedForward(hidden_dim=128, ff_dim=256)
        x = torch.randn(1, 4, 128)

        # Get intermediate values
        x_norm = ff.norm(x)
        x_linear = ff.linear1(x_norm)

        # Split for GLU
        gate, value = x_linear.chunk(2, dim=-1)
        glu_output = torch.sigmoid(gate) * value

        # This should match the internal computation
        assert glu_output.shape == (1, 4, 128)

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        ff = GLUFeedForward(hidden_dim=128, ff_dim=256)
        x = torch.randn(1, 5, 128, requires_grad=True)

        output = ff(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_parameter_count(self):
        """Test parameter count calculation."""
        hidden_dim = 256
        ff_dim = 1024

        ff = GLUFeedForward(hidden_dim=hidden_dim, ff_dim=ff_dim)

        # LayerNorm: hidden_dim * 2 (weight + bias)
        layernorm_params = hidden_dim * 2
        # Linear1: hidden_dim * ff_dim + ff_dim (weight + bias)
        linear1_params = hidden_dim * ff_dim + ff_dim
        # Linear2: (ff_dim//2) * hidden_dim + hidden_dim (weight + bias)
        linear2_params = (ff_dim // 2) * hidden_dim + hidden_dim

        expected_params = layernorm_params + linear1_params + linear2_params
        actual_params = sum(p.numel() for p in ff.parameters())

        assert actual_params == expected_params


class TestSwiGLUFeedForward:
    """Test cases for SwiGLUFeedForward module."""

    def test_initialization(self):
        """Test SwiGLUFeedForward initialization."""
        ff = SwiGLUFeedForward(hidden_dim=512, ff_dim=2048)

        assert ff.hidden_dim == 512
        assert ff.ff_dim == 2048
        assert ff.dropout_rate == 0.1
        assert isinstance(ff.norm, nn.LayerNorm)
        assert isinstance(ff.w1, nn.Linear)
        assert isinstance(ff.w2, nn.Linear)
        assert isinstance(ff.w3, nn.Linear)
        assert isinstance(ff.dropout, nn.Dropout)

    def test_initialization_no_bias(self):
        """Test SwiGLUFeedForward initialization without bias."""
        ff = SwiGLUFeedForward(hidden_dim=256, ff_dim=1024, bias=False)

        assert ff.w1.bias is None
        assert ff.w2.bias is None
        assert ff.w3.bias is None

    def test_forward_pass(self):
        """Test forward pass with SwiGLU activation."""
        ff = SwiGLUFeedForward(hidden_dim=256, ff_dim=1024)
        x = torch.randn(2, 10, 256)

        output = ff(x)

        assert output.shape == x.shape
        assert not torch.allclose(output, x)

    def test_swiglu_activation_mechanism(self):
        """Test that SwiGLU activation works correctly."""
        ff = SwiGLUFeedForward(hidden_dim=128, ff_dim=256)
        x = torch.randn(1, 4, 128)

        # Get intermediate values
        x_norm = ff.norm(x)

        # SwiGLU: SiLU(W1(x)) * W3(x)
        gate = F.silu(ff.w1(x_norm))
        value = ff.w3(x_norm)
        swiglu_output = gate * value

        # This should match the internal computation
        assert swiglu_output.shape == (1, 4, 256)

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        ff = SwiGLUFeedForward(hidden_dim=128, ff_dim=256)
        x = torch.randn(1, 5, 128, requires_grad=True)

        output = ff(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_parameter_count(self):
        """Test parameter count calculation."""
        hidden_dim = 256
        ff_dim = 1024

        ff = SwiGLUFeedForward(hidden_dim=hidden_dim, ff_dim=ff_dim)

        # LayerNorm: hidden_dim * 2 (weight + bias)
        layernorm_params = hidden_dim * 2
        # W1: hidden_dim * ff_dim (weight only, no bias by default)
        w1_params = hidden_dim * ff_dim
        # W2: ff_dim * hidden_dim (weight only, no bias by default)
        w2_params = ff_dim * hidden_dim
        # W3: hidden_dim * ff_dim (weight only, no bias by default)
        w3_params = hidden_dim * ff_dim

        expected_params = layernorm_params + w1_params + w2_params + w3_params
        actual_params = sum(p.numel() for p in ff.parameters())

        assert actual_params == expected_params

    def test_different_ff_dim(self):
        """Test with different feed forward dimensions."""
        ff = SwiGLUFeedForward(hidden_dim=512, ff_dim=4096)
        x = torch.randn(1, 8, 512)

        output = ff(x)
        assert output.shape == x.shape


class TestFeedForwardIntegration:
    """Integration tests for feed forward modules."""

    def test_all_variants_same_shape(self):
        """Test that all variants produce same output shape."""
        hidden_dim = 256
        ff_dim = 1024
        batch_size = 2
        seq_len = 10

        x = torch.randn(batch_size, seq_len, hidden_dim)

        ff_standard = FeedForward(hidden_dim=hidden_dim, ff_dim=ff_dim)
        ff_glu = GLUFeedForward(hidden_dim=hidden_dim, ff_dim=ff_dim)
        ff_swiglu = SwiGLUFeedForward(hidden_dim=hidden_dim, ff_dim=ff_dim)

        output_standard = ff_standard(x)
        output_glu = ff_glu(x)
        output_swiglu = ff_swiglu(x)

        assert output_standard.shape == x.shape
        assert output_glu.shape == x.shape
        assert output_swiglu.shape == x.shape

    def test_dropout_consistency(self):
        """Test that dropout works consistently across variants."""
        hidden_dim = 128
        ff_dim = 512

        variants = [
            FeedForward(hidden_dim=hidden_dim, ff_dim=ff_dim, dropout_rate=0.5),
            GLUFeedForward(hidden_dim=hidden_dim, ff_dim=ff_dim, dropout_rate=0.5),
            SwiGLUFeedForward(hidden_dim=hidden_dim, ff_dim=ff_dim, dropout_rate=0.5),
        ]

        x = torch.randn(1, 4, hidden_dim)

        for variant in variants:
            variant.eval()  # Disable dropout
            output1 = variant(x)
            output2 = variant(x)
            assert torch.allclose(output1, output2)

    def test_parameter_comparison(self):
        """Compare parameter counts across variants."""
        hidden_dim = 256
        ff_dim = 1024

        ff_standard = FeedForward(hidden_dim=hidden_dim, ff_dim=ff_dim)
        ff_glu = GLUFeedForward(hidden_dim=hidden_dim, ff_dim=ff_dim)
        ff_swiglu = SwiGLUFeedForward(hidden_dim=hidden_dim, ff_dim=ff_dim)

        params_standard = sum(p.numel() for p in ff_standard.parameters())
        params_glu = sum(p.numel() for p in ff_glu.parameters())
        params_swiglu = sum(p.numel() for p in ff_swiglu.parameters())

        # All should have different parameter counts
        assert params_standard != params_glu
        assert params_standard != params_swiglu
        assert params_glu != params_swiglu

        # SwiGLU should have the most parameters (3 linear layers)
        assert params_swiglu > params_standard
        assert params_swiglu > params_glu


if __name__ == "__main__":
    pytest.main([__file__])
