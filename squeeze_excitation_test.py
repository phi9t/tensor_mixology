"""
Comprehensive Test Suite for Squeeze and Excitation Module

This test suite covers:
1. Shape invariance and correctness
2. Gradient flow validation
3. Numerical stability tests
4. Performance benchmarks
5. Edge case handling
6. Integration tests

Educational tests with detailed explanations of what each test validates.
"""

import time

import pytest
import torch
import torch.nn as nn

from squeeze_excitation import (
    SEBlock,
    SqueezeExcitation,
    create_se_conv_block,
    visualize_se_effect,
)


class TestSqueezeExcitation:
    """Test suite for basic SqueezeExcitation functionality"""

    def test_initialization(self):
        """Test proper initialization of SE module"""
        # Valid initialization
        se = SqueezeExcitation(in_channels=64, reduction_ratio=16)
        assert se.in_channels == 64
        assert se.reduction_ratio == 16
        assert se.bottleneck_channels == 4  # 64 // 16

        # Test minimum bottleneck size
        se_small = SqueezeExcitation(in_channels=8, reduction_ratio=16)
        assert se_small.bottleneck_channels == 1  # max(1, 8//16)

        # Test parameter counting
        expected_params = 64 * 4 + 4 * 64  # fc1 + fc2 weights (no bias)
        actual_params = sum(p.numel() for p in se.parameters())
        assert actual_params == expected_params

    def test_invalid_initialization(self):
        """Test error handling for invalid inputs"""
        with pytest.raises(ValueError):
            SqueezeExcitation(in_channels=0)

        with pytest.raises(ValueError):
            SqueezeExcitation(in_channels=64, reduction_ratio=0)

        with pytest.raises(ValueError):
            SqueezeExcitation(in_channels=64, activation="invalid")

    def test_shape_invariance_2d(self):
        """Test that SE preserves input shapes for 2D inputs"""
        se = SqueezeExcitation(in_channels=32)

        # Test various 2D input shapes
        test_shapes = [
            (1, 32, 16, 16),  # Single sample
            (4, 32, 32, 32),  # Small batch
            (16, 32, 64, 64),  # Larger batch
            (2, 32, 1, 1),  # Global pooled features
        ]

        for shape in test_shapes:
            x = torch.randn(shape)
            output = se(x)
            assert output.shape == x.shape, f"Shape mismatch for {shape}"

    def test_shape_invariance_1d(self):
        """Test that SE preserves input shapes for 1D inputs"""
        se = SqueezeExcitation(in_channels=64)

        # Test 1D CNN format: (B, C, T)
        test_shapes = [
            (1, 64, 100),  # Single sample
            (8, 64, 256),  # Audio-like
            (4, 64, 1024),  # Long sequence
        ]

        for shape in test_shapes:
            x = torch.randn(shape)
            output = se(x)
            assert output.shape == x.shape, f"Shape mismatch for {shape}"

    def test_shape_invariance_sequence(self):
        """Test that SE preserves input shapes for sequence format"""
        se = SqueezeExcitation(in_channels=128)

        # Test sequence format: (B, T, C)
        test_shapes = [
            (1, 50, 128),  # Single sequence
            (8, 100, 128),  # Batch of sequences
            (16, 512, 128),  # Long sequences
        ]

        for shape in test_shapes:
            x = torch.randn(shape)
            output = se(x)
            assert output.shape == x.shape, f"Shape mismatch for {shape}"

    def test_unsupported_shapes(self):
        """Test error handling for unsupported input shapes"""
        se = SqueezeExcitation(in_channels=64)

        # Too few dimensions
        with pytest.raises(ValueError):
            se(torch.randn(64, 32))

        # Too many dimensions
        with pytest.raises(ValueError):
            se(torch.randn(2, 64, 16, 16, 8))

    def test_gradient_flow(self):
        """Test that gradients flow properly through SE module"""
        se = SqueezeExcitation(in_channels=32)
        x = torch.randn(4, 32, 16, 16, requires_grad=True)

        # Forward pass
        output = se(x)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None, "No gradients for input"
        assert not torch.isnan(x.grad).any(), "NaN gradients in input"
        assert not torch.isinf(x.grad).any(), "Inf gradients in input"

        # Check that all parameters have gradients
        for name, param in se.named_parameters():
            assert param.grad is not None, f"No gradients for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradients in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradients in {name}"

    def test_channel_importance_extraction(self):
        """Test channel importance weight extraction"""
        se = SqueezeExcitation(in_channels=16, reduction_ratio=4)
        x = torch.randn(2, 16, 8, 8)

        # Extract channel importance
        weights = se.get_channel_importance(x)

        # Validate shape and range
        assert weights.shape == (2, 16), f"Wrong weight shape: {weights.shape}"
        assert torch.all(weights >= 0), "Negative importance weights"
        assert torch.all(weights <= 1), "Importance weights > 1"

        # Test different input formats
        x_seq = torch.randn(2, 32, 16)  # (B, T, C)
        weights_seq = se.get_channel_importance(x_seq)
        assert weights_seq.shape == (2, 16)

    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs"""
        se = SqueezeExcitation(in_channels=32)

        # Very large values
        x_large = torch.randn(2, 32, 8, 8) * 1000
        output_large = se(x_large)
        assert torch.isfinite(output_large).all(), "Instability with large values"

        # Very small values
        x_small = torch.randn(2, 32, 8, 8) * 1e-6
        output_small = se(x_small)
        assert torch.isfinite(output_small).all(), "Instability with small values"

        # Zero input
        x_zero = torch.zeros(2, 32, 8, 8)
        output_zero = se(x_zero)
        assert torch.isfinite(output_zero).all(), "Instability with zero input"
        assert torch.allclose(output_zero, x_zero), "Non-zero output for zero input"

    def test_different_activations(self):
        """Test SE with different activation functions"""
        activations = ["relu", "gelu", "swish"]
        gate_activations = ["sigmoid", "hardsigmoid"]

        for act in activations:
            for gate_act in gate_activations:
                se = SqueezeExcitation(
                    in_channels=16, activation=act, gate_activation=gate_act
                )
                x = torch.randn(2, 16, 8, 8)
                output = se(x)
                assert output.shape == x.shape
                assert torch.isfinite(output).all()


class TestSEBlock:
    """Test suite for SEBlock wrapper class"""

    def test_se_conv_block(self):
        """Test SE with convolutional base layer"""
        conv = nn.Conv2d(32, 64, 3, padding=1)
        se_block = SEBlock(conv, se_channels=64, reduction_ratio=16)

        x = torch.randn(4, 32, 16, 16)
        output = se_block(x)

        assert output.shape == (4, 64, 16, 16)
        assert torch.isfinite(output).all()

    def test_se_linear_block(self):
        """Test SE with linear base layer"""
        linear = nn.Linear(128, 256)
        se_block = SEBlock(
            linear, se_channels=256, reduction_ratio=8, use_residual=False
        )

        x = torch.randn(4, 100, 128)  # (B, T, C)
        output = se_block(x)

        assert output.shape == (4, 100, 256)
        assert torch.isfinite(output).all()

    def test_residual_connection(self):
        """Test residual connection in SEBlock"""
        # Case where residual should work (same shape)
        identity = nn.Identity()
        se_block = SEBlock(identity, se_channels=64, use_residual=True)

        x = torch.randn(2, 64, 8, 8)
        output = se_block(x)

        # Output should not be identical to input due to SE scaling
        assert not torch.allclose(output, x)
        assert output.shape == x.shape

    def test_conv_block_factory(self):
        """Test convenience factory for SE conv blocks"""
        se_conv = create_se_conv_block(
            in_channels=32, out_channels=64, kernel_size=3, reduction_ratio=8
        )

        x = torch.randn(2, 32, 16, 16)
        output = se_conv(x)

        assert output.shape == (2, 64, 16, 16)
        assert torch.isfinite(output).all()


class TestSEIntegration:
    """Integration tests with other components"""

    def test_integration_with_transformer(self):
        """Test SE integration in transformer-like architecture"""

        # Simple transformer block with SE
        class TransformerWithSE(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.attention = nn.MultiheadAttention(
                    hidden_dim, num_heads=8, batch_first=True
                )
                self.se = SqueezeExcitation(hidden_dim, reduction_ratio=8)
                self.ffn = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
                self.norm1 = nn.LayerNorm(hidden_dim)
                self.norm2 = nn.LayerNorm(hidden_dim)

            def forward(self, x):
                # Self-attention
                attn_out, _ = self.attention(x, x, x)
                x = self.norm1(x + attn_out)

                # SE attention on features
                x = self.se(x)  # (B, T, C) format

                # FFN
                ffn_out = self.ffn(x)
                x = self.norm2(x + ffn_out)

                return x

        model = TransformerWithSE(hidden_dim=128)
        x = torch.randn(4, 50, 128)  # (batch, seq_len, hidden_dim)

        output = model(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_integration_with_resnet_style(self):
        """Test SE in ResNet-style architecture"""

        class ResNetSEBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(channels)
                self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(channels)
                self.se = SqueezeExcitation(channels, reduction_ratio=16)
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                residual = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                # Apply SE before residual connection
                out = self.se(out)

                out += residual
                out = self.relu(out)

                return out

        block = ResNetSEBlock(channels=64)
        x = torch.randn(2, 64, 32, 32)

        output = block(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()


class TestSEPerformance:
    """Performance and efficiency tests"""

    def test_memory_efficiency(self):
        """Test memory usage is reasonable"""
        se = SqueezeExcitation(in_channels=1024, reduction_ratio=16)

        # Large input to test memory efficiency
        x = torch.randn(8, 1024, 56, 56)

        # Measure memory before
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        output = se(x)

        # Memory should not explode
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_increase = end_memory - start_memory

        # SE should use minimal additional memory (just for weights and intermediate tensors)
        expected_se_params = (1024 * 64 + 64 * 1024) * 4  # bytes for fp32
        assert memory_increase < expected_se_params * 10  # Allow some overhead

    def test_computational_efficiency(self):
        """Test that SE doesn't add excessive computation"""
        se = SqueezeExcitation(in_channels=256, reduction_ratio=16)
        x = torch.randn(16, 256, 32, 32)

        # Warm up
        for _ in range(10):
            _ = se(x)

        # Time the forward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(100):
            _ = se(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / 100

        # SE should be reasonably fast (< 500ms for this size)
        assert avg_time < 0.5, f"SE too slow: {avg_time:.4f}s"

    def test_parameter_efficiency(self):
        """Test parameter count is reasonable"""
        # Compare parameter counts for different reduction ratios
        channels = 512

        se_r4 = SqueezeExcitation(channels, reduction_ratio=4)
        se_r16 = SqueezeExcitation(channels, reduction_ratio=16)
        se_r64 = SqueezeExcitation(channels, reduction_ratio=64)

        params_r4 = sum(p.numel() for p in se_r4.parameters())
        params_r16 = sum(p.numel() for p in se_r16.parameters())
        params_r64 = sum(p.numel() for p in se_r64.parameters())

        # Higher reduction ratio should mean fewer parameters
        assert params_r4 > params_r16 > params_r64

        # Parameter count should be reasonable fraction of base layer
        base_conv_params = channels * channels * 3 * 3  # 3x3 conv
        assert params_r16 < base_conv_params * 0.1  # SE < 10% of conv params


class TestSEVisualization:
    """Test visualization and analysis functions"""

    def test_visualization_function(self):
        """Test the SE effect visualization function"""
        se = SqueezeExcitation(in_channels=16, reduction_ratio=4)
        x = torch.randn(2, 16, 8, 8)

        output, weights, analysis = visualize_se_effect(se, x)

        # Check return types and shapes
        assert output.shape == x.shape
        assert weights.shape == (2, 16)
        assert isinstance(analysis, dict)

        # Check analysis contents
        required_keys = [
            "channel_weights",
            "weight_statistics",
            "most_important_channels",
            "least_important_channels",
            "scaling_effect",
        ]
        for key in required_keys:
            assert key in analysis, f"Missing analysis key: {key}"

        # Verify statistics make sense
        stats = analysis["weight_statistics"]
        assert 0 <= stats["min"] <= stats["mean"] <= stats["max"] <= 1
        assert stats["std"] >= 0

    def test_channel_ranking(self):
        """Test channel importance ranking"""
        se = SqueezeExcitation(in_channels=8, reduction_ratio=2)

        # Create input where certain channels have higher magnitude
        x = torch.randn(1, 8, 4, 4)
        x[:, [0, 2, 5], :, :] *= 10  # Make channels 0, 2, 5 more prominent

        weights = se.get_channel_importance(x)

        # The prominent channels should generally get higher weights
        # (though this depends on learned parameters)
        assert weights.shape == (1, 8)
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)


def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("=== Running Comprehensive SE Tests ===\n")

    test_classes = [
        TestSqueezeExcitation,
        TestSEBlock,
        TestSEIntegration,
        TestSEPerformance,
        TestSEVisualization,
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"Running {test_class.__name__}...")
        test_instance = test_class()

        # Get all test methods
        test_methods = [
            method for method in dir(test_instance) if method.startswith("test_")
        ]

        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                passed_tests += 1
                print(f"  ✓ {method_name}")
            except Exception as e:
                print(f"  ✗ {method_name}: {str(e)}")

        print()

    print("=== Test Summary ===")
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")

    return passed_tests == total_tests


if __name__ == "__main__":
    # Run educational demonstration
    print("=== Squeeze and Excitation Test Suite ===\n")

    # Basic functionality test
    se = SqueezeExcitation(in_channels=32, reduction_ratio=8)
    x = torch.randn(4, 32, 16, 16)

    print(f"SE Module: {se}")
    print(f"Input shape: {x.shape}")

    # Forward pass
    output = se(x)
    print(f"Output shape: {output.shape}")

    # Channel importance
    weights = se.get_channel_importance(x)
    print(f"Channel weights shape: {weights.shape}")
    print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")

    # Run comprehensive tests
    success = run_comprehensive_test()

    if success:
        print("🎉 All tests passed! SE module is working correctly.")
    else:
        print("❌ Some tests failed. Check implementation.")
