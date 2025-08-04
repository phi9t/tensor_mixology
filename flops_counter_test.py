#!/usr/bin/env python3
"""Simple test script for FLOPs counter functionality."""

import torch
import torch.nn as nn

from flops_counter import (
    FLOPsCounter,
    compare_flops_efficiency,
    count_attention_flops,
    count_token_mixer_flops,
    format_flops,
)


def test_basic_flops_counter():
    """Test basic FLOPs counter functionality."""
    print("Testing basic FLOPs counter...")

    counter = FLOPsCounter()

    # Test initialization
    assert counter.flops == 0
    assert counter.params == 0
    assert counter.module_flops == {}
    assert counter.module_params == {}
    print("✓ Initialization test passed")

    # Test reset
    counter.flops = 1000
    counter.params = 500
    counter.reset()
    assert counter.flops == 0
    assert counter.params == 0
    print("✓ Reset test passed")


def test_linear_layer_flops():
    """Test linear layer FLOPs counting."""
    print("Testing linear layer FLOPs...")

    counter = FLOPsCounter()
    linear = nn.Linear(256, 512)
    x = torch.randn(2, 10, 256)
    y = linear(x)

    flops = counter._count_linear_flops(linear, x, y)

    # Should be non-zero
    assert flops > 0
    print(f"✓ Linear layer FLOPs: {flops}")
    print("✓ Linear layer test passed")


def test_layernorm_flops():
    """Test layer normalization FLOPs counting."""
    print("Testing layer normalization FLOPs...")

    counter = FLOPsCounter()
    layernorm = nn.LayerNorm(256)
    x = torch.randn(2, 10, 256)
    y = layernorm(x)

    flops = counter._count_layernorm_flops(layernorm, x, y)

    # Should be non-zero
    assert flops > 0
    print(f"✓ LayerNorm FLOPs: {flops}")
    print("✓ LayerNorm test passed")


def test_token_mixer_flops():
    """Test TokenMixer FLOPs counting."""
    print("Testing TokenMixer FLOPs...")

    try:
        result = count_token_mixer_flops(
            num_tokens=8, hidden_dim=256, num_heads=8, batch_size=2
        )

        assert result["flops"] > 0
        assert result["params"] > 0
        assert "theoretical_flops" in result
        assert "theoretical_breakdown" in result

        print(f"✓ TokenMixer FLOPs: {result['flops']}")
        print(f"✓ TokenMixer Parameters: {result['params']}")
        print(f"✓ Theoretical FLOPs: {result['theoretical_flops']}")
        print("✓ TokenMixer test passed")

    except ImportError as e:
        print(f"⚠ TokenMixer not available: {e}")


def test_attention_flops():
    """Test attention FLOPs counting."""
    print("Testing attention FLOPs...")

    try:
        result = count_attention_flops(
            embed_dim=256, num_heads=8, seq_len=10, batch_size=2
        )

        assert result["flops"] > 0
        assert result["params"] > 0
        assert "theoretical_flops" in result
        assert "theoretical_breakdown" in result

        print(f"✓ Attention FLOPs: {result['flops']}")
        print(f"✓ Attention Parameters: {result['params']}")
        print(f"✓ Theoretical FLOPs: {result['theoretical_flops']}")
        print("✓ Attention test passed")

    except ImportError as e:
        print(f"⚠ Attention module not available: {e}")


def test_format_flops():
    """Test FLOPs formatting."""
    print("Testing FLOPs formatting...")

    # Test different scales
    assert format_flops(1000) == "1.00 KFLOPS"
    assert format_flops(1000000) == "1.00 MFLOPS"
    assert format_flops(1000000000) == "1.00 GFLOPS"
    assert format_flops(1000000000000) == "1.00 TFLOPS"

    print("✓ Format FLOPs test passed")


def test_efficiency_comparison():
    """Test efficiency comparison."""
    print("Testing efficiency comparison...")

    result = compare_flops_efficiency(
        module1_flops=1000, module1_params=100, module2_flops=2000, module2_params=200
    )

    assert "flops_ratio" in result
    assert "params_ratio" in result
    assert "flops_per_param_ratio" in result

    print(f"✓ Efficiency comparison: {result}")
    print("✓ Efficiency comparison test passed")


def test_complexity_analysis():
    """Test complexity analysis between TokenMixer and Attention."""
    print("Testing complexity analysis...")

    try:
        # Test with small sizes for quick comparison
        # Note: TokenMixer requires num_tokens == num_heads
        token_mixer_result = count_token_mixer_flops(
            num_tokens=8, hidden_dim=256, num_heads=8, batch_size=1
        )

        attention_result = count_attention_flops(
            embed_dim=256, num_heads=8, seq_len=8, batch_size=1
        )

        # TokenMixer should have fewer FLOPs (linear vs quadratic)
        print(f"TokenMixer FLOPs: {format_flops(token_mixer_result['flops'])}")
        print(f"Attention FLOPs: {format_flops(attention_result['flops'])}")

        efficiency = compare_flops_efficiency(
            token_mixer_result["flops"],
            token_mixer_result["params"],
            attention_result["flops"],
            attention_result["params"],
        )

        print(f"FLOPs ratio (TokenMixer/Attention): {efficiency['flops_ratio']:.3f}")
        print(f"Parameters ratio: {efficiency['params_ratio']:.3f}")
        print(f"FLOPs per param ratio: {efficiency['flops_per_param_ratio']:.3f}")

        # TokenMixer should be more efficient (lower FLOPs ratio)
        assert efficiency["flops_ratio"] < 1.0
        print("✓ TokenMixer is more efficient than Attention")
        print("✓ Complexity analysis test passed")

    except ImportError as e:
        print(f"⚠ Cannot perform complexity analysis: {e}")


def main():
    """Run all tests."""
    print("🧪 Running FLOPs Counter Tests\n")

    test_basic_flops_counter()
    print()

    test_linear_layer_flops()
    print()

    test_layernorm_flops()
    print()

    test_token_mixer_flops()
    print()

    test_attention_flops()
    print()

    test_format_flops()
    print()

    test_efficiency_comparison()
    print()

    test_complexity_analysis()
    print()

    print("🎉 All tests completed successfully!")


if __name__ == "__main__":
    main()
