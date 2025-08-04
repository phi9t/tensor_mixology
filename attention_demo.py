"""Demo script for custom attention implementation with Rich formatting."""

import time

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from attention import MultiHeadAttention, SelfAttention

# Initialize Rich console
console = Console()


def demo_self_attention():
    """Demo self-attention functionality."""
    console.print(
        Panel(
            "[bold blue]Custom Attention Implementation Demo[/bold blue]\n"
            "Demonstrating self-attention functionality",
            title="Self Attention",
            border_style="blue",
        )
    )

    # Create attention layer
    embed_dim = 512
    num_heads = 8
    attn = SelfAttention(embed_dim, num_heads)

    console.print(f"[green]Created SelfAttention:[/green] {attn}")
    console.print(
        f"[yellow]Embed dim:[/yellow] {embed_dim}, [yellow]Heads:[/yellow] {num_heads}"
    )
    console.print()

    # Test with sample input
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, embed_dim)

    console.print(f"[yellow]Input tensor shape:[/yellow] {x.shape}")
    console.print(f"[yellow]Input tensor dtype:[/yellow] {x.dtype}")
    console.print(f"[yellow]Input tensor device:[/yellow] {x.device}")
    console.print()

    # Forward pass
    output, weights = attn(x)

    console.print(f"[green]Output tensor shape:[/green] {output.shape}")
    console.print(f"[green]Attention weights shape:[/green] {weights.shape}")
    console.print(f"[green]Shape invariance:[/green] {output.shape == x.shape}")
    console.print(f"[green]Output is finite:[/green] {torch.isfinite(output).all()}")
    console.print()


def demo_attention_weights():
    """Demo attention weights visualization."""
    console.print(
        Panel(
            "[bold blue]Attention Weights Demo[/bold blue]\n"
            "Visualizing attention weights and their properties",
            title="Attention Weights",
            border_style="blue",
        )
    )

    # Create attention layer
    embed_dim = 256
    num_heads = 4
    attn = SelfAttention(embed_dim, num_heads)

    # Test with sample input
    batch_size = 1
    seq_len = 5
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Forward pass
    output, weights = attn(x)

    # Average attention weights across heads
    avg_weights = weights.mean(dim=1)  # Average across heads

    console.print("[yellow]Attention weights (averaged across heads):[/yellow]")
    console.print(avg_weights[0])  # Shape: (seq_len, seq_len)
    console.print(f"[green]Attention weights shape:[/green] {weights.shape}")

    # Check that attention weights sum to 1
    weights_sum = weights.sum(dim=-1)
    ones = torch.ones_like(weights_sum)
    console.print(
        f"[green]Sum of attention weights per position:[/green] {weights_sum}"
    )
    console.print(
        f"[green]All weights sum to 1:[/green] {torch.allclose(weights_sum, ones)}"
    )
    console.print()


def demo_cross_attention():
    """Demo cross-attention functionality."""
    console.print(
        Panel(
            "[bold blue]Cross-Attention Demo[/bold blue]\n"
            "Demonstrating cross-attention between different sequences",
            title="Cross Attention",
            border_style="blue",
        )
    )

    # Create multi-head attention layer
    embed_dim = 512
    num_heads = 8
    attn = MultiHeadAttention(embed_dim, num_heads)

    # Test with different query, key, value
    batch_size = 2
    query_len = 8
    key_len = 12
    query = torch.randn(batch_size, query_len, embed_dim)
    key = torch.randn(batch_size, key_len, embed_dim)
    value = torch.randn(batch_size, key_len, embed_dim)

    console.print(f"[yellow]Query shape:[/yellow] {query.shape}")
    console.print(f"[yellow]Key shape:[/yellow] {key.shape}")
    console.print(f"[yellow]Value shape:[/yellow] {value.shape}")
    console.print()

    # Forward pass
    output, weights = attn(query, key, value)

    console.print(f"[green]Output shape:[/green] {output.shape}")
    console.print(f"[green]Attention weights shape:[/green] {weights.shape}")
    console.print(f"[green]Expected output shape:[/green] {query.shape}")
    console.print(f"[green]Shape matches:[/green] {output.shape == query.shape}")
    console.print()


def demo_attention_masks():
    """Demo attention with different types of masks."""
    console.print(
        Panel(
            "[bold blue]Attention with Masks Demo[/bold blue]\n"
            "Demonstrating padding masks, attention masks, and causal attention",
            title="Attention Masks",
            border_style="blue",
        )
    )

    # Create attention layer
    embed_dim = 256
    num_heads = 4
    attn = MultiHeadAttention(embed_dim, num_heads)

    # Test with sample input
    batch_size = 1
    seq_len = 6
    x = torch.randn(batch_size, seq_len, embed_dim)

    console.print("[bold cyan]1. Padding mask:[/bold cyan]")
    # Create padding mask (last 2 positions are padding)
    padding_mask = torch.tensor([[True, True, True, True, False, False]])
    output, weights = attn(x, x, x, key_padding_mask=padding_mask)

    console.print(f"[green]Output shape:[/green] {output.shape}")
    console.print(f"[green]Weights shape:[/green] {weights.shape}")
    console.print(
        f"[green]Padded positions ignored:[/green] {weights[0, :, 4:].sum() < 1e-6}"
    )
    console.print()

    console.print("[bold cyan]2. Attention mask:[/bold cyan]")
    # Create attention mask (prevent attending to future tokens)
    attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    output, weights = attn(x, x, x, attn_mask=attn_mask)

    console.print(f"[green]Output shape:[/green] {output.shape}")
    console.print(f"[green]Weights shape:[/green] {weights.shape}")
    console.print(
        f"[green]Causal mask applied:[/green] {weights[0, 0, 1:].sum() < 1e-6}"
    )
    console.print()

    console.print("[bold cyan]3. Causal attention:[/bold cyan]")
    # Test causal attention
    causal_attn = SelfAttention(embed_dim, num_heads)
    output, weights = causal_attn(x, is_causal=True)

    console.print(f"[green]Output shape:[/green] {output.shape}")
    console.print(f"[green]Weights shape:[/green] {weights.shape}")
    console.print(
        f"[green]Causal attention applied:[/green] {weights[0, 0, 1:].sum() < 1e-6}"
    )
    console.print()


def demo_performance_comparison():
    """Demo performance comparison with PyTorch implementation."""
    console.print(
        Panel(
            "[bold blue]Performance Comparison[/bold blue]\n"
            "Comparing our implementation with PyTorch's MultiheadAttention",
            title="Performance",
            border_style="blue",
        )
    )

    import torch.nn as nn

    # Test configurations
    configs = [
        (256, 4, "Small"),
        (512, 8, "Medium"),
        (768, 12, "Large"),
    ]

    performance_table = Table(
        title="Performance Comparison", show_header=True, header_style="bold magenta"
    )
    performance_table.add_column("Config", style="cyan", no_wrap=True)
    performance_table.add_column("Embed Dim", style="green", justify="right")
    performance_table.add_column("Heads", style="yellow", justify="right")
    performance_table.add_column("Our Time (ms)", style="blue", justify="right")
    performance_table.add_column("PyTorch Time (ms)", style="magenta", justify="right")
    performance_table.add_column("Speed Ratio", style="cyan", justify="right")

    for embed_dim, num_heads, description in configs:
        # Create our implementation
        our_attn = SelfAttention(embed_dim, num_heads)

        # Create PyTorch implementation
        pytorch_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Test input
        batch_size = 4
        seq_len = 32
        x = torch.randn(batch_size, seq_len, embed_dim)

        # Benchmark our implementation
        with torch.no_grad():
            for _ in range(10):  # Warmup
                _ = our_attn(x)

            start_time = time.time()
            for _ in range(100):
                _ = our_attn(x)
            our_time = (time.time() - start_time) / 100

        # Benchmark PyTorch implementation
        with torch.no_grad():
            for _ in range(10):  # Warmup
                _ = pytorch_attn(x, x, x)

            start_time = time.time()
            for _ in range(100):
                _ = pytorch_attn(x, x, x)
            pytorch_time = (time.time() - start_time) / 100

        performance_table.add_row(
            description,
            str(embed_dim),
            str(num_heads),
            f"{our_time*1000:.2f}",
            f"{pytorch_time*1000:.2f}",
            f"{pytorch_time/our_time:.2f}x",
        )

    console.print(performance_table)
    console.print()


def demo_parameter_comparison():
    """Demo parameter count comparison with PyTorch implementation."""
    console.print(
        Panel(
            "[bold blue]Parameter Count Comparison[/bold blue]\n"
            "Comparing parameter counts with PyTorch's MultiheadAttention",
            title="Parameters",
            border_style="blue",
        )
    )

    import torch.nn as nn

    # Test configurations
    configs = [
        (256, 4),
        (512, 8),
        (768, 12),
    ]

    param_table = Table(
        title="Parameter Count Comparison",
        show_header=True,
        header_style="bold magenta",
    )
    param_table.add_column("Embed Dim", style="green", justify="right")
    param_table.add_column("Heads", style="yellow", justify="right")
    param_table.add_column("Our Params", style="blue", justify="right")
    param_table.add_column("PyTorch Params", style="magenta", justify="right")
    param_table.add_column("Match", style="cyan", justify="center")
    param_table.add_column("Efficiency", style="green", justify="right")

    for embed_dim, num_heads in configs:
        # Create our implementation
        our_attn = SelfAttention(embed_dim, num_heads)

        # Create PyTorch implementation
        pytorch_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Count parameters
        our_params = sum(p.numel() for p in our_attn.parameters())
        pytorch_params = sum(p.numel() for p in pytorch_attn.parameters())

        param_table.add_row(
            str(embed_dim),
            str(num_heads),
            f"{our_params:,}",
            f"{pytorch_params:,}",
            "✅" if our_params == pytorch_params else "❌",
            f"{our_params/pytorch_params:.2f}x",
        )

    console.print(param_table)
    console.print()


def main():
    """Main demo function."""
    console.print(
        Panel(
            "[bold blue]Custom Attention Implementation Demo[/bold blue]\n"
            "Comprehensive demonstration of attention mechanisms",
            title="Welcome",
            border_style="blue",
        )
    )

    demo_self_attention()
    demo_attention_weights()
    demo_cross_attention()
    demo_attention_masks()
    demo_performance_comparison()
    demo_parameter_comparison()

    console.print(
        Panel(
            "[bold green]Demo Complete[/bold green]\n"
            "All attention functionality demonstrated successfully!",
            title="Success",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
