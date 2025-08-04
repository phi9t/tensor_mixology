"""
Tensor Mixology: Educational Examples

This module provides comprehensive educational examples to help engineers and
researchers understand the inner workings of the Token Mixer and Feed Forward
architectures. Each example is designed to build understanding progressively.

Learning Path:
1. Basic tensor operations and shapes
2. Understanding the Token Mixer step-by-step
3. Exploring Feed Forward Network variants
4. Comparing architectures and their trade-offs
5. Performance analysis and optimization

Run this file to see interactive demonstrations of each concept.
"""

import time

import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from feed_forward import FeedForward, SwiGLUFeedForward
from model_configs import RankMixerConfig
from rank_mixer_architecture import RankMixerModel
from token_mixer import TokenMixer

console = Console()


def educational_banner(title: str):
    """Print a nice banner for educational sections using rich."""
    banner = Panel(
        f"🎓 {title}",
        expand=True,
        style="bold magenta",
        box=box.DOUBLE,
        padding=(1, 2),
    )
    console.print(banner)


def example_1_tensor_shapes():
    """
    Example 1: Understanding Tensor Shapes in Sequence Models

    This example helps you understand the basic tensor shapes used throughout
    transformer-like architectures.
    """
    educational_banner("Example 1: Understanding Tensor Shapes")

    console.print(
        "[bold]In sequence models, we work with 3D tensors: (Batch, Sequence, Features)[/bold]"
    )
    console.print("Let's create some example tensors and explore their properties:\n")

    # Common configurations
    batch_size = 4  # Number of examples processed together
    seq_length = 8  # Number of tokens in each sequence
    hidden_dim = 12  # Dimension of each token's embedding

    config_table = Table(title="Configuration", box=box.SIMPLE)
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="bold yellow")
    config_table.add_row(
        "Batch size", f"{batch_size} (processing {batch_size} sequences simultaneously)"
    )
    config_table.add_row(
        "Sequence length", f"{seq_length} (each sequence has {seq_length} tokens)"
    )
    config_table.add_row(
        "Hidden dimension",
        f"{hidden_dim} (each token is represented by {hidden_dim} numbers)",
    )
    console.print(config_table)

    # Create example tensor
    x = torch.randn(batch_size, seq_length, hidden_dim)
    console.print(
        f"\n:bar_chart: [bold]Created tensor with shape:[/bold] [green]{tuple(x.shape)}[/green]"
    )
    console.print(f"   [dim]Total elements:[/dim] [yellow]{x.numel():,}[/yellow]")
    console.print(
        f"   [dim]Memory usage:[/dim] [yellow]{x.numel() * 4} bytes[/yellow] (assuming float32)"
    )

    # Show what each dimension represents
    console.print("\n:mag: [bold]Understanding each dimension:[/bold]")
    console.print(
        f"   [cyan]x[0][/cyan] = first sequence in batch, shape: [green]{tuple(x[0].shape)}[/green]"
    )
    console.print(
        f"   [cyan]x[0, 0][/cyan] = first token of first sequence, shape: [green]{tuple(x[0, 0].shape)}[/green]"
    )
    console.print(
        f"   [cyan]x[0, 0, 0][/cyan] = first feature of first token: [yellow]{x[0, 0, 0].item():.3f}[/yellow]"
    )

    # Show how sequence length affects memory
    console.print(
        "\n:chart_with_upwards_trend: [bold]Memory scaling with sequence length:[/bold]"
    )
    mem_table = Table(box=box.MINIMAL_DOUBLE_HEAD)
    mem_table.add_column("Seq length", justify="right", style="cyan")
    mem_table.add_column("Elements", justify="right", style="yellow")
    mem_table.add_column("Memory (MB)", justify="right", style="green")
    for seq_len in [128, 512, 1024, 2048]:
        elements = batch_size * seq_len * hidden_dim
        memory_mb = elements * 4 / (1024 * 1024)  # 4 bytes per float32
        mem_table.add_row(f"{seq_len}", f"{elements:,}", f"{memory_mb:5.1f}")
    console.print(mem_table)


def example_2_token_mixer_deep_dive():
    """
    Example 2: Token Mixer Deep Dive

    This example walks through the Token Mixer operation step-by-step,
    showing how tensor reshaping achieves global token mixing.
    """
    educational_banner("Example 2: Token Mixer Deep Dive")

    console.print(
        "[bold]The Token Mixer achieves O(T) complexity through clever tensor operations.[/bold]"
    )
    console.print("Let's trace through a simple example to understand how it works:\n")

    # Small example for clarity
    batch_size = 2
    num_tokens = 4
    hidden_dim = 8
    num_heads = 4

    config_table = Table(title="Configuration", box=box.SIMPLE)
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="bold yellow")
    config_table.add_row("Batch size", f"{batch_size}")
    config_table.add_row("Number of tokens", f"{num_tokens}")
    config_table.add_row("Hidden dimension", f"{hidden_dim}")
    config_table.add_row("Number of heads", f"{num_heads}")
    config_table.add_row("Head dimension", f"{hidden_dim // num_heads}")
    console.print(config_table)

    # Create Token Mixer
    mixer = TokenMixer(
        num_tokens=num_tokens, hidden_dim=hidden_dim, num_heads=num_heads
    )

    # Create simple input for tracing
    x = torch.arange(batch_size * num_tokens * hidden_dim, dtype=torch.float32)
    x = x.view(batch_size, num_tokens, hidden_dim)

    console.print(
        f"\n:inbox_tray: [bold]Input tensor shape:[/bold] [green]{tuple(x.shape)}[/green]"
    )
    input_table = Table(title="Input data (first batch)", box=box.MINIMAL)
    input_table.add_column("Token", style="cyan")
    input_table.add_column("Values", style="yellow")
    for i in range(num_tokens):
        input_table.add_row(f"{i}", str([float(f"{v:.1f}") for v in x[0, i].tolist()]))
    console.print(input_table)

    # Use the visualization method
    console.print("\n:arrows_counterclockwise: [bold]Token Mixing Process:[/bold]")
    steps = mixer.visualize_mixing_process(x)

    for step_name, step_info in steps.items():
        panel = Panel(
            f"{step_info['explanation']}\n[dim]Shape:[/dim] [green]{step_info['shape']}[/green]",
            title=f"[bold]{step_name}[/bold]",
            style="blue" if step_name != "step6_final" else "bold green",
            expand=False,
        )
        console.print(panel)
        if step_name == "step6_final":
            mixed_table = Table(title="Mixed tokens (first batch)", box=box.MINIMAL)
            mixed_table.add_column("Token", style="cyan")
            mixed_table.add_column("Values", style="yellow")
            for i in range(num_tokens):
                token_data = step_info["tensor"][0, i].tolist()
                mixed_table.add_row(f"{i}", str([f"{x:.3f}" for x in token_data]))
            console.print(mixed_table)


def example_3_feed_forward_variants():
    """
    Example 3: Comparing Feed Forward Network Variants

    This example compares different FFN architectures and their characteristics.
    """
    educational_banner("Example 3: Feed Forward Network Variants")

    console.print(
        "[bold]Feed Forward Networks come in different variants. Let's compare them:[/bold]\n"
    )

    # Configuration
    hidden_dim = 768
    ff_dim = 3072  # 4x expansion
    batch_size = 4
    seq_length = 8

    # Create different FFN variants
    ffn_standard = FeedForward(hidden_dim, ff_dim, activation="gelu")
    ffn_swiglu = SwiGLUFeedForward(hidden_dim, ff_dim)

    # Create input
    x = torch.randn(batch_size, seq_length, hidden_dim)

    config_table = Table(title="FFN Configuration", box=box.SIMPLE)
    config_table.add_column("Input shape", style="cyan")
    config_table.add_column("Structure", style="yellow")
    config_table.add_row(
        str(tuple(x.shape)), f"{hidden_dim} -> {ff_dim} -> {hidden_dim}"
    )
    console.print(config_table)

    # Compare parameter counts
    console.print("\n:bar_chart: [bold]Parameter Comparison:[/bold]")
    standard_params = sum(p.numel() for p in ffn_standard.parameters())
    swiglu_params = 3 * hidden_dim * ff_dim + 2 * hidden_dim  # LayerNorm

    param_table = Table(box=box.MINIMAL_DOUBLE_HEAD)
    param_table.add_column("Variant", style="cyan")
    param_table.add_column("Parameters", style="yellow", justify="right")
    param_table.add_column("Ratio", style="green", justify="right")
    param_table.add_row("Standard FFN", f"{standard_params:,}", "1.00x")
    param_table.add_row(
        "SwiGLU FFN", f"{swiglu_params:,}", f"{swiglu_params / standard_params:.2f}x"
    )
    console.print(param_table)

    # Test forward passes with timing
    console.print("\n:stopwatch: [bold]Performance Comparison:[/bold]")

    # Standard FFN
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = ffn_standard(x)
    standard_time = (time.time() - start_time) * 10  # Convert to ms

    # SwiGLU FFN
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = ffn_swiglu(x)
    swiglu_time = (time.time() - start_time) * 10  # Convert to ms

    perf_table = Table(box=box.MINIMAL_DOUBLE_HEAD)
    perf_table.add_column("Variant", style="cyan")
    perf_table.add_column("Time (ms)", style="yellow", justify="right")
    perf_table.add_column("Ratio", style="green", justify="right")
    perf_table.add_row("Standard FFN", f"{standard_time:.2f}", "1.00x")
    perf_table.add_row(
        "SwiGLU FFN", f"{swiglu_time:.2f}", f"{swiglu_time / standard_time:.2f}x"
    )
    console.print(perf_table)

    # Show verbose forward pass for educational purposes
    console.print("\n:mag: [bold]Detailed Standard FFN Forward Pass:[/bold]")
    _ = ffn_standard(x[:1, :2])  # Small subset for clarity


def example_4_architecture_comparison():
    """
    Example 4: Architecture Comparison - Token Mixer vs Self-Attention

    This example compares Token Mixer with traditional self-attention in terms
    of computational complexity and memory usage.
    """
    educational_banner("Example 4: Architecture Comparison")

    console.print(
        "[bold]Let's compare Token Mixer with self-attention across different sequence lengths:[/bold]\n"
    )

    hidden_dim = 768
    num_heads = 12
    batch_size = 8

    sequence_lengths = [64, 128, 256, 512, 1024]

    config_table = Table(title="Configuration", box=box.SIMPLE)
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="bold yellow")
    config_table.add_row("Hidden dimension", f"{hidden_dim}")
    config_table.add_row("Number of heads", f"{num_heads}")
    config_table.add_row("Batch size", f"{batch_size}")
    console.print(config_table)

    console.print("\n:bar_chart: [bold]Computational Complexity Comparison:[/bold]")
    comp_table = Table(box=box.MINIMAL_DOUBLE_HEAD)
    comp_table.add_column("Seq Len", justify="right", style="cyan")
    comp_table.add_column("TokenMixer FLOPs", justify="right", style="yellow")
    comp_table.add_column("Self-Att FLOPs", justify="right", style="magenta")
    comp_table.add_column("Speedup", justify="right", style="green")
    for seq_len in sequence_lengths:
        tm_flops = 8 * batch_size * seq_len * hidden_dim
        sa_flops = (
            batch_size * seq_len * seq_len * hidden_dim
            + 4 * batch_size * seq_len * hidden_dim * hidden_dim
        )
        speedup = sa_flops / tm_flops
        comp_table.add_row(
            f"{seq_len}", f"{tm_flops:,}", f"{sa_flops:,}", f"{speedup:6.1f}x"
        )
    console.print(comp_table)

    # Memory comparison
    console.print("\n:floppy_disk: [bold]Memory Usage Comparison:[/bold]")
    mem_table = Table(box=box.MINIMAL_DOUBLE_HEAD)
    mem_table.add_column("Seq Len", justify="right", style="cyan")
    mem_table.add_column("TokenMixer (MB)", justify="right", style="yellow")
    mem_table.add_column("Self-Att (MB)", justify="right", style="magenta")
    mem_table.add_column("Ratio", justify="right", style="green")
    for seq_len in sequence_lengths:
        tm_memory = (
            batch_size * seq_len * hidden_dim * 4 / (1024 * 1024)
        )  # 4 bytes per float
        sa_memory = batch_size * num_heads * seq_len * seq_len * 4 / (1024 * 1024)
        ratio = sa_memory / tm_memory
        mem_table.add_row(
            f"{seq_len}", f"{tm_memory:13.1f}", f"{sa_memory:13.1f}", f"{ratio:6.1f}x"
        )
    console.print(mem_table)


def example_5_complete_model_walkthrough():
    """
    Example 5: Complete Model Walkthrough

    This example shows how Token Mixer and Feed Forward components work together
    in a complete transformer-like model.
    """
    educational_banner("Example 5: Complete Model Walkthrough")

    console.print(
        "[bold]Let's see how Token Mixer and FFN work together in a complete model:[/bold]\n"
    )

    # Create a small model for educational purposes
    config = RankMixerConfig(
        hidden_dim=256,
        num_tokens=8,
        num_layers=2,
        num_heads=8,
        ff_dim=1024,
        dropout_rate=0.0,  # Disable dropout for clearer output
    )

    config_table = Table(title="Model Configuration", box=box.SIMPLE)
    config_table.add_column("Key", style="cyan")
    config_table.add_column("Value", style="yellow")
    for key, value in config.__dict__.items():
        config_table.add_row(str(key), str(value))
    console.print(config_table)

    model = RankMixerModel(
        num_tokens=config.num_tokens,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout_rate=config.dropout_rate,
    )

    # Create input (using embeddings instead of token IDs)
    batch_size = 2
    seq_length = 8
    input_embeddings = torch.randn(batch_size, seq_length, config.hidden_dim)

    input_panel = Panel(
        f"[bold]Shape:[/bold] [green]{tuple(input_embeddings.shape)}[/green]\n"
        f"[bold]Sample values:[/bold] [yellow]{input_embeddings[0, 0, :5].tolist()}[/yellow]",
        title=":inbox_tray: Input Embeddings",
        style="blue",
    )
    console.print(input_panel)

    # Forward pass with analysis
    console.print("\n:arrows_counterclockwise: [bold]Model Forward Pass:[/bold]")

    with torch.no_grad():
        outputs = model(input_embeddings)

    output_panel = Panel(
        f"[bold]Last hidden state shape:[/bold] [green]{tuple(outputs['last_hidden_state'].shape)}[/green]\n"
        f"[bold]Output keys:[/bold] [yellow]{list(outputs.keys())}[/yellow]",
        title=":outbox_tray: Output",
        style="green",
    )
    console.print(output_panel)

    # Analyze parameter distribution
    total_params = sum(p.numel() for p in model.parameters())
    mixer_params = sum(
        p.numel() for n, p in model.named_parameters() if "token_mixer" in n
    )
    ffn_params = sum(
        p.numel() for n, p in model.named_parameters() if "feed_forward" in n
    )
    other_params = total_params - mixer_params - ffn_params

    analysis_table = Table(title="Model Analysis", box=box.MINIMAL_DOUBLE_HEAD)
    analysis_table.add_column("Component", style="cyan")
    analysis_table.add_column("Parameters", style="yellow", justify="right")
    analysis_table.add_column("Percent", style="green", justify="right")
    analysis_table.add_row("Total", f"{total_params:,}", "100%")
    analysis_table.add_row(
        "Token Mixer", f"{mixer_params:,}", f"{100*mixer_params/total_params:.1f}%"
    )
    analysis_table.add_row(
        "Feed Forward", f"{ffn_params:,}", f"{100*ffn_params/total_params:.1f}%"
    )
    analysis_table.add_row(
        "Other (LN, etc.)", f"{other_params:,}", f"{100*other_params/total_params:.1f}%"
    )
    console.print(analysis_table)


def example_6_optimization_techniques():
    """
    Example 6: Optimization Techniques

    This example demonstrates various optimization techniques that can be applied
    to Token Mixer models.
    """
    educational_banner("Example 6: Optimization Techniques")

    console.print(
        "[bold]Let's explore optimization techniques for Token Mixer models:[/bold]\n"
    )

    # Create models with different optimizations
    hidden_dim = 768
    num_tokens = 128
    num_heads = 12
    batch_size = 4

    console.print(":hammer_and_wrench: [bold]Optimization Techniques:[/bold]")

    # 1. Mixed Precision
    console.print("\n[bold]1. Mixed Precision Training:[/bold]")
    mixer_fp32 = TokenMixer(num_tokens, hidden_dim, num_heads)
    x_fp32 = torch.randn(batch_size, num_tokens, hidden_dim)

    # Simulate fp16 (for educational purposes)
    mixer_fp16 = TokenMixer(num_tokens, hidden_dim, num_heads).half()
    x_fp16 = x_fp32.half()

    fp32_memory = sum(p.numel() * 4 for p in mixer_fp32.parameters()) / (
        1024 * 1024
    )  # MB
    fp16_memory = sum(p.numel() * 2 for p in mixer_fp16.parameters()) / (
        1024 * 1024
    )  # MB

    mem_table = Table(box=box.MINIMAL_DOUBLE_HEAD)
    mem_table.add_column("Model", style="cyan")
    mem_table.add_column("Memory (MB)", style="yellow", justify="right")
    mem_table.add_row("FP32", f"{fp32_memory:.2f}")
    mem_table.add_row("FP16", f"{fp16_memory:.2f}")
    mem_table.add_row("Savings", f"{100 * (1 - fp16_memory/fp32_memory):.1f}%")
    console.print(mem_table)

    # 2. Gradient Checkpointing Concept
    console.print("\n[bold]2. Gradient Checkpointing:[/bold]")
    checkpoint_panel = Panel(
        "- Trades computation for memory during backpropagation\n"
        "- Useful for training larger models or longer sequences\n"
        "- Can reduce memory usage by ~50% with ~20% slowdown",
        style="cyan",
    )
    console.print(checkpoint_panel)

    # 3. Sequence Length Scaling
    console.print("\n[bold]3. Sequence Length Scaling Analysis:[/bold]")
    scaling_table = Table(box=box.MINIMAL_DOUBLE_HEAD)
    scaling_table.add_column("Length", justify="right", style="cyan")
    scaling_table.add_column("Memory (MB)", justify="right", style="yellow")
    scaling_table.add_column("Relative", justify="right", style="green")
    base_memory = batch_size * 128 * hidden_dim * 4 / (1024 * 1024)
    for length in [128, 256, 512, 1024, 2048]:
        memory = batch_size * length * hidden_dim * 4 / (1024 * 1024)
        relative = memory / base_memory
        scaling_table.add_row(f"{length}", f"{memory:10.2f}", f"{relative:8.2f}x")
    console.print(scaling_table)

    insights_panel = Panel(
        "[bold]- Token Mixer scales linearly O(T) vs quadratic O(T²) for attention[/bold]\n"
        "[bold]- Memory usage is dominated by sequence length, not model complexity[/bold]\n"
        "[bold]- Mixed precision can halve memory usage with minimal accuracy loss[/bold]\n"
        "[bold]- Gradient checkpointing enables longer sequences on limited hardware[/bold]",
        title=":bulb: Key Optimization Insights",
        style="green",
    )
    console.print(insights_panel)


def run_all_examples():
    """Run all educational examples in sequence using rich."""
    console.print(
        Panel(
            "[bold magenta]Welcome to Tensor Mixology Educational Examples![/bold magenta]\n"
            "This interactive tutorial will help you understand the architecture step-by-step.",
            style="bold magenta",
            expand=True,
            box=box.DOUBLE,
        )
    )

    examples = [
        example_1_tensor_shapes,
        example_2_token_mixer_deep_dive,
        example_3_feed_forward_variants,
        example_4_architecture_comparison,
        example_5_complete_model_walkthrough,
        example_6_optimization_techniques,
    ]

    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
            console.print(
                f"\n[bold green]:white_check_mark: Example {i} completed successfully![/bold green]"
            )

            if i < len(examples):
                Prompt.ask(
                    "\n[bold]Press Enter to continue to the next example...[/bold]",
                    default="",
                    show_default=False,
                )

        except Exception as e:
            console.print(f"\n[bold red]:x: Error in Example {i}: {str(e)}[/bold red]")
            console.print("[yellow]Continuing to next example...[/yellow]")

    summary_panel = Panel(
        "[bold green]🎉 All educational examples completed![/bold green]\n"
        "You now have a comprehensive understanding of:\n"
        "  [green]✓[/green] Tensor shapes and operations in sequence models\n"
        "  [green]✓[/green] How Token Mixer achieves linear complexity\n"
        "  [green]✓[/green] Different Feed Forward Network architectures\n"
        "  [green]✓[/green] Performance comparisons with traditional attention\n"
        "  [green]✓[/green] Complete model architecture and parameter analysis\n"
        "  [green]✓[/green] Optimization techniques for production deployment",
        style="bold green",
        expand=True,
        box=box.DOUBLE,
    )
    console.print(summary_panel)


if __name__ == "__main__":
    # Check if running interactively
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_all_examples()
    else:
        console.print(
            Panel(
                "[bold]Educational Examples Available:[/bold]\n"
                "1. [cyan]run_all_examples()[/cyan] - Complete tutorial walkthrough\n"
                "2. Individual examples: [cyan]example_1_tensor_shapes()[/cyan], [cyan]example_2_token_mixer_deep_dive()[/cyan], etc.\n\n"
                "To run all examples: [yellow]python educational_examples.py --all[/yellow]\n"
                "To run interactively: [yellow]python -i educational_examples.py[/yellow]",
                style="magenta",
                expand=True,
                box=box.SIMPLE,
            )
        )

        # Run just the first example as a demo
        example_1_tensor_shapes()
