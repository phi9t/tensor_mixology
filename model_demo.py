"""Demo script for RankMixer models with Rich formatting."""

import time

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from model_configs import RankMixerConfig
from rank_mixer_models import RankMixer1B, RankMixer100M, RankMixerFactory

# Initialize Rich console
console = Console()


def demo_basic_usage():
    """Demo basic model usage."""
    console.print(
        Panel(
            "[bold blue]RankMixer Model Demo[/bold blue]\n"
            "Basic usage and testing of RankMixer models",
            title="Basic Usage",
            border_style="blue",
        )
    )

    console.print("\n[bold cyan]1. Creating models...[/bold cyan]")

    model_100m = RankMixer100M()
    model_1b = RankMixer1B()

    console.print(
        f"   [green]100M Model:[/green] {model_100m.config.total_parameters:,} parameters"
    )
    console.print(
        f"   [green]1B Model:[/green] {model_1b.config.total_parameters:,} parameters"
    )
    console.print()

    console.print("[bold cyan]2. Testing with sample input...[/bold cyan]")

    # Sample input embeddings
    sample_embeddings_100m = torch.randn(1, 12, 768)  # 1 batch, 12 tokens, 768 dim
    sample_embeddings_1b = torch.randn(1, 24, 1536)  # 1 batch, 24 tokens, 1536 dim

    console.print(f"   [yellow]Input 100M:[/yellow] {sample_embeddings_100m.shape}")
    console.print(f"   [yellow]Input 1B:[/yellow] {sample_embeddings_1b.shape}")

    # Forward pass
    with torch.no_grad():
        output_100m = model_100m(sample_embeddings_100m)
        output_1b = model_1b(sample_embeddings_1b)

    console.print(
        f"   [green]Output 100M:[/green] {output_100m['last_hidden_state'].shape}"
    )
    console.print(
        f"   [green]Output 1B:[/green] {output_1b['last_hidden_state'].shape}"
    )
    console.print()

    console.print(
        Panel("✅ Basic usage demo completed!", title="Success", border_style="green")
    )


def demo_factory_usage():
    """Demo factory usage."""
    console.print(
        Panel(
            "[bold blue]Factory Usage Demo[/bold blue]\n"
            "Using RankMixerFactory to create models",
            title="Factory Usage",
            border_style="blue",
        )
    )

    console.print("\n[bold cyan]1. Creating models using factory...[/bold cyan]")

    factory_model_100m = RankMixerFactory.create_model("100M")
    factory_model_1b = RankMixerFactory.create_model("1B")

    console.print(
        f"   [green]Factory 100M:[/green] {factory_model_100m.config.total_parameters:,} parameters"
    )
    console.print(
        f"   [green]Factory 1B:[/green] {factory_model_1b.config.total_parameters:,} parameters"
    )
    console.print()

    console.print("[bold cyan]2. Getting configurations...[/bold cyan]")

    config_100m = RankMixerFactory.get_config("100M")
    config_1b = RankMixerFactory.get_config("1B")

    console.print(f"   [yellow]100M Config:[/yellow] {config_100m}")
    console.print(f"   [yellow]1B Config:[/yellow] {config_1b}")
    console.print()

    console.print(
        Panel("✅ Factory usage demo completed!", title="Success", border_style="green")
    )


def demo_custom_config():
    """Demo custom configuration."""
    console.print(
        Panel(
            "[bold blue]Custom Configuration Demo[/bold blue]\n"
            "Creating models with custom configurations",
            title="Custom Config",
            border_style="blue",
        )
    )

    console.print("\n[bold cyan]1. Custom configuration created...[/bold cyan]")

    custom_config = RankMixerConfig(
        hidden_dim=1024,  # D = 1024
        num_tokens=16,  # T = 16
        num_layers=3,  # L = 3
        num_heads=16,
    )

    console.print(f"   [yellow]Hidden Dim:[/yellow] {custom_config.hidden_dim}")
    console.print(f"   [yellow]Num Tokens:[/yellow] {custom_config.num_tokens}")
    console.print(f"   [yellow]Num Layers:[/yellow] {custom_config.num_layers}")
    console.print(f"   [yellow]Num Heads:[/yellow] {custom_config.num_heads}")
    console.print(
        f"   [green]Total Parameters:[/green] {custom_config.total_parameters:,}"
    )
    console.print(f"   [green]Model Size:[/green] {custom_config.model_size_mb:.2f} MB")
    console.print()

    console.print("[bold cyan]2. Creating model with custom config...[/bold cyan]")

    custom_model = RankMixer100M(custom_config)

    console.print(
        f"   [green]Model Parameters:[/green] {custom_model.config.total_parameters:,}"
    )
    console.print(
        f"   [green]Model Size:[/green] {custom_model.config.model_size_mb:.2f} MB"
    )
    console.print()

    console.print("[bold cyan]3. Testing custom model...[/bold cyan]")

    # Test with custom input
    sample_input = torch.randn(1, custom_config.num_tokens, custom_config.hidden_dim)

    with torch.no_grad():
        output = custom_model(sample_input)

    console.print(f"   [yellow]Input Shape:[/yellow] {sample_input.shape}")
    console.print(
        f"   [green]Output Shape:[/green] {output['last_hidden_state'].shape}"
    )
    console.print()

    console.print(
        Panel(
            "✅ Custom configuration demo completed!",
            title="Success",
            border_style="green",
        )
    )


def demo_model_comparison():
    """Demo model comparison."""
    console.print(
        Panel(
            "[bold blue]Model Comparison Demo[/bold blue]\n"
            "Comparing different model configurations and performance",
            title="Model Comparison",
            border_style="blue",
        )
    )

    console.print("\n[bold cyan]Model Configurations:[/bold cyan]")
    console.print("─" * 60)

    # Create all model variants
    models = {
        "100M (D=768, T=12, L=2)": RankMixer100M(),
        "1B (D=1536, T=24, L=2)": RankMixer1B(),
    }

    config_table = Table(
        title="Model Configurations", show_header=True, header_style="bold magenta"
    )
    config_table.add_column("Model", style="cyan", no_wrap=True)
    config_table.add_column("Parameters", style="green", justify="right")
    config_table.add_column("Size (MB)", style="yellow", justify="right")
    config_table.add_column("Hidden Dim", style="blue", justify="right")
    config_table.add_column("Tokens", style="magenta", justify="right")
    config_table.add_column("Layers", style="cyan", justify="right")
    config_table.add_column("Heads", style="green", justify="right")

    for name, model in models.items():
        config = model.config
        config_table.add_row(
            name,
            f"{config.total_parameters:,}",
            f"{config.model_size_mb:.2f}",
            str(config.hidden_dim),
            str(config.num_tokens),
            str(config.num_layers),
            str(config.num_heads),
        )

    console.print(config_table)

    console.print("\n[bold cyan]Performance Comparison (batch_size=1):[/bold cyan]")
    console.print("─" * 60)

    performance_table = Table(
        title="Performance Comparison", show_header=True, header_style="bold magenta"
    )
    performance_table.add_column("Model", style="cyan", no_wrap=True)
    performance_table.add_column("Avg Time (ms)", style="green", justify="right")
    performance_table.add_column(
        "Throughput (tokens/sec)", style="yellow", justify="right"
    )

    for name, model in models.items():
        # Generate appropriate input
        if "100M" in name:
            input_tensor = torch.randn(1, 12, 768)
        else:
            input_tensor = torch.randn(1, 24, 1536)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(10):
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        tokens_per_sec = input_tensor.shape[1] / avg_time

        performance_table.add_row(name, f"{avg_time*1000:.2f}", f"{tokens_per_sec:.0f}")

    console.print(performance_table)

    console.print("\n" + "─" * 50)
    console.print(
        Panel(
            "✅ Model comparison demo completed!", title="Success", border_style="green"
        )
    )


def main():
    """Main demo function."""
    console.print(
        Panel(
            "[bold blue]RankMixer Model Demo[/bold blue]\n"
            "Comprehensive demonstration of RankMixer models",
            title="Welcome",
            border_style="blue",
        )
    )

    demo_basic_usage()
    console.print("\n" + "─" * 50)
    demo_factory_usage()
    console.print("\n" + "─" * 50)
    demo_custom_config()
    console.print("\n" + "─" * 50)
    demo_model_comparison()

    # Usage examples
    usage_text = Text()
    usage_text.append("🎉 All demos completed successfully!\n\n", style="bold green")
    usage_text.append("Usage Examples:\n", style="bold blue")
    usage_text.append("  # Create 100M model\n", style="cyan")
    usage_text.append("  model = RankMixer100M()\n\n", style="white")
    usage_text.append("  # Create 1B model\n", style="cyan")
    usage_text.append("  model = RankMixer1B()\n\n", style="white")
    usage_text.append("  # Use factory\n", style="cyan")
    usage_text.append(
        "  model = RankMixerFactory.create_model('100M')\n\n", style="white"
    )
    usage_text.append("  # Custom configuration\n", style="cyan")
    usage_text.append(
        "  config = RankMixerConfig(hidden_dim=1024, num_tokens=24, num_layers=3)\n",
        style="white",
    )
    usage_text.append("  model = RankMixer100M(config)", style="white")

    console.print(Panel(usage_text, title="Usage Examples", border_style="green"))


if __name__ == "__main__":
    main()
