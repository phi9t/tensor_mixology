"""RankMixer model implementations with Rich formatting."""

import time

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from model_configs import (
    RANK_MIXER_1B,
    RANK_MIXER_100M,
    SE_RANK_MIXER_1B,
    SE_RANK_MIXER_100M,
    RankMixerConfig,
)
from rank_mixer_architecture import RankMixerModel

# Initialize Rich console
console = Console()


class RankMixer100M(RankMixerModel):
    """RankMixer 100M parameter model using TokenMixer."""

    def __init__(self, config: RankMixerConfig | None = None):
        config = config or RANK_MIXER_100M
        super().__init__(
            num_tokens=config.num_tokens,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout_rate=config.dropout_rate,
            ff_activation=config.ff_activation,
            use_swiglu=config.use_swiglu,
            prenorm=config.prenorm,
            use_position_embedding=config.use_position_embedding,
            max_position_embeddings=config.max_position_embeddings,
            layer_norm_eps=config.layer_norm_eps,
            initializer_range=config.initializer_range,
            use_se=config.use_se,
            se_reduction_ratio=config.se_reduction_ratio,
            se_activation=config.se_activation,
            se_gate_activation=config.se_gate_activation,
        )
        self.config = config


class RankMixer1B(RankMixerModel):
    """RankMixer 1B parameter model using TokenMixer."""

    def __init__(self, config: RankMixerConfig | None = None):
        config = config or RANK_MIXER_1B
        super().__init__(
            num_tokens=config.num_tokens,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout_rate=config.dropout_rate,
            ff_activation=config.ff_activation,
            use_swiglu=config.use_swiglu,
            prenorm=config.prenorm,
            use_position_embedding=config.use_position_embedding,
            max_position_embeddings=config.max_position_embeddings,
            layer_norm_eps=config.layer_norm_eps,
            initializer_range=config.initializer_range,
            use_se=config.use_se,
            se_reduction_ratio=config.se_reduction_ratio,
            se_activation=config.se_activation,
            se_gate_activation=config.se_gate_activation,
        )
        self.config = config


class SERankMixer100M(RankMixerModel):
    """SE-RankMixer 100M parameter model using Squeeze and Excitation."""

    def __init__(self, config: RankMixerConfig | None = None):
        config = config or SE_RANK_MIXER_100M
        super().__init__(
            num_tokens=config.num_tokens,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout_rate=config.dropout_rate,
            ff_activation=config.ff_activation,
            use_swiglu=config.use_swiglu,
            prenorm=config.prenorm,
            use_position_embedding=config.use_position_embedding,
            max_position_embeddings=config.max_position_embeddings,
            layer_norm_eps=config.layer_norm_eps,
            initializer_range=config.initializer_range,
            use_se=config.use_se,
            se_reduction_ratio=config.se_reduction_ratio,
            se_activation=config.se_activation,
            se_gate_activation=config.se_gate_activation,
        )
        self.config = config


class SERankMixer1B(RankMixerModel):
    """SE-RankMixer 1B parameter model using Squeeze and Excitation."""

    def __init__(self, config: RankMixerConfig | None = None):
        config = config or SE_RANK_MIXER_1B
        super().__init__(
            num_tokens=config.num_tokens,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout_rate=config.dropout_rate,
            ff_activation=config.ff_activation,
            use_swiglu=config.use_swiglu,
            prenorm=config.prenorm,
            use_position_embedding=config.use_position_embedding,
            max_position_embeddings=config.max_position_embeddings,
            layer_norm_eps=config.layer_norm_eps,
            initializer_range=config.initializer_range,
            use_se=config.use_se,
            se_reduction_ratio=config.se_reduction_ratio,
            se_activation=config.se_activation,
            se_gate_activation=config.se_gate_activation,
        )
        self.config = config


class RankMixerFactory:
    """Factory for creating RankMixer models."""

    _models = {
        "100M": RankMixer100M,
        "1B": RankMixer1B,
        "SE-100M": SERankMixer100M,
        "SE-1B": SERankMixer1B,
    }

    _configs = {
        "100M": RANK_MIXER_100M,
        "1B": RANK_MIXER_1B,
        "SE-100M": SE_RANK_MIXER_100M,
        "SE-1B": SE_RANK_MIXER_1B,
    }

    @classmethod
    def create_model(cls, model_type: str) -> RankMixerModel:
        """Create a RankMixer model by type."""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._models[model_type]()

    @classmethod
    def get_config(cls, model_type: str) -> RankMixerConfig:
        """Get configuration for a model type."""
        if model_type not in cls._configs:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._configs[model_type]

    @classmethod
    def list_models(cls) -> list[str]:
        """List available model types."""
        return list(cls._models.keys())


def demo_models():
    """Demo the RankMixer models."""
    console.print(
        Panel(
            "[bold blue]RankMixer Model Demo[/bold blue]\n"
            "Demonstrating TokenMixer and SE-based models",
            title="Model Demo",
            border_style="blue",
        )
    )

    console.print("\n[bold cyan]1. Creating TokenMixer-based models...[/bold cyan]")

    model_100m = RankMixer100M()
    config_100m = model_100m.config

    console.print(
        f"   [green]Configuration:[/green] D={config_100m.hidden_dim}, T={config_100m.num_tokens}, L={config_100m.num_layers}"
    )
    console.print("   [green]Model Type:[/green] TokenMixer-based")
    console.print(
        f"   [green]Total Parameters:[/green] {config_100m.total_parameters:,}"
    )
    console.print(f"   [green]Model Size:[/green] {config_100m.model_size_mb:.2f} MB")
    console.print()

    model_1b = RankMixer1B()
    config_1b = model_1b.config

    console.print(
        f"   [green]Configuration:[/green] D={config_1b.hidden_dim}, T={config_1b.num_tokens}, L={config_1b.num_layers}"
    )
    console.print("   [green]Model Type:[/green] TokenMixer-based")
    console.print(f"   [green]Total Parameters:[/green] {config_1b.total_parameters:,}")
    console.print(f"   [green]Model Size:[/green] {config_1b.model_size_mb:.2f} MB")
    console.print()

    console.print("\n[bold cyan]2. Creating SE-based models...[/bold cyan]")

    se_model_100m = SERankMixer100M()
    se_config_100m = se_model_100m.config

    console.print(
        f"   [green]Configuration:[/green] D={se_config_100m.hidden_dim}, T={se_config_100m.num_tokens}, L={se_config_100m.num_layers}"
    )
    console.print("   [green]Model Type:[/green] SE-based")
    console.print(
        f"   [green]SE Reduction Ratio:[/green] {se_config_100m.se_reduction_ratio}"
    )
    console.print(
        f"   [green]Total Parameters:[/green] {se_config_100m.total_parameters:,}"
    )
    console.print(
        f"   [green]Model Size:[/green] {se_config_100m.model_size_mb:.2f} MB"
    )
    console.print()

    se_model_1b = SERankMixer1B()
    se_config_1b = se_model_1b.config

    console.print(
        f"   [green]Configuration:[/green] D={se_config_1b.hidden_dim}, T={se_config_1b.num_tokens}, L={se_config_1b.num_layers}"
    )
    console.print("   [green]Model Type:[/green] SE-based")
    console.print(
        f"   [green]SE Reduction Ratio:[/green] {se_config_1b.se_reduction_ratio}"
    )
    console.print(
        f"   [green]Total Parameters:[/green] {se_config_1b.total_parameters:,}"
    )
    console.print(f"   [green]Model Size:[/green] {se_config_1b.model_size_mb:.2f} MB")
    console.print()

    console.print("[bold cyan]3. Testing forward pass...[/bold cyan]")

    # Test with sample input
    input_embeddings_100m = torch.randn(
        1, config_100m.num_tokens, config_100m.hidden_dim
    )
    input_embeddings_1b = torch.randn(1, config_1b.num_tokens, config_1b.hidden_dim)
    input_embeddings_se_100m = torch.randn(
        1, se_config_100m.num_tokens, se_config_100m.hidden_dim
    )
    input_embeddings_se_1b = torch.randn(
        1, se_config_1b.num_tokens, se_config_1b.hidden_dim
    )

    with torch.no_grad():
        output_100m = model_100m(input_embeddings_100m)
        output_1b = model_1b(input_embeddings_1b)
        output_se_100m = se_model_100m(input_embeddings_se_100m)
        output_se_1b = se_model_1b(input_embeddings_se_1b)

    console.print(
        f"   [yellow]TokenMixer 100M Input Shape:[/yellow] {input_embeddings_100m.shape}"
    )
    console.print(
        f"   [green]TokenMixer 100M Output Shape:[/green] {output_100m['last_hidden_state'].shape}"
    )
    console.print()
    console.print(
        f"   [yellow]TokenMixer 1B Input Shape:[/yellow] {input_embeddings_1b.shape}"
    )
    console.print(
        f"   [green]TokenMixer 1B Output Shape:[/green] {output_1b['last_hidden_state'].shape}"
    )
    console.print()
    console.print(
        f"   [yellow]SE 100M Input Shape:[/yellow] {input_embeddings_se_100m.shape}"
    )
    console.print(
        f"   [green]SE 100M Output Shape:[/green] {output_se_100m['last_hidden_state'].shape}"
    )
    console.print()
    console.print(
        f"   [yellow]SE 1B Input Shape:[/yellow] {input_embeddings_se_1b.shape}"
    )
    console.print(
        f"   [green]SE 1B Output Shape:[/green] {output_se_1b['last_hidden_state'].shape}"
    )
    console.print()

    console.print("[bold cyan]4. Testing factory...[/bold cyan]")

    factory_model_100m = RankMixerFactory.create_model("100M")
    factory_model_1b = RankMixerFactory.create_model("1B")
    factory_se_model_100m = RankMixerFactory.create_model("SE-100M")
    factory_se_model_1b = RankMixerFactory.create_model("SE-1B")

    console.print(
        f"   [green]Factory TokenMixer 100M Parameters:[/green] {factory_model_100m.config.total_parameters:,}"
    )
    console.print(
        f"   [green]Factory TokenMixer 1B Parameters:[/green] {factory_model_1b.config.total_parameters:,}"
    )
    console.print(
        f"   [green]Factory SE 100M Parameters:[/green] {factory_se_model_100m.config.total_parameters:,}"
    )
    console.print(
        f"   [green]Factory SE 1B Parameters:[/green] {factory_se_model_1b.config.total_parameters:,}"
    )
    console.print()

    console.print(
        Panel(
            "✅ All models created and tested successfully!",
            title="Success",
            border_style="green",
        )
    )


def benchmark_models():
    """Benchmark the RankMixer models."""
    console.print(
        Panel(
            "[bold blue]RankMixer Model Benchmark[/bold blue]\n"
            "Performance benchmarking of TokenMixer and SE-based models",
            title="Benchmark",
            border_style="blue",
        )
    )

    # Create models
    model_100m = RankMixer100M()
    model_1b = RankMixer1B()
    se_model_100m = SERankMixer100M()
    se_model_1b = SERankMixer1B()

    # Benchmark configurations
    batch_sizes = [1, 2, 4, 8]

    benchmark_table = Table(
        title="Model Performance Benchmark",
        show_header=True,
        header_style="bold magenta",
    )
    benchmark_table.add_column("Batch Size", style="cyan", justify="right")
    benchmark_table.add_column("Model", style="green", no_wrap=True)
    benchmark_table.add_column("Avg Time (ms)", style="yellow", justify="right")
    benchmark_table.add_column("Throughput (tokens/sec)", style="blue", justify="right")

    for batch_size in batch_sizes:
        console.print(f"\n[bold cyan]Batch Size: {batch_size}[/bold cyan]")
        console.print("─" * 30)

        # Test TokenMixer 100M model
        input_100m = torch.randn(
            batch_size, model_100m.config.num_tokens, model_100m.config.hidden_dim
        )

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model_100m(input_100m)

        # Benchmark
        times_100m = []
        with torch.no_grad():
            for _ in range(20):
                start_time = time.time()
                _ = model_100m(input_100m)
                end_time = time.time()
                times_100m.append(end_time - start_time)

        avg_time_100m = sum(times_100m) / len(times_100m)
        tokens_per_sec_100m = (
            batch_size * model_100m.config.num_tokens
        ) / avg_time_100m

        benchmark_table.add_row(
            str(batch_size),
            "TokenMixer 100M",
            f"{avg_time_100m*1000:.2f}",
            f"{tokens_per_sec_100m:.0f}",
        )

        # Test TokenMixer 1B model
        input_1b = torch.randn(
            batch_size, model_1b.config.num_tokens, model_1b.config.hidden_dim
        )

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model_1b(input_1b)

        # Benchmark
        times_1b = []
        with torch.no_grad():
            for _ in range(20):
                start_time = time.time()
                _ = model_1b(input_1b)
                end_time = time.time()
                times_1b.append(end_time - start_time)

        avg_time_1b = sum(times_1b) / len(times_1b)
        tokens_per_sec_1b = (batch_size * model_1b.config.num_tokens) / avg_time_1b

        benchmark_table.add_row(
            str(batch_size),
            "TokenMixer 1B",
            f"{avg_time_1b*1000:.2f}",
            f"{tokens_per_sec_1b:.0f}",
        )

        # Test SE 100M model
        input_se_100m = torch.randn(
            batch_size, se_model_100m.config.num_tokens, se_model_100m.config.hidden_dim
        )

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = se_model_100m(input_se_100m)

        # Benchmark
        times_se_100m = []
        with torch.no_grad():
            for _ in range(20):
                start_time = time.time()
                _ = se_model_100m(input_se_100m)
                end_time = time.time()
                times_se_100m.append(end_time - start_time)

        avg_time_se_100m = sum(times_se_100m) / len(times_se_100m)
        tokens_per_sec_se_100m = (
            batch_size * se_model_100m.config.num_tokens
        ) / avg_time_se_100m

        benchmark_table.add_row(
            str(batch_size),
            "SE 100M",
            f"{avg_time_se_100m*1000:.2f}",
            f"{tokens_per_sec_se_100m:.0f}",
        )

        # Test SE 1B model
        input_se_1b = torch.randn(
            batch_size, se_model_1b.config.num_tokens, se_model_1b.config.hidden_dim
        )

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = se_model_1b(input_se_1b)

        # Benchmark
        times_se_1b = []
        with torch.no_grad():
            for _ in range(20):
                start_time = time.time()
                _ = se_model_1b(input_se_1b)
                end_time = time.time()
                times_se_1b.append(end_time - start_time)

        avg_time_se_1b = sum(times_se_1b) / len(times_se_1b)
        tokens_per_sec_se_1b = (
            batch_size * se_model_1b.config.num_tokens
        ) / avg_time_se_1b

        benchmark_table.add_row(
            str(batch_size),
            "SE 1B",
            f"{avg_time_se_1b*1000:.2f}",
            f"{tokens_per_sec_se_1b:.0f}",
        )

    console.print(benchmark_table)

    # Summary
    summary_text = Text()
    summary_text.append("Performance Summary:\n", style="bold green")
    summary_text.append(
        "  • TokenMixer models use linear complexity O(T)\n", style="green"
    )
    summary_text.append(
        "  • SE models use channel-wise attention mechanism\n", style="green"
    )
    summary_text.append("  • Both model types scale with batch size\n", style="green")
    summary_text.append(
        "  • SE models may have different parameter efficiency\n", style="green"
    )

    console.print(Panel(summary_text, title="Summary", border_style="green"))
    console.print("\n" + "─" * 50)


if __name__ == "__main__":
    demo_models()
    benchmark_models()
