"""Model configurations for RankMixer models."""

from dataclasses import dataclass
from typing import Any


@dataclass
class RankMixerConfig:
    """Configuration for RankMixer models."""

    # Model dimensions
    hidden_dim: int  # D - token dimension
    num_tokens: int  # T - fixed number of tokens
    num_layers: int  # L - number of RankMixer layers

    # Attention settings
    num_heads: int = 12
    ff_dim: int | None = None  # Will be set to 4 * hidden_dim if None

    # Training settings
    dropout_rate: float = 0.1
    ff_activation: str = "gelu"
    use_swiglu: bool = False
    prenorm: bool = True

    # Embedding settings
    use_position_embedding: bool = True
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02

    # SE settings (for SE-based models)
    use_se: bool = False
    se_reduction_ratio: int = 16
    se_activation: str = "relu"
    se_gate_activation: str = "sigmoid"

    def __post_init__(self):
        """Set default values after initialization."""
        if self.ff_dim is None:
            self.ff_dim = 4 * self.hidden_dim

    @property
    def total_parameters(self) -> int:
        """Calculate total number of parameters."""
        # Position embedding (if used)
        embedding_params = 0
        if self.use_position_embedding:
            embedding_params += self.max_position_embeddings * self.hidden_dim

        # RankMixer layers
        layer_params = self._calculate_layer_parameters()
        total_layers_params = layer_params * self.num_layers

        return embedding_params + total_layers_params

    def _calculate_layer_parameters(self) -> int:
        """Calculate parameters per RankMixer layer."""
        if self.use_se:
            # SE-based RankMixer parameters
            # SE module parameters
            bottleneck_channels = max(1, self.hidden_dim // self.se_reduction_ratio)
            se_params = (
                self.hidden_dim * bottleneck_channels  # fc1
                + bottleneck_channels * self.hidden_dim  # fc2
            )
        else:
            # TokenMixer parameters (only layer norm)
            se_params = self.hidden_dim

        # Feed forward parameters
        ff_params = (
            self.hidden_dim * self.ff_dim
            + self.ff_dim  # First linear
            + self.ff_dim * self.hidden_dim
            + self.hidden_dim
        )  # Second linear

        return se_params + ff_params

    @property
    def model_size_mb(self) -> float:
        """Calculate model size in MB."""
        return self.total_parameters * 4 / (1024 * 1024)  # 4 bytes per parameter

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "hidden_dim": self.hidden_dim,
            "num_tokens": self.num_tokens,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
            "ff_activation": self.ff_activation,
            "use_swiglu": self.use_swiglu,
            "prenorm": self.prenorm,
            "use_position_embedding": self.use_position_embedding,
            "max_position_embeddings": self.max_position_embeddings,
            "layer_norm_eps": self.layer_norm_eps,
            "initializer_range": self.initializer_range,
            "use_se": self.use_se,
            "se_reduction_ratio": self.se_reduction_ratio,
            "se_activation": self.se_activation,
            "se_gate_activation": self.se_gate_activation,
            "total_parameters": self.total_parameters,
            "model_size_mb": self.model_size_mb,
        }


# Predefined configurations for TokenMixer-based models
RANK_MIXER_100M = RankMixerConfig(
    hidden_dim=768,  # D = 768
    num_tokens=12,  # T = 12
    num_layers=2,  # L = 2
    num_heads=12,
)

RANK_MIXER_1B = RankMixerConfig(
    hidden_dim=1536,  # D = 1536
    num_tokens=24,  # T = 24
    num_layers=2,  # L = 2
    num_heads=24,
)

# Adjusted configurations to get closer to target parameter counts
RANK_MIXER_100M_ADJUSTED = RankMixerConfig(
    hidden_dim=768,  # D = 768
    num_tokens=16,  # T = 16
    num_layers=3,  # L = 3 (increased layers)
    num_heads=12,
)

RANK_MIXER_1B_ADJUSTED = RankMixerConfig(
    hidden_dim=1536,  # D = 1536
    num_tokens=32,  # T = 32
    num_layers=3,  # L = 3 (increased layers)
    num_heads=24,
)

# Additional configurations for comparison
RANK_MIXER_100M_VARIANT = RankMixerConfig(
    hidden_dim=768,
    num_tokens=16,
    num_layers=4,  # More layers, same total params
    num_heads=12,
)

RANK_MIXER_1B_VARIANT = RankMixerConfig(
    hidden_dim=1536,
    num_tokens=32,
    num_layers=4,  # More layers, same total params
    num_heads=24,
)

# SE-based RankMixer configurations
SE_RANK_MIXER_100M = RankMixerConfig(
    hidden_dim=768,  # D = 768
    num_tokens=12,  # T = 12
    num_layers=2,  # L = 2
    num_heads=12,  # Not used in SE but kept for compatibility
    use_se=True,
    se_reduction_ratio=16,
)

SE_RANK_MIXER_1B = RankMixerConfig(
    hidden_dim=1536,  # D = 1536
    num_tokens=24,  # T = 24
    num_layers=2,  # L = 2
    num_heads=24,  # Not used in SE but kept for compatibility
    use_se=True,
    se_reduction_ratio=16,
)

# SE-based configurations with different reduction ratios
SE_RANK_MIXER_100M_REDUCTION_8 = RankMixerConfig(
    hidden_dim=768,
    num_tokens=12,
    num_layers=2,
    num_heads=12,
    use_se=True,
    se_reduction_ratio=8,  # More parameters, less reduction
)

SE_RANK_MIXER_1B_REDUCTION_8 = RankMixerConfig(
    hidden_dim=1536,
    num_tokens=24,
    num_layers=2,
    num_heads=24,
    use_se=True,
    se_reduction_ratio=8,  # More parameters, less reduction
)

SE_RANK_MIXER_100M_REDUCTION_32 = RankMixerConfig(
    hidden_dim=768,
    num_tokens=12,
    num_layers=2,
    num_heads=12,
    use_se=True,
    se_reduction_ratio=32,  # Fewer parameters, more reduction
)

SE_RANK_MIXER_1B_REDUCTION_32 = RankMixerConfig(
    hidden_dim=1536,
    num_tokens=24,
    num_layers=2,
    num_heads=24,
    use_se=True,
    se_reduction_ratio=32,  # Fewer parameters, more reduction
)


def print_model_info(config: RankMixerConfig, name: str):
    """Print detailed model information."""
    model_type = "SE-RankMixer" if config.use_se else "RankMixer"
    print(f"\n=== {name} ({model_type}) ===")
    print(
        f"Configuration: D={config.hidden_dim}, T={config.num_tokens}, L={config.num_layers}"
    )
    if config.use_se:
        print(f"SE Reduction Ratio: {config.se_reduction_ratio}")
    else:
        print(f"Number of Heads: {config.num_heads}")
    print(f"Total Parameters: {config.total_parameters:,}")
    print(f"Model Size: {config.model_size_mb:.2f} MB")
    print(f"Parameters per layer: {config._calculate_layer_parameters():,}")
    print(f"Hidden dimension: {config.hidden_dim}")
    print(f"Number of tokens: {config.num_tokens}")
    print(f"Number of layers: {config.num_layers}")
    print(f"Feed forward dimension: {config.ff_dim}")


def main():
    """Print information about all model configurations."""
    print("RankMixer Model Configurations")
    print("=" * 50)

    # TokenMixer-based models
    print_model_info(RANK_MIXER_100M, "RankMixer 100M (Original)")
    print_model_info(RANK_MIXER_1B, "RankMixer 1B (Original)")
    print_model_info(RANK_MIXER_100M_ADJUSTED, "RankMixer 100M (Adjusted)")
    print_model_info(RANK_MIXER_1B_ADJUSTED, "RankMixer 1B (Adjusted)")
    print_model_info(RANK_MIXER_100M_VARIANT, "RankMixer 100M (4 layers)")
    print_model_info(RANK_MIXER_1B_VARIANT, "RankMixer 1B (4 layers)")

    # SE-based models
    print_model_info(SE_RANK_MIXER_100M, "SE-RankMixer 100M (Reduction 16)")
    print_model_info(SE_RANK_MIXER_1B, "SE-RankMixer 1B (Reduction 16)")
    print_model_info(SE_RANK_MIXER_100M_REDUCTION_8, "SE-RankMixer 100M (Reduction 8)")
    print_model_info(SE_RANK_MIXER_1B_REDUCTION_8, "SE-RankMixer 1B (Reduction 8)")
    print_model_info(
        SE_RANK_MIXER_100M_REDUCTION_32, "SE-RankMixer 100M (Reduction 32)"
    )
    print_model_info(SE_RANK_MIXER_1B_REDUCTION_32, "SE-RankMixer 1B (Reduction 32)")

    print("\n" + "=" * 50)
    print("Configuration Summary:")
    print("TokenMixer Models:")
    print(f"  100M Original: {RANK_MIXER_100M.total_parameters:,} parameters")
    print(f"  1B Original: {RANK_MIXER_1B.total_parameters:,} parameters")
    print(f"  100M Adjusted: {RANK_MIXER_100M_ADJUSTED.total_parameters:,} parameters")
    print(f"  1B Adjusted: {RANK_MIXER_1B_ADJUSTED.total_parameters:,} parameters")

    print("SE-RankMixer Models:")
    print(f"  100M (R16): {SE_RANK_MIXER_100M.total_parameters:,} parameters")
    print(f"  1B (R16): {SE_RANK_MIXER_1B.total_parameters:,} parameters")
    print(
        f"  100M (R8): {SE_RANK_MIXER_100M_REDUCTION_8.total_parameters:,} parameters"
    )
    print(f"  1B (R8): {SE_RANK_MIXER_1B_REDUCTION_8.total_parameters:,} parameters")
    print(
        f"  100M (R32): {SE_RANK_MIXER_100M_REDUCTION_32.total_parameters:,} parameters"
    )
    print(f"  1B (R32): {SE_RANK_MIXER_1B_REDUCTION_32.total_parameters:,} parameters")


if __name__ == "__main__":
    main()
