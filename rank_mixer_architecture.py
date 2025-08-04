from typing import Any

import torch
from torch import nn

from feed_forward import FeedForward, SwiGLUFeedForward
from squeeze_excitation import SqueezeExcitation
from token_mixer import TokenMixer


class RankMixerBlock(nn.Module):
    """
    A single RankMixer block combining Token Mixing and Feed Forward layers.

    Similar to a Transformer block but uses TokenMixer instead of self-attention.
    Architecture: TokenMixer -> FeedForward with optional residual connections.

    Args:
        num_tokens (int): Number of input tokens
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads for token mixing
        ff_dim (int): Feed forward intermediate dimension (typically 4x hidden_dim)
        dropout_rate (float): Dropout probability (default: 0.1)
        ff_activation (str): Feed forward activation function (default: 'gelu')
        use_swiglu (bool): Whether to use SwiGLU instead of standard FFN (default: False)
        prenorm (bool): Whether to use pre-normalization (default: True)
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        ff_activation: str = "gelu",
        use_swiglu: bool = False,
        prenorm: bool = True,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.prenorm = prenorm

        # Token Mixer layer
        self.token_mixer = TokenMixer(
            num_tokens=num_tokens, hidden_dim=hidden_dim, num_heads=num_heads
        )

        # Feed Forward layer
        if use_swiglu:
            self.feed_forward = SwiGLUFeedForward(
                hidden_dim=hidden_dim, ff_dim=ff_dim, dropout_rate=dropout_rate
            )
        else:
            self.feed_forward = FeedForward(
                hidden_dim=hidden_dim,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                activation=ff_activation,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RankMixer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, hidden_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_tokens, hidden_dim)
        """
        # Token mixing
        x = self.token_mixer(x)

        # Feed forward
        x = self.feed_forward(x)

        return x


class SERankMixerBlock(nn.Module):
    """
    A single RankMixer block using Squeeze and Excitation instead of TokenMixer.

    Architecture: SE Module -> FeedForward with optional residual connections.
    SE provides channel-wise attention mechanism as an alternative to token mixing.

    Args:
        hidden_dim (int): Hidden dimension size
        ff_dim (int): Feed forward intermediate dimension (typically 4x hidden_dim)
        dropout_rate (float): Dropout probability (default: 0.1)
        ff_activation (str): Feed forward activation function (default: 'gelu')
        use_swiglu (bool): Whether to use SwiGLU instead of standard FFN (default: False)
        prenorm (bool): Whether to use pre-normalization (default: True)
        se_reduction_ratio (int): SE reduction ratio (default: 16)
        se_activation (str): SE activation function (default: 'relu')
        se_gate_activation (str): SE gate activation (default: 'sigmoid')
    """

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        ff_activation: str = "gelu",
        use_swiglu: bool = False,
        prenorm: bool = True,
        se_reduction_ratio: int = 16,
        se_activation: str = "relu",
        se_gate_activation: str = "sigmoid",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.prenorm = prenorm

        # Squeeze and Excitation layer
        self.se_module = SqueezeExcitation(
            in_channels=hidden_dim,
            reduction_ratio=se_reduction_ratio,
            activation=se_activation,
            gate_activation=se_gate_activation,
        )

        # Feed Forward layer
        if use_swiglu:
            self.feed_forward = SwiGLUFeedForward(
                hidden_dim=hidden_dim, ff_dim=ff_dim, dropout_rate=dropout_rate
            )
        else:
            self.feed_forward = FeedForward(
                hidden_dim=hidden_dim,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                activation=ff_activation,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SE RankMixer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, hidden_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_tokens, hidden_dim)
        """
        # Squeeze and Excitation (treats tokens as channels)
        # Reshape from (B, T, D) to (B, D, T) for SE, then back
        batch_size, num_tokens, hidden_dim = x.shape
        x_se = x.transpose(1, 2)  # (B, D, T)
        x_se = self.se_module(x_se)  # Apply SE
        x = x_se.transpose(1, 2)  # Back to (B, T, D)

        # Feed forward
        x = self.feed_forward(x)

        return x


class RankMixerModel(nn.Module):
    """
    Complete RankMixer model with input embedding concatenation and partitioning.

    Input embeddings are concatenated and partitioned into T tokens each with D dimensions.
    No vocabulary size is required as the model works with continuous embeddings.

    Args:
        num_tokens (int): Number of tokens to partition input into (T)
        hidden_dim (int): Hidden dimension size (D)
        num_layers (int): Number of RankMixer layers (L)
        num_heads (int): Number of attention heads (only used for TokenMixer)
        ff_dim (int): Feed forward intermediate dimension
        dropout_rate (float): Dropout probability
        ff_activation (str): Feed forward activation function
        use_swiglu (bool): Whether to use SwiGLU instead of standard FFN
        prenorm (bool): Whether to use pre-normalization
        use_position_embedding (bool): Whether to add position embeddings
        max_position_embeddings (int): Maximum position embeddings
        layer_norm_eps (float): Layer normalization epsilon
        initializer_range (float): Weight initialization range
        use_se (bool): Whether to use SE instead of TokenMixer (default: False)
        se_reduction_ratio (int): SE reduction ratio (default: 16)
        se_activation (str): SE activation function (default: 'relu')
        se_gate_activation (str): SE gate activation (default: 'sigmoid')
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        ff_activation: str = "gelu",
        use_swiglu: bool = False,
        prenorm: bool = True,
        use_position_embedding: bool = True,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        use_se: bool = False,
        se_reduction_ratio: int = 16,
        se_activation: str = "relu",
        se_gate_activation: str = "sigmoid",
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.use_position_embedding = use_position_embedding
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.use_se = use_se

        # Position embeddings
        if use_position_embedding:
            self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)

        # Layer normalization for input
        self.input_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # RankMixer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_se:
                # Use SE-based RankMixer block
                layer = SERankMixerBlock(
                    hidden_dim=hidden_dim,
                    ff_dim=ff_dim,
                    dropout_rate=dropout_rate,
                    ff_activation=ff_activation,
                    use_swiglu=use_swiglu,
                    prenorm=prenorm,
                    se_reduction_ratio=se_reduction_ratio,
                    se_activation=se_activation,
                    se_gate_activation=se_gate_activation,
                )
            else:
                # Use TokenMixer-based RankMixer block
                layer = RankMixerBlock(
                    num_tokens=num_tokens,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout_rate=dropout_rate,
                    ff_activation=ff_activation,
                    use_swiglu=use_swiglu,
                    prenorm=prenorm,
                )
            self.layers.append(layer)

        # Final layer normalization
        self.output_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_position_embeddings(
        self, seq_length: int, device: torch.device
    ) -> torch.Tensor:
        """Get position embeddings for the given sequence length."""
        position_ids = torch.arange(seq_length, device=device)
        return self.position_embeddings(position_ids)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the RankMixer model.

        Args:
            input_embeddings (torch.Tensor): Input embeddings of shape (batch_size, seq_len, hidden_dim)
            attention_mask (torch.Tensor, optional): Attention mask
            position_ids (torch.Tensor, optional): Position IDs
            output_hidden_states (bool): Whether to output hidden states

        Returns:
            dict: Dictionary containing output and optional hidden states
        """
        batch_size, seq_length, hidden_dim = input_embeddings.shape

        # Validate input dimensions
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Input hidden dimension {hidden_dim} does not match model hidden dimension {self.hidden_dim}"
            )

        # Add position embeddings if enabled
        if self.use_position_embedding:
            if position_ids is None:
                position_embeddings = self.get_position_embeddings(
                    seq_length, input_embeddings.device
                )
            else:
                position_embeddings = self.position_embeddings(position_ids)
            input_embeddings = input_embeddings + position_embeddings

        # Apply input normalization
        x = self.input_norm(input_embeddings)

        # Process input through layers
        hidden_states = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if output_hidden_states:
                hidden_states.append(x)

        # Apply output normalization
        x = self.output_norm(x)

        # Prepare output
        output = {"last_hidden_state": x}
        if output_hidden_states:
            output["hidden_states"] = hidden_states

        return output


def create_rankmixer_config(
    num_tokens: int = 512,
    hidden_dim: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    ff_dim: int = 3072,
    dropout_rate: float = 0.1,
    use_se: bool = False,
    se_reduction_ratio: int = 16,
    **kwargs,
) -> dict[str, Any]:
    """
    Create a configuration dictionary for RankMixer models.

    Args:
        num_tokens (int): Number of tokens
        hidden_dim (int): Hidden dimension
        num_layers (int): Number of layers
        num_heads (int): Number of heads (only used for TokenMixer)
        ff_dim (int): Feed forward dimension
        dropout_rate (float): Dropout rate
        use_se (bool): Whether to use SE instead of TokenMixer
        se_reduction_ratio (int): SE reduction ratio
        **kwargs: Additional arguments

    Returns:
        dict: Configuration dictionary
    """
    return {
        "num_tokens": num_tokens,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ff_dim": ff_dim,
        "dropout_rate": dropout_rate,
        "use_se": use_se,
        "se_reduction_ratio": se_reduction_ratio,
        **kwargs,
    }
