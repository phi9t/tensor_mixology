"""Custom attention layer implementation from scratch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head attention implementation from scratch.

    This module implements the standard multi-head attention mechanism as described
    in "Attention Is All You Need" with support for both self-attention and
    cross-attention patterns.

    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        embed_dim: Total embedding dimension
        scale: Scaling factor for attention scores
        q_proj: Query projection layer
        k_proj: Key projection layer
        v_proj: Value projection layer
        out_proj: Output projection layer
        dropout: Dropout layer for attention weights
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
    ):
        """Initialize the multi-head attention module.

        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights
            bias: Whether to add bias to linear layers
            add_bias_kv: Whether to add bias to key/value projections
            add_zero_attn: Whether to add zero attention
            kdim: Key dimension (defaults to embed_dim)
            vdim: Value dimension (defaults to embed_dim)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.scale = self.head_dim**-0.5
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn

        # Use provided dimensions or default to embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Optional bias for key/value
        if self.add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
            self._reset_parameters()

    def _reset_parameters(self):
        """Reset parameters using Xavier initialization."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

        if self.add_bias_kv:
            nn.init.xavier_uniform_(self.bias_k)
            nn.init.xavier_uniform_(self.bias_v)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim)
            key: Key tensor of shape (batch_size, seq_len, kdim)
            value: Value tensor of shape (batch_size, seq_len, vdim)
            key_padding_mask: Mask for padded positions in key
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            average_attn_weights: Whether to average attention weights
            is_causal: Whether to use causal attention

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, embed_dim = query.shape

        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Add bias if specified
        if self.add_bias_kv:
            k = k + self.bias_k
            v = v + self.bias_v

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Add zero attention if specified
        if self.add_zero_attn:
            # Get the current sequence length
            zero_attn_shape = (batch_size, 1, self.num_heads, self.head_dim)
            k = torch.cat(
                [k, torch.zeros(zero_attn_shape, device=k.device, dtype=k.dtype)], dim=2
            )
            v = torch.cat(
                [v, torch.zeros(zero_attn_shape, device=v.device, dtype=v.dtype)], dim=2
            )

        # Compute attention scores
        attn_output, attn_weights = self._scaled_dot_product_attention(
            q,
            k,
            v,
            key_padding_mask,
            need_weights,
            attn_mask,
            average_attn_weights,
            is_causal,
        )

        # Reshape output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, embed_dim)
        )

        # Final linear projection
        output = self.out_proj(attn_output)

        if need_weights:
            return output, attn_weights
        else:
            return output, None

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute scaled dot-product attention.

        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            v: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
            key_padding_mask: Mask for padded positions
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            average_attn_weights: Whether to average attention weights
            is_causal: Whether to use causal attention

        Returns:
            Tuple of (attention_output, attention_weights)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        key_len = k.size(2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask

        # Apply causal mask if specified
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, key_len, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # Reshape mask to match attention scores
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, float("-inf"))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        if self.dropout > 0:
            attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training
            )

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)

        # Average attention weights across heads if requested
        if need_weights and average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)

        return attn_output, attn_weights if need_weights else None


class SelfAttention(MultiHeadAttention):
    """Self-attention layer that applies attention to the same sequence."""

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            key_padding_mask: Mask for padded positions
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            average_attn_weights: Whether to average attention weights
            is_causal: Whether to use causal attention

        Returns:
            Tuple of (output, attention_weights)
        """
        return super().forward(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
