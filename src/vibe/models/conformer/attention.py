import math
from typing import Optional, Tuple

import torch
from torch import nn

from .rope_utils import google_apply_rotary_emb, llama_apply_rotary_emb

T_CACHE = Tuple[torch.Tensor, torch.Tensor]


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    This implementation follows the paper "Attention Is All You Need".
    It computes scaled dot product attention with multiple heads.

    Args:
        n_head (int): The number of attention heads.
        n_feat (int): The dimension of features.
        dropout_rate (float): Dropout rate applied to attention weights.
        query_bias (bool, optional): Whether to use bias in the query projection. Defaults to True.
        key_bias (bool, optional): Whether to use bias in the key projection. Defaults to True.
        value_bias (bool, optional): Whether to use bias in the value projection. Defaults to True.
        use_sdpa (bool, optional): Whether to use scaled dot product attention. Defaults to False.
        n_kv_head (int, optional): Number of key/value heads for grouped queries. Defaults to None.
        head_dim (int, optional): Dimension of each head. Defaults to None.
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        use_sdpa: bool = False,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        """Initialize the MultiHeadedAttention module."""
        super().__init__()
        assert n_feat % n_head == 0, "Feature dimension must be divisible by number of heads"
        
        # Dimension of each head and number of heads
        self.d_k = n_feat // n_head if head_dim is None else head_dim
        self.h = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.dropout_rate = dropout_rate
        self.use_sdpa = use_sdpa
        
        # Linear projections for query, key, value
        self.linear_q = nn.Linear(n_feat, n_head * self.d_k, bias=query_bias)
        self.linear_k = nn.Linear(n_feat, self.n_kv_head * self.d_k, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, self.n_kv_head * self.d_k, bias=value_bias)
        
        # Output projection
        self.linear_out = nn.Linear(n_head * self.d_k, n_feat)
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(p=dropout_rate)

    def _forward_linearx(self, name, x, head_first=True):
        """Forward linear projection for query/key/value.
        
        Args:
            name (str): Name of the projection (query, key, or value).
            x (torch.Tensor): Input tensor.
            head_first (bool): If True, output shape will be (batch, head, time, d_k).
                               If False, output shape will be (batch, time, head, d_k).
                               
        Returns:
            torch.Tensor: Projected tensor.
        """
        n_batch = x.size(0)
        linear = getattr(self, f"linear_{name[0]}")
        h = self.h if name == "query" else self.n_kv_head
        
        x = linear(x).view(n_batch, -1, h, self.d_k)
        if head_first:
            x = x.transpose(1, 2)
            
        return x

    def _update_kv_and_cache(self, k, v, cache, head_first=True):
        """Update key-value cache for efficient decoding.
        
        Args:
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            cache (Tuple[torch.Tensor, torch.Tensor]): Previous key-value cache.
            head_first (bool): If True, input tensors have shape (batch, head, time, d_k).
                               If False, input tensors have shape (batch, time, head, d_k).
                               
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - Updated key tensor
                - Updated value tensor
                - New cache
        """
        key_cache, value_cache = cache
        
        if key_cache.size(0) == 0:
            # No previous cache
            return k, v, (k, v)
        
        if head_first:
            key_cache = torch.cat([key_cache, k], dim=2)
            value_cache = torch.cat([value_cache, v], dim=2)
            return key_cache, value_cache, (key_cache, value_cache)
        else:
            key_cache = torch.cat([key_cache, k], dim=1)
            value_cache = torch.cat([value_cache, v], dim=1)
            return key_cache, value_cache, (key_cache, value_cache)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value tensors for multi-head attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Transformed query tensor (#batch, n_head, time1, d_k)
                - Transformed key tensor (#batch, n_head, time2, d_k)
                - Transformed value tensor (#batch, n_head, time2, d_k)
        """
        n_batch = query.size(0)
        
        # Linear projection and split heads
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.n_kv_head, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.n_kv_head, self.d_k)
        
        # Transpose to (#batch, head, time, d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(
        self, value: torch.Tensor, scores: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention context vector from scores and values.

        Args:
            value (torch.Tensor): Transformed value tensor (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score tensor (#batch, n_head, time1, time2).
            mask (torch.Tensor, optional): Mask tensor (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor after attention (#batch, time1, d_model).
        """
        n_batch = value.size(0)
        
        # Apply mask if provided
        if mask is not None and mask.size(0) > 0:
            # Convert mask to binary mask where 0s are positions to mask
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            
            # Fill masked positions with -inf before softmax
            scores = scores.masked_fill(mask, -float('inf'))
            
            # Apply softmax to get attention weights
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            # Simple softmax if no mask
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        # Apply dropout to attention weights
        p_attn = self.dropout(attn)
        
        # Weighted sum of values
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        
        # Combine heads and apply output projection
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos_emb: torch.Tensor = torch.empty(0),
        cache: T_CACHE = (torch.zeros((0, 0, 0, 0)), torch.zeros((0, 0, 0, 0))),
    ) -> torch.Tensor:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor, optional): Mask tensor with following possible shapes:
                1. Cross attention between decoder and encoder: (#batch, 1, T)
                2. Self attention of encoder: (#batch, T, T)
                3. Self attention of decoder: (#batch, L, L)
                4. Different position in decoder: (#batch, L, T)
            pos_emb (torch.Tensor, optional): Positional embedding tensor.
                Default is an empty tensor.
            cache (Tuple[torch.Tensor, torch.Tensor], optional): Cache for keys and values
                for efficient decoding.

        Returns:
            torch.Tensor: Output tensor after attention (#batch, time1, d_model).
            Tuple[torch.Tensor, torch.Tensor]: Updated cache.
        """
        q, k, v = self.forward_qkv(query, key, value)
        
        # Update cache if needed
        if cache[0].size(0) > 0:
            k, v, new_cache = self._update_kv_and_cache(k, v, cache)
        else:
            new_cache = cache
        
        # Compute attention scores with scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply attention weights to values
        output = self.forward_attention(v, scores, mask)
        
        return output, new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.

    This implementation is based on the paper:
    "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    (https://arxiv.org/abs/1901.02860)

    Args:
        n_head (int): The number of attention heads.
        n_feat (int): The dimension of features.
        dropout_rate (float): Dropout rate applied to attention weights.
    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Initialize the RelPositionMultiHeadedAttention module."""
        super().__init__(n_head, n_feat, dropout_rate)
        
        # Linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        
        # Learnable biases for relative position attention
        # Used in matrices c and d as described in the paper Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        
        # Initialize parameters with Xavier uniform
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor, zero_triu: bool = False) -> torch.Tensor:
        """Compute relative positional encoding shift.

        This function shifts the input tensor to implement relative positional
        encoding as described in the Transformer-XL paper.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, time2).
            zero_triu (bool): If True, return the lower triangular part of the matrix.

        Returns:
            torch.Tensor: Shifted output tensor.
        """
        # Create zero padding
        zero_pad = torch.zeros(
            (x.size(0), x.size(1), x.size(2), 1),
            device=x.device,
            dtype=x.dtype
        )
        
        # Concatenate zero padding to the right
        x_padded = torch.cat([zero_pad, x], dim=-1)
        
        # Reshape and slice to shift the matrix
        x_padded = x_padded.view(
            x.size(0),
            x.size(1),
            x.size(3) + 1,
            x.size(2)
        )
        x = x_padded[:, :, 1:].view_as(x)
        
        # Apply lower triangular mask if requested
        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos_emb: torch.Tensor = None,
        cache: T_CACHE = (torch.zeros((0, 0, 0, 0)), torch.zeros((0, 0, 0, 0))),
    ) -> torch.Tensor:
        """Compute scaled dot product attention with relative positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor, optional): Mask tensor (#batch, 1, time2) or (#batch, time1, time2).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time2, size).
            cache (Tuple[torch.Tensor, torch.Tensor], optional): Cache for keys and values
                for efficient decoding.

        Returns:
            torch.Tensor: Output tensor after attention (#batch, time1, d_model).
            Tuple[torch.Tensor, torch.Tensor]: Updated cache.
        """
        # Transform query, key, value with multi-head projections
        q, k, v = self.forward_qkv(query, key, value)
        
        # Update cache if needed
        if cache[0].size(0) > 0:
            k, v, new_cache = self._update_kv_and_cache(k, v, cache)
        else:
            new_cache = cache
        
        # Reshape query for relative positional encoding
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        # Process positional embeddings
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # Add learnable biases to the query for content-based and position-based attention
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)  # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)  # (batch, head, time1, d_k)

        # Compute attention scores
        # Matrix a and c from the paper (content-to-content and content-to-position)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))  # (batch, head, time1, time2)

        # Matrix b and d from the paper (position-to-content)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))  # (batch, head, time1, time2)
        
        # Note: rel_shift is commented out as it's not needed for speech recognition
        # and requires special handling for streaming applications
        # matrix_bd = self.rel_shift(matrix_bd)

        # Combine scores and apply scaling
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        # Apply attention weights to values
        output = self.forward_attention(v, scores, mask)
        
        return output, new_cache


# Dictionary mapping style names to implementation functions
APPLY_ROTARY_EMB = {
    'google': google_apply_rotary_emb,
    'llama': llama_apply_rotary_emb,
}


class RopeMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention with Rotary Position Embedding (RoPE).
    
    This implementation applies rotary positional embeddings to queries and keys
    before computing attention scores, as described in the RoPE paper:
    "Roformer: Enhanced transformer with rotary position embedding"
    
    Args:
        n_head (int): The number of attention heads.
        n_feat (int): The dimension of features.
        dropout_rate (float): Dropout rate applied to attention weights.
        query_bias (bool, optional): Whether to use bias in query projection. Defaults to True.
        key_bias (bool, optional): Whether to use bias in key projection. Defaults to True.
        value_bias (bool, optional): Whether to use bias in value projection. Defaults to True.
        use_sdpa (bool, optional): Whether to use scaled dot product attention. Defaults to False.
        n_kv_head (int, optional): Number of key/value heads for grouped queries. Defaults to None.
        head_dim (int, optional): Dimension of each head. Defaults to None.
        style (str, optional): Style of RoPE implementation ('google' or 'llama'). Defaults to 'google'.
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        use_sdpa: bool = False,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        style='google'
    ):
        """Initialize the RopeMultiHeadedAttention module."""
        super().__init__(
            n_head, n_feat, dropout_rate, query_bias, key_bias,
            value_bias, use_sdpa, n_kv_head, head_dim
        )
        self.style = style

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: T_CACHE = (torch.zeros((0, 0, 0, 0)), torch.zeros((0, 0, 0, 0)))
    ) -> Tuple[torch.Tensor, T_CACHE]:
        """Compute RoPE scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor, optional): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1. When applying cross attention between decoder and encoder,
                   the batch padding mask for input is in (#batch, 1, T) shape.
                2. When applying self attention of encoder,
                   the mask is in (#batch, T, T) shape.
                3. When applying self attention of decoder,
                   the mask is in (#batch, L, L) shape.
                4. If different positions in decoder see different blocks
                   of the encoder, such as Mocha, the mask could be
                   in (#batch, L, T) shape.
            pos_emb (torch.Tensor, optional): Positional embedding tensor.
            cache (Tuple[torch.Tensor, torch.Tensor], optional): Cache tensor
                (1, head, cache_t, d_k * 2), where `cache_t == chunk_size *
                num_decoding_left_chunks` and `head * d_k == size`.

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            Tuple[torch.Tensor, torch.Tensor]: Updated cache tensor
                (1, head, cache_t + time1, d_k * 2) where
                `cache_t == chunk_size * num_decoding_left_chunks` and
                `head * d_k == size`.
        """
        # Project inputs to multi-head queries, keys, and values
        q = self._forward_linearx('query', query, head_first=False)
        k = self._forward_linearx('key', key, head_first=False)
        v = self._forward_linearx('value', value, head_first=False)
        
        # Apply rotary position embeddings
        q = APPLY_ROTARY_EMB[self.style](q, pos_emb)
        k = APPLY_ROTARY_EMB[self.style](k, pos_emb)

        # Update key-value cache
        k, v, new_cache = self._update_kv_and_cache(k, v, cache, head_first=False)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        
        if not self.use_sdpa:
            # Traditional scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            output = self.forward_attention(v, scores, mask)
        else:
            # Use PyTorch's optimized scaled dot product attention
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask.unsqueeze(1) if mask.size(0) > 0 else None,
                dropout_p=self.dropout_rate if self.training else 0.0,
                scale=1 / math.sqrt(self.d_k),
            )
            # Reshape output
            output = (output.transpose(1, 2).contiguous().view(
                query.size(0), -1, self.h * self.d_k))  # (batch, time1, d_model)
            output = self.linear_out(output)
            
        return output, new_cache