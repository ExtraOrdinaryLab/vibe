#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer and Conformer encoder implementations.

This module provides implementations of the BaseEncoder class along with
specialized Transformer and Conformer encoders for speech processing tasks.
These encoder architectures support various features like streaming inference
with chunk-based processing, different positional encoding schemes, and
subsampling methods.
"""

from typing import Tuple, List, Optional

import torch
import torch.nn as nn

from .attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    RopeMultiHeadedAttention
)
from .convolution import ConvolutionModule
from .embedding import (
    PositionalEncoding,
    RelPositionalEncoding,
    RopePositionalEncoding,
    NoPositionalEncoding
)
from .encoder_layer import (
    TransformerEncoderLayer,
    ConformerEncoderLayer
)
from .positionwise_feed_forward import PositionwiseFeedForward
from .subsampling import (
    LinearNoSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8
)
from .common import get_activation
from .mask import (
    make_pad_mask,
    add_optional_chunk_mask
)


class BaseEncoder(nn.Module):
    """
    Base encoder class for speech processing models.
    
    This class implements common encoder functionality for both Transformer
    and Conformer architectures, including chunked processing for streaming
    inference, different subsampling strategies, and positional encoding options.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: Optional[nn.Module] = None,
        use_dynamic_left_chunk: bool = False,
    ):
        """
        Initialize a BaseEncoder instance.
        
        Args:
            input_size (int): Dimension of input features
            output_size (int): Dimension of attention/hidden layers
            attention_heads (int): Number of attention heads for multi-head attention
            linear_units (int): Number of units in position-wise feed-forward layers
            num_blocks (int): Number of encoder blocks
            dropout_rate (float): Dropout rate for most layers
            positional_dropout_rate (float): Dropout rate for positional encoding
            attention_dropout_rate (float): Dropout rate in attention layers
            input_layer (str): Input layer type for subsampling
                Options: ["linear", "conv2d", "conv2d2", "conv2d6", "conv2d8"]
            pos_enc_layer_type (str): Positional encoding layer type
                Options: ["abs_pos", "rel_pos", "no_pos"]
            normalize_before (bool): If True, use layer normalization before each block
                If False, use layer normalization after each block
            concat_after (bool): If True, concat attention input and output
                If False, use residual connection for attention
            static_chunk_size (int): Chunk size for static chunk training/inference
            use_dynamic_chunk (bool): Whether to use dynamic chunk size for training
            global_cmvn (Optional[nn.Module]): Global CMVN (Cepstral Mean and 
                Variance Normalization) module
            use_dynamic_left_chunk (bool): Whether to use dynamic left chunk size
                in dynamic chunk training
        """
        super().__init__()
        
        # Total output size is the encoder output size multiplied by the number of blocks
        self._output_size = output_size * num_blocks

        # Select positional encoding class based on configuration
        if pos_enc_layer_type == "abs_pos":
            pos_enc = PositionalEncoding(output_size, positional_dropout_rate, max_len=50_000)
        elif pos_enc_layer_type == "rel_pos":
            pos_enc = RelPositionalEncoding(output_size, positional_dropout_rate, max_len=100_000)
        elif pos_enc_layer_type == "rope_pos":
            pos_enc = RopePositionalEncoding(output_size, positional_dropout_rate, max_len=100_000, head_dim=output_size//attention_heads)
        elif pos_enc_layer_type == "no_pos":
            pos_enc = NoPositionalEncoding(output_size, positional_dropout_rate)
        else:
            raise ValueError(f"Unknown positional encoding layer type: {pos_enc_layer_type}")

        # Select input layer (subsampling) class based on configuration
        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        elif input_layer == "conv2d2":
            subsampling_class = Conv2dSubsampling2
        else:
            raise ValueError(f"Unknown input layer type: {input_layer}")

        # Store configurations
        self.global_cmvn = global_cmvn
        
        # Create embedding with specified subsampling and positional encoding
        self.embed = subsampling_class(
            input_size,
            output_size,
            dropout_rate,
            pos_enc,
        )

        # Additional configurations
        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size * num_blocks, eps=1e-12)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

    def output_size(self) -> int:
        """
        Get the output dimension of the encoder.
        
        Returns:
            int: Output dimension size
        """
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the encoder.
        
        Args:
            xs (torch.Tensor): Padded input tensor of shape (B, T, D)
                where B is batch size, T is maximum input length, D is input dimension
            xs_lens (torch.Tensor): Tensor of actual input lengths of shape (B,)
            decoding_chunk_size (int): Chunk size for dynamic chunk-based decoding
                - 0: Use random chunk size during training
                - <0: Use full context during decoding
                - >0: Use fixed chunk size during decoding
            num_decoding_left_chunks (int): Number of left chunks to attend to
                - >=0: Use specific number of left chunks
                - <0: Use all left chunks
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - xs (torch.Tensor): Encoder output tensor of shape (B, T', D')
                  where T' is the subsampled sequence length and D' is output dimension
                - masks (torch.Tensor): Padding mask after subsampling (B, 1, T')
        """
        # Create padding mask: True for non-padded positions, False for padding
        masks = ~make_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, T)
        
        # Apply global CMVN if provided
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        
        # Apply subsampling and get positional embeddings
        xs, pos_emb, masks = self.embed(xs, masks)
        
        # Save original padding mask for attention calculations
        mask_pad = masks  # (B, 1, T/subsample_rate)
        
        # Add optional chunk mask for streaming processing
        chunk_masks = add_optional_chunk_mask(
            xs, masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size,
            num_decoding_left_chunks
        )
        
        # Apply encoder layers and collect outputs from all layers
        out = []
        for layer in self.encoders:
            xs, chunk_masks, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
            out.append(xs)
        
        # Concatenate outputs from all encoder layers
        xs = torch.cat(out, dim=-1)
        
        # Apply final normalization if normalize_before is True
        if self.normalize_before:
            xs = self.after_norm(xs)
        
        # Return output and the padding mask (for potential use in attention with decoder)
        return xs, masks

    def forward_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        subsampling_cache: Optional[torch.Tensor] = None,
        elayers_output_cache: Optional[List[torch.Tensor]] = None,
        conformer_cnn_cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Process a single chunk of input for streaming inference.
        
        Args:
            xs (torch.Tensor): Input chunk tensor of shape (1, chunk_length, D)
            offset (int): Current offset in encoder output time dimension
            required_cache_size (int): Cache size required for next chunk computation
                - >=0: Actual cache size to keep
                - <0: Keep all history cache
            subsampling_cache (Optional[torch.Tensor]): Cache for subsampling output
            elayers_output_cache (Optional[List[torch.Tensor]]): Cache for encoder layers output
            conformer_cnn_cache (Optional[List[torch.Tensor]]): Cache for Conformer CNN modules
        
        Returns:
            Tuple containing:
                - torch.Tensor: Output for current chunk
                - torch.Tensor: Updated subsampling cache for next chunk
                - List[torch.Tensor]: Updated encoder layers cache for next chunk
                - List[torch.Tensor]: Updated Conformer CNN cache for next chunk
        """
        # Ensure we're processing a single sequence (batch size = 1)
        assert xs.size(0) == 1
        
        # Create dummy mask for interface compatibility (all positions attended to)
        tmp_masks = torch.ones(
            1, xs.size(1), device=xs.device, dtype=torch.bool
        ).unsqueeze(1)
        
        # Apply global CMVN if provided
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        
        # Apply subsampling with positional encoding
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
        
        # Prepend cached output if available
        if subsampling_cache is not None:
            cache_size = subsampling_cache.size(1)
            xs = torch.cat((subsampling_cache, xs), dim=1)
        else:
            cache_size = 0
        
        # Get positional encoding that includes the cache positions
        pos_emb = self.embed.position_encoding(offset - cache_size, xs.size(1))
        
        # Determine next cache start position based on required_cache_size
        if required_cache_size < 0:
            # Keep all history
            next_cache_start = 0
        elif required_cache_size == 0:
            # Don't cache anything
            next_cache_start = xs.size(1)
        else:
            # Cache the required number of frames
            next_cache_start = max(xs.size(1) - required_cache_size, 0)
        
        # Create cache for subsampling output
        r_subsampling_cache = xs[:, next_cache_start:, :]
        
        # Create mask for transformer/conformer layers (all positions attended to)
        masks = torch.ones(
            1, xs.size(1), device=xs.device, dtype=torch.bool
        ).unsqueeze(1)
        
        # Initialize output caches
        r_elayers_output_cache = []
        r_conformer_cnn_cache = []
        
        # Process through each encoder layer
        for i, layer in enumerate(self.encoders):
            # Get cached values for current layer, if available
            attn_cache = None if elayers_output_cache is None else elayers_output_cache[i]
            cnn_cache = None if conformer_cnn_cache is None else conformer_cnn_cache[i]
            
            # Process through the layer
            xs, _, new_cnn_cache = layer(
                xs,
                masks,
                pos_emb,
                output_cache=attn_cache,
                cnn_cache=cnn_cache
            )
            
            # Update caches for next chunk
            r_elayers_output_cache.append(xs[:, next_cache_start:, :])
            r_conformer_cnn_cache.append(new_cnn_cache)
        
        # Apply final normalization if needed
        if self.normalize_before:
            xs = self.after_norm(xs)

        # Return current chunk output and updated caches
        return (
            xs[:, cache_size:, :],  # Current chunk output (exclude cache)
            r_subsampling_cache,    # Subsampling cache for next chunk
            r_elayers_output_cache, # Encoder layers cache for next chunk
            r_conformer_cnn_cache   # Conformer CNN cache for next chunk
        )

    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input sequence chunk by chunk for streaming inference.
        
        This method handles the streaming decoding scenario by processing
        the input sequence in chunks with attention to previous chunks.
        
        Args:
            xs (torch.Tensor): Input tensor of shape (1, max_len, dim)
            decoding_chunk_size (int): Chunk size for decoding
            num_decoding_left_chunks (int): Number of left chunks to attend to
                - >=0: Use specific number of left chunks
                - <0: Use all left chunks
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ys (torch.Tensor): Output tensor
                - masks (torch.Tensor): Mask tensor for the output
        """
        assert decoding_chunk_size > 0
        # The model must be trained with either static or dynamic chunking
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        
        # Get subsampling rate and context from embedding layer
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        
        # Calculate stride and window size for chunk processing
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        
        # Initialize caches
        subsampling_cache: Optional[torch.Tensor] = None
        elayers_output_cache: Optional[List[torch.Tensor]] = None
        conformer_cnn_cache: Optional[List[torch.Tensor]] = None
        
        # Initialize outputs and offset tracking
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Process input sequence chunk by chunk
        for cur in range(0, num_frames - context + 1, stride):
            # Determine end of current chunk
            end = min(cur + decoding_window, num_frames)
            
            # Extract current chunk
            chunk_xs = xs[:, cur:end, :]
            
            # Process current chunk and update caches
            (y, subsampling_cache, elayers_output_cache,
             conformer_cnn_cache) = self.forward_chunk(
                chunk_xs, offset,
                required_cache_size,
                subsampling_cache,
                elayers_output_cache,
                conformer_cnn_cache
            )
            
            # Collect outputs
            outputs.append(y)
            
            # Update offset for next chunk
            offset += y.size(1)
        
        # Concatenate all chunk outputs
        ys = torch.cat(outputs, 1)
        
        # Create mask for output (all positions are valid)
        masks = torch.ones(1, ys.size(1), device=ys.device, dtype=torch.bool)
        masks = masks.unsqueeze(1)
        
        return ys, masks


class TransformerEncoder(BaseEncoder):
    """
    Transformer encoder implementation.
    
    This encoder uses standard Transformer architecture with multi-head
    self-attention and position-wise feed-forward networks.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: Optional[nn.Module] = None,
        use_dynamic_left_chunk: bool = False,
    ):
        """
        Initialize a TransformerEncoder instance.
        
        Args:
            See BaseEncoder for argument descriptions.
        """
        super().__init__(
            input_size, output_size, attention_heads,
            linear_units, num_blocks, dropout_rate,
            positional_dropout_rate, attention_dropout_rate,
            input_layer, pos_enc_layer_type, normalize_before,
            concat_after, static_chunk_size, use_dynamic_chunk,
            global_cmvn, use_dynamic_left_chunk
        )
        
        # Create encoder layers with standard multi-head attention
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads, 
                    output_size,
                    attention_dropout_rate
                ),
                PositionwiseFeedForward(
                    output_size, 
                    linear_units,
                    dropout_rate
                ), 
                dropout_rate,
                normalize_before, 
                concat_after
            ) for _ in range(num_blocks)
        ])


class ConformerEncoder(BaseEncoder):
    """
    Conformer encoder implementation.
    
    This encoder combines self-attention and convolution modules for improved
    speech recognition performance, incorporating techniques from both
    Transformer and CNN architectures.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: Optional[nn.Module] = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
    ):
        """
        Initialize a ConformerEncoder instance.
        
        Args:
            input_size to use_dynamic_left_chunk: See BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size for position-wise conv1d
            macaron_style (bool): Whether to use macaron style for position-wise layer
            selfattention_layer_type (str): Type of self-attention layer
                (parameter kept for configuration compatibility)
            activation_type (str): Type of activation function
            use_cnn_module (bool): Whether to use convolutional module
            cnn_module_kernel (int): Kernel size for convolutional module
            causal (bool): Whether to use causal convolution
            cnn_module_norm (str): Normalization type for CNN module
        """
        super().__init__(
            input_size, output_size, attention_heads,
            linear_units, num_blocks, dropout_rate,
            positional_dropout_rate, attention_dropout_rate,
            input_layer, pos_enc_layer_type, normalize_before,
            concat_after, static_chunk_size, use_dynamic_chunk,
            global_cmvn, use_dynamic_left_chunk
        )
        
        # Get activation function
        activation = get_activation(activation_type)

        # Select self-attention layer type based on positional encoding
        if pos_enc_layer_type == "no_pos":
            encoder_selfattn_layer = MultiHeadedAttention
        elif pos_enc_layer_type == "rel_pos":
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
        elif pos_enc_layer_type == "rope_pos":
            encoder_selfattn_layer = RopeMultiHeadedAttention
        
        # Define arguments for self-attention layer
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
        )
        
        # Define feed-forward module and its arguments
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        
        # Define convolution module and its arguments
        convolution_layer = ConvolutionModule
        convolution_layer_args = (
            output_size, 
            cnn_module_kernel, 
            activation,
            cnn_module_norm, 
            causal
        )

        # Create Conformer encoder layers
        self.encoders = nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                # Self-attention module
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                # Feed-forward module (after attention)
                positionwise_layer(*positionwise_layer_args),
                # Feed-forward module (before attention, macaron style)
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                # Convolution module
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(num_blocks)
        ])
