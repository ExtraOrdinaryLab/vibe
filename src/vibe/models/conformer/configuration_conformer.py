from typing import List, Optional

from transformers.configuration_utils import PretrainedConfig


class ConformerConfig(PretrainedConfig):
    """Configuration class"""
    model_type = "conformer"
    
    def __init__(
        self,
        num_mel_bins: int = 80,
        num_blocks: int = 6, 
        output_hidden_size: int = 256, 
        attention_heads: int = 4,
        linear_units: int = 2048,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        conv_bias: bool = True,
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        mlp_type: str = 'position_wise_feed_forward',
        mlp_bias: bool = True,
        n_expert: int = 8,
        n_expert_activated: int = 2,
        conv_norm_eps: float = 1e-5,
        conv_inner_factor: int = 2,
        final_norm: bool = True,
        use_mfa: bool = True,  # Add MFA option
        emb_sizes: int = 192, 
        pool_mode: str = 'attention', 
        attention_channels: int = 128, 
        num_time_masks: int = 2, 
        time_mask_width: int = 5, 
        num_freq_masks: int = 2, 
        freq_mask_width: int = 10, 
        scale: float = 30.0, 
        margin: float = 0.2, 
        pad_token_id: int = 0, 
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.num_mel_bins = num_mel_bins
        self.num_blocks = num_blocks
        self.output_hidden_size = output_hidden_size
        self.input_layer = input_layer
        self.pos_enc_layer_type = pos_enc_layer_type
        self.attention_heads = attention_heads
        self.linear_units = linear_units
        self.dropout_rate = dropout_rate
        self.positional_dropout_rate = positional_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.normalize_before = normalize_before
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.positionwise_conv_kernel_size = positionwise_conv_kernel_size
        self.macaron_style = macaron_style
        self.selfattention_layer_type = selfattention_layer_type
        self.activation_type = activation_type
        self.use_cnn_module = use_cnn_module
        self.cnn_module_kernel = cnn_module_kernel
        self.causal = causal
        self.cnn_module_norm = cnn_module_norm
        self.query_bias = query_bias
        self.key_bias = key_bias
        self.value_bias = value_bias
        self.conv_bias = conv_bias
        self.gradient_checkpointing = gradient_checkpointing
        self.use_sdpa = use_sdpa
        self.layer_norm_type = layer_norm_type
        self.norm_eps = norm_eps
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.mlp_type = mlp_type
        self.mlp_bias = mlp_bias
        self.n_expert = n_expert
        self.n_expert_activated = n_expert_activated
        self.conv_norm_eps = conv_norm_eps
        self.conv_inner_factor = conv_inner_factor
        self.final_norm = final_norm
        self.use_mfa = use_mfa

        self.emb_sizes = emb_sizes
        self.pool_mode = pool_mode
        self.attention_channels = attention_channels

        self.num_time_masks = num_time_masks
        self.time_mask_width = time_mask_width
        self.num_freq_masks = num_freq_masks
        self.freq_mask_width = freq_mask_width

        self.scale = scale
        self.margin = margin