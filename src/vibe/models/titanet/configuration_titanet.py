from transformers.configuration_utils import PretrainedConfig


class TitaNetConfig(PretrainedConfig):
    """Configuration class for TitaNet"""
    model_type = "titanet"
    
    def __init__(
        self,
        num_mel_bins: int = 80,
        num_mega_blocks: int = 18, # 18, 10, 5
        num_sub_blocks: int = 3,
        encoder_hidden_size: int = 256, # 256, 512, 1024
        encoder_output_size: int = 1536,
        emb_sizes: int = 192,
        mega_block_kernel_size: int = 3, # 3, 7, 11
        prolog_kernel_size: int =3,
        epilog_kernel_size: int = 1,
        attention_hidden_size: int = 128,
        se_reduction: int = 16,
        dropout: float = 0.1,
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
        self.num_mega_blocks = num_mega_blocks
        self.num_sub_blocks = num_sub_blocks
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_output_size = encoder_output_size
        self.emb_sizes = emb_sizes
        self.mega_block_kernel_size = mega_block_kernel_size
        self.prolog_kernel_size = prolog_kernel_size
        self.epilog_kernel_size = epilog_kernel_size
        self.attention_hidden_size = attention_hidden_size
        self.se_reduction = se_reduction
        self.dropout = dropout

        self.num_time_masks = num_time_masks
        self.time_mask_width = time_mask_width
        self.num_freq_masks = num_freq_masks
        self.freq_mask_width = freq_mask_width

        self.scale = scale
        self.margin = margin