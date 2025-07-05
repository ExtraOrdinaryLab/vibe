from typing import List

from transformers.configuration_utils import PretrainedConfig


class CAMPPlusConfig(PretrainedConfig):
    """Configuration class"""
    model_type = "campplus"
    
    def __init__(
        self,
        num_mel_bins: int = 80,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
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
        self.growth_rate = growth_rate
        self.bn_size = bn_size
        self.init_channels = init_channels

        self.emb_sizes = emb_sizes
        self.pool_mode = pool_mode
        self.attention_channels = attention_channels

        self.num_time_masks = num_time_masks
        self.time_mask_width = time_mask_width
        self.num_freq_masks = num_freq_masks
        self.freq_mask_width = freq_mask_width

        self.scale = scale
        self.margin = margin