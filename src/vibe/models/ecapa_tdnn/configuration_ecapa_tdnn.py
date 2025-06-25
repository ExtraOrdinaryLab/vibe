from typing import List

from transformers.configuration_utils import PretrainedConfig


class EcapaTdnnConfig(PretrainedConfig):
    """Configuration class"""
    model_type = "ecapa_tdnn"
    
    def __init__(
        self,
        num_mel_bins: int = 80,
        filters: List[int] = [512, 512, 512, 512, 1536],
        kernel_sizes: List[int] = [5, 3, 3, 3, 1],
        dilations: List[int] = [1, 2, 3, 4, 1],
        res2net_scale: int = 8,
        se_channels: int = 128,
        groups: List[int] = [1, 1, 1, 1, 1],
        dropout: float = 0.0,
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
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.res2net_scale = res2net_scale
        self.se_channels = se_channels
        self.groups = groups
        self.dropout = dropout

        self.emb_sizes = emb_sizes
        self.pool_mode = pool_mode
        self.attention_channels = attention_channels

        self.num_time_masks = num_time_masks
        self.time_mask_width = time_mask_width
        self.num_freq_masks = num_freq_masks
        self.freq_mask_width = freq_mask_width

        self.scale = scale
        self.margin = margin