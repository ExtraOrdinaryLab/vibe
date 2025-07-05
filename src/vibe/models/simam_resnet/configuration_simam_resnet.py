from typing import List

from transformers.configuration_utils import PretrainedConfig


class SimAMResNetConfig(PretrainedConfig):
    """Configuration class for SimAMResNet"""
    model_type = "simam_resnet"
    
    def __init__(
        self,
        num_mel_bins: int = 80,
        in_planes: int = 64,
        emb_sizes: int = 192, 
        num_blocks: List[int] = [3, 4, 6, 3], 
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
        self.in_planes = in_planes
        # ResNet34 [3, 4, 6, 3]
        # ResNet100 [6, 16, 24, 3]
        # ResNet221 [6, 16, 48, 3]
        # ResNet293 [10, 20, 64, 3]
        self.num_blocks = num_blocks
        self.emb_sizes = emb_sizes

        self.num_time_masks = num_time_masks
        self.time_mask_width = time_mask_width
        self.num_freq_masks = num_freq_masks
        self.freq_mask_width = freq_mask_width

        self.scale = scale
        self.margin = margin