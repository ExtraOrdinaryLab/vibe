import math
from typing import List, Union

from transformers.configuration_utils import PretrainedConfig


class MambaConfig(PretrainedConfig):
    """
    Configuration class for Mamba model.
    
    This class stores the configuration parameters for the Mamba architecture,
    which is a state space model with selective scan mechanism.
    """
    model_type = "mamba"
    
    def __init__(
        self,
        # Input features
        num_mel_bins: int = 80,  # Number of mel filterbank features
        
        # Core Mamba architecture parameters
        d_model: int = 256,  # Hidden dimension size (D in the paper)
        n_layers: int = 12,   # Number of Mamba layers
        dt_rank: Union[int, str] = 'auto',  # Rank of Δ projection. 'auto' sets it to d_model//16
        d_state: int = 16,   # SSM state size (N in the paper)
        expand_factor: int = 2,  # Expansion factor for hidden dimension (E in the paper)
        d_conv: int = 4,     # Kernel size for 1D convolution
        
        # Delta (Δ) parameters
        dt_min: float = 0.001,  # Minimum value for delta parameter
        dt_max: float = 0.1,    # Maximum value for delta parameter
        dt_init: str = "random",  # Delta initialization: "random" or "constant"
        dt_scale: float = 1.0,   # Scaling factor for delta values
        dt_init_floor: float = 1e-4,  # Minimum value for random initialization
        
        # Normalization and regularization
        rms_norm_eps: float = 1e-5,  # Epsilon for RMS normalization
        base_std: float = 0.02,      # Base standard deviation for weight initialization
        bias: bool = False,          # Whether to use bias in linear layers
        conv_bias: bool = True,      # Whether to use bias in convolutional layers
        inner_layernorms: bool = False,  # Whether to apply layer norms to internal activations
        
        # μP (mu-Parametrization) settings
        mup: bool = False,           # Whether to use μP (mu-Parametrization)
        mup_base_width: float = 128,  # Base width for μP (width=d_model)
        
        # Computation mode
        pscan: bool = True,  # Use parallel scan mode (True) or sequential mode (False) during training
        
        # Speaker embedding parameters
        emb_sizes: int = 192,  # Size of the final speaker embedding
        pool_mode: str = 'attention',  # Pooling method: 'attention', 'mean', or 'max'
        attention_channels: int = 128,  # Number of channels for attention-based pooling
        
        # Data augmentation parameters
        num_time_masks: int = 2,      # Number of time masks for SpecAugment
        time_mask_width: int = 5,     # Maximum width of time masks
        num_freq_masks: int = 2,      # Number of frequency masks for SpecAugment
        freq_mask_width: int = 10,    # Maximum width of frequency masks
        
        # Loss function parameters
        scale: float = 30.0,  # Scale factor for AAM softmax
        margin: float = 0.2,  # Margin for AAM softmax
        
        # Tokenizer settings
        pad_token_id: int = 0,  # ID for padding token
        
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        # Input features
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        
        # Core Mamba architecture parameters
        self.n_layers = n_layers
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_conv = d_conv
        
        # Delta (Δ) parameters
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        
        # Normalization and regularization
        self.rms_norm_eps = rms_norm_eps
        self.base_std = base_std
        self.bias = bias
        self.conv_bias = conv_bias
        self.inner_layernorms = inner_layernorms
        
        # μP (mu-Parametrization) settings
        self.mup = mup
        self.mup_base_width = mup_base_width
        
        # Computation mode
        self.pscan = pscan
        
        # Speaker embedding parameters
        self.emb_sizes = emb_sizes
        self.pool_mode = pool_mode
        self.attention_channels = attention_channels
        
        # Data augmentation parameters
        self.num_time_masks = num_time_masks
        self.time_mask_width = time_mask_width
        self.num_freq_masks = num_freq_masks
        self.freq_mask_width = freq_mask_width
        
        # Loss function parameters
        self.scale = scale
        self.margin = margin

        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width
