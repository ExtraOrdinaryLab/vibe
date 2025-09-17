"""
CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking.

This module implements the CAM++ architecture as described in Wang et al. (Interspeech 2023).
CAM++ combines a front-end convolution module (FCM) with a densely connected time delay 
neural network (D-TDNN) backbone enhanced with context-aware masking.
"""
from collections import OrderedDict
from typing import List, Tuple, Optional, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


def get_nonlinear(config_str: str, channels: int) -> nn.Sequential:
    """
    Create a sequential module with the specified nonlinear operations.
    
    Args:
        config_str: String containing nonlinear operations separated by '-'.
        channels: Number of channels for the operations.
        
    Returns:
        nn.Sequential: A sequential module with the specified nonlinear operations.
    """
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f'Unexpected module ({name}).')
    return nonlinear


class TDNNLayer(nn.Module):
    """
    Time Delay Neural Network (TDNN) layer.
    
    Applies one-dimensional convolution along the time axis to capture local temporal context information.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = 'batchnorm-relu'
    ):
        """
        Initialize the TDNN layer.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            stride: Stride of the convolution.
            padding: Padding added to both sides of the input. If negative, calculated automatically.
            dilation: Spacing between kernel elements.
            bias: Whether to include a bias term.
            config_str: Configuration string for the nonlinear operations.
        """
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, f'Expect equal paddings, but got even kernel size ({kernel_size})'
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TDNN layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, time_steps).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, time_steps).
        """
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):
    """
    Context-Aware Masking (CAM) layer.
    
    As described in the paper, CAM performs feature map masking using contextual information
    to focus on the speaker of interest and blur unrelated noise.
    """
    def __init__(
        self,
        bn_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        bias: bool,
        reduction: int = 2
    ):
        """
        Initialize the CAM layer.
        
        Args:
            bn_channels: Number of bottleneck channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            stride: Stride of the convolution.
            padding: Padding added to both sides of the input.
            dilation: Spacing between kernel elements.
            bias: Whether to include a bias term.
            reduction: Reduction factor for the attention mechanism.
        """
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CAM layer.
        
        The CAM layer applies feature map masking using multi-granularity pooling
        to capture both global and segment-level contextual information.
        
        Args:
            x: Input tensor of shape (batch_size, bn_channels, time_steps).
            
        Returns:
            torch.Tensor: Masked output tensor of shape (batch_size, out_channels, time_steps).
        """
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(
        self, 
        x: torch.Tensor, 
        seg_len: int = 100, 
        stype: str = 'avg'
    ) -> torch.Tensor:
        """
        Segment pooling operation for multi-granularity pooling.
        
        As described in the paper, segment pooling captures local contextual information
        which is essential for generating a more accurate attention mask.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time_steps).
            seg_len: Length of each segment.
            stype: Pooling type ('avg' or 'max').
            
        Returns:
            torch.Tensor: Pooled tensor with segment-level contextual information.
        """
        if stype == 'avg':
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == 'max':
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError('Wrong segment pooling type.')
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., :x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    """
    CAM Dense TDNN Layer with context-aware masking.
    
    This layer integrates context-aware masking into the basic D-TDNN layer
    to enhance the model's ability to focus on speaker characteristics.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = 'batchnorm-relu',
        memory_efficient: bool = False
    ):
        """
        Initialize the CAMDenseTDNNLayer.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            bn_channels: Number of bottleneck channels.
            kernel_size: Size of the convolutional kernel.
            stride: Stride of the convolution.
            dilation: Spacing between kernel elements.
            bias: Whether to include a bias term.
            config_str: Configuration string for the nonlinear operations.
            memory_efficient: Whether to use memory-efficient implementation with checkpointing.
        """
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, f'Expect equal paddings, but got even kernel size ({kernel_size})'
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )

    def bn_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bottleneck function for efficient implementation with checkpointing.
        
        Args:
            x: Input tensor.
            
        Returns:
            torch.Tensor: Output of the bottleneck function.
        """
        return self.linear1(self.nonlinear1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CAMDenseTDNNLayer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, time_steps).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, time_steps).
        """
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x, use_reentrant=True)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    """
    CAM Dense TDNN Block with dense connectivity.
    
    This block consists of multiple CAMDenseTDNNLayers with dense connections.
    As described in the paper, dense connectivity involves direct connections among all layers
    in a feed-forward manner, making the network more parameter-efficient.
    """
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = 'batchnorm-relu',
        memory_efficient: bool = False
    ):
        """
        Initialize the CAMDenseTDNNBlock.
        
        Args:
            num_layers: Number of layers in the block.
            in_channels: Number of input channels.
            out_channels: Number of output channels for each layer (growth rate).
            bn_channels: Number of bottleneck channels.
            kernel_size: Size of the convolutional kernel.
            stride: Stride of the convolution.
            dilation: Spacing between kernel elements.
            bias: Whether to include a bias term.
            config_str: Configuration string for the nonlinear operations.
            memory_efficient: Whether to use memory-efficient implementation.
        """
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient
            )
            self.add_module(f'tdnnd{i+1}', layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CAMDenseTDNNBlock.
        
        With dense connectivity, the output of each layer is concatenated with all preceding layers
        and serves as the input for the next layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, time_steps).
            
        Returns:
            torch.Tensor: Output tensor with concatenated features.
        """
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    """
    Transit Layer.
    
    As described in the paper, transit layers are used after each CAMDenseTDNNBlock
    to reduce the number of channels and control the network's complexity.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        config_str: str = 'batchnorm-relu'
    ):
        """
        Initialize the TransitLayer.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            bias: Whether to include a bias term.
            config_str: Configuration string for the nonlinear operations.
        """
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TransitLayer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, time_steps).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, time_steps).
        """
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    """
    Dense Layer for the final embedding extraction.
    
    This layer transforms the pooled statistics into the final speaker embedding.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        config_str: str = 'batchnorm-relu'
    ):
        """
        Initialize the DenseLayer.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (embedding size).
            bias: Whether to include a bias term.
            config_str: Configuration string for the nonlinear operations.
        """
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DenseLayer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, time_steps) or (batch_size, in_channels).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, time_steps) or (batch_size, out_channels).
        """
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class BasicResBlock(nn.Module):
    """
    Basic Residual Block for the Front-end Convolution Module (FCM).
    
    As described in the paper, residual blocks enhance the model's ability to be
    invariant to frequency shifts in the input features.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        """
        Initialize the BasicResBlock.
        
        Args:
            in_planes: Number of input planes.
            planes: Number of output planes.
            stride: Stride of the first convolutional layer.
        """
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=(stride, 1),
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the BasicResBlock.
        
        Args:
            x: Input tensor of shape (batch_size, in_planes, freq, time).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, planes, freq, time).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCM(nn.Module):
    """
    Front-end Convolution Module (FCM).
    
    As described in the paper, FCM uses two-dimensional convolutions with residual connections
    to enhance the network's ability to capture patterns in both time and frequency domains.
    """
    def __init__(
        self,
        block: Type[BasicResBlock] = BasicResBlock,
        num_blocks: List[int] = [2, 2],
        m_channels: int = 32,
        feat_dim: int = 80
    ):
        """
        Initialize the FCM.
        
        Args:
            block: Residual block class to use.
            num_blocks: List of number of blocks for each layer.
            m_channels: Number of channels for the module.
            feat_dim: Feature dimension.
        """
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)

        self.conv2 = nn.Conv2d(
            m_channels, 
            m_channels, 
            kernel_size=3, 
            stride=(2, 1), 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(
        self, 
        block: Type[BasicResBlock], 
        planes: int, 
        num_blocks: int, 
        stride: int
    ) -> nn.Sequential:
        """
        Create a layer of residual blocks.
        
        Args:
            block: Residual block class to use.
            planes: Number of output planes.
            num_blocks: Number of blocks in the layer.
            stride: Stride for the first block.
            
        Returns:
            nn.Sequential: Sequential layer of residual blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FCM.
        
        The output feature map is flattened along the channel and frequency dimensions
        and serves as input for the D-TDNN backbone.
        
        Args:
            x: Input tensor of shape (batch_size, freq, time).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, time).
        """
        x = x.unsqueeze(1)  # (B, 1, F, T)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1]*shape[2], shape[3])
        return out


def statistics_pooling(
    x: torch.Tensor, 
    dim: int = -1, 
    keepdim: bool = False, 
    unbiased: bool = True, 
    eps: float = 1e-2
) -> torch.Tensor:
    """
    Compute mean and standard deviation for pooling.
    
    Args:
        x: Input tensor.
        dim: Dimension along which to compute statistics.
        keepdim: Whether to keep the dimension.
        unbiased: Whether to use unbiased standard deviation.
        eps: Small constant for numerical stability.
        
    Returns:
        torch.Tensor: Concatenated mean and standard deviation.
    """
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    """
    Statistical Pooling layer.
    
    This layer computes the mean and standard deviation of the input along the time dimension,
    which is essential for transforming frame-level features into utterance-level embeddings.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the StatsPool layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time_steps).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels*2).
        """
        return statistics_pooling(x)


class CAMPPlus(nn.Module):
    """
    CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking.
    
    As described in the paper, CAM++ is an efficient network for speaker verification that 
    uses context-aware masking with multi-granularity pooling and a front-end convolution
    module to enhance the D-TDNN backbone.
    """
    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 512,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
        config_str: str = 'batchnorm-relu',
        memory_efficient: bool = True
    ):
        """
        Initialize the CAMPPlus model.
        
        Args:
            feat_dim: Feature dimension (input features).
            embedding_size: Size of the output speaker embedding.
            growth_rate: Growth rate for the dense blocks (output channels for each layer).
            bn_size: Bottleneck size multiplier.
            init_channels: Initial number of channels.
            config_str: Configuration string for the nonlinear operations.
            memory_efficient: Whether to use memory-efficient implementation.
        """
        super(CAMPPlus, self).__init__()

        # Front-end Convolution Module (FCM)
        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        # D-TDNN backbone with CAM
        self.xvector = nn.Sequential(
            OrderedDict([
                ('tdnn',
                 TDNNLayer(
                     channels,
                     init_channels,
                     5,
                     stride=2,
                     dilation=1,
                     padding=-1,
                     config_str=config_str
                 )),
            ]))
        channels = init_channels
        
        # Three CAMDenseTDNNBlocks with different configurations
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient
            )
            self.xvector.add_module(f'block{i+1}', block)
            channels = channels + num_layers * growth_rate
            
            # Transit layer to reduce the number of channels
            self.xvector.add_module(
                f'transit{i+1}',
                TransitLayer(
                    channels,
                    channels // 2,
                    bias=False,
                    config_str=config_str
                )
            )
            channels //= 2

        # Final nonlinear transformation before pooling
        self.xvector.add_module(
            'out_nonlinear', get_nonlinear(config_str, channels)
        )

        # Statistical pooling and final embedding layer
        self.xvector.add_module('stats', StatsPool())
        self.xvector.add_module(
            'dense',
            DenseLayer(channels * 2, embedding_size, config_str='batchnorm_')
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CAMPPlus model.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, features).
            
        Returns:
            torch.Tensor: Speaker embedding tensor of shape (batch_size, embedding_size).
        """
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        return x