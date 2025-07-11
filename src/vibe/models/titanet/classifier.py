import torch
import torch.nn as nn
import torch.nn.functional as F


def get_nonlinear(config_str, channels):
    """
    Creates a sequential module with specified non-linear operations.
    
    Args:
        config_str: String with hyphen-separated non-linear operations.
        channels: Number of input/output channels.
        
    Returns:
        nn.Sequential: Module with specified non-linear operations.
        
    Supported operations:
        - relu: ReLU activation
        - prelu: PReLU activation with learnable parameters
        - batchnorm: BatchNorm1d with affine parameters
        - batchnorm_: BatchNorm1d without affine parameters
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


class DenseLayer(nn.Module):
    """
    A dense layer implemented as a 1D convolution followed by non-linear operations.
    
    This layer supports both 2D and 3D input tensors, automatically handling 
    the reshaping operations needed for the 1D convolution.
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
        config_str='batchnorm-relu'
    ):
        """
        Initialize a DenseLayer.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            bias: Whether to include bias parameters in the convolution.
            config_str: Configuration string for non-linear operations.
        """
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        """
        Forward pass through the DenseLayer.
        
        Args:
            x: Input tensor of shape [B, C] or [B, C, T]
            
        Returns:
            Processed tensor after linear transformation and non-linear operations.
        """
        if len(x.shape) == 2:
            # Handle 2D input by adding and then removing a dimension
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
            
        x = self.nonlinear(x)
        return x


class CosineClassifier(nn.Module):
    """
    A classifier that computes cosine similarity between features and weights.
    
    This classifier normalizes both the input features and the weight vectors
    before computing their dot product, effectively calculating cosine similarity.
    """
    
    def __init__(
        self,
        input_dim,
        num_blocks=0,
        inter_dim=512,
        out_neurons=1000,
    ):
        """
        Initialize a CosineClassifier.
        
        Args:
            input_dim: Dimension of input features.
            num_blocks: Number of DenseLayer blocks before classification.
            inter_dim: Dimension of intermediate representations.
            out_neurons: Number of output classes.
        """
        super().__init__()
        self.blocks = nn.ModuleList()

        # Add intermediate dense blocks if specified
        for _ in range(num_blocks):
            self.blocks.append(
                DenseLayer(input_dim, inter_dim, config_str='batchnorm')
            )
            input_dim = inter_dim

        # Create trainable weights for classification
        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_dim)
        )
        # Initialize with Xavier uniform distribution
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """
        Forward pass through the CosineClassifier.
        
        Args:
            x: Input feature tensor of shape [B, dim]
            
        Returns:
            Tensor of shape [B, out_neurons] containing class scores.
        """
        # Pass through intermediate blocks
        for layer in self.blocks:
            x = layer(x)

        # Compute normalized cosine similarity
        x = F.linear(F.normalize(x), F.normalize(self.weight))
        return x


class LinearClassifier(nn.Module):
    """
    A classifier that uses standard linear projection with ReLU activation.
    
    This classifier applies ReLU activation to the input, optionally passes
    the features through several dense blocks, then applies a final linear
    projection to obtain class scores.
    """
    
    def __init__(
        self,
        input_dim,
        num_blocks=0,
        inter_dim=512,
        out_neurons=1000,
    ):
        """
        Initialize a LinearClassifier.
        
        Args:
            input_dim: Dimension of input features.
            num_blocks: Number of DenseLayer blocks before classification.
            inter_dim: Dimension of intermediate representations.
            out_neurons: Number of output classes.
        """
        super().__init__()
        self.blocks = nn.ModuleList()

        # Initial ReLU activation
        self.nonlinear = nn.ReLU(inplace=True)
        
        # Add intermediate dense blocks if specified
        for _ in range(num_blocks):
            self.blocks.append(
                DenseLayer(input_dim, inter_dim, bias=True)
            )
            input_dim = inter_dim

        # Final linear classification layer
        self.linear = nn.Linear(input_dim, out_neurons, bias=True)

    def forward(self, x):
        """
        Forward pass through the LinearClassifier.
        
        Args:
            x: Input feature tensor of shape [B, dim]
            
        Returns:
            Tensor of shape [B, out_neurons] containing class scores.
        """
        # Initial activation
        x = self.nonlinear(x)
        
        # Pass through intermediate blocks
        for layer in self.blocks:
            x = layer(x)
            
        # Final linear projection
        x = self.linear(x)
        return x