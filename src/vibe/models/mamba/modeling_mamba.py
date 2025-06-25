import math
from typing import List, Optional

from rich.console import Console

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import SpecAugment
from huggingface_hub import PyTorchModelHubMixin
from transformers.modeling_outputs import ModelOutput

from ...logging import get_logger
from .configuration_mamba import MambaConfig

logger = get_logger()


class InputNormalization(torch.nn.Module):
    """Performs sentence-level mean and variance normalization of the input tensor."""

    def __init__(self, mean_norm=True, std_norm=True, eps=1e-10):
        super().__init__()
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.eps = eps

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """Apply sentence-level normalization.

        Arguments
        ---------
        x : torch.Tensor
            A batch of input tensors with shape [batch, time, feat_dim].
        lengths : torch.Tensor
            A tensor containing the relative length of each sentence.

        Returns
        -------
        x : torch.Tensor
            The normalized tensor.
        """
        N_batches = x.shape[0]
        out = torch.empty_like(x)

        for snt_id in range(N_batches):
            actual_size = torch.round(lengths[snt_id] * x.shape[1]).int()
            x_snt = x[snt_id, 0:actual_size]

            mean, std = self._compute_current_stats(x_snt)
            out[snt_id] = (x[snt_id] - mean) / std

        return out

    def _compute_current_stats(self, x):
        """Compute mean and std for a single sentence."""
        mean = torch.mean(x, dim=0) if self.mean_norm else torch.tensor([0.0], device=x.device)
        std = torch.std(x, dim=0) if self.std_norm else torch.tensor([1.0], device=x.device)
        std = torch.max(std, self.eps * torch.ones_like(std))  # numerical stability
        return mean.detach(), std.detach()


def next_power_of_2(length):
    """
    Returns the next power of 2 above length
    """
    return 2 ** math.ceil(math.log2(length))


def pad_to_next_power_of_2(tensor):
    """
    Pads input length dim to the next power of 2

    Args:
        tensor: (B, L, D, N)

    Returns:
        padded_tensor: (B, next_power_of_2(L), D, N)
    """
    len_next_power_of_2 = next_power_of_2(tensor.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_next_power_of_2 - tensor.size(1))
    return F.pad(tensor, pad_tuple, "constant", 0)


class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A: (B, D, L, N)
        # X: (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values:
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        # only supports L that is a power of two (mainly for a clearer code)
        
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps-2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        Xa = X[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 2**k-1:L:2**k]
            Xa = X[:, :, 2**k-1:L:2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        # A: (B, D, L, N)
        # X: (B, D, L, N)

        # the same function as above, but in reverse
        # (if you flip the input, call pscan, then flip the output, you get what this function outputs)
        # it is used in the backward pass

        # only supports L that is a power of two (mainly for a clearer code)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps-2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
                    
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        # we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 0:L:2**(num_steps-2)]
        Xa = X[:, :, 0:L:2**(num_steps-2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 0:L:2**k]
            Xa = X[:, :, 0:L:2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.
        If you can, privilege sequence lengths that are powers of two.

        Args:
            A_in: (B, L, D, N)
            X_in: (B, L, D, N)

        Returns:
            H: (B, L, D, N)
        """
        L = X_in.size(1)

        # cloning is required because of the in-place ops
        if L == next_power_of_2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            # pad tensors (and clone btw)
            A = pad_to_next_power_of_2(A_in)  # (B, next_power_of_2(L), D, N)
            X = pad_to_next_power_of_2(X_in)  # (B, next_power_of_2(L), D, N)
        
        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, next_power_of_2(L), N)
        X = X.transpose(2, 1)  # (B, D, next_power_of_2(L), N)

        # parallel scan (modifies X in-place)
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)
        
        # slice [:, :L] (cut if there was padding)
        return X.transpose(2, 1)[:, :L]
    
    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx: A_in: (B, L, D, N), X: (B, D, L, N)
            grad_output_in: (B, L, D, N)

        Returns:
            grad_A: (B, L, D, N), grad_X: (B, L, D, N)
        """
        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        # cloning is required because of the in-place ops
        if L == next_power_of_2(L):
            grad_output = grad_output_in.clone()
            # the next padding will clone A_in
        else:
            grad_output = pad_to_next_power_of_2(grad_output_in)  # (B, next_power_of_2(L), D, N)
            A_in = pad_to_next_power_of_2(A_in)  # (B, next_power_of_2(L), D, N)

        # prepare tensors
        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1)  # (B, D, next_power_of_2(L), N)
        A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1))  # (B, D, next_power_of_2(L), N) shift 1 to the left (see hand derivation)

        # reverse parallel scan (modifies grad_output in-place)
        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]


# Create a function to apply PScan
pscan = PScan.apply


class BatchNorm1d(nn.Module):
    """Wrapper for BatchNorm1d with appropriate shape handling for speaker recognition."""
    
    def __init__(self, input_size):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size)
        
    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor
            Input to normalize with shape (batch, time, channels) or (batch, channels, time).
        """
        shape_or = x.shape
        if len(shape_or) == 3 and shape_or[1] != self.norm.num_features:
            x = x.transpose(1, 2)
            
        x_n = self.norm(x)
        
        if len(shape_or) == 3 and shape_or[1] != self.norm.num_features:
            x_n = x_n.transpose(1, 2)
            
        return x_n


class TDNNBlock(nn.Module):
    """Time-Delay Neural Network block.
    
    Arguments
    ---------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Kernel size of the convolutional layers.
    dilation : int
        Dilation factor of the convolution.
    activation : torch.nn.Module
        Activation function.
    groups : int
        Number of groups in the convolution.
    dropout : float
        Dropout rate.
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        activation=nn.ReLU,
        groups=1,
        dropout=0.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, lengths=None):
        """
        Arguments
        ---------
        x : torch.Tensor
            Input tensor with shape (batch, time, channels) or (batch, channels, time).
        lengths : torch.Tensor
            Length of each sequence for proper masking.
        
        Returns
        -------
        x : torch.Tensor
            Output tensor with shape (batch, channels, time).
        """
        # Handle 3D inputs (batch, time, channels)
        if x.ndim == 3 and x.shape[1] != self.conv.in_channels:
            x = x.transpose(1, 2)
            
        x = self.conv(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling layer for speaker embeddings.
    
    Arguments
    ---------
    channels : int
        Number of input channels.
    attention_channels : int
        Number of attention channels.
    global_context : bool
        Whether to include global context in attention.
    """
    
    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()
        self.eps = 1e-12
        self.global_context = global_context
        
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
            
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(
            in_channels=attention_channels, 
            out_channels=channels, 
            kernel_size=1
        )
        
    def forward(self, x, lengths=None):
        """
        Arguments
        ---------
        x : torch.Tensor
            Input tensor with shape (batch, channels, time).
        lengths : torch.Tensor
            Length of each sequence for proper masking.
            
        Returns
        -------
        x : torch.Tensor
            Output tensor with shape (batch, channels*2).
        """
        # Create masks if lengths is provided
        if lengths is not None:
            mask = length_to_mask(lengths, max_len=x.size(2), device=x.device)
            mask = mask.unsqueeze(1)
        else:
            mask = torch.ones(
                x.size(0), 1, x.size(2), device=x.device
            ).float()
            
        # Compute required statistics
        mean = torch.sum(x * mask, dim=2) / (torch.sum(mask, dim=2) + self.eps)
        std = torch.sqrt(
            torch.sum(((x - mean.unsqueeze(2)) ** 2) * mask, dim=2) / 
            (torch.sum(mask, dim=2) + self.eps)
        )
        mean = mean.unsqueeze(2).repeat(1, 1, x.size(2))
        std = std.unsqueeze(2).repeat(1, 1, x.size(2))
        
        # Compute attention
        if self.global_context:
            # Concatenate mean, std, and input
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x
            
        attn = self.tdnn(attn)
        attn = self.tanh(attn)
        attn = self.conv(attn)
        attn = F.softmax(attn, dim=2)
        
        # Apply attention to input
        mean = torch.sum(x * attn, dim=2)
        std = torch.sqrt(
            torch.sum(((x - mean.unsqueeze(2)) ** 2) * attn, dim=2)
        )
        
        # Concatenate mean and std
        pooled = torch.cat([mean, std], dim=1)
        return pooled


def length_to_mask(length, max_len=None, device=None):
    """Creates a binary mask for each sequence in a batch.
    
    Arguments
    ---------
    length : torch.Tensor
        Containing the length of each sequence in the batch.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    device : torch.device
        Device of the mask tensor.
        
    Returns
    -------
    mask : torch.Tensor
        The binary mask.
    """
    if max_len is None:
        max_len = length.max().long().item()
        
    mask = torch.arange(
        max_len, device=device if device else length.device
    ).expand(len(length), max_len) < length.unsqueeze(1)
    
    return mask


class MambaBlock(nn.Module):
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        # Projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        # Depthwise convolution over time
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1
        )
        
        # Projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # Projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True  # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo: explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # why store A in log? to keep A < 0 (cf -torch.exp(...))? for gradient stability?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # Projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # Used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps, config.mup)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

    def _apply_layernorms(self, dt, B, C):
        """Apply layer normalization to internal activations if configured"""
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L, D)
        
        Returns:
            Output tensor of shape (B, L, D)
        """
        _, seq_len, _ = x.shape

        # Split into two branches
        xz = self.in_proj(x)  # (B, L, 2*ED)
        x_branch, z_branch = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)

        # x branch: Apply 1D convolution
        x_branch = x_branch.transpose(1, 2)  # (B, ED, L)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]  # depthwise convolution over time, with a short filter
        x_branch = x_branch.transpose(1, 2)  # (B, L, ED)

        x_branch = F.silu(x_branch)
        y = self.ssm(x_branch, z_branch)

        # z branch: Apply activation
        z_branch = F.silu(z_branch)

        # Combine branches and project back to model dimension
        output = y * z_branch
        output = self.out_proj(output)  # (B, L, D)

        return output
    
    def ssm(self, x, z):
        """
        Apply the state space model computation
        
        Args:
            x: Input tensor of shape (B, L, ED)
            z: Second branch tensor (not used directly here)
        
        Returns:
            Output tensor of shape (B, L, ED)
        """
        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        # Project x to get delta, B, C parameters
        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(
            deltaBC, 
            [self.config.dt_rank, self.config.d_state, self.config.d_state], 
            dim=-1
        )  # (B, L, dt_rank), (B, L, N), (B, L, N)
        
        # Apply layer normalization if configured
        delta, B, C = self._apply_layernorms(delta, B, C)
        
        # Project delta
        delta = self.dt_proj.weight @ delta.transpose(1, 2)  # (ED, dt_rank) @ (B, dt_rank, L) -> (B, ED, L)
        # Here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # The rest will be applied later (fused if using cuda)
        
        delta = delta.transpose(1, 2)
        delta = F.softplus(delta + self.dt_proj.bias)

        # Choose between parallel or sequential scan implementation
        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        """
        Parallel scan implementation of selective SSM
        
        Args:
            x: Input tensor of shape (B, L, ED)
            delta: Delta tensor of shape (B, L, ED)
            A: A tensor of shape (ED, N)
            B: B tensor of shape (B, L, N)
            C: C tensor of shape (B, L, N)
            D: D tensor of shape (ED)
        
        Returns:
            Output tensor of shape (B, L, ED)
        """
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        """
        Sequential scan implementation of selective SSM
        
        Args:
            x: Input tensor of shape (B, L, ED)
            delta: Delta tensor of shape (B, L, ED)
            A: A tensor of shape (ED, N)
            B: B tensor of shape (B, L, N)
            C: C tensor of shape (B, L, N)
            D: D tensor of shape (ED)
        
        Returns:
            Output tensor of shape (B, L, ED)
        """
        batch_size, seq_len, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        h = torch.zeros(batch_size, self.config.d_inner, self.config.d_state, device=deltaA.device)  # (B, ED, N)
        hidden_states = []

        for t in range(0, seq_len):
            h = deltaA[:, t] * h + BX[:, t]
            hidden_states.append(h)
            
        hidden_states = torch.stack(hidden_states, dim=1)  # (B, L, ED, N)

        y = (hidden_states @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
        y = y + D * x

        return y
    
    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba: inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things:
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, also outputs the updated cache for the next call.

    The cache object is initialized as follows: (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """
    
    def step(self, x, cache):
        """
        Single step forward for autoregressive inference
        
        Args:
            x: Input tensor of shape (B, D)
            cache: Tuple (h, inputs)
                  h: Hidden state tensor of shape (B, ED, N)
                  inputs: Previous inputs of shape (B, ED, d_conv-1)
        
        Returns:
            output: Output tensor of shape (B, D)
            updated_cache: Updated cache for next step
        """
        h, inputs = cache
        
        # Split into two branches
        xz = self.in_proj(x)  # (B, 2*ED)
        x_branch, z_branch = xz.chunk(2, dim=1)  # (B, ED), (B, ED)

        # x branch: Apply 1D convolution using cached inputs
        x_cache = x_branch.unsqueeze(2)
        x_branch = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1]  # (B, ED)

        x_branch = F.silu(x_branch)
        y, h = self.ssm_step(x_branch, h)

        # z branch: Apply activation
        z_branch = F.silu(z_branch)

        # Combine branches and project back to model dimension
        output = y * z_branch
        output = self.out_proj(output)  # (B, D)

        # Update cache for next step
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)  # (B, ED, d_conv-1)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        """
        Single step SSM computation for autoregressive inference
        
        Args:
            x: Input tensor of shape (B, ED)
            h: Hidden state tensor of shape (B, ED, N) or None
        
        Returns:
            y: Output tensor of shape (B, ED)
            updated_h: Updated hidden state for next step
        """
        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        # Project x to get delta, B, C parameters
        deltaBC = self.x_proj(x)  # (B, dt_rank+2*N)
        delta, B, C = torch.split(
            deltaBC, 
            [self.config.dt_rank, self.config.d_state, self.config.d_state], 
            dim=-1
        )  # (B, dt_rank), (B, N), (B, N)
        
        # Apply layer normalization if configured
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta))  # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, ED, N)

        # Initialize hidden state if None
        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  # (B, ED, N)

        # Update hidden state
        h = deltaA * h + BX  # (B, ED, N)

        # Compute output
        y = (h @ C.unsqueeze(-1)).squeeze(2)  # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)
        y = y + D * x

        return y, h


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()
        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        Apply RMS normalization
        
        Args:
            x: Input tensor
        
        Returns:
            Normalized tensor
        """
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output


class ResidualBlock(nn.Module):

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L, D)
        
        Returns:
            Output tensor of shape (B, L, D)
        """
        output = self.mixer(self.norm(x)) + x
        return output
    
    def step(self, x, cache):
        """
        Single step forward for autoregressive inference
        
        Args:
            x: Input tensor of shape (B, D)
            cache: Tuple (h, inputs)
                  h: Hidden state tensor of shape (B, ED, N)
                  inputs: Previous inputs of shape (B, ED, d_conv-1)
        
        Returns:
            output: Output tensor of shape (B, D)
            updated_cache: Updated cache for next step
        """
        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class Mamba(nn.Module):

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L, D)
        
        Returns:
            Output tensor of shape (B, L, D)
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def step(self, x, caches):
        """
        Single step forward for autoregressive inference
        
        Args:
            x: Input tensor of shape (B, L, D)
            caches: List of cache tuples for all layers, each cache: (h, inputs)
        
        Returns:
            output: Output tensor of shape (B, L, D)
            updated_caches: Updated caches for next step
        """
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])
        return x, caches


class MambaEncoder(nn.Module):
    """
    Mamba-based Speaker Recognition Model
    
    This model combines a TDNN front-end for feature extraction with a Mamba model
    for sequence modeling and an attentive statistics pooling layer for aggregation.
    
    Arguments
    ---------
    input_size : int
        Number of input features (mel bins).
    d_model : int
        Dimension of the model.
    n_layers : int
        Number of Mamba layers.
    attention_channels : int
        Number of attention channels in pooling.
    lin_neurons : int
        Number of neurons in the final linear layer.
    """
    
    def __init__(
        self,
        input_size=80,
        d_model=512,
        n_layers=6,
        d_state=16,
        expand_factor=2,
        d_conv=4,
        attention_channels=128,
        lin_neurons=192,
    ):
        super().__init__()
        
        # Initial TDNN layer to map input features to model dimension
        self.tdnn = TDNNBlock(
            in_channels=input_size,
            out_channels=d_model,
            kernel_size=5,
            dilation=1,
        )
        
        # Mamba configuration
        mamba_config = MambaConfig(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv,
        )
        
        # Mamba sequence model
        self.mamba = Mamba(mamba_config)
        
        # Attentive Statistics Pooling
        self.asp = AttentiveStatisticsPooling(
            channels=d_model,
            attention_channels=attention_channels,
        )
        
        # Batch norm after pooling
        self.norm = BatchNorm1d(input_size=d_model * 2)
        
        # Final linear layer
        self.fc = nn.Linear(d_model * 2, lin_neurons)
        
    def forward(self, x, lengths=None):
        """
        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, features).
        lengths : torch.Tensor
            Length of each sequence.
            
        Returns
        -------
        embeddings : torch.Tensor
            Speaker embeddings.
        """
        # Initial feature extraction with TDNN
        x = self.tdnn(x)  # (B, d_model, T)
        
        # Transpose for Mamba (expects B, T, d_model)
        x = x.transpose(1, 2)
        
        # Apply Mamba sequence model
        x = self.mamba(x)  # (B, T, d_model)
        
        # Transpose back for pooling (expects B, d_model, T)
        x = x.transpose(1, 2)
        
        # Apply attentive statistics pooling
        x = self.asp(x, lengths)  # (B, d_model*2)
        
        # Apply batch normalization
        x = self.norm(x)
        
        # Final linear projection
        embeddings = self.fc(x)
        
        return embeddings
    

class Linear(torch.nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape : tuple
        It is the shape of the input tensor.
    input_size : int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    max_norm : float
        weight max-norm.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        bias=True,
        max_norm=None,
        combine_dims=False,
    ):
        super().__init__()
        self.max_norm = max_norm
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]

        # Weights are initialized following pytorch approach
        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly.

        Returns
        -------
        wx : torch.Tensor
            The linearly transformed outputs.
        """
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        if self.max_norm is not None:
            self.w.weight.data = torch.renorm(
                self.w.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )

        wx = self.w(x)

        return wx


class Classifier(torch.nn.Module):
    """This class implements the cosine similarity on the top of features.

    Arguments
    ---------
    input_size : int
        Expected size of input dimension.
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of classes.

    Example
    -------
    >>> classify = Classifier(input_size=2, lin_neurons=2, out_neurons=2)
    >>> outputs = torch.tensor([ [1., -1.], [-9., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outputs = outputs.unsqueeze(1)
    >>> cos = classify(outputs)
    >>> (cos < -1.0).long().sum()
    tensor(0)
    >>> (cos > 1.0).long().sum()
    tensor(0)
    """

    def __init__(
        self,
        input_size,
        device="cpu",
        lin_blocks=0,
        lin_neurons=192,
        out_neurons=1211,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    BatchNorm1d(input_size=input_size),
                    Linear(input_size=input_size, n_neurons=lin_neurons),
                ]
            )
            input_size = lin_neurons

        # Final Layer
        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_size, device=device)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """Returns the output probabilities over speakers.

        Arguments
        ---------
        x : torch.Tensor
            Torch tensor.

        Returns
        -------
        out : torch.Tensor
            Output probabilities over speakers.
        """
        for layer in self.blocks:
            x = layer(x)

        # Need to be normalized
        x = F.linear(F.normalize(x.squeeze(1)), F.normalize(self.weight))
        return x.unsqueeze(1)


class AngularMargin(nn.Module):
    """
    An implementation of Angular Margin (AM) proposed in the following
    paper: '''Margin Matters: Towards More Discriminative Deep Neural Network
    Embeddings for Speaker Recognition''' (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similarity
    scale : float
        The scale for cosine similarity

    Example
    -------
    >>> pred = AngularMargin()
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> targets = torch.tensor([ [1., 0.], [0., 1.], [ 1., 0.], [0.,  1.] ])
    >>> predictions = pred(outputs, targets)
    >>> predictions[:,0] > predictions[:,1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, outputs, targets):
        """Compute AM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Returns
        -------
        predictions : torch.Tensor
        """
        outputs = outputs - self.margin * targets
        return self.scale * outputs


class AdditiveAngularMargin(AngularMargin):
    """
    An implementation of Additive Angular Margin (AAM) proposed
    in the following paper: '''Margin Matters: Towards More Discriminative Deep
    Neural Network Embeddings for Speaker Recognition'''
    (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similarity.
    scale : float
        The scale for cosine similarity.
    easy_margin : bool

    Example
    -------
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> targets = torch.tensor([ [1., 0.], [0., 1.], [ 1., 0.], [0.,  1.] ])
    >>> pred = AdditiveAngularMargin()
    >>> predictions = pred(outputs, targets)
    >>> predictions[:,0] > predictions[:,1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0, easy_margin=False):
        super().__init__(margin, scale)
        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, outputs, targets):
        """
        Compute AAM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Returns
        -------
        predictions : torch.Tensor
        """
        cosine = outputs.float()
        cosine = torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        return self.scale * outputs


class LogSoftmaxWrapper(nn.Module):
    """
    Arguments
    ---------
    loss_fn : Callable
        The LogSoftmax function to wrap.

    Example
    -------
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outputs = outputs.unsqueeze(1)
    >>> targets = torch.tensor([ [0], [1], [0], [1] ])
    >>> log_prob = LogSoftmaxWrapper(nn.Identity())
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    >>> log_prob = LogSoftmaxWrapper(AngularMargin(margin=0.2, scale=32))
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> log_prob = LogSoftmaxWrapper(AdditiveAngularMargin(margin=0.3, scale=32))
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    """

    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
        self.criterion = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets, length=None):
        """
        Arguments
        ---------
        outputs : torch.Tensor
            Network output tensor, of shape
            [batch, 1, outdim].
        targets : torch.Tensor
            Target tensor, of shape [batch, 1].
        length : torch.Tensor
            The lengths of the corresponding inputs.

        Returns
        -------
        loss: torch.Tensor
            Loss for current examples.
        """
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)
        targets = F.one_hot(targets.long(), outputs.shape[1]).float()
        try:
            predictions = self.loss_fn(outputs, targets)
        except TypeError:
            predictions = self.loss_fn(outputs)

        predictions = F.log_softmax(predictions, dim=1)
        loss = self.criterion(predictions, targets) / targets.sum()
        return loss


class MambaForSpeakerClassification(nn.Module, PyTorchModelHubMixin):

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.num_classes = config.num_labels
        self.mean_var_norm = InputNormalization(std_norm=False)
        self.encoder = MambaEncoder(
            input_size=config.num_mel_bins, 
            d_model=config.d_model,
            n_layers=config.n_layers,
            d_state=config.d_state,
            expand_factor=config.expand_factor,
            d_conv=config.d_conv,
            attention_channels=config.attention_channels,
            lin_neurons=config.emb_sizes
        )
        self.classifier = Classifier(
            input_size=config.emb_sizes, 
            lin_blocks=0, 
            lin_neurons=config.emb_sizes, 
            out_neurons=self.num_classes
        )
        self.loss_fn = LogSoftmaxWrapper(
            AdditiveAngularMargin(config.margin, config.scale)
        )

    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        B, C, T = input_features.shape
        # Input tensor should be (B, T, C) for InputNormalization
        input_features = self.mean_var_norm(
            input_features.transpose(1, 2), torch.ones(B).to(input_features.device)
        )
        if self.training:
            spec_augment = SpecAugment(
                n_time_masks=self.config.num_time_masks, 
                time_mask_param=self.config.time_mask_width, 
                n_freq_masks=self.config.num_freq_masks, 
                freq_mask_param=self.config.freq_mask_width, 
                zero_masking=True
            )
            # Input tensor should be (B, C, T) for SpecAugment
            input_features = spec_augment(input_features.transpose(1, 2))
            input_features = input_features.transpose(1, 2)
        embeddings = self.encoder(input_features)
        logits = self.classifier(embeddings.view(B, 1, -1))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(B, 1, self.num_classes), labels.view(-1, 1))

        return ModelOutput(
            loss=loss,
            logits=logits.view(B, self.num_classes),
            embeddings=embeddings.view(B, self.config.emb_sizes)
        )