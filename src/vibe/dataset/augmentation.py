import random
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import numpy as np
from scipy import signal

from ..utils import load_wav_scp


def add_reverb(wav: torch.Tensor, rir_wav: torch.Tensor) -> torch.Tensor:
    """
    Apply room impulse response (RIR) to a waveform to add reverberation.
    
    Args:
        wav: Input waveform tensor of shape [T].
        rir_wav: Room impulse response tensor of shape [T].
        
    Returns:
        Reverberated waveform tensor of shape [T].
    """
    # Convert tensors to numpy arrays for processing
    wav_np = wav.numpy()
    rir_np = rir_wav.numpy()
    
    # Get the length of the waveform
    wav_len = wav_np.shape[0]
    
    # Normalize the RIR
    rir_np = rir_np / np.sqrt(np.sum(rir_np**2))
    
    # Apply convolution and trim to original length
    out_wav = signal.convolve(wav_np, rir_np, mode='full')[:wav_len]
    
    # Normalize the output waveform
    out_wav = out_wav / (np.max(np.abs(out_wav)) + 1e-6)
    
    return torch.from_numpy(out_wav)


def add_noise(
    wav: torch.Tensor, 
    noise: Optional[torch.Tensor] = None, 
    snr_high: float = 15, 
    snr_low: float = 0
) -> torch.Tensor:
    """
    Add noise to a waveform with a specified signal-to-noise ratio.
    
    Args:
        wav: Input waveform tensor of shape [T].
        noise: Noise waveform tensor. If None, Gaussian noise will be used.
        snr_high: Maximum signal-to-noise ratio in dB.
        snr_low: Minimum signal-to-noise ratio in dB.
        
    Returns:
        Noisy waveform tensor of shape [T].
    """
    # Generate Gaussian noise if noise is not provided
    if noise is None:
        noise = torch.randn_like(wav)
        
    # Convert tensors to numpy arrays for processing
    noise_np = noise.numpy()
    wav_np = wav.numpy()

    # Get lengths of waveform and noise
    wav_len = wav_np.shape[0]
    noise_len = noise_np.shape[0]
    
    # Adjust noise length to match waveform length
    if noise_len >= wav_len:
        # If noise is longer, select a random segment
        start = random.randint(0, noise_len - wav_len)
        noise_np = noise_np[start:start + wav_len]
    else:
        # If noise is shorter, repeat it to fill the required length
        noise_np = noise_np.repeat(wav_len // noise_len + 1)
        noise_np = noise_np[:wav_len]

    # Calculate signal and noise power in dB
    wav_db = 10 * np.log10(np.mean(wav_np**2) + 1e-6)
    noise_db = 10 * np.log10(np.mean(noise_np**2) + 1e-6)
    
    # Generate random SNR within specified range
    noise_snr = random.uniform(snr_low, snr_high)
    
    # Scale noise to achieve target SNR
    noise_np = np.sqrt(10**((wav_db - noise_db - noise_snr) / 10)) * noise_np
    
    # Add noise to the waveform
    out_wav = wav_np + noise_np
    
    # Normalize the output waveform
    out_wav = out_wav / (np.max(np.abs(out_wav)) + 1e-6)
    
    return torch.from_numpy(out_wav)


class NoiseReverbCorrupter:
    """
    A class that applies noise and reverberation augmentation to audio signals.
    
    This corrupts audio signals by adding environmental noise and room reverberation
    with specified probabilities, enhancing training data variety for speech models.
    """
    
    def __init__(
        self,
        noise_prob: float = 0.0,
        reverb_prob: float = 0.0,
        noise_file: Optional[str] = None,
        reverb_file: Optional[str] = None,
        noise_snr_low: float = 0,
        noise_snr_high: float = 15,
    ):
        """
        Initialize the NoiseReverbCorrupter.
        
        Args:
            noise_prob: Probability of applying noise (0.0 to 1.0).
            reverb_prob: Probability of applying reverberation (0.0 to 1.0).
            noise_file: Path to noise.scp file containing noise audio paths.
            reverb_file: Path to rir.scp file containing room impulse response paths.
            noise_snr_low: Minimum signal-to-noise ratio in dB.
            noise_snr_high: Maximum signal-to-noise ratio in dB.
        """
        # Initialize reverberation resources if needed
        if reverb_prob > 0.0:
            if reverb_file is None:
                raise ValueError('Reverb_file must be specified when reverb_prob > 0.')
            self.add_reverb = add_reverb
            self.reverb_data = load_wav_scp(reverb_file)
            self.reverb_data_keys = list(self.reverb_data.keys())

        # Initialize noise resources if needed
        if noise_prob > 0.0:
            if noise_file is None:
                raise ValueError('Noise_file must be specified when noise_prob > 0.')
            self.add_noise = add_noise
            self.noise_data = load_wav_scp(noise_file)
            self.noise_data_keys = list(self.noise_data.keys())

        # Store configuration parameters
        self.reverb_prob = reverb_prob
        self.noise_prob = noise_prob
        self.noise_snr_low = noise_snr_low
        self.noise_snr_high = noise_snr_high

    def __call__(self, wav: torch.Tensor, fs: int = 16000) -> torch.Tensor:
        """
        Apply noise and reverberation to the input waveform based on the specified probabilities.
        
        Args:
            wav: Input waveform tensor.
            fs: Sampling frequency of the audio in Hz.
            
        Returns:
            Processed waveform tensor.
        """
        # Apply reverberation with probability reverb_prob
        if self.reverb_prob > random.random():
            # Select a random room impulse response
            reverb_key = random.choice(self.reverb_data_keys)
            reverb_path = self.reverb_data[reverb_key]
            
            # Load the impulse response
            reverb, fs_rir = torchaudio.load(reverb_path, backend='sox')
            assert fs_rir == fs, f"RIR sampling rate ({fs_rir}Hz) doesn't match input ({fs}Hz)"
            
            # Apply reverberation
            wav = self.add_reverb(wav, reverb[0])
            
        # Apply noise with probability noise_prob
        if self.noise_prob > random.random():
            # Select a random noise sample
            noise_key = random.choice(self.noise_data_keys)
            noise_path = self.noise_data[noise_key]
            
            # Load the noise audio
            noise, fs_noise = torchaudio.load(noise_path)
            assert fs_noise == fs, f"Noise sampling rate ({fs_noise}Hz) doesn't match input ({fs}Hz)"
            
            # Apply noise
            wav = self.add_noise(
                wav, 
                noise[0],
                snr_high=self.noise_snr_high,
                snr_low=self.noise_snr_low,
            )
            
        return wav