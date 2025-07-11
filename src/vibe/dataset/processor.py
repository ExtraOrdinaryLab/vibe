import json
import random
import pickle
from typing import Dict, Tuple, List, Optional, Any

import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi

from .augmentation import NoiseReverbCorrupter


class WaveformReader:
    """
    A class for reading and preprocessing audio waveforms.
    
    Reads audio files, optionally applies speed perturbation,
    and returns a fixed-duration segment.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 3.0,
        speed_pertub: bool = False,
    ):
        """
        Initialize the WavReader.
        
        Args:
            sample_rate: Audio sample rate in Hz.
            duration: Target segment duration in seconds.
            speed_pertub: Whether to apply speed perturbation.
        
        Note:
            Speed perturbation might be more appropriate in augmentation.py.
        """
        self.duration = duration
        self.sample_rate = sample_rate
        self.speed_pertub = speed_pertub

    def __call__(self, wav_path: str, start: float, stop: float) -> Tuple[torch.Tensor, int]:
        """
        Read and process an audio file.
        
        Args:
            wav_path: Path to the audio file.
            start: Start time in seconds.
            stop: Stop time in seconds.
            
        Returns:
            Tuple of (processed waveform, speed index)
        """
        # Load the audio file
        wav, sr = torchaudio.load(wav_path)
        assert sr == self.sample_rate, f"Sample rate mismatch: expected {self.sample_rate}, got {sr}"
        
        # Take the first channel if multi-channel
        wav = wav[0]
        
        # Extract the segment specified by start and stop times
        wav = wav[int(sr * start):int(sr * stop)]

        # Apply speed perturbation if enabled
        if self.speed_pertub:
            speeds = [1.0, 0.9, 1.1]  # Original, slower, faster
            speed_idx = random.randint(0, 2)
            if speed_idx > 0:  # Skip if speed_idx is 0 (no perturbation)
                wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                    wav.unsqueeze(0), 
                    sample_rate=self.sample_rate, 
                    effects=[
                        ['speed', str(speeds[speed_idx])],  # Change speed
                        ['rate', str(self.sample_rate)]     # Restore sample rate
                    ]
                )
        else:
            speed_idx = 0

        # Ensure wav is 1D
        wav = wav.squeeze(0)
        data_len = wav.shape[0]

        # Extract or pad to target duration
        chunk_len = int(self.duration * sr)
        if data_len >= chunk_len:
            # If segment is longer than needed, extract a random chunk
            start = random.randint(0, data_len - chunk_len)
            end = start + chunk_len
            wav = wav[start:end]
        else:
            # If segment is shorter than needed, pad with zeros
            wav = F.pad(wav, (0, chunk_len - data_len))

        return wav, speed_idx


class SpeakerLabelEncoder:
    """
    A class for encoding speaker labels to indices.
    
    Maps speaker IDs to numerical indices for model training.
    """
    
    def __init__(self, data_file: str):
        """
        Initialize the SpkLabelEncoder.
        
        Args:
            data_file: Path to the JSONL file containing speaker information.
        """
        self.lab2ind = {}  # Maps speaker labels to indices
        self.ind2lab = {}  # Maps indices to speaker labels
        self.starting_index = -1  # Start indexing from 0
        self.load_from_jsonl(data_file)

    def __call__(self, spk: str, speed_idx: int = 0) -> int:
        """
        Convert a speaker ID to its numerical index.
        
        Args:
            spk: Speaker identifier.
            speed_idx: Speed perturbation index (0=normal, 1=slow, 2=fast).
            
        Returns:
            Speaker index, modified by speed index if applicable.
        """
        spkid = self.lab2ind[spk]
        # Adjust the ID based on speed perturbation to create virtual speakers
        spkid = spkid + len(self.lab2ind) * speed_idx
        return spkid

    def load_from_jsonl(self, path: str) -> None:
        """
        Load speaker information from a JSONL file.
        
        Args:
            path: Path to the JSONL file containing speaker information.
        """
        with open(path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            data = json.loads(line.strip())
            # Extract speaker ID from the JSON object
            if 'spk_id' in data:
                self.add(data['spk_id'])
            elif 'spk' in data:
                self.add(data['spk'])
            else:
                raise ValueError(f"No speaker ID field found in JSONL data: {data}")

    def add(self, label: str) -> None:
        """
        Add a new speaker label to the encoder.
        
        Args:
            label: Speaker identifier.
        """
        if label in self.lab2ind:
            return  # Skip if already added
        
        index = self._next_index()
        self.lab2ind[label] = index
        self.ind2lab[index] = label

    def _next_index(self) -> int:
        """
        Get the next available index for a new speaker.
        
        Returns:
            Next index value.
        """
        self.starting_index += 1
        return self.starting_index

    def __len__(self) -> int:
        """
        Get the number of speakers in the encoder.
        
        Returns:
            Number of unique speakers.
        """
        return len(self.lab2ind)

    def save(self, path: str) -> None:
        """
        Save the speaker label mapping to a file.
        
        Args:
            path: Output file path.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.lab2ind, f)

    def load(self, path: str) -> None:
        """
        Load speaker label mapping from a file.
        
        Args:
            path: Input file path.
        """
        self.lab2ind = {}
        self.ind2lab = {}
        
        with open(path, 'rb') as f:
            self.lab2ind = pickle.load(f)
            
        # Rebuild the reverse mapping
        for label in self.lab2ind:
            self.ind2lab[self.lab2ind[label]] = label


class SpeakerVerificationAugmentation:
    """
    A class for speaker verification augmentation.
    
    Applies various audio augmentations for training speaker verification models.
    """
    
    def __init__(
        self,
        aug_prob: float = 0.0,
        noise_file: Optional[str] = None,
        reverb_file: Optional[str] = None,
    ):
        """
        Initialize the SpeakerVerificationAugmentation augmenter.
        
        Args:
            aug_prob: Probability of applying augmentation (0.0 to 1.0).
            noise_file: Path to noise.scp file for noise augmentation.
            reverb_file: Path to rir.scp file for reverberation augmentation.
        """
        self.aug_prob = aug_prob
        
        # Initialize augmentation types if augmentation is enabled
        if aug_prob > 0:
            # Noise-only augmentation
            self.add_noise = NoiseReverbCorrupter(
                noise_prob=1.0,
                noise_file=noise_file,
            )
            
            # Reverberation-only augmentation
            self.add_rir = NoiseReverbCorrupter(
                reverb_prob=1.0,
                reverb_file=reverb_file,
            )
            
            # Combined noise and reverberation augmentation
            self.add_rir_noise = NoiseReverbCorrupter(
                noise_prob=1.0,
                reverb_prob=1.0,
                noise_file=noise_file,
                reverb_file=reverb_file,
            )

            # List of available augmentation methods
            self.augmentations = [self.add_noise, self.add_rir, self.add_rir_noise]

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to a waveform.
        
        Args:
            wav: Input waveform tensor.
            
        Returns:
            Augmented waveform tensor.
        """
        sample_rate = 16000  # Fixed sample rate for augmentation
        
        # Apply augmentation with specified probability
        if self.aug_prob > random.random():
            # Randomly select an augmentation method
            aug = random.choice(self.augmentations)
            wav = aug(wav, sample_rate)

        return wav


class FBank:
    """
    A class for extracting Mel filterbank features from audio waveforms.
    
    Converts time-domain signals to frequency-domain features using Kaldi's
    implementation of filterbank analysis.
    """
    
    def __init__(
        self,
        num_mel_bins: int,
        sample_rate: int,
        normalize: bool = False,
    ):
        """
        Initialize the FBank feature extractor.
        
        Args:
            num_mel_bins: Number of Mel filterbank channels.
            sample_rate: Audio sample rate in Hz.
            normalize: Whether to apply mean normalization to features.
        """
        self.num_mel_bins = num_mel_bins
        self.sample_rate = sample_rate
        self.normalize = normalize

    def __call__(self, wav: torch.Tensor, dither: float = 0) -> torch.Tensor:
        """
        Extract Mel filterbank features from a waveform.
        
        Args:
            wav: Input waveform tensor.
            dither: Amount of dithering to apply (usually 0 for evaluation, >0 for training).
            
        Returns:
            Mel filterbank features of shape [Time, n_mels].
        """
        sr = 16000
        assert sr == self.sample_rate, f"Sample rate mismatch: expected {self.sample_rate}, got {sr}"
        
        # Ensure input is 2D [channels, samples]
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
            
        # Select first channel if multi-channel
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
            
        # Validate shape
        assert len(wav.shape) == 2 and wav.shape[0] == 1, f"Unexpected waveform shape: {wav.shape}"
        
        # Extract filterbank features
        feat = Kaldi.fbank(
            wav, 
            num_mel_bins=self.num_mel_bins,
            sample_frequency=sr, 
            dither=dither
        )
        # feat shape: [Time, num_mel_bins]
        
        # Apply mean normalization if enabled
        if self.normalize:
            feat = feat - feat.mean(0, keepdim=True)
            
        return feat