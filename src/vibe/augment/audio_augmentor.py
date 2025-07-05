import os
import glob
import random
import subprocess
import tempfile
import platform
from typing import List, Tuple, Optional, Union, Dict, Any

import numpy as np
import soundfile as sf

from ..logging import get_logger

logger = get_logger()


class AudioAugmentor:
    """
    Audio augmentation class that applies noise addition, reverberation,
    and SoX/FFmpeg-based audio effects.
    
    This class can add background noise, music, speech, room impulse responses (RIR),
    as well as apply tempo changes, gain adjustments, fade effects, and audio compression
    to make training more robust.
    
    Args:
        musan_path: Path to MUSAN dataset containing noise, music, and speech
        rir_path: Path to RIR dataset containing room impulse responses
        noise_prob: Probability of applying noise augmentation
        music_prob: Probability of applying music augmentation
        speech_prob: Probability of applying speech augmentation
        rir_prob: Probability of applying RIR augmentation
        tempo_prob: Probability of applying tempo changes
        gain_prob: Probability of applying gain adjustments
        fade_prob: Probability of applying fade effects
        compression_prob: Probability of applying audio compression
        noise_snr_range: Range of SNR values for noise (in dB)
        music_snr_range: Range of SNR values for music (in dB)
        speech_snr_range: Range of SNR values for speech (in dB)
        tempo_range: Range of tempo stretch factors (e.g., [0.8, 1.2])
        gain_range: Range of gain adjustments in dB (e.g., [-5, 5])
        fade_range: Range of fade duration in seconds (e.g., [0.1, 1.0])
        sample_rate: Sample rate of the audio signals
        auto_install: Whether to attempt installation of missing tools
        random_single_method: If True, randomly select only one augmentation method per call
    """
    
    def __init__(
        self, 
        musan_path=None, 
        rir_path=None, 
        noise_prob=0.6,
        music_prob=0.3, 
        speech_prob=0.3, 
        rir_prob=0.6,
        tempo_prob=0.3,
        gain_prob=0.3,
        fade_prob=0.2,
        compression_prob=0.3,
        noise_snr_range=(0, 15),
        music_snr_range=(5, 15),
        speech_snr_range=(13, 20),
        tempo_range=(0.8, 1.2),
        gain_range=(-5, 5),
        fade_range=(0.1, 1.0),
        sample_rate=16000,
        auto_install=True,
        random_single_method=False,
    ):
        self.sample_rate = sample_rate
        self.noise_prob = noise_prob
        self.music_prob = music_prob
        self.speech_prob = speech_prob
        self.rir_prob = rir_prob
        self.tempo_prob = tempo_prob
        self.gain_prob = gain_prob
        self.fade_prob = fade_prob
        self.compression_prob = compression_prob
        self.noise_snr_range = noise_snr_range
        self.music_snr_range = music_snr_range
        self.speech_snr_range = speech_snr_range
        self.tempo_range = tempo_range
        self.gain_range = gain_range
        self.fade_range = fade_range
        self.auto_install = auto_install
        self.random_single_method = random_single_method
        
        # Check if external tools are available
        self.sox_available = self._check_tool_available("sox", auto_install)
        self.ffmpeg_available = self._check_tool_available("ffmpeg", auto_install)
        
        # Load MUSAN data
        self.noises = []
        self.musics = []
        self.speeches = []
        if musan_path is not None and os.path.exists(musan_path):
            self._load_musan(musan_path)
        
        # Load RIR data
        self.rirs = []
        if rir_path is not None and os.path.exists(rir_path):
            self._load_rir(rir_path)
        
        # Define available augmentation methods for random selection
        self._define_augmentation_methods()
    
    def _define_augmentation_methods(self):
        """Define all available augmentation methods with their conditions."""
        self.augmentation_methods = []
        
        # Add RIR method if RIRs are available
        if self.rirs:
            self.augmentation_methods.append(('rir', self.rir_prob))
        
        # Add SoX-based methods if SoX is available
        if self.sox_available:
            self.augmentation_methods.extend([
                ('tempo', self.tempo_prob),
                ('gain', self.gain_prob),
                ('fade', self.fade_prob)
            ])
        
        # Add compression method if FFmpeg is available
        if self.ffmpeg_available:
            self.augmentation_methods.append(('compression', self.compression_prob))
        
        # Add noise-based methods if datasets are available
        if self.noises:
            self.augmentation_methods.append(('noise', self.noise_prob))
        if self.musics:
            self.augmentation_methods.append(('music', self.music_prob))
        if self.speeches:
            self.augmentation_methods.append(('speech', self.speech_prob))
    
    def _check_tool_available(self, tool: str, try_install: bool = False) -> bool:
        """
        Check if an external tool (sox, ffmpeg) is available on the system.
        Optionally tries to install the tool if not available.
        
        Args:
            tool: Name of the tool to check
            try_install: Whether to attempt installation if not available
            
        Returns:
            True if tool is available or successfully installed, False otherwise
        """
        try:
            subprocess.run([tool, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            logger.log(f"Found {tool} installed on the system.")
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.log(f"Warning: {tool} not found.")
            
            if try_install and platform.system() == "Linux":
                try:
                    logger.log(f"Attempting to install {tool}...")
                    
                    # Update package lists
                    subprocess.run(["sudo", "apt", "update"], check=True)
                    
                    # Install the required tool
                    subprocess.run(["sudo", "apt", "install", "-y", tool], check=True)
                    
                    # Verify installation
                    subprocess.run([tool, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    logger.log(f"Successfully installed {tool}.")
                    return True
                except Exception as e:
                    logger.log(f"Failed to install {tool}: {e}")
                    logger.log(f"{tool} augmentations will be disabled.")
                    return False
            else:
                logger.log(f"{tool} augmentations will be disabled.")
                return False
    
    def _load_musan(self, musan_path: str) -> None:
        """
        Load MUSAN dataset files (noise, music, speech)
        
        Args:
            musan_path: Path to MUSAN dataset
        """
        if not os.path.exists(musan_path):
            raise FileNotFoundError(
                'Please download MUSAN from `https://www.openslr.org/17`, '
                'and run command `tar zxvf musan.tar.gz`'
            )
        noise_path = os.path.join(musan_path, 'noise')
        if os.path.exists(noise_path):
            _, noise_wavs = fast_scandir(noise_path, extensions=['.wav'], recursive=True)
            self.noises.extend(noise_wavs)
        
        music_path = os.path.join(musan_path, 'music')
        if os.path.exists(music_path):
            _, music_wavs = fast_scandir(music_path, extensions=['.wav'], recursive=True)
            self.musics.extend(music_wavs)
        
        speech_path = os.path.join(musan_path, 'speech')
        if os.path.exists(speech_path):
            _, speech_wavs = fast_scandir(speech_path, extensions=['.wav'], recursive=True)
            self.speeches.extend(speech_wavs)
    
    def _load_rir(self, rir_path: str) -> None:
        """
        Load Room Impulse Response files
        
        Args:
            rir_path: Path to RIR dataset
        """
        if not os.path.exists(rir_path):
            raise FileNotFoundError(
                'Please download MUSAN from `https://www.openslr.org/28`, '
                'and run command `unzip rirs_noises.zip`'
            )
        _, self.rirs = fast_scandir(
            rir_path, extensions=['.wav'], recursive=True
        )
    
    def _compute_amplitude(self, audio: np.ndarray) -> float:
        """
        Compute RMS amplitude of audio signal
        
        Args:
            audio: Audio signal
            
        Returns:
            RMS amplitude value
        """
        return np.sqrt(np.mean(audio ** 2))
    
    def _add_noise(self, audio: np.ndarray, noise_file: str, snr: float) -> np.ndarray:
        """
        Add noise to audio signal with specified SNR
        
        Args:
            audio: Clean audio signal
            noise_file: Path to noise file
            snr: Signal-to-Noise Ratio in dB
            
        Returns:
            Noisy audio signal
        """
        try:
            # Load noise file
            noise, sr = sf.read(noise_file)
            
            # Ensure noise is mono
            if len(noise.shape) > 1:
                noise = noise[:, 0]
            
            # Repeat or trim noise to match audio length
            if len(noise) < len(audio):
                noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
                noise = noise[:len(audio)]
            
            if len(noise) > len(audio):
                start = random.randint(0, len(noise) - len(audio))
                noise = noise[start:start + len(audio)]
            
            # Calculate amplitudes for SNR scaling
            clean_amp = self._compute_amplitude(audio)
            noise_amp = self._compute_amplitude(noise)
            
            # Avoid division by zero
            if noise_amp < 1e-10:
                return audio
                
            # Calculate gain for target SNR
            snr_linear = 10 ** (snr / 10)
            gain = clean_amp / (noise_amp * snr_linear)
            
            # Apply gain and add noise
            return audio + gain * noise
            
        except Exception as e:
            logger.log(f"Warning: Error adding noise from {noise_file}: {e}")
            return audio
    
    def _add_rir(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply room impulse response to audio signal
        
        Args:
            audio: Clean audio signal
            
        Returns:
            Reverberated audio signal
        """
        if not self.rirs:
            return audio
            
        try:
            # Select random RIR file
            rir_file = random.choice(self.rirs)
            
            # Load RIR file
            rir, sr = sf.read(rir_file)
            
            # Ensure RIR is mono
            if len(rir.shape) > 1:
                rir = rir[:, 0]
            
            # Normalize RIR
            rir = rir / np.max(np.abs(rir))
            
            # Limit RIR length to reduce computation
            max_rir_len = int(self.sample_rate * 0.5)
            if len(rir) > max_rir_len:
                rir = rir[:max_rir_len]
            
            # Ensure arrays are contiguous and correct type
            audio_cont = np.ascontiguousarray(audio, dtype=np.float64)
            rir_cont = np.ascontiguousarray(rir, dtype=np.float64)
            
            # Apply convolution for reverberation
            augmented_audio = np.convolve(audio_cont, rir_cont, mode='full')[:len(audio)]
            
            # Normalize output amplitude
            if np.max(np.abs(augmented_audio)) > 0:
                max_orig = np.max(np.abs(audio))
                max_augmented = np.max(np.abs(augmented_audio))
                augmented_audio = augmented_audio * (max_orig / max_augmented)
                
            return augmented_audio
            
        except Exception as e:
            rir_file_name = rir_file if 'rir_file' in locals() else 'unknown'
            logger.log(f"Warning: Error applying RIR from {rir_file_name}: {e}")
            return audio
    
    def _apply_sox_effects(self, audio: np.ndarray, effects: List[str]) -> np.ndarray:
        """
        Apply multiple SoX effects to audio (tempo, gain, fade, etc.)
        
        Args:
            audio: Input audio signal
            effects: List of SoX effect strings to apply
            
        Returns:
            Processed audio signal
        """
        if not self.sox_available or not effects:
            return audio
            
        try:
            # Create temporary files for input and output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as in_file, \
                 tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as out_file:
                in_path = in_file.name
                out_path = out_file.name
            
            # Write input audio to temporary file
            sf.write(in_path, audio, self.sample_rate)
            
            # Prepare SoX command with all effects
            cmd = ['sox', in_path, out_path] + effects
            
            # Run SoX command
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Read the processed audio
            modified_audio, _ = sf.read(out_path)
            
            # Clean up temporary files
            os.unlink(in_path)
            os.unlink(out_path)
            
            # Ensure output has the correct length
            if len(modified_audio) < len(audio):
                # Pad with zeros if output is shorter
                padding = len(audio) - len(modified_audio)
                modified_audio = np.pad(modified_audio, (0, padding))
            elif len(modified_audio) > len(audio):
                # Truncate if output is longer
                modified_audio = modified_audio[:len(audio)]
                
            return modified_audio
            
        except Exception as e:
            logger.log(f"Warning: Error applying SoX effects {effects}: {e}")
            return audio
    
    def _apply_compression(self, audio: np.ndarray, codec: str) -> np.ndarray:
        """
        Apply audio compression using FFmpeg
        
        Args:
            audio: Input audio signal
            codec: Compression codec ('opus' or 'aac')
            
        Returns:
            Compressed audio signal
        """
        if not self.ffmpeg_available:
            return audio
            
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as in_file, \
                 tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as out_file:
                in_path = in_file.name
                out_path = out_file.name
            
            # Write input audio to temporary file
            sf.write(in_path, audio, self.sample_rate)
            
            # Set codec-specific parameters
            if codec == 'opus':
                bitrate = random.choice(['6k', '12k', '24k'])
                format_params = ['-c:a', 'libopus', '-b:a', bitrate]
            else:  # aac
                bitrate = random.choice(['8k', '16k', '32k'])
                format_params = ['-c:a', 'aac', '-b:a', bitrate]
            
            # Apply compression with FFmpeg (via intermediate file)
            temp_compressed = tempfile.NamedTemporaryFile(suffix=f'.{codec}', delete=False).name
            
            # Compress
            compress_cmd = [
                'ffmpeg', 
                '-hide_banner', 
                '-loglevel', 'error', 
                '-y', '-i', in_path
            ] + format_params + [temp_compressed]
            subprocess.run(compress_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Decompress
            decompress_cmd = [
                'ffmpeg', 
                '-hide_banner', 
                '-loglevel', 'error', 
                '-y', '-i', temp_compressed, out_path
            ]
            subprocess.run(decompress_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Read the processed audio
            compressed_audio, _ = sf.read(out_path)
            
            # Clean up temporary files
            os.unlink(in_path)
            os.unlink(out_path)
            os.unlink(temp_compressed)
            
            # Match original length
            if len(compressed_audio) < len(audio):
                compressed_audio = np.pad(compressed_audio, (0, len(audio) - len(compressed_audio)))
            elif len(compressed_audio) > len(audio):
                compressed_audio = compressed_audio[:len(audio)]
                
            return compressed_audio
            
        except Exception as e:
            logger.log(f"Warning: Error applying {codec} compression with FFmpeg: {e}")
            return audio
    
    def _apply_single_augmentation(self, audio: np.ndarray, method: str) -> np.ndarray:
        """
        Apply a single augmentation method to the audio.
        
        Args:
            audio: Input audio signal
            method: Name of the augmentation method to apply
            
        Returns:
            Augmented audio signal
        """
        if method == 'rir':
            return self._add_rir(audio)
        
        elif method == 'tempo':
            tempo_factor = random.uniform(self.tempo_range[0], self.tempo_range[1])
            sox_effects = ['stretch', str(tempo_factor)]
            return self._apply_sox_effects(audio, sox_effects)
        
        elif method == 'gain':
            gain_db = random.uniform(self.gain_range[0], self.gain_range[1])
            sox_effects = ['gain', str(gain_db)]
            return self._apply_sox_effects(audio, sox_effects)
        
        elif method == 'fade':
            fade_duration = random.uniform(self.fade_range[0], self.fade_range[1])
            sox_effects = ['fade', 'p', '0', str(fade_duration)]
            return self._apply_sox_effects(audio, sox_effects)
        
        elif method == 'compression':
            codec = random.choice(['opus', 'aac'])
            return self._apply_compression(audio, codec)
        
        elif method == 'noise':
            noise_file = random.choice(self.noises)
            snr = random.uniform(self.noise_snr_range[0], self.noise_snr_range[1])
            return self._add_noise(audio, noise_file, snr)
        
        elif method == 'music':
            music_file = random.choice(self.musics)
            snr = random.uniform(self.music_snr_range[0], self.music_snr_range[1])
            return self._add_noise(audio, music_file, snr)
        
        elif method == 'speech':
            speech_file = random.choice(self.speeches)
            snr = random.uniform(self.speech_snr_range[0], self.speech_snr_range[1])
            return self._add_noise(audio, speech_file, snr)
        
        else:
            logger.log(f"Warning: Unknown augmentation method: {method}")
            return audio
    
    def augment(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply audio augmentation to input signal
        
        Args:
            audio: Input audio signal
            
        Returns:
            Augmented audio signal
        """
        # Validate input
        if not isinstance(audio, np.ndarray):
            logger.log(f"Warning: Input audio is not a numpy array, type: {type(audio)}")
            return audio
            
        # Ensure input is mono
        if len(audio.shape) > 1:
            audio = audio[:, 0]
            
        # Create a copy to avoid modifying the original
        augmented_audio = audio.copy()
        
        # If random_single_method is enabled, select one method randomly
        if self.random_single_method:
            if not self.augmentation_methods:
                logger.log("Warning: No augmentation methods available")
                return augmented_audio
            
            # Select a random method based on their probabilities
            methods, probabilities = zip(*self.augmentation_methods)
            
            # Create weighted selection based on probabilities
            total_prob = sum(probabilities)
            if total_prob == 0:
                return augmented_audio
            
            # Normalize probabilities
            normalized_probs = [p/total_prob for p in probabilities]
            
            # Select one method randomly
            selected_method = np.random.choice(methods, p=normalized_probs)
            
            # Apply the selected method
            augmented_audio = self._apply_single_augmentation(augmented_audio, selected_method)
            
            # logger.log(f"Applied single augmentation method: {selected_method}")
            
        else:
            # Original behavior: apply all augmentations based on their probabilities
            
            # Apply RIR (reverberation)
            if self.rirs and random.random() < self.rir_prob:
                augmented_audio = self._add_rir(augmented_audio)
            
            # Apply SoX effects
            if self.sox_available:
                sox_effects = []
                
                # Add tempo stretch effect
                if random.random() < self.tempo_prob:
                    tempo_factor = random.uniform(self.tempo_range[0], self.tempo_range[1])
                    # Using 'stretch' effect as per SoX documentation
                    sox_effects.extend(['stretch', str(tempo_factor)])
                
                # Add gain effect
                if random.random() < self.gain_prob:
                    gain_db = random.uniform(self.gain_range[0], self.gain_range[1])
                    sox_effects.extend(['gain', str(gain_db)])
                
                # Add fade effect
                if random.random() < self.fade_prob:
                    fade_duration = random.uniform(self.fade_range[0], self.fade_range[1])
                    # 'p' means that fade starts at the end of the audio
                    sox_effects.extend(['fade', 'p', '0', str(fade_duration)])
                
                # Apply all SoX effects if any
                if sox_effects:
                    augmented_audio = self._apply_sox_effects(augmented_audio, sox_effects)
            
            # Apply audio compression using FFmpeg
            if self.ffmpeg_available and random.random() < self.compression_prob:
                codec = random.choice(['opus', 'aac'])
                augmented_audio = self._apply_compression(augmented_audio, codec)
            
            # Add background noise
            if self.noises and random.random() < self.noise_prob:
                noise_file = random.choice(self.noises)
                snr = random.uniform(self.noise_snr_range[0], self.noise_snr_range[1])
                augmented_audio = self._add_noise(augmented_audio, noise_file, snr)
            
            # Add background music
            if self.musics and random.random() < self.music_prob:
                music_file = random.choice(self.musics)
                snr = random.uniform(self.music_snr_range[0], self.music_snr_range[1])
                augmented_audio = self._add_noise(augmented_audio, music_file, snr)
            
            # Add background speech
            if self.speeches and random.random() < self.speech_prob:
                speech_file = random.choice(self.speeches)
                snr = random.uniform(self.speech_snr_range[0], self.speech_snr_range[1])
                augmented_audio = self._add_noise(augmented_audio, speech_file, snr)
        
        return augmented_audio


def fast_scandir(path: str, extensions: List[str], recursive: bool = False):
    """
    Scan files recursively faster than glob
    
    Args:
        path: Directory path to scan
        extensions: List of file extensions to include
        recursive: Whether to scan subdirectories
        
    Returns:
        Tuple of (subfolders, files)
    """
    subfolders, files = [], []

    try:
        for f in os.scandir(path):
            try:
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    if os.path.splitext(f.name)[1].lower() in extensions:
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    if recursive:
        for path in list(subfolders):
            sf, f = fast_scandir(path, extensions, recursive=recursive)
            subfolders.extend(sf)
            files.extend(f)

    return subfolders, files