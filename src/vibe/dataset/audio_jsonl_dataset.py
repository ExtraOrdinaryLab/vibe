import math
import json

import torch
import torchaudio
from torch.utils.data import Dataset


class JsonlAudioDataset(Dataset):

    def __init__(self, jsonl_file, preprocessor=None, chunkify=False, chunk_length=3.0, chunk_overlap=0.0):
        """
        Dataset for loading speaker data from jsonl files with optional chunking
        
        Args:
            jsonl_file (str): Path to the jsonl file
            preprocessor (dict): Dictionary containing preprocessing functions
            chunkify (bool): Whether to split audio files into chunks
            chunk_length (float): Length of each chunk in seconds
            chunk_overlap (float): Overlap between chunks in seconds
        """
        self.data_points = self.read_jsonl(jsonl_file)
        self.preprocessor = preprocessor if preprocessor is not None else {}
        self.chunkify = chunkify
        self.chunk_length = chunk_length
        self.chunk_overlap = chunk_overlap
        
        # Process chunking if enabled
        if self.chunkify:
            self.create_chunks()
    
    def read_jsonl(self, jsonl_file):
        """Read data from jsonl file"""
        data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def create_chunks(self):
        """Create chunks from audio files"""
        chunked_data = []
        for item in self.data_points:
            duration = item['duration']
            
            # Skip if audio is shorter than chunk_length
            if duration < self.chunk_length:
                chunked_data.append(item)
                continue
                
            # Calculate number of chunks with overlap
            step = self.chunk_length - self.chunk_overlap
            num_chunks = max(1, math.floor((duration - self.chunk_length) / step) + 1)
            
            # Create chunks
            for i in range(num_chunks):
                start_time = i * step
                end_time = start_time + self.chunk_length
                
                if end_time > duration:
                    end_time = duration
                    start_time = max(0, end_time - self.chunk_length)
                
                chunk = item.copy()
                chunk['chunk_info'] = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'original_duration': duration
                }
                chunked_data.append(chunk)
        
        self.data_points = chunked_data
    
    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, index):
        """Get item from dataset"""
        item = self.data_points[index]
        audio_path = item['audio']['path']
        spk_id = item['spk_id']
        
        # Handle chunked or full audio
        if 'chunk_info' in item and self.chunkify:
            start_time = item['chunk_info']['start_time']
            end_time = item['chunk_info']['end_time']
        else:
            start_time = 0
            end_time = item['duration']
        
        # Load and preprocess audio
        if 'waveform_reader' in self.preprocessor:
            wav, speed_idx = self.preprocessor['waveform_reader'](audio_path, start_time, end_time)
        
        # Apply speaker ID encoding if available
        if 'label_encoder' in self.preprocessor:
            spk_id = self.preprocessor['label_encoder'](spk_id, speed_idx)
        
        # Apply augmentations if available
        if 'augmentations' in self.preprocessor:
            wav = self.preprocessor['augmentations'](wav)
        
        # Extract features if available
        if 'feature_extractor' in self.preprocessor:
            feat = self.preprocessor['feature_extractor'](wav)
        else:
            feat = wav
            
        return feat, spk_id
    
