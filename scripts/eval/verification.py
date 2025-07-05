import os
import sys
import json
import time
import math
import logging
import argparse
import tempfile
from pathlib import Path
from typing import List, Callable, Dict, Tuple, Set

import numpy as np
from tqdm import tqdm
from sklearn import metrics
from rich.console import Console

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import soundfile as sf
from accelerate import Accelerator  # Import Accelerator

from vibe.augment.audio_augmentor import AudioAugmentor
from vibe.models.ecapa_tdnn.configuration_ecapa_tdnn import EcapaTdnnConfig
from vibe.models.ecapa_tdnn.feature_extractor_ecapa_tdnn import EcapaTdnnFeatureExtractor
from vibe.models.ecapa_tdnn.modeling_ecapa_tdnn import EcapaTdnnForSpeakerClassification

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--manifest_paths", default=None, nargs='+', help="")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="")
    parser.add_argument("--output_dir", type=str, default=None, help="")
    parser.add_argument("--trial_file", nargs='+', help="")
    parser.add_argument("--device", type=str, default='cuda', help="")
    parser.add_argument("--trust_remote_code", action="store_true", help="")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--audio_column_name", type=str, default='audio', help="Name of the audio column in the manifest")
    parser.add_argument("--label_column_name", type=str, default='label', help="Name of the label column in the manifest")
    parser.add_argument("--use_disk_cache", action="store_true", help="Cache embeddings to disk to reduce memory usage")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to store embedding cache")
    # Add score normalization arguments
    parser.add_argument("--score_norm", type=str, choices=["none", "z-norm", "t-norm", "s-norm"], default="none", 
                        help="Score normalization method. Options: none, z-norm, t-norm, s-norm")
    parser.add_argument("--cohort_manifest", nargs='+', default=None, 
                        help="Manifest file(s) containing cohort data for score normalization")
    parser.add_argument("--cohort_size", type=int, default=None, 
                        help="Number of top cohort scores to use for normalization. If None, all scores are used.")
    parser.add_argument("--num_cohort_samples", type=int, default=None,
                        help="Maximum number of samples to use from cohort. If None, all available samples are used.")
    return parser.parse_args()


class JsonlAudioDataset(Dataset):
    
    def __init__(
        self,
        manifest_paths: List[str],
        transform: Callable = None,
        audio_column_name: str = "audio",
        label_column_name: str = "label",
        max_duration: float = 30.0,
        sample_rate: int = 16000,
        label2id: dict = None,
        filter_files: Set[str] = None,  # Only include these files (absolute paths)
        max_samples: int = None  # Maximum number of samples to include
    ):
        self.entries = []
        self.audio_column_name = audio_column_name
        self.label_column_name = label_column_name
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.transform = transform
        self.filter_files = filter_files

        for manifest_path in manifest_paths:
            with open(manifest_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    
                    # Get absolute file path
                    audio_path = item[audio_column_name]["path"]
                    
                    # Skip if not in the filter list (if filter is active)
                    # Now comparing absolute paths directly
                    if filter_files is not None and audio_path not in filter_files:
                        continue
                    
                    # Check duration constraints
                    if max_duration is not None:
                        duration = float(item.get("duration", -1))
                        if duration <= 0 or duration > max_duration:
                            continue
                    
                    self.entries.append(item)
                    
                    # Break if we've reached the max samples limit
                    if max_samples is not None and len(self.entries) >= max_samples:
                        break
            
            # Break outer loop too if we've reached the max samples
            if max_samples is not None and len(self.entries) >= max_samples:
                break

        if label2id is None:
            unique_labels = sorted({entry[label_column_name] for entry in self.entries})
            self.label2id = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label2id = label2id

        self.id2label = {i: l for l, i in self.label2id.items()}

        console.log(f"Loaded {len(self.entries)} valid samples" + 
                   (f" (<= {max_duration}s)" if max_duration is not None else ""))
        console.log(f"Found {len(self.label2id)} unique speakers")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        audio_path = item[self.audio_column_name]["path"]
        spk = item[self.label_column_name]

        audio_array, sr = sf.read(audio_path)
        if len(audio_array.shape) > 1:
            audio_array = audio_array[:, 0]
        if sr != self.sample_rate:
            raise ValueError(f"Sample rate mismatch: expected {self.sample_rate}, got {sr}")

        label_id = self.label2id[spk]

        # Use absolute path directly for filename to match with trials
        return {
            self.audio_column_name: {"array": audio_array, "path": audio_path, "sampling_rate": sr},
            "label": label_id,
            "filename": audio_path  # Using absolute path for matching with trials
        }


class EmbeddingCache:
    """Class to manage embeddings with optional disk caching to reduce memory usage"""
    
    def __init__(self, use_disk=False, cache_dir=None):
        """
        Initialize embedding cache
        
        Args:
            use_disk: Whether to use disk caching
            cache_dir: Directory to store cache files (if None, uses temp dir)
        """
        self.embeddings = {}  # In-memory cache
        self.use_disk = use_disk
        
        if use_disk:
            if cache_dir is None:
                self.cache_dir = Path(tempfile.mkdtemp(prefix="embedding_cache_"))
            else:
                self.cache_dir = Path(cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
            
            console.log(f"Using disk cache at: {self.cache_dir}")
    
    def add(self, filename, embedding):
        """Add embedding to cache"""
        if self.use_disk:
            # Generate a safe filename from the path
            safe_name = filename.replace('/', '_').replace('\\', '_') + ".pt"
            cache_path = self.cache_dir / safe_name
            torch.save(embedding.cpu(), cache_path)
        else:
            self.embeddings[filename] = embedding.cpu()
    
    def get(self, filename):
        """Get embedding from cache"""
        if self.use_disk:
            safe_name = filename.replace('/', '_').replace('\\', '_') + ".pt"
            cache_path = self.cache_dir / safe_name
            if cache_path.exists():
                return torch.load(cache_path)
            return None
        else:
            return self.embeddings.get(filename)
    
    def contains(self, filename):
        """Check if embedding exists in cache"""
        if self.use_disk:
            safe_name = filename.replace('/', '_').replace('\\', '_') + ".pt"
            return (self.cache_dir / safe_name).exists()
        else:
            return filename in self.embeddings
    
    def clear(self):
        """Clear the cache"""
        if self.use_disk:
            for file_path in self.cache_dir.glob("*.pt"):
                os.remove(file_path)
        self.embeddings.clear()
    
    def get_all_embeddings(self):
        """Get all embeddings as a list"""
        if self.use_disk:
            embeddings = []
            for file_path in self.cache_dir.glob("*.pt"):
                embeddings.append(torch.load(file_path))
            return embeddings
        else:
            return list(self.embeddings.values())


def eer_score(positive_scores, negative_scores, return_threshold=False):
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d

    scores = positive_scores + negative_scores
    labels = [1] * len(positive_scores) + [0] * len(negative_scores)
    fpr, tpr, thresholds_ = metrics.roc_curve(labels, scores, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresholds = interp1d(fpr, thresholds_)(eer)
    
    if return_threshold:
        return float(eer), float(thresholds)
    return float(eer)


def mindcf_score(positive_scores, negative_scores, p_target=0.01, c_miss=1.0, c_fa=1.0, return_threshold=False):
    scores = positive_scores + negative_scores
    labels = [1] * len(positive_scores) + [0] * len(negative_scores)
    fprs, tprs, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnrs = 1 - tprs
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    if return_threshold:
        return float(min_dcf), float(min_c_det_threshold)
    return min_dcf


def inference_transforms(batch, feature_extractor):
    """Apply transforms for inference across a batch."""
    audio_arrays = [item["audio"]["array"] for item in batch]
    
    # Center crop or pad to feature_extractor's chunk length
    max_length = max([len(array) for array in audio_arrays])
    processed_arrays = []
    for wav in audio_arrays:
        if len(wav) > max_length:
            # Center crop
            start = (len(wav) - max_length) // 2
            wav_processed = wav[start:start + max_length]
        else:
            # Pad with zeros if needed
            wav_processed = np.pad(wav, (0, max(0, max_length - len(wav))), mode='constant')
        processed_arrays.append(wav_processed)
        
    inputs = feature_extractor(
        processed_arrays, 
        max_length=max_length, 
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors='pt'
    )

    return {
        "input_features": inputs["input_features"],
        "filenames": [item["filename"] for item in batch]
    }


def extract_trial_files(trial_files):
    """Extract all filenames referenced in trial files"""
    required_files = set()
    
    for trial_file in trial_files:
        with open(trial_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # Add both enrollment and test files
                    required_files.add(parts[1])
                    required_files.add(parts[2])
    
    console.log(f"Extracted {len(required_files)} unique files from trial lists")
    return required_files


def compute_embeddings(dataset, dataloader, model, embedding_cache=None, accelerator=None):
    """Compute embeddings for all files in the dataset and store in cache"""
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Computing Embeddings')):
            # No need to manually move inputs to device, Accelerator handles device placement
            outputs = model(input_features=batch["input_features"])
            embeddings = outputs.embeddings
            
            # Use Accelerator to gather results from all processes (in distributed settings)
            if accelerator:
                embeddings = accelerator.gather(embeddings)
                filenames = accelerator.gather_for_metrics(batch["filenames"])
            else:
                filenames = batch["filenames"]
            
            # Store embeddings
            for i, filename in enumerate(filenames):
                if i < len(embeddings) and embedding_cache is not None:
                    embedding_cache.add(filename, embeddings[i])
            
            # Free up memory periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()


def evaluate_trials(trial_file, embedding_cache, cohort_dict=None, score_norm="none", cohort_size=None):
    """
    Evaluate verification trials using pre-computed embeddings with optional score normalization
    
    Args:
        trial_file: Path to the trial file
        embedding_cache: Cache containing trial embeddings
        cohort_dict: Dictionary of cohort embeddings for score normalization
        score_norm: Score normalization method (none, z-norm, t-norm, s-norm)
        cohort_size: Number of top cohort scores to use for normalization
    """
    similarity_fn = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    
    # Determine trial name
    if Path(trial_file).name == 'veri_test2.txt':
        trial_name = 'voxceleb1-o'
    elif Path(trial_file).name == 'list_test_hard2.txt':
        trial_name = 'voxceleb1-h'
    elif Path(trial_file).name == 'list_test_all2.txt':
        trial_name = 'voxceleb1-e'
    else:
        trial_name = Path(trial_file).stem
    
    # Load the verification trial file
    with open(trial_file, "r") as f:
        trial_list = [line.strip().split() for line in f]
    trial_list = [(int(parts[0]), parts[1], parts[2]) for parts in trial_list]
    
    console.log(f"Processing {len(trial_list)} trials for {trial_name}")
    
    positive_scores, negative_scores, trial_scores = [], [], []
    missing_files = set()
    
    # Create cohort tensor for score normalization if needed
    apply_score_norm = score_norm != "none" and cohort_dict is not None
    cohort_embeddings = None
    
    if apply_score_norm:
        # Stack all cohort embeddings
        cohort_embeddings = torch.stack(list(cohort_dict.values()))
        console.log(f"Using {len(cohort_embeddings)} cohort embeddings for score normalization")
    
    # Save scores to file (like SpeechBrain)
    save_file = os.path.join(os.path.dirname(trial_file), f"scores_{trial_name}_{score_norm}.txt")
    s_file = open(save_file, "w", encoding="utf-8")
    
    # Process trials in batches to reduce memory usage
    batch_size = 1000  # Adjust batch size based on your memory constraints
    total_batches = (len(trial_list) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(trial_list))
        batch_trials = trial_list[start_idx:end_idx]
        
        console.log(f"Processing batch {batch_idx+1}/{total_batches} ({start_idx}-{end_idx-1}) of trials")
        
        # Process each trial in the batch
        for parts in tqdm(batch_trials, desc=f'Trial batch {batch_idx+1}/{total_batches}'):
            label, enrol_filename, test_filename = parts
            
            # Get embeddings from cache
            enrol_embedding = embedding_cache.get(enrol_filename)
            test_embedding = embedding_cache.get(test_filename)
            
            if enrol_embedding is None or test_embedding is None:
                missing_files.add(enrol_filename if enrol_embedding is None else test_filename)
                continue
            
            # Ensure embeddings are properly shaped for similarity calculation
            if len(enrol_embedding.shape) == 1:
                enrol_embedding = enrol_embedding.unsqueeze(0)
            if len(test_embedding.shape) == 1:
                test_embedding = test_embedding.unsqueeze(0)
            
            # Compute raw similarity score
            score = similarity_fn(enrol_embedding, test_embedding).item()
            
            # Apply score normalization if requested
            if apply_score_norm:
                try:
                    # Calculate scores against the cohort
                    # Fix: Ensure proper reshaping for broadcasting
                    if enrol_embedding.dim() == 2 and cohort_embeddings.dim() == 2:
                        # Calculate similarity between enrollment embedding and all cohort embeddings
                        score_e_c = torch.zeros(cohort_embeddings.shape[0])
                        for i in range(cohort_embeddings.shape[0]):
                            score_e_c[i] = similarity_fn(enrol_embedding, cohort_embeddings[i:i+1])
                        
                        # Calculate similarity between test embedding and all cohort embeddings
                        score_t_c = torch.zeros(cohort_embeddings.shape[0])
                        for i in range(cohort_embeddings.shape[0]):
                            score_t_c[i] = similarity_fn(test_embedding, cohort_embeddings[i:i+1])
                        
                        # Apply cohort size limit if specified
                        if cohort_size is not None:
                            score_e_c, _ = torch.topk(score_e_c, k=min(cohort_size, len(score_e_c)))
                            score_t_c, _ = torch.topk(score_t_c, k=min(cohort_size, len(score_t_c)))
                        
                        # Calculate statistics for normalization
                        mean_e_c = torch.mean(score_e_c)
                        std_e_c = torch.std(score_e_c)
                        mean_t_c = torch.mean(score_t_c)
                        std_t_c = torch.std(score_t_c)
                        
                        # Apply the selected normalization technique
                        if score_norm == "z-norm":
                            score = (score - mean_e_c) / (std_e_c if std_e_c > 0 else 1.0)
                        elif score_norm == "t-norm":
                            score = (score - mean_t_c) / (std_t_c if std_t_c > 0 else 1.0)
                        elif score_norm == "s-norm":
                            # Fix: Use item() to get scalar values from tensors
                            score_e = (score - mean_e_c.item()) / (std_e_c.item() if std_e_c.item() > 0 else 1.0)
                            score_t = (score - mean_t_c.item()) / (std_t_c.item() if std_t_c.item() > 0 else 1.0)
                            score = 0.5 * (score_e + score_t)
                    else:
                        console.log(f"Warning: Dimension mismatch. enrol: {enrol_embedding.shape}, cohort: {cohort_embeddings.shape}")
                except Exception as e:
                    console.log(f"Error during score normalization: {e}")
                    # Continue with raw score if normalization fails
            
            # Store scores
            if label == 1:
                positive_scores.append(score)
            else:
                negative_scores.append(score)
            
            trial_scores.append((score, enrol_filename, test_filename))
            
            # Write score to file (following SpeechBrain format)
            enrol_id = os.path.basename(enrol_filename).split('.')[0]
            test_id = os.path.basename(test_filename).split('.')[0]
            s_file.write("%s %s %i %f\n" % (enrol_id, test_id, label, score))
        
        # Free up memory after each batch
        torch.cuda.empty_cache()
    
    s_file.close()
    
    if missing_files:
        console.log(f"Warning: {len(missing_files)} files missing from cache")
        console.log(f"First few missing files: {list(missing_files)[:5]}")
    
    # Calculate metrics
    if len(positive_scores) > 0 and len(negative_scores) > 0:
        console.log(f"Calculating metrics with {len(positive_scores)} positive and {len(negative_scores)} negative scores")
        eer, eer_threshold = eer_score(positive_scores, negative_scores, return_threshold=True)
        mindcf, mindcf_threshold = mindcf_score(
            positive_scores, negative_scores, p_target=0.01, c_miss=1.0, c_fa=1.0, return_threshold=True
        )
        
        norm_info = f" ({score_norm})" if apply_score_norm else ""
        console.log(f"{trial_name}{norm_info}: EER = {eer:.6f}, minDCF = {mindcf:.6f}")
    else:
        console.log(f"Error: No valid scores found for {trial_name}")
        eer, eer_threshold = 1.0, 0.0
        mindcf, mindcf_threshold = 1.0, 0.0
    
    return {
        'trial_name': trial_name,
        'score_norm': score_norm,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'mindcf': mindcf,
        'mindcf_threshold': mindcf_threshold,
        'trial_scores': trial_scores,
        'positive_scores': positive_scores,
        'negative_scores': negative_scores
    }


def main(args):
    if args.output_dir is None:
        raise ValueError("No output directory specified. Set `--output_dir`.")
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Accelerator
    accelerator = Accelerator()
    console.log(f"Accelerator config: {accelerator.state}")
    
    # Initialize model and feature extractor
    feature_extractor = EcapaTdnnFeatureExtractor.from_pretrained(
        args.model_name_or_path, 
        chunk_length=30
    )
    
    config = EcapaTdnnConfig.from_pretrained(args.model_name_or_path)
    model = EcapaTdnnForSpeakerClassification.from_pretrained(args.model_name_or_path, config=config)
    model.eval()
    
    # Extract required files from trial lists
    required_files = extract_trial_files(args.trial_file)
    
    # Create dataset with only the required files
    test_dataset = JsonlAudioDataset(
        manifest_paths=args.manifest_paths,
        audio_column_name=args.audio_column_name,
        label_column_name=args.label_column_name,
        max_duration=None,
        sample_rate=feature_extractor.sampling_rate,
        filter_files=required_files
    )
    
    # Create dataloader for batch processing
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: inference_transforms(batch, feature_extractor),
        pin_memory=True
    )
    
    # Use Accelerator to wrap model and data loaders
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    
    # Initialize embedding cache
    embedding_cache = EmbeddingCache(
        use_disk=args.use_disk_cache,
        cache_dir=args.cache_dir
    )
    
    # Compute embeddings for all required files
    compute_embeddings(
        test_dataset, 
        test_dataloader, 
        model, 
        embedding_cache,
        accelerator
    )
    
    # Process cohort data if score normalization is requested
    cohort_dict = None
    if args.score_norm != "none" and args.cohort_manifest:
        console.log("Processing cohort data for score normalization...")
        
        # Create cohort dataset - limiting to the specified number of samples if provided
        cohort_dataset = JsonlAudioDataset(
            manifest_paths=args.cohort_manifest,
            audio_column_name=args.audio_column_name,
            label_column_name=args.label_column_name,
            max_duration=30.0,  # Limiting cohort utterance duration
            sample_rate=feature_extractor.sampling_rate,
            max_samples=args.num_cohort_samples  # Limit the number of cohort samples
        )
        
        # Create dataloader for cohort processing
        cohort_dataloader = DataLoader(
            cohort_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda batch: inference_transforms(batch, feature_extractor),
            pin_memory=True
        )
        
        # Use Accelerator to wrap cohort_dataloader
        cohort_dataloader = accelerator.prepare(cohort_dataloader)
        
        # Create a separate cache for cohort embeddings
        cohort_cache_dir = None
        if args.cache_dir:
            cohort_cache_dir = os.path.join(args.cache_dir, "cohort_cache")
        
        cohort_cache = EmbeddingCache(
            use_disk=args.use_disk_cache,
            cache_dir=cohort_cache_dir
        )
        
        # Compute embeddings for cohort files
        compute_embeddings(
            cohort_dataset, 
            cohort_dataloader, 
            model, 
            cohort_cache,
            accelerator
        )
        
        # Create dictionary of cohort embeddings (like SpeechBrain's train_dict)
        cohort_dict = {}
        for i, item in enumerate(cohort_dataset.entries):
            filename = item[args.audio_column_name]["path"]
            embedding = cohort_cache.get(filename)
            if embedding is not None:
                cohort_dict[f"cohort_{i}"] = embedding
        
        console.log(f"Created cohort dictionary with {len(cohort_dict)} embeddings")
    elif args.score_norm != "none" and not args.cohort_manifest:
        console.log("Warning: Score normalization requested but no cohort manifest provided.")
    
    # Free up GPU memory
    accelerator.free_memory()
    torch.cuda.empty_cache()
    
    # Ensure main process handles evaluation (in distributed environment)
    if accelerator.is_main_process:
        # Evaluate all trial files
        for trial_file in args.trial_file:
            results = evaluate_trials(
                trial_file, 
                embedding_cache,
                cohort_dict=cohort_dict,
                score_norm=args.score_norm,
                cohort_size=args.cohort_size
            )
            
            # Save results
            trial_name = results['trial_name']
            score_norm_suffix = f"_{args.score_norm}" if args.score_norm != "none" else ""
            
            with open(os.path.join(args.output_dir, f'verification_results_{trial_name}{score_norm_suffix}.json'), "w") as f:
                result = {
                    'score_norm': args.score_norm,
                    'cohort_size': args.cohort_size,
                    'eer': {'value': results['eer'], 'threshold': results['eer_threshold']}, 
                    'mindcf': {'value': results['mindcf'], 'threshold': results['mindcf_threshold']}, 
                }
                json.dump(result, f, indent=4)

            with open(os.path.join(args.output_dir, f'trial_scores_{trial_name}{score_norm_suffix}.txt'), "w") as f:
                for data_point in results['trial_scores']:
                    score, enrol_filename, test_filename = data_point
                    f.write(f'{score}\t{enrol_filename}\t{test_filename}\n')
    
    # Wait for all processes to complete
    accelerator.wait_for_everyone()
    
    # Clean up cache if using disk
    if args.use_disk_cache and args.cache_dir is None:
        embedding_cache.clear()


if __name__ == "__main__":
    args = parse_args()
    main(args)