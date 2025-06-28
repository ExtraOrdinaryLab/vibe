import os
import json
import math
import argparse
from pathlib import Path
from random import randint
from typing import List, Callable, Dict, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm
from rich.console import Console
from sklearn.model_selection import train_test_split

import torch
import evaluate
import torch.nn.functional as F
from accelerate.utils import set_seed
from accelerate import Accelerator
from huggingface_hub import HfApi
from transformers import SchedulerType, get_scheduler
from torch.utils.data import Dataset, DataLoader, Subset

from vibe.logging import get_logger
from vibe.augment.audio_augmentor import AudioAugmentor
from vibe.grad_cache import GradCache, NTXentLoss
from vibe.optimizers.lars import LARS
from vibe.models.ecapa_tdnn.configuration_ecapa_tdnn import EcapaTdnnConfig
from vibe.models.ecapa_tdnn.feature_extractor_ecapa_tdnn import EcapaTdnnFeatureExtractor
from vibe.models.ecapa_tdnn.modeling_ecapa_tdnn import EcapaTdnnForSpeakerClassification

logger = get_logger()


class JsonlAudioDataset(Dataset):
    
    def __init__(
        self,
        manifest_paths: List[str],
        transform: Callable = None,
        audio_column_name: str = "audio",
        label_column_name: str = "label",
        max_duration: float = 30.0,
        sample_rate: int = 16000,
        label2id: dict = None
    ):
        self.entries = []
        self.audio_column_name = audio_column_name
        self.label_column_name = label_column_name
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.transform = transform

        for manifest_path in manifest_paths:
            with open(manifest_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if max_duration is None:
                        self.entries.append(item)
                        continue
                    duration = float(item.get("duration", -1))
                    if 0 < duration <= max_duration:
                        self.entries.append(item)

        if label2id is None:
            unique_labels = sorted({entry[label_column_name] for entry in self.entries})
            self.label2id = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label2id = label2id

        self.id2label = {i: l for l, i in self.label2id.items()}

        logger.log(f"Loaded {len(self.entries)} valid samples (<= {max_duration}s)")
        logger.log(f"Found {len(self.label2id)} unique speakers")
        
        # Store labels for stratified split
        self.labels = [self.label2id[entry[label_column_name]] for entry in self.entries]

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

        return {
            self.audio_column_name: {"array": audio_array, "path": audio_path, "sampling_rate": sr},
            "label": label_id,
        }


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]


def create_train_val_split(dataset: JsonlAudioDataset, val_size: float = 0.1, random_state: int = 42) -> Tuple[Subset, Subset]:
    """
    Create a stratified train-validation split from a JsonlAudioDataset
    
    Args:
        dataset: The original dataset
        val_size: Proportion of data to use for validation (default: 0.1)
        random_state: Random seed for reproducibility
        
    Returns:
        A tuple of (train_subset, val_subset)
    """
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=val_size,
        random_state=random_state,
        stratify=dataset.labels
    )
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def evaluate_speaker_verification_with_sampling(
    model, val_loader, n_positives=5, n_negatives=5, num_runs=1, stratified=True, accelerator=None
):
    """
    Enhanced evaluation with multiple runs and stratified sampling
    """
    all_eer = []
    all_min_dcf = []
    
    for run in range(num_runs):
        model.eval()
        eval_progress_bar = tqdm(
            val_loader, 
            desc="Evaluating...", 
            leave=False,
            disable=not accelerator.is_local_main_process
        )
        
        # Get all embeddings and speaker IDs
        all_embeddings = []
        all_speaker_ids = []
        
        with torch.no_grad():
            for batch in eval_progress_bar:
                speaker_id = batch.pop('labels')
                embeddings = model(**batch).embeddings
                all_embeddings.append(embeddings.cpu())
                all_speaker_ids.extend(speaker_id.tolist())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_speaker_ids = torch.tensor(all_speaker_ids)
        
        # Initialize lists to store scores
        positive_scores = []
        negative_scores = []
        
        # Get unique speakers for stratified sampling
        unique_speakers = torch.unique(all_speaker_ids)
        
        if stratified:
            # Stratified sampling: ensure all speakers are represented
            for speaker in unique_speakers:
                # Get indices for this speaker
                speaker_indices = torch.where(all_speaker_ids == speaker)[0]
                
                # Skip if too few samples
                if len(speaker_indices) <= 1:
                    continue
                
                # Sample anchors for this speaker (up to 10% of samples or at least 1)
                num_anchors = max(1, int(0.1 * len(speaker_indices)))
                anchor_indices = speaker_indices[torch.randperm(len(speaker_indices))[:num_anchors]]
                
                for anchor_idx in anchor_indices:
                    anchor_embedding = all_embeddings[anchor_idx]
                    
                    # Find other samples from same speaker (excluding self)
                    same_speaker_indices = speaker_indices[speaker_indices != anchor_idx]
                    
                    # Find samples from different speakers
                    diff_speaker_indices = torch.where(all_speaker_ids != speaker)[0]
                    
                    # Sample positives
                    if len(same_speaker_indices) > 0:
                        n_pos = min(n_positives, len(same_speaker_indices))
                        pos_indices = same_speaker_indices[torch.randperm(len(same_speaker_indices))[:n_pos]]
                        
                        for pos_idx in pos_indices:
                            pos_embedding = all_embeddings[pos_idx]
                            score = F.cosine_similarity(anchor_embedding.unsqueeze(0), 
                                                      pos_embedding.unsqueeze(0)).item()
                            positive_scores.append(score)
                    
                    # Sample negatives
                    if len(diff_speaker_indices) > 0:
                        n_neg = min(n_negatives, len(diff_speaker_indices))
                        neg_indices = diff_speaker_indices[torch.randperm(len(diff_speaker_indices))[:n_neg]]
                        
                        for neg_idx in neg_indices:
                            neg_embedding = all_embeddings[neg_idx]
                            score = F.cosine_similarity(anchor_embedding.unsqueeze(0), 
                                                      neg_embedding.unsqueeze(0)).item()
                            negative_scores.append(score)
        else:
            # Random sampling across all samples
            for i in range(len(all_embeddings)):
                anchor_embedding = all_embeddings[i]
                anchor_speaker = all_speaker_ids[i]
                
                # Find indices of same speaker (excluding self)
                same_speaker_indices = torch.where((all_speaker_ids == anchor_speaker) & 
                                                  (torch.arange(len(all_speaker_ids)) != i))[0]
                
                # Find indices of different speakers
                diff_speaker_indices = torch.where(all_speaker_ids != anchor_speaker)[0]
                
                # Sample positives
                if len(same_speaker_indices) > 0:
                    n_pos = min(n_positives, len(same_speaker_indices))
                    pos_indices = same_speaker_indices[torch.randperm(len(same_speaker_indices))[:n_pos]]
                    
                    for pos_idx in pos_indices:
                        pos_embedding = all_embeddings[pos_idx]
                        score = F.cosine_similarity(anchor_embedding.unsqueeze(0), 
                                                  pos_embedding.unsqueeze(0)).item()
                        positive_scores.append(score)
                
                # Sample negatives
                if len(diff_speaker_indices) > 0:
                    n_neg = min(n_negatives, len(diff_speaker_indices))
                    neg_indices = diff_speaker_indices[torch.randperm(len(diff_speaker_indices))[:n_neg]]
                    
                    for neg_idx in neg_indices:
                        neg_embedding = all_embeddings[neg_idx]
                        score = F.cosine_similarity(anchor_embedding.unsqueeze(0), 
                                                  neg_embedding.unsqueeze(0)).item()
                        negative_scores.append(score)
        
        # Calculate EER and minDCF using your functions
        eer = eer_score(positive_scores, negative_scores)
        min_dcf = mindcf_score(positive_scores, negative_scores)
        
        all_eer.append(eer)
        all_min_dcf.append(min_dcf)
    
    # Average results over multiple runs
    avg_eer = sum(all_eer) / len(all_eer)
    avg_min_dcf = sum(all_min_dcf) / len(all_min_dcf)
    
    return avg_eer, avg_min_dcf


def eer_score(positive_scores, negative_scores, return_threshold=False):
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve

    scores = positive_scores + negative_scores
    labels = [1] * len(positive_scores) + [0] * len(negative_scores)
    fpr, tpr, thresholds_ = roc_curve(labels, scores, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresholds = interp1d(fpr, thresholds_)(eer)
    
    if return_threshold:
        return float(eer), float(thresholds)
    return float(eer)


def mindcf_score(positive_scores, negative_scores, p_target=0.01, c_miss=1.0, c_fa=1.0, return_threshold=False):
    from sklearn.metrics import roc_curve

    scores = positive_scores + negative_scores
    labels = [1] * len(positive_scores) + [0] * len(negative_scores)
    fprs, tprs, thresholds = roc_curve(labels, scores, pos_label=1)
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


def main():
    parser = argparse.ArgumentParser(description="Train Vibe model for audio classification")
    parser.add_argument(
        "--manifest_paths",
        nargs='+', 
        required=True, 
        help="Paths to JSONL manifest files containing audio data"
    )
    parser.add_argument("--audio_column_name", type=str, default='audio')
    parser.add_argument("--label_column_name", type=str, default='label')
    parser.add_argument("--max_length_seconds", type=float, default=30)
    parser.add_argument("--validation_split", type=float, default=0.01, 
                       help="Proportion of training data to use for validation")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "cyclic", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_cycles", type=int, default=4, help="Number of cycles for cyclic scheduler"
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log training metrics every X steps."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")

    parser.add_argument("--no_augment", action="store_true", help="Disable audio augmentation")
    parser.add_argument("--random_single_method", action="store_true", help="Apply only one random augmentation method instead of stacking multiple")
    parser.add_argument("--musan_path", type=str, default=None, help="MUSAN path")
    parser.add_argument("--rir_path", type=str, default=None, help="RIR path")
    parser.add_argument("--noise_prob", type=float, default=0.6, help="Probability of applying noise augmentation")
    parser.add_argument("--music_prob", type=float, default=0.3, help="Probability of applying music augmentation")
    parser.add_argument("--speech_prob", type=float, default=0.3, help="Probability of applying speech augmentation")
    parser.add_argument("--rir_prob", type=float, default=0.6, help="Probability of applying RIR augmentation")
    parser.add_argument("--tempo_prob", type=float, default=0.3, help="Probability of applying tempo changes")
    parser.add_argument("--gain_prob", type=float, default=0.3, help="Probability of applying gain adjustments")
    parser.add_argument("--fade_prob", type=float, default=0.2, help="Probability of applying fade effects")
    parser.add_argument("--compression_prob", type=float, default=0.3, help="Probability of applying audio compression")
    parser.add_argument("--tempo_range", type=float, nargs=2, default=[0.8, 1.2], help="Range of tempo stretch factors")
    parser.add_argument("--gain_range", type=float, nargs=2, default=[-5, 5], help="Range of gain adjustments in dB")
    parser.add_argument("--fade_range", type=float, nargs=2, default=[0.1, 1.0], help="Range of fade duration in seconds")
    parser.add_argument("--auto_install_tools", action="store_true", help="Automatically install SoX and FFmpeg if missing")
    
    # Add new arguments for contrastive learning
    parser.add_argument("--num_views", type=int, default=6, help="Number of augmented views to generate per sample for contrastive learning")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature parameter for SupCon loss")
    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    logger.log(accelerator.state)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    feature_extractor = EcapaTdnnFeatureExtractor(chunk_length=args.max_length_seconds)
    model_input_name = feature_extractor.model_input_names[0]

    audio_augmentor = None
    if not args.no_augment and (args.musan_path is not None or args.rir_path is not None):
        audio_augmentor = AudioAugmentor(
            musan_path=args.musan_path,
            rir_path=args.rir_path,
            noise_prob=args.noise_prob,
            music_prob=args.music_prob,
            speech_prob=args.speech_prob,
            rir_prob=args.rir_prob,
            tempo_prob=args.tempo_prob,
            gain_prob=args.gain_prob,
            fade_prob=args.fade_prob,
            compression_prob=args.compression_prob,
            tempo_range=args.tempo_range,
            gain_range=args.gain_range,
            fade_range=args.fade_range,
            sample_rate=feature_extractor.sampling_rate,
            auto_install=args.auto_install_tools, 
            random_single_method=args.random_single_method
        )
        logger.log(f"***** Enable audio augmentation *****")

    # Modified to create multiple views for contrastive learning
    def train_transforms_supcon(batch):
        """
        Transform function for supervised contrastive learning.
        Creates multiple augmented views of each sample for contrastive learning.
        """
        audio_arrays = [item["audio"]["array"] for item in batch]
        labels = [item["label"] for item in batch]
        
        # Create exactly 2 views per sample as expected by GradCache
        all_views = []
        
        for wav in audio_arrays:
            # First view: original audio (with random subsample)
            original_wav = random_subsample(wav, max_length=args.max_length_seconds, 
                                        sample_rate=feature_extractor.sampling_rate)
            
            # Second view: augmented audio
            if audio_augmentor is not None:
                augmented_wav = audio_augmentor.augment(
                    random_subsample(wav, max_length=args.max_length_seconds, 
                                    sample_rate=feature_extractor.sampling_rate)
                )
            else:
                # If no augmentor, just use another random subsample
                augmented_wav = random_subsample(wav, max_length=args.max_length_seconds, 
                                            sample_rate=feature_extractor.sampling_rate)
            
            # Add both views
            all_views.append(original_wav)
            all_views.append(augmented_wav)
        
        # Process all views through feature extractor
        inputs = feature_extractor(
            all_views, 
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors='pt'
        )
        
        # Return flat input features - GradCache will handle the reshaping
        return {
            "input_features": inputs["input_features"],
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    def val_transforms(batch):
        """Apply val_transforms across a batch."""
        audio_arrays = [item["audio"]["array"] for item in batch]
        subsampled_wavs = [
            # Use center crop for validation instead of random subsample
            wav[:int(feature_extractor.sampling_rate * args.max_length_seconds)] 
            if len(wav) > int(feature_extractor.sampling_rate * args.max_length_seconds) 
            else wav
            for wav in audio_arrays
        ]
        
        inputs = feature_extractor(
            subsampled_wavs, 
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors='pt'
        )

        return {
            "input_features": inputs["input_features"],
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long)
        }

    # Create the full dataset
    full_dataset = JsonlAudioDataset(
        manifest_paths=args.manifest_paths,
        transform=None,
        audio_column_name=args.audio_column_name,
        label_column_name=args.label_column_name,
        max_duration=None,
        sample_rate=feature_extractor.sampling_rate
    )
    
    # Create stratified train-val split
    train_dataset, val_dataset = create_train_val_split(
        full_dataset, 
        val_size=args.validation_split,
        random_state=args.seed
    )
    
    logger.log(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    logger.log(f"***** Using contrastive learning with {args.num_views} views per sample *****")

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.per_device_train_batch_size, 
        collate_fn=train_transforms_supcon,  # Use the contrastive learning transform
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    eval_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.per_device_eval_batch_size, 
        collate_fn=val_transforms, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    # Initialize model
    config = EcapaTdnnConfig()
    config.num_labels = len(full_dataset.label2id)
    config.label2id = full_dataset.label2id
    config.id2label = full_dataset.id2label
    model = EcapaTdnnForSpeakerClassification(config)
    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize SupCon loss
    # loss_fn = SupConLoss(temperature=args.temperature)
    loss_fn = NTXentLoss(temperature=args.temperature)
    
    # Initialize GradCache
    grad_cache = GradCache(
        models=[model, model],
        chunk_sizes=32,
        loss_fn=loss_fn,
        split_input_fn=None,
        get_rep_fn=lambda v: v.embeddings, 
        accelerator=accelerator
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    optimizer = LARS(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_training_steps = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    if args.lr_scheduler_type == 'cyclic':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer, 
            base_lr=1e-8,
            max_lr=args.learning_rate,
            step_size_up=(num_training_steps // 2) // args.num_cycles,
            mode='triangular2',
            cycle_momentum=False
        )
    else:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
            num_training_steps=num_training_steps,
        )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("audio_classification_vibe", experiment_config)

    # Get the metric function
    metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.log("***** Running training *****")
    logger.log(f"  Num examples = {len(train_dataset)}")
    logger.log(f"  Num validation examples = {len(val_dataset)}")
    logger.log(f"  Num Epochs = {args.num_train_epochs}")
    logger.log(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.log(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.log(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.log(f"  Total optimization steps = {args.max_train_steps}")
    logger.log(f"  Logging steps = {args.logging_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), leave=False, disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
            
        for step, batch in enumerate(active_dataloader):
            # Modified training loop for contrastive learning with GradCache
            input_features = batch["input_features"]
            labels = batch["labels"]
            
            # Process in chunks using GradCache
            batch_size = input_features.shape[0]
            loss = grad_cache(
                input_features[:batch_size//2, ...], 
                input_features[batch_size//2:, ...], 
                reduction='mean'
            )
            
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
                
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if args.logging_steps > 0 and completed_steps % args.logging_steps == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    epoch_fraction = epoch + (step + 1) / len(train_dataloader)
                    log_data = {
                        'loss': loss.item(), 
                        'learning_rate': current_lr, 
                        'epoch': epoch_fraction
                    }
                    if accelerator.is_main_process:
                        progress_bar.clear()
                        tqdm.write(str(log_data))
                        with open(os.path.join(args.output_dir, 'train_log.txt'), 'a') as log_file:
                            log_file.write(str(log_data) + '\n')

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        # Evaluation code remains the same
        model.eval()
        
        eer, mindcf = evaluate_speaker_verification_with_sampling(
            model, eval_dataloader, n_positives=5, n_negatives=5, num_runs=1, stratified=True, accelerator=accelerator
        )

        # eval_metric = metric.compute()
        eval_metric = {'eer': eer, 'mindcf': mindcf}
        if accelerator.is_main_process:
            tqdm.write(f"epoch {epoch}: {eval_metric}")
            with open(os.path.join(args.output_dir, 'val_log.txt'), 'a') as log_file:
                log_file.write(f"epoch {epoch}: {eval_metric}" + '\n')

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                feature_extractor.save_pretrained(args.output_dir)
                config.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            config.save_pretrained(args.output_dir)
            feature_extractor.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
            all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)

    accelerator.wait_for_everyone()
    accelerator.end_training()
            

if __name__ == "__main__":
    main()