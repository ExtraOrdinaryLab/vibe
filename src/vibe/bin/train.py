"""
Speaker recognition model training script using distributed training.
This module handles the complete training pipeline for speaker embedding models.
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as torch_dist
from torch.utils.data.distributed import DistributedSampler

from vibe.utils import (
    build_config,
    get_logger,
    build,
    set_seed,
    EpochLogger,
    AverageMeters,
    ProgressMeter
)
from vibe.models.utils import convert_sync_batchnorm, revert_sync_batchnorm


def accuracy(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Calculate top-1 accuracy between predictions and targets.
    
    Args:
        inputs (torch.Tensor): Predictions tensor of shape [*, C]
        targets (torch.Tensor): Target labels tensor of shape [*,]
        
    Returns:
        float: Accuracy percentage (0-100)
    """
    _, pred = inputs.topk(1)
    pred = pred.squeeze(-1)
    acc = pred.eq(targets).float().mean()
    return acc * 100


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    lr_scheduler,
    margin_scheduler,
    logger,
    config,
    rank
):
    """
    Train the model for one epoch.
    
    Args:
        train_loader: DataLoader for training data
        model: Model to be trained
        criterion: Loss function
        optimizer: Optimizer for parameter updates
        epoch: Current epoch number
        lr_scheduler: Learning rate scheduler
        margin_scheduler: Margin scheduler for loss function
        logger: Logger for printing information
        config: Configuration object
        rank: Process rank in distributed training
        
    Returns:
        dict: Key training statistics
    """
    # Initialize tracking metrics
    train_stats = AverageMeters()
    train_stats.add('time', ':6.3f')
    train_stats.add('loss', ':4.4f')
    train_stats.add('accuracy@1', ':2.4f')
    train_stats.add('learning_rate', ':.8f')
    train_stats.add('margin', ':.3f')
    train_stats.add('grad_norm', ':.4f')
    train_stats.add('target_logit', ':.4f')  # Cosine similarity for correct class
    
    progress = ProgressMeter(
        len(train_loader),
        train_stats,
        prefix=f"Epoch: [{epoch}]"
    )

    # Set model to training mode
    model.train()

    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        # Update learning rate and margin for current iteration
        iter_num = (epoch - 1) * len(train_loader) + i
        lr_scheduler.step(iter_num)
        margin_scheduler.step(iter_num)

        # Move data to GPU
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        acc1 = accuracy(output, y)

        # Monitor logits statistics
        # For margin-based losses, these represent cosine similarities
        with torch.no_grad():
            # Get cosine values for the correct classes only
            batch_size = y.size(0)
            correct_class_logit = output[torch.arange(batch_size), y].mean().item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        # Calculate gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()

        # Record training statistics
        train_stats.update('loss', loss.item(), x.size(0))
        train_stats.update('accuracy@1', acc1.item(), x.size(0))
        train_stats.update('learning_rate', optimizer.param_groups[0]["lr"])
        train_stats.update('margin', margin_scheduler.get_margin())
        train_stats.update('grad_norm', total_norm)
        train_stats.update('target_logit', correct_class_logit)
        train_stats.update('time', time.time() - end)

        # Log progress at specified frequency
        if rank == 0 and i % config.log_batch_freq == 0:
            logger.info(progress.display(i))

        end = time.time()

    # Return key statistics for epoch logging
    key_stats = {
        'train_loss': train_stats.avg('loss'),
        'train_acc': train_stats.avg('accuracy@1'),
        'learning_rate': train_stats.val('learning_rate'), 
        'grad_norm': train_stats.avg('grad_norm'), 
        'target_logit': train_stats.avg('target_logit')
    }
    return key_stats


def validate(
    val_loader,
    model,
    criterion,
    logger,
    config,
    rank,
    epoch
):
    """
    Evaluate the model on validation data.
    
    Args:
        val_loader: DataLoader for validation data
        model: Model to be evaluated
        criterion: Loss function
        logger: Logger for printing information
        config: Configuration object
        rank: Process rank in distributed training
        epoch: Current epoch number
        
    Returns:
        dict: Key validation statistics
    """
    # Initialize tracking metrics
    val_stats = AverageMeters()
    val_stats.add('time', ':6.3f')
    val_stats.add('loss', ':4.4f')
    val_stats.add('accuracy@1', ':2.4f')
    
    progress = ProgressMeter(
        len(val_loader),
        val_stats,
        prefix=f"Validation: [{epoch}]"
    )

    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (x, y) in enumerate(val_loader):
            # Move data to GPU
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            # Forward pass
            output = model(x)
            loss = criterion(output, y)
            acc1 = accuracy(output, y)

            # Record validation statistics
            val_stats.update('loss', loss.item(), x.size(0))
            val_stats.update('accuracy@1', acc1.item(), x.size(0))
            val_stats.update('time', time.time() - end)

            # Log progress at specified frequency
            if rank == 0 and i % config.log_batch_freq == 0:
                logger.info(progress.display(i))

            end = time.time()

    # Log overall validation results
    if rank == 0:
        logger.info(f' * Validation Accuracy {val_stats.avg("accuracy@1"):.3f}')
        logger.info(f' * Validation Loss {val_stats.avg("loss"):.4f}')

    # Return key statistics for epoch logging
    key_stats = {
        'val_loss': val_stats.avg('loss'),
        'val_acc': val_stats.avg('accuracy@1')
    }
    return key_stats


def main():
    """
    Main function to setup and run the training process.
    Handles argument parsing, distributed training setup, and the training loop.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Speaker Network Training')
    parser.add_argument('--config', default='', type=str, help='Config file for training')
    parser.add_argument('--gpu', nargs='+', help='GPU id to use')
    parser.add_argument('--resume', default=True, type=bool, help='Resume from recent checkpoint or not')

    args, overrides = parser.parse_known_args(sys.argv[1:])
    config = build_config(args.config, overrides, copy=True)

    # Auto detect GPUs if not specified
    if args.gpu is None:
        args.gpu = list(range(torch.cuda.device_count()))
    
    # Check if we are running in single-GPU or multi-GPU mode
    is_distributed = len(args.gpu) > 1
    
    if is_distributed:
        # Setup distributed training environment
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(args.gpu[rank])
        torch.cuda.set_device(gpu)
        torch_dist.init_process_group(backend='nccl')
        set_seed()
    else:
        # Single GPU mode
        rank = 0
        world_size = 1
        gpu = int(args.gpu[0]) if args.gpu else 0
        torch.cuda.set_device(gpu)
        set_seed()

    # Create experiment directory and setup logging
    os.makedirs(config.exp_dir, exist_ok=True)
    logger = get_logger(fpath=os.path.join(config.exp_dir, 'train.log'))
    logger.info(f"Use GPU: {gpu} for training.")
    if is_distributed:
        logger.info(f"Distributed training enabled with {world_size} GPUs")
    else:
        logger.info("Running in single-GPU mode")

    # Initialize training dataset
    train_dataset = build('train_dataset', config)
    
    # Handle validation data setup
    # If validation_manifest is specified in config, use it for validation
    # Otherwise, skip validation
    val_dataset = None
    val_dataloader = None
    has_validation = False
    
    if hasattr(config, 'validation_manifest') and config.validation_manifest:
        # Use provided validation manifest
        if rank == 0:
            logger.info("Using provided validation manifest")
        val_dataset = build('validation_dataset', config)
        has_validation = True
    else:
        if rank == 0:
            logger.info("No validation manifest provided. Skipping validation.")
    
    # Setup training dataloader with appropriate sampler
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    config.train_dataloader['args']['batch_size'] = int(config.total_batch_size / world_size)
    if is_distributed:
        config.train_dataloader['args']['sampler'] = train_sampler
        config.train_dataloader['args']['shuffle'] = False
    else:
        config.train_dataloader['args']['shuffle'] = True

    train_dataloader = build('train_dataloader', config)
    
    # Setup validation dataloader only if validation dataset exists
    if has_validation:
        if is_distributed:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            config.validation_dataloader['args']['sampler'] = val_sampler
        else:
            val_sampler = None
            config.validation_dataloader['args']['shuffle'] = False
        
        config.validation_dataloader['args']['batch_size'] = int(config.total_batch_size / world_size)
        val_dataloader = build('validation_dataloader', config)

    # Build model components
    embedding_model = build('embedding_model', config)
    
    # Determine number of classes
    if hasattr(config, 'speed_pertub') and config.speed_pertub:
        config.num_classes = len(config.label_encoder) * 3
    else:
        config.num_classes = len(config.label_encoder)

    # Setup classifier and complete model
    config.classifier['args']['out_neurons'] = config.num_classes
    classifier = build('classifier', config)
    model = nn.Sequential(embedding_model, classifier)
    model = model.to('cuda')

    # Convert BatchNorm to SyncBatchNorm in distributed mode
    if is_distributed:
        model = convert_sync_batchnorm(model)
        logger.info("Converted BatchNorm to SyncBatchNorm for distributed training")
    
    # Wrap model with DDP only in distributed mode
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    if rank == 0:
        logger.info(f"Embedding model parameters: {sum(p.numel() for p in embedding_model.parameters()):,}")
        logger.info(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize optimizer and loss function
    config.optimizer['args']['params'] = model.parameters()
    optimizer = build('optimizer', config)
    criterion = build('loss', config)

    # Setup learning rate and margin schedulers
    config.lr_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    lr_scheduler = build('lr_scheduler', config)
    config.margin_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    margin_scheduler = build('margin_scheduler', config)

    # Setup training utilities
    epoch_counter = build('epoch_counter', config)
    checkpointer = build('checkpointer', config)
    epoch_logger = EpochLogger(save_file=os.path.join(config.exp_dir, 'train_epoch.log'))

    # Resume from checkpoint if requested
    if args.resume:
        checkpointer.recover_if_possible(device='cuda')
        
    # Enable CUDNN benchmark for better performance
    cudnn.benchmark = True

    # Log dataset information
    if rank == 0:
        step_per_epoch = len(train_dataloader)
        num_epochs = config.num_epochs
        num_training_steps = int(step_per_epoch) * int(num_epochs)
        
        logger.info("***** Running training *****")
        logger.info(f"  Num training examples = {len(train_dataset):,}")
        if has_validation:
            logger.info(f"  Num validation examples = {len(val_dataset):,}")
        logger.info(f"  Number of classes: {config.num_classes:,}")
        logger.info(f"  Number of Epochs = {config.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {int(config.total_batch_size / world_size)}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {config.total_batch_size}")
        logger.info(f"  Total optimization steps = {num_training_steps:,}")

    # Training loop
    for epoch in epoch_counter:
        # Set epoch for distributed sampler only in distributed mode
        if is_distributed:
            train_sampler.set_epoch(epoch)
            if has_validation and val_sampler:
                val_sampler.set_epoch(epoch)

        # Train one epoch
        train_stats = train(
            train_dataloader,
            model,
            criterion,
            optimizer,
            epoch,
            lr_scheduler,
            margin_scheduler,
            logger,
            config,
            rank,
        )
        
        # Initialize val_stats as an empty dict
        val_stats = {}
        
        # Validate after each epoch only if validation data is available
        if has_validation:
            val_stats = validate(
                val_dataloader,
                model,
                criterion,
                logger,
                config,
                rank,
                epoch
            )

        # Log results and save checkpoints (rank 0 only)
        if rank == 0:
            # Combine training and validation stats
            combined_stats = {**train_stats, **val_stats}
            
            # Log epoch statistics
            epoch_logger.log_stats(
                stats_meta={"epoch": epoch},
                stats=combined_stats,
                stage=''
            )
            
            # Save checkpoint at specified frequency or if best validation performance
            if epoch % config.save_epoch_freq == 0:
                checkpointer.save_checkpoint(epoch=epoch)
                
        # Synchronize processes in distributed mode only
        if is_distributed:
            torch_dist.barrier(device_ids=[gpu])

    if is_distributed and rank == 0:
        # Get model without DDP wrapper
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        
        # Convert SyncBatchNorm back to regular BatchNorm
        model = revert_sync_batchnorm(model)
        logger.info("Reverted SyncBatchNorm to BatchNorm for final model saving")
        
        # Save the final model with regular BatchNorm
        checkpointer.save_checkpoint(epoch=epoch)

    # Cleanup distributed training
    if is_distributed:
        torch_dist.barrier(device_ids=[gpu])
        if torch_dist.is_initialized():
            torch_dist.destroy_process_group()


if __name__ == '__main__':
    main()