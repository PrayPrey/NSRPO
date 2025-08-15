#!/usr/bin/env python3
"""
GRPO+Null-Decoder Training Script
Task 7: Implement GRPO+Null-Decoder Training Script

Main training script for NSRPO (Null-Space Regularized Policy Optimization) model.
Supports both baseline GRPO training and GRPO with Null-Space Decoder regularization.
"""

import argparse
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time
import traceback
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
    get_scheduler,
    set_seed
)
from tqdm.auto import tqdm

# Import our custom modules
from models import NSRPOModel, NullDecoder, create_nsrpo_model
from utils.dataset import get_dataloader, create_dummy_data
from utils.svd_utils import extract_lora_null_basis, extract_base_null_basis
from utils.paths import (
    get_outputs_dir, get_checkpoints_dir, get_logs_dir,
    get_timestamped_filename, get_checkpoint_path, get_output_path
)


def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Task 7.5: Implement Checkpointing and Logging (component)
    Set up logging configuration.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger


def parse_arguments() -> argparse.Namespace:
    """
    Task 7.1: Implement Command-Line Argument Parsing and Configuration
    Parse command line arguments for the training script.
    """
    parser = argparse.ArgumentParser(
        description='Train GRPO model with Null-Space Decoder regularization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to base model (HuggingFace model name or local path)'
    )
    parser.add_argument(
        '--use_null_decoder', action='store_true',
        help='Whether to use Null-Space Decoder regularization'
    )
    parser.add_argument(
        '--null_basis_path', type=str, default=None,
        help='Path to saved null-basis tensor (required if use_null_decoder=True)'
    )
    parser.add_argument(
        '--extract_null_basis', action='store_true',
        help='Extract null-basis from base model if not provided'
    )
    
    # Null-decoder specific parameters
    parser.add_argument(
        '--decoder_layers', type=int, default=3,
        help='Number of transformer decoder layers in null decoder'
    )
    parser.add_argument(
        '--decoder_heads', type=int, default=8,
        help='Number of attention heads in null decoder'
    )
    parser.add_argument(
        '--decoder_dropout', type=float, default=0.1,
        help='Dropout rate for null decoder'
    )
    
    # Loss weighting parameters (as specified in PRD)
    parser.add_argument(
        '--alpha_1', type=float, default=0.1,
        help='Weight for Cross-Entropy loss (CE loss)'
    )
    parser.add_argument(
        '--alpha_2', type=float, default=0.1, 
        help='Weight for Cosine similarity loss'
    )
    parser.add_argument(
        '--alpha_3', type=float, default=0.05,
        help='Weight for Norm preservation loss'
    )
    
    # Training parameters
    parser.add_argument(
        '--learning_rate', type=float, default=5e-5,
        help='Learning rate for optimizer'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Training batch size'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--max_length', type=int, default=512,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--warmup_ratio', type=float, default=0.1,
        help='Ratio of warmup steps to total training steps'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.01,
        help='Weight decay for optimizer'
    )
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=1,
        help='Number of steps to accumulate gradients'
    )
    parser.add_argument(
        '--max_grad_norm', type=float, default=1.0,
        help='Maximum gradient norm for clipping'
    )
    
    # Data parameters
    parser.add_argument(
        '--train_data_path', type=str, default=None,
        help='Path to training data (will create dummy data if not provided)'
    )
    parser.add_argument(
        '--eval_data_path', type=str, default=None,
        help='Path to evaluation data (will create dummy data if not provided)'
    )
    parser.add_argument(
        '--num_dummy_samples', type=int, default=1000,
        help='Number of dummy samples to create if no data path provided'
    )
    
    # Output and logging parameters  
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory to save outputs, logs, and checkpoints (default: auto-generated)'
    )
    parser.add_argument(
        '--save_every', type=int, default=1,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--eval_every', type=int, default=1,
        help='Run evaluation every N epochs'
    )
    parser.add_argument(
        '--log_every', type=int, default=100,
        help='Log training progress every N steps'
    )
    parser.add_argument(
        '--log_level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    # Scheduler parameters
    parser.add_argument(
        '--scheduler_type', type=str, default='linear',
        choices=['linear', 'cosine', 'constant'],
        help='Learning rate scheduler type'
    )
    
    # Experiment parameters
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--fp16', action='store_true',
        help='Use mixed precision training'
    )
    parser.add_argument(
        '--resume_from_checkpoint', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    
    # Fast execution flags
    parser.add_argument(
        '--cpu_only', action='store_true',
        help='Force CPU-only execution (overrides --device)'
    )
    parser.add_argument(
        '--max_steps', type=int, default=None,
        help='Maximum number of training steps (overrides --num_epochs)'
    )
    parser.add_argument(
        '--limit_train_samples', type=int, default=None,
        help='Limit the number of training samples'
    )
    parser.add_argument(
        '--limit_eval_samples', type=int, default=None,
        help='Limit the number of evaluation samples'
    )
    parser.add_argument(
        '--fast', '--smoke', action='store_true',
        help='Fast smoke test mode (small samples, few steps)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_null_decoder and not args.null_basis_path and not args.extract_null_basis:
        parser.error("--null_basis_path is required when --use_null_decoder is True, or use --extract_null_basis")
    
    # Apply fast/smoke mode settings
    if args.fast:
        args.num_epochs = min(args.num_epochs, 1)
        args.max_steps = 10
        args.limit_train_samples = 20
        args.limit_eval_samples = 10
        args.batch_size = min(args.batch_size, 2)
        args.log_every = 1
        print("Fast/smoke mode enabled: limiting to 10 steps, 20 train samples, 10 eval samples")
    
    # Apply CPU-only mode
    if args.cpu_only:
        args.device = 'cpu'
        args.fp16 = False  # Disable mixed precision for CPU
        print("CPU-only mode enabled")
    
    return args


def create_dummy_model():
    """Create a dummy model for offline testing"""
    import torch.nn as nn
    from types import SimpleNamespace
    
    class DummyGPT2(nn.Module):
        """Minimal dummy model for testing"""
        def __init__(self, vocab_size=50257, hidden_size=768):
            super().__init__()
            self.config = SimpleNamespace(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                n_layer=12,
                n_head=12
            )
            self.transformer = nn.Linear(hidden_size, hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)
            
        def forward(self, input_ids, attention_mask=None, labels=None, 
                    output_hidden_states=False, output_attentions=False, 
                    return_dict=True, **kwargs):
            batch_size, seq_len = input_ids.shape
            hidden_size = self.config.hidden_size
            
            # Dummy hidden states
            hidden_states = torch.randn(batch_size, seq_len, hidden_size)
            hidden_states = self.transformer(hidden_states)
            logits = self.lm_head(hidden_states)
            
            loss = None
            if labels is not None:
                # Simple cross-entropy loss
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            
            # Return in transformers format
            outputs = SimpleNamespace(
                logits=logits, 
                loss=loss,
                hidden_states=[hidden_states] if output_hidden_states else None,
                attentions=None
            )
            return outputs
        
        def generate(self, *args, **kwargs):
            """Dummy generate for compatibility"""
            # Return input_ids repeated
            input_ids = args[0] if args else kwargs.get('input_ids')
            return input_ids.repeat(1, 2)  # Just double the input
    
    return DummyGPT2()


def load_model_and_tokenizer(args: argparse.Namespace, logger: logging.Logger) -> tuple:
    """
    Task 7.2: Implement Model Initialization and Data Loading (component)
    Load and initialize the model and tokenizer.
    """
    # Check for dummy model
    if args.model_path == "dummy":
        logger.info("Using dummy model for offline testing")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        base_model = create_dummy_model()
    else:
        logger.info(f"Loading tokenizer from {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        logger.info(f"Loading base model from {args.model_path}")
        # Load model with CPU-only consideration
        model_kwargs = {
            'torch_dtype': torch.float16 if args.fp16 and not args.cpu_only else torch.float32,
        }
        
        if not args.cpu_only and args.device == 'auto':
            model_kwargs['device_map'] = 'auto'
        
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            **model_kwargs
        )
    
    # For CPU-only mode, ensure model is on CPU
    if args.cpu_only:
        base_model = base_model.to('cpu')
    
    # Determine device
    if args.cpu_only:
        device = torch.device('cpu')
    elif args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    if args.use_null_decoder:
        logger.info("Initializing NSRPO model with Null-Space Decoder")
    
        # null_basis_path 확정
        if args.null_basis_path and os.path.exists(args.null_basis_path):
            logger.info(f"Using null-basis from {args.null_basis_path}")
            null_basis_path = args.null_basis_path
        elif args.extract_null_basis:
            logger.info("Extracting null-basis from base model")
            try:
                null_basis = extract_lora_null_basis(base_model, rank=16, epsilon_factor=1e-3)
                logger.info(f"Extracted LoRA null-basis with shape {null_basis.shape}")
            except Exception as e:
                logger.warning(f"LoRA extraction failed: {e}, trying base model extraction")
                null_basis = extract_base_null_basis(base_model, epsilon_factor=1e-3)
                logger.info(f"Extracted base null-basis with shape {null_basis.shape}")
    
            # Save extracted null-basis
            null_basis_save_path = get_checkpoint_path('extracted_null_basis', timestamped=True)
            torch.save(null_basis, null_basis_save_path)
            logger.info(f"Saved extracted null-basis to {null_basis_save_path}")
            null_basis_path = str(null_basis_save_path)
        else:
            raise ValueError("No null-basis provided and extraction not requested")
    
        # Create NSRPO model (경로를 직접 넘김)
        model = create_nsrpo_model(
            base_model=base_model,
            null_basis_path=null_basis_path,
            vocab_size=tokenizer.vocab_size,
            hidden_size=getattr(base_model.config, 'hidden_size', None),
            num_layers=args.decoder_layers,
            nhead=args.decoder_heads,
            dropout=args.decoder_dropout,
            alpha_1=args.alpha_1,
            alpha_2=args.alpha_2,
            alpha_3=args.alpha_3
        )
    
        logger.info(f"Created NSRPO model using null-basis from {null_basis_path}")

    else:
        logger.info("Using baseline GRPO model (no Null-Space Decoder)")
        model = base_model
    
    # Move model to device if not already done
    if args.device != 'auto':
        model = model.to(device)
    
    return model, tokenizer, device


def create_dataloaders(args: argparse.Namespace, tokenizer, logger: logging.Logger) -> tuple:
    """
    Task 7.2: Implement Model Initialization and Data Loading (component)
    Create training and evaluation dataloaders.
    """
    # Prepare data
    if args.train_data_path:
        # Check if it's a HuggingFace dataset
        if args.train_data_path.startswith('hf:'):
            logger.info(f"Loading HuggingFace dataset: {args.train_data_path}")
            train_dataloader = get_dataloader(
                data_path=args.train_data_path,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                shuffle=True,
                num_samples=args.limit_train_samples
            )
        elif os.path.exists(args.train_data_path):
            logger.info(f"Loading training data from {args.train_data_path}")
            train_dataloader = get_dataloader(
                data_path=args.train_data_path,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                shuffle=True
            )
        else:
            logger.warning(f"Training data path not found: {args.train_data_path}, using dummy data")
            num_samples = args.limit_train_samples or args.num_dummy_samples
            logger.info(f"Creating {num_samples} dummy training samples")
            dummy_train_data = create_dummy_data(num_samples)
            train_dataloader = get_dataloader(
                data=dummy_train_data,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                shuffle=True
            )
    else:
        num_samples = args.limit_train_samples or args.num_dummy_samples
        logger.info(f"Creating {num_samples} dummy training samples")
        dummy_train_data = create_dummy_data(num_samples)
        train_dataloader = get_dataloader(
            data=dummy_train_data,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            shuffle=True
        )
    
    if args.eval_data_path:
        # Check if it's a HuggingFace dataset
        if args.eval_data_path.startswith('hf:'):
            logger.info(f"Loading HuggingFace dataset for evaluation: {args.eval_data_path}")
            eval_dataloader = get_dataloader(
                data_path=args.eval_data_path,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                shuffle=False,
                num_samples=args.limit_eval_samples
            )
        elif os.path.exists(args.eval_data_path):
            logger.info(f"Loading evaluation data from {args.eval_data_path}")
            eval_dataloader = get_dataloader(
                data_path=args.eval_data_path,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                shuffle=False
            )
        else:
            logger.warning(f"Evaluation data path not found: {args.eval_data_path}, using dummy data")
            num_eval_samples = args.limit_eval_samples or 200
            logger.info(f"Creating {num_eval_samples} dummy evaluation samples")
            dummy_eval_data = create_dummy_data(num_eval_samples)
            eval_dataloader = get_dataloader(
                data=dummy_eval_data,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                shuffle=False
            )
    else:
        num_eval_samples = args.limit_eval_samples or 200
        logger.info(f"Creating {num_eval_samples} dummy evaluation samples")
        dummy_eval_data = create_dummy_data(num_eval_samples)
        eval_dataloader = get_dataloader(
            data=dummy_eval_data,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            shuffle=False
        )
    
    logger.info(f"Created dataloaders: train={len(train_dataloader)} batches, "
                f"eval={len(eval_dataloader)} batches")
    
    return train_dataloader, eval_dataloader


def create_optimizer_and_scheduler(model, args: argparse.Namespace, num_training_steps: int):
    """
    Create optimizer and learning rate scheduler.
    """
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    if args.scheduler_type == 'linear':
        scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif args.scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    else:  # constant
        scheduler = get_scheduler(
            name='constant',
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    return optimizer, scheduler


def evaluate_model(model, eval_dataloader, device, logger: logging.Logger) -> Dict[str, float]:
    """
    Task 7.4: Implement Evaluation During Training
    Evaluate the model on the evaluation dataset.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    loss_components = {}
    
    eval_progress = tqdm(eval_dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch in eval_progress:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if isinstance(model, NSRPOModel):
                # Use input_ids as labels if response_input_ids not available or mismatched
                labels = batch.get('response_input_ids', batch['input_ids'])
                # Ensure labels and input_ids have compatible shapes
                if labels.shape[1] != batch['input_ids'].shape[1]:
                    # Use input_ids as labels for consistent training
                    labels = batch['input_ids']
                    logger.debug("Using input_ids as labels due to shape mismatch")
                
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=labels
                )
                
                # Accumulate loss components
                if outputs.loss_components:
                    for key, value in outputs.loss_components.items():
                        if key not in loss_components:
                            loss_components[key] = 0.0
                        loss_components[key] += value.item()
            else:
                # Baseline model
                labels = batch.get('response_input_ids', batch['input_ids'])
                # Ensure labels and input_ids have compatible shapes
                if labels.shape[1] != batch['input_ids'].shape[1]:
                    # Use input_ids as labels for consistent training
                    labels = batch['input_ids']
                    logger.debug("Using input_ids as labels due to shape mismatch")
                
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=labels
                )
            
            if outputs.loss is not None:
                total_loss += outputs.loss.item()
                total_samples += 1
            
            eval_progress.set_postfix({'loss': f"{total_loss/max(total_samples, 1):.4f}"})
    
    # Calculate averages
    avg_loss = total_loss / max(total_samples, 1)
    avg_loss_components = {k: v / max(total_samples, 1) for k, v in loss_components.items()}
    
    eval_metrics = {
        'eval_loss': avg_loss,
        **avg_loss_components
    }
    
    logger.info(f"Evaluation completed. Average loss: {avg_loss:.4f}")
    if avg_loss_components:
        component_str = ", ".join(f"{k}: {v:.4f}" for k, v in avg_loss_components.items())
        logger.info(f"Loss components: {component_str}")
    
    return eval_metrics


def save_checkpoint(model, optimizer, scheduler, epoch: int, step: int, args: argparse.Namespace, 
                   metrics: Dict[str, float], logger: logging.Logger):
    """
    Task 7.5: Implement Checkpointing and Logging (component)
    Save model checkpoint.
    """
    checkpoint_dir = get_checkpoints_dir() / f"checkpoint-epoch-{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model state
    model_path = checkpoint_dir / "model.pt"
    if isinstance(model, NSRPOModel):
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'args': vars(args),
            'model_config': model.get_null_space_info()
        }, model_path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'args': vars(args)
        }, model_path)
    
    # Save optimizer and scheduler states
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    
    # Save metrics
    with open(checkpoint_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Checkpoint saved to {checkpoint_dir}")


def train_epoch(model, train_dataloader, optimizer, scheduler, device, 
               args: argparse.Namespace, epoch: int, global_step: int, 
               writer: SummaryWriter, logger: logging.Logger) -> tuple:
    """
    Task 7.3: Implement Training Loop with Optimization Steps
    Train model for one epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    accumulated_loss = 0.0
    loss_components_acc = {}
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        if isinstance(model, NSRPOModel):
            # Use input_ids as labels if response_input_ids not available or mismatched
            labels = batch.get('response_input_ids', batch['input_ids'])
            # Ensure labels and input_ids have compatible shapes
            if labels.shape[1] != batch['input_ids'].shape[1]:
                # Use input_ids as labels for consistent training
                labels = batch['input_ids']
                # Don't log in training loop to avoid spam
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                labels=labels
            )
            
            # Accumulate loss components for logging
            if outputs.loss_components:
                for key, value in outputs.loss_components.items():
                    if key not in loss_components_acc:
                        loss_components_acc[key] = 0.0
                    loss_components_acc[key] += value.item()
        else:
            # Baseline model
            labels = batch.get('response_input_ids', batch['input_ids'])
            # Ensure labels and input_ids have compatible shapes
            if labels.shape[1] != batch['input_ids'].shape[1]:
                # Use input_ids as labels for consistent training
                labels = batch['input_ids']
                # Don't log in training loop to avoid spam
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                labels=labels
            )
        
        loss = outputs.loss
        if loss is None:
            continue
            
        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        if args.fp16:
            # For mixed precision training (simplified)
            loss.backward()
        else:
            loss.backward()
        
        accumulated_loss += loss.item()
        total_loss += loss.item() * args.gradient_accumulation_steps
        total_samples += 1
        
        # Optimization step
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Check max_steps limit
            if args.max_steps and global_step >= args.max_steps:
                logger.info(f"Reached max_steps limit ({args.max_steps}). Stopping training.")
                # Create epoch_metrics before returning
                epoch_metrics = {
                    'train_loss': total_loss / max(total_samples, 1),
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                if loss_components_acc:
                    for key, value in loss_components_acc.items():
                        epoch_metrics[f'train_{key}'] = value / max(total_samples, 1)
                return global_step, epoch_metrics
            
            # Logging
            if global_step % args.log_every == 0:
                avg_loss = total_loss / max(total_samples, 1)
                current_lr = scheduler.get_last_lr()[0]
                
                # Log to tensorboard
                writer.add_scalar('train/loss', avg_loss, global_step)
                writer.add_scalar('train/learning_rate', current_lr, global_step)
                
                if loss_components_acc:
                    avg_components = {k: v / max(total_samples, 1) 
                                    for k, v in loss_components_acc.items()}
                    for key, value in avg_components.items():
                        writer.add_scalar(f'train/{key}', value, global_step)
                
                logger.info(f"Step {global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")
            
            accumulated_loss = 0.0
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{total_loss/max(total_samples, 1):.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    # Final metrics for the epoch
    epoch_metrics = {
        'train_loss': total_loss / max(total_samples, 1),
        'learning_rate': scheduler.get_last_lr()[0]
    }
    
    if loss_components_acc:
        avg_components = {k: v / max(total_samples, 1) for k, v in loss_components_acc.items()}
        epoch_metrics.update(avg_components)
    
    return global_step, epoch_metrics


def train(args: argparse.Namespace):
    """
    Main training function.
    """
    # Setup
    set_seed(args.seed)
    
    # Use provided output_dir or create timestamped one
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = get_outputs_dir() / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(output_dir)  # Update args for consistency
    
    logger = setup_logging(output_dir, args.log_level)
    logger.info("Starting NSRPO training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(output_dir / 'runs')
    
    try:
        # Load model and tokenizer
        model, tokenizer, device = load_model_and_tokenizer(args, logger)
        
        # Create dataloaders
        train_dataloader, eval_dataloader = create_dataloaders(args, tokenizer, logger)
        
        # Create optimizer and scheduler
        if args.max_steps:
            num_training_steps = args.max_steps
            effective_epochs = (args.max_steps * args.gradient_accumulation_steps + len(train_dataloader) - 1) // len(train_dataloader)
            args.num_epochs = min(args.num_epochs, effective_epochs)
        else:
            num_training_steps = args.num_epochs * len(train_dataloader) // args.gradient_accumulation_steps
        
        optimizer, scheduler = create_optimizer_and_scheduler(model, args, num_training_steps)
        
        logger.info(f"Total training steps: {num_training_steps}")
        
        # Resume from checkpoint if specified
        start_epoch = 0
        global_step = 0
        
        if args.resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['step']
            logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
        
        # Training loop
        logger.info("Starting training loop")
        
        for epoch in range(start_epoch, args.num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
            
            # Train for one epoch
            global_step, train_metrics = train_epoch(
                model, train_dataloader, optimizer, scheduler, device,
                args, epoch, global_step, writer, logger
            )
            
            # Evaluation
            if (epoch + 1) % args.eval_every == 0:
                logger.info("Running evaluation")
                eval_metrics = evaluate_model(model, eval_dataloader, device, logger)
                
                # Log evaluation metrics
                for key, value in eval_metrics.items():
                    writer.add_scalar(f'eval/{key}', value, global_step)
                
                # Combined metrics
                all_metrics = {**train_metrics, **eval_metrics}
            else:
                all_metrics = train_metrics
            
            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(model, optimizer, scheduler, epoch + 1, 
                              global_step, args, all_metrics, logger)
        
        # Save final model in checkpoint format for compatibility with evaluate.py
        final_model_path = output_dir / "model_final.pt"
        if isinstance(model, NSRPOModel):
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': model.get_null_space_info(),
                'args': vars(args)
            }, final_model_path)
        else:
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': vars(args)
            }, final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Calculate final policy gradient variance if using NSRPO
        if isinstance(model, NSRPOModel):
            logger.info("Calculating policy gradient variance")
            variance_metrics = model.get_policy_gradient_variance(
                train_dataloader, max_batches=10
            )
            logger.info(f"Policy gradient variance: {variance_metrics}")
            
            # Save variance metrics
            with open(output_dir / 'variance_metrics.json', 'w') as f:
                json.dump(variance_metrics, f, indent=2)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    finally:
        writer.close()


def main():
    """Main entry point for the training script."""
    args = parse_arguments()
    train(args)


if __name__ == '__main__':
    main()
