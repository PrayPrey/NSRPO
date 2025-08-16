#!/usr/bin/env python3
"""
NSRPO Model Evaluation Script
Task 9: Implement Inference/Evaluation Script with Basic Metrics

Comprehensive evaluation script for NSRPO models with support for:
- Accuracy metrics (token-level, sequence-level)
- Perplexity calculation
- KL divergence analysis
- Training efficiency metrics
- Comparative evaluation against baselines
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from tqdm.auto import tqdm

from models import NSRPOModel, NullDecoder, create_nsrpo_model
from utils.dataset import get_dataloader, create_dummy_data


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class NSRPOEvaluator:
    """
    Comprehensive evaluator for NSRPO models.
    
    Provides evaluation metrics including accuracy, perplexity, KL divergence,
    and training efficiency comparisons.
    """
    
    def __init__(self, model, tokenizer, device: str = 'auto'):
        """
        Initialize evaluator.
        
        Args:
            model: NSRPO model or baseline model
            tokenizer: Tokenizer for the model
            device: Device to use for evaluation
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Set device
        if device == 'cpu' or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        elif device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Evaluation cache
        self.results_cache = {}
        
    def compute_accuracy_metrics(
        self, 
        dataloader: DataLoader,
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute token-level and sequence-level accuracy metrics.
        
        Args:
            dataloader: DataLoader with evaluation data
            max_batches: Maximum number of batches to evaluate
            
        Returns:
            Dictionary with accuracy metrics
        """
        total_tokens = 0
        correct_tokens = 0
        total_sequences = 0
        correct_sequences = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing accuracy")):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                    
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get model predictions
                if isinstance(self.model, NSRPOModel):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask')
                    )
                    logits = outputs.logits
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask')
                    )
                    logits = outputs.logits
                
                # Get predictions and targets
                predictions = torch.argmax(logits, dim=-1)
                # For accuracy computation, use input_ids as both input and target
                # This is standard for language modeling evaluation
                targets = batch['input_ids']
                
                # Ensure predictions and targets have the same shape
                min_length = min(predictions.size(1), targets.size(1))
                
                # Shift for next-token prediction if needed
                if min_length > 1:
                    # For autoregressive models, we predict the next token
                    predictions = predictions[:, :min_length-1].contiguous()
                    targets = targets[:, 1:min_length].contiguous()
                else:
                    # If sequences are too short, just truncate to same length
                    predictions = predictions[:, :min_length].contiguous()
                    targets = targets[:, :min_length].contiguous()
                
                # Create attention mask for valid tokens
                if 'attention_mask' in batch:
                    # Adjust attention mask to match the truncated length
                    if min_length > 1:
                        attention_mask = batch['attention_mask'][:, 1:min_length]
                    else:
                        attention_mask = batch['attention_mask'][:, :min_length]
                else:
                    attention_mask = torch.ones_like(targets)
                
                # Ensure all tensors have the same shape
                assert predictions.shape == targets.shape, f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
                assert predictions.shape == attention_mask.shape, f"Shape mismatch: predictions {predictions.shape} vs attention_mask {attention_mask.shape}"
                
                # Token-level accuracy
                valid_tokens = attention_mask.bool()
                correct_token_mask = (predictions == targets) & valid_tokens
                
                batch_correct_tokens = correct_token_mask.sum().item()
                batch_total_tokens = valid_tokens.sum().item()
                
                correct_tokens += batch_correct_tokens
                total_tokens += batch_total_tokens
                
                # Sequence-level accuracy
                sequence_correct = (correct_token_mask | ~valid_tokens).all(dim=1)
                correct_sequences += sequence_correct.sum().item()
                total_sequences += sequence_correct.size(0)
        
        # Calculate metrics
        token_accuracy = correct_tokens / max(total_tokens, 1)
        sequence_accuracy = correct_sequences / max(total_sequences, 1)
        
        return {
            'token_accuracy': token_accuracy,
            'sequence_accuracy': sequence_accuracy,
            'total_tokens': total_tokens,
            'total_sequences': total_sequences
        }
    
    def compute_perplexity(
        self, 
        dataloader: DataLoader,
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute perplexity metrics.
        
        Args:
            dataloader: DataLoader with evaluation data
            max_batches: Maximum number of batches to evaluate
            
        Returns:
            Dictionary with perplexity metrics
        """
        total_loss = 0.0
        total_tokens = 0
        losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing perplexity")):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                    
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                # Use input_ids as both input and target for language modeling
                # This is standard for autoregressive language models
                input_ids = batch['input_ids']
                attention_mask = batch.get('attention_mask')
                
                # For perplexity, we typically use the same sequence as input and target
                # The model will internally shift for next-token prediction
                if isinstance(self.model, NSRPOModel):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None)

                
                # Get loss from model output if available
                if outputs.loss is not None:
                    # Model already computed the loss correctly
                    loss = outputs.loss
                    losses.append(loss.item())
                    
                    # Count tokens for perplexity calculation
                    if attention_mask is not None:
                        n_tokens = attention_mask.sum().item()
                    else:
                        n_tokens = input_ids.numel()
                    
                    total_loss += loss.item() * n_tokens
                    total_tokens += n_tokens
                else:
                    # Calculate loss manually if not provided
                    logits = outputs.logits
                    if logits.size(1) > 1:
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = input_ids[:, 1:].contiguous()  # Use input_ids for labels
                    
                    # Create attention mask
                    if 'attention_mask' in batch:
                        shift_attention = batch['attention_mask'][:, 1:].contiguous()
                    else:
                        shift_attention = torch.ones_like(shift_labels)
                    
                    # Calculate loss per token
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                           shift_labels.view(-1))
                    token_losses = token_losses.view(shift_labels.shape)
                    
                    # Apply attention mask
                    valid_losses = token_losses * shift_attention.float()
                    batch_loss = valid_losses.sum()
                    batch_tokens = shift_attention.sum()
                    
                    total_loss += batch_loss.item()
                    total_tokens += batch_tokens.item()
                    
                    # Store per-sample losses for statistics
                    sample_losses = valid_losses.sum(dim=1) / shift_attention.sum(dim=1).clamp(min=1)
                    losses.extend(sample_losses.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Additional statistics
        losses = np.array(losses)
        loss_std = np.std(losses) if len(losses) > 1 else 0.0
        
        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'loss_std': loss_std,
            'total_tokens': total_tokens
        }
    
    def compute_kl_divergence(
        self, 
        baseline_model,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute KL divergence between model and baseline.
        
        Args:
            baseline_model: Baseline model for comparison
            dataloader: DataLoader with evaluation data
            max_batches: Maximum number of batches to evaluate
            temperature: Temperature for softmax
            
        Returns:
            Dictionary with KL divergence metrics
        """
        baseline_model.to(self.device)
        baseline_model.eval()
        
        total_kl_div = 0.0
        total_tokens = 0
        kl_values = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing KL divergence")):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                    
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get model predictions
                if isinstance(self.model, NSRPOModel):
                    model_outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask')
                    )
                    model_logits = model_outputs.logits
                else:
                    model_outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask')
                    )
                    model_logits = model_outputs.logits
                
                # Get baseline predictions
                baseline_outputs = baseline_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask')
                )
                baseline_logits = baseline_outputs.logits
                
                # Apply temperature
                model_logits = model_logits / temperature
                baseline_logits = baseline_logits / temperature
                
                # Convert to probabilities
                model_probs = F.softmax(model_logits, dim=-1)
                baseline_log_probs = F.log_softmax(baseline_logits, dim=-1)
                
                # Calculate KL divergence
                kl_div = F.kl_div(baseline_log_probs, model_probs, reduction='none').sum(dim=-1)
                
                # Apply attention mask
                if 'attention_mask' in batch:
                    attention_mask = batch['attention_mask'].float()
                else:
                    attention_mask = torch.ones(kl_div.shape, device=self.device)
                
                valid_kl = kl_div * attention_mask
                batch_kl = valid_kl.sum()
                batch_tokens = attention_mask.sum()
                
                total_kl_div += batch_kl.item()
                total_tokens += batch_tokens.item()
                
                # Store per-sample KL values
                sample_kl = valid_kl.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
                kl_values.extend(sample_kl.cpu().numpy())
        
        # Calculate metrics
        avg_kl_div = total_kl_div / max(total_tokens, 1)
        
        # Additional statistics
        kl_values = np.array(kl_values)
        kl_std = np.std(kl_values) if len(kl_values) > 1 else 0.0
        
        return {
            'kl_divergence': avg_kl_div,
            'kl_std': kl_std,
            'total_tokens': total_tokens
        }
    
    def compute_training_efficiency_metrics(
        self,
        dataloader: DataLoader,
        max_batches: int = 10
    ) -> Dict[str, float]:
        """
        Compute training efficiency metrics like gradient variance.
        
        Args:
            dataloader: DataLoader with training data
            max_batches: Maximum number of batches to process
            
        Returns:
            Dictionary with efficiency metrics
        """
        if not isinstance(self.model, NSRPOModel):
            return {'gradient_variance': 0.0, 'note': 'Not available for baseline model'}
        
        # Use the model's built-in method
        variance_metrics = self.model.get_policy_gradient_variance(
            dataloader, max_batches=max_batches
        )
        
        return variance_metrics
    
    def generate_samples(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text samples from the model.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate per prompt
            
        Returns:
            List of generated texts
        """
        generated_texts = []
        
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="Generating samples"):
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                
                # Generate
                if isinstance(self.model, NSRPOModel):
                    # Use base model for generation
                    base_model = self.model.base_model
                else:
                    base_model = self.model
                
                outputs = base_model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=inputs.input_ids.size(1) + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode generated texts
                for output in outputs:
                    generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                    # Remove the input prompt
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                    generated_texts.append(generated_text)
        
        return generated_texts
    
    def comprehensive_evaluation(
        self,
        eval_dataloader: DataLoader,
        baseline_model = None,
        max_batches: Optional[int] = None,
        include_generation: bool = False,
        generation_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation with all metrics.
        
        Args:
            eval_dataloader: DataLoader for evaluation
            baseline_model: Baseline model for KL divergence
            max_batches: Maximum batches to evaluate
            include_generation: Whether to include text generation
            generation_prompts: Prompts for text generation
            
        Returns:
            Complete evaluation results
        """
        results = {}
        
        # Basic metrics
        print("Computing accuracy metrics...")
        accuracy_metrics = self.compute_accuracy_metrics(eval_dataloader, max_batches)
        results['accuracy'] = accuracy_metrics
        
        print("Computing perplexity...")
        perplexity_metrics = self.compute_perplexity(eval_dataloader, max_batches)
        results['perplexity'] = perplexity_metrics
        
        # KL divergence (if baseline provided)
        if baseline_model is not None:
            print("Computing KL divergence...")
            kl_metrics = self.compute_kl_divergence(baseline_model, eval_dataloader, max_batches)
            results['kl_divergence'] = kl_metrics
        
        # Training efficiency (for NSRPO models)
        if isinstance(self.model, NSRPOModel):
            print("Computing training efficiency metrics...")
            efficiency_metrics = self.compute_training_efficiency_metrics(eval_dataloader)
            results['training_efficiency'] = efficiency_metrics
            
            # Null-space specific metrics
            null_space_info = self.model.get_null_space_info()
            results['null_space_info'] = null_space_info
        
        # Text generation (if requested)
        if include_generation and generation_prompts:
            print("Generating text samples...")
            generated_texts = self.generate_samples(generation_prompts)
            results['generation'] = {
                'prompts': generation_prompts,
                'generated_texts': generated_texts
            }
        
        # Add evaluation metadata
        results['metadata'] = {
            'model_type': type(self.model).__name__,
            'device': str(self.device),
            'evaluation_time': time.time()
        }
        
        return results


def load_model_from_checkpoint(
    checkpoint_path: str,
    base_model_path: str,
    tokenizer_path: str,
    device: str = 'auto',
    model_type: str = 'auto'
) -> Tuple[Any, Any]:
    """Load model and tokenizer from checkpoint."""
    logger = logging.getLogger(__name__)
    
    # 토크나이저
    if base_model_path == "dummy" or tokenizer_path == "dummy":
        # Use GPT-2 tokenizer for dummy model
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Determine model type
    if model_type == 'auto':
        # Auto-detect from checkpoint
        if 'model_config' in checkpoint and checkpoint['model_config'].get('model_type') == 'NSRPOModel':
            model_type = 'nspo'
        else:
            model_type = 'baseline'
        logger.info(f"Auto-detected model type: {model_type}")
    
    # Load model based on type
    if model_type in ['nspo', 'nsrpo']:
        # 베이스 모델
        if base_model_path == "dummy":
            base_model = create_dummy_model()
        else:
            base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

        # checkpoint에서 null_basis 크기를 확인
        state_dict = checkpoint['model_state_dict']
        
        # null_basis 차원 추출
        if 'null_decoder.null_basis' in state_dict:
            saved_null_basis = state_dict['null_decoder.null_basis']
            basis_hidden_size, null_dim = saved_null_basis.shape
            logger.info(f"Found null_basis in checkpoint with shape {saved_null_basis.shape}")
            
            # 저장된 null_basis 사용
            null_basis = saved_null_basis.contiguous()
        else:
            # Fallback: 랜덤 생성
            basis_hidden_size = base_model.config.hidden_size
            null_dim = 64
            logger.warning(f"null_basis not found in checkpoint, generating random basis with dim={null_dim}")
            q, _ = torch.linalg.qr(torch.randn(basis_hidden_size, null_dim))
            null_basis = q[:, :null_dim].contiguous()

        # 실제 hidden_size는 base_model에서 가져옴
        hidden_size = base_model.config.hidden_size
        
        # NullDecoder 직접 생성
        null_decoder_info = checkpoint['model_config'].get('null_decoder_info', {})
        null_decoder = NullDecoder(
            hidden_size=hidden_size,
            null_basis=null_basis,
            vocab_size=tokenizer.vocab_size,
            num_layers=null_decoder_info.get('num_layers', 3),
            nhead=null_decoder_info.get('nhead', 8),
            dropout=null_decoder_info.get('dropout', 0.1),
        )

        # NSRPOModel 조립 (손실 가중치 체크포인트에서 사용)
        loss_weights = checkpoint['model_config'].get('loss_weights', {})
        model = NSRPOModel(
            base_model=base_model,
            null_decoder=null_decoder,
            alpha_1=loss_weights.get('alpha_1', 0.1),
            alpha_2=loss_weights.get('alpha_2', 0.1),
            alpha_3=loss_weights.get('alpha_3', 0.05),
        )

        # state_dict 로드 (strict=False for dimension mismatches in dummy models)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        # 일반 CausalLM 체크포인트
        if base_model_path == "dummy":
            model = create_dummy_model()
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    return model, tokenizer



def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate NSRPO models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to model checkpoint or HuggingFace model'
    )
    parser.add_argument(
        '--model_type', type=str, default='auto',
        choices=['auto', 'nspo', 'nsrpo', 'baseline', 'grpo'],
        help='Type of model (auto-detect from checkpoint, nspo/nsrpo for NSRPO model, baseline/grpo for standard model)'
    )
    parser.add_argument(
        '--base_model_path', type=str, default=None,
        help='Path to base model (required for checkpoint evaluation)'
    )
    parser.add_argument(
        '--baseline_model_path', type=str, default=None,
        help='Path to baseline model for KL divergence comparison'
    )
    parser.add_argument(
        '--tokenizer_path', type=str, default=None,
        help='Path to tokenizer (defaults to model_path)'
    )
    
    # Evaluation configuration
    parser.add_argument(
        '--eval_data_path', type=str, default=None,
        help='Path to evaluation data'
    )
    parser.add_argument(
        '--max_batches', type=int, default=None,
        help='Maximum number of batches to evaluate'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Evaluation batch size'
    )
    parser.add_argument(
        '--max_length', type=int, default=512,
        help='Maximum sequence length'
    )
    
    # Generation configuration
    parser.add_argument(
        '--include_generation', action='store_true',
        help='Include text generation evaluation'
    )
    parser.add_argument(
        '--generation_prompts', type=str, nargs='+',
        default=["The future of AI is", "In the year 2030,", "Climate change will"],
        help='Prompts for text generation'
    )
    
    # Output configuration
    parser.add_argument(
        '--output_path', type=str, default='evaluation_results.json',
        help='Path to save evaluation results'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for evaluation'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--log_level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    # Fast execution flags
    parser.add_argument(
        '--fast', '--smoke', action='store_true',
        help='Fast smoke test mode (limit samples and batches)'
    )
    parser.add_argument(
        '--limit_eval_samples', type=int, default=None,
        help='Limit the number of evaluation samples'
    )
    parser.add_argument(
        '--cpu_only', action='store_true',
        help='Force CPU-only execution'
    )
    
    args = parser.parse_args()
    
    # Apply fast/smoke mode settings
    if args.fast:
        args.max_batches = 5
        args.limit_eval_samples = 50
        args.batch_size = min(args.batch_size, 2)
        args.include_generation = False  # Skip generation in fast mode
        print("Fast/smoke mode enabled: limiting to 5 batches, 50 samples")
    
    # Apply CPU-only mode
    if args.cpu_only:
        args.device = 'cpu'
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


def main():
    """Main evaluation function."""
    args = parse_arguments()
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting NSRPO model evaluation")
    
    try:
        # Load model and tokenizer
        logger.info(f"Loading model from {args.model_path}")
        
        # Check for dummy model (offline testing)
        if args.model_path == "dummy":
            logger.info("Using dummy model for offline testing")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = create_dummy_model()
        # Determine if this is a checkpoint or HuggingFace model
        elif args.model_path.endswith('.pt') or args.model_path.endswith('.pth'):
            # Checkpoint
            if not args.base_model_path:
                raise ValueError("--base_model_path required for checkpoint evaluation")
            
            tokenizer_path = args.tokenizer_path or args.base_model_path
            model, tokenizer = load_model_from_checkpoint(
                args.model_path, args.base_model_path, tokenizer_path, args.device, args.model_type
            )
        else:
            # HuggingFace model
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(args.model_path)
        
        # Load baseline model if specified
        baseline_model = None
        if args.baseline_model_path:
            logger.info(f"Loading baseline model from {args.baseline_model_path}")
            baseline_model = AutoModelForCausalLM.from_pretrained(args.baseline_model_path)
        
        # Create evaluator
        evaluator = NSRPOEvaluator(model, tokenizer, args.device)
        
        # Create evaluation dataloader
        if args.eval_data_path and os.path.exists(args.eval_data_path):
            logger.info(f"Loading evaluation data from {args.eval_data_path}")
            eval_dataloader = get_dataloader(
                data_path=args.eval_data_path,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                shuffle=False
            )
        else:
            num_samples = args.limit_eval_samples or 500
            logger.info(f"Creating {num_samples} dummy evaluation samples")
            dummy_data = create_dummy_data(num_samples)
            eval_dataloader = get_dataloader(
                data=dummy_data,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                shuffle=False
            )
        
        # Run comprehensive evaluation
        logger.info("Running comprehensive evaluation")
        results = evaluator.comprehensive_evaluation(
            eval_dataloader=eval_dataloader,
            baseline_model=baseline_model,
            max_batches=args.max_batches,
            include_generation=args.include_generation,
            generation_prompts=args.generation_prompts
        )
        
        # Save results
        logger.info(f"Saving results to {args.output_path}")
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print("\n=== EVALUATION RESULTS SUMMARY ===")
        if 'accuracy' in results:
            acc = results['accuracy']
            print(f"Token Accuracy: {acc['token_accuracy']:.4f}")
            print(f"Sequence Accuracy: {acc['sequence_accuracy']:.4f}")
        
        if 'perplexity' in results:
            ppl = results['perplexity']
            print(f"Perplexity: {ppl['perplexity']:.2f}")
            print(f"Average Loss: {ppl['avg_loss']:.4f}")
        
        if 'kl_divergence' in results:
            kl = results['kl_divergence']
            print(f"KL Divergence: {kl['kl_divergence']:.4f}")
        
        if 'training_efficiency' in results:
            eff = results['training_efficiency']
            print(f"Gradient Variance: {eff['variance']:.6f}")
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()
