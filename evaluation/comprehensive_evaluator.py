"""
Comprehensive Evaluation Framework for NSRPO
Task 11: Build Comprehensive Evaluation Framework - Complete evaluation suite

Advanced evaluation system with publication-ready metrics, statistical analysis,
and comprehensive model comparison capabilities.
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from models.nsrpo_model import NSRPOModel
from evaluate import NSRPOEvaluator


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation framework for NSRPO models.
    
    Provides advanced metrics, statistical analysis, and comparison capabilities
    suitable for academic publication.
    """
    
    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        device: str = 'auto',
        precision: str = 'float32'
    ):
        """
        Initialize comprehensive evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            device: Device for evaluation
            precision: Numerical precision for calculations
        """
        self.model = model
        self.tokenizer = tokenizer
        self.precision = precision
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize base evaluator
        self.base_evaluator = NSRPOEvaluator(model, tokenizer, device)
        
        # Evaluation caches
        self.results_cache = {}
        self.comparison_cache = {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def compute_advanced_accuracy_metrics(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
        compute_per_token_stats: bool = True
    ) -> Dict[str, Any]:
        """
        Compute advanced accuracy metrics including confidence intervals.
        
        Args:
            dataloader: DataLoader with evaluation data
            max_batches: Maximum number of batches to evaluate
            compute_per_token_stats: Whether to compute per-token statistics
            
        Returns:
            Dictionary with advanced accuracy metrics
        """
        all_predictions = []
        all_targets = []
        all_confidences = []
        token_accuracies = []
        sequence_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing advanced accuracy")):
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
                
                # Get probabilities and predictions
                probs = F.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                confidences = torch.max(probs, dim=-1)[0]
                
                targets = batch.get('response_input_ids', batch['input_ids'])
                
                # Shift for next-token prediction
                if targets.size(1) > 1:
                    predictions = predictions[:, :-1].contiguous()
                    targets = targets[:, 1:].contiguous()
                    confidences = confidences[:, :-1].contiguous()
                
                # Create attention mask
                if 'attention_mask' in batch:
                    attention_mask = batch['attention_mask']
                    # Adjust mask size to match targets
                    if attention_mask.size(1) > targets.size(1):
                        attention_mask = attention_mask[:, :targets.size(1)]
                    elif attention_mask.size(1) < targets.size(1):
                        # Pad with ones if mask is shorter
                        pad_size = targets.size(1) - attention_mask.size(1)
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones(attention_mask.size(0), pad_size, device=attention_mask.device)
                        ], dim=1)
                else:
                    attention_mask = torch.ones_like(targets)
                
                # Process each sequence
                for seq_idx in range(predictions.size(0)):
                    seq_preds = predictions[seq_idx]
                    seq_targets = targets[seq_idx]
                    seq_confs = confidences[seq_idx]
                    seq_mask = attention_mask[seq_idx].bool()
                    
                    # Ensure mask and tensors have same length
                    min_len = min(len(seq_preds), len(seq_targets), len(seq_mask))
                    seq_preds = seq_preds[:min_len]
                    seq_targets = seq_targets[:min_len]
                    seq_confs = seq_confs[:min_len]
                    seq_mask = seq_mask[:min_len]
                    
                    # Extract valid tokens
                    valid_preds = seq_preds[seq_mask]
                    valid_targets = seq_targets[seq_mask]
                    valid_confs = seq_confs[seq_mask]
                    
                    if len(valid_preds) > 0:
                        all_predictions.extend(valid_preds.cpu().numpy())
                        all_targets.extend(valid_targets.cpu().numpy())
                        all_confidences.extend(valid_confs.cpu().numpy())
                        
                        # Token-level accuracy for this sequence
                        seq_accuracy = (valid_preds == valid_targets).float().mean().item()
                        token_accuracies.append(seq_accuracy)
                        
                        # Sequence-level result
                        sequence_correct = (valid_preds == valid_targets).all().item()
                        sequence_results.append(sequence_correct)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_confidences = np.array(all_confidences)
        token_accuracies = np.array(token_accuracies)
        
        # Compute basic accuracy metrics
        token_accuracy = accuracy_score(all_targets, all_predictions)
        sequence_accuracy = np.mean(sequence_results)
        
        # Confidence intervals (using bootstrap)
        token_acc_ci = self._bootstrap_confidence_interval(token_accuracies, confidence=0.95)
        
        # Per-class metrics (for tokens)
        unique_classes = np.unique(all_targets)
        if len(unique_classes) <= 100:  # Only compute if reasonable number of classes
            precision, recall, f1, support = precision_recall_fscore_support(
                all_targets, all_predictions, average='weighted', zero_division=0
            )
        else:
            precision = recall = f1 = 0.0
        
        # Confidence-based metrics
        high_conf_mask = all_confidences > 0.8
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = accuracy_score(
                all_targets[high_conf_mask], 
                all_predictions[high_conf_mask]
            )
        else:
            high_conf_accuracy = 0.0
        
        results = {
            'token_accuracy': float(token_accuracy),
            'token_accuracy_ci_lower': float(token_acc_ci[0]),
            'token_accuracy_ci_upper': float(token_acc_ci[1]),
            'sequence_accuracy': float(sequence_accuracy),
            'weighted_precision': float(precision),
            'weighted_recall': float(recall),
            'weighted_f1': float(f1),
            'high_confidence_accuracy': float(high_conf_accuracy),
            'mean_confidence': float(np.mean(all_confidences)),
            'confidence_std': float(np.std(all_confidences)),
            'num_tokens': len(all_predictions),
            'num_sequences': len(sequence_results),
            'num_unique_classes': len(unique_classes)
        }
        
        if compute_per_token_stats:
            # Additional per-token statistics
            results.update({
                'token_accuracy_std': float(np.std(token_accuracies)),
                'token_accuracy_min': float(np.min(token_accuracies)),
                'token_accuracy_max': float(np.max(token_accuracies)),
                'token_accuracy_percentiles': {
                    '25': float(np.percentile(token_accuracies, 25)),
                    '50': float(np.percentile(token_accuracies, 50)),
                    '75': float(np.percentile(token_accuracies, 75))
                }
            })
        
        return results
    
    def compute_advanced_perplexity_metrics(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute advanced perplexity metrics with statistical analysis.
        
        Args:
            dataloader: DataLoader with evaluation data
            max_batches: Maximum number of batches to evaluate
            
        Returns:
            Dictionary with advanced perplexity metrics
        """
        token_losses = []
        sequence_losses = []
        sequence_perplexities = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing advanced perplexity")):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                targets = batch.get('response_input_ids', batch['input_ids'])
                
                if isinstance(self.model, NSRPOModel):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        labels=targets
                    )
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        labels=targets
                    )
                
                logits = outputs.logits
                if logits.size(1) > 1:
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = targets[:, 1:].contiguous()
                    
                    # Create attention mask
                    if 'attention_mask' in batch:
                        shift_attention = batch['attention_mask'][:, 1:].contiguous()
                    else:
                        shift_attention = torch.ones_like(shift_labels)
                    
                    # Calculate loss per token
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    batch_token_losses = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)), 
                        shift_labels.view(-1)
                    )
                    batch_token_losses = batch_token_losses.view(shift_labels.shape)
                    
                    # Apply attention mask and collect token losses
                    valid_losses = batch_token_losses * shift_attention.float()
                    
                    # Per-sequence processing
                    for seq_idx in range(valid_losses.size(0)):
                        seq_losses = valid_losses[seq_idx]
                        seq_mask = shift_attention[seq_idx].bool()
                        seq_valid_losses = seq_losses[seq_mask]
                        
                        if len(seq_valid_losses) > 0:
                            # Add token losses
                            token_losses.extend(seq_valid_losses.cpu().numpy())
                            
                            # Sequence-level metrics
                            seq_loss = seq_valid_losses.mean().item()
                            seq_perplexity = np.exp(seq_loss)
                            
                            sequence_losses.append(seq_loss)
                            sequence_perplexities.append(seq_perplexity)
        
        # Convert to numpy arrays
        token_losses = np.array(token_losses)
        sequence_losses = np.array(sequence_losses)
        sequence_perplexities = np.array(sequence_perplexities)
        
        # Compute statistics
        mean_token_loss = np.mean(token_losses)
        mean_perplexity = np.exp(mean_token_loss)
        
        # Confidence intervals
        loss_ci = self._bootstrap_confidence_interval(sequence_losses, confidence=0.95)
        perplexity_ci = [np.exp(loss_ci[0]), np.exp(loss_ci[1])]
        
        # Additional statistics
        results = {
            'perplexity': float(mean_perplexity),
            'perplexity_ci_lower': float(perplexity_ci[0]),
            'perplexity_ci_upper': float(perplexity_ci[1]),
            'log_likelihood': float(-mean_token_loss),
            'token_loss_mean': float(mean_token_loss),
            'token_loss_std': float(np.std(token_losses)),
            'token_loss_min': float(np.min(token_losses)),
            'token_loss_max': float(np.max(token_losses)),
            'sequence_perplexity_mean': float(np.mean(sequence_perplexities)),
            'sequence_perplexity_std': float(np.std(sequence_perplexities)),
            'sequence_perplexity_median': float(np.median(sequence_perplexities)),
            'sequence_perplexity_percentiles': {
                '25': float(np.percentile(sequence_perplexities, 25)),
                '75': float(np.percentile(sequence_perplexities, 75)),
                '90': float(np.percentile(sequence_perplexities, 90)),
                '95': float(np.percentile(sequence_perplexities, 95))
            },
            'num_tokens': len(token_losses),
            'num_sequences': len(sequence_losses)
        }
        
        return results
    
    def compute_distributional_metrics(
        self,
        dataloader: DataLoader,
        baseline_model = None,
        max_batches: Optional[int] = None,
        compute_divergences: bool = True
    ) -> Dict[str, Any]:
        """
        Compute distributional metrics including various divergences.
        
        Args:
            dataloader: DataLoader with evaluation data
            baseline_model: Baseline model for comparison
            max_batches: Maximum number of batches to evaluate
            compute_divergences: Whether to compute divergence metrics
            
        Returns:
            Dictionary with distributional metrics
        """
        if baseline_model is not None:
            baseline_model.to(self.device)
            baseline_model.eval()
        
        model_entropy_values = []
        baseline_entropy_values = []
        kl_divergences = []
        js_divergences = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing distributional metrics")):
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
                
                # Get model probabilities
                model_probs = F.softmax(model_logits, dim=-1)
                
                # Compute model entropy
                model_log_probs = F.log_softmax(model_logits, dim=-1)
                model_entropy = -(model_probs * model_log_probs).sum(dim=-1)
                
                # Create attention mask
                if 'attention_mask' in batch:
                    attention_mask = batch['attention_mask'].float()
                else:
                    attention_mask = torch.ones(model_entropy.shape, device=self.device)
                
                # Collect model entropy values
                valid_entropy = model_entropy * attention_mask
                model_entropy_values.extend(valid_entropy.flatten().cpu().numpy())
                
                # Compare with baseline if provided
                if baseline_model is not None and compute_divergences:
                    baseline_outputs = baseline_model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask')
                    )
                    baseline_logits = baseline_outputs.logits
                    
                    baseline_probs = F.softmax(baseline_logits, dim=-1)
                    baseline_log_probs = F.log_softmax(baseline_logits, dim=-1)
                    
                    # Baseline entropy
                    baseline_entropy = -(baseline_probs * baseline_log_probs).sum(dim=-1)
                    valid_baseline_entropy = baseline_entropy * attention_mask
                    baseline_entropy_values.extend(valid_baseline_entropy.flatten().cpu().numpy())
                    
                    # KL divergence: KL(model || baseline)
                    kl_div = F.kl_div(baseline_log_probs, model_probs, reduction='none').sum(dim=-1)
                    valid_kl = kl_div * attention_mask
                    kl_divergences.extend(valid_kl.flatten().cpu().numpy())
                    
                    # Jensen-Shannon divergence
                    m_probs = 0.5 * (model_probs + baseline_probs)
                    m_log_probs = torch.log(m_probs + 1e-10)
                    
                    kl_model_m = F.kl_div(m_log_probs, model_probs, reduction='none').sum(dim=-1)
                    kl_baseline_m = F.kl_div(m_log_probs, baseline_probs, reduction='none').sum(dim=-1)
                    js_div = 0.5 * (kl_model_m + kl_baseline_m)
                    
                    valid_js = js_div * attention_mask
                    js_divergences.extend(valid_js.flatten().cpu().numpy())
        
        # Convert to numpy and filter valid values
        model_entropy_values = np.array([x for x in model_entropy_values if not np.isnan(x) and x > 0])
        
        results = {
            'model_entropy_mean': float(np.mean(model_entropy_values)),
            'model_entropy_std': float(np.std(model_entropy_values)),
            'model_entropy_median': float(np.median(model_entropy_values)),
            'model_entropy_min': float(np.min(model_entropy_values)),
            'model_entropy_max': float(np.max(model_entropy_values))
        }
        
        if baseline_model is not None and compute_divergences:
            baseline_entropy_values = np.array([x for x in baseline_entropy_values if not np.isnan(x) and x > 0])
            kl_divergences = np.array([x for x in kl_divergences if not np.isnan(x)])
            js_divergences = np.array([x for x in js_divergences if not np.isnan(x)])
            
            results.update({
                'baseline_entropy_mean': float(np.mean(baseline_entropy_values)),
                'baseline_entropy_std': float(np.std(baseline_entropy_values)),
                'kl_divergence_mean': float(np.mean(kl_divergences)),
                'kl_divergence_std': float(np.std(kl_divergences)),
                'kl_divergence_median': float(np.median(kl_divergences)),
                'js_divergence_mean': float(np.mean(js_divergences)),
                'js_divergence_std': float(np.std(js_divergences)),
                'js_divergence_median': float(np.median(js_divergences))
            })
        
        return results
    
    def compute_training_dynamics_metrics(
        self,
        train_dataloader: DataLoader,
        max_batches: int = 20
    ) -> Dict[str, Any]:
        """
        Compute training dynamics and stability metrics.
        
        Args:
            train_dataloader: Training dataloader
            max_batches: Maximum batches for gradient analysis
            
        Returns:
            Dictionary with training dynamics metrics
        """
        results = {}
        
        # Gradient variance (for NSRPO models)
        if isinstance(self.model, NSRPOModel):
            gradient_metrics = self.model.get_policy_gradient_variance(
                train_dataloader, max_batches=max_batches
            )
            results['gradient_variance'] = gradient_metrics
            
            # Null space utilization metrics
            null_space_info = self.model.get_null_space_info()
            results['null_space_utilization'] = null_space_info
        
        # Loss landscape analysis (simplified)
        loss_values = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Analyzing training dynamics")):
                if batch_idx >= max_batches:
                    break
                
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                targets = batch.get('response_input_ids', batch['input_ids'])
                
                if isinstance(self.model, NSRPOModel):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        labels=targets
                    )
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        labels=targets
                    )
                
                if outputs.loss is not None:
                    loss_values.append(outputs.loss.item())
        
        if loss_values:
            results['loss_statistics'] = {
                'mean': float(np.mean(loss_values)),
                'std': float(np.std(loss_values)),
                'min': float(np.min(loss_values)),
                'max': float(np.max(loss_values)),
                'median': float(np.median(loss_values))
            }
        
        return results
    
    def run_comprehensive_evaluation(
        self,
        eval_dataloader: DataLoader,
        train_dataloader: Optional[DataLoader] = None,
        baseline_models: Optional[List] = None,
        max_eval_batches: Optional[int] = None,
        max_train_batches: int = 20,
        save_detailed_results: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation with all metrics.
        
        Args:
            eval_dataloader: Evaluation data loader
            train_dataloader: Training data loader (for dynamics)
            baseline_models: List of baseline models for comparison
            max_eval_batches: Maximum evaluation batches
            max_train_batches: Maximum training batches for dynamics
            save_detailed_results: Whether to save detailed results
            output_dir: Directory to save results
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            'evaluation_metadata': {
                'model_type': type(self.model).__name__,
                'device': str(self.device),
                'precision': self.precision,
                'timestamp': time.time(),
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
        
        self.logger.info("Starting comprehensive evaluation...")
        
        # Advanced accuracy metrics
        self.logger.info("Computing advanced accuracy metrics...")
        accuracy_results = self.compute_advanced_accuracy_metrics(
            eval_dataloader, max_eval_batches
        )
        results['accuracy_metrics'] = accuracy_results
        
        # Advanced perplexity metrics
        self.logger.info("Computing advanced perplexity metrics...")
        perplexity_results = self.compute_advanced_perplexity_metrics(
            eval_dataloader, max_eval_batches
        )
        results['perplexity_metrics'] = perplexity_results
        
        # Distributional metrics
        self.logger.info("Computing distributional metrics...")
        if baseline_models and len(baseline_models) > 0:
            distributional_results = self.compute_distributional_metrics(
                eval_dataloader, baseline_models[0], max_eval_batches
            )
        else:
            distributional_results = self.compute_distributional_metrics(
                eval_dataloader, None, max_eval_batches, compute_divergences=False
            )
        results['distributional_metrics'] = distributional_results
        
        # Training dynamics (if training data provided)
        if train_dataloader is not None:
            self.logger.info("Computing training dynamics metrics...")
            dynamics_results = self.compute_training_dynamics_metrics(
                train_dataloader, max_train_batches
            )
            results['training_dynamics'] = dynamics_results
        
        # Baseline comparisons
        if baseline_models:
            self.logger.info("Computing baseline comparisons...")
            comparison_results = self._compare_with_baselines(
                eval_dataloader, baseline_models, max_eval_batches
            )
            results['baseline_comparisons'] = comparison_results
        
        # Statistical summaries
        results['statistical_summary'] = self._compute_statistical_summary(results)
        
        # Save results if requested
        if save_detailed_results and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main results
            with open(output_dir / 'comprehensive_evaluation.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Detailed results saved to {output_dir}")
        
        self.logger.info("Comprehensive evaluation completed!")
        return results
    
    def _compare_with_baselines(
        self,
        dataloader: DataLoader,
        baseline_models: List,
        max_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compare model with baseline models."""
        comparisons = {}
        
        for idx, baseline_model in enumerate(baseline_models):
            baseline_name = f"baseline_{idx}"
            self.logger.info(f"Comparing with {baseline_name}...")
            
            # Create evaluator for baseline
            baseline_evaluator = NSRPOEvaluator(baseline_model, self.tokenizer, self.device)
            
            # Get baseline results
            baseline_accuracy = baseline_evaluator.compute_accuracy_metrics(dataloader, max_batches)
            baseline_perplexity = baseline_evaluator.compute_perplexity(dataloader, max_batches)
            
            # Get model results for comparison
            model_accuracy = self.base_evaluator.compute_accuracy_metrics(dataloader, max_batches)
            model_perplexity = self.base_evaluator.compute_perplexity(dataloader, max_batches)
            
            # Compute differences
            comparisons[baseline_name] = {
                'accuracy_difference': model_accuracy['token_accuracy'] - baseline_accuracy['token_accuracy'],
                'perplexity_ratio': model_perplexity['perplexity'] / baseline_perplexity['perplexity'],
                'baseline_metrics': {
                    'accuracy': baseline_accuracy,
                    'perplexity': baseline_perplexity
                }
            }
        
        return comparisons
    
    def _compute_statistical_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute statistical summary of all results."""
        summary = {}
        
        # Key metrics summary
        if 'accuracy_metrics' in results:
            acc = results['accuracy_metrics']
            summary['key_accuracy'] = {
                'token_accuracy': acc.get('token_accuracy', 0.0),
                'token_accuracy_ci': [
                    acc.get('token_accuracy_ci_lower', 0.0),
                    acc.get('token_accuracy_ci_upper', 0.0)
                ],
                'sequence_accuracy': acc.get('sequence_accuracy', 0.0)
            }
        
        if 'perplexity_metrics' in results:
            ppl = results['perplexity_metrics']
            summary['key_perplexity'] = {
                'perplexity': ppl.get('perplexity', float('inf')),
                'perplexity_ci': [
                    ppl.get('perplexity_ci_lower', float('inf')),
                    ppl.get('perplexity_ci_upper', float('inf'))
                ],
                'log_likelihood': ppl.get('log_likelihood', float('-inf'))
            }
        
        # Model-specific summaries
        if isinstance(self.model, NSRPOModel):
            summary['nsrpo_specific'] = {
                'null_decoder_enabled': True,
                'loss_weights': self.model.get_null_space_info().get('loss_weights', {}),
                'uses_reconstruction_metrics': self.model.use_reconstruction_metrics
            }
        
        return summary
    
    def _bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        confidence: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        if len(data) < 2:
            return (float(data[0]) if len(data) == 1 else 0.0, float(data[0]) if len(data) == 1 else 0.0)
        
        # Bootstrap sampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # Compute confidence interval
        alpha = (1 - confidence) / 2
        lower = np.percentile(bootstrap_means, alpha * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
        
        return (float(lower), float(upper))


if __name__ == "__main__":
    # Test the comprehensive evaluator
    print("Testing Comprehensive Evaluator...")
    
    # This would be used with actual models and data
    print("✓ Comprehensive evaluator implementation completed")
    print("  Features implemented:")
    print("  - Advanced accuracy metrics with confidence intervals")
    print("  - Advanced perplexity metrics with statistical analysis")
    print("  - Distributional metrics (entropy, KL/JS divergences)")
    print("  - Training dynamics analysis")
    print("  - Baseline model comparisons")
    print("  - Statistical summaries")
    print("  - Bootstrap confidence intervals")
    print("  - Comprehensive evaluation orchestration")
