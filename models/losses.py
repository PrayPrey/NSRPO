"""
Loss Functions for Null-Space Decoder
Task 3: Implement Loss Functions for Null-Space Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import warnings


class NullDecoderLoss(nn.Module):
    """
    Combined loss function for Null-Space Decoder training.
    
    Combines:
    1. Cross-Entropy Loss: Standard token prediction loss
    2. Cosine Similarity Loss: Measure similarity between original and reconstructed hidden states
    3. Norm-Preservation Loss: Ensure the norm of hidden states is preserved after reconstruction
    """
    
    def __init__(
        self,
        alpha_ce: float = 1.0,
        alpha_cos: float = 1.0,
        alpha_preserve: float = 1.0,
        ignore_index: int = -100,
        label_smoothing: float = 0.0
    ):
        """
        Initialize the combined loss function.
        
        Args:
            alpha_ce: Weight for Cross-Entropy loss
            alpha_cos: Weight for Cosine similarity loss 
            alpha_preserve: Weight for Norm preservation loss
            ignore_index: Index to ignore in CE loss calculation
            label_smoothing: Label smoothing factor for CE loss
        """
        super().__init__()
        
        self.alpha_ce = alpha_ce
        self.alpha_cos = alpha_cos
        self.alpha_preserve = alpha_preserve
        
        # Cross-entropy loss for token prediction
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        
        # Store loss components for monitoring
        self.loss_components = {}
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        original_hidden: torch.Tensor,
        reconstructed_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate combined loss.
        
        Args:
            logits: Decoder output logits (batch_size, seq_len, vocab_size)
            targets: Target token IDs (batch_size, seq_len)
            original_hidden: Original hidden states (batch_size, seq_len, hidden_size)
            reconstructed_hidden: Reconstructed hidden states (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual loss components
        """
        
        # 1. Cross-Entropy Loss
        ce_loss = self._compute_ce_loss(logits, targets)
        
        # 2. Cosine Similarity Loss 
        cos_loss = self._compute_cosine_loss(original_hidden, reconstructed_hidden, attention_mask)
        
        # 3. Norm Preservation Loss
        norm_loss = self._compute_norm_loss(original_hidden, reconstructed_hidden, attention_mask)
        
        # Combine losses with weights
        total_loss = (
            self.alpha_ce * ce_loss +
            self.alpha_cos * cos_loss + 
            self.alpha_preserve * norm_loss
        )
        
        # Store components for monitoring
        loss_dict = {
            'ce_loss': ce_loss.detach(),
            'cos_loss': cos_loss.detach(),
            'norm_loss': norm_loss.detach(),
            'total_loss': total_loss.detach()
        }
        
        return total_loss, loss_dict
    
    def _compute_ce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Cross-Entropy loss."""
        # Reshape for CE loss: (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        ce_loss = self.ce_loss(logits_flat, targets_flat)
        return ce_loss
    
    def _compute_cosine_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Cosine Similarity loss.
        
        Uses 1 - cosine_similarity to convert similarity to loss (minimize for better similarity).
        """
        # Flatten to 2D: (batch_size * seq_len, hidden_size)
        orig_flat = original.view(-1, original.size(-1))
        recon_flat = reconstructed.view(-1, reconstructed.size(-1))
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(orig_flat, recon_flat, dim=-1)
        
        # Convert to loss (1 - similarity)
        cos_loss_flat = 1 - cos_sim
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask_flat = attention_mask.view(-1)
            cos_loss_flat = cos_loss_flat * mask_flat
            cos_loss = cos_loss_flat.sum() / (mask_flat.sum() + 1e-8)
        else:
            cos_loss = cos_loss_flat.mean()
        
        return cos_loss
    
    def _compute_norm_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Norm Preservation loss.
        
        Minimizes the relative difference between original and reconstructed norms.
        """
        # Compute norms along the hidden dimension
        orig_norm = torch.norm(original, dim=-1)  # (batch_size, seq_len)
        recon_norm = torch.norm(reconstructed, dim=-1)  # (batch_size, seq_len)
        
        # Compute relative norm difference
        norm_diff = torch.abs(orig_norm - recon_norm) / (orig_norm + 1e-8)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            norm_diff = norm_diff * attention_mask
            norm_loss = norm_diff.sum() / (attention_mask.sum() + 1e-8)
        else:
            norm_loss = norm_diff.mean()
        
        return norm_loss
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return {
            'alpha_ce': self.alpha_ce,
            'alpha_cos': self.alpha_cos,
            'alpha_preserve': self.alpha_preserve
        }
    
    def update_loss_weights(
        self,
        alpha_ce: Optional[float] = None,
        alpha_cos: Optional[float] = None,
        alpha_preserve: Optional[float] = None
    ):
        """Update loss weights during training."""
        if alpha_ce is not None:
            self.alpha_ce = alpha_ce
        if alpha_cos is not None:
            self.alpha_cos = alpha_cos
        if alpha_preserve is not None:
            self.alpha_preserve = alpha_preserve


class ReconstructionMetrics(nn.Module):
    """
    Additional metrics to monitor reconstruction quality.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction quality metrics.
        
        Args:
            original: Original hidden states
            reconstructed: Reconstructed hidden states  
            attention_mask: Attention mask
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Mean Squared Error
        mse = F.mse_loss(reconstructed, original, reduction='none').mean(dim=-1)
        if attention_mask is not None:
            mse = (mse * attention_mask).sum() / (attention_mask.sum() + 1e-8)
        else:
            mse = mse.mean()
        metrics['mse'] = mse
        
        # Mean Absolute Error
        mae = F.l1_loss(reconstructed, original, reduction='none').mean(dim=-1)
        if attention_mask is not None:
            mae = (mae * attention_mask).sum() / (attention_mask.sum() + 1e-8)
        else:
            mae = mae.mean()
        metrics['mae'] = mae
        
        # Cosine Similarity (average)
        orig_flat = original.view(-1, original.size(-1))
        recon_flat = reconstructed.view(-1, reconstructed.size(-1))
        cos_sim = F.cosine_similarity(orig_flat, recon_flat, dim=-1)
        
        if attention_mask is not None:
            mask_flat = attention_mask.view(-1)
            cos_sim = (cos_sim * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        else:
            cos_sim = cos_sim.mean()
        metrics['cosine_similarity'] = cos_sim
        
        # Relative Norm Error
        orig_norm = torch.norm(original, dim=-1)
        recon_norm = torch.norm(reconstructed, dim=-1)
        relative_norm_error = torch.abs(orig_norm - recon_norm) / (orig_norm + 1e-8)
        
        if attention_mask is not None:
            relative_norm_error = (relative_norm_error * attention_mask).sum() / (attention_mask.sum() + 1e-8)
        else:
            relative_norm_error = relative_norm_error.mean()
        metrics['relative_norm_error'] = relative_norm_error
        
        # Signal-to-Noise Ratio (SNR)
        signal_power = torch.mean(original ** 2, dim=-1)
        noise_power = torch.mean((original - reconstructed) ** 2, dim=-1)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        
        if attention_mask is not None:
            snr = (snr * attention_mask).sum() / (attention_mask.sum() + 1e-8)
        else:
            snr = snr.mean()
        metrics['snr_db'] = snr
        
        return metrics


class AdaptiveLossScheduler:
    """
    Scheduler to adaptively adjust loss weights during training.
    """
    
    def __init__(
        self,
        loss_fn: NullDecoderLoss,
        mode: str = 'cosine_annealing',
        min_ratio: float = 0.1,
        max_ratio: float = 2.0,
        warmup_steps: int = 1000
    ):
        """
        Initialize adaptive loss scheduler.
        
        Args:
            loss_fn: Loss function to schedule
            mode: Scheduling mode ('cosine_annealing', 'linear', 'exponential')
            min_ratio: Minimum ratio for weight adjustment
            max_ratio: Maximum ratio for weight adjustment  
            warmup_steps: Number of warmup steps
        """
        self.loss_fn = loss_fn
        self.mode = mode
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.warmup_steps = warmup_steps
        
        # Store initial weights
        self.initial_weights = loss_fn.get_loss_weights()
        
    def step(self, global_step: int, loss_components: Dict[str, float]):
        """
        Update loss weights based on current step and loss components.
        
        Args:
            global_step: Current training step
            loss_components: Dictionary of current loss components
        """
        if global_step < self.warmup_steps:
            # Gradual warmup
            ratio = global_step / self.warmup_steps
            self._update_weights_with_ratio(ratio)
        else:
            # Adaptive adjustment based on loss components
            self._adaptive_adjustment(loss_components)
    
    def _update_weights_with_ratio(self, ratio: float):
        """Update weights with a given ratio."""
        for key, initial_weight in self.initial_weights.items():
            new_weight = initial_weight * (self.min_ratio + ratio * (1.0 - self.min_ratio))
            setattr(self.loss_fn, key, new_weight)
    
    def _adaptive_adjustment(self, loss_components: Dict[str, float]):
        """Adaptively adjust weights based on loss component magnitudes."""
        # Simple heuristic: increase weight for components that are too small
        ce_loss = loss_components.get('ce_loss', 1.0)
        cos_loss = loss_components.get('cos_loss', 1.0) 
        norm_loss = loss_components.get('norm_loss', 1.0)
        
        # Normalize by CE loss
        cos_ratio = cos_loss / (ce_loss + 1e-8)
        norm_ratio = norm_loss / (ce_loss + 1e-8)
        
        # Adjust weights (simple proportional control)
        if cos_ratio < 0.1:  # Cosine loss too small
            self.loss_fn.alpha_cos = min(self.loss_fn.alpha_cos * 1.1, self.initial_weights['alpha_cos'] * self.max_ratio)
        elif cos_ratio > 1.0:  # Cosine loss too large
            self.loss_fn.alpha_cos = max(self.loss_fn.alpha_cos * 0.9, self.initial_weights['alpha_cos'] * self.min_ratio)
            
        if norm_ratio < 0.1:  # Norm loss too small
            self.loss_fn.alpha_preserve = min(self.loss_fn.alpha_preserve * 1.1, self.initial_weights['alpha_preserve'] * self.max_ratio)
        elif norm_ratio > 1.0:  # Norm loss too large
            self.loss_fn.alpha_preserve = max(self.loss_fn.alpha_preserve * 0.9, self.initial_weights['alpha_preserve'] * self.min_ratio)


def create_loss_function(
    alpha_ce: float = 1.0,
    alpha_cos: float = 1.0,
    alpha_preserve: float = 1.0,
    **kwargs
) -> NullDecoderLoss:
    """
    Factory function to create a NullDecoderLoss with specified weights.
    
    Args:
        alpha_ce: Cross-entropy loss weight
        alpha_cos: Cosine similarity loss weight
        alpha_preserve: Norm preservation loss weight
        **kwargs: Additional arguments for NullDecoderLoss
        
    Returns:
        Configured NullDecoderLoss instance
    """
    return NullDecoderLoss(
        alpha_ce=alpha_ce,
        alpha_cos=alpha_cos,
        alpha_preserve=alpha_preserve,
        **kwargs
    )


if __name__ == "__main__":
    # Test the loss functions
    print("Testing NullDecoderLoss implementation...")
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    hidden_size = 64
    vocab_size = 1000
    
    # Create test data
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    original_hidden = torch.randn(batch_size, seq_len, hidden_size)
    reconstructed_hidden = original_hidden + 0.1 * torch.randn(batch_size, seq_len, hidden_size)  # Add some noise
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Create loss function
    loss_fn = NullDecoderLoss(alpha_ce=1.0, alpha_cos=0.5, alpha_preserve=0.5)
    
    # Test forward pass
    total_loss, loss_dict = loss_fn(logits, targets, original_hidden, reconstructed_hidden, attention_mask)
    
    print(f"Total loss: {total_loss.item():.4f}")
    for key, value in loss_dict.items():
        print(f"{key}: {value.item():.4f}")
    
    # Test reconstruction metrics
    metrics_fn = ReconstructionMetrics()
    metrics = metrics_fn(original_hidden, reconstructed_hidden, attention_mask)
    
    print("\nReconstruction metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value.item():.4f}")
    
    print("✓ NullDecoderLoss test completed successfully!")