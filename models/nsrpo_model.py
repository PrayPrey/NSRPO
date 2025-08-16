"""
Null-Space Augmented GRPO Model
Task 4: Implement Null-Space Augmented GRPO Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

from .null_decoder import NullDecoder
from .losses import NullDecoderLoss, ReconstructionMetrics


@dataclass
class ModelOutput:
    """
    Output class for NSRPOModel with all necessary components.
    """
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    decoder_logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    reconstructed_hidden: Optional[torch.Tensor] = None
    loss_components: Optional[Dict[str, torch.Tensor]] = None
    reconstruction_metrics: Optional[Dict[str, torch.Tensor]] = None
    past_key_values: Optional[Tuple] = None
    attentions: Optional[Tuple] = None


class NSRPOModel(nn.Module):
    """
    Null-Space Augmented GRPO Model.
    
    This model combines a base language model with a Null-Space Decoder
    to provide regularization during reinforcement learning from human feedback (RLHF).
    
    The key idea is to project hidden representations to a null space and use a 
    small transformer decoder to reconstruct them, serving as a regularization mechanism
    that preserves the existing representation space while expanding representation capability.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        null_decoder: NullDecoder,
        alpha_1: float = 0.1,
        alpha_2: float = 0.1, 
        alpha_3: float = 0.05,
        use_reconstruction_metrics: bool = True,
        freeze_base_model: bool = False
    ):
        """
        Initialize the NSRPO Model.
        
        Args:
            base_model: Pre-trained base language model (e.g., GPT, LLaMA)
            null_decoder: Null-space decoder for regularization
            alpha_1: Weight for Cross-Entropy loss (CE loss)
            alpha_2: Weight for Cosine similarity loss
            alpha_3: Weight for Norm preservation loss
            use_reconstruction_metrics: Whether to compute additional reconstruction metrics
            freeze_base_model: Whether to freeze the base model parameters
        """
        super().__init__()
        
        # Task 4.1: Implement Base Model Integration
        self.base_model = base_model
        self.null_decoder = null_decoder
        
        # Loss weighting parameters as described in PRD
        self.alpha_1 = alpha_1  # CE loss weight
        self.alpha_2 = alpha_2  # Cosine loss weight  
        self.alpha_3 = alpha_3  # Norm preservation weight
        
        # Initialize loss functions
        self.null_decoder_loss = NullDecoderLoss(
            alpha_ce=alpha_1,
            alpha_cos=alpha_2, 
            alpha_preserve=alpha_3
        )
        
        # Optional reconstruction metrics
        self.use_reconstruction_metrics = use_reconstruction_metrics
        if use_reconstruction_metrics:
            self.reconstruction_metrics = ReconstructionMetrics()
        
        # Optionally freeze base model parameters
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Validate compatibility
        self._validate_model_compatibility()

    def _align_labels_to_inputs(self, input_ids: torch.Tensor, labels: torch.Tensor | None, pad_value: int = -100):
        """labels 길이를 input_ids(seq_len)에 맞춰 잘라내거나 패딩(-100)합니다.
           labels가 없으면 LM 학습 기본값으로 input_ids를 반환합니다.
        """
        if labels is None:
            return input_ids  # 기본 LM 학습: labels=input_ids
    
        # (batch, seq) 형태 보장
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
    
        seq_len = input_ids.size(1)
        if labels.size(1) == seq_len:
            return labels
        if labels.size(1) > seq_len:
            return labels[:, :seq_len]  # 잘라냄
        # labels.size(1) < seq_len 인 경우: 뒤쪽을 -100으로 패딩(무시)
        pad = seq_len - labels.size(1)
        return F.pad(labels, (0, pad), value=pad_value)
        
    def _validate_model_compatibility(self):
        """Validate that base model and null decoder are compatible."""
        # Check if base model has the required interface
        if not hasattr(self.base_model, 'forward'):
            raise ValueError("Base model must have a forward method")
        
        # Test with dummy input to check compatibility
        try:
            dummy_input = torch.randint(0, 1000, (1, 10))
            with torch.no_grad():
                outputs = self.base_model(dummy_input, output_hidden_states=True)
                if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
                    warnings.warn("Base model may not support output_hidden_states=True")
        except Exception as e:
            warnings.warn(f"Model compatibility check failed: {e}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = True
    ) -> Union[ModelOutput, Tuple]:
        """
        Forward pass through the NSRPO model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Target labels for training (batch_size, seq_len)
            return_dict: Whether to return ModelOutput or tuple
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            
        Returns:
            ModelOutput containing loss, logits, and additional information
        """
        # Task 4.3: Implement Forward Pass Logic
        labels = self._align_labels_to_inputs(input_ids, labels)  # <<< 추가
        # Get base model outputs with hidden states
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True
        )
        
        # Extract hidden states (use last layer)
        if hasattr(base_outputs, 'hidden_states') and base_outputs.hidden_states is not None:
            hidden_states = base_outputs.hidden_states[-1]  # Last layer hidden states
        else:
            # Fallback: try to get hidden states from last_hidden_state
            hidden_states = getattr(base_outputs, 'last_hidden_state', None)
            if hidden_states is None:
                raise RuntimeError("Could not extract hidden states from base model")
        
        # Task 4.2: Implement Null-Decoder Integration
        # Apply null decoder to hidden states
        null_decoder_outputs = self.null_decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            return_projections=True
        )
        
        decoder_logits = null_decoder_outputs['logits']
        reconstructed_hidden = null_decoder_outputs['reconstructed_hidden']
        
        # Task 4.4: Implement Loss Calculation Logic
        # Initialize loss components
        total_loss = None
        loss_components = {}
        reconstruction_metrics = {}
        
        # Get base RL loss if available
        rl_loss = getattr(base_outputs, 'loss', None)
        if rl_loss is not None:
            loss_components['rl_loss'] = rl_loss.detach()
        
        # Calculate null decoder regularization loss if labels are provided
        if labels is not None:
            null_loss, null_components = self.null_decoder_loss(
                logits=decoder_logits,
                targets=labels,
                original_hidden=hidden_states,
                reconstructed_hidden=reconstructed_hidden,
                attention_mask=attention_mask
            )
            
            # Combine losses: L_total = L_RL + L_null
            if rl_loss is not None:
                total_loss = rl_loss + null_loss
            else:
                total_loss = null_loss
            
            # Store loss components
            loss_components.update(null_components)
            loss_components['null_decoder_loss'] = null_loss.detach()
        else:
            total_loss = rl_loss
        
        # Compute reconstruction metrics if requested
        if self.use_reconstruction_metrics and self.training:
            reconstruction_metrics = self.reconstruction_metrics(
                original=hidden_states,
                reconstructed=reconstructed_hidden,
                attention_mask=attention_mask
            )
        
        # Prepare output
        if return_dict:
            return ModelOutput(
                loss=total_loss,
                logits=getattr(base_outputs, 'logits', None),
                decoder_logits=decoder_logits,
                hidden_states=hidden_states,
                reconstructed_hidden=reconstructed_hidden,
                loss_components=loss_components,
                reconstruction_metrics=reconstruction_metrics,
                past_key_values=getattr(base_outputs, 'past_key_values', None),
                attentions=getattr(base_outputs, 'attentions', None)
            )
        else:
            return (
                total_loss,
                getattr(base_outputs, 'logits', None),
                decoder_logits,
                hidden_states,
                reconstructed_hidden
            )
    
    def get_policy_gradient_variance(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_batches: int = 10,
        token_level: bool = True
    ) -> Dict[str, float]:
        """Compute (approx.) policy gradient variance over a few batches."""
        import torch
        import torch.nn.functional as F
    
        # 유지하던 학습/평가 모드 기억
        was_training = self.training
        self.eval()  # 드롭아웃 고정(원하면 was_training 그대로 둬도 무방)
    
        # 모델 파라미터가 올라간 대표 디바이스
        device = next(self.parameters()).device
        grads_cpu = []
        n_batches = 0
    
        with torch.enable_grad():  # <-- 핵심: 그래드 추적 활성화
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
    
                # 배치를 모델 디바이스로
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    
                # labels 준비(없으면 input_ids 사용) — forward에 길이 정렬 가드가 있음
                labels = batch.get('response_input_ids', batch.get('input_ids'))
    
                # 그래프가 있는 forward 호출
                outputs = self.forward(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=labels
                )
    
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    # 필요 시 CE loss를 직접 구성
                    logits = getattr(outputs, "logits", None)
                    if logits is None:
                        continue
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = batch['input_ids'][:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction='mean'
                    )
    
                # 그래드 계산
                self.zero_grad(set_to_none=True)
                loss.backward()  # retain_graph 불필요
    
                # base_model 파라미터 그래드 수집(디바이스 섞임 방지를 위해 CPU로 이동)
                flat_grads = []
                for p in self.base_model.parameters():
                    if p.grad is not None:
                        flat_grads.append(p.grad.detach().to('cpu').reshape(-1))
    
                if flat_grads:
                    grads_cpu.append(torch.cat(flat_grads))
                    n_batches += 1
    
        # 원래 모드 복구
        if was_training:
            self.train()
    
        if n_batches == 0:
            return {"variance": 0.0, "mean": 0.0, "std": 0.0, "num_batches": 0}
    
        G = torch.stack(grads_cpu, dim=0)  # (n_batches, total_params)
        param_var = G.var(dim=0, unbiased=False)
        return {
            "variance": param_var.mean().item(),
            "mean": G.mean().item(),
            "std": param_var.std().item(),
            "num_batches": n_batches
        }

    def update_loss_weights(
        self,
        alpha_1: Optional[float] = None,
        alpha_2: Optional[float] = None,
        alpha_3: Optional[float] = None
    ):
        """Update loss weights during training."""
        if alpha_1 is not None:
            self.alpha_1 = alpha_1
            self.null_decoder_loss.alpha_ce = alpha_1
        
        if alpha_2 is not None:
            self.alpha_2 = alpha_2
            self.null_decoder_loss.alpha_cos = alpha_2
            
        if alpha_3 is not None:
            self.alpha_3 = alpha_3
            self.null_decoder_loss.alpha_preserve = alpha_3
    
    def get_null_space_info(self) -> Dict[str, Any]:
        """Get information about the null space configuration."""
        return {
            'model_type': 'NSRPOModel',
            'loss_weights': {
                'alpha_1': self.alpha_1,
                'alpha_2': self.alpha_2, 
                'alpha_3': self.alpha_3
            },
            'null_decoder_info': self.null_decoder.get_null_space_info(),
            'base_model_type': type(self.base_model).__name__,
            'use_reconstruction_metrics': self.use_reconstruction_metrics
        }
    
    def enable_null_decoder(self):
        """Enable null decoder processing."""
        for param in self.null_decoder.parameters():
            param.requires_grad = True
    
    def disable_null_decoder(self):
        """Disable null decoder processing (baseline GRPO mode)."""
        for param in self.null_decoder.parameters():
            param.requires_grad = False
    
    def save_null_decoder(self, path: str):
        """Save null decoder separately."""
        torch.save(self.null_decoder.state_dict(), path)
    
    def load_null_decoder(self, path: str):
        """Load null decoder from saved state."""
        self.null_decoder.load_state_dict(torch.load(path, map_location='cpu'))


def create_nsrpo_model(
    base_model: nn.Module,
    null_basis_path: str,
    vocab_size: int,
    hidden_size: Optional[int] = None,
    **kwargs
) -> NSRPOModel:
    """
    Factory function to create an NSRPOModel.
    
    Args:
        base_model: Pre-trained base model
        null_basis_path: Path to saved null-basis tensor
        vocab_size: Vocabulary size
        hidden_size: Hidden size (inferred from base_model if not provided)
        **kwargs: Additional arguments for NSRPOModel and NullDecoder
        
    Returns:
        Initialized NSRPOModel
    """
    # Infer hidden size from base model if not provided
    if hidden_size is None:
        if hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
            hidden_size = base_model.config.hidden_size
        else:
            # Try to infer from a dummy forward pass
            dummy_input = torch.randint(0, 1000, (1, 10))
            with torch.no_grad():
                outputs = base_model(dummy_input, output_hidden_states=True)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    hidden_size = outputs.hidden_states[-1].shape[-1]
                else:
                    raise ValueError("Could not infer hidden_size from base_model")
    
    # Load null basis
    null_basis = torch.load(null_basis_path, map_location='cpu')
    
    # Get lm_head from base model if available
    lm_head = None
    if hasattr(base_model, 'lm_head'):
        lm_head = base_model.lm_head
    elif hasattr(base_model, 'get_output_embeddings'):
        lm_head = base_model.get_output_embeddings()
    
    # Create null decoder with shared lm_head
    null_decoder = NullDecoder(
        hidden_size=hidden_size,
        null_basis=null_basis,
        vocab_size=vocab_size,
        lm_head=lm_head,
        **{k: v for k, v in kwargs.items() 
           if k in ['num_layers', 'nhead', 'dropout', 'dim_feedforward', 'activation']}
    )
    
    # Create NSRPO model
    nsrpo_model = NSRPOModel(
        base_model=base_model,
        null_decoder=null_decoder,
        **{k: v for k, v in kwargs.items() 
           if k in ['alpha_1', 'alpha_2', 'alpha_3', 'use_reconstruction_metrics', 'freeze_base_model']}
    )
    
    return nsrpo_model


if __name__ == "__main__":
    # Test the NSRPOModel
    print("Testing NSRPOModel implementation...")
    
    # Mock base model for testing
    class MockBaseModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False, **kwargs):
            hidden_states = self.embedding(input_ids)
            logits = self.lm_head(hidden_states)
            
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            class MockOutput:
                def __init__(self):
                    self.logits = logits
                    self.loss = loss
                    self.hidden_states = [hidden_states] if output_hidden_states else None
            
            return MockOutput()
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    hidden_size = 64
    null_dim = 16
    
    # Create models
    base_model = MockBaseModel(vocab_size=vocab_size, hidden_size=hidden_size)
    null_basis = torch.qr(torch.randn(hidden_size, null_dim))[0]
    null_decoder = NullDecoder(
        hidden_size=hidden_size,
        null_basis=null_basis,
        vocab_size=vocab_size,
        num_layers=2,
        nhead=4
    )
    
    # Create NSRPO model
    nsrpo_model = NSRPOModel(
        base_model=base_model,
        null_decoder=null_decoder,
        alpha_1=1.0,
        alpha_2=0.5,
        alpha_3=0.5
    )
    
    # Test input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test forward pass
    with torch.no_grad():
        outputs = nsrpo_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Base logits shape: {outputs.logits.shape}")
    print(f"Decoder logits shape: {outputs.decoder_logits.shape}")
    print(f"Hidden states shape: {outputs.hidden_states.shape}")
    print(f"Reconstructed hidden shape: {outputs.reconstructed_hidden.shape}")
    print(f"Total loss: {outputs.loss.item():.4f}")
    
    # Test loss components
    if outputs.loss_components:
        print("Loss components:")
        for key, value in outputs.loss_components.items():
            print(f"  {key}: {value.item():.4f}")
    
    # Test model info
    info = nsrpo_model.get_null_space_info()
    print(f"Model info: {info}")
    
    print("✓ NSRPOModel test completed successfully!")
