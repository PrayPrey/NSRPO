"""
Null-Space Decoder Architecture
Task 2: Implement Null-Space Decoder Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class NullDecoder(nn.Module):
    """
    Null-Space Decoder that projects hidden states to null-space and reconstructs them.
    
    This decoder implements the core idea of projecting hidden representations to a null-space
    (low-variance dimensions) and using a small transformer decoder to reconstruct the original
    hidden states, serving as a regularization mechanism.
    """
    
    def __init__(
        self,
        hidden_size: int,
        null_basis: torch.Tensor,
        vocab_size: int,
        num_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
        lm_head: Optional[nn.Module] = None
    ):
        """
        Initialize the Null-Space Decoder.
        
        Args:
            hidden_size: Size of the hidden representations
            null_basis: Null-basis vectors (shape: [hidden_size, null_dim])
            vocab_size: Size of the vocabulary for output projection
            num_layers: Number of transformer decoder layers
            nhead: Number of attention heads
            dropout: Dropout probability
            dim_feedforward: Dimension of feedforward network
            activation: Activation function for transformer layers
            lm_head: Optional shared language model head from base model
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        
        # Register null-basis as buffer (not a parameter)
        self.register_buffer('null_basis', null_basis)
        self.null_dim = null_basis.shape[1]
        
        # Ensure null_basis is properly shaped and normalized
        self.basis_hidden = null_basis.shape[0]
        if self.basis_hidden == self.hidden_size:
            self.align_in = nn.Identity()
            self.align_out = nn.Identity()
        else:
            # hidden_size -> basis_hidden (투영 전 정렬)
            self.align_in = nn.Linear(self.hidden_size, self.basis_hidden, bias=False)
            # basis_hidden -> hidden_size (복원 후 정렬)
            self.align_out = nn.Linear(self.basis_hidden, self.hidden_size, bias=False)
            nn.init.xavier_uniform_(self.align_in.weight)
            nn.init.xavier_uniform_(self.align_out.weight)
        # Null-space projection operations
        self.null_projection = NullSpaceProjection(null_basis)
        
        # Decoder architecture layers
        self.input_projection = nn.Linear(self.null_dim, hidden_size)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # Use batch_first=True for easier handling
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection to vocabulary (use shared lm_head if provided)
        # Don't register as a submodule to avoid double gradient updates
        if lm_head is not None:
            # Store reference without registering as parameter
            self._lm_head = lm_head
            self.shared_lm_head = True
        else:
            self.output_projection = nn.Linear(hidden_size, vocab_size)
            self.shared_lm_head = False
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize input projection
        nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)
        
        # Initialize output projection only if not shared
        if not self.shared_lm_head:
            nn.init.xavier_uniform_(self.output_projection.weight)
            if self.output_projection.bias is not None:
                nn.init.zeros_(self.output_projection.bias)
        
        # Initialize transformer weights (they have their own initialization)
        for layer in self.transformer_decoder.layers:
            # Additional initialization for stability
            nn.init.xavier_uniform_(layer.linear1.weight, gain=1.0)
            nn.init.xavier_uniform_(layer.linear2.weight, gain=1.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_projections: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Null-Space Decoder.
        
        Args:
            hidden_states: Input hidden states (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask for the decoder
            memory: Memory for cross-attention (if None, uses self-attention)
            memory_mask: Mask for memory attention
            return_projections: Whether to return intermediate projections
            
        Returns:
            Dictionary containing:
                - logits: Output logits (batch_size, seq_len, vocab_size)
                - reconstructed_hidden: Reconstructed hidden states
                - null_projection: Projection in null space (if return_projections=True)
        """
        device = hidden_states.device
        # 모듈/버퍼가 hidden_states와 다른 디바이스면 전체를 이동
        if self.input_projection.weight.device != device:
            self.to(device)
            
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to null space
        pre_hidden = self.align_in(hidden_states)
        null_projection = self.null_projection.project_to_null(pre_hidden)
            
        # Project null-space features to decoder input dimension
        decoder_input = self.input_projection(null_projection)
        
        # Apply layer normalization for stability
        decoder_input = self.layer_norm(decoder_input)
        
        # If no memory is provided, use self-attention
        if memory is None:
            memory = decoder_input
        # (1) tgt_mask: 항상 causal mask 사용 (부울 정방 행렬)
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(device)
    
        # (2) tgt_key_padding_mask: attention_mask(1=유효, 0=패딩)를 bool로 변환
        tgt_key_padding_mask = None
        if attention_mask is not None:
            # 기대 형태: (batch, seq_len)
            if attention_mask.dim() == 2 and attention_mask.size(0) == batch_size:
                tgt_key_padding_mask = (attention_mask == 0).to(device=device, dtype=torch.bool)
            else:
                # 형태가 맞지 않으면 무시(필요시 로깅)
                tgt_key_padding_mask = None
        
        # Transformer 디코더 호출
        decoder_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=tgt_mask,                               # 부울 정방 마스크
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask        # 패딩 위치 마스크
        )
            
        # Project to vocabulary
        if self.shared_lm_head:
            logits = self._lm_head(decoder_output)
        else:
            logits = self.output_projection(decoder_output)
        
        # null-space에서 복원한 뒤 basis_hidden -> hidden_size로 역정렬
        reconstructed_basis = self.null_projection.project_from_null(null_projection)
        reconstructed_hidden = self.align_out(reconstructed_basis)
        # Prepare output
        output = {
            'logits': logits,
            'reconstructed_hidden': reconstructed_hidden
        }
        
        if return_projections:
            output['null_projection'] = null_projection
            output['decoder_input'] = decoder_input
            output['decoder_output'] = decoder_output
        
        return output
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence. The masked positions are filled with True."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def get_null_space_info(self) -> Dict[str, any]:
        """Get information about the null space."""
        return {
            'null_dim': self.null_dim,
            'hidden_size': self.hidden_size,
            'compression_ratio': self.null_dim / self.hidden_size,
            'null_basis_shape': self.null_basis.shape,
            'shared_lm_head': self.shared_lm_head
        }


class NullSpaceProjection(nn.Module):
    """
    Handles null-space projection operations.
    
    Subtask 2.1: Implement Null-Space Projection Operations
    """
    
    def __init__(self, null_basis: torch.Tensor):
        """
        Initialize null-space projection.
        
        Args:
            null_basis: Null-basis vectors (shape: [hidden_size, null_dim])
        """
        super().__init__()
        
        # Register as buffer so it moves with the model but isn't a parameter
        self.register_buffer('null_basis', null_basis)
        self.register_buffer('null_basis_t', null_basis.T)  # Precompute transpose
        
        self.hidden_size = null_basis.shape[0]
        self.null_dim = null_basis.shape[1]
        
        # Validate null basis properties
        self._validate_null_basis()
    
    def _validate_null_basis(self):
        """Validate properties of the null basis."""
        # Check for reasonable orthogonality
        gram_matrix = torch.matmul(self.null_basis.T, self.null_basis)
        identity = torch.eye(self.null_dim, device=self.null_basis.device, dtype=self.null_basis.dtype)
        orthogonality_error = (gram_matrix - identity).abs().max().item()
        
        if orthogonality_error > 0.1:
            import warnings
            warnings.warn(f"Null basis may not be well orthogonalized. Max error: {orthogonality_error:.4f}")
    
    def project_to_null(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to null space.
        
        Args:
            hidden_states: Input tensor (..., hidden_size)
            
        Returns:
            Null-space projection (..., null_dim)
        """
        # Validate input dimensions
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected last dim to be {self.hidden_size}, got {hidden_states.shape[-1]}")
        
        # Project: H @ N = (..., hidden_size) @ (hidden_size, null_dim) -> (..., null_dim)
        return torch.matmul(hidden_states, self.null_basis)
    
    def project_from_null(self, null_projection: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct hidden states from null-space projection.
        
        Args:
            null_projection: Null-space projection (..., null_dim)
            
        Returns:
            Reconstructed hidden states (..., hidden_size)
        """
        # Validate input dimensions
        if null_projection.shape[-1] != self.null_dim:
            raise ValueError(f"Expected last dim to be {self.null_dim}, got {null_projection.shape[-1]}")
        
        # Reconstruct: P @ N^T = (..., null_dim) @ (null_dim, hidden_size) -> (..., hidden_size)
        return torch.matmul(null_projection, self.null_basis_t)
    
    def compute_reconstruction_error(
        self,
        original: torch.Tensor,
        return_relative: bool = True
    ) -> torch.Tensor:
        """
        Compute reconstruction error after null-space projection.
        
        Args:
            original: Original hidden states
            return_relative: Whether to return relative error
            
        Returns:
            Reconstruction error
        """
        # Project to null space and back
        null_proj = self.project_to_null(original)
        reconstructed = self.project_from_null(null_proj)
        
        # Compute error
        error = (original - reconstructed).norm(dim=-1)
        
        if return_relative:
            original_norm = original.norm(dim=-1)
            error = error / (original_norm + 1e-8)
        
        return error


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer decoder.
    
    Subtask 2.2: Build Transformer Decoder Architecture (component)
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def create_null_decoder(
    null_basis_path: str,
    hidden_size: int,
    vocab_size: int,
    **kwargs
) -> NullDecoder:
    """
    Factory function to create a NullDecoder from a saved null-basis.
    
    Args:
        null_basis_path: Path to saved null-basis tensor
        hidden_size: Hidden size of the model
        vocab_size: Vocabulary size
        **kwargs: Additional arguments for NullDecoder
        
    Returns:
        Initialized NullDecoder
    """
    # Load null-basis
    null_basis = torch.load(null_basis_path, map_location='cpu')
    
    return NullDecoder(
        hidden_size=hidden_size,
        null_basis=null_basis,
        vocab_size=vocab_size,
        **kwargs
    )


if __name__ == "__main__":
    # Test the NullDecoder
    print("Testing NullDecoder implementation...")
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    hidden_size = 64
    null_dim = 16
    vocab_size = 1000
    
    # Create random null basis
    null_basis = torch.qr(torch.randn(hidden_size, null_dim))[0]  # Orthogonal basis
    
    # Create model
    decoder = NullDecoder(
        hidden_size=hidden_size,
        null_basis=null_basis,
        vocab_size=vocab_size,
        num_layers=2,
        nhead=4
    )
    
    # Test input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    with torch.no_grad():
        output = decoder(hidden_states, return_projections=True)
    
    # Check outputs
    print(f"Input shape: {hidden_states.shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Reconstructed hidden shape: {output['reconstructed_hidden'].shape}")
    print(f"Null projection shape: {output['null_projection'].shape}")
    
    # Check reconstruction error
    reconstruction_error = (hidden_states - output['reconstructed_hidden']).norm()
    print(f"Reconstruction error: {reconstruction_error:.4f}")
    
    print("✓ NullDecoder test completed successfully!")
