"""
SVD Utility Functions for Null-Space Extraction
Task 1: Implement SVD Utility Functions
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import numpy as np
from sklearn.decomposition import PCA


def extract_lora_null_basis(
    model: nn.Module, 
    rank: int = 16, 
    epsilon_factor: float = 1e-3
) -> torch.Tensor:
    """
    Extract null-basis from LoRA ΔW using SVD.
    
    Args:
        model: Model with LoRA weights
        rank: Target rank for null-basis extraction
        epsilon_factor: Factor for epsilon threshold calculation
        
    Returns:
        null_basis: Tensor of shape (hidden_size, rank) containing null-basis vectors
    """
    lora_weights = []
    
    # Extract LoRA weights from model
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # Compute ΔW = B @ A for LoRA
            delta_w = module.lora_B.weight @ module.lora_A.weight
            lora_weights.append(delta_w)
    
    if not lora_weights:
        raise ValueError("No LoRA weights found in model")
    
    # Stack all LoRA weight matrices
    combined_weights = torch.cat([w.flatten() for w in lora_weights]).reshape(-1, lora_weights[0].size(-1))
    
    # Perform SVD using torch.svd_lowrank for efficiency
    U, S, Vt = torch.svd_lowrank(combined_weights, q=min(rank * 2, combined_weights.size(0)))
    
    # Calculate epsilon threshold: ε ≈ epsilon_factor × σ_max
    epsilon = epsilon_factor * S[0].item()
    
    # Find indices where singular values are below threshold
    null_indices = torch.where(S < epsilon)[0]
    
    if len(null_indices) == 0:
        # If no null space found, take the smallest singular values
        null_indices = torch.arange(max(0, len(S) - rank), len(S))
    
    # Extract corresponding right singular vectors as null-basis
    null_basis = Vt[null_indices[:rank]].T  # Shape: (hidden_size, rank)
    
    return null_basis


def extract_base_null_basis(
    model: nn.Module, 
    epsilon_factor: float = 1e-3,
    layer_name: str = "lm_head"
) -> torch.Tensor:
    """
    Extract null-basis from base model weights using truncated SVD.
    
    Args:
        model: Base model
        epsilon_factor: Factor for epsilon threshold calculation
        layer_name: Name of the layer to extract weights from
        
    Returns:
        null_basis: Tensor containing null-basis vectors
    """
    # Find the target layer
    target_layer = None
    for name, module in model.named_modules():
        if layer_name in name:
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Get weight matrix
    if hasattr(target_layer, 'weight'):
        weight_matrix = target_layer.weight.data
    else:
        raise ValueError(f"Layer {layer_name} does not have weight attribute")
    
    # Use randomized SVD for large matrices
    rank = min(256, weight_matrix.size(0) // 4, weight_matrix.size(1) // 4)
    U, S, Vt = torch.svd_lowrank(weight_matrix, q=rank)
    
    # Calculate epsilon threshold
    epsilon = epsilon_factor * S[0].item()
    
    # Find null space
    null_indices = torch.where(S < epsilon)[0]
    
    if len(null_indices) == 0:
        # Take bottom 10% of singular values as null space
        num_null = max(1, len(S) // 10)
        null_indices = torch.arange(len(S) - num_null, len(S))
    
    # Extract null-basis vectors
    null_basis = Vt[null_indices].T
    
    return null_basis


def extract_activation_null_basis(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epsilon_factor: float = 1e-3,
    max_samples: int = 1000,
    layer_name: str = "model.norm"
) -> torch.Tensor:
    """
    Extract null-basis using PCA on model activations.
    
    Args:
        model: Model to extract activations from
        dataloader: DataLoader for sampling activations
        epsilon_factor: Factor for epsilon threshold calculation
        max_samples: Maximum number of samples to use
        layer_name: Name of layer to extract activations from
        
    Returns:
        null_basis: Tensor containing null-basis vectors from activation space
    """
    model.eval()
    activations = []
    
    # Hook to capture activations
    activation_hook = None
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        # Store activation (flatten batch and sequence dimensions)
        act = output.detach().cpu()
        if act.dim() == 3:  # [batch, seq, hidden]
            act = act.reshape(-1, act.size(-1))
        activations.append(act)
    
    # Register hook
    for name, module in model.named_modules():
        if layer_name in name:
            activation_hook = module.register_forward_hook(hook_fn)
            break
    
    if activation_hook is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Collect activations
    samples_collected = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if samples_collected >= max_samples:
                break
            
            # Forward pass to collect activations
            if isinstance(batch, dict) and 'input_ids' in batch:
                _ = model(input_ids=batch['input_ids'])
            else:
                _ = model(batch)
            
            samples_collected += batch['input_ids'].size(0) if isinstance(batch, dict) else batch.size(0)
    
    # Remove hook
    activation_hook.remove()
    
    # Concatenate all activations
    all_activations = torch.cat(activations, dim=0).numpy()
    
    # Apply PCA
    pca = PCA(n_components=min(all_activations.shape[1], 256))
    pca.fit(all_activations)
    
    # Find components with low explained variance
    explained_variance = pca.explained_variance_
    epsilon = epsilon_factor * explained_variance[0]
    
    # Find null space components
    null_indices = np.where(explained_variance < epsilon)[0]
    
    if len(null_indices) == 0:
        # Take components with lowest variance
        num_null = max(1, len(explained_variance) // 10)
        null_indices = np.arange(len(explained_variance) - num_null, len(explained_variance))
    
    # Extract null-basis from PCA components
    null_basis = torch.from_numpy(pca.components_[null_indices].T).float()
    
    return null_basis


def filter_by_variance(
    basis: torch.Tensor,
    data: Optional[torch.Tensor] = None,
    threshold: float = 1e-5
) -> torch.Tensor:
    """
    Filter basis vectors by post-projection variance.
    
    Args:
        basis: Basis vectors to filter (shape: [hidden_size, num_basis])
        data: Optional data to compute variance on (shape: [num_samples, hidden_size])
        threshold: Variance threshold for filtering
        
    Returns:
        filtered_basis: Filtered basis vectors with variance above threshold
    """
    if data is None:
        # Generate random data for variance computation if not provided
        data = torch.randn(1000, basis.size(0))
    
    # Project data onto basis
    projections = data @ basis  # Shape: [num_samples, num_basis]
    
    # Compute variance for each basis dimension
    variances = projections.var(dim=0)
    
    # Filter basis vectors with variance above threshold
    valid_indices = torch.where(variances > threshold)[0]
    
    if len(valid_indices) == 0:
        # If all variances are below threshold, keep the top-k
        k = min(5, basis.size(1))
        _, valid_indices = torch.topk(variances, k)
    
    filtered_basis = basis[:, valid_indices]
    
    return filtered_basis


def compute_orthogonality_score(basis: torch.Tensor) -> float:
    """
    Compute orthogonality score of basis vectors.
    
    Args:
        basis: Basis vectors (shape: [hidden_size, num_basis])
        
    Returns:
        score: Orthogonality score (1.0 = perfectly orthogonal)
    """
    # Normalize basis vectors
    basis_norm = basis / (basis.norm(dim=0, keepdim=True) + 1e-8)
    
    # Compute Gram matrix
    gram = basis_norm.T @ basis_norm
    
    # Ideal Gram matrix for orthogonal vectors is identity
    identity = torch.eye(gram.size(0), device=gram.device)
    
    # Compute orthogonality score
    off_diagonal_sum = (gram - identity).abs().sum()
    max_off_diagonal = gram.size(0) * (gram.size(0) - 1)
    
    score = 1.0 - (off_diagonal_sum / max_off_diagonal).item()
    
    return score


def project_to_null_space(
    data: torch.Tensor,
    null_basis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project data to null space and compute reconstruction.
    
    Args:
        data: Input data (shape: [batch_size, seq_len, hidden_size] or [batch_size, hidden_size])
        null_basis: Null-basis vectors (shape: [hidden_size, null_dim])
        
    Returns:
        null_projection: Projection in null space
        reconstruction: Reconstructed data from null space
    """
    original_shape = data.shape
    
    # Flatten to 2D if needed
    if data.dim() == 3:
        data_flat = data.reshape(-1, data.size(-1))
    else:
        data_flat = data
    
    # Project to null space
    null_projection = data_flat @ null_basis  # Shape: [batch*seq, null_dim]
    
    # Reconstruct from null space
    reconstruction = null_projection @ null_basis.T  # Shape: [batch*seq, hidden_size]
    
    # Reshape back to original shape if needed
    if len(original_shape) == 3:
        null_projection = null_projection.reshape(original_shape[0], original_shape[1], -1)
        reconstruction = reconstruction.reshape(original_shape)
    
    return null_projection, reconstruction


if __name__ == "__main__":
    # Simple test
    print("SVD Utilities Module Loaded Successfully")
    
    # Test orthogonality score
    test_basis = torch.eye(10)[:, :5]  # 5 orthogonal vectors
    score = compute_orthogonality_score(test_basis)
    print(f"Orthogonality score for identity basis: {score:.4f}")
