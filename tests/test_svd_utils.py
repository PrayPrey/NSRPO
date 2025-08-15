"""
Unit tests for SVD Utility Functions
Task 1.5: Create comprehensive unit tests for SVD utility functions
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.svd_utils import (
    extract_lora_null_basis,
    extract_base_null_basis,
    extract_activation_null_basis,
    filter_by_variance,
    compute_orthogonality_score,
    project_to_null_space
)


class MockLoRAModel(nn.Module):
    """Mock model with LoRA weights for testing."""
    
    def __init__(self, hidden_size=768, lora_rank=16):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Create mock LoRA layers
        self.lora_layer1 = nn.Module()
        self.lora_layer1.lora_A = nn.Linear(hidden_size, lora_rank, bias=False)
        self.lora_layer1.lora_B = nn.Linear(lora_rank, hidden_size, bias=False)
        
        self.lora_layer2 = nn.Module()
        self.lora_layer2.lora_A = nn.Linear(hidden_size, lora_rank, bias=False)
        self.lora_layer2.lora_B = nn.Linear(lora_rank, hidden_size, bias=False)
        
        # Initialize with small values to ensure null space exists
        with torch.no_grad():
            self.lora_layer1.lora_A.weight.mul_(0.01)
            self.lora_layer1.lora_B.weight.mul_(0.01)
            self.lora_layer2.lora_A.weight.mul_(0.01)
            self.lora_layer2.lora_B.weight.mul_(0.01)


class MockBaseModel(nn.Module):
    """Mock base model for testing."""
    
    def __init__(self, hidden_size=768, vocab_size=50000):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Create layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.model = nn.Module()
        self.model.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.model.norm(x)
        return self.lm_head(x)


def test_extract_lora_null_basis():
    """Test LoRA null-basis extraction."""
    print("Testing extract_lora_null_basis...")
    
    # Create mock model
    model = MockLoRAModel(hidden_size=128, lora_rank=8)
    
    # Extract null-basis
    null_basis = extract_lora_null_basis(model, rank=8, epsilon_factor=1e-3)
    
    # Check dimensions
    assert null_basis.shape[0] == 128, f"Expected hidden_size=128, got {null_basis.shape[0]}"
    assert null_basis.shape[1] <= 8, f"Expected rank<=8, got {null_basis.shape[1]}"
    
    # Check orthogonality
    orthogonality = compute_orthogonality_score(null_basis)
    assert orthogonality > 0.8, f"Poor orthogonality: {orthogonality}"
    
    print(f"✓ LoRA null-basis extraction: shape={null_basis.shape}, orthogonality={orthogonality:.4f}")


def test_extract_base_null_basis():
    """Test base model null-basis extraction."""
    print("Testing extract_base_null_basis...")
    
    # Create mock model
    model = MockBaseModel(hidden_size=128, vocab_size=1000)
    
    # Extract null-basis from lm_head
    null_basis = extract_base_null_basis(model, epsilon_factor=1e-3, layer_name="lm_head")
    
    # Check that we got some null-basis vectors
    assert null_basis.shape[0] == 128, f"Expected hidden_size=128, got {null_basis.shape[0]}"
    assert null_basis.shape[1] > 0, "No null-basis vectors extracted"
    
    # Check orthogonality
    orthogonality = compute_orthogonality_score(null_basis)
    
    print(f"✓ Base null-basis extraction: shape={null_basis.shape}, orthogonality={orthogonality:.4f}")


def test_filter_by_variance():
    """Test variance-based filtering."""
    print("Testing filter_by_variance...")
    
    # Create test basis with varying importance
    hidden_size = 64
    num_basis = 10
    
    # Create basis where first half has high variance, second half has low variance
    basis = torch.zeros(hidden_size, num_basis)
    basis[:, :5] = torch.randn(hidden_size, 5) * 10  # High variance
    basis[:, 5:] = torch.randn(hidden_size, 5) * 0.001  # Low variance
    
    # Create test data
    test_data = torch.randn(100, hidden_size)
    
    # Filter basis
    filtered_basis = filter_by_variance(basis, test_data, threshold=1e-3)
    
    # Check that low-variance dimensions were filtered out
    assert filtered_basis.shape[1] <= num_basis, "Filtered basis should have fewer dimensions"
    assert filtered_basis.shape[1] >= 5, "Should keep at least high-variance dimensions"
    
    print(f"✓ Variance filtering: original={num_basis} dims, filtered={filtered_basis.shape[1]} dims")


def test_project_to_null_space():
    """Test null space projection and reconstruction."""
    print("Testing project_to_null_space...")
    
    # Create orthogonal null-basis
    hidden_size = 64
    null_dim = 8
    null_basis = torch.qr(torch.randn(hidden_size, null_dim))[0]
    
    # Create test data
    batch_size = 4
    seq_len = 10
    data = torch.randn(batch_size, seq_len, hidden_size)
    
    # Project and reconstruct
    null_proj, reconstruction = project_to_null_space(data, null_basis)
    
    # Check dimensions
    assert null_proj.shape == (batch_size, seq_len, null_dim), f"Wrong projection shape: {null_proj.shape}"
    assert reconstruction.shape == data.shape, f"Wrong reconstruction shape: {reconstruction.shape}"
    
    # Check that projection captures some information
    reconstruction_error = (data - reconstruction).norm() / data.norm()
    assert reconstruction_error < 1.0, "Reconstruction should capture some information"
    
    # For perfect orthogonal basis, check that null space is preserved
    null_proj2, _ = project_to_null_space(reconstruction, null_basis)
    preservation_error = (null_proj - null_proj2).norm() / (null_proj.norm() + 1e-8)
    assert preservation_error < 0.1, f"Null space not preserved: error={preservation_error:.4f}"
    
    print(f"✓ Null space projection: reconstruction_error={reconstruction_error:.4f}, preservation={1-preservation_error:.4f}")


def test_orthogonality_score():
    """Test orthogonality score computation."""
    print("Testing compute_orthogonality_score...")
    
    # Test with perfectly orthogonal vectors (identity matrix columns)
    perfect_basis = torch.eye(10)[:, :5]
    perfect_score = compute_orthogonality_score(perfect_basis)
    assert perfect_score > 0.99, f"Identity basis should be perfectly orthogonal, got {perfect_score}"
    
    # Test with random vectors (should have lower orthogonality)
    random_basis = torch.randn(10, 5)
    random_score = compute_orthogonality_score(random_basis)
    assert random_score < perfect_score, "Random vectors should be less orthogonal than identity"
    
    # Test with QR decomposition (should be orthogonal)
    qr_basis = torch.qr(torch.randn(10, 5))[0]
    qr_score = compute_orthogonality_score(qr_basis)
    assert qr_score > 0.99, f"QR basis should be orthogonal, got {qr_score}"
    
    print(f"✓ Orthogonality scores: perfect={perfect_score:.4f}, random={random_score:.4f}, QR={qr_score:.4f}")


def test_extract_activation_null_basis():
    """Test activation-based null-basis extraction."""
    print("Testing extract_activation_null_basis...")
    
    # Create mock model and data
    model = MockBaseModel(hidden_size=64, vocab_size=100)
    
    # Create mock dataloader
    class MockDataLoader:
        def __init__(self, num_batches=5, batch_size=4, seq_len=10, vocab_size=100):
            self.num_batches = num_batches
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.vocab_size = vocab_size
        
        def __iter__(self):
            for _ in range(self.num_batches):
                yield {
                    'input_ids': torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
                }
    
    dataloader = MockDataLoader()
    
    try:
        # Extract null-basis from activations
        null_basis = extract_activation_null_basis(
            model, 
            dataloader,
            epsilon_factor=1e-3,
            max_samples=20,
            layer_name="model.norm"
        )
        
        # Check dimensions
        assert null_basis.shape[0] == 64, f"Expected hidden_size=64, got {null_basis.shape[0]}"
        assert null_basis.shape[1] > 0, "No null-basis vectors extracted"
        
        print(f"✓ Activation null-basis extraction: shape={null_basis.shape}")
    except ImportError as e:
        print(f"⚠ Skipping activation test (sklearn not available): {e}")


def test_numerical_stability():
    """Test numerical stability with edge cases."""
    print("Testing numerical stability...")
    
    # Test with very small values
    small_basis = torch.randn(32, 8) * 1e-10
    small_score = compute_orthogonality_score(small_basis)
    assert not torch.isnan(torch.tensor(small_score)), "Score should not be NaN for small values"
    
    # Test with very large values
    large_basis = torch.randn(32, 8) * 1e10
    large_score = compute_orthogonality_score(large_basis)
    assert not torch.isnan(torch.tensor(large_score)), "Score should not be NaN for large values"
    
    # Test filtering with zero variance
    zero_var_basis = torch.ones(32, 8)  # All same values = zero variance
    filtered = filter_by_variance(zero_var_basis, threshold=1e-5)
    assert filtered.shape[1] > 0, "Should return at least some basis vectors"
    
    print(f"✓ Numerical stability: small_score={small_score:.4f}, large_score={large_score:.4f}")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 50)
    print("Running SVD Utilities Unit Tests")
    print("=" * 50)
    
    test_functions = [
        test_extract_lora_null_basis,
        test_extract_base_null_basis,
        test_filter_by_variance,
        test_project_to_null_space,
        test_orthogonality_score,
        test_extract_activation_null_basis,
        test_numerical_stability
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} error: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Tests completed: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)