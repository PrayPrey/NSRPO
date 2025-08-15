#!/usr/bin/env python3
"""
Example usage of the NSRPO framework.

This script demonstrates how to use the NSRPO (Null-Space Regularized Policy Optimization)
framework for training language models with null-space decoder regularization.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import NSRPOModel, NullDecoder, create_nsrpo_model
from utils.dataset import create_dummy_data, get_dataloader
from utils.svd_utils import extract_lora_null_basis


def example_baseline_training():
    """Example of baseline GRPO training without null-space decoder."""
    print("=== Baseline GRPO Training Example ===")
    
    # This would typically be run with train.py:
    command = """
    python train.py \\
        --model_path "gpt2" \\
        --batch_size 4 \\
        --num_epochs 1 \\
        --learning_rate 5e-5 \\
        --output_dir ./outputs/baseline \\
        --log_every 50
    """
    print(f"Command: {command}")


def example_nsrpo_training():
    """Example of NSRPO training with null-space decoder."""
    print("=== NSRPO Training Example ===")
    
    # This would typically be run with train.py:
    command = """
    python train.py \\
        --model_path "gpt2" \\
        --use_null_decoder \\
        --extract_null_basis \\
        --batch_size 4 \\
        --num_epochs 1 \\
        --learning_rate 5e-5 \\
        --alpha_1 0.1 \\
        --alpha_2 0.1 \\
        --alpha_3 0.05 \\
        --decoder_layers 3 \\
        --decoder_heads 8 \\
        --output_dir ./outputs/nsrpo \\
        --log_every 50
    """
    print(f"Command: {command}")


def example_programmatic_usage():
    """Example of using NSRPO components programmatically."""
    print("=== Programmatic Usage Example ===")
    
    try:
        # Note: This requires PyTorch and transformers to be installed
        print("Loading model and tokenizer...")
        
        # Create a small model for demonstration (this would fail without PyTorch)
        # tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        print("Creating dummy data...")
        dummy_data = create_dummy_data(num_samples=10)
        print(f"Created {len(dummy_data)} dummy samples")
        print(f"Sample: {dummy_data[0]}")
        
        print("\n✓ Programmatic usage example completed (without actual model loading)")
        
    except ImportError as e:
        print(f"Note: {e}")
        print("Install requirements: pip install -r requirements.txt")


def example_null_space_analysis():
    """Example of null-space analysis utilities."""
    print("=== Null-Space Analysis Example ===")
    
    # Create example null-basis
    hidden_size = 768
    null_dim = 64
    null_basis = torch.qr(torch.randn(hidden_size, null_dim))[0]
    
    print(f"Null-basis shape: {null_basis.shape}")
    print(f"Compression ratio: {null_dim/hidden_size:.2%}")
    
    # Check orthogonality
    gram_matrix = null_basis.T @ null_basis
    identity = torch.eye(null_dim)
    orthogonality_error = (gram_matrix - identity).abs().max()
    print(f"Orthogonality error: {orthogonality_error:.6f}")
    
    print("✓ Null-space analysis completed")


def example_loss_components():
    """Example of understanding loss components in NSRPO."""
    print("=== Loss Components Example ===")
    
    print("NSRPO uses a combined loss function:")
    print("L_total = L_RL + α₁·L_CE + α₂·L_cos + α₃·L_preserve")
    print("")
    print("Where:")
    print("- L_RL: Reinforcement learning loss from base model")
    print("- L_CE: Cross-entropy loss from null decoder")
    print("- L_cos: Cosine similarity loss (1 - cosine_similarity)")
    print("- L_preserve: Norm preservation loss")
    print("")
    print("Default weights (from PRD):")
    print("- α₁ = 0.1 (CE loss weight)")
    print("- α₂ = 0.1 (Cosine loss weight)")  
    print("- α₃ = 0.05 (Norm preservation weight)")


def main():
    """Run all examples."""
    print("NSRPO Framework Examples")
    print("=" * 50)
    
    example_baseline_training()
    print()
    
    example_nsrpo_training()
    print()
    
    example_programmatic_usage()
    print()
    
    example_null_space_analysis()
    print()
    
    example_loss_components()
    print()
    
    print("=" * 50)
    print("For actual training, run:")
    print("  python train.py --help")
    print("")
    print("For testing implementations:")
    print("  python -m pytest tests/")


if __name__ == "__main__":
    main()