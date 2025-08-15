#!/usr/bin/env python3
"""
Simple test script for evaluation functionality
"""
import os
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all imports work"""
    try:
        from models import NSRPOModel, NullDecoder, create_nsrpo_model
        print("[OK] Models import successful")
    except Exception as e:
        print(f"[FAIL] Models import failed: {e}")
        return False
    
    try:
        from utils.dataset import get_dataloader, create_dummy_data
        print("[OK] Dataset utils import successful")
    except Exception as e:
        print(f"[FAIL] Dataset utils import failed: {e}")
        return False
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("[OK] Transformers import successful")
    except Exception as e:
        print(f"[FAIL] Transformers import failed: {e}")
        return False
    
    return True

def test_dummy_evaluation():
    """Test evaluation with dummy model and data"""
    try:
        from transformers import AutoTokenizer
        from utils.dataset import create_dummy_data, get_dataloader
        from models import NSRPOModel, NullDecoder
        
        print("\n=== Testing Dummy Evaluation ===")
        
        # Create dummy tokenizer
        print("Creating dummy tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("[OK] Tokenizer created")
        
        # Create dummy data
        print("Creating dummy data...")
        dummy_data = create_dummy_data(10)  # Just 10 samples for quick test
        print(f"[OK] Created {len(dummy_data)} dummy samples")
        
        # Create dataloader
        print("Creating dataloader...")
        dataloader = get_dataloader(
            data=dummy_data,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=128,
            shuffle=False
        )
        print("[OK] Dataloader created")
        
        # Test iteration
        print("Testing dataloader iteration...")
        for i, batch in enumerate(dataloader):
            if i == 0:
                print(f"[OK] First batch shape: input_ids={batch['input_ids'].shape}")
                if 'attention_mask' in batch:
                    print(f"  attention_mask={batch['attention_mask'].shape}")
            if i >= 2:  # Just test first few batches
                break
        print("[OK] Dataloader iteration successful")
        
        # Test simple model creation (without loading full GPT-2)
        print("\nTesting model components...")
        hidden_size = 768  # GPT-2 hidden size
        vocab_size = 50257  # GPT-2 vocab size
        null_dim = 64
        
        # Create null basis
        q, _ = torch.linalg.qr(torch.randn(hidden_size, null_dim))
        null_basis = q[:, :null_dim].contiguous()
        print("[OK] Null basis created")
        
        # Create null decoder
        null_decoder = NullDecoder(
            hidden_size=hidden_size,
            null_basis=null_basis,
            vocab_size=vocab_size,
            num_layers=2,  # Small for testing
            nhead=4,
            dropout=0.1
        )
        print("[OK] Null decoder created")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Dummy evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== NSPO Evaluation Test Suite ===\n")
    
    # Test imports
    if not test_imports():
        print("\n[FAIL] Import tests failed. Please check dependencies.")
        return 1
    
    # Test dummy evaluation
    if not test_dummy_evaluation():
        print("\n[FAIL] Dummy evaluation test failed.")
        return 1
    
    print("\n=== All Tests Passed ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())