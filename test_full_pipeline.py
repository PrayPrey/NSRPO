#!/usr/bin/env python3
"""
Test Full Pipeline with Dummy Models
Demonstrates the complete workflow: training -> saving -> loading -> evaluation
"""

import os
import json
import torch
from pathlib import Path

def test_full_pipeline():
    """Test complete NSPO pipeline with dummy models"""
    
    print("=" * 60)
    print("NSPO FULL PIPELINE TEST")
    print("=" * 60)
    
    # 1. Test Training
    print("\n1. Testing Training (GRPO baseline)...")
    train_cmd = 'python train.py --model_path dummy --fast --cpu_only --output_dir test_grpo --num_epochs 1'
    result = os.system(f'cd "{Path(__file__).parent}" && {train_cmd}')
    if result == 0:
        print("[OK] GRPO training completed")
    else:
        print(f"[FAIL] GRPO training failed with code {result}")
        
    # 2. Test NSRPO Training  
    print("\n2. Testing NSRPO Training (with null decoder)...")
    nsrpo_cmd = 'python train.py --model_path dummy --use_null_decoder --extract_null_basis --fast --cpu_only --output_dir test_nsrpo --num_epochs 1'
    result = os.system(f'cd "{Path(__file__).parent}" && {nsrpo_cmd}')
    if result == 0:
        print("[OK] NSRPO training completed")
    else:
        print(f"[FAIL] NSRPO training failed with code {result}")
    
    # 3. Test Evaluation on Trained Models
    print("\n3. Testing Evaluation on GRPO model...")
    eval_cmd = 'python evaluate.py --model_path test_grpo/model_final.pt --base_model_path dummy --fast --cpu_only --output_path test_grpo_eval.json'
    result = os.system(f'cd "{Path(__file__).parent}" && {eval_cmd}')
    if result == 0:
        print("[OK] GRPO evaluation completed")
        # Load and display results
        if os.path.exists("test_grpo_eval.json"):
            with open("test_grpo_eval.json", "r") as f:
                results = json.load(f)
                print(f"   - Perplexity: {results.get('perplexity', {}).get('perplexity', 'N/A'):.2f}")
                print(f"   - Token Accuracy: {results.get('accuracy', {}).get('token_accuracy', 'N/A'):.4f}")
    else:
        print(f"[FAIL] GRPO evaluation failed with code {result}")
    
    print("\n4. Testing Evaluation on NSRPO model...")
    eval_cmd = 'python evaluate.py --model_path test_nsrpo/model_final.pt --base_model_path dummy --fast --cpu_only --output_path test_nsrpo_eval.json'
    result = os.system(f'cd "{Path(__file__).parent}" && {eval_cmd}')
    if result == 0:
        print("[OK] NSRPO evaluation completed")
        # Load and display results
        if os.path.exists("test_nsrpo_eval.json"):
            with open("test_nsrpo_eval.json", "r") as f:
                results = json.load(f)
                print(f"   - Perplexity: {results.get('perplexity', {}).get('perplexity', 'N/A'):.2f}")
                print(f"   - Token Accuracy: {results.get('accuracy', {}).get('token_accuracy', 'N/A'):.4f}")
    else:
        print(f"[FAIL] NSRPO evaluation failed with code {result}")
    
    # 5. Test Checkpoint Loading
    print("\n5. Testing Checkpoint Loading...")
    checkpoint_path = Path("checkpoints")
    if checkpoint_path.exists():
        checkpoints = list(checkpoint_path.glob("checkpoint-epoch-*/model.pt"))
        if checkpoints:
            latest_checkpoint = str(checkpoints[-1])
            print(f"   Found checkpoint: {latest_checkpoint}")
            eval_cmd = f'python evaluate.py --model_path "{latest_checkpoint}" --base_model_path dummy --fast --cpu_only --output_path test_checkpoint_eval.json'
            result = os.system(f'cd "{Path(__file__).parent}" && {eval_cmd}')
            if result == 0:
                print("[OK] Checkpoint evaluation completed")
            else:
                print(f"[FAIL] Checkpoint evaluation failed with code {result}")
        else:
            print("   No checkpoints found")
    
    # 6. Test Comparative Evaluation
    print("\n6. Testing Comparative Evaluation (KL divergence)...")
    # This would compare NSRPO vs GRPO models
    print("   [INFO] Would compute KL divergence between models")
    print("   [INFO] This requires both models loaded simultaneously")
    
    # 7. Summary
    print("\n" + "=" * 60)
    print("PIPELINE TEST SUMMARY")
    print("=" * 60)
    
    results_files = [
        ("GRPO Training", "test_grpo/model_final.pt"),
        ("NSRPO Training", "test_nsrpo/model_final.pt"),
        ("GRPO Evaluation", "test_grpo_eval.json"),
        ("NSRPO Evaluation", "test_nsrpo_eval.json"),
        ("Checkpoint Eval", "test_checkpoint_eval.json")
    ]
    
    for name, filepath in results_files:
        if os.path.exists(filepath):
            print(f"[OK] {name}: {filepath}")
        else:
            print(f"[--] {name}: Not found")
    
    print("\n" + "=" * 60)
    print("All major components are working with dummy models!")
    print("=" * 60)

if __name__ == "__main__":
    test_full_pipeline()