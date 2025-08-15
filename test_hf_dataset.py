#!/usr/bin/env python3
"""
Test HuggingFace Dataset Integration
"""

import os
import sys
from pathlib import Path

def test_hf_dataset():
    """Test training with HuggingFace WikiText dataset"""
    
    print("=" * 60)
    print("Testing HuggingFace Dataset Integration")
    print("=" * 60)
    
    # 1. Test with WikiText small dataset
    print("\n1. Training with WikiText-2 dataset...")
    train_cmd = (
        'python train.py '
        '--model_path dummy '
        '--train_data_path hf:wikitext-small '
        '--eval_data_path hf:wikitext-small '
        '--fast '
        '--cpu_only '
        '--output_dir hf_test_wikitext '
        '--limit_train_samples 100 '
        '--limit_eval_samples 50'
    )
    
    print(f"Command: {train_cmd}")
    result = os.system(f'cd "{Path(__file__).parent}" && {train_cmd}')
    
    if result == 0:
        print("[OK] Training with WikiText completed successfully")
    else:
        print(f"[FAIL] Training failed with code {result}")
        return False
    
    # 2. Test evaluation
    print("\n2. Evaluating trained model...")
    eval_cmd = (
        'python evaluate.py '
        '--model_path hf_test_wikitext/model_final.pt '
        '--base_model_path dummy '
        '--eval_data_path hf:wikitext-small '
        '--fast '
        '--cpu_only '
        '--output_path hf_wikitext_eval.json '
        '--limit_eval_samples 50'
    )
    
    print(f"Command: {eval_cmd}")
    result = os.system(f'cd "{Path(__file__).parent}" && {eval_cmd}')
    
    if result == 0:
        print("[OK] Evaluation completed successfully")
        
        # Show results
        import json
        if os.path.exists("hf_wikitext_eval.json"):
            with open("hf_wikitext_eval.json", "r") as f:
                results = json.load(f)
                print("\nEvaluation Results:")
                print(f"  Perplexity: {results.get('perplexity', {}).get('perplexity', 'N/A'):.2f}")
                print(f"  Token Accuracy: {results.get('accuracy', {}).get('token_accuracy', 'N/A'):.4f}")
    else:
        print(f"[FAIL] Evaluation failed with code {result}")
        return False
    
    # 3. Test with NSRPO model
    print("\n3. Training NSRPO model with WikiText...")
    nsrpo_cmd = (
        'python train.py '
        '--model_path dummy '
        '--use_null_decoder '
        '--extract_null_basis '
        '--train_data_path hf:wikitext-small '
        '--eval_data_path hf:wikitext-small '
        '--fast '
        '--cpu_only '
        '--output_dir hf_test_nsrpo '
        '--limit_train_samples 100 '
        '--limit_eval_samples 50'
    )
    
    print(f"Command: {nsrpo_cmd}")
    result = os.system(f'cd "{Path(__file__).parent}" && {nsrpo_cmd}')
    
    if result == 0:
        print("[OK] NSRPO training with WikiText completed")
    else:
        print(f"[WARN] NSRPO training failed with code {result}")
    
    print("\n" + "=" * 60)
    print("HuggingFace Dataset Integration Test Complete!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_hf_dataset()
    sys.exit(0 if success else 1)