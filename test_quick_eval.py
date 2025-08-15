#!/usr/bin/env python3
"""
Quick evaluation test without downloading models
"""
import os
import sys
import torch
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def run_quick_evaluation():
    """Run evaluation with local dummy model"""
    from transformers import AutoTokenizer
    from utils.dataset import create_dummy_data, get_dataloader
    from models import NSRPOModel, NullDecoder
    import torch.nn as nn
    
    print("=== Quick Evaluation Test ===\n")
    
    # 1. Setup
    print("1. Setting up components...")
    
    # Create tokenizer (will download once and cache)
    print("   Creating tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dummy model instead of downloading full GPT-2
    print("   Creating dummy model...")
    class DummyGPT2(nn.Module):
        """Minimal dummy model for testing"""
        def __init__(self, vocab_size=50257, hidden_size=768):
            super().__init__()
            self.config = type('Config', (), {
                'hidden_size': hidden_size,
                'vocab_size': vocab_size,
                'n_layer': 12,
                'n_head': 12
            })()
            self.transformer = nn.Linear(hidden_size, hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)
            
        def forward(self, input_ids, attention_mask=None, labels=None, 
                    output_hidden_states=False, output_attentions=False, 
                    return_dict=True, **kwargs):
            batch_size, seq_len = input_ids.shape
            hidden_size = self.config.hidden_size
            
            # Dummy hidden states
            hidden_states = torch.randn(batch_size, seq_len, hidden_size)
            hidden_states = self.transformer(hidden_states)
            logits = self.lm_head(hidden_states)
            
            loss = None
            if labels is not None:
                # Simple cross-entropy loss
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            
            # Return in transformers format
            from types import SimpleNamespace
            outputs = SimpleNamespace(
                logits=logits, 
                loss=loss,
                hidden_states=[hidden_states] if output_hidden_states else None,
                attentions=None
            )
            return outputs
    
    base_model = DummyGPT2()
    
    # Create NSRPO model
    print("   Creating NSRPO model...")
    hidden_size = base_model.config.hidden_size
    vocab_size = base_model.config.vocab_size
    null_dim = 64
    
    # Create null basis
    q, _ = torch.linalg.qr(torch.randn(hidden_size, null_dim))
    null_basis = q[:, :null_dim].contiguous()
    
    # Create null decoder
    null_decoder = NullDecoder(
        hidden_size=hidden_size,
        null_basis=null_basis,
        vocab_size=vocab_size,
        num_layers=2,
        nhead=4,
        dropout=0.1
    )
    
    # Create NSRPO model
    model = NSRPOModel(
        base_model=base_model,
        null_decoder=null_decoder,
        alpha_1=0.1,
        alpha_2=0.1,
        alpha_3=0.05
    )
    
    print("[OK] Model setup complete\n")
    
    # 2. Create data
    print("2. Creating evaluation data...")
    dummy_data = create_dummy_data(20)  # Small dataset for quick test
    eval_dataloader = get_dataloader(
        data=dummy_data,
        tokenizer=tokenizer,
        batch_size=4,
        max_length=128,
        shuffle=False
    )
    print(f"[OK] Created dataloader with {len(dummy_data)} samples\n")
    
    # 3. Run evaluation
    print("3. Running evaluation...")
    from evaluate import NSRPOEvaluator
    
    evaluator = NSRPOEvaluator(model, tokenizer, device='cpu')
    
    # Test individual metrics
    print("   Computing accuracy metrics...")
    accuracy_metrics = evaluator.compute_accuracy_metrics(eval_dataloader, max_batches=2)
    print(f"   Token Accuracy: {accuracy_metrics['token_accuracy']:.4f}")
    print(f"   Sequence Accuracy: {accuracy_metrics['sequence_accuracy']:.4f}")
    
    print("\n   Computing perplexity...")
    perplexity_metrics = evaluator.compute_perplexity(eval_dataloader, max_batches=2)
    print(f"   Perplexity: {perplexity_metrics['perplexity']:.2f}")
    print(f"   Average Loss: {perplexity_metrics['avg_loss']:.4f}")
    
    print("\n   Computing training efficiency metrics...")
    efficiency_metrics = evaluator.compute_training_efficiency_metrics(eval_dataloader, max_batches=2)
    print(f"   Gradient Variance: {efficiency_metrics.get('variance', 0.0):.6f}")
    
    # Save results
    results = {
        'accuracy': accuracy_metrics,
        'perplexity': perplexity_metrics,
        'training_efficiency': efficiency_metrics,
        'model_type': 'NSRPOModel (dummy)',
        'status': 'success'
    }
    
    output_file = 'quick_eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n[OK] Results saved to {output_file}")
    print("\n=== Evaluation Complete ===")
    
    return True

if __name__ == "__main__":
    try:
        success = run_quick_evaluation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAIL] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)