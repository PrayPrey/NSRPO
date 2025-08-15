"""
Smoke tests for evaluation functionality.
Tests basic evaluation operations with minimal resources.
"""

import pytest
import torch
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluate import (
    NSRPOEvaluator,
    parse_arguments,
    setup_logging
)
from utils.dataset import create_dummy_data, get_dataloader


@pytest.mark.smoke
class TestEvalSmoke:
    """Smoke tests for evaluation functionality."""
    
    def test_parse_arguments_fast_mode(self):
        """Test evaluation argument parsing with fast mode."""
        test_args = ['--model_path', 'gpt2', '--fast']
        sys.argv = ['evaluate.py'] + test_args
        
        args = parse_arguments()
        
        assert args.fast is True
        assert args.max_batches == 5
        assert args.limit_eval_samples == 50
        assert args.include_generation is False
    
    def test_evaluator_initialization(self, mock_tokenizer):
        """Test NSRPOEvaluator initialization."""
        # Create a simple model
        model = torch.nn.Linear(10, 10)
        
        evaluator = NSRPOEvaluator(
            model=model,
            tokenizer=mock_tokenizer,
            device='cpu'
        )
        
        assert evaluator.model is not None
        assert evaluator.tokenizer is not None
        assert evaluator.device == torch.device('cpu')
    
    @pytest.mark.slow
    def test_accuracy_computation(self, mock_tokenizer, synthetic_eval_data):
        """Test accuracy metrics computation."""
        # Create a mock model that returns predictable outputs
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
            
            def forward(self, input_ids, attention_mask=None, labels=None):
                batch_size, seq_len = input_ids.shape
                vocab_size = 50000
                
                # Create mock logits
                logits = torch.randn(batch_size, seq_len, vocab_size)
                
                # Create mock loss
                loss = torch.tensor(0.5)
                
                # Return mock output
                class Output:
                    def __init__(self):
                        self.logits = logits
                        self.loss = loss
                
                return Output()
        
        model = MockModel()
        evaluator = NSRPOEvaluator(model, mock_tokenizer, device='cpu')
        
        # Create dataloader
        dataloader = get_dataloader(
            data=synthetic_eval_data,
            tokenizer=mock_tokenizer,
            batch_size=2,
            max_length=128,
            shuffle=False
        )
        
        # Compute accuracy metrics
        metrics = evaluator.compute_accuracy_metrics(
            dataloader, max_batches=2
        )
        
        assert 'token_accuracy' in metrics
        assert 'sequence_accuracy' in metrics
        assert 0 <= metrics['token_accuracy'] <= 1
        assert 0 <= metrics['sequence_accuracy'] <= 1
    
    @pytest.mark.slow
    def test_perplexity_computation(self, mock_tokenizer, synthetic_eval_data):
        """Test perplexity computation."""
        # Create a mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, input_ids, attention_mask=None, labels=None):
                # Return consistent loss for testing
                class Output:
                    def __init__(self):
                        self.loss = torch.tensor(2.0)  # Log perplexity
                
                return Output()
        
        model = MockModel()
        evaluator = NSRPOEvaluator(model, mock_tokenizer, device='cpu')
        
        # Create dataloader
        dataloader = get_dataloader(
            data=synthetic_eval_data,
            tokenizer=mock_tokenizer,
            batch_size=2,
            max_length=128,
            shuffle=False
        )
        
        # Compute perplexity
        metrics = evaluator.compute_perplexity(dataloader, max_batches=2)
        
        assert 'perplexity' in metrics
        assert 'avg_loss' in metrics
        assert metrics['avg_loss'] == 2.0
        assert metrics['perplexity'] == pytest.approx(torch.exp(torch.tensor(2.0)).item(), rel=1e-3)
    
    def test_comprehensive_evaluation(self, mock_tokenizer, synthetic_eval_data, temp_dir):
        """Test comprehensive evaluation pipeline."""
        # Create a mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, input_ids, attention_mask=None, labels=None):
                batch_size, seq_len = input_ids.shape
                vocab_size = 50000
                
                class Output:
                    def __init__(self):
                        self.logits = torch.randn(batch_size, seq_len, vocab_size)
                        self.loss = torch.tensor(1.5)
                
                return Output()
            
            def generate(self, input_ids, **kwargs):
                # Mock generation
                batch_size = input_ids.shape[0]
                max_length = kwargs.get('max_length', 20)
                return torch.randint(0, 50000, (batch_size, max_length))
        
        model = MockModel()
        evaluator = NSRPOEvaluator(model, mock_tokenizer, device='cpu')
        
        # Create dataloader
        dataloader = get_dataloader(
            data=synthetic_eval_data,
            tokenizer=mock_tokenizer,
            batch_size=2,
            max_length=128,
            shuffle=False
        )
        
        # Run comprehensive evaluation
        results = evaluator.comprehensive_evaluation(
            eval_dataloader=dataloader,
            baseline_model=None,
            max_batches=2,
            include_generation=False
        )
        
        assert 'accuracy' in results
        assert 'perplexity' in results
        assert 'metadata' in results
        
        # Save results
        results_file = temp_dir / "eval_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        assert results_file.exists()


@pytest.mark.smoke
def test_fast_evaluation_execution(temp_dir):
    """Test that fast evaluation mode executes quickly."""
    import subprocess
    
    # Create a test script
    test_script = temp_dir / "test_fast_eval.py"
    test_script.write_text(f"""
import sys
sys.path.insert(0, r'{str(Path(__file__).parent.parent)}')

from evaluate import parse_arguments
import sys

sys.argv = ['evaluate.py', '--model_path', 'gpt2', '--fast', '--cpu_only',
            '--output_path', r'{str(temp_dir / "results.json")}']

# Test argument parsing
args = parse_arguments()
assert args.fast is True
assert args.max_batches == 5
assert args.limit_eval_samples == 50
print("Fast evaluation mode test passed")
""")
    
    # Run the test script
    result = subprocess.run(
        [sys.executable, str(test_script)],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    assert result.returncode == 0, f"Fast eval test failed: {result.stderr}"
    assert "Fast evaluation mode test passed" in result.stdout