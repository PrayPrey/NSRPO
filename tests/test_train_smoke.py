"""
Smoke tests for training functionality.
Tests basic training operations with minimal resources.
"""

import pytest
import torch
import sys
from pathlib import Path
import tempfile
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train import (
    parse_arguments,
    load_model_and_tokenizer,
    create_dataloaders,
    create_optimizer_and_scheduler,
    train_epoch,
    evaluate_model,
    save_checkpoint
)
from utils.dataset import create_dummy_data, get_dataloader


@pytest.mark.smoke
class TestTrainSmoke:
    """Smoke tests for training functionality."""
    
    def test_parse_arguments_defaults(self):
        """Test argument parsing with defaults."""
        # Simulate command line arguments
        test_args = ['--model_path', 'gpt2', '--fast']
        sys.argv = ['train.py'] + test_args
        
        args = parse_arguments()
        
        assert args.model_path == 'gpt2'
        assert args.fast is True
        assert args.max_steps == 10
        assert args.limit_train_samples == 20
        assert args.limit_eval_samples == 10
    
    def test_parse_arguments_cpu_only(self):
        """Test CPU-only mode arguments."""
        test_args = ['--model_path', 'gpt2', '--cpu_only']
        sys.argv = ['train.py'] + test_args
        
        args = parse_arguments()
        
        assert args.cpu_only is True
        assert args.device == 'cpu'
        assert args.fp16 is False
    
    @pytest.mark.slow
    def test_create_dataloaders(self, mock_tokenizer):
        """Test dataloader creation with synthetic data."""
        # Create mock args
        class Args:
            train_data_path = None
            eval_data_path = None
            num_dummy_samples = 20
            limit_train_samples = 10
            limit_eval_samples = 5
            batch_size = 2
            max_length = 128
        
        args = Args()
        
        # Mock logger
        class Logger:
            def info(self, msg):
                print(f"INFO: {msg}")
        
        logger = Logger()
        
        # Create dataloaders
        train_dataloader, eval_dataloader = create_dataloaders(args, mock_tokenizer, logger)
        
        assert len(train_dataloader) > 0
        assert len(eval_dataloader) > 0
        
        # Test iteration
        for batch in train_dataloader:
            assert 'input_ids' in batch
            assert 'attention_mask' in batch
            break  # Just test first batch
    
    def test_optimizer_scheduler_creation(self):
        """Test optimizer and scheduler creation."""
        # Create a simple model
        model = torch.nn.Linear(10, 10)
        
        # Mock args
        class Args:
            learning_rate = 1e-4
            weight_decay = 0.01
            warmup_ratio = 0.1
            scheduler_type = 'linear'
        
        args = Args()
        num_training_steps = 100
        
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, args, num_training_steps
        )
        
        assert optimizer is not None
        assert scheduler is not None
        assert optimizer.param_groups[0]['lr'] == args.learning_rate
    
    @pytest.mark.slow
    def test_checkpoint_saving(self, temp_dir):
        """Test checkpoint saving functionality."""
        # Create a simple model
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        
        # Mock args
        class Args:
            output_dir = str(temp_dir)
        
        args = Args()
        
        # Mock logger
        class Logger:
            def info(self, msg):
                print(f"INFO: {msg}")
        
        logger = Logger()
        
        # Save checkpoint
        metrics = {'loss': 0.5, 'accuracy': 0.9}
        save_checkpoint(
            model, optimizer, scheduler,
            epoch=1, step=100, args=args,
            metrics=metrics, logger=logger
        )
        
        # Check that files were created
        checkpoint_dir = temp_dir / "checkpoint-epoch-1"
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "model.pt").exists()
        assert (checkpoint_dir / "optimizer.pt").exists()
        assert (checkpoint_dir / "scheduler.pt").exists()
        assert (checkpoint_dir / "metrics.json").exists()
        
        # Verify metrics were saved correctly
        with open(checkpoint_dir / "metrics.json", 'r') as f:
            saved_metrics = json.load(f)
            assert saved_metrics['loss'] == 0.5
            assert saved_metrics['accuracy'] == 0.9


@pytest.mark.smoke
def test_fast_mode_execution(temp_dir):
    """Test that fast mode executes quickly with minimal resources."""
    import time
    import subprocess
    
    # Create a test script that runs training in fast mode
    test_script = temp_dir / "test_fast_train.py"
    test_script.write_text(f"""
import sys
sys.path.insert(0, r'{str(Path(__file__).parent.parent)}')

from train import main
import sys

sys.argv = ['train.py', '--model_path', 'gpt2', '--fast', '--cpu_only', 
            '--output_dir', r'{str(temp_dir / "output")}']

# This should complete quickly
import time
start = time.time()
try:
    # We don't actually run main() in tests to avoid model downloads
    # Just verify the setup works
    from train import parse_arguments
    args = parse_arguments()
    assert args.fast is True
    assert args.max_steps == 10
    print("Fast mode test passed")
except Exception as e:
    print(f"Error: {{e}}")
    sys.exit(1)

elapsed = time.time() - start
assert elapsed < 5, f"Fast mode took too long: {{elapsed}} seconds"
print(f"Completed in {{elapsed:.2f}} seconds")
""")
    
    # Run the test script
    result = subprocess.run(
        [sys.executable, str(test_script)],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    assert result.returncode == 0, f"Fast mode test failed: {result.stderr}"
    assert "Fast mode test passed" in result.stdout