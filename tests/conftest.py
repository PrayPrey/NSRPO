"""
Pytest configuration and fixtures for NSRPO tests.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import utilities for fixture creation
from utils.dataset import create_dummy_data, generate_synthetic_eval_data


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""
    return create_dummy_data(num_samples=50)


@pytest.fixture
def synthetic_eval_data():
    """Generate synthetic evaluation data."""
    return generate_synthetic_eval_data(num_samples=20)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self):
            self.pad_token = '[PAD]'
            self.eos_token = '[EOS]'
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.vocab_size = 50000
        
        def __call__(self, texts, **kwargs):
            # Simple mock tokenization
            if isinstance(texts, str):
                texts = [texts]
            
            max_length = kwargs.get('max_length', 10)
            padding = kwargs.get('padding', 'longest')
            
            input_ids = []
            attention_mask = []
            
            for text in texts:
                tokens = text.split()[:max_length]
                ids = list(range(2, 2 + len(tokens)))  # Start from 2 (after special tokens)
                mask = [1] * len(ids)
                
                # Pad if necessary
                if padding == 'max_length':
                    while len(ids) < max_length:
                        ids.append(0)
                        mask.append(0)
                
                input_ids.append(ids)
                attention_mask.append(mask)
            
            # Convert to tensors
            return {
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)
            }
        
        def decode(self, token_ids, skip_special_tokens=True):
            """Mock decoding."""
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            return " ".join([f"token_{id}" for id in token_ids if id > 1 or not skip_special_tokens])
    
    return MockTokenizer()


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return torch.device('cpu')  # Always use CPU for tests


@pytest.fixture
def seed():
    """Fixed seed for reproducibility."""
    seed_value = 42
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    return seed_value


# Configure pytest settings
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests"
    )