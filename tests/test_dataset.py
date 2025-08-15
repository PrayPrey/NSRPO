"""
Unit tests for Dataset Utilities
Task 6: Test Dataset and DataLoader Utilities
"""

import torch
import json
import tempfile
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset import (
    RLDataset,
    rl_collate_fn,
    load_data,
    get_dataloader,
    create_dummy_data,
    BatchSampler
)


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.pad_token = '[PAD]'
        self.eos_token = '[EOS]'
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab = {'[PAD]': 0, '[EOS]': 1, 'hello': 2, 'world': 3, 'test': 4}
    
    def __call__(self, texts, padding='longest', truncation=True, max_length=512, return_tensors='pt'):
        """Simple mock tokenization."""
        if isinstance(texts, str):
            texts = [texts]
        
        input_ids = []
        attention_mask = []
        
        for text in texts:
            # Simple word-based tokenization
            tokens = text.lower().split()[:max_length-2]  # Leave room for special tokens
            ids = [self.vocab.get(token, len(self.vocab)) for token in tokens]
            
            # Add EOS token
            ids.append(self.eos_token_id)
            mask = [1] * len(ids)
            
            input_ids.append(ids)
            attention_mask.append(mask)
        
        # Pad to same length if padding is enabled
        if padding == 'longest':
            max_len = max(len(ids) for ids in input_ids)
        else:
            max_len = max_length
        
        for i in range(len(input_ids)):
            while len(input_ids[i]) < max_len:
                input_ids[i].append(self.pad_token_id)
                attention_mask[i].append(0)
            
            # Truncate if too long
            input_ids[i] = input_ids[i][:max_len]
            attention_mask[i] = attention_mask[i][:max_len]
        
        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
        
        return result


def test_rl_dataset_creation():
    """Test RLDataset creation and basic functionality."""
    print("Testing RLDataset creation...")
    
    # Create sample data
    data = [
        {'prompt': 'What is AI?', 'response': 'AI is artificial intelligence.', 'reward': 0.8},
        {'prompt': 'Explain ML.', 'response': 'ML is machine learning.', 'reward': 0.9},
        {'prompt': 'Define DL.', 'response': 'DL is deep learning.', 'reward': 0.7}
    ]
    
    tokenizer = MockTokenizer()
    dataset = RLDataset(data, tokenizer, max_length=128)
    
    # Test dataset length
    assert len(dataset) == 3, f"Expected 3 items, got {len(dataset)}"
    
    # Test item retrieval
    item = dataset[0]
    assert 'prompt' in item, "Missing 'prompt' in item"
    assert 'response' in item, "Missing 'response' in item"
    assert 'reward' in item, "Missing 'reward' in item"
    assert item['reward'] == 0.8, f"Expected reward 0.8, got {item['reward']}"
    
    # Test without rewards
    dataset_no_rewards = RLDataset(data, tokenizer, include_rewards=False)
    item_no_reward = dataset_no_rewards[0]
    assert 'reward' not in item_no_reward, "Reward should not be included"
    
    print("✓ RLDataset creation and item access working correctly")


def test_collate_function():
    """Test the rl_collate_fn function."""
    print("Testing rl_collate_fn...")
    
    # Create sample batch
    batch = [
        {'prompt': 'What is AI?', 'response': 'AI is artificial intelligence.', 'reward': 0.8},
        {'prompt': 'Explain ML.', 'response': 'ML is machine learning.', 'reward': 0.9}
    ]
    
    tokenizer = MockTokenizer()
    collated = rl_collate_fn(batch, tokenizer, max_length=32)
    
    # Check required keys
    required_keys = ['prompt_input_ids', 'prompt_attention_mask', 
                    'response_input_ids', 'response_attention_mask',
                    'input_ids', 'attention_mask', 'rewards']
    
    for key in required_keys:
        assert key in collated, f"Missing key '{key}' in collated output"
    
    # Check tensor shapes
    batch_size = len(batch)
    assert collated['prompt_input_ids'].shape[0] == batch_size, "Wrong batch size for prompts"
    assert collated['response_input_ids'].shape[0] == batch_size, "Wrong batch size for responses"
    assert collated['rewards'].shape[0] == batch_size, "Wrong batch size for rewards"
    
    # Check data types
    assert collated['prompt_input_ids'].dtype == torch.long, "Wrong dtype for input_ids"
    assert collated['rewards'].dtype == torch.float, "Wrong dtype for rewards"
    
    # Check reward values
    expected_rewards = torch.tensor([0.8, 0.9])
    assert torch.allclose(collated['rewards'], expected_rewards), "Incorrect reward values"
    
    print("✓ Collate function working correctly")


def test_data_loading():
    """Test data loading from files."""
    print("Testing data loading...")
    
    # Test data
    test_data = [
        {'prompt': 'Hello', 'response': 'Hi there!', 'reward': 1.0},
        {'prompt': 'Goodbye', 'response': 'See you later!', 'reward': 0.5}
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test JSON format
        json_file = tmpdir / 'test_data.json'
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
        
        loaded_data = load_data(json_file)
        assert len(loaded_data) == 2, f"Expected 2 items, got {len(loaded_data)}"
        assert loaded_data[0]['prompt'] == 'Hello', "Incorrect data loaded"
        
        # Test JSONL format
        jsonl_file = tmpdir / 'test_data.jsonl'
        with open(jsonl_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        loaded_jsonl = load_data(jsonl_file)
        assert len(loaded_jsonl) == 2, f"Expected 2 items from JSONL, got {len(loaded_jsonl)}"
        assert loaded_jsonl[1]['response'] == 'See you later!', "Incorrect JSONL data"
        
        # Test auto-detection
        auto_loaded = load_data(json_file, data_format='auto')
        assert len(auto_loaded) == 2, "Auto-detection failed for JSON"
    
    print("✓ Data loading from files working correctly")


def test_dataloader_creation():
    """Test DataLoader creation and iteration."""
    print("Testing DataLoader creation...")
    
    # Create test data
    data = create_dummy_data(10)
    tokenizer = MockTokenizer()
    
    # Create dataloader
    dataloader = get_dataloader(
        data=data,
        tokenizer=tokenizer,
        batch_size=3,
        max_length=64,
        shuffle=False
    )
    
    # Test iteration
    batch_count = 0
    total_items = 0
    
    for batch in dataloader:
        batch_count += 1
        batch_size = batch['input_ids'].shape[0]
        total_items += batch_size
        
        # Check batch structure
        assert 'input_ids' in batch, "Missing input_ids in batch"
        assert 'attention_mask' in batch, "Missing attention_mask in batch"
        assert 'rewards' in batch, "Missing rewards in batch"
        
        # Check tensor properties
        assert batch['input_ids'].dim() == 2, "input_ids should be 2D"
        assert batch['attention_mask'].shape == batch['input_ids'].shape, "Mask shape mismatch"
        assert batch['rewards'].shape[0] == batch_size, "Wrong number of rewards"
    
    # Check that all data was processed
    expected_batches = (len(data) + 2) // 3  # ceil(10/3) = 4
    assert batch_count == expected_batches, f"Expected {expected_batches} batches, got {batch_count}"
    assert total_items == len(data), f"Expected {len(data)} total items, processed {total_items}"
    
    print(f"✓ DataLoader creation and iteration: {batch_count} batches, {total_items} items")


def test_batch_sampler():
    """Test custom BatchSampler."""
    print("Testing BatchSampler...")
    
    # Create dummy dataset
    class DummyDataset:
        def __init__(self, size):
            self.size = size
        def __len__(self):
            return self.size
    
    dataset = DummyDataset(10)
    sampler = BatchSampler(dataset, batch_size=3, shuffle=False, drop_last=False)
    
    # Test sampling
    batches = list(sampler)
    assert len(batches) == 4, f"Expected 4 batches (ceil(10/3)), got {len(batches)}"
    
    # Check batch sizes
    assert len(batches[0]) == 3, "First batch should have 3 items"
    assert len(batches[1]) == 3, "Second batch should have 3 items"
    assert len(batches[2]) == 3, "Third batch should have 3 items"
    assert len(batches[3]) == 1, "Last batch should have 1 item"
    
    # Test with drop_last=True
    sampler_drop = BatchSampler(dataset, batch_size=3, shuffle=False, drop_last=True)
    batches_drop = list(sampler_drop)
    assert len(batches_drop) == 3, f"With drop_last, expected 3 batches, got {len(batches_drop)}"
    
    print("✓ BatchSampler working correctly")


def test_dummy_data_creation():
    """Test dummy data creation and saving."""
    print("Testing dummy data creation...")
    
    # Create dummy data
    data = create_dummy_data(5)
    assert len(data) == 5, f"Expected 5 samples, got {len(data)}"
    
    # Check structure
    for item in data:
        assert 'prompt' in item, "Missing 'prompt' in dummy data"
        assert 'response' in item, "Missing 'response' in dummy data"
        assert 'reward' in item, "Missing 'reward' in dummy data"
        assert isinstance(item['reward'], float), "Reward should be float"
        assert -1.0 <= item['reward'] <= 1.0, "Reward should be in range [-1, 1]"
    
    # Test saving
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test JSON saving
        json_path = tmpdir / 'dummy.json'
        saved_data = create_dummy_data(3, save_path=json_path)
        assert json_path.exists(), "JSON file was not created"
        
        # Load and verify
        loaded = load_data(json_path)
        assert len(loaded) == 3, "Saved data not loaded correctly"
        
        # Test JSONL saving
        jsonl_path = tmpdir / 'dummy.jsonl'
        create_dummy_data(2, save_path=jsonl_path)
        assert jsonl_path.exists(), "JSONL file was not created"
        
        loaded_jsonl = load_data(jsonl_path)
        assert len(loaded_jsonl) == 2, "JSONL data not loaded correctly"
    
    print("✓ Dummy data creation and saving working correctly")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")
    
    tokenizer = MockTokenizer()
    
    # Empty data
    empty_dataset = RLDataset([], tokenizer)
    assert len(empty_dataset) == 0, "Empty dataset should have length 0"
    
    # Data without rewards
    data_no_reward = [{'prompt': 'Test', 'response': 'Response'}]
    dataset_no_reward = RLDataset(data_no_reward, tokenizer, include_rewards=True)
    item = dataset_no_reward[0]
    assert item['reward'] == 0.0, "Default reward should be 0.0"
    
    # Very long sequences (should be truncated)
    long_text = 'word ' * 1000
    long_data = [{'prompt': long_text, 'response': long_text, 'reward': 1.0}]
    dataset_long = RLDataset(long_data, tokenizer, max_length=10)
    
    # Should not crash during collation
    batch = rl_collate_fn([dataset_long[0]], tokenizer, max_length=10)
    assert batch['prompt_input_ids'].shape[1] <= 10, "Sequence should be truncated"
    
    # Test error cases
    try:
        get_dataloader(tokenizer=None)
        assert False, "Should raise ValueError for missing tokenizer"
    except ValueError:
        pass  # Expected
    
    try:
        get_dataloader(data_path=None, data=None, tokenizer=tokenizer)
        assert False, "Should raise ValueError for missing data"
    except ValueError:
        pass  # Expected
    
    print("✓ Edge cases handled correctly")


def run_all_tests():
    """Run all dataset utility tests."""
    print("=" * 50)
    print("Running Dataset Utilities Unit Tests")
    print("=" * 50)
    
    test_functions = [
        test_rl_dataset_creation,
        test_collate_function,
        test_data_loading,
        test_dataloader_creation,
        test_batch_sampler,
        test_dummy_data_creation,
        test_edge_cases
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