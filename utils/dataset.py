"""
Dataset and DataLoader Utilities for GRPO Training
Task 6: Implement Dataset and DataLoader Utilities
"""

import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import default_data_collator
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


class RLDataset(Dataset):
    """
    Dataset for Reinforcement Learning with Human Feedback (RLHF) / GRPO training.
    """
    
    def __init__(
        self, 
        data: List[Dict[str, Any]], 
        tokenizer,
        max_length: int = 512,
        include_rewards: bool = True
    ):
        """
        Initialize RLDataset.
        
        Args:
            data: List of dictionaries containing 'prompt', 'response', and optionally 'reward'
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            include_rewards: Whether to include rewards in the dataset
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_rewards = include_rewards
        
        # Set padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing the processed item
        """
        item = self.data[idx]
        
        # Basic item structure
        processed_item = {
            'prompt': item.get('prompt', ''),
            'response': item.get('response', ''),
        }
        
        # Add reward if available and requested
        if self.include_rewards and 'reward' in item:
            processed_item['reward'] = float(item['reward'])
        elif self.include_rewards:
            # Default reward if not provided
            processed_item['reward'] = 0.0
        
        # Add any additional fields from the original item
        for key in item:
            if key not in ['prompt', 'response', 'reward']:
                processed_item[key] = item[key]
        
        return processed_item


def rl_collate_fn(
    batch: List[Dict[str, Any]], 
    tokenizer,
    max_length: int = 512,
    padding: str = 'longest',
    return_tensors: str = 'pt'
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for RL datasets.
    
    Args:
        batch: List of items from the dataset
        tokenizer: Tokenizer for encoding text
        max_length: Maximum sequence length
        padding: Padding strategy ('longest', 'max_length')
        return_tensors: Type of tensors to return
        
    Returns:
        Dictionary containing batched and tokenized data
    """
    # Extract fields from batch
    prompts = [item['prompt'] for item in batch]
    responses = [item['response'] for item in batch]
    
    # Tokenize prompts
    prompt_inputs = tokenizer(
        prompts,
        padding=padding,
        truncation=True,
        max_length=max_length,
        return_tensors=return_tensors
    )
    
    # Tokenize responses
    response_inputs = tokenizer(
        responses,
        padding=padding,
        truncation=True,
        max_length=max_length,
        return_tensors=return_tensors
    )
    
    # Combine prompt and response for full sequence
    combined_texts = [p + r for p, r in zip(prompts, responses)]
    combined_inputs = tokenizer(
        combined_texts,
        padding=padding,
        truncation=True,
        max_length=max_length * 2,  # Allow longer combined sequence
        return_tensors=return_tensors
    )
    
    # Prepare output dictionary
    collated = {
        "prompt_input_ids": prompt_inputs.input_ids,
        "prompt_attention_mask": prompt_inputs.attention_mask,
        "response_input_ids": response_inputs.input_ids,
        "response_attention_mask": response_inputs.attention_mask,
        "input_ids": combined_inputs.input_ids,
        "attention_mask": combined_inputs.attention_mask,
    }
    
    # Add rewards if present
    if 'reward' in batch[0]:
        rewards = torch.tensor([item['reward'] for item in batch], dtype=torch.float)
        collated["rewards"] = rewards
    
    # Add any additional tensor fields
    for key in batch[0]:
        if key not in ['prompt', 'response', 'reward'] and isinstance(batch[0][key], (int, float)):
            values = torch.tensor([item[key] for item in batch])
            collated[key] = values
    
    return collated


def load_data(
    data_path: Union[str, Path],
    data_format: str = 'auto'
) -> List[Dict[str, Any]]:
    """
    Load data from file.
    
    Args:
        data_path: Path to the data file
        data_format: Format of the data ('json', 'jsonl', 'auto')
        
    Returns:
        List of data items
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Auto-detect format from extension
    if data_format == 'auto':
        if data_path.suffix == '.jsonl':
            data_format = 'jsonl'
        elif data_path.suffix == '.json':
            data_format = 'json'
        else:
            # Try to infer from content
            with open(data_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith('[') or first_line.startswith('{'):
                    data_format = 'json'
                else:
                    data_format = 'jsonl'
    
    data = []
    
    if data_format == 'jsonl':
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    elif data_format == 'json':
        with open(data_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                data = loaded
            else:
                data = [loaded]
    else:
        raise ValueError(f"Unsupported data format: {data_format}")
    
    return data


def get_dataloader(
    data_path: Optional[Union[str, Path]] = None,
    data: Optional[List[Dict[str, Any]]] = None,
    tokenizer = None,
    batch_size: int = 8,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    include_rewards: bool = True,
    padding: str = 'longest',
    num_samples: Optional[int] = None
) -> DataLoader:
    """
    Create a DataLoader for RL training.
    
    Args:
        data_path: Path to data file or HuggingFace dataset (e.g., 'hf:wikitext-small')
        data: Direct data list (if data_path is not provided)
        tokenizer: Tokenizer for encoding text
        batch_size: Batch size for training
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for CUDA
        include_rewards: Whether to include rewards in the dataset
        padding: Padding strategy
        num_samples: Limit number of samples (for HuggingFace datasets)
        
    Returns:
        DataLoader instance
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    
    # Load data if path is provided
    if data is None:
        if data_path is None:
            raise ValueError("Either data_path or data must be provided")
        
        # Check if it's a HuggingFace dataset
        if isinstance(data_path, str) and data_path.startswith('hf:'):
            # Import HuggingFace dataset loader
            from .hf_dataset import load_huggingface_dataset, DATASET_CONFIGS
            
            # Parse dataset name
            dataset_key = data_path[3:]  # Remove 'hf:' prefix
            
            if dataset_key in DATASET_CONFIGS:
                config = DATASET_CONFIGS[dataset_key]
                data = load_huggingface_dataset(
                    dataset_name=config['dataset_name'],
                    dataset_config=config.get('dataset_config'),
                    split='train' if 'train' in str(data_path) else 'validation',
                    num_samples=num_samples
                )
            else:
                # Try direct dataset name
                parts = dataset_key.split('/')
                dataset_name = parts[0]
                dataset_config = parts[1] if len(parts) > 1 else None
                data = load_huggingface_dataset(
                    dataset_name=dataset_name,
                    dataset_config=dataset_config,
                    split='train',
                    num_samples=num_samples
                )
        else:
            data = load_data(data_path)
    
    # Create dataset
    dataset = RLDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
        include_rewards=include_rewards
    )
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: rl_collate_fn(
            batch, 
            tokenizer=tokenizer,
            max_length=max_length,
            padding=padding
        ),
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader


def create_dummy_data(
    num_samples: int = 100,
    save_path: Optional[Union[str, Path]] = None,
    data_type: str = 'preference'  # 'preference', 'instruction', 'conversation'
) -> List[Dict[str, Any]]:
    """
    Create synthetic data for testing.
    
    Args:
        num_samples: Number of samples to create
        save_path: Optional path to save the data
        data_type: Type of data to generate
        
    Returns:
        List of synthetic data items
    """
    import random
    
    if data_type == 'preference':
        return generate_synthetic_preference_data(num_samples, save_path)
    elif data_type == 'instruction':
        return generate_synthetic_instruction_data(num_samples, save_path)
    elif data_type == 'conversation':
        return generate_synthetic_conversation_data(num_samples, save_path)
    else:
        # Default simple data
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing.",
            "How do I bake a cake?",
            "What is machine learning?",
            "Write a poem about nature.",
            "Solve this math problem:",
            "Translate this sentence:",
            "Summarize this article:",
            "What are the benefits of exercise?",
            "How does photosynthesis work?"
        ]
        
        responses = [
            "The capital of France is Paris.",
            "Quantum computing uses quantum bits or qubits.",
            "To bake a cake, you need flour, eggs, and sugar.",
            "Machine learning is a subset of artificial intelligence.",
            "In the forest deep and green, nature's beauty can be seen.",
            "The answer is 42.",
            "La traduction est ici.",
            "This article discusses important topics.",
            "Exercise improves physical and mental health.",
            "Photosynthesis converts light into chemical energy."
        ]
        
        data = []
        for i in range(num_samples):
            item = {
                'prompt': random.choice(prompts),
                'response': random.choice(responses),
                'reward': random.uniform(-1.0, 1.0)
            }
            data.append(item)
        
        # Save if path is provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if save_path.suffix == '.jsonl':
                with open(save_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(json.dumps(item) + '\n')
            else:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
        
        return data


class BatchSampler:
    """
    Custom batch sampler for specific sampling strategies.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize batch sampler.
        
        Args:
            dataset: Dataset to sample from
            batch_size: Size of each batch
            shuffle: Whether to shuffle indices
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = self.indices
        
        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def generate_synthetic_preference_data(
    num_samples: int = 100,
    save_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Generate synthetic preference dataset for RLHF/GRPO training.
    """
    import random
    
    # Define template prompts for preference learning
    prompt_templates = [
        "Explain {} in simple terms.",
        "What are the benefits of {}?",
        "How does {} work?",
        "Compare {} and {}.",
        "Write a short story about {}.",
        "What is the importance of {} in {}?",
        "Describe the process of {}.",
        "What are the main features of {}?",
        "How can I improve my {} skills?",
        "What should I know about {}?"
    ]
    
    topics = [
        "machine learning", "quantum computing", "climate change",
        "renewable energy", "artificial intelligence", "blockchain",
        "gene editing", "space exploration", "virtual reality",
        "cybersecurity", "data science", "robotics"
    ]
    
    response_qualities = [
        {"quality": "good", "reward": random.uniform(0.5, 1.0)},
        {"quality": "medium", "reward": random.uniform(-0.2, 0.5)},
        {"quality": "poor", "reward": random.uniform(-1.0, -0.2)}
    ]
    
    data = []
    for i in range(num_samples):
        # Generate prompt
        template = random.choice(prompt_templates)
        if "{}" in template and template.count("{}") == 2:
            prompt = template.format(
                random.choice(topics),
                random.choice(topics)
            )
        elif "{}" in template and template.count("{}") == 1:
            prompt = template.format(random.choice(topics))
        else:
            prompt = template
        
        # Generate response with quality indicator
        quality = random.choice(response_qualities)
        
        if quality["quality"] == "good":
            response = f"Here's a comprehensive explanation: [Detailed, accurate response about {prompt}]"
        elif quality["quality"] == "medium":
            response = f"The answer is: [Partially correct response about {prompt}]"
        else:
            response = f"I think: [Brief or incorrect response about {prompt}]"
        
        item = {
            'prompt': prompt,
            'response': response,
            'reward': quality["reward"],
            'quality': quality["quality"]
        }
        data.append(item)
    
    # Save if path is provided
    if save_path:
        _save_data(data, save_path)
    
    return data


def generate_synthetic_instruction_data(
    num_samples: int = 100,
    save_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Generate synthetic instruction-following dataset.
    """
    import random
    
    instruction_types = [
        {"type": "code", "prefix": "Write Python code to"},
        {"type": "explain", "prefix": "Explain how to"},
        {"type": "list", "prefix": "List 5 ways to"},
        {"type": "compare", "prefix": "Compare and contrast"},
        {"type": "summarize", "prefix": "Summarize the key points about"},
        {"type": "create", "prefix": "Create a plan for"},
        {"type": "analyze", "prefix": "Analyze the impact of"}
    ]
    
    tasks = [
        "sort a list", "implement a binary search", "train a neural network",
        "optimize database queries", "handle exceptions", "manage memory",
        "improve code performance", "debug complex issues", "design APIs",
        "secure applications", "scale systems", "test software"
    ]
    
    data = []
    for i in range(num_samples):
        instruction = random.choice(instruction_types)
        task = random.choice(tasks)
        
        prompt = f"{instruction['prefix']} {task}"
        
        # Generate response based on instruction type
        if instruction["type"] == "code":
            response = f"```python\n# Solution for {task}\ndef solution():\n    pass  # Implementation here\n```"
        elif instruction["type"] == "list":
            response = f"1. First approach to {task}\n2. Second method\n3. Third technique\n4. Fourth strategy\n5. Fifth option"
        else:
            response = f"Here's the {instruction['type']} for {task}: [Detailed response]"
        
        # Assign reward based on response completeness
        reward = random.uniform(-0.5, 1.0) if random.random() > 0.3 else random.uniform(-1.0, -0.5)
        
        item = {
            'prompt': prompt,
            'response': response,
            'reward': reward,
            'instruction_type': instruction["type"]
        }
        data.append(item)
    
    if save_path:
        _save_data(data, save_path)
    
    return data


def generate_synthetic_conversation_data(
    num_samples: int = 100,
    save_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Generate synthetic conversational dataset.
    """
    import random
    
    conversation_starters = [
        "Tell me about", "What do you think about", "Can you help me understand",
        "I'm curious about", "What's your opinion on", "How would you approach",
        "What's the best way to", "Could you explain", "I need advice on"
    ]
    
    topics = [
        "learning programming", "staying motivated", "work-life balance",
        "career development", "problem solving", "creativity",
        "team collaboration", "time management", "decision making"
    ]
    
    response_styles = [
        {"style": "helpful", "reward_range": (0.5, 1.0)},
        {"style": "brief", "reward_range": (0.0, 0.5)},
        {"style": "off-topic", "reward_range": (-1.0, -0.3)}
    ]
    
    data = []
    for i in range(num_samples):
        starter = random.choice(conversation_starters)
        topic = random.choice(topics)
        style = random.choice(response_styles)
        
        prompt = f"{starter} {topic}."
        
        if style["style"] == "helpful":
            response = f"I'd be happy to help with {topic}. Here are some key insights: [Detailed helpful response]"
        elif style["style"] == "brief":
            response = f"Regarding {topic}: [Short response]"
        else:
            response = "That's interesting. [Unrelated or unhelpful response]"
        
        reward = random.uniform(*style["reward_range"])
        
        item = {
            'prompt': prompt,
            'response': response,
            'reward': reward,
            'style': style["style"]
        }
        data.append(item)
    
    if save_path:
        _save_data(data, save_path)
    
    return data


def _save_data(data: List[Dict[str, Any]], save_path: Union[str, Path]):
    """Helper function to save data."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.suffix == '.jsonl':
        with open(save_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    else:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


def generate_synthetic_train_data(
    num_samples: int = 100,
    save_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Generate synthetic training data optimized for GRPO/NSRPO training.
    Matches expected format for preference learning.
    """
    return generate_synthetic_preference_data(num_samples, save_path)


def generate_synthetic_eval_data(
    num_samples: int = 100,
    save_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Generate synthetic evaluation data with known ground truth.
    Useful for testing metrics and evaluation pipelines.
    """
    import random
    
    # Create evaluation data with controlled properties
    data = []
    
    # Generate samples with varying quality levels
    quality_distribution = [
        {"quality": "excellent", "reward": 0.9, "count": int(num_samples * 0.2)},
        {"quality": "good", "reward": 0.5, "count": int(num_samples * 0.3)},
        {"quality": "average", "reward": 0.0, "count": int(num_samples * 0.3)},
        {"quality": "poor", "reward": -0.5, "count": int(num_samples * 0.2)}
    ]
    
    sample_id = 0
    for quality_level in quality_distribution:
        for _ in range(quality_level["count"]):
            prompt = f"Test prompt {sample_id}: Evaluate this response quality."
            response = f"Response with {quality_level['quality']} quality: [Content matching quality level]"
            
            item = {
                'prompt': prompt,
                'response': response,
                'reward': quality_level["reward"] + random.uniform(-0.1, 0.1),  # Add small noise
                'ground_truth_quality': quality_level["quality"],
                'sample_id': sample_id
            }
            data.append(item)
            sample_id += 1
    
    # Ensure we have exactly num_samples
    while len(data) < num_samples:
        item = {
            'prompt': f"Test prompt {len(data)}: Additional sample.",
            'response': "Additional response for testing.",
            'reward': random.uniform(-1.0, 1.0),
            'ground_truth_quality': 'random',
            'sample_id': len(data)
        }
        data.append(item)
    
    data = data[:num_samples]  # Trim if necessary
    
    if save_path:
        _save_data(data, save_path)
    
    return data


if __name__ == "__main__":
    # Test the module
    print("Dataset Utilities Module Loaded Successfully")
    
    # Create and test dummy data
    dummy_data = create_dummy_data(10)
    print(f"Created {len(dummy_data)} dummy samples")
    print(f"Sample item: {dummy_data[0]}")
    
    # Test with a mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_token = '[PAD]'
            self.eos_token = '[EOS]'
        
        def __call__(self, texts, **kwargs):
            # Simple mock tokenization
            if isinstance(texts, str):
                texts = [texts]
            
            input_ids = []
            attention_mask = []
            
            for text in texts:
                tokens = text.split()[:10]  # Simple word tokenization
                ids = list(range(len(tokens)))
                mask = [1] * len(ids)
                
                # Pad to max length
                while len(ids) < 10:
                    ids.append(0)
                    mask.append(0)
                
                input_ids.append(ids)
                attention_mask.append(mask)
            
            return {
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)
            }
    
    # Test dataset creation
    mock_tokenizer = MockTokenizer()
    dataset = RLDataset(dummy_data, mock_tokenizer, max_length=512)
    print(f"Dataset size: {len(dataset)}")
    
    # Test dataloader creation
    dataloader = get_dataloader(
        data=dummy_data,
        tokenizer=mock_tokenizer,
        batch_size=2,
        shuffle=False
    )
    
    # Test iteration
    for i, batch in enumerate(dataloader):
        if i == 0:
            print(f"Batch keys: {batch.keys()}")
            print(f"Batch shapes: input_ids={batch['input_ids'].shape}, rewards={batch['rewards'].shape}")
        if i >= 1:  # Only show first 2 batches
            break