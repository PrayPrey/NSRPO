"""
HuggingFace Dataset Integration for NSRPO Training
Supports loading real datasets from HuggingFace Hub
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from torch.utils.data import Dataset, DataLoader


def load_huggingface_dataset(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    num_samples: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load dataset from HuggingFace.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        dataset_config: Configuration of the dataset
        split: Which split to load (train/validation/test)
        num_samples: Limit number of samples (None for all)
        cache_dir: Directory to cache the dataset
        
    Returns:
        List of processed data items
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets"])
        from datasets import load_dataset
    
    # Load the dataset
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir)
    else:
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    
    # Process based on dataset type
    processed_data = []
    
    if dataset_name == "wikitext":
        # Process WikiText for language modeling
        for idx, item in enumerate(dataset):
            if num_samples and idx >= num_samples:
                break
            
            text = item.get('text', '')
            if text.strip():  # Skip empty lines
                # Split into chunks for prompt-response pairs
                sentences = text.split('. ')
                if len(sentences) > 1:
                    prompt = sentences[0] + '.'
                    response = '. '.join(sentences[1:])
                else:
                    # For single sentences, use partial text
                    words = text.split()
                    if len(words) > 5:
                        prompt = ' '.join(words[:len(words)//2])
                        response = ' '.join(words[len(words)//2:])
                    else:
                        continue
                
                processed_data.append({
                    'prompt': prompt.strip(),
                    'response': response.strip(),
                    'reward': 0.0  # Neutral reward for unsupervised data
                })
    
    elif dataset_name == "imdb":
        # Process IMDB for sentiment-aware responses
        for idx, item in enumerate(dataset):
            if num_samples and idx >= num_samples:
                break
            
            text = item.get('text', '')
            label = item.get('label', 0)
            
            # Create prompt-response pairs from reviews
            if len(text) > 100:
                prompt = "Review the following movie: " + text[:100] + "..."
                response = text[100:300] if len(text) > 300 else text[100:]
                # Positive reviews get higher rewards
                reward = 0.5 if label == 1 else -0.5
                
                processed_data.append({
                    'prompt': prompt,
                    'response': response,
                    'reward': reward,
                    'sentiment': 'positive' if label == 1 else 'negative'
                })
    
    elif dataset_name == "squad":
        # Process SQuAD for question-answering
        for idx, item in enumerate(dataset):
            if num_samples and idx >= num_samples:
                break
            
            question = item.get('question', '')
            context = item.get('context', '')
            answers = item.get('answers', {})
            
            if question and context and answers:
                prompt = f"Question: {question}\nContext: {context[:200]}..."
                response = answers.get('text', [''])[0] if answers.get('text') else ''
                
                if response:
                    processed_data.append({
                        'prompt': prompt,
                        'response': response,
                        'reward': 1.0  # Correct answers get high reward
                    })
    
    elif dataset_name == "c4":
        # Process C4 for general language modeling
        for idx, item in enumerate(dataset):
            if num_samples and idx >= num_samples:
                break
            
            text = item.get('text', '')
            if len(text) > 100:
                # Split into prompt-response
                sentences = text.split('. ')
                if len(sentences) > 2:
                    prompt = '. '.join(sentences[:len(sentences)//2]) + '.'
                    response = '. '.join(sentences[len(sentences)//2:])
                    
                    processed_data.append({
                        'prompt': prompt[:512],  # Limit length
                        'response': response[:512],
                        'reward': 0.0
                    })
    
    else:
        # Generic processing for other datasets
        for idx, item in enumerate(dataset):
            if num_samples and idx >= num_samples:
                break
            
            # Try to find text field
            text_field = None
            for field in ['text', 'sentence', 'content', 'input', 'prompt']:
                if field in item:
                    text_field = field
                    break
            
            if text_field:
                text = item[text_field]
                if isinstance(text, str) and len(text) > 20:
                    # Simple split for prompt-response
                    mid = len(text) // 2
                    prompt = text[:mid]
                    response = text[mid:]
                    
                    processed_data.append({
                        'prompt': prompt,
                        'response': response,
                        'reward': 0.0
                    })
    
    return processed_data


def get_hf_dataloader(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    tokenizer = None,
    batch_size: int = 8,
    max_length: int = 512,
    num_samples: Optional[int] = None,
    shuffle: bool = True,
    cache_dir: Optional[str] = None
) -> DataLoader:
    """
    Create DataLoader from HuggingFace dataset.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        dataset_config: Configuration name
        split: Dataset split to use
        tokenizer: Tokenizer for encoding
        batch_size: Batch size
        max_length: Maximum sequence length
        num_samples: Limit number of samples
        shuffle: Whether to shuffle
        cache_dir: Cache directory
        
    Returns:
        DataLoader instance
    """
    # Import the original dataset utilities
    from dataset import RLDataset, rl_collate_fn
    
    # Load HuggingFace data
    data = load_huggingface_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        num_samples=num_samples,
        cache_dir=cache_dir
    )
    
    if not data:
        raise ValueError(f"No data loaded from {dataset_name}")
    
    # Create dataset
    dataset = RLDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
        include_rewards=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: rl_collate_fn(
            batch, 
            tokenizer=tokenizer,
            max_length=max_length,
            padding='longest'
        ),
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False
    )
    
    return dataloader


# Predefined dataset configurations for easy use
DATASET_CONFIGS = {
    'wikitext-small': {
        'dataset_name': 'wikitext',
        'dataset_config': 'wikitext-2-raw-v1',
        'description': 'Small WikiText dataset for language modeling'
    },
    'wikitext-large': {
        'dataset_name': 'wikitext',
        'dataset_config': 'wikitext-103-raw-v1',
        'description': 'Large WikiText dataset for language modeling'
    },
    'imdb': {
        'dataset_name': 'imdb',
        'dataset_config': None,
        'description': 'IMDB movie reviews for sentiment analysis'
    },
    'squad': {
        'dataset_name': 'squad',
        'dataset_config': None,
        'description': 'Stanford Question Answering Dataset'
    },
    'c4-small': {
        'dataset_name': 'c4',
        'dataset_config': 'en',
        'description': 'Colossal Clean Crawled Corpus (small sample)'
    },
    'glue-cola': {
        'dataset_name': 'glue',
        'dataset_config': 'cola',
        'description': 'GLUE CoLA dataset for linguistic acceptability'
    },
    'glue-sst2': {
        'dataset_name': 'glue', 
        'dataset_config': 'sst2',
        'description': 'GLUE SST-2 dataset for sentiment analysis'
    }
}


def list_available_datasets():
    """List all available pre-configured datasets."""
    print("Available HuggingFace Datasets:")
    print("-" * 50)
    for key, config in DATASET_CONFIGS.items():
        print(f"{key:15} - {config['description']}")
    print("-" * 50)
    print("\nUsage example:")
    print("  python train.py --train_data_path hf:wikitext-small")


if __name__ == "__main__":
    # Test the module
    print("Testing HuggingFace Dataset Integration\n")
    
    # List available datasets
    list_available_datasets()
    
    # Test loading a small sample
    print("\nTesting WikiText loading...")
    try:
        data = load_huggingface_dataset(
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            split="train",
            num_samples=5
        )
        
        print(f"Loaded {len(data)} samples")
        if data:
            print(f"Sample item:")
            print(f"  Prompt: {data[0]['prompt'][:100]}...")
            print(f"  Response: {data[0]['response'][:100]}...")
            print(f"  Reward: {data[0]['reward']}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("This is normal if you're offline or don't have the datasets library installed.")