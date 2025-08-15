"""
Reproducibility utilities for NSRPO project.
Ensures deterministic behavior across runs.
"""

import random
import numpy as np
import torch
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic mode (may impact performance)
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        # Enable deterministic mode in PyTorch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set deterministic algorithms (PyTorch 1.8+)
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
                # For some operations, we need to set this environment variable
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            except:
                # Some operations might not have deterministic implementations
                pass
    
    return seed


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Get current reproducibility settings and system information.
    
    Returns:
        Dictionary with reproducibility information
    """
    import platform
    import sys
    
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'random_state': random.getstate()[1][0],  # Get first element of random state
        'numpy_random_state': np.random.get_state()[1][0],  # Get first element
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'deterministic_mode': torch.backends.cudnn.deterministic if torch.cuda.is_available() else None,
        'benchmark_mode': torch.backends.cudnn.benchmark if torch.cuda.is_available() else None,
    }
    
    # Add environment variables
    env_vars = ['PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']
    info['environment'] = {var: os.environ.get(var) for var in env_vars}
    
    return info


def save_reproducibility_info(save_path: Union[str, Path], **kwargs):
    """
    Save reproducibility information to a file.
    
    Args:
        save_path: Path to save the information
        **kwargs: Additional information to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    info = get_reproducibility_info()
    info.update(kwargs)
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2, default=str)


def load_reproducibility_info(load_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load reproducibility information from a file.
    
    Args:
        load_path: Path to load the information from
        
    Returns:
        Dictionary with reproducibility information
    """
    with open(load_path, 'r') as f:
        return json.load(f)


def compute_data_hash(data: Any) -> str:
    """
    Compute a hash of data for verification.
    
    Args:
        data: Data to hash (will be converted to string)
        
    Returns:
        SHA256 hash of the data
    """
    if isinstance(data, (list, dict)):
        data_str = json.dumps(data, sort_keys=True)
    elif isinstance(data, torch.Tensor):
        data_str = str(data.cpu().numpy().tolist())
    elif isinstance(data, np.ndarray):
        data_str = str(data.tolist())
    else:
        data_str = str(data)
    
    return hashlib.sha256(data_str.encode()).hexdigest()


def verify_reproducibility(
    results1: Dict[str, Any],
    results2: Dict[str, Any],
    tolerance: float = 1e-6
) -> Dict[str, bool]:
    """
    Verify if two results are reproducible within tolerance.
    
    Args:
        results1: First set of results
        results2: Second set of results
        tolerance: Numerical tolerance for floating point comparisons
        
    Returns:
        Dictionary indicating which fields are reproducible
    """
    verification = {}
    
    for key in results1.keys():
        if key not in results2:
            verification[key] = False
            continue
        
        val1 = results1[key]
        val2 = results2[key]
        
        if isinstance(val1, (int, str, bool)):
            verification[key] = val1 == val2
        elif isinstance(val1, float):
            verification[key] = abs(val1 - val2) < tolerance
        elif isinstance(val1, (list, tuple)):
            if len(val1) != len(val2):
                verification[key] = False
            else:
                verification[key] = all(
                    abs(a - b) < tolerance if isinstance(a, float) else a == b
                    for a, b in zip(val1, val2)
                )
        elif isinstance(val1, dict):
            sub_verification = verify_reproducibility(val1, val2, tolerance)
            verification[key] = all(sub_verification.values())
        elif isinstance(val1, torch.Tensor):
            verification[key] = torch.allclose(val1, val2, atol=tolerance)
        elif isinstance(val1, np.ndarray):
            verification[key] = np.allclose(val1, val2, atol=tolerance)
        else:
            verification[key] = str(val1) == str(val2)
    
    return verification


class ReproducibilityManager:
    """
    Manager class for handling reproducibility across experiments.
    """
    
    def __init__(self, experiment_dir: Union[str, Path], seed: int = 42):
        """
        Initialize reproducibility manager.
        
        Args:
            experiment_dir: Directory for experiment outputs
            seed: Random seed
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.info_file = self.experiment_dir / 'reproducibility_info.json'
        
    def setup(self, deterministic: bool = True) -> int:
        """
        Setup reproducibility for the experiment.
        
        Args:
            deterministic: Whether to enable deterministic mode
            
        Returns:
            The seed used
        """
        seed = set_seed(self.seed, deterministic)
        save_reproducibility_info(self.info_file, seed=seed, deterministic=deterministic)
        return seed
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any], name: str = "checkpoint"):
        """
        Save a checkpoint with reproducibility information.
        
        Args:
            checkpoint_data: Data to save in checkpoint
            name: Name for the checkpoint
        """
        checkpoint_path = self.experiment_dir / f"{name}.pt"
        
        # Add reproducibility info to checkpoint
        checkpoint_data['reproducibility_info'] = get_reproducibility_info()
        checkpoint_data['data_hash'] = compute_data_hash(checkpoint_data.get('model_state_dict', {}))
        
        torch.save(checkpoint_data, checkpoint_path)
        
    def load_checkpoint(self, name: str = "checkpoint") -> Dict[str, Any]:
        """
        Load a checkpoint and verify reproducibility information.
        
        Args:
            name: Name of the checkpoint
            
        Returns:
            Checkpoint data
        """
        checkpoint_path = self.experiment_dir / f"{name}.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Verify data integrity
        if 'data_hash' in checkpoint:
            current_hash = compute_data_hash(checkpoint.get('model_state_dict', {}))
            if current_hash != checkpoint['data_hash']:
                print("Warning: Checkpoint data hash mismatch. Data may have been modified.")
        
        return checkpoint
    
    def log_results(self, results: Dict[str, Any], name: str = "results"):
        """
        Log results with reproducibility information.
        
        Args:
            results: Results to log
            name: Name for the results file
        """
        results_file = self.experiment_dir / f"{name}.json"
        
        # Add reproducibility info
        results['reproducibility_info'] = get_reproducibility_info()
        results['results_hash'] = compute_data_hash(results)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def verify_experiment(self, other_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Verify reproducibility against another experiment.
        
        Args:
            other_dir: Directory of the other experiment
            
        Returns:
            Verification results
        """
        other_dir = Path(other_dir)
        
        # Load results from both experiments
        results1_file = self.experiment_dir / "results.json"
        results2_file = other_dir / "results.json"
        
        if results1_file.exists() and results2_file.exists():
            with open(results1_file) as f:
                results1 = json.load(f)
            with open(results2_file) as f:
                results2 = json.load(f)
            
            return verify_reproducibility(results1, results2)
        else:
            return {"error": "Results files not found in one or both experiments"}


# Utility functions for common use cases

def make_deterministic():
    """Enable full deterministic mode for maximum reproducibility."""
    set_seed(42, deterministic=True)
    print("Deterministic mode enabled with seed 42")


def get_experiment_hash(config: Dict[str, Any]) -> str:
    """
    Generate a unique hash for an experiment configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Hash string
    """
    return compute_data_hash(config)[:12]  # Use first 12 characters


if __name__ == "__main__":
    # Test reproducibility utilities
    print("Testing reproducibility utilities...")
    
    # Set seed
    seed = set_seed(42)
    print(f"Seed set to: {seed}")
    
    # Get reproducibility info
    info = get_reproducibility_info()
    print(f"System info: Python {info['python_version'].split()[0]}, "
          f"PyTorch {info['torch_version']}, "
          f"CUDA: {info['cuda_available']}")
    
    # Test data hashing
    test_data = {"model": "test", "accuracy": 0.95}
    hash_value = compute_data_hash(test_data)
    print(f"Data hash: {hash_value}")
    
    # Test reproducibility verification
    results1 = {"loss": 0.5, "accuracy": 0.85}
    results2 = {"loss": 0.5, "accuracy": 0.85}
    verification = verify_reproducibility(results1, results2)
    print(f"Reproducibility verification: {verification}")
    
    print("Reproducibility utilities test completed!")