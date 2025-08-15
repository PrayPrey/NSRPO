"""
NSRPO Experiment Configuration System
Task 10: Implement Experiment Configuration - Hyperparameter and experiment management

Comprehensive configuration management system for reproducible NSRPO experiments.
Supports hyperparameter search, experiment tracking, and configuration validation.
"""

import json
import os
import yaml
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import itertools
import hashlib
from datetime import datetime


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    base_model_path: str = "microsoft/DialoGPT-medium"
    use_null_decoder: bool = True
    extract_null_basis: bool = False
    null_basis_path: Optional[str] = None
    
    # Null decoder architecture
    decoder_layers: int = 3
    decoder_heads: int = 8
    decoder_dropout: float = 0.1
    
    # Loss weighting (as specified in PRD)
    alpha_1: float = 0.1  # CE loss weight
    alpha_2: float = 0.1  # Cosine similarity loss weight
    alpha_3: float = 0.05  # Norm preservation loss weight
    
    # Advanced model options
    freeze_base_model: bool = False
    use_reconstruction_metrics: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.use_null_decoder and not self.null_basis_path and not self.extract_null_basis:
            raise ValueError("null_basis_path required when use_null_decoder=True or set extract_null_basis=True")


@dataclass 
class TrainingConfig:
    """Configuration for training parameters."""
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 3
    max_length: int = 512
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Scheduler options
    scheduler_type: str = "linear"  # linear, cosine, constant
    
    # Mixed precision and optimization
    fp16: bool = False
    dataloader_num_workers: int = 4
    
    # Checkpointing and logging
    save_every: int = 1
    eval_every: int = 1
    log_every: int = 100
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.scheduler_type not in ["linear", "cosine", "constant"]:
            raise ValueError(f"Invalid scheduler_type: {self.scheduler_type}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass
class DataConfig:
    """Configuration for data parameters."""
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    num_dummy_samples: int = 1000
    
    # Data processing options
    shuffle_train: bool = True
    preprocessing_num_workers: int = 4
    cache_preprocessed: bool = True
    
    # Data augmentation (future extension)
    use_data_augmentation: bool = False
    augmentation_prob: float = 0.1


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters."""
    eval_batch_size: int = 16
    max_eval_batches: Optional[int] = None
    
    # Metrics to compute
    compute_accuracy: bool = True
    compute_perplexity: bool = True
    compute_kl_divergence: bool = False
    compute_training_efficiency: bool = True
    
    # Text generation evaluation
    include_generation: bool = False
    generation_prompts: List[str] = field(default_factory=lambda: [
        "The future of AI is",
        "In the year 2030,", 
        "Climate change will"
    ])
    generation_max_length: int = 100
    generation_temperature: float = 1.0
    generation_top_p: float = 0.9
    
    # Baseline comparison
    baseline_model_path: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Main experiment configuration combining all components."""
    experiment_name: str = "nsrpo_experiment"
    experiment_description: str = ""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Environment settings
    seed: int = 42
    device: str = "auto"  # auto, cuda, cpu
    output_dir: str = "./experiments"
    log_level: str = "INFO"
    
    # Experiment tracking
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Create experiment hash for unique identification
        config_dict = asdict(self)
        config_str = json.dumps(config_dict, sort_keys=True)
        self.experiment_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Set up output directory with timestamp and hash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{self.experiment_name}_{timestamp}_{self.experiment_hash}"
        self.full_output_dir = os.path.join(self.output_dir, self.experiment_id)
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save configuration to file.
        
        Args:
            path: Optional path to save to (defaults to output directory)
            
        Returns:
            Path where configuration was saved
        """
        if path is None:
            os.makedirs(self.full_output_dir, exist_ok=True)
            path = os.path.join(self.full_output_dir, "config.yaml")
        
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, indent=2)
        
        return path
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Reconstruct nested dataclasses
        config = cls()
        
        # Update with loaded values
        for key, value in config_dict.items():
            if key == 'model' and isinstance(value, dict):
                config.model = ModelConfig(**value)
            elif key == 'training' and isinstance(value, dict):
                config.training = TrainingConfig(**value)
            elif key == 'data' and isinstance(value, dict):
                config.data = DataConfig(**value)
            elif key == 'evaluation' and isinstance(value, dict):
                config.evaluation = EvaluationConfig(**value)
            elif hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def validate(self) -> bool:
        """Validate the entire configuration."""
        try:
            # Validate sub-configurations (handled in __post_init__)
            self.model.__post_init__()
            self.training.__post_init__()
            
            # Additional cross-configuration validation
            if self.evaluation.baseline_model_path and not self.evaluation.compute_kl_divergence:
                print("Warning: baseline_model_path specified but compute_kl_divergence=False")
            
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


class HyperparameterSearch:
    """
    Hyperparameter search configuration and execution.
    
    Supports grid search, random search, and custom parameter sweeps.
    """
    
    def __init__(self, base_config: ExperimentConfig):
        """Initialize with base configuration."""
        self.base_config = base_config
        self.search_space = {}
        self.search_results = []
    
    def add_parameter_range(
        self, 
        parameter_path: str, 
        values: List[Any],
        search_type: str = "grid"
    ):
        """
        Add parameter to search space.
        
        Args:
            parameter_path: Dot-notation path to parameter (e.g., "model.alpha_1")
            values: List of values to search over
            search_type: Type of search ("grid", "random")
        """
        self.search_space[parameter_path] = {
            'values': values,
            'type': search_type
        }
    
    def generate_configurations(
        self, 
        max_configs: Optional[int] = None,
        random_seed: int = 42
    ) -> List[ExperimentConfig]:
        """
        Generate configurations for hyperparameter search.
        
        Args:
            max_configs: Maximum number of configurations to generate
            random_seed: Random seed for random search
            
        Returns:
            List of experiment configurations
        """
        import random
        random.seed(random_seed)
        
        # Separate grid and random parameters
        grid_params = {k: v for k, v in self.search_space.items() if v['type'] == 'grid'}
        random_params = {k: v for k, v in self.search_space.items() if v['type'] == 'random'}
        
        configs = []
        
        if grid_params:
            # Generate all combinations for grid search
            param_names = list(grid_params.keys())
            param_values = [grid_params[name]['values'] for name in param_names]
            
            for combination in itertools.product(*param_values):
                config = self._create_config_variant(
                    dict(zip(param_names, combination)),
                    random_params,
                    random
                )
                configs.append(config)
        else:
            # Pure random search
            num_configs = max_configs or 10
            for _ in range(num_configs):
                config = self._create_config_variant({}, random_params, random)
                configs.append(config)
        
        # Limit to max_configs if specified
        if max_configs and len(configs) > max_configs:
            configs = random.sample(configs, max_configs)
        
        return configs
    
    def _create_config_variant(
        self, 
        grid_assignment: Dict[str, Any],
        random_params: Dict[str, Dict],
        random_gen
    ) -> ExperimentConfig:
        """Create configuration variant with parameter assignments."""
        # Deep copy base config
        config_dict = asdict(self.base_config)
        
        # Apply grid assignments
        for param_path, value in grid_assignment.items():
            self._set_nested_value(config_dict, param_path, value)
        
        # Apply random assignments
        for param_path, param_info in random_params.items():
            value = random_gen.choice(param_info['values'])
            self._set_nested_value(config_dict, param_path, value)
        
        # Recreate config object
        config = ExperimentConfig()
        
        # Update nested structures
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'evaluation' in config_dict:
            config.evaluation = EvaluationConfig(**config_dict['evaluation'])
        
        # Update top-level attributes
        for key, value in config_dict.items():
            if key not in ['model', 'training', 'data', 'evaluation'] and hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def _set_nested_value(self, config_dict: Dict[str, Any], path: str, value: Any):
        """Set value in nested dictionary using dot notation path."""
        keys = path.split('.')
        current = config_dict
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value


class ConfigurationValidator:
    """Validate experiment configurations for common issues."""
    
    @staticmethod
    def validate_paths(config: ExperimentConfig) -> List[str]:
        """Validate file paths in configuration."""
        issues = []
        
        # Check model paths
        if config.model.base_model_path and not config.model.base_model_path.startswith(('http', 'huggingface')):
            if not os.path.exists(config.model.base_model_path):
                issues.append(f"Base model path does not exist: {config.model.base_model_path}")
        
        if config.model.null_basis_path and not os.path.exists(config.model.null_basis_path):
            issues.append(f"Null basis path does not exist: {config.model.null_basis_path}")
        
        # Check data paths
        if config.data.train_data_path and not os.path.exists(config.data.train_data_path):
            issues.append(f"Training data path does not exist: {config.data.train_data_path}")
        
        if config.data.eval_data_path and not os.path.exists(config.data.eval_data_path):
            issues.append(f"Evaluation data path does not exist: {config.data.eval_data_path}")
        
        return issues
    
    @staticmethod
    def validate_resources(config: ExperimentConfig) -> List[str]:
        """Validate resource requirements and constraints."""
        issues = []
        
        # Check batch size constraints
        if config.training.batch_size * config.training.gradient_accumulation_steps > 128:
            issues.append("Effective batch size very large, may cause OOM")
        
        # Check sequence length
        if config.training.max_length > 2048:
            issues.append("Very long sequences may cause memory issues")
        
        # Check epoch vs save frequency
        if config.training.save_every > config.training.num_epochs:
            issues.append("save_every > num_epochs, no checkpoints will be saved")
        
        return issues
    
    @staticmethod
    def validate_hyperparameters(config: ExperimentConfig) -> List[str]:
        """Validate hyperparameter ranges and combinations."""
        issues = []
        
        # Check learning rate
        if config.training.learning_rate > 1e-3:
            issues.append("Learning rate may be too high")
        
        # Check loss weights
        total_weight = config.model.alpha_1 + config.model.alpha_2 + config.model.alpha_3
        if total_weight > 1.0:
            issues.append(f"Total loss weight ({total_weight:.3f}) > 1.0, may destabilize training")
        
        # Check warmup ratio
        if config.training.warmup_ratio > 0.3:
            issues.append("Warmup ratio > 0.3 may be too high")
        
        return issues


def create_ablation_configs(
    base_config: ExperimentConfig,
    experiment_name_prefix: str = "ablation"
) -> List[ExperimentConfig]:
    """
    Create configurations for ablation study.
    
    Args:
        base_config: Base experiment configuration
        experiment_name_prefix: Prefix for ablation experiment names
        
    Returns:
        List of ablation study configurations
    """
    configs = []
    
    # Baseline: No null decoder
    baseline_config = ExperimentConfig(**asdict(base_config))
    baseline_config.model.use_null_decoder = False
    baseline_config.experiment_name = f"{experiment_name_prefix}_baseline"
    baseline_config.experiment_description = "Baseline GRPO without null decoder"
    baseline_config.tags.append("ablation_baseline")
    configs.append(baseline_config)
    
    # Full NSRPO
    full_config = ExperimentConfig(**asdict(base_config))
    full_config.experiment_name = f"{experiment_name_prefix}_full"
    full_config.experiment_description = "Full NSRPO with all components"
    full_config.tags.append("ablation_full")
    configs.append(full_config)
    
    # Ablation: No cosine loss (alpha_2 = 0)
    no_cosine_config = ExperimentConfig(**asdict(base_config))
    no_cosine_config.model.alpha_2 = 0.0
    no_cosine_config.experiment_name = f"{experiment_name_prefix}_no_cosine"
    no_cosine_config.experiment_description = "NSRPO without cosine similarity loss"
    no_cosine_config.tags.append("ablation_no_cosine")
    configs.append(no_cosine_config)
    
    # Ablation: No norm preservation (alpha_3 = 0)
    no_norm_config = ExperimentConfig(**asdict(base_config))
    no_norm_config.model.alpha_3 = 0.0
    no_norm_config.experiment_name = f"{experiment_name_prefix}_no_norm"
    no_norm_config.experiment_description = "NSRPO without norm preservation loss"
    no_norm_config.tags.append("ablation_no_norm")
    configs.append(no_norm_config)
    
    # Ablation: Different decoder architectures
    small_decoder_config = ExperimentConfig(**asdict(base_config))
    small_decoder_config.model.decoder_layers = 1
    small_decoder_config.model.decoder_heads = 4
    small_decoder_config.experiment_name = f"{experiment_name_prefix}_small_decoder"
    small_decoder_config.experiment_description = "NSRPO with small decoder architecture"
    small_decoder_config.tags.append("ablation_small_decoder")
    configs.append(small_decoder_config)
    
    return configs


def load_config_from_args(args) -> ExperimentConfig:
    """
    Create experiment configuration from command-line arguments.
    
    Args:
        args: Parsed command-line arguments (from argparse)
        
    Returns:
        ExperimentConfig instance
    """
    # Model configuration
    model_config = ModelConfig(
        base_model_path=args.model_path,
        use_null_decoder=args.use_null_decoder,
        extract_null_basis=args.extract_null_basis,
        null_basis_path=args.null_basis_path,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_dropout=args.decoder_dropout,
        alpha_1=args.alpha_1,
        alpha_2=args.alpha_2,
        alpha_3=args.alpha_3
    )
    
    # Training configuration
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        scheduler_type=args.scheduler_type,
        fp16=args.fp16,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_every=args.log_every
    )
    
    # Data configuration
    data_config = DataConfig(
        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path,
        num_dummy_samples=args.num_dummy_samples
    )
    
    # Create main configuration
    config = ExperimentConfig(
        experiment_name=getattr(args, 'experiment_name', 'nsrpo_experiment'),
        model=model_config,
        training=training_config,
        data=data_config,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        log_level=args.log_level
    )
    
    return config


if __name__ == "__main__":
    # Example usage and testing
    print("Testing NSRPO Configuration System...")
    
    # Create default configuration
    config = ExperimentConfig(
        experiment_name="test_experiment",
        experiment_description="Test configuration system"
    )
    
    # Validate configuration
    if config.validate():
        print("✓ Configuration validation passed")
    else:
        print("✗ Configuration validation failed")
    
    # Save configuration
    config_path = config.save("test_config.yaml")
    print(f"✓ Configuration saved to {config_path}")
    
    # Load configuration
    loaded_config = ExperimentConfig.load(config_path)
    print("✓ Configuration loaded successfully")
    
    # Test hyperparameter search
    search = HyperparameterSearch(config)
    search.add_parameter_range("model.alpha_1", [0.05, 0.1, 0.2])
    search.add_parameter_range("model.alpha_2", [0.05, 0.1, 0.2])
    search.add_parameter_range("training.learning_rate", [1e-5, 5e-5, 1e-4])
    
    search_configs = search.generate_configurations(max_configs=5)
    print(f"✓ Generated {len(search_configs)} search configurations")
    
    # Test ablation study configs
    ablation_configs = create_ablation_configs(config)
    print(f"✓ Generated {len(ablation_configs)} ablation configurations")
    
    # Test validation
    validator = ConfigurationValidator()
    path_issues = validator.validate_paths(config)
    resource_issues = validator.validate_resources(config)
    hyperparam_issues = validator.validate_hyperparameters(config)
    
    total_issues = len(path_issues) + len(resource_issues) + len(hyperparam_issues)
    print(f"✓ Configuration validation found {total_issues} potential issues")
    
    # Clean up test file
    if os.path.exists(config_path):
        os.remove(config_path)
    
    print("✓ Configuration system test completed successfully!")