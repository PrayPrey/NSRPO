#!/usr/bin/env python3
"""
Test visualization with dummy data
"""

import json
from pathlib import Path
from visualize_results import create_comparison_plots

# Create dummy evaluation results
dummy_nspo_results = {
    "accuracy": {
        "token_accuracy": 0.723,
        "sequence_accuracy": 0.412
    },
    "perplexity": {
        "perplexity": 45.32,
        "avg_loss": 3.814,
        "loss_std": 0.245
    },
    "training_efficiency": {
        "variance": 3.27e-07,
        "mean": 0.00012,
        "std": 1.23e-05,
        "num_batches": 10
    },
    "null_space_info": {
        "model_type": "NSRPOModel",
        "loss_weights": {
            "alpha_1": 0.1,
            "alpha_2": 0.1,
            "alpha_3": 0.05
        },
        "null_decoder_info": {
            "null_dim": 19,
            "hidden_size": 768,
            "compression_ratio": 0.024739583333333332,
            "null_basis_shape": [192, 19],
            "shared_lm_head": True
        }
    }
}

dummy_baseline_results = {
    "accuracy": {
        "token_accuracy": 0.651,
        "sequence_accuracy": 0.328
    },
    "perplexity": {
        "perplexity": 67.89,
        "avg_loss": 4.218,
        "loss_std": 0.312
    }
}

# Save dummy results
Path("test_outputs").mkdir(exist_ok=True)

with open("test_outputs/dummy_nspo_results.json", "w") as f:
    json.dump(dummy_nspo_results, f, indent=2)

with open("test_outputs/dummy_baseline_results.json", "w") as f:
    json.dump(dummy_baseline_results, f, indent=2)

print("Creating visualization with dummy data...")

# Generate plots
create_comparison_plots(
    dummy_nspo_results,
    dummy_baseline_results,
    output_dir="test_outputs/visualization_test"
)

print("\n[DONE] Test complete! Check test_outputs/visualization_test/ for generated plots")