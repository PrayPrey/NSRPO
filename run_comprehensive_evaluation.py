#!/usr/bin/env python3
"""
Comprehensive Evaluation Pipeline for NSRPO
Complete evaluation pipeline demonstrating all implemented evaluation components.

This script showcases the full evaluation framework including:
- Model evaluation with comprehensive metrics
- Statistical significance testing
- Automated ablation studies
- Paper-ready visualizations
- LaTeX table generation

Usage:
    python run_comprehensive_evaluation.py --config config.yaml --output-dir ./evaluation_results
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import our evaluation framework
from config import ExperimentConfig, create_ablation_configs
from evaluation import (
    ComprehensiveEvaluator, 
    StatisticalTester,
    AblationStudyFramework,
    LaTeXTableGenerator
)
from visualization import PaperPlotGenerator
from evaluate import NSRPOEvaluator
from models import NSRPOModel, create_nsrpo_model
from utils.dataset import get_dataloader, create_dummy_data


def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    log_file = output_dir / "evaluation.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def create_mock_models_and_data(config: ExperimentConfig, logger: logging.Logger):
    """
    Create mock models and data for demonstration.
    In a real scenario, you would load your trained models and actual datasets.
    """
    logger.info("Creating mock models and data for demonstration...")
    
    # Create tokenizer (using a small model for demo)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create mock models
    models = {}
    
    # Mock NSRPO model
    base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    
    # Extract null basis from the base model
    from utils.svd_utils import extract_base_null_basis
    null_basis = extract_base_null_basis(base_model, epsilon_factor=1e-3)
    
    # Save null basis to a temporary file
    import tempfile
    import os
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
    null_basis_path = temp_file.name
    torch.save(null_basis, null_basis_path)
    temp_file.close()
    
    nsrpo_model = create_nsrpo_model(
        base_model=base_model,
        null_basis_path=null_basis_path,
        vocab_size=tokenizer.vocab_size,
        hidden_size=base_model.config.n_embd,
        alpha_1=config.model.alpha_1,
        alpha_2=config.model.alpha_2,
        alpha_3=config.model.alpha_3
    )
    
    # Clean up temporary file
    os.unlink(null_basis_path)
    
    models['NSRPO'] = nsrpo_model
    
    # Mock baseline model
    models['GRPO_Baseline'] = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    
    # Create mock evaluation data
    dummy_data = create_dummy_data(500)
    eval_dataloader = get_dataloader(
        data=dummy_data,
        tokenizer=tokenizer,
        batch_size=8,
        max_length=256,
        shuffle=False
    )
    
    # Create mock training data for dynamics analysis
    train_dummy_data = create_dummy_data(200)
    train_dataloader = get_dataloader(
        data=train_dummy_data,
        tokenizer=tokenizer,
        batch_size=8,
        max_length=256,
        shuffle=True
    )
    
    return models, tokenizer, eval_dataloader, train_dataloader


def run_comprehensive_evaluation(
    models: Dict[str, Any],
    tokenizer: Any,
    eval_dataloader: Any,
    train_dataloader: Any,
    config: ExperimentConfig,
    output_dir: Path,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Run comprehensive evaluation on all models."""
    logger.info("Starting comprehensive evaluation...")
    
    results = {
        'model_comparison': {},
        'detailed_evaluations': {},
        'training_dynamics': {},
        'metadata': {
            'evaluation_timestamp': time.time(),
            'config': config.to_dict()
        }
    }
    
    # Evaluate each model
    for model_name, model in models.items():
        logger.info(f"Evaluating model: {model_name}")
        
        # Create comprehensive evaluator
        evaluator = ComprehensiveEvaluator(model, tokenizer, device='auto')
        
        # Run comprehensive evaluation
        model_results = evaluator.run_comprehensive_evaluation(
            eval_dataloader=eval_dataloader,
            train_dataloader=train_dataloader,
            baseline_models=[models['GRPO_Baseline']] if model_name != 'GRPO_Baseline' else None,
            max_eval_batches=10,  # Limited for demo
            save_detailed_results=True,
            output_dir=output_dir / f"{model_name}_detailed"
        )
        
        results['detailed_evaluations'][model_name] = model_results
        
        # Extract key metrics for comparison
        results['model_comparison'][model_name] = {
            'accuracy': model_results['accuracy_metrics']['token_accuracy'],
            'perplexity': model_results['perplexity_metrics']['perplexity'],
            'kl_divergence': model_results.get('distributional_metrics', {}).get('kl_divergence_mean', 0.0),
            'training_time': np.random.uniform(3600, 7200),  # Mock training time
            'num_parameters': model_results['evaluation_metadata']['num_parameters']
        }
    
    return results


def run_statistical_analysis(
    results: Dict[str, Any],
    output_dir: Path,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Run statistical significance testing."""
    logger.info("Running statistical significance analysis...")
    
    tester = StatisticalTester(alpha=0.05)
    statistical_results = {}
    
    # Compare models pairwise
    model_names = list(results['model_comparison'].keys())
    
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            comparison_name = f"{model1}_vs_{model2}"
            
            # Extract accuracy values for comparison (mock multiple runs)
            acc1_values = np.random.normal(
                results['model_comparison'][model1]['accuracy'], 0.01, 10
            )
            acc2_values = np.random.normal(
                results['model_comparison'][model2]['accuracy'], 0.01, 10
            )
            
            # Perform t-test
            test_result = tester.compare_two_groups(
                acc1_values, acc2_values, test_type='auto'
            )
            
            statistical_results[comparison_name] = {
                'test_name': test_result.test_name,
                'statistic': test_result.statistic,
                'p_value': test_result.p_value,
                'effect_size': test_result.effect_size,
                'significant': test_result.p_value < 0.05,
                'interpretation': test_result.interpretation
            }
    
    # Save statistical results
    with open(output_dir / 'statistical_analysis.json', 'w') as f:
        json.dump(statistical_results, f, indent=2, default=str)
    
    return statistical_results


def run_ablation_study(
    config: ExperimentConfig,
    output_dir: Path,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Run automated ablation study."""
    logger.info("Running automated ablation study...")
    
    # Create ablation study framework
    ablation_framework = AblationStudyFramework(
        base_config=config,
        output_dir=output_dir / "ablation_studies"
    )
    
    # Mock ablation results (in reality, this would train and evaluate models)
    from evaluation.ablation_study import AblationResult
    mock_ablation_results = []
    
    # Mock results for alpha_1 ablation
    alpha_1_values = [0.0, 0.05, 0.1, 0.15, 0.2]
    for value in alpha_1_values:
        result = AblationResult(
            component_name='alpha_1_ce_loss',
            component_value=value,
            metrics={
                'accuracy': 0.75 + 0.08 * np.sin(value * 10) + np.random.normal(0, 0.01),
                'perplexity': 3.0 - 0.5 * value + np.random.normal(0, 0.1)
            },
            training_time=np.random.uniform(100, 200),
            model_size=None
        )
        mock_ablation_results.append(result)
    
    # Mock results for decoder layers ablation
    layer_values = [1, 2, 3, 4, 5]
    for value in layer_values:
        result = AblationResult(
            component_name='decoder_layers',
            component_value=value,
            metrics={
                'accuracy': 0.78 + 0.05 * np.log(value) + np.random.normal(0, 0.01),
                'perplexity': 2.8 - 0.1 * value + np.random.normal(0, 0.05)
            },
            training_time=np.random.uniform(100, 200),
            model_size=None
        )
        mock_ablation_results.append(result)
    
    # Analyze ablation results
    analyses = ablation_framework._analyze_ablation_results(mock_ablation_results)
    summary = ablation_framework._generate_ablation_summary(mock_ablation_results, analyses)
    
    # Save results
    ablation_results = {
        'results': [result.__dict__ for result in mock_ablation_results],
        'analyses': [analysis.__dict__ for analysis in analyses],
        'summary': summary
    }
    
    with open(output_dir / 'ablation_study.json', 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)
    
    return ablation_results


def generate_visualizations(
    results: Dict[str, Any],
    statistical_results: Dict[str, Any],
    ablation_results: Dict[str, Any],
    output_dir: Path,
    logger: logging.Logger
) -> Dict[str, str]:
    """Generate paper-ready visualizations."""
    logger.info("Generating paper-ready visualizations...")
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    plotter = PaperPlotGenerator(output_dir=str(viz_dir))
    generated_plots = {}
    
    # Model comparison plot
    comparison_plot = plotter.plot_model_comparison(
        results['model_comparison'],
        metrics=['accuracy', 'perplexity', 'kl_divergence'],
        title="NSRPO vs. Baseline Model Performance",
        save_name="model_comparison",
        statistical_tests=statistical_results
    )
    generated_plots['model_comparison'] = comparison_plot
    
    # Ablation study plots
    for component in ['alpha_1_ce_loss', 'decoder_layers']:
        try:
            ablation_plot = plotter.plot_ablation_study(
                ablation_results['results'],
                component,
                metric='accuracy',
                title=f"Ablation Study: {component.replace('_', ' ').title()}",
                save_name=f"ablation_{component}",
                show_baseline=True
            )
            generated_plots[f'ablation_{component}'] = ablation_plot
        except Exception as e:
            logger.warning(f"Failed to create ablation plot for {component}: {e}")
    
    # Performance heatmap
    models = list(results['model_comparison'].keys())
    metrics = ['accuracy', 'perplexity', 'kl_divergence']
    
    matrix = np.array([
        [results['model_comparison'][model][metric] for metric in metrics]
        for model in models
    ])
    
    heatmap_plot = plotter.plot_performance_heatmap(
        matrix, models, metrics,
        title="Model Performance Matrix",
        save_name="performance_heatmap"
    )
    generated_plots['performance_heatmap'] = heatmap_plot
    
    return generated_plots


def generate_latex_tables(
    results: Dict[str, Any],
    statistical_results: Dict[str, Any],
    ablation_results: Dict[str, Any],
    config: ExperimentConfig,
    output_dir: Path,
    logger: logging.Logger
) -> Dict[str, str]:
    """Generate LaTeX tables for paper."""
    logger.info("Generating LaTeX tables...")
    
    latex_dir = output_dir / "latex_tables"
    latex_dir.mkdir(exist_ok=True)
    
    table_generator = LaTeXTableGenerator(output_dir=str(latex_dir))
    generated_tables = {}
    
    # Model comparison table
    comparison_table = table_generator.create_model_comparison_table(
        results['model_comparison'],
        metrics=['accuracy', 'perplexity', 'kl_divergence'],
        caption="Performance comparison of NSRPO and baseline models on evaluation dataset",
        label="nsrpo_comparison",
        statistical_tests=statistical_results
    )
    generated_tables['model_comparison'] = comparison_table
    
    # Statistical significance table
    significance_table = table_generator.create_statistical_significance_table(
        statistical_results,
        caption="Statistical significance tests for model performance comparisons",
        label="nsrpo_significance"
    )
    generated_tables['statistical_significance'] = significance_table
    
    # Hyperparameter configuration table
    hyperparams = {
        'learning_rate': 5e-5,
        'batch_size': 16,
        'alpha_1': config.model.alpha_1,
        'alpha_2': config.model.alpha_2,
        'alpha_3': config.model.alpha_3,
        'decoder_layers': config.model.decoder_layers,
        'decoder_heads': config.model.decoder_heads,
        'dropout': config.model.decoder_dropout
    }
    
    hyperparams_table = table_generator.create_hyperparameter_table(
        hyperparams,
        caption="Hyperparameter configuration for NSRPO training",
        label="nsrpo_hyperparams"
    )
    generated_tables['hyperparameters'] = hyperparams_table
    
    # Ablation study tables
    for component in ['alpha_1_ce_loss', 'decoder_layers']:
        try:
            ablation_table = table_generator.create_ablation_study_table(
                ablation_results['results'],
                component,
                metrics=['accuracy', 'perplexity'],
                caption=f"Ablation study results for {component.replace('_', ' ')} parameter",
                label=f"nsrpo_ablation_{component}"
            )
            generated_tables[f'ablation_{component}'] = ablation_table
        except Exception as e:
            logger.warning(f"Failed to create ablation table for {component}: {e}")
    
    return generated_tables


def create_evaluation_summary(
    results: Dict[str, Any],
    statistical_results: Dict[str, Any],
    ablation_results: Dict[str, Any],
    generated_plots: Dict[str, str],
    generated_tables: Dict[str, str],
    output_dir: Path
):
    """Create comprehensive evaluation summary."""
    summary = {
        'evaluation_overview': {
            'models_evaluated': list(results['model_comparison'].keys()),
            'metrics_computed': ['accuracy', 'perplexity', 'kl_divergence'],
            'statistical_tests_performed': len(statistical_results),
            'ablation_components_tested': len(set(
                r['component_name'] for r in ablation_results['results']
            )),
            'visualizations_generated': len(generated_plots),
            'tables_generated': len(generated_tables)
        },
        
        'key_findings': {
            'best_model': max(
                results['model_comparison'].keys(),
                key=lambda k: results['model_comparison'][k]['accuracy']
            ),
            'largest_performance_gap': max(
                results['model_comparison'][k]['accuracy'] 
                for k in results['model_comparison'].keys()
            ) - min(
                results['model_comparison'][k]['accuracy'] 
                for k in results['model_comparison'].keys()
            ),
            'significant_comparisons': sum(
                1 for test in statistical_results.values() if test['significant']
            ),
            'most_important_component': ablation_results['analyses'][0]['component_name'] if ablation_results['analyses'] else None
        },
        
        'performance_summary': results['model_comparison'],
        'statistical_summary': {
            name: {k: v for k, v in test.items() if k != 'interpretation'}
            for name, test in statistical_results.items()
        },
        'ablation_summary': ablation_results['summary'],
        
        'output_files': {
            'visualizations': generated_plots,
            'tables': generated_tables,
            'detailed_results': {
                'statistical_analysis': 'statistical_analysis.json',
                'ablation_study': 'ablation_study.json',
                'evaluation_log': 'evaluation.log'
            }
        }
    }
    
    # Save summary
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return summary


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation pipeline for NSRPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to experiment configuration file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./comprehensive_evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--max-eval-batches', type=int, default=20,
        help='Maximum number of evaluation batches (for demo speed)'
    )
    parser.add_argument(
        '--skip-ablation', action='store_true',
        help='Skip ablation study (faster for testing)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for evaluation'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation pipeline."""
    args = parse_arguments()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir, args.log_level)
    logger.info("Starting comprehensive NSRPO evaluation pipeline")
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    try:
        # Load or create configuration
        if args.config:
            config = ExperimentConfig.load(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            from config import ModelConfig, TrainingConfig, DataConfig, EvaluationConfig
            config = ExperimentConfig(
                experiment_name="comprehensive_evaluation_demo",
                experiment_description="Comprehensive evaluation pipeline demonstration",
                model=ModelConfig(
                    base_model_path="microsoft/DialoGPT-small",
                    use_null_decoder=True,
                    extract_null_basis=True,  # Extract null basis automatically
                    alpha_1=0.1,
                    alpha_2=0.1,
                    alpha_3=0.05
                ),
                training=TrainingConfig(batch_size=8, num_epochs=3),
                data=DataConfig(num_dummy_samples=500),
                evaluation=EvaluationConfig(
                    eval_batch_size=8,
                    compute_accuracy=True,
                    compute_perplexity=True,
                    compute_kl_divergence=True
                )
            )
            logger.info("Using default configuration")
        
        # Save configuration
        config.save(str(output_dir / "evaluation_config.yaml"))
        
        # Create models and data (mock for demonstration)
        models, tokenizer, eval_dataloader, train_dataloader = create_mock_models_and_data(
            config, logger
        )
        
        # Run comprehensive evaluation
        evaluation_results = run_comprehensive_evaluation(
            models, tokenizer, eval_dataloader, train_dataloader,
            config, output_dir, logger
        )
        
        # Run statistical analysis
        statistical_results = run_statistical_analysis(
            evaluation_results, output_dir, logger
        )
        
        # Run ablation study (optional)
        if not args.skip_ablation:
            ablation_results = run_ablation_study(config, output_dir, logger)
        else:
            ablation_results = {'results': [], 'analyses': [], 'summary': {}}
            logger.info("Skipping ablation study")
        
        # Generate visualizations
        generated_plots = generate_visualizations(
            evaluation_results, statistical_results, ablation_results,
            output_dir, logger
        )
        
        # Generate LaTeX tables
        generated_tables = generate_latex_tables(
            evaluation_results, statistical_results, ablation_results,
            config, output_dir, logger
        )
        
        # Create evaluation summary
        summary = create_evaluation_summary(
            evaluation_results, statistical_results, ablation_results,
            generated_plots, generated_tables, output_dir
        )
        
        # Print summary
        logger.info("=== EVALUATION PIPELINE COMPLETED ===")
        logger.info(f"Models evaluated: {summary['evaluation_overview']['models_evaluated']}")
        logger.info(f"Best performing model: {summary['key_findings']['best_model']}")
        logger.info(f"Significant statistical comparisons: {summary['key_findings']['significant_comparisons']}")
        logger.info(f"Visualizations generated: {summary['evaluation_overview']['visualizations_generated']}")
        logger.info(f"LaTeX tables generated: {summary['evaluation_overview']['tables_generated']}")
        logger.info(f"Complete results saved to: {output_dir.absolute()}")
        
        print("\n🎉 Comprehensive NSRPO evaluation completed successfully!")
        print(f"📁 Results available in: {output_dir.absolute()}")
        print(f"📊 Key findings: {summary['key_findings']['best_model']} performed best")
        print(f"📈 {len(generated_plots)} visualizations and {len(generated_tables)} LaTeX tables generated")
        print(f"📋 See evaluation_summary.json for complete results")
        
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")
        logger.error(f"Error details: {str(e)}")
        raise


if __name__ == '__main__':
    main()