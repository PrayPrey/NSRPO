# NSRPO Comprehensive Evaluation Framework

This document describes the comprehensive evaluation and testing framework implemented for the NSRPO (Null-Space Regularized Policy Optimization) project. The framework provides publication-ready evaluation tools for academic research.

## Overview

The evaluation framework consists of seven main components:

1. **Basic Inference/Evaluation Script** (`evaluate.py`)
2. **Experiment Configuration System** (`config/`)
3. **Comprehensive Evaluation Framework** (`evaluation/comprehensive_evaluator.py`)
4. **Statistical Significance Testing** (`evaluation/statistical_tests.py`)
5. **Automated Ablation Study** (`evaluation/ablation_study.py`)
6. **Paper-Ready Visualizations** (`visualization/paper_plots.py`)
7. **LaTeX Table Generation** (`evaluation/latex_tables.py`)

## Quick Start

### Basic Evaluation

```bash
# Run basic model evaluation
python evaluate.py --model_path ./trained_model --eval_data_path ./eval_data.json

# Compare with baseline
python evaluate.py --model_path ./trained_model --baseline_model_path ./baseline_model --eval_data_path ./eval_data.json
```

### Comprehensive Evaluation Pipeline

```bash
# Run complete evaluation pipeline
python run_comprehensive_evaluation.py --output-dir ./results

# With custom configuration
python run_comprehensive_evaluation.py --config config.yaml --output-dir ./results
```

## Component Details

### 1. Basic Inference/Evaluation (`evaluate.py`)

**Purpose**: Core evaluation script with fundamental metrics.

**Features**:
- Token-level and sequence-level accuracy
- Perplexity calculation with confidence intervals
- KL divergence analysis between models
- Training efficiency metrics (gradient variance)
- Text generation evaluation

**Key Metrics**:
- `token_accuracy`: Per-token prediction accuracy
- `sequence_accuracy`: Complete sequence accuracy
- `perplexity`: Model perplexity with statistical bounds
- `kl_divergence`: KL divergence from baseline
- `gradient_variance`: Policy gradient variance (NSRPO-specific)

**Usage**:
```python
from evaluate import NSRPOEvaluator

evaluator = NSRPOEvaluator(model, tokenizer)
results = evaluator.comprehensive_evaluation(eval_dataloader)
```

### 2. Experiment Configuration (`config/`)

**Purpose**: Systematic hyperparameter and experiment management.

**Features**:
- Hierarchical configuration with validation
- Hyperparameter search space definition
- Reproducibility with configuration hashing
- Ablation study configuration generation

**Configuration Structure**:
```python
@dataclass
class ExperimentConfig:
    experiment_name: str
    model: ModelConfig          # Model architecture & loss weights
    training: TrainingConfig    # Optimization parameters  
    data: DataConfig           # Dataset configuration
    evaluation: EvaluationConfig # Evaluation settings
```

**Usage**:
```python
from config import ExperimentConfig

config = ExperimentConfig.load('config.yaml')
config.save('experiment_config.yaml')

# Hyperparameter search
search = HyperparameterSearch(config)
search.add_parameter_range('model.alpha_1', [0.05, 0.1, 0.2])
configs = search.generate_configurations()
```

### 3. Comprehensive Evaluation (`evaluation/comprehensive_evaluator.py`)

**Purpose**: Advanced evaluation with publication-ready metrics.

**Features**:
- Bootstrap confidence intervals
- Per-token statistics and distributions
- Cross-model distributional comparisons
- Training dynamics analysis
- Comprehensive metric aggregation

**Advanced Metrics**:
- Confidence intervals via bootstrap sampling
- Weighted precision/recall/F1 scores
- Entropy and distributional measures
- Jensen-Shannon divergence
- Training efficiency analysis

**Usage**:
```python
from evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator(model, tokenizer)
results = evaluator.run_comprehensive_evaluation(
    eval_dataloader=eval_data,
    baseline_models=[baseline_model],
    save_detailed_results=True
)
```

### 4. Statistical Significance Testing (`evaluation/statistical_tests.py`)

**Purpose**: Rigorous statistical analysis for model comparisons.

**Features**:
- Parametric and non-parametric tests
- Multiple comparison corrections
- Effect size calculations
- Power analysis and sample size recommendations
- Assumption testing (normality, equal variance)

**Supported Tests**:
- Student's t-test (paired/independent)
- Wilcoxon signed-rank test
- Mann-Whitney U test
- One-way ANOVA / Kruskal-Wallis
- Chi-square / Fisher's exact test
- McNemar's test for paired proportions

**Usage**:
```python
from evaluation import StatisticalTester

tester = StatisticalTester(alpha=0.05)

# Compare two models
result = tester.compare_two_groups(model1_scores, model2_scores)
print(f"p-value: {result.p_value}, Effect size: {result.effect_size}")

# Multiple comparisons with correction
corrected = tester.correct_multiple_comparisons(p_values, method='bonferroni')
```

### 5. Automated Ablation Study (`evaluation/ablation_study.py`)

**Purpose**: Systematic component importance analysis.

**Features**:
- Automated configuration generation
- Component importance ranking
- Statistical significance testing
- Recommendation generation
- Comprehensive result reporting

**Standard NSRPO Components**:
- Loss weights (`alpha_1`, `alpha_2`, `alpha_3`)
- Architecture parameters (layers, heads, dropout)
- Training hyperparameters
- Full null decoder ablation

**Usage**:
```python
from evaluation import AblationStudyFramework

framework = AblationStudyFramework(base_config)
study_results = framework.run_ablation_study(
    components=['alpha_1_ce_loss', 'decoder_layers'],
    train_func=train_model,
    eval_func=evaluate_model
)
```

### 6. Paper-Ready Visualizations (`visualization/paper_plots.py`)

**Purpose**: Publication-quality figures for academic papers.

**Features**:
- IEEE/ACM conference formatting standards
- Statistical significance annotations
- Vector graphics output (PDF, EPS, PNG)
- Multiple color schemes and styling options
- Multi-panel summary figures

**Supported Plot Types**:
- Model comparison with error bars
- Training curves with smoothing
- Ablation study results
- Performance heatmaps
- Statistical comparison plots
- Correlation matrices

**Usage**:
```python
from visualization import PaperPlotGenerator

plotter = PaperPlotGenerator(output_dir='./figures')

# Model comparison
plotter.plot_model_comparison(
    results, metrics=['accuracy', 'perplexity'],
    statistical_tests=significance_tests
)

# Ablation study
plotter.plot_ablation_study(ablation_data, 'alpha_1_ce_loss')
```

### 7. LaTeX Table Generation (`evaluation/latex_tables.py`)

**Purpose**: Academic-quality LaTeX tables.

**Features**:
- IEEE/ACM/Springer formatting standards
- Statistical significance indicators
- Confidence interval formatting
- Booktabs and siunitx package support
- Complete document generation

**Table Types**:
- Model comparison with significance tests
- Ablation study results
- Statistical test summaries
- Hyperparameter configurations
- Correlation matrices
- Training summaries

**Usage**:
```python
from evaluation import LaTeXTableGenerator

generator = LaTeXTableGenerator(output_dir='./latex_tables')

# Model comparison table
latex_table = generator.create_model_comparison_table(
    results, metrics=['accuracy', 'perplexity'],
    caption="Model Performance Comparison",
    statistical_tests=significance_results
)
```

## Example Workflows

### Complete Evaluation Pipeline

```python
# 1. Load configuration and models
config = ExperimentConfig.load('config.yaml')
models = load_trained_models()
eval_data = load_evaluation_data()

# 2. Run comprehensive evaluation
evaluator = ComprehensiveEvaluator(models['nsrpo'], tokenizer)
eval_results = evaluator.run_comprehensive_evaluation(
    eval_dataloader=eval_data,
    baseline_models=[models['baseline']]
)

# 3. Statistical analysis
tester = StatisticalTester()
stat_results = tester.compare_multiple_groups([
    eval_results['nsrpo']['accuracy_scores'],
    eval_results['baseline']['accuracy_scores']
])

# 4. Ablation study
ablation = AblationStudyFramework(config)
ablation_results = ablation.run_ablation_study(
    components=['alpha_1_ce_loss', 'decoder_layers']
)

# 5. Generate visualizations
plotter = PaperPlotGenerator()
plotter.plot_model_comparison(eval_results, stat_results)
plotter.plot_ablation_study(ablation_results, 'alpha_1_ce_loss')

# 6. Generate LaTeX tables
table_gen = LaTeXTableGenerator()
table_gen.create_model_comparison_table(eval_results, stat_results)
table_gen.create_ablation_study_table(ablation_results, 'alpha_1_ce_loss')
```

### Hyperparameter Search with Evaluation

```python
# Define search space
search = HyperparameterSearch(base_config)
search.add_parameter_range('model.alpha_1', [0.05, 0.1, 0.15, 0.2])
search.add_parameter_range('model.alpha_2', [0.05, 0.1, 0.15, 0.2])
search.add_parameter_range('training.learning_rate', [1e-5, 5e-5, 1e-4])

# Generate configurations
configs = search.generate_configurations(max_configs=20)

# Train and evaluate each configuration
results = []
for config in configs:
    model = train_model(config)
    eval_result = evaluate_model(model, eval_data)
    results.append({
        'config': config,
        'results': eval_result
    })

# Analyze best hyperparameters
best_config = max(results, key=lambda x: x['results']['accuracy'])
print(f"Best configuration: {best_config['config']}")
```

## Output Structure

The evaluation framework generates organized outputs:

```
evaluation_results/
├── evaluation_summary.json           # Overall summary
├── evaluation_config.yaml            # Configuration used
├── evaluation.log                    # Detailed logs
├── statistical_analysis.json         # Statistical test results
├── ablation_study.json              # Ablation study results
├── visualizations/                   # Paper-ready figures
│   ├── model_comparison.pdf
│   ├── ablation_alpha_1_ce_loss.pdf
│   └── performance_heatmap.pdf
├── latex_tables/                     # LaTeX tables
│   ├── model_comparison.tex
│   ├── ablation_alpha_1_ce_loss.tex
│   └── statistical_significance.tex
└── detailed_results/                 # Per-model detailed results
    ├── NSRPO_detailed/
    └── baseline_detailed/
```

## Statistical Rigor

The framework implements academic-level statistical rigor:

### Multiple Comparisons
- Bonferroni correction
- Holm-Sidak method  
- False Discovery Rate (FDR) control
- Family-wise error rate control

### Effect Sizes
- Cohen's d for continuous measures
- Odds ratios for categorical data
- Rank-biserial correlation for non-parametric tests
- Eta-squared for ANOVA

### Confidence Intervals
- Bootstrap confidence intervals
- Parametric confidence intervals
- Bayesian credible intervals (optional)

### Power Analysis
- Post-hoc power calculation
- Sample size recommendations
- Effect size interpretation

## Best Practices

### For Academic Papers

1. **Always report confidence intervals**: Use bootstrap CIs for robust estimates
2. **Correct for multiple comparisons**: Apply appropriate correction methods
3. **Report effect sizes**: Include practical significance alongside statistical significance
4. **Use appropriate tests**: Check assumptions and choose parametric vs non-parametric
5. **Provide complete statistics**: Include test statistics, p-values, effect sizes, and CIs

### For Reproducibility

1. **Save all configurations**: Use the configuration system for full reproducibility
2. **Set random seeds**: Ensure deterministic results
3. **Document hyperparameter choices**: Use ablation studies to justify selections
4. **Version control results**: Track model versions and evaluation results
5. **Provide statistical details**: Include test assumptions and method justifications

### For Efficient Evaluation

1. **Use batched evaluation**: Process multiple samples simultaneously
2. **Cache intermediate results**: Avoid recomputation of expensive metrics
3. **Parallel processing**: Leverage multiple cores for statistical tests
4. **Early stopping**: Use validation metrics to determine convergence
5. **Resource monitoring**: Track memory and compute usage

## Advanced Features

### Custom Metrics
```python
def custom_metric(predictions, targets, **kwargs):
    # Implement custom evaluation metric
    return metric_value

evaluator.add_custom_metric('my_metric', custom_metric)
```

### Custom Ablation Components
```python
custom_component = AblationComponent(
    name='custom_component',
    description='Custom component for ablation',
    config_path='model.custom_param',
    ablation_values=[0.1, 0.2, 0.5, 1.0],
    baseline_value=0.2
)

framework.add_custom_component(custom_component)
```

### Custom Statistical Tests
```python
def custom_test(group1, group2, **kwargs):
    # Implement custom statistical test
    return StatisticalTestResult(...)

tester.register_custom_test('my_test', custom_test)
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch sizes or use gradient accumulation
2. **Slow evaluation**: Enable caching and reduce max_batches
3. **Statistical test failures**: Check data distributions and sample sizes
4. **LaTeX compilation errors**: Ensure required packages are installed
5. **Visualization issues**: Check matplotlib backend and display settings

### Performance Optimization

1. **Use GPU acceleration**: Move models to GPU for faster inference
2. **Batch operations**: Process multiple samples simultaneously  
3. **Cache results**: Save intermediate computations
4. **Parallel evaluation**: Use multiple processes for independent evaluations
5. **Memory management**: Clear unused variables and use del statements

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@misc{nsrpo_evaluation_framework,
    title={Comprehensive Evaluation Framework for Null-Space Regularized Policy Optimization},
    author={NSRPO Development Team},
    year={2024},
    note={https://github.com/nsrpo/evaluation-framework}
}
```