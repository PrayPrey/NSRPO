"""
Automated Ablation Study Framework for NSRPO
Task 13: Create Automated Ablation Study - Component importance analysis

Comprehensive ablation study system for analyzing the contribution of different
components in NSRPO models. Provides automated experiment generation, execution,
and statistical analysis of component importance.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
import copy

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from config import ExperimentConfig, ModelConfig, create_ablation_configs
from models import NSRPOModel, create_nsrpo_model
from .comprehensive_evaluator import ComprehensiveEvaluator
from .statistical_tests import StatisticalTester, StatisticalTestResult


@dataclass
class AblationComponent:
    """Definition of a component for ablation study."""
    name: str
    description: str
    config_path: str  # Dot notation path to config parameter
    ablation_values: List[Any]  # Values to test (including baseline)
    baseline_value: Any  # Original/baseline value
    component_type: str = "parameter"  # parameter, module, loss_component


@dataclass
class AblationResult:
    """Results from a single ablation experiment."""
    component_name: str
    component_value: Any
    metrics: Dict[str, float]
    training_time: Optional[float] = None
    inference_time: Optional[float] = None
    model_size: Optional[int] = None
    convergence_epoch: Optional[int] = None


@dataclass
class AblationAnalysis:
    """Statistical analysis of ablation results."""
    component_name: str
    importance_score: float
    statistical_significance: Dict[str, Any]
    effect_sizes: Dict[str, float]
    ranking: int
    interpretation: str


class AblationStudyFramework:
    """
    Automated ablation study framework for NSRPO models.
    
    Provides systematic analysis of component contributions through
    controlled experiments and statistical evaluation.
    """
    
    def __init__(
        self,
        base_config: ExperimentConfig,
        evaluator_factory: Optional[Callable] = None,
        output_dir: str = "./ablation_studies",
        random_seed: int = 42
    ):
        """
        Initialize ablation study framework.
        
        Args:
            base_config: Base experiment configuration
            evaluator_factory: Factory function to create model evaluators
            output_dir: Directory for ablation study outputs
            random_seed: Random seed for reproducibility
        """
        self.base_config = base_config
        self.evaluator_factory = evaluator_factory or self._default_evaluator_factory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        
        # Initialize statistical tester
        self.statistical_tester = StatisticalTester(alpha=0.05, random_state=random_seed)
        
        # Results storage
        self.ablation_results = []
        self.analyses = []
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Define standard ablation components for NSRPO
        self.standard_components = self._define_standard_components()
    
    def _define_standard_components(self) -> List[AblationComponent]:
        """Define standard ablation components for NSRPO."""
        components = []
        
        # Loss weight components
        components.append(AblationComponent(
            name="alpha_1_ce_loss",
            description="Cross-entropy loss weight (α₁)",
            config_path="model.alpha_1",
            ablation_values=[0.0, 0.05, 0.1, 0.2, 0.5],
            baseline_value=0.1,
            component_type="loss_component"
        ))
        
        components.append(AblationComponent(
            name="alpha_2_cosine_loss",
            description="Cosine similarity loss weight (α₂)",
            config_path="model.alpha_2", 
            ablation_values=[0.0, 0.05, 0.1, 0.2, 0.5],
            baseline_value=0.1,
            component_type="loss_component"
        ))
        
        components.append(AblationComponent(
            name="alpha_3_norm_preservation",
            description="Norm preservation loss weight (α₃)",
            config_path="model.alpha_3",
            ablation_values=[0.0, 0.025, 0.05, 0.1, 0.2],
            baseline_value=0.05,
            component_type="loss_component"
        ))
        
        # Architecture components
        components.append(AblationComponent(
            name="decoder_layers",
            description="Number of decoder transformer layers",
            config_path="model.decoder_layers",
            ablation_values=[1, 2, 3, 4, 6],
            baseline_value=3,
            component_type="module"
        ))
        
        components.append(AblationComponent(
            name="decoder_heads",
            description="Number of decoder attention heads",
            config_path="model.decoder_heads",
            ablation_values=[2, 4, 8, 12, 16],
            baseline_value=8,
            component_type="module"
        ))
        
        components.append(AblationComponent(
            name="decoder_dropout",
            description="Decoder dropout rate",
            config_path="model.decoder_dropout",
            ablation_values=[0.0, 0.05, 0.1, 0.2, 0.3],
            baseline_value=0.1,
            component_type="parameter"
        ))
        
        # Training components
        components.append(AblationComponent(
            name="learning_rate",
            description="Learning rate for optimization",
            config_path="training.learning_rate",
            ablation_values=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
            baseline_value=5e-5,
            component_type="parameter"
        ))
        
        components.append(AblationComponent(
            name="use_null_decoder",
            description="Whether to use null decoder (full ablation)",
            config_path="model.use_null_decoder",
            ablation_values=[False, True],
            baseline_value=True,
            component_type="module"
        ))
        
        return components
    
    def add_custom_component(self, component: AblationComponent):
        """Add a custom ablation component."""
        self.standard_components.append(component)
        self.logger.info(f"Added custom ablation component: {component.name}")
    
    def generate_ablation_configs(
        self,
        components: Optional[List[str]] = None,
        include_baseline: bool = True
    ) -> List[Tuple[str, Any, ExperimentConfig]]:
        """
        Generate experiment configurations for ablation study.
        
        Args:
            components: List of component names to ablate (None = all)
            include_baseline: Whether to include baseline configuration
            
        Returns:
            List of (component_name, component_value, config) tuples
        """
        configs = []
        
        # Filter components
        if components is None:
            components_to_test = self.standard_components
        else:
            components_to_test = [c for c in self.standard_components if c.name in components]
        
        # Add baseline configuration
        if include_baseline:
            baseline_config = copy.deepcopy(self.base_config)
            baseline_config.experiment_name = f"{self.base_config.experiment_name}_baseline"
            baseline_config.experiment_description = "Baseline configuration for ablation study"
            configs.append(("baseline", "full_model", baseline_config))
        
        # Generate ablation configurations
        for component in components_to_test:
            for value in component.ablation_values:
                if value == component.baseline_value and include_baseline:
                    continue  # Skip duplicate baseline
                
                # Create modified configuration
                config = copy.deepcopy(self.base_config)
                config.experiment_name = f"{self.base_config.experiment_name}_ablate_{component.name}_{value}"
                config.experiment_description = f"Ablation: {component.description} = {value}"
                
                # Set the ablated value
                self._set_config_value(config, component.config_path, value)
                
                # Add tags
                config.tags.extend(["ablation", f"ablate_{component.name}", f"value_{value}"])
                
                configs.append((component.name, value, config))
        
        return configs
    
    def run_ablation_study(
        self,
        components: Optional[List[str]] = None,
        train_func: Optional[Callable] = None,
        eval_func: Optional[Callable] = None,
        max_experiments: Optional[int] = None,
        parallel_execution: bool = False,
        save_intermediate_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete ablation study.
        
        Args:
            components: Components to ablate (None = all)
            train_func: Function to train models (component_name, config) -> trained_model
            eval_func: Function to evaluate models (model, config) -> metrics
            max_experiments: Maximum number of experiments to run
            parallel_execution: Whether to run experiments in parallel
            save_intermediate_results: Whether to save results after each experiment
            
        Returns:
            Dictionary with complete ablation study results
        """
        self.logger.info("Starting automated ablation study...")
        
        # Generate configurations
        ablation_configs = self.generate_ablation_configs(components)
        
        if max_experiments:
            ablation_configs = ablation_configs[:max_experiments]
        
        self.logger.info(f"Generated {len(ablation_configs)} ablation experiments")
        
        # Run experiments
        if parallel_execution:
            results = self._run_experiments_parallel(ablation_configs, train_func, eval_func)
        else:
            results = self._run_experiments_sequential(ablation_configs, train_func, eval_func)
        
        # Analyze results
        self.logger.info("Analyzing ablation results...")
        analyses = self._analyze_ablation_results(results)
        
        # Generate summary
        summary = self._generate_ablation_summary(results, analyses)
        
        # Save results
        if save_intermediate_results:
            self._save_ablation_results(results, analyses, summary)
        
        return {
            'results': results,
            'analyses': analyses,
            'summary': summary,
            'configs': ablation_configs
        }
    
    def _run_experiments_sequential(
        self,
        ablation_configs: List[Tuple[str, Any, ExperimentConfig]],
        train_func: Optional[Callable],
        eval_func: Optional[Callable]
    ) -> List[AblationResult]:
        """Run ablation experiments sequentially."""
        results = []
        
        for component_name, component_value, config in tqdm(ablation_configs, desc="Running ablation experiments"):
            try:
                self.logger.info(f"Running experiment: {component_name} = {component_value}")
                
                # Time the experiment
                start_time = time.time()
                
                # Train model
                if train_func:
                    model = train_func(component_name, config)
                else:
                    model = self._default_train_func(component_name, config)
                
                training_time = time.time() - start_time
                
                # Evaluate model
                eval_start_time = time.time()
                if eval_func:
                    metrics = eval_func(model, config)
                else:
                    metrics = self._default_eval_func(model, config)
                
                inference_time = time.time() - eval_start_time
                
                # Get model size
                model_size = sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else None
                
                # Create result
                result = AblationResult(
                    component_name=component_name,
                    component_value=component_value,
                    metrics=metrics,
                    training_time=training_time,
                    inference_time=inference_time,
                    model_size=model_size
                )
                
                results.append(result)
                
                self.logger.info(f"Completed experiment: {component_name} = {component_value}")
                
            except Exception as e:
                self.logger.error(f"Experiment failed: {component_name} = {component_value}, Error: {e}")
                # Add failed result
                result = AblationResult(
                    component_name=component_name,
                    component_value=component_value,
                    metrics={'error': str(e)},
                    training_time=None,
                    inference_time=None,
                    model_size=None
                )
                results.append(result)
        
        return results
    
    def _run_experiments_parallel(
        self,
        ablation_configs: List[Tuple[str, Any, ExperimentConfig]],
        train_func: Optional[Callable],
        eval_func: Optional[Callable]
    ) -> List[AblationResult]:
        """Run ablation experiments in parallel (simplified implementation)."""
        self.logger.warning("Parallel execution not fully implemented, falling back to sequential")
        return self._run_experiments_sequential(ablation_configs, train_func, eval_func)
    
    def _analyze_ablation_results(self, results: List[AblationResult]) -> List[AblationAnalysis]:
        """Analyze ablation results for component importance."""
        analyses = []
        
        # Group results by component
        component_groups = {}
        for result in results:
            if result.component_name not in component_groups:
                component_groups[result.component_name] = []
            component_groups[result.component_name].append(result)
        
        # Find baseline results
        baseline_results = [r for r in results if r.component_name == "baseline"]
        baseline_metrics = baseline_results[0].metrics if baseline_results else {}
        
        # Analyze each component
        for component_name, component_results in component_groups.items():
            if component_name == "baseline":
                continue
            
            analysis = self._analyze_component_importance(
                component_name, component_results, baseline_metrics
            )
            analyses.append(analysis)
        
        # Rank components by importance
        analyses.sort(key=lambda x: x.importance_score, reverse=True)
        for i, analysis in enumerate(analyses):
            analysis.ranking = i + 1
        
        return analyses
    
    def _analyze_component_importance(
        self,
        component_name: str,
        component_results: List[AblationResult],
        baseline_metrics: Dict[str, float]
    ) -> AblationAnalysis:
        """Analyze importance of a single component."""
        # Extract metrics for analysis
        metric_names = set()
        for result in component_results:
            metric_names.update(result.metrics.keys())
        metric_names = list(metric_names - {'error'})
        
        # Calculate importance scores
        importance_scores = []
        effect_sizes = {}
        statistical_tests = {}
        
        for metric_name in metric_names:
            if metric_name not in baseline_metrics:
                continue
                
            baseline_value = baseline_metrics[metric_name]
            component_values = []
            
            for result in component_results:
                if metric_name in result.metrics and 'error' not in result.metrics:
                    component_values.append(result.metrics[metric_name])
            
            if not component_values:
                continue
            
            # Calculate importance as variance explained
            metric_variance = np.var(component_values) if len(component_values) > 1 else 0.0
            baseline_diff = np.mean([abs(v - baseline_value) for v in component_values])
            
            # Normalize importance score
            if baseline_value != 0:
                relative_importance = baseline_diff / abs(baseline_value)
            else:
                relative_importance = baseline_diff
            
            importance_scores.append(relative_importance)
            
            # Statistical comparison with baseline
            if len(component_values) > 1:
                baseline_array = np.array([baseline_value] * len(component_values))
                component_array = np.array(component_values)
                
                stat_result = self.statistical_tester.compare_two_groups(
                    component_array, baseline_array, paired=False
                )
                
                statistical_tests[metric_name] = {
                    'p_value': stat_result.p_value,
                    'effect_size': stat_result.effect_size,
                    'significant': stat_result.p_value < 0.05
                }
                
                if stat_result.effect_size is not None:
                    effect_sizes[metric_name] = stat_result.effect_size
        
        # Overall importance score
        overall_importance = np.mean(importance_scores) if importance_scores else 0.0
        
        # Generate interpretation
        interpretation = self._interpret_component_analysis(
            component_name, overall_importance, statistical_tests
        )
        
        return AblationAnalysis(
            component_name=component_name,
            importance_score=overall_importance,
            statistical_significance=statistical_tests,
            effect_sizes=effect_sizes,
            ranking=0,  # Will be set later
            interpretation=interpretation
        )
    
    def _interpret_component_analysis(
        self,
        component_name: str,
        importance_score: float,
        statistical_tests: Dict[str, Any]
    ) -> str:
        """Generate interpretation of component analysis."""
        interpretations = []
        
        # Importance level
        if importance_score > 0.2:
            importance_level = "high"
        elif importance_score > 0.1:
            importance_level = "moderate"
        elif importance_score > 0.05:
            importance_level = "low"
        else:
            importance_level = "negligible"
        
        interpretations.append(f"Component shows {importance_level} importance (score: {importance_score:.3f})")
        
        # Statistical significance
        significant_metrics = [name for name, test in statistical_tests.items() if test['significant']]
        if significant_metrics:
            interpretations.append(f"Statistically significant effects on: {', '.join(significant_metrics)}")
        else:
            interpretations.append("No statistically significant effects detected")
        
        return ". ".join(interpretations)
    
    def _generate_ablation_summary(
        self,
        results: List[AblationResult],
        analyses: List[AblationAnalysis]
    ) -> Dict[str, Any]:
        """Generate summary of ablation study."""
        summary = {
            'study_metadata': {
                'total_experiments': len(results),
                'successful_experiments': len([r for r in results if 'error' not in r.metrics]),
                'failed_experiments': len([r for r in results if 'error' in r.metrics]),
                'components_tested': len(set(r.component_name for r in results if r.component_name != 'baseline')),
                'analysis_timestamp': time.time()
            },
            'component_rankings': [
                {
                    'component': analysis.component_name,
                    'rank': analysis.ranking,
                    'importance_score': analysis.importance_score,
                    'interpretation': analysis.interpretation
                }
                for analysis in sorted(analyses, key=lambda x: x.ranking)
            ],
            'key_findings': self._extract_key_findings(analyses),
            'recommendations': self._generate_recommendations(analyses)
        }
        
        return summary
    
    def _extract_key_findings(self, analyses: List[AblationAnalysis]) -> List[str]:
        """Extract key findings from ablation analysis."""
        findings = []
        
        # Most important component
        if analyses:
            most_important = analyses[0]  # Already sorted by importance
            findings.append(f"Most important component: {most_important.component_name} "
                          f"(importance: {most_important.importance_score:.3f})")
        
        # Least important component
        if len(analyses) > 1:
            least_important = analyses[-1]
            findings.append(f"Least important component: {least_important.component_name} "
                          f"(importance: {least_important.importance_score:.3f})")
        
        # High-importance components
        high_importance = [a for a in analyses if a.importance_score > 0.2]
        if high_importance:
            findings.append(f"High-importance components ({len(high_importance)}): "
                          f"{', '.join(a.component_name for a in high_importance)}")
        
        # Negligible components
        negligible = [a for a in analyses if a.importance_score < 0.05]
        if negligible:
            findings.append(f"Components with negligible impact ({len(negligible)}): "
                          f"{', '.join(a.component_name for a in negligible)}")
        
        return findings
    
    def _generate_recommendations(self, analyses: List[AblationAnalysis]) -> List[str]:
        """Generate recommendations based on ablation analysis."""
        recommendations = []
        
        # Components to focus on
        important_components = [a for a in analyses if a.importance_score > 0.1]
        if important_components:
            recommendations.append("Focus hyperparameter tuning on: " + 
                                 ", ".join(a.component_name for a in important_components))
        
        # Components that can be simplified
        negligible_components = [a for a in analyses if a.importance_score < 0.05]
        if negligible_components:
            recommendations.append("Consider simplifying or removing: " + 
                                 ", ".join(a.component_name for a in negligible_components))
        
        # Architecture recommendations
        arch_components = [a for a in analyses if any('decoder' in a.component_name for a in analyses)]
        if arch_components:
            arch_important = [a for a in arch_components if a.importance_score > 0.1]
            if arch_important:
                recommendations.append("Critical architectural components: " + 
                                     ", ".join(a.component_name for a in arch_important))
        
        return recommendations
    
    def _save_ablation_results(
        self,
        results: List[AblationResult],
        analyses: List[AblationAnalysis],
        summary: Dict[str, Any]
    ):
        """Save ablation study results to files."""
        # Create timestamped output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        study_dir = self.output_dir / f"ablation_study_{timestamp}"
        study_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_data = [asdict(result) for result in results]
        with open(study_dir / "ablation_results.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save analyses
        analyses_data = [asdict(analysis) for analysis in analyses]
        with open(study_dir / "ablation_analyses.json", 'w') as f:
            json.dump(analyses_data, f, indent=2, default=str)
        
        # Save summary
        with open(study_dir / "ablation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save as CSV for easy analysis
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(study_dir / "ablation_results.csv", index=False)
        
        analyses_df = pd.DataFrame(analyses_data)
        analyses_df.to_csv(study_dir / "ablation_analyses.csv", index=False)
        
        self.logger.info(f"Ablation study results saved to {study_dir}")
    
    def _set_config_value(self, config: ExperimentConfig, path: str, value: Any):
        """Set nested configuration value using dot notation."""
        parts = path.split('.')
        current = config
        
        for part in parts[:-1]:
            current = getattr(current, part)
        
        setattr(current, parts[-1], value)
    
    def _default_evaluator_factory(self, model, tokenizer):
        """Default factory for creating model evaluators."""
        return ComprehensiveEvaluator(model, tokenizer)
    
    def _default_train_func(self, component_name: str, config: ExperimentConfig):
        """Default training function (placeholder)."""
        self.logger.warning("Using placeholder training function - implement actual training")
        
        # This would be replaced with actual model training
        # For now, return a mock model
        class MockModel:
            def parameters(self):
                return [torch.randn(100, 100)]
        
        return MockModel()
    
    def _default_eval_func(self, model, config: ExperimentConfig) -> Dict[str, float]:
        """Default evaluation function (placeholder)."""
        self.logger.warning("Using placeholder evaluation function - implement actual evaluation")
        
        # This would be replaced with actual model evaluation
        # For now, return mock metrics
        return {
            'accuracy': np.random.uniform(0.7, 0.9),
            'perplexity': np.random.uniform(2.0, 4.0),
            'loss': np.random.uniform(0.5, 1.5)
        }


def create_nsrpo_ablation_study(
    base_config: ExperimentConfig,
    output_dir: str = "./nsrpo_ablation"
) -> AblationStudyFramework:
    """
    Create ablation study framework specifically for NSRPO models.
    
    Args:
        base_config: Base NSRPO experiment configuration
        output_dir: Output directory for ablation results
        
    Returns:
        Configured ablation study framework
    """
    return AblationStudyFramework(
        base_config=base_config,
        output_dir=output_dir
    )


if __name__ == "__main__":
    # Test the ablation study framework
    print("Testing Ablation Study Framework...")
    
    # Create mock base configuration
    from ..config import ExperimentConfig, ModelConfig
    
    base_config = ExperimentConfig(
        experiment_name="test_nsrpo",
        model=ModelConfig(
            base_model_path="microsoft/DialoGPT-medium",
            use_null_decoder=True,
            alpha_1=0.1,
            alpha_2=0.1,
            alpha_3=0.05
        )
    )
    
    # Create ablation study
    ablation_study = create_nsrpo_ablation_study(base_config)
    
    # Test configuration generation
    configs = ablation_study.generate_ablation_configs(
        components=["alpha_1_ce_loss", "decoder_layers"],
        include_baseline=True
    )
    
    print(f"✓ Generated {len(configs)} ablation configurations")
    
    # Test component analysis (with mock results)
    mock_results = [
        AblationResult("alpha_1_ce_loss", 0.0, {"accuracy": 0.75, "perplexity": 3.2}),
        AblationResult("alpha_1_ce_loss", 0.1, {"accuracy": 0.82, "perplexity": 2.8}),
        AblationResult("alpha_1_ce_loss", 0.2, {"accuracy": 0.80, "perplexity": 2.9}),
        AblationResult("baseline", "full_model", {"accuracy": 0.81, "perplexity": 2.85})
    ]
    
    analyses = ablation_study._analyze_ablation_results(mock_results)
    print(f"✓ Generated {len(analyses)} component analyses")
    
    # Test summary generation
    summary = ablation_study._generate_ablation_summary(mock_results, analyses)
    print(f"✓ Generated ablation study summary with {len(summary['key_findings'])} key findings")
    
    print("✓ Ablation study framework test completed successfully!")
    print("  Features implemented:")
    print("  - Standard NSRPO component definitions")
    print("  - Automatic configuration generation")
    print("  - Statistical importance analysis")
    print("  - Component ranking and recommendations")
    print("  - Comprehensive result saving and reporting")
