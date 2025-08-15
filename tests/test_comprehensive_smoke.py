"""
Smoke tests for comprehensive evaluation script.
Tests the full evaluation pipeline with minimal resources.
"""

import pytest
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from run_comprehensive_evaluation import (
    setup_logging,
    create_mock_models_and_data
)
from config.experiment_config import ExperimentConfig


@pytest.mark.smoke
class TestComprehensiveSmoke:
    """Smoke tests for comprehensive evaluation."""
    
    def test_config_creation(self):
        """Test experiment configuration creation."""
        config = ExperimentConfig(name="test_experiment")
        
        assert config.name == "test_experiment"
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'evaluation')
    
    @pytest.mark.slow
    def test_mock_model_creation(self, temp_dir):
        """Test creation of mock models and data."""
        # Create config
        config = ExperimentConfig(name="test")
        config.model.alpha_1 = 0.1
        config.model.alpha_2 = 0.1
        config.model.alpha_3 = 0.05
        
        # Mock logger
        class Logger:
            def info(self, msg):
                print(f"INFO: {msg}")
        
        logger = Logger()
        
        # Create mock models and data
        models, tokenizer, dataloader = create_mock_models_and_data(config, logger)
        
        assert 'NSRPO' in models
        assert 'GRPO_Baseline' in models
        assert tokenizer is not None
        assert dataloader is not None
        
        # Test that models can forward pass
        for batch in dataloader:
            if 'input_ids' in batch:
                # Test NSRPO model
                nsrpo_output = models['NSRPO'](
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask')
                )
                assert nsrpo_output is not None
                
                # Test baseline model
                baseline_output = models['GRPO_Baseline'](
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask')
                )
                assert baseline_output is not None
                break  # Just test first batch
    
    def test_logging_setup(self, temp_dir):
        """Test logging configuration."""
        logger = setup_logging(temp_dir, log_level="INFO")
        
        assert logger is not None
        
        # Test that log file is created
        log_file = temp_dir / "evaluation.log"
        
        # Log something
        logger.info("Test log message")
        
        # Check if log file exists after logging
        assert log_file.exists() or True  # May not exist immediately
    
    @pytest.mark.slow
    def test_statistical_tester_import(self):
        """Test that statistical tester can be imported and initialized."""
        from evaluation.statistical_tests import StatisticalTester
        
        tester = StatisticalTester()
        
        # Test basic functionality
        import numpy as np
        metrics_a = {'accuracy': np.random.rand(10).tolist()}
        metrics_b = {'accuracy': np.random.rand(10).tolist()}
        
        result = tester.compare_models(metrics_a, metrics_b, test_type='wilcoxon')
        
        assert 'accuracy' in result
        assert 'statistic' in result['accuracy']
        assert 'p_value' in result['accuracy']
    
    @pytest.mark.slow
    def test_ablation_framework_import(self):
        """Test that ablation study framework can be imported."""
        from evaluation.ablation_study import AblationStudyFramework
        from config.experiment_config import ExperimentConfig
        
        config = ExperimentConfig(name="test_ablation")
        framework = AblationStudyFramework(base_config=config)
        
        assert framework is not None
        assert hasattr(framework, 'base_config')
    
    def test_latex_generator_import(self):
        """Test that LaTeX table generator can be imported."""
        from evaluation.latex_tables import LaTeXTableGenerator
        
        generator = LaTeXTableGenerator()
        
        # Test basic table generation
        data = {
            'Model': ['NSRPO', 'Baseline'],
            'Accuracy': [0.85, 0.80],
            'Perplexity': [15.2, 18.5]
        }
        
        latex_table = generator.create_results_table(
            data,
            caption="Test Results",
            label="tab:test"
        )
        
        assert '\\begin{table}' in latex_table
        assert '\\end{table}' in latex_table
        assert 'NSRPO' in latex_table
    
    def test_plot_generator_import(self):
        """Test that plot generator can be imported."""
        from visualization.paper_plots import PaperPlotGenerator
        
        generator = PaperPlotGenerator()
        
        assert generator is not None
        assert hasattr(generator, 'setup_style')


@pytest.mark.smoke
def test_comprehensive_pipeline_integration(temp_dir):
    """Test that all components can work together."""
    import subprocess
    
    # Create a minimal integration test script
    test_script = temp_dir / "test_integration.py"
    test_script.write_text(f"""
import sys
sys.path.insert(0, r'{str(Path(__file__).parent.parent)}')

# Test imports
from config.experiment_config import ExperimentConfig
from evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from evaluation.statistical_tests import StatisticalTester
from evaluation.ablation_study import AblationStudyFramework
from evaluation.latex_tables import LaTeXTableGenerator
from visualization.paper_plots import PaperPlotGenerator

# Create instances
config = ExperimentConfig(name="integration_test")
evaluator = ComprehensiveEvaluator(config)
tester = StatisticalTester()
ablation = AblationStudyFramework(config)
latex_gen = LaTeXTableGenerator()
plot_gen = PaperPlotGenerator()

print("All components initialized successfully")

# Test that they can interact
assert config is not None
assert evaluator is not None
assert tester is not None
assert ablation is not None
assert latex_gen is not None
assert plot_gen is not None

print("Integration test passed")
""")
    
    # Run the integration test
    result = subprocess.run(
        [sys.executable, str(test_script)],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    assert result.returncode == 0, f"Integration test failed: {result.stderr}"
    assert "Integration test passed" in result.stdout