"""
Test module imports for NSRPO project.
Verifies all modules can be imported without errors.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.smoke
class TestImports:
    """Test that all project modules can be imported."""
    
    def test_import_models(self):
        """Test importing model modules."""
        try:
            from models import nsrpo_model
            from models import null_decoder
            from models import losses
            from models import NSRPOModel, NullDecoder, create_nsrpo_model
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import models: {e}")
    
    def test_import_utils(self):
        """Test importing utility modules."""
        try:
            from utils import dataset
            from utils import svd_utils
            from utils import paths
            from utils.dataset import get_dataloader, create_dummy_data
            from utils.svd_utils import extract_lora_null_basis, extract_base_null_basis
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import utils: {e}")
    
    def test_import_evaluation(self):
        """Test importing evaluation modules."""
        try:
            from evaluation import comprehensive_evaluator
            from evaluation import ablation_study
            from evaluation import statistical_tests
            from evaluation import latex_tables
            from evaluation.comprehensive_evaluator import ComprehensiveEvaluator
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import evaluation: {e}")
    
    def test_import_visualization(self):
        """Test importing visualization modules."""
        try:
            from visualization import paper_plots
            from visualization.paper_plots import PaperPlotGenerator
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import visualization: {e}")
    
    def test_import_config(self):
        """Test importing configuration modules."""
        try:
            from config import experiment_config
            from config.experiment_config import ExperimentConfig
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import config: {e}")
    
    def test_import_main_scripts(self):
        """Test importing main executable scripts."""
        try:
            import train
            import evaluate
            import run_comprehensive_evaluation
            import example_usage
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import main scripts: {e}")
    
    def test_verify_installation_script(self):
        """Test the installation verification script."""
        try:
            import verify_installation
            assert hasattr(verify_installation, 'main')
            assert hasattr(verify_installation, 'test_import')
            assert hasattr(verify_installation, 'verify_core_dependencies')
        except ImportError as e:
            pytest.fail(f"Failed to import verify_installation: {e}")


@pytest.mark.smoke
def test_torch_availability():
    """Test that PyTorch is available and functional."""
    import torch
    
    # Check PyTorch version
    assert hasattr(torch, '__version__')
    
    # Test basic tensor operations
    x = torch.tensor([1.0, 2.0, 3.0])
    y = x * 2
    assert y.tolist() == [2.0, 4.0, 6.0]
    
    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device is not None


@pytest.mark.smoke
def test_transformers_availability():
    """Test that Transformers library is available."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import transformers: {e}")