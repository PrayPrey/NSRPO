#!/usr/bin/env python3
"""
Dependency Verification Script for NSRPO
Tests all package imports and validates installation integrity.
"""

import sys
import warnings
from typing import List, Tuple, Dict
import platform

def print_system_info():
    """Print system information."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print()

def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """
    Test if a module can be imported.
    
    Args:
        module_name: Name of the module to import
        package_name: Optional package name if different from module name
        
    Returns:
        Tuple of (success, version_or_error)
    """
    if package_name is None:
        package_name = module_name
        
    try:
        module = __import__(module_name)
        
        # Try to get version
        version = "Unknown"
        for attr in ['__version__', 'VERSION', 'version']:
            if hasattr(module, attr):
                version = getattr(module, attr)
                if callable(version):
                    version = version()
                break
        
        # Special cases
        if module_name == 'torch':
            import torch
            version = torch.__version__
            cuda_available = torch.cuda.is_available()
            device_info = f" (CUDA: {'Available' if cuda_available else 'Not Available'})"
            version += device_info
            
        return True, str(version)
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error: {str(e)}"

def verify_core_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Verify core dependencies."""
    core_deps = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('accelerate', 'Accelerate'),
        ('numpy', 'NumPy'),
        ('sklearn', 'scikit-learn'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('statsmodels', 'StatsModels'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
        ('tensorboard', 'TensorBoard'),
        ('json5', 'JSON5'),
    ]
    
    results = {}
    for module_name, display_name in core_deps:
        success, version = test_import(module_name)
        results[display_name] = (success, version)
    
    return results

def verify_dev_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Verify development dependencies."""
    dev_deps = [
        ('pytest', 'pytest'),
        ('black', 'Black'),
        ('isort', 'isort'),
    ]
    
    results = {}
    for module_name, display_name in dev_deps:
        success, version = test_import(module_name)
        results[display_name] = (success, version)
    
    return results

def verify_project_modules() -> Dict[str, Tuple[bool, str]]:
    """Verify that project modules can be imported."""
    project_modules = [
        'models.nsrpo_model',
        'models.null_decoder',
        'models.losses',
        'utils.dataset',
        'utils.svd_utils',
        'evaluation.comprehensive_evaluator',
        'evaluation.ablation_study',
        'evaluation.statistical_tests',
        'evaluation.latex_tables',
        'visualization.paper_plots',
        'config.experiment_config',
    ]
    
    results = {}
    for module_path in project_modules:
        try:
            # Add current directory to path if needed
            import os
            if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            module = __import__(module_path, fromlist=[''])
            results[module_path] = (True, "Successfully imported")
        except ImportError as e:
            results[module_path] = (False, str(e))
        except Exception as e:
            results[module_path] = (False, f"Error: {str(e)}")
    
    return results

def print_results(title: str, results: Dict[str, Tuple[bool, str]]):
    """Print verification results."""
    print("=" * 60)
    print(title)
    print("=" * 60)
    
    success_count = sum(1 for success, _ in results.values() if success)
    total_count = len(results)
    
    for name, (success, info) in results.items():
        status = "✓" if success else "✗"
        color = "\033[92m" if success else "\033[91m"  # Green if success, red if failure
        reset = "\033[0m"
        
        if success:
            print(f"{color}{status}{reset} {name:<25} {info}")
        else:
            print(f"{color}{status}{reset} {name:<25} FAILED: {info}")
    
    print()
    print(f"Summary: {success_count}/{total_count} passed")
    
    if success_count < total_count:
        print(f"\033[93mWarning: {total_count - success_count} dependencies failed!\033[0m")
    else:
        print(f"\033[92mAll dependencies successfully verified!{reset}")
    print()
    
    return success_count == total_count

def test_torch_functionality():
    """Test basic PyTorch functionality."""
    print("=" * 60)
    print("PYTORCH FUNCTIONALITY TEST")
    print("=" * 60)
    
    try:
        import torch
        
        # Test tensor creation
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"✓ Tensor creation successful: {x}")
        
        # Test basic operations
        y = x * 2
        print(f"✓ Basic operations successful: {y}")
        
        # Test device availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Default device: {device}")
        
        # Test model creation
        model = torch.nn.Linear(10, 5)
        print(f"✓ Model creation successful: Linear(10, 5)")
        
        # Test gradient computation
        x = torch.randn(1, 10, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        print(f"✓ Gradient computation successful")
        
        print("\033[92mAll PyTorch functionality tests passed!\033[0m")
        return True
        
    except Exception as e:
        print(f"\033[91m✗ PyTorch functionality test failed: {str(e)}\033[0m")
        return False
    
    print()

def main():
    """Main verification function."""
    print("\n" + "=" * 60)
    print("NSRPO INSTALLATION VERIFICATION")
    print("=" * 60 + "\n")
    
    # Print system info
    print_system_info()
    
    # Verify core dependencies
    core_results = verify_core_dependencies()
    core_success = print_results("CORE DEPENDENCIES", core_results)
    
    # Verify development dependencies
    dev_results = verify_dev_dependencies()
    dev_success = print_results("DEVELOPMENT DEPENDENCIES (Optional)", dev_results)
    
    # Verify project modules
    project_results = verify_project_modules()
    project_success = print_results("PROJECT MODULES", project_results)
    
    # Test PyTorch functionality
    torch_success = test_torch_functionality()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    if core_success and project_success:
        print("\033[92m✓ Installation verification PASSED!\033[0m")
        print("All core dependencies and project modules are working correctly.")
        
        if not dev_success:
            print("\033[93m⚠ Development dependencies are optional.\033[0m")
            print("Install with: pip install -r requirements-dev.txt")
            
        return 0
    else:
        print("\033[91m✗ Installation verification FAILED!\033[0m")
        print("Please check the errors above and install missing dependencies.")
        print("\nSuggested fixes:")
        print("1. For core dependencies: pip install -r requirements.txt")
        print("2. For CPU-only PyTorch: pip install -r requirements-cpu.txt")
        print("3. For development tools: pip install -r requirements-dev.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())