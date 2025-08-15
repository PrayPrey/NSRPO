"""
Simple tests without external dependencies to verify code structure.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test SVD utils import
        import utils.svd_utils
        print("+ SVD utils imported successfully")
        
        # Check that functions exist
        functions = [
            'extract_lora_null_basis',
            'extract_base_null_basis', 
            'extract_activation_null_basis',
            'filter_by_variance',
            'compute_orthogonality_score',
            'project_to_null_space'
        ]
        
        for func_name in functions:
            assert hasattr(utils.svd_utils, func_name), f"Missing function: {func_name}"
        
        print(f"+ All {len(functions)} SVD utility functions found")
        
    except ImportError as e:
        print(f"- SVD utils import failed: {e}")
        return False
    
    try:
        # Test dataset utils import
        import utils.dataset
        print("+ Dataset utils imported successfully")
        
        # Check that classes and functions exist
        items = [
            'RLDataset',
            'rl_collate_fn',
            'load_data',
            'get_dataloader',
            'create_dummy_data',
            'BatchSampler'
        ]
        
        for item_name in items:
            assert hasattr(utils.dataset, item_name), f"Missing item: {item_name}"
        
        print(f"+ All {len(items)} dataset utility items found")
        
    except ImportError as e:
        print(f"- Dataset utils import failed: {e}")
        return False
    
    return True


def test_code_structure():
    """Test basic code structure and syntax."""
    print("Testing code structure...")
    
    # Test that the files can be parsed without syntax errors
    import ast
    
    # Test SVD utils
    svd_file = Path(__file__).parent.parent / 'utils' / 'svd_utils.py'
    with open(svd_file, 'r', encoding='utf-8') as f:
        svd_code = f.read()
    
    try:
        ast.parse(svd_code)
        print("+ SVD utils syntax is valid")
    except SyntaxError as e:
        print(f"- SVD utils syntax error: {e}")
        return False
    
    # Test dataset utils
    dataset_file = Path(__file__).parent.parent / 'utils' / 'dataset.py'
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset_code = f.read()
    
    try:
        ast.parse(dataset_code)
        print("+ Dataset utils syntax is valid")
    except SyntaxError as e:
        print(f"- Dataset utils syntax error: {e}")
        return False
    
    return True


def test_docstrings():
    """Test that functions have proper docstrings."""
    print("Testing docstrings...")
    
    import utils.svd_utils as svd
    import utils.dataset as dataset
    
    # Test SVD function docstrings
    svd_functions = [
        svd.extract_lora_null_basis,
        svd.extract_base_null_basis,
        svd.filter_by_variance,
        svd.compute_orthogonality_score,
        svd.project_to_null_space
    ]
    
    for func in svd_functions:
        if not func.__doc__ or len(func.__doc__.strip()) < 20:
            print(f"- {func.__name__} missing or short docstring")
            return False
    
    print(f"+ {len(svd_functions)} SVD functions have docstrings")
    
    # Test dataset class/function docstrings
    dataset_items = [
        dataset.RLDataset,
        dataset.rl_collate_fn,
        dataset.load_data,
        dataset.get_dataloader,
        dataset.create_dummy_data
    ]
    
    for item in dataset_items:
        if not item.__doc__ or len(item.__doc__.strip()) < 20:
            print(f"- {item.__name__} missing or short docstring")
            return False
    
    print(f"+ {len(dataset_items)} dataset items have docstrings")
    
    return True


def run_simple_tests():
    """Run all simple tests."""
    print("=" * 50)
    print("Running Simple Structure Tests (No Dependencies)")
    print("=" * 50)
    
    tests = [
        test_code_structure,
        test_imports,
        test_docstrings
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"- {test_func.__name__} error: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Simple tests completed: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_simple_tests()
    print("\nNote: Full tests with PyTorch will require 'pip install torch transformers sklearn'")
    sys.exit(0 if success else 1)