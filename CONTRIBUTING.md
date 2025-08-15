# Contributing to NSPO

Thank you for your interest in contributing to NSPO! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## How to Contribute

### Reporting Issues

1. Check existing issues to avoid duplicates
2. Use issue templates when available
3. Include:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, PyTorch version)
   - Error messages and stack traces

### Suggesting Enhancements

1. Check if the enhancement has already been suggested
2. Open a discussion or issue with:
   - Clear description of the enhancement
   - Use cases and benefits
   - Potential implementation approach

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest tests/`
6. Format your code: `black . && isort .`
7. Commit with descriptive messages
8. Push to your fork
9. Open a pull request

## Development Setup

### Setting Up Your Environment

```bash
# Clone your fork
git clone https://github.com/yourusername/NSPO.git
cd NSPO

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional)
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_train_smoke.py

# Run only smoke tests
pytest tests/ -m smoke
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint code
flake8 .

# Type checking (if configured)
mypy .
```

## Coding Standards

### Python Style

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Write docstrings for all public functions/classes
- Add type hints where beneficial

### Documentation

- Update README.md for user-facing changes
- Add docstrings following Google style
- Include examples in docstrings
- Update CHANGELOG.md

### Testing

- Write tests for new features
- Maintain or improve code coverage
- Use pytest fixtures for common test data
- Mark slow tests with `@pytest.mark.slow`
- Add smoke tests for critical paths

### Commit Messages

Follow conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

Examples:
```
feat(model): add null-space projection layer
fix(train): resolve memory leak in dataloader
docs: update installation instructions
test: add smoke tests for evaluation pipeline
```

## Project Structure

```
NSPO/
├── models/          # Model implementations
├── utils/           # Utility functions
├── evaluation/      # Evaluation framework
├── visualization/   # Plotting utilities
├── config/          # Configuration
├── tests/           # Test files
├── docs/            # Documentation
└── scripts/         # Helper scripts
```

## Adding New Features

1. **Discuss First**: For major changes, open an issue for discussion
2. **Design**: Consider the architecture and integration
3. **Implement**: Follow existing patterns and conventions
4. **Test**: Add comprehensive tests
5. **Document**: Update relevant documentation
6. **Benchmark**: For performance-critical code, add benchmarks

## Testing Guidelines

### Test Categories

- **Unit Tests**: Test individual functions/methods
- **Integration Tests**: Test component interactions
- **Smoke Tests**: Quick validation tests
- **End-to-End Tests**: Full pipeline tests

### Writing Tests

```python
# Example test structure
import pytest
from module import function_to_test

class TestFeature:
    def test_normal_case(self):
        """Test normal operation."""
        result = function_to_test(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge cases."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
    
    @pytest.mark.slow
    def test_performance(self):
        """Test performance-critical paths."""
        # Performance test implementation
```

## Documentation

### Docstring Format

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description of the function.
    
    Longer description if needed, explaining the purpose,
    algorithm, or important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input is provided
        
    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
    """
```

## Release Process

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Run full test suite
4. Create release PR
5. Tag release after merge
6. Deploy to PyPI (maintainers only)

## Getting Help

- Open an issue for bugs or questions
- Join discussions for design decisions
- Check existing documentation
- Ask in pull request comments

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to NSPO!