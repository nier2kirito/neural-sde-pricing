# Contributing to Neural SDEs for Robust Pricing and Hedging

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Neural SDE pricing and hedging framework.

## 🎯 How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **Bug Reports**: Found a bug? Please report it!
- **Feature Requests**: Have an idea for improvement? Let us know!
- **Code Contributions**: Fix bugs, add features, or improve documentation
- **Documentation**: Help improve our docs, examples, or tutorials
- **Testing**: Add test cases or improve existing tests

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/yourusername/NeuralSDE_pricing_hedging.git
cd NeuralSDE_pricing_hedging
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install black flake8 pytest jupyter
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number
```

## 📝 Code Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

- **Line Length**: 88 characters (Black default)
- **Imports**: Group imports (standard library, third-party, local)
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Use type hints for function signatures

### Example Function

```python
import torch
import numpy as np
from typing import Tuple, Optional

def price_option(
    model: torch.nn.Module,
    S0: float,
    strikes: np.ndarray,
    maturities: list,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Price vanilla options using Neural SDE model.
    
    Args:
        model: Trained Neural SDE model
        S0: Initial stock price
        strikes: Array of strike prices
        maturities: List of maturity indices
        device: PyTorch device for computation
        
    Returns:
        Tuple of (option_prices, price_variances)
    """
    # Implementation here
    pass
```

### Neural Network Guidelines

- **Initialization**: Use appropriate weight initialization (Xavier/Kaiming)
- **Activation Functions**: Document choice of activation functions
- **Architecture**: Keep architectures modular and configurable
- **Memory Management**: Be mindful of GPU memory usage

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_networks.py

# Run with coverage
python -m pytest --cov=. tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names
- Test both success and failure cases

Example test:

```python
import pytest
import torch
from networks import Net_FFN

def test_net_ffn_forward_pass():
    """Test that Net_FFN produces correct output shape."""
    model = Net_FFN(dim=2, nOut=1, n_layers=3, vNetWidth=10)
    x = torch.randn(100, 2)
    output = model(x)
    
    assert output.shape == (100, 1)
    assert not torch.isnan(output).any()
```

## 📊 Experiments and Results

### Adding New Experiments

When adding new experiments:

1. **Create a new directory**: `nsde_LV_your_experiment/`
2. **Follow naming convention**: `nsde_LV_your_experiment_model.py`
3. **Include results directory**: `results/` with outputs
4. **Document parameters**: Add clear documentation of hyperparameters

### Result Standards

- **Save metrics**: Always save training/validation metrics
- **Reproducibility**: Set random seeds for reproducible results
- **Visualization**: Generate plots for key results
- **Comparison**: Compare against existing baselines

## 🐛 Bug Reports

When reporting bugs, please include:

- **Environment**: Python version, PyTorch version, OS
- **Reproduction Steps**: Clear steps to reproduce the issue
- **Expected Behavior**: What should have happened
- **Actual Behavior**: What actually happened
- **Code Sample**: Minimal code that reproduces the issue

## 💡 Feature Requests

For feature requests:

- **Use Case**: Describe the problem you're trying to solve
- **Proposed Solution**: Outline your proposed approach
- **Alternatives**: Consider alternative solutions
- **Impact**: Who would benefit from this feature?

## 📋 Pull Request Process

### Before Submitting

1. **Test Your Code**: Ensure all tests pass
2. **Format Code**: Run `black *.py` to format
3. **Check Style**: Run `flake8` for style issues
4. **Update Documentation**: Update relevant docs
5. **Add Tests**: Include tests for new functionality

### PR Guidelines

- **Clear Title**: Use descriptive PR titles
- **Description**: Explain what your PR does and why
- **Link Issues**: Reference related issues with `#issue-number`
- **Small PRs**: Keep PRs focused and reasonably sized
- **Review Ready**: Mark PR as ready for review when complete

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature  
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## 🔄 Development Workflow

### Typical Workflow

1. **Plan**: Discuss significant changes in issues first
2. **Develop**: Work in feature branches
3. **Test**: Ensure your changes work correctly
4. **Document**: Update relevant documentation
5. **Submit**: Create a pull request
6. **Iterate**: Address review feedback
7. **Merge**: Maintainer merges approved PRs

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add LSTM support for LSV model
fix: resolve gradient explosion in training
docs: update installation instructions  
test: add unit tests for Net_timegrid
refactor: simplify network initialization
```

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: Maintainers will review PRs and provide feedback

## 🏆 Recognition

Contributors will be recognized in:
- **README**: Contributors section
- **Releases**: Release notes mention contributors
- **Documentation**: Author attribution where appropriate

Thank you for contributing to Neural SDEs for Robust Pricing and Hedging! 🎉
