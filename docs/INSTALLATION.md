# Installation Guide

This guide provides detailed installation instructions for the Neural SDE pricing and hedging framework.

## 📋 Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **Memory**: At least 8GB RAM (16GB+ recommended for training)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Python Environment

We recommend using a virtual environment to avoid dependency conflicts:

```bash
# Using venv (recommended)
python3 -m venv neural-sde-env
source neural-sde-env/bin/activate  # On Windows: neural-sde-env\Scripts\activate

# Or using conda
conda create -n neural-sde python=3.9
conda activate neural-sde
```

## 🚀 Installation Methods

### Method 1: From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/NeuralSDE_pricing_hedging.git
cd NeuralSDE_pricing_hedging

# Install dependencies
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .
```

### Method 2: Using pip (when available on PyPI)

```bash
# Install from PyPI
pip install neural-sde-pricing

# With optional dependencies
pip install neural-sde-pricing[dev,notebooks,advanced]
```

### Method 3: Using conda (when available)

```bash
# Install from conda-forge
conda install -c conda-forge neural-sde-pricing
```

## 🔧 Dependency Details

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥1.9.0 | Deep learning framework |
| NumPy | ≥1.21.0 | Numerical computations |
| SciPy | ≥1.7.0 | Scientific computing |
| Matplotlib | ≥3.4.0 | Plotting and visualization |
| Pandas | ≥1.3.0 | Data manipulation |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| geomloss | ≥0.2.3 | Wasserstein distance calculations |
| jupyter | ≥1.0.0 | Interactive notebooks |
| black | ≥21.0.0 | Code formatting |
| pytest | ≥6.0.0 | Testing framework |

## 🖥 GPU Setup

### CUDA Installation

For GPU acceleration, install CUDA-compatible PyTorch:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Verify GPU Setup

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

## ✅ Verification

### Test Installation

```bash
# Run quick start example
python examples/quick_start.py

# Run tests (if pytest is installed)
pytest tests/

# Import main modules
python -c "from networks import Net_FFN; print('✅ Installation successful!')"
```

### Common Issues and Solutions

#### Issue 1: ModuleNotFoundError

```bash
# Error: No module named 'torch'
# Solution: Install PyTorch
pip install torch

# Error: No module named 'networks'
# Solution: Add project to Python path or install in development mode
pip install -e .
```

#### Issue 2: CUDA Errors

```bash
# Error: CUDA out of memory
# Solution: Reduce batch size in training scripts
python nsde_LV.py --batch_size 20000

# Error: CUDA version mismatch
# Solution: Install compatible PyTorch version
pip install torch --upgrade
```

#### Issue 3: Import Errors

```bash
# Error: cannot import name 'Net_LV'
# Solution: Check file paths and ensure proper installation
cd /path/to/NeuralSDE_pricing_hedging
python -c "from nsde_LV import Net_LV"
```

## 🐳 Docker Installation (Optional)

For containerized deployment:

```dockerfile
# Dockerfile example
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "examples/quick_start.py"]
```

```bash
# Build and run
docker build -t neural-sde .
docker run --gpus all neural-sde
```

## 🔄 Development Installation

For contributing to the project:

```bash
# Clone with development setup
git clone https://github.com/yourusername/NeuralSDE_pricing_hedging.git
cd NeuralSDE_pricing_hedging

# Install with development dependencies
pip install -r requirements.txt
pip install black flake8 pytest pytest-cov mypy

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Verify development setup
pre-commit run --all-files
pytest tests/
```

## 🌐 Platform-Specific Notes

### Linux/Ubuntu

```bash
# Install system dependencies
sudo apt update
sudo apt install python3-dev python3-pip

# For GPU support
sudo apt install nvidia-cuda-toolkit
```

### macOS

```bash
# Using Homebrew
brew install python

# For Apple Silicon Macs
pip install torch torchvision torchaudio
```

### Windows

```bash
# Using Anaconda (recommended)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Or using pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🆘 Getting Help

If you encounter installation issues:

1. **Check Requirements**: Ensure you meet all system requirements
2. **Update pip**: Run `pip install --upgrade pip`
3. **Virtual Environment**: Try installing in a fresh virtual environment
4. **Search Issues**: Check [GitHub Issues](https://github.com/yourusername/NeuralSDE_pricing_hedging/issues) for similar problems
5. **Create Issue**: If problem persists, create a new issue with details

## 📚 Next Steps

After successful installation:

1. **Run Examples**: Try `python examples/quick_start.py`
2. **Read Documentation**: Check `docs/EXAMPLES.md` for tutorials
3. **Explore Notebooks**: Open `black_sholes_advanced_analysis/` notebooks
4. **Train Models**: Start with basic LV model training

---

**Note**: This project is primarily designed for research and educational purposes. For production use, please validate all results independently.
