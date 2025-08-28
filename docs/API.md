# API Reference

This document provides detailed API documentation for the Neural SDE framework.

## 🧠 Core Models

### `Net_LV` Class

Local Volatility Neural SDE model for option pricing.

```python
class Net_LV(nn.Module):
    def __init__(self, dim, timegrid, strikes_call, n_layers, vNetWidth, device, rate, maturities, n_maturities)
```

#### Parameters

- **`dim`** (int): Dimension of the state space (typically 1 for single asset)
- **`timegrid`** (torch.Tensor): Time discretization grid
- **`strikes_call`** (array-like): Strike prices for vanilla options
- **`n_layers`** (int): Number of hidden layers in neural networks
- **`vNetWidth`** (int): Width of hidden layers
- **`device`** (str): PyTorch device ('cpu' or 'cuda:X')
- **`rate`** (float): Risk-free interest rate
- **`maturities`** (list): List of maturity time indices
- **`n_maturities`** (int): Number of different maturities

#### Methods

##### `forward(S0, z, MC_samples, ind_T, period_length=30)`

Forward pass for pricing options.

**Parameters:**
- `S0` (float): Initial stock price
- `z` (torch.Tensor): Random noise tensor for Monte Carlo simulation
- `MC_samples` (int): Number of Monte Carlo samples
- `ind_T` (int): Final time index
- `period_length` (int): Period length for time grouping

**Returns:**
- `price_vanilla_cv` (torch.Tensor): Vanilla option prices with control variates
- `var_price_vanilla_cv` (torch.Tensor): Variance of vanilla option prices
- `exotic_option_price` (torch.Tensor): Exotic option prices
- `exotic_price_mean` (torch.Tensor): Mean exotic option price
- `exotic_price_var` (torch.Tensor): Variance of exotic option price
- `error` (torch.Tensor): Hedging error

## 🏗 Neural Network Components

### `Net_FFN` Class

Feed-forward neural network with configurable activation functions.

```python
class Net_FFN(nn.Module):
    def __init__(self, dim, nOut, n_layers, vNetWidth, activation="relu", activation_output="id", batchnorm=False)
```

#### Parameters

- **`dim`** (int): Input dimension
- **`nOut`** (int): Output dimension
- **`n_layers`** (int): Number of hidden layers
- **`vNetWidth`** (int): Width of hidden layers
- **`activation`** (str): Activation function ('relu', 'silu', 'tanh')
- **`activation_output`** (str): Output activation ('id', 'softplus')
- **`batchnorm`** (bool): Whether to use batch normalization

### `Net_timegrid` Class

Time-dependent neural network with separate networks for different time periods.

```python
class Net_timegrid(nn.Module):
    def __init__(self, dim, nOut, n_layers, vNetWidth, n_maturities, activation="relu", activation_output="id")
```

#### Methods

##### `forward_idx(idnet, x)`

Forward pass for specific time index.

**Parameters:**
- `idnet` (int): Network index corresponding to time period
- `x` (torch.Tensor): Input tensor

**Returns:**
- `y` (torch.Tensor): Network output

##### `freeze(*args)` / `unfreeze(*args)`

Freeze or unfreeze network parameters for alternating optimization.

## 🎯 Training Functions

### `train_nsde(model, z_test, config)`

Main training function for Neural SDE models.

#### Parameters

- **`model`** (Net_LV): Neural SDE model to train
- **`z_test`** (torch.Tensor): Test data for validation
- **`config`** (dict): Training configuration dictionary

#### Configuration Dictionary

```python
config = {
    "batch_size": int,           # Batch size for training
    "n_epochs": int,             # Number of training epochs
    "maturities": list,          # List of maturity indices
    "n_maturities": int,         # Number of maturities
    "strikes_call": array,       # Strike prices
    "timegrid": torch.Tensor,    # Time discretization
    "n_steps": int,              # Number of time steps
    "target_data": torch.Tensor  # Target option prices
}
```

#### Returns

- **`model_best`** (Net_LV): Best model found during training

## 📊 Data Generation

### Black-Scholes Generator

#### `bs_call_price(S, K, T, r, sigma)`

Calculate Black-Scholes call option price.

**Parameters:**
- `S` (float): Current stock price
- `K` (float): Strike price
- `T` (float): Time to maturity
- `r` (float): Risk-free rate
- `sigma` (float): Volatility

**Returns:**
- `price` (float): Black-Scholes call option price

#### `generate_option_prices(S, r, sigma, maturities, strikes, option_type='call')`

Generate option prices for multiple strikes and maturities.

**Parameters:**
- `S` (float): Current stock price
- `r` (float): Risk-free rate
- `sigma` (float): Volatility
- `maturities` (list): List of maturity times
- `strikes` (list): List of strike prices
- `option_type` (str): 'call' or 'put'

**Returns:**
- `prices` (np.ndarray): Option prices matrix (maturities × strikes)

## 🔧 Utility Functions

### `init_weights(m)`

Initialize neural network weights using Xavier normal initialization.

**Parameters:**
- `m` (nn.Module): Neural network module to initialize

### Weight Initialization Strategies

```python
# Xavier Normal (default)
def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1.5)

# Kaiming Normal (for ReLU networks)
def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')

# Custom initialization
def init_weights_custom(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.1)
        nn.init.constant_(m.bias.data, 0.0)
```

## 📈 Advanced Models

### LSV Model Components

The Local Stochastic Volatility model includes additional components:

```python
# Drift function for volatility
self.driftV = Net_timegrid(
    dim=dim+2,  # (t, S, V)
    nOut=1,
    n_layers=n_layers,
    vNetWidth=vNetWidth,
    n_maturities=n_maturities
)

# Diffusion function for volatility
self.diffusionV = Net_timegrid(
    dim=dim+2,
    nOut=1,
    n_layers=n_layers,
    vNetWidth=vNetWidth,
    n_maturities=n_maturities,
    activation_output="softplus"
)

# Correlation parameter
self.rho = nn.Parameter(torch.tensor(0.0))
```

## 🎛 Configuration Options

### Training Configuration

```python
DEFAULT_CONFIG = {
    # Training parameters
    "batch_size": 40000,
    "n_epochs": 1000,
    "learning_rate_sde": 0.001,
    "learning_rate_cv": 0.001,
    
    # Model parameters
    "n_layers": 10,
    "vNetWidth": 50,
    "activation": "relu",
    
    # Market parameters
    "rate": 0.025,
    "S0": 1.0,
    
    # Discretization
    "n_steps": 96,
    "period_length": 16,
    
    # Optimization
    "gradient_clip_sde": 5.0,
    "gradient_clip_cv": 3.0,
    "scheduler_milestones": [500, 800],
    "scheduler_gamma": 0.2,
    
    # Early stopping
    "early_stop_threshold": 2e-5,
    "patience": 100,
    
    # Logging
    "log_interval": 10,
    "save_interval": 100,
    "validate_interval": 50
}
```

### Device Configuration

```python
def setup_device(device_id=0):
    """Set up PyTorch device configuration."""
    
    if torch.cuda.is_available() and device_id >= 0:
        device = f'cuda:{device_id}'
        torch.cuda.set_device(device_id)
        print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
    else:
        device = "cpu"
        print("Using CPU")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if device.startswith('cuda'):
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    return device
```

## 🔍 Error Handling

### Common Error Patterns

```python
class NeuralSDEError(Exception):
    """Base exception for Neural SDE errors."""
    pass

class TrainingError(NeuralSDEError):
    """Raised when training fails."""
    pass

class DataError(NeuralSDEError):
    """Raised when data is invalid."""
    pass

# Usage in code
try:
    model = train_nsde(model, z_test, config)
except TrainingError as e:
    print(f"Training failed: {e}")
    # Handle gracefully
```

## 📋 Type Hints

For better code documentation and IDE support:

```python
from typing import Dict, List, Tuple, Optional, Union
import torch
import numpy as np

def train_nsde(
    model: torch.nn.Module,
    z_test: torch.Tensor,
    config: Dict[str, Union[int, float, List, torch.Tensor]]
) -> torch.nn.Module:
    """Train Neural SDE model with type hints."""
    pass

def evaluate_model(
    model: torch.nn.Module,
    test_data: torch.Tensor,
    device: str
) -> Tuple[float, float, Optional[torch.Tensor]]:
    """Evaluate model performance with type hints."""
    pass
```

## 🧪 Testing Framework

### Unit Test Examples

```python
import pytest
import torch
from networks import Net_FFN, Net_timegrid

class TestNetFFN:
    """Test suite for Net_FFN class."""
    
    def test_initialization(self):
        """Test proper network initialization."""
        model = Net_FFN(dim=2, nOut=1, n_layers=3, vNetWidth=10)
        assert len(model.h_h) == 2  # n_layers - 1
    
    def test_forward_pass_shape(self):
        """Test output shape correctness."""
        model = Net_FFN(dim=2, nOut=3, n_layers=3, vNetWidth=10)
        x = torch.randn(100, 2)
        y = model(x)
        assert y.shape == (100, 3)
    
    def test_activation_functions(self):
        """Test different activation functions."""
        activations = ['relu', 'silu', 'tanh']
        for act in activations:
            model = Net_FFN(dim=2, nOut=1, n_layers=2, vNetWidth=5, activation=act)
            x = torch.randn(10, 2)
            y = model(x)
            assert not torch.isnan(y).any()

class TestNetTimegrid:
    """Test suite for Net_timegrid class."""
    
    def test_freeze_unfreeze(self):
        """Test parameter freezing functionality."""
        model = Net_timegrid(dim=2, nOut=1, n_layers=2, vNetWidth=5, n_maturities=3)
        
        # Test freeze
        model.freeze()
        for param in model.parameters():
            assert not param.requires_grad
        
        # Test unfreeze
        model.unfreeze()
        for param in model.parameters():
            assert param.requires_grad
```

---

This API reference provides comprehensive documentation for all major components. For implementation details, please refer to the source code and examples.
