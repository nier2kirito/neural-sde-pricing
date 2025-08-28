# Examples and Tutorials

This document provides detailed examples and tutorials for using the Neural SDE framework for option pricing and hedging.

## 📚 Table of Contents

1. [Basic Usage](#basic-usage)
2. [Model Comparison](#model-comparison)
3. [Custom Data Generation](#custom-data-generation)
4. [Advanced Training](#advanced-training)
5. [Jupyter Notebook Examples](#jupyter-notebook-examples)

## 🚀 Basic Usage

### Example 1: Training a Simple Local Volatility Model

```python
import torch
import numpy as np
from nsde_LV import Net_LV, train_nsde

# Set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define market parameters
S0 = 1.0  # Initial stock price
rate = 0.025  # Risk-free rate
strikes_call = np.arange(0.8, 1.21, 0.02)  # Strike prices
n_steps = 96  # Time discretization steps
timegrid = torch.linspace(0, 1, n_steps + 1).to(device)
maturities = list(range(16, 65, 16))  # Maturity indices

# Create model
model = Net_LV(
    dim=1,
    timegrid=timegrid,
    strikes_call=strikes_call,
    n_layers=10,
    vNetWidth=30,
    device=device,
    rate=rate,
    maturities=maturities,
    n_maturities=len(maturities)
)

# Load target data
target_data = torch.load("Call_prices_59.pt")

# Training configuration
config = {
    "batch_size": 20000,
    "n_epochs": 500,
    "maturities": maturities,
    "n_maturities": len(maturities),
    "strikes_call": strikes_call,
    "timegrid": timegrid,
    "n_steps": n_steps,
    "target_data": target_data
}

# Generate test data
MC_samples_test = 100000
z_test = torch.randn(MC_samples_test, n_steps, device=device)
z_test = torch.cat([z_test, -z_test], 0)  # Antithetic sampling

# Train the model
trained_model = train_nsde(model, z_test, config)
```

### Example 2: Evaluating Model Performance

```python
# Evaluate the trained model
with torch.no_grad():
    pred_prices, price_vars, exotic_price, exotic_mean, exotic_var, hedge_error = model(
        S0, z_test, z_test.shape[0], maturities[-1], period_length=16
    )

# Calculate pricing errors
target_prices = torch.tensor(target_data[:len(maturities), :len(strikes_call)], device=device)
pricing_error = torch.sqrt(torch.mean((pred_prices - target_prices)**2))

print(f"RMSE Pricing Error: {pricing_error:.6f}")
print(f"Exotic Option Price: {exotic_mean:.6f}")
print(f"Hedging Error (L2): {torch.mean(hedge_error**2):.6f}")
```

## 🔄 Model Comparison

### Example 3: Comparing Different Architectures

```python
import matplotlib.pyplot as plt
from collections import defaultdict

def compare_models():
    """Compare different Neural SDE architectures."""
    
    models_config = {
        'Standard NN': {
            'script': 'nsde_LV_nn/nsde_LV.py',
            'n_layers': 10,
            'activation': 'relu'
        },
        'LSTM': {
            'script': 'nsde_LV_LSTM/nsde_LV_LSTM.py',
            'n_layers': 5,
            'activation': 'tanh'
        },
        'Improved NN': {
            'script': 'nsde_LV_improved_nn/nsde_LV_improved.py',
            'n_layers': 15,
            'activation': 'silu'
        }
    }
    
    results = defaultdict(list)
    
    for model_name, config in models_config.items():
        # Train each model (pseudo-code)
        model = create_model(config)
        final_error = train_and_evaluate(model)
        results[model_name].append(final_error)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    for model_name, errors in results.items():
        plt.bar(model_name, np.mean(errors), yerr=np.std(errors))
    
    plt.ylabel('RMSE Pricing Error')
    plt.title('Model Architecture Comparison')
    plt.yscale('log')
    plt.show()

# Run comparison
compare_models()
```

## 📊 Custom Data Generation

### Example 4: Generate Black-Scholes Training Data

```python
from BS_generator import generate_option_prices, save_option_prices_to_file

# Define market parameters
S0 = 1.0
r = 0.025
sigma = 0.2
strikes = np.linspace(0.8, 1.2, 21)
maturities = [0.25, 0.5, 0.75, 1.0]

# Generate option prices
call_prices = generate_option_prices(
    S=S0, r=r, sigma=sigma, 
    maturities=maturities, 
    strikes=strikes, 
    option_type='call'
)

# Save to file
save_option_prices_to_file("BS_call_prices.pt", call_prices)

print(f"Generated {call_prices.shape[0]} x {call_prices.shape[1]} option prices")
print(f"Price range: {call_prices.min():.4f} - {call_prices.max():.4f}")
```

### Example 5: Generate Heston Model Data

```python
import torch
import numpy as np

def heston_monte_carlo(S0, V0, r, kappa, theta, sigma_v, rho, T, K, n_paths=100000, n_steps=100):
    """Generate Heston model option prices via Monte Carlo."""
    
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Initialize arrays
    S = np.full(n_paths, S0)
    V = np.full(n_paths, V0)
    
    # Simulate paths
    for i in range(n_steps):
        # Generate correlated random numbers
        Z1 = np.random.normal(0, 1, n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n_paths)
        
        # Update variance (Feller condition)
        V_new = V + kappa * (theta - np.maximum(V, 0)) * dt + sigma_v * np.sqrt(np.maximum(V, 0)) * sqrt_dt * Z2
        V = np.maximum(V_new, 0)  # Ensure non-negative variance
        
        # Update stock price
        S = S * np.exp((r - 0.5 * V) * dt + np.sqrt(V) * sqrt_dt * Z1)
    
    # Calculate option payoffs
    payoffs = np.maximum(S - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price

# Generate Heston prices for multiple strikes and maturities
heston_params = {
    'S0': 1.0, 'V0': 0.04, 'r': 0.025,
    'kappa': 2.0, 'theta': 0.04, 'sigma_v': 0.3, 'rho': -0.7
}

strikes = np.arange(0.8, 1.21, 0.02)
maturities = [0.25, 0.5, 0.75, 1.0]

heston_prices = np.zeros((len(maturities), len(strikes)))

for i, T in enumerate(maturities):
    for j, K in enumerate(strikes):
        heston_prices[i, j] = heston_monte_carlo(**heston_params, T=T, K=K)

torch.save(torch.tensor(heston_prices), "heston_call_prices.pt")
```

## 🎛 Advanced Training

### Example 6: Custom Loss Function

```python
import torch.nn as nn

class CustomLoss(nn.Module):
    """Custom loss combining MSE and variance penalty."""
    
    def __init__(self, mse_weight=1.0, var_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.var_weight = var_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred_prices, target_prices, price_variances):
        mse_loss = self.mse_loss(pred_prices, target_prices)
        var_penalty = torch.mean(price_variances)
        return self.mse_weight * mse_loss + self.var_weight * var_penalty

# Use in training
loss_fn = CustomLoss(mse_weight=1.0, var_weight=0.05)
```

### Example 7: Learning Rate Scheduling

```python
import torch.optim as optim

# Set up optimizers with different learning rates
optimizer_sde = optim.Adam(model.diffusion.parameters(), lr=0.001)
optimizer_cv = optim.Adam(
    list(model.control_variate_vanilla.parameters()) + 
    list(model.control_variate_exotics.parameters()), 
    lr=0.0005
)

# Learning rate schedulers
scheduler_sde = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_sde, mode='min', factor=0.5, patience=50, verbose=True
)

scheduler_cv = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_cv, T_max=100, eta_min=1e-6
)

# In training loop
for epoch in range(n_epochs):
    # ... training code ...
    
    # Update learning rates
    scheduler_sde.step(validation_loss)
    scheduler_cv.step()
```

## 📓 Jupyter Notebook Examples

### Running the Analysis Notebooks

```bash
# Navigate to analysis directory
cd black_sholes_advanced_analysis/

# Start Jupyter
jupyter notebook

# Open desired notebook:
# - BS_baseline.ipynb: Black-Scholes baseline analysis
# - BS_LSTM.ipynb: LSTM model comparison
# - CEV_baseline.ipynb: CEV model analysis
# - ML_models.ipynb: Machine learning model comparisons
```

### Example Notebook Cell

```python
# Cell 1: Setup
import torch
import numpy as np
import matplotlib.pyplot as plt
from nsde_LV import Net_LV

# Cell 2: Load and visualize target data
target_data = torch.load("../Call_prices_59.pt")
strikes = np.arange(0.8, 1.21, 0.02)
maturities = [0.25, 0.5, 0.75, 1.0]

plt.figure(figsize=(12, 8))
for i, T in enumerate(maturities):
    plt.subplot(2, 2, i+1)
    plt.plot(strikes, target_data[i, :len(strikes)])
    plt.title(f'Call Prices at T={T}')
    plt.xlabel('Strike')
    plt.ylabel('Option Price')
plt.tight_layout()
plt.show()

# Cell 3: Train model and compare
# ... training code ...

# Cell 4: Analyze results
# ... analysis code ...
```

## 🔧 Hyperparameter Tuning

### Example 8: Grid Search for Optimal Parameters

```python
import itertools
from sklearn.model_selection import ParameterGrid

# Define parameter grid
param_grid = {
    'n_layers': [5, 10, 15, 20],
    'vNetWidth': [20, 30, 50, 80],
    'batch_size': [20000, 40000, 60000],
    'learning_rate': [0.001, 0.0005, 0.0001]
}

def grid_search_neural_sde(param_grid, target_data):
    """Perform grid search for optimal hyperparameters."""
    
    best_params = None
    best_error = float('inf')
    results = []
    
    for params in ParameterGrid(param_grid):
        print(f"Testing parameters: {params}")
        
        # Create and train model
        model = Net_LV(
            dim=1,
            timegrid=timegrid,
            strikes_call=strikes_call,
            n_layers=params['n_layers'],
            vNetWidth=params['vNetWidth'],
            device=device,
            rate=0.025,
            maturities=maturities,
            n_maturities=len(maturities)
        )
        
        config = {
            "batch_size": params['batch_size'],
            "n_epochs": 200,  # Reduced for grid search
            "target_data": target_data
        }
        
        # Train and evaluate
        trained_model = train_nsde(model, z_test, config)
        final_error = evaluate_model(trained_model)
        
        results.append({
            'params': params,
            'error': final_error
        })
        
        if final_error < best_error:
            best_error = final_error
            best_params = params
    
    return best_params, best_error, results

# Run grid search
best_params, best_error, all_results = grid_search_neural_sde(param_grid, target_data)
print(f"Best parameters: {best_params}")
print(f"Best error: {best_error:.6f}")
```

## 📈 Visualization and Analysis

### Example 9: Plotting Training Progress

```python
import matplotlib.pyplot as plt

def plot_training_progress(training_errors, validation_errors):
    """Plot training and validation errors over epochs."""
    
    plt.figure(figsize=(12, 5))
    
    # Training error
    plt.subplot(1, 2, 1)
    plt.semilogy(training_errors, label='Training Error')
    plt.semilogy(validation_errors, label='Validation Error')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE Error')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    # Error improvement rate
    plt.subplot(1, 2, 2)
    improvement_rate = np.diff(np.log(training_errors))
    plt.plot(improvement_rate, label='Log Error Improvement Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Log(Error) Improvement')
    plt.title('Learning Rate Analysis')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Usage during training
training_errors = []
validation_errors = []

for epoch in range(n_epochs):
    # ... training code ...
    training_errors.append(train_loss.item())
    validation_errors.append(val_loss.item())
    
    if epoch % 100 == 0:
        plot_training_progress(training_errors, validation_errors)
```

### Example 10: Implied Volatility Surface Analysis

```python
from scipy.optimize import brentq
from BS_generator import bs_call_price

def implied_volatility(market_price, S, K, T, r):
    """Calculate implied volatility from market price."""
    
    def objective(sigma):
        return bs_call_price(S, K, T, r, sigma) - market_price
    
    try:
        iv = brentq(objective, 0.001, 5.0)
        return iv
    except ValueError:
        return np.nan

def plot_iv_surface(prices, strikes, maturities, S0=1.0, r=0.025):
    """Plot implied volatility surface."""
    
    # Calculate implied volatilities
    iv_surface = np.zeros_like(prices)
    
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            price = prices[i, j]
            iv_surface[i, j] = implied_volatility(price, S0, K, T, r)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    T_mesh, K_mesh = np.meshgrid(maturities, strikes, indexing='ij')
    
    surface = ax.plot_surface(
        T_mesh, K_mesh, iv_surface,
        cmap='viridis', alpha=0.8
    )
    
    ax.set_xlabel('Maturity')
    ax.set_ylabel('Strike')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Implied Volatility Surface')
    
    fig.colorbar(surface)
    plt.show()
    
    return iv_surface

# Generate and plot IV surface
target_data = torch.load("Call_prices_59.pt").numpy()
strikes = np.arange(0.8, 1.21, 0.02)
maturities = [0.25, 0.5, 0.75, 1.0]

iv_surface = plot_iv_surface(target_data, strikes, maturities)
```

## 🧠 Advanced Neural Network Configurations

### Example 11: Custom Network Architecture

```python
from networks import Net_timegrid, Net_FFN

class CustomNeuralSDE(nn.Module):
    """Custom Neural SDE with residual connections."""
    
    def __init__(self, dim, timegrid, strikes_call, n_layers, vNetWidth, device, rate, maturities, n_maturities):
        super().__init__()
        
        self.dim = dim
        self.timegrid = timegrid
        self.device = device
        self.rate = rate
        
        # Custom diffusion network with residual connections
        self.diffusion = ResidualNet_timegrid(
            dim=dim+1, 
            nOut=1, 
            n_layers=n_layers, 
            vNetWidth=vNetWidth,
            n_maturities=n_maturities,
            activation="silu",
            activation_output="softplus"
        )
        
        # Enhanced control variates
        self.control_variate_vanilla = AttentionNet_timegrid(
            dim=dim+1,
            nOut=len(strikes_call)*n_maturities,
            n_layers=5,
            vNetWidth=40,
            n_maturities=n_maturities
        )
    
    def forward(self, S0, z, MC_samples, ind_T, period_length=30):
        # Custom forward pass implementation
        # ... implementation details ...
        pass

class ResidualNet_timegrid(Net_timegrid):
    """Neural network with residual connections."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add residual connection layers
        self.residual_layers = nn.ModuleList([
            nn.Linear(kwargs['vNetWidth'], kwargs['vNetWidth'])
            for _ in range(kwargs['n_layers'] // 2)
        ])
```

## 🎯 Performance Optimization

### Example 12: Memory-Efficient Training

```python
def memory_efficient_training(model, config, checkpoint_interval=100):
    """Training with gradient checkpointing and mixed precision."""
    
    from torch.cuda.amp import GradScaler, autocast
    
    scaler = GradScaler()
    
    for epoch in range(config["n_epochs"]):
        for batch_idx in range(0, 20 * config["batch_size"], config["batch_size"]):
            
            # Generate batch
            batch_z = torch.randn(config["batch_size"], config["n_steps"], device=device)
            
            with autocast():  # Mixed precision
                pred, var, _, exotic_price, exotic_var, _ = model(
                    S0, batch_z, config["batch_size"], config["maturities"][-1]
                )
                
                loss = calculate_loss(pred, config["target_data"])
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Clear cache periodically
            if batch_idx % checkpoint_interval == 0:
                torch.cuda.empty_cache()
```

## 🔍 Debugging and Troubleshooting

### Example 13: Gradient Analysis

```python
def analyze_gradients(model, loss):
    """Analyze gradient flow for debugging."""
    
    total_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            print(f"{name}: grad_norm = {param_norm:.6f}")
    
    total_norm = total_norm ** (1. / 2)
    print(f"Total gradient norm: {total_norm:.6f}")
    
    # Check for gradient explosion/vanishing
    if total_norm > 10:
        print("⚠️  Warning: Gradient explosion detected!")
    elif total_norm < 1e-6:
        print("⚠️  Warning: Vanishing gradients detected!")

# Use during training
loss.backward()
analyze_gradients(model, loss)
optimizer.step()
```

## 📝 Best Practices

### Training Tips

1. **Start Small**: Begin with smaller networks and increase complexity
2. **Monitor Metrics**: Track both pricing accuracy and hedging efficiency
3. **Use Checkpoints**: Save model checkpoints regularly
4. **Validate Early**: Check validation error to detect overfitting
5. **Experiment Logging**: Keep detailed logs of hyperparameters and results

### Code Organization

1. **Modular Design**: Keep models, training, and evaluation separate
2. **Configuration Files**: Use config dictionaries or files
3. **Reproducibility**: Set random seeds for consistent results
4. **Documentation**: Document all hyperparameters and design choices

### Performance Tips

1. **Batch Size**: Use largest batch size that fits in GPU memory
2. **Data Loading**: Precompute and cache expensive operations
3. **Mixed Precision**: Use automatic mixed precision for speed
4. **Profiling**: Profile code to identify bottlenecks

## 🎓 Further Reading

- [Original Paper](https://arxiv.org/abs/2007.04154)
- [Neural SDEs Tutorial](https://github.com/patrick-kidger/diffrax)
- [Stochastic Calculus Background](https://web.stanford.edu/~shreve/StochasticCalculus/)
- [Option Pricing Theory](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)

---

For more examples and detailed tutorials, check the Jupyter notebooks in the `black_sholes_advanced_analysis/` directory.
