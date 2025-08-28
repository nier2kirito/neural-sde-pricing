#!/usr/bin/env python3
"""
Quick Start Example for Neural SDE Framework

This script demonstrates basic usage of the Neural SDE framework for option pricing.
Run this script to get started quickly with a simple example.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from BS_generator import generate_option_prices, save_option_prices_to_file
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("📦 Please install dependencies with: pip install -r requirements.txt")
    DEPENDENCIES_AVAILABLE = False


def main():
    """Run a quick start example."""
    
    print("🚀 Neural SDE Quick Start Example")
    print("=" * 50)
    
    if not DEPENDENCIES_AVAILABLE:
        print("\n⚠️  Dependencies not installed. Please run:")
        print("   pip install -r requirements.txt")
        print("\nThen try again!")
        return
    
    # Check PyTorch and device availability
    print(f"PyTorch version: {torch.__version__}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print()
    
    # Generate some sample data
    print("📊 Generating Black-Scholes option prices...")
    
    # Market parameters
    S0 = 1.0  # Initial stock price
    r = 0.025  # Risk-free rate  
    sigma = 0.2  # Volatility
    strikes = np.arange(0.8, 1.21, 0.02)
    maturities = [0.25, 0.5, 0.75, 1.0]
    
    # Generate option prices
    call_prices = generate_option_prices(
        S=S0, r=r, sigma=sigma,
        maturities=maturities,
        strikes=strikes,
        option_type='call'
    )
    
    print(f"Generated {call_prices.shape[0]} x {call_prices.shape[1]} option prices")
    print(f"Price range: {call_prices.min():.4f} - {call_prices.max():.4f}")
    
    # Save generated data
    save_option_prices_to_file("examples/sample_bs_prices.pt", call_prices)
    
    # Visualize the data
    print("\n📈 Visualizing option prices...")
    
    plt.figure(figsize=(12, 8))
    
    for i, T in enumerate(maturities):
        plt.subplot(2, 2, i+1)
        plt.plot(strikes, call_prices[i, :], 'b-', linewidth=2, label=f'T={T}')
        plt.xlabel('Strike Price')
        plt.ylabel('Option Price')
        plt.title(f'Call Prices at Maturity T={T}')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('examples/sample_option_prices.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'examples/sample_option_prices.png'")
    
    # Display next steps
    print("\n✅ Quick start completed successfully!")
    print("\n🎯 Next steps:")
    print("1. Train a Neural SDE model:")
    print("   python nsde_LV.py --device 0 --vNetWidth 50 --n_layers 10")
    print()
    print("2. Explore the Jupyter notebooks:")
    print("   cd black_sholes_advanced_analysis/")
    print("   jupyter notebook")
    print()
    print("3. Read the documentation:")
    print("   - docs/EXAMPLES.md for detailed examples")
    print("   - docs/API.md for API reference")
    print()
    print("4. Check out different model variants:")
    print("   - nsde_LV_LSTM/ for LSTM-based models")
    print("   - nsde_LSV/ for Local Stochastic Volatility models")


if __name__ == "__main__":
    main()
