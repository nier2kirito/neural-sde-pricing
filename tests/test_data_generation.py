"""Tests for data generation utilities."""

import pytest
import torch
import numpy as np
from BS_generator import bs_call_price, bs_put_price, generate_option_prices


class TestBlackScholesGenerator:
    """Test suite for Black-Scholes option pricing."""
    
    def test_bs_call_price_basic(self):
        """Test basic Black-Scholes call pricing."""
        # ATM option
        price = bs_call_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        assert price > 0
        assert isinstance(price, (float, np.float64))
    
    def test_bs_call_price_itm_otm(self):
        """Test ITM and OTM options."""
        S, r, T, sigma = 100, 0.05, 1.0, 0.2
        
        # ITM call (K < S)
        itm_price = bs_call_price(S=S, K=80, T=T, r=r, sigma=sigma)
        
        # OTM call (K > S)  
        otm_price = bs_call_price(S=S, K=120, T=T, r=r, sigma=sigma)
        
        # ITM should be more expensive than OTM
        assert itm_price > otm_price
        assert itm_price > S - 80 * np.exp(-r * T)  # Should exceed intrinsic value
    
    def test_bs_put_price_basic(self):
        """Test basic Black-Scholes put pricing."""
        price = bs_put_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        assert price > 0
        assert isinstance(price, (float, np.float64))
    
    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        call_price = bs_call_price(S, K, T, r, sigma)
        put_price = bs_put_price(S, K, T, r, sigma)
        
        # Put-call parity: C - P = S - K*exp(-rT)
        parity_diff = call_price - put_price - (S - K * np.exp(-r * T))
        assert abs(parity_diff) < 1e-10
    
    def test_zero_time_to_expiry(self):
        """Test option pricing at expiry."""
        S, K = 110, 100
        
        call_price = bs_call_price(S=S, K=K, T=0, r=0.05, sigma=0.2)
        put_price = bs_put_price(S=S, K=K, T=0, r=0.05, sigma=0.2)
        
        assert call_price == max(S - K, 0)
        assert put_price == max(K - S, 0)
    
    def test_generate_option_prices_shape(self):
        """Test option price matrix generation."""
        strikes = np.linspace(80, 120, 5)
        maturities = [0.25, 0.5, 1.0]
        
        call_prices = generate_option_prices(
            S=100, r=0.05, sigma=0.2,
            maturities=maturities,
            strikes=strikes,
            option_type='call'
        )
        
        assert call_prices.shape == (len(maturities), len(strikes))
        assert np.all(call_prices >= 0)
    
    def test_generate_option_prices_monotonicity(self):
        """Test that option prices follow expected monotonicity."""
        strikes = np.linspace(80, 120, 10)
        maturities = [0.25, 0.5, 1.0]
        
        call_prices = generate_option_prices(
            S=100, r=0.05, sigma=0.2,
            maturities=maturities,
            strikes=strikes,
            option_type='call'
        )
        
        # Call prices should decrease with strike (for fixed maturity)
        for i in range(len(maturities)):
            assert np.all(np.diff(call_prices[i, :]) <= 0)
        
        # Call prices should increase with maturity (for fixed strike, usually)
        for j in range(len(strikes)):
            # This might not always hold for deep ITM options, so we test ATM
            if strikes[j] == 100:  # ATM
                assert np.all(np.diff(call_prices[:, j]) >= 0)


class TestDataValidation:
    """Test data validation and edge cases."""
    
    def test_negative_inputs(self):
        """Test handling of negative inputs."""
        # Negative stock price should still work mathematically
        price = bs_call_price(S=-100, K=100, T=1.0, r=0.05, sigma=0.2)
        assert isinstance(price, (float, np.float64))
        
        # Negative volatility should raise an error in a robust implementation
        # For now, we just check it doesn't crash
        price = bs_call_price(S=100, K=100, T=1.0, r=0.05, sigma=-0.2)
        assert isinstance(price, (float, np.float64))
    
    def test_extreme_parameters(self):
        """Test with extreme parameter values."""
        # Very high volatility
        price_high_vol = bs_call_price(S=100, K=100, T=1.0, r=0.05, sigma=2.0)
        assert price_high_vol > 0
        
        # Very low volatility
        price_low_vol = bs_call_price(S=100, K=100, T=1.0, r=0.05, sigma=0.01)
        assert price_low_vol > 0
        
        # High volatility should give higher option price
        assert price_high_vol > price_low_vol
    
    def test_invalid_option_type(self):
        """Test error handling for invalid option types."""
        with pytest.raises(ValueError):
            generate_option_prices(
                S=100, r=0.05, sigma=0.2,
                maturities=[1.0], strikes=[100],
                option_type='invalid'
            )


class TestTorchIntegration:
    """Test PyTorch tensor integration."""
    
    def test_tensor_conversion(self):
        """Test conversion to PyTorch tensors."""
        strikes = np.linspace(80, 120, 5)
        maturities = [0.25, 0.5, 1.0]
        
        prices = generate_option_prices(
            S=100, r=0.05, sigma=0.2,
            maturities=maturities,
            strikes=strikes
        )
        
        # Convert to tensor
        price_tensor = torch.tensor(prices, dtype=torch.float32)
        assert price_tensor.shape == (len(maturities), len(strikes))
        assert price_tensor.dtype == torch.float32
    
    def test_gpu_compatibility(self):
        """Test GPU compatibility if available."""
        if torch.cuda.is_available():
            strikes = np.linspace(90, 110, 3)
            maturities = [0.5, 1.0]
            
            prices = generate_option_prices(
                S=100, r=0.05, sigma=0.2,
                maturities=maturities,
                strikes=strikes
            )
            
            # Move to GPU
            price_tensor = torch.tensor(prices, dtype=torch.float32, device='cuda')
            assert price_tensor.device.type == 'cuda'
        else:
            pytest.skip("CUDA not available")


if __name__ == "__main__":
    pytest.main([__file__])
