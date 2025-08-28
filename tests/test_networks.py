"""Tests for neural network components."""

import pytest
import torch
import numpy as np
from networks import Net_FFN, Net_timegrid


class TestNetFFN:
    """Test suite for Net_FFN class."""
    
    def test_initialization(self):
        """Test proper network initialization."""
        model = Net_FFN(dim=2, nOut=1, n_layers=3, vNetWidth=10)
        assert len(model.h_h) == 2  # n_layers - 1
        assert model.dim == 2
        assert model.nOut == 1
    
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
            assert y.shape == (10, 1)
    
    def test_output_activations(self):
        """Test different output activation functions."""
        output_activations = ['id', 'softplus']
        for out_act in output_activations:
            model = Net_FFN(dim=2, nOut=1, n_layers=2, vNetWidth=5, activation_output=out_act)
            x = torch.randn(10, 2)
            y = model(x)
            
            if out_act == 'softplus':
                assert torch.all(y >= 0)  # Softplus ensures non-negative output
    
    def test_invalid_activation(self):
        """Test error handling for invalid activation functions."""
        with pytest.raises(ValueError):
            Net_FFN(dim=2, nOut=1, n_layers=2, vNetWidth=5, activation='invalid')
        
        with pytest.raises(ValueError):
            Net_FFN(dim=2, nOut=1, n_layers=2, vNetWidth=5, activation_output='invalid')


class TestNetTimegrid:
    """Test suite for Net_timegrid class."""
    
    def test_initialization(self):
        """Test proper initialization."""
        model = Net_timegrid(dim=2, nOut=1, n_layers=2, vNetWidth=5, n_maturities=3)
        assert len(model.net_t) == 3  # One network per maturity
        assert model.dim == 2
        assert model.nOut == 1
    
    def test_forward_idx(self):
        """Test forward pass with specific index."""
        model = Net_timegrid(dim=2, nOut=1, n_layers=2, vNetWidth=5, n_maturities=3)
        x = torch.randn(10, 2)
        
        for idx in range(3):
            y = model.forward_idx(idx, x)
            assert y.shape == (10, 1)
            assert not torch.isnan(y).any()
    
    def test_freeze_unfreeze(self):
        """Test parameter freezing functionality."""
        model = Net_timegrid(dim=2, nOut=1, n_layers=2, vNetWidth=5, n_maturities=3)
        
        # Test freeze all
        model.freeze()
        for param in model.parameters():
            assert not param.requires_grad
        
        # Test unfreeze all
        model.unfreeze()
        for param in model.parameters():
            assert param.requires_grad
        
        # Test selective freeze
        model.freeze(0, 1)
        frozen_count = sum(1 for param in model.net_t[0].parameters() if not param.requires_grad)
        unfrozen_count = sum(1 for param in model.net_t[2].parameters() if param.requires_grad)
        assert frozen_count > 0
        assert unfrozen_count > 0
    
    def test_different_activations(self):
        """Test different activation functions."""
        activations = ['relu', 'silu', 'tanh']
        for act in activations:
            model = Net_timegrid(
                dim=2, nOut=1, n_layers=2, vNetWidth=5, 
                n_maturities=2, activation=act
            )
            x = torch.randn(5, 2)
            y = model.forward_idx(0, x)
            assert not torch.isnan(y).any()


class TestModelIntegration:
    """Integration tests for complete models."""
    
    def test_net_lv_creation(self):
        """Test Net_LV model creation."""
        # This test requires importing Net_LV
        try:
            from nsde_LV import Net_LV
            
            device = "cpu"
            timegrid = torch.linspace(0, 1, 10)
            strikes_call = np.arange(0.9, 1.11, 0.1)
            maturities = [3, 6, 9]
            
            model = Net_LV(
                dim=1,
                timegrid=timegrid,
                strikes_call=strikes_call,
                n_layers=2,
                vNetWidth=5,
                device=device,
                rate=0.025,
                maturities=maturities,
                n_maturities=len(maturities)
            )
            
            assert model is not None
            assert hasattr(model, 'diffusion')
            assert hasattr(model, 'control_variate_vanilla')
            
        except ImportError:
            pytest.skip("Net_LV not available for testing")


if __name__ == "__main__":
    pytest.main([__file__])
