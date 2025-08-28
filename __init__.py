"""
Neural SDEs for Robust Pricing and Hedging

This package implements Neural Stochastic Differential Equations for option pricing
and hedging strategies as described in "Robust pricing and hedging via neural SDEs"
by Gierjatowicz et al. (2020).
"""

__version__ = "1.0.0"
__author__ = "Patryk Gierjatowicz, Marc Sabate-Vidales, David Šiška, Lukasz Szpruch, Žan Žurič"

# Import main classes
from .nsde_LV import Net_LV
from .networks import Net_FFN, Net_timegrid

__all__ = ["Net_LV", "Net_FFN", "Net_timegrid"]
