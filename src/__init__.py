"""
Physics-Informed Machine Learning for Solar Cell Materials

A framework for accelerated discovery of photovoltaic materials using
physics-based descriptors and constrained machine learning.

Author: Nabil Khossossi
Date: September 2025
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Nabil Khossossi"

from .data_extraction import MaterialsProjectExtractor
from .descriptors import PhotovoltaicDescriptors
from .models import PhysicsInformedModel, MultiObjectiveOptimizer
from .visualization import PVVisualization

__all__ = [
    'MaterialsProjectExtractor',
    'PhotovoltaicDescriptors',
    'PhysicsInformedModel',
    'MultiObjectiveOptimizer',
    'PVVisualization',
]
