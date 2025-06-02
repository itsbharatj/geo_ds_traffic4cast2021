# src/experiments/__init__.py
"""
Experiment runners for Traffic4Cast
"""

from .spatial_transfer import SpatialTransferExperiment
from .temporal_transfer import TemporalTransferExperiment
from .spatiotemporal_transfer import SpatioTemporalTransferExperiment

__all__ = ['SpatialTransferExperiment', 'TemporalTransferExperiment', 'SpatioTemporalTransferExperiment']

def create_experiment(config):
    """Factory function to create appropriate experiment"""
    
    if config.experiment.type == "spatial_transfer":
        return SpatialTransferExperiment(config)
    elif config.experiment.type == "temporal_transfer":
        return TemporalTransferExperiment(config)
    elif config.experiment.type == "spatiotemporal_transfer":
        return SpatioTemporalTransferExperiment(config)
    else:
        raise ValueError(f"Unknown experiment type: {config.experiment.type}")

__all__ = [
    'SpatialTransferExperiment', 
    'SpatioTemporalTransferExperiment',
    'EnhancedSpatioTemporalTransferExperiment'
]