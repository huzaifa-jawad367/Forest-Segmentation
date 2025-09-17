"""
LoveDA dataset package for forest segmentation.

This package provides:
- TiledAerialDataset: Main dataset class for tiled aerial images
- Transform utilities: Various transforms for training, validation, and inference
# - BalancedDataset: Balanced subset creation with target pixel ratios (Deprecated)
"""

from .tiled_dataset import TiledAerialDataset
from .transforms import (
    TiledTransform,
    TrainTransforms,
    ValTransforms,
    TestTransforms,
    InferenceTransforms,
    ForestBinaryTransform,
    get_transforms
)
# from .balanced_dataset import BalancedDataset, create_balanced_datasets  # Deprecated

__all__ = [
    'TiledAerialDataset',
    'TiledTransform',
    'TrainTransforms',
    'ValTransforms', 
    'TestTransforms',
    'InferenceTransforms',
    'ForestBinaryTransform',
    'get_transforms',
    # 'BalancedDataset',  # Deprecated
    # 'create_balanced_datasets'  # Deprecated
]
