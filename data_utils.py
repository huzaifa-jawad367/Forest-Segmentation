"""
Data loading utilities and helper functions for tiled aerial image dataset.

This module provides utilities for creating data loaders, handling batch processing,
and managing the dataset pipeline.
"""

import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from Dataset.LoveDa import TiledAerialDataset, get_transforms, ForestBinaryTransform


def create_data_loaders(
    root: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    tile_size: int = 512,
    original_size: int = 1024,
    resize: Optional[Tuple[int, int]] = None,
    forest_class_id: int = 6,
    subset_ratio: Optional[float] = None,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    Create data loaders for train, validation, and test splits.
    
    Args:
        root: Root directory containing Train, Val, Test folders
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        tile_size: Size of each tile
        original_size: Size of original images
        resize: Optional resize dimensions for tiles
        forest_class_id: Class ID for forest in original masks
        subset_ratio: Optional ratio to subset the dataset (for quick testing)
        **kwargs: Additional arguments for transforms
        
    Returns:
        Dictionary containing train, val, and test data loaders
    """
    # Create datasets
    train_dataset = TiledAerialDataset(
        root=root,
        split="train",
        tile_size=tile_size,
        original_size=original_size,
        transforms=ForestBinaryTransform(forest_class_id=forest_class_id, mode="train"),
        return_metadata=False
    )
    
    val_dataset = TiledAerialDataset(
        root=root,
        split="val",
        tile_size=tile_size,
        original_size=original_size,
        transforms=ForestBinaryTransform(forest_class_id=forest_class_id, mode="val"),
        return_metadata=False
    )
    
    test_dataset = TiledAerialDataset(
        root=root,
        split="test",
        tile_size=tile_size,
        original_size=original_size,
        transforms=ForestBinaryTransform(forest_class_id=forest_class_id, mode="test"),
        return_metadata=True  # Return metadata for test set
    )
    
    # Create subsets if requested
    if subset_ratio is not None:
        train_size = int(len(train_dataset) * subset_ratio)
        val_size = int(len(val_dataset) * subset_ratio)
        train_dataset = Subset(train_dataset, range(train_size))
        val_dataset = Subset(val_dataset, range(val_size))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def create_inference_dataset(
    root: Union[str, Path],
    split: str = "test",
    tile_size: int = 512,
    original_size: int = 1024,
    resize: Optional[Tuple[int, int]] = None,
    **kwargs
) -> TiledAerialDataset:
    """
    Create dataset for inference with metadata.
    
    Args:
        root: Root directory containing data
        split: Dataset split to use for inference
        tile_size: Size of each tile
        original_size: Size of original images
        resize: Optional resize dimensions
        **kwargs: Additional arguments for transforms
        
    Returns:
        Dataset configured for inference
    """
    return TiledAerialDataset(
        root=root,
        split=split,
        tile_size=tile_size,
        original_size=original_size,
        transforms=get_transforms("inference", resize=resize, **kwargs),
        return_metadata=True
    )


def get_dataset_stats(dataset: TiledAerialDataset) -> Dict[str, int]:
    """
    Get basic statistics about the dataset.
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_tiles': len(dataset),
        'total_images': len(dataset.files),
        'tiles_per_image': dataset.tiles_per_image,
        'tile_size': dataset.tile_size,
        'original_size': dataset.original_size
    }
    
    return stats


def count_class_pixels(dataset: TiledAerialDataset, device: str = 'cpu') -> Dict[str, int]:
    """
    Count pixels for each class in the dataset.
    
    Args:
        dataset: Dataset to analyze
        device: Device to use for computation
        
    Returns:
        Dictionary with pixel counts for each class
    """
    forest_count = 0
    background_count = 0
    
    for i in range(len(dataset)):
        sample = dataset[i]
        if 'mask' in sample:
            mask = sample['mask'].to(device)
            forest_count += (mask == 1).sum().item()
            background_count += (mask == 0).sum().item()
    
    return {
        'forest_pixels': forest_count,
        'background_pixels': background_count,
        'total_pixels': forest_count + background_count,
        'forest_ratio': forest_count / (forest_count + background_count) if (forest_count + background_count) > 0 else 0
    }


def create_balanced_subset(
    dataset: TiledAerialDataset,
    target_ratio: float = 0.5,
    max_images: Optional[int] = None,
    device: str = 'cpu'
) -> Subset:
    """
    Create a balanced subset of the dataset.
    
    Args:
        dataset: Original dataset
        target_ratio: Target forest/background ratio
        max_images: Maximum number of images to include
        device: Device to use for computation
        
    Returns:
        Balanced subset of the dataset
    """
    # Calculate forest ratio for each image
    image_ratios = []
    tiles_per_image = dataset.tiles_per_image
    
    for i in range(len(dataset.files)):
        forest_pixels = 0
        total_pixels = 0
        
        # Sample tiles from this image
        for tile_idx in range(0, tiles_per_image, max(1, tiles_per_image // 4)):  # Sample every 4th tile
            global_idx = i * tiles_per_image + tile_idx
            if global_idx < len(dataset):
                sample = dataset[global_idx]
                if 'mask' in sample:
                    mask = sample['mask'].to(device)
                    forest_pixels += (mask == 1).sum().item()
                    total_pixels += mask.numel()
        
        if total_pixels > 0:
            ratio = forest_pixels / total_pixels
            image_ratios.append((i, ratio))
    
    # Sort by how close they are to target ratio
    image_ratios.sort(key=lambda x: abs(x[1] - target_ratio))
    
    # Select images
    if max_images is not None:
        selected_images = image_ratios[:max_images]
    else:
        selected_images = image_ratios
    
    # Create indices for selected images
    selected_indices = []
    for img_idx, _ in selected_images:
        start_idx = img_idx * tiles_per_image
        end_idx = start_idx + tiles_per_image
        selected_indices.extend(range(start_idx, end_idx))
    
    return Subset(dataset, selected_indices)


def visualize_sample(
    sample: Dict[str, torch.Tensor],
    title: str = "Sample",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a sample from the dataset.
    
    Args:
        sample: Sample dictionary from dataset
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    
    image = sample['image']
    mask = sample.get('mask')
    
    # Denormalize image for visualization
    if image.min() < 0:  # Assume normalized
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
    
    # Convert to numpy for plotting
    image_np = image.permute(1, 2, 0).numpy()
    
    if mask is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot image
        axes[0].imshow(image_np)
        axes[0].set_title(f"{title} - Image")
        axes[0].axis('off')
        
        # Plot mask
        axes[1].imshow(mask.numpy(), cmap='gray')
        axes[1].set_title(f"{title} - Mask")
        axes[1].axis('off')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(image_np)
        ax.set_title(f"{title} - Image")
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """
    Move batch to specified device.
    
    Args:
        batch: Batch dictionary
        device: Target device
        
    Returns:
        Batch moved to device
    """
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value 
            for key, value in batch.items()}


# Example usage and configuration
def get_default_config() -> Dict:
    """
    Get default configuration for the dataset.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'batch_size': 32,
        'num_workers': 4,
        'pin_memory': True,
        'tile_size': 512,
        'original_size': 1024,
        'forest_class_id': 6,
        'resize': None,  # Keep original tile size
        'subset_ratio': None,  # Use full dataset
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
