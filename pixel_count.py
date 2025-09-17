"""
Pixel counting script for tiled aerial image dataset.

This script loops through all training and validation data to calculate
the number of background (0) and foreground (1) pixels in binary masks.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Dataset.LoveDa import TiledAerialDataset, ForestBinaryTransform


def count_pixels_in_dataset(dataset, dataset_name="Dataset", device='cpu'):
    """
    Count background and foreground pixels in a dataset.
    
    Args:
        dataset: Dataset to analyze
        dataset_name: Name for display purposes
        device: Device to use for computation
        
    Returns:
        Dictionary with pixel counts and statistics
    """
    print(f"\nAnalyzing {dataset_name}...")
    print(f"Total samples: {len(dataset)}")
    
    total_background = 0
    total_foreground = 0
    total_pixels = 0
    samples_with_foreground = 0
    
    # Progress bar
    pbar = tqdm(dataset, desc=f"Processing {dataset_name}")
    
    for sample in pbar:
        if 'mask' in sample:
            mask = sample['mask']
            
            # Count pixels
            background_pixels = (mask == 0).sum().item()
            foreground_pixels = (mask == 1).sum().item()
            
            total_background += background_pixels
            total_foreground += foreground_pixels
            total_pixels += mask.numel()
            
            # Count samples with foreground
            if foreground_pixels > 0:
                samples_with_foreground += 1
            
            # Update progress bar with current stats
            pbar.set_postfix({
                'Foreground': f"{total_foreground:,}",
                'Background': f"{total_background:,}",
                'Ratio': f"{total_foreground/(total_foreground+total_background)*100:.2f}%" if (total_foreground+total_background) > 0 else "0%"
            })
    
    # Calculate statistics
    stats = {
        'dataset_name': dataset_name,
        'total_samples': len(dataset),
        'samples_with_foreground': samples_with_foreground,
        'samples_without_foreground': len(dataset) - samples_with_foreground,
        'total_background_pixels': total_background,
        'total_foreground_pixels': total_foreground,
        'total_pixels': total_pixels,
        'foreground_ratio': total_foreground / total_pixels if total_pixels > 0 else 0,
        'background_ratio': total_background / total_pixels if total_pixels > 0 else 0,
        'foreground_per_sample': total_foreground / len(dataset) if len(dataset) > 0 else 0,
        'background_per_sample': total_background / len(dataset) if len(dataset) > 0 else 0
    }
    
    return stats


def print_statistics(stats):
    """Print formatted statistics."""
    print(f"\n{'='*60}")
    print(f"PIXEL STATISTICS - {stats['dataset_name'].upper()}")
    print(f"{'='*60}")
    
    print(f"Dataset Info:")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Samples with foreground: {stats['samples_with_foreground']:,}")
    print(f"  Samples without foreground: {stats['samples_without_foreground']:,}")
    print(f"  Foreground sample ratio: {stats['samples_with_foreground']/stats['total_samples']*100:.2f}%")
    
    print(f"\nPixel Counts:")
    print(f"  Total pixels: {stats['total_pixels']:,}")
    print(f"  Background pixels: {stats['total_background_pixels']:,}")
    print(f"  Foreground pixels: {stats['total_foreground_pixels']:,}")
    
    print(f"\nPixel Ratios:")
    print(f"  Background ratio: {stats['background_ratio']*100:.4f}%")
    print(f"  Foreground ratio: {stats['foreground_ratio']*100:.4f}%")
    
    print(f"\nPer Sample Averages:")
    print(f"  Background pixels per sample: {stats['background_per_sample']:,.0f}")
    print(f"  Foreground pixels per sample: {stats['foreground_per_sample']:,.0f}")
    
    print(f"{'='*60}")


def compare_datasets(train_stats, val_stats):
    """Compare statistics between training and validation datasets."""
    print(f"\n{'='*60}")
    print("DATASET COMPARISON")
    print(f"{'='*60}")
    
    print(f"Sample Counts:")
    print(f"  Training: {train_stats['total_samples']:,} samples")
    print(f"  Validation: {val_stats['total_samples']:,} samples")
    print(f"  Ratio (Train/Val): {train_stats['total_samples']/val_stats['total_samples']:.2f}")
    
    print(f"\nForeground Ratios:")
    print(f"  Training: {train_stats['foreground_ratio']*100:.4f}%")
    print(f"  Validation: {val_stats['foreground_ratio']*100:.4f}%")
    print(f"  Difference: {(train_stats['foreground_ratio'] - val_stats['foreground_ratio'])*100:.4f}%")
    
    print(f"\nSamples with Foreground:")
    print(f"  Training: {train_stats['samples_with_foreground']:,} ({train_stats['samples_with_foreground']/train_stats['total_samples']*100:.2f}%)")
    print(f"  Validation: {val_stats['samples_with_foreground']:,} ({val_stats['samples_with_foreground']/val_stats['total_samples']*100:.2f}%)")
    
    print(f"{'='*60}")


def main():
    """Main function to run pixel counting analysis."""
    print("PIXEL COUNTING ANALYSIS")
    print("="*60)
    
    # Set up paths
    root_path = Path("LoveDa")
    
    if not root_path.exists():
        print(f"Error: Dataset path '{root_path}' does not exist!")
        print("Please make sure the LoveDA dataset is extracted in the current directory.")
        return
    
    # Create datasets
    print("Creating datasets...")
    
    train_dataset = TiledAerialDataset(
        root=root_path,
        split="train",
        tile_size=512,
        original_size=1024,
        transforms=ForestBinaryTransform(forest_class_id=6, mode="train"),
        return_metadata=False
    )
    
    val_dataset = TiledAerialDataset(
        root=root_path,
        split="val",
        tile_size=512,
        original_size=1024,
        transforms=ForestBinaryTransform(forest_class_id=6, mode="val"),
        return_metadata=False
    )
    
    print(f"Train dataset: {len(train_dataset)} tiles")
    print(f"Val dataset: {len(val_dataset)} tiles")
    
    # Count pixels in training dataset
    train_stats = count_pixels_in_dataset(train_dataset, "Training Dataset")
    
    # Count pixels in validation dataset
    val_stats = count_pixels_in_dataset(val_dataset, "Validation Dataset")
    
    # Print individual statistics
    print_statistics(train_stats)
    print_statistics(val_stats)
    
    # Compare datasets
    compare_datasets(train_stats, val_stats)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total training pixels analyzed: {train_stats['total_pixels']:,}")
    print(f"Total validation pixels analyzed: {val_stats['total_pixels']:,}")
    print(f"Combined foreground ratio: {(train_stats['total_foreground_pixels'] + val_stats['total_foreground_pixels']) / (train_stats['total_pixels'] + val_stats['total_pixels']) * 100:.4f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
