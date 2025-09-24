#!/usr/bin/env python3
"""
Test script for the CSV-based dataset functionality.
"""

import sys
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

# Add the Dataset directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Dataset', 'LoveDa_Dataloader'))

from create_precise_balanced_dataset import CSVBasedDataset
from transforms import ForestBinaryTransform

def test_dataset():
    """Test the CSV-based dataset."""
    print("🧪 Testing CSV-based Dataset...")
    print("=" * 60)
    
    # Test with percent split CSV
    print("1. Testing with percent split CSV...")
    dataset = CSVBasedDataset(
        csv_path="subsampling/Subsets/training_data_percent_split_sorted.csv",
        tile_size=512,
        original_size=1024,
        transforms=ForestBinaryTransform(forest_class_id=6, mode="train"),
        forest_class_id=6
    )
    
    print(f"   📊 Dataset length: {len(dataset)}")
    
    # Test getting samples
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"   🖼️ Sample image type: {type(sample['image'])}")
        print(f"   🖼️ Sample image shape: {sample['image'].shape}")
        print(f"   🎭 Sample mask type: {type(sample['mask'])}")
        print(f"   🎭 Sample mask shape: {sample['mask'].shape}")
        print(f"   🎯 Sample mask unique values: {sample['mask'].unique()}")
        print(f"   📈 Sample mask value range: {sample['mask'].min().item()} to {sample['mask'].max().item()}")
    
    # Test with pixel count CSV
    print(f"\n2. Testing with pixel count CSV...")
    dataset2 = CSVBasedDataset(
        csv_path="subsampling/Subsets/training_data_pixel_count_sorted.csv",
        tile_size=512,
        original_size=1024,
        transforms=ForestBinaryTransform(forest_class_id=6, mode="val"),
        forest_class_id=6
    )
    
    print(f"   📊 Dataset2 length: {len(dataset2)}")
    
    if len(dataset2) > 0:
        sample2 = dataset2[0]
        print(f"   🖼️ Sample2 image type: {type(sample2['image'])}")
        print(f"   🖼️ Sample2 image shape: {sample2['image'].shape}")
        print(f"   🎭 Sample2 mask type: {type(sample2['mask'])}")
        print(f"   🎭 Sample2 mask shape: {sample2['mask'].shape}")
        print(f"   🎯 Sample2 mask unique values: {sample2['mask'].unique()}")
        print(f"   📈 Sample2 mask value range: {sample2['mask'].min().item()} to {sample2['mask'].max().item()}")
    
    # Plot samples
    print(f"\n3. Plotting samples...")
    plot_multiple_windows(dataset, "Percent Split Dataset", samples_per_window=4, total_samples=20)
    plot_multiple_windows(dataset2, "Pixel Count Dataset", samples_per_window=4, total_samples=20)
    
    print(f"\n🎉 Testing completed!")

def plot_multiple_windows(dataset, title, samples_per_window=4, total_samples=20):
    """Plot samples in multiple windows, each showing 4 image-mask pairs."""
    if len(dataset) == 0:
        print(f"   ❌ No samples to plot for {title}")
        return
    
    num_windows = (total_samples + samples_per_window - 1) // samples_per_window  # Ceiling division
    
    for window_idx in range(num_windows):
        start_idx = window_idx * samples_per_window
        end_idx = min(start_idx + samples_per_window, total_samples)
        
        if start_idx >= len(dataset):
            break
            
        # Create a 2x4 grid for each window (2 rows, 4 columns: 4 image-mask pairs)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i in range(start_idx, end_idx):
            if i >= len(dataset):
                break
                
            sample = dataset[i]
            image = sample['image']
            mask = sample['mask']
            
            # Denormalize image if needed
            if image.min() < 0:  # Check if normalized
                image = denormalize_image(image)
            
            # Convert to numpy for plotting
            image_np = image.permute(1, 2, 0).numpy()
            mask_np = mask.numpy()
            
            # Calculate position in the 2x4 grid
            local_idx = i - start_idx
            row = local_idx // 2  # Which row (0 or 1)
            col = local_idx % 2   # Which column (0 or 1)
            
            # Plot image (left column of each pair)
            axes[row, col*2].imshow(image_np)
            axes[row, col*2].set_title(f"Image {i}")
            axes[row, col*2].axis('off')
            
            # Plot mask (right column of each pair)
            axes[row, col*2+1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            
            # Count pixels for debugging
            zero_pixels = (mask_np == 0).sum()
            one_pixels = (mask_np == 1).sum()
            total_pixels = mask_np.size
            
            axes[row, col*2+1].set_title(f"Mask {i}\n0s: {zero_pixels}, 1s: {one_pixels}")
            axes[row, col*2+1].axis('off')
            
            # Print pixel counts for debugging
            print(f"   Sample {i}: 0 pixels = {zero_pixels}, 1 pixels = {one_pixels}, total = {total_pixels}")
        
        # Hide unused subplots
        for i in range(end_idx - start_idx, 4):
            local_idx = i
            row = local_idx // 2
            col = local_idx % 2
            axes[row, col*2].axis('off')
            axes[row, col*2+1].axis('off')
        
        plt.suptitle(f"{title} - Window {window_idx + 1}/{num_windows}")
        plt.tight_layout()
        plt.show()
        
        print(f"   ✅ Plotted window {window_idx + 1}/{num_windows} with samples {start_idx}-{end_idx-1}")

def plot_samples(dataset, title, num_samples=4):
    """Plot samples from the dataset."""
    if len(dataset) == 0:
        print(f"   ❌ No samples to plot for {title}")
        return
    
    # Create a grid layout for 20 samples (4 rows x 10 columns: 10 image-mask pairs)
    if num_samples == 20:
        rows, cols = 4, 10  # 4 rows, 10 columns (10 pairs of image-mask)
        fig, axes = plt.subplots(rows, cols, figsize=(25, 10))
    elif num_samples == 10:
        rows, cols = 2, 10  # 2 rows, 10 columns (5 pairs of image-mask)
        fig, axes = plt.subplots(rows, cols, figsize=(25, 5))
    else:
        fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
        if num_samples == 1:
            axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        if i >= len(dataset):
            break
            
        sample = dataset[i]
        image = sample['image']
        mask = sample['mask']
        
        # Denormalize image if needed
        if image.min() < 0:  # Check if normalized
            image = denormalize_image(image)
        
        # Convert to numpy for plotting
        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.numpy()
        
        if num_samples == 20:
            # For 20 samples, show 10 pairs of image-mask (4 rows, 10 columns)
            row = i // 5  # Which row (0-3)
            col = i % 5   # Which column (0-4)
            
            # Plot image (left column of each pair)
            axes[row, col*2].imshow(image_np)
            axes[row, col*2].set_title(f"Image {i}")
            axes[row, col*2].axis('off')
            
            # Plot mask (right column of each pair)
            axes[row, col*2+1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            axes[row, col*2+1].set_title(f"Mask {i}")
            axes[row, col*2+1].axis('off')
        elif num_samples == 10:
            # For 10 samples, show 5 pairs of image-mask (2 rows, 10 columns)
            row = i // 5  # Which row (0 or 1)
            col = i % 5   # Which column (0-4)
            
            # Plot image (left column of each pair)
            axes[row, col*2].imshow(image_np)
            axes[row, col*2].set_title(f"Image {i}")
            axes[row, col*2].axis('off')
            
            # Plot mask (right column of each pair)
            axes[row, col*2+1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            axes[row, col*2+1].set_title(f"Mask {i}")
            axes[row, col*2+1].axis('off')
        else:
            # Original layout for smaller numbers
            # Plot image
            axes[0, i].imshow(image_np)
            axes[0, i].set_title(f"Image {i}")
            axes[0, i].axis('off')
            
            # Plot mask
            axes[1, i].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f"Mask {i}")
            axes[1, i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    print(f"   ✅ Plotted {min(num_samples, len(dataset))} samples from {title}")

def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Denormalize a normalized image tensor."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

if __name__ == "__main__":
    test_dataset()
