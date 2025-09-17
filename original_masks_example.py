"""
Visualize original LoveDA masks without relabeling to binary.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from Dataset.LoveDa.tiled_dataset import TiledAerialDataset


def visualize_original_masks(dataset, num_samples=15, samples_per_plot=3, title_prefix="Sample"):
    """
    Visualize multiple samples with original LoveDA masks.
    """
    # Get random indices
    total_samples = len(dataset)
    random_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    # Create plots in batches
    for batch_idx in range(0, len(random_indices), samples_per_plot):
        batch_indices = random_indices[batch_idx:batch_idx + samples_per_plot]
        
        # Create subplot
        fig, axes = plt.subplots(2, len(batch_indices), figsize=(5 * len(batch_indices), 10))
        if len(batch_indices) == 1:
            axes = axes.reshape(2, 1)
        
        for i, idx in enumerate(batch_indices):
            sample = dataset[idx]
            image = sample['image']
            mask = sample.get('mask')
            
            # Convert to numpy for plotting (images are already in [0,1] range)
            image_np = image.permute(1, 2, 0).numpy()
            
            # Plot image
            axes[0, i].imshow(image_np)
            axes[0, i].set_title(f"{title_prefix} {idx} - Image")
            axes[0, i].axis('off')
            
            # Plot original mask
            if mask is not None:
                # Use a colormap that shows all 8 classes clearly
                im = axes[1, i].imshow(mask.numpy(), cmap='tab10', vmin=0, vmax=7)
                axes[1, i].set_title(f"{title_prefix} {idx} - Original Mask")
            else:
                axes[1, i].text(0.5, 0.5, 'No Mask', ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].set_title(f"{title_prefix} {idx} - No Mask")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Displayed samples: {batch_indices}")


def analyze_original_masks(dataset, num_samples=100):
    """Analyze the distribution of original LoveDA classes."""
    class_counts = {i: 0 for i in range(8)}  # LoveDA has 8 classes (0-7)
    
    print(f"Analyzing original masks in {min(num_samples, len(dataset))} samples...")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        if 'mask' in sample:
            mask = sample['mask'].numpy()
            unique_values, counts = np.unique(mask, return_counts=True)
            for val, count in zip(unique_values, counts):
                if val in class_counts:
                    class_counts[val] += count
    
    total_pixels = sum(class_counts.values())
    print(f"\nOriginal LoveDA Class Distribution:")
    print(f"{'Class':<10} {'Name':<15} {'Pixels':<12} {'Percentage':<10}")
    print("-" * 50)
    
    for class_id, count in class_counts.items():
        percentage = (count / total_pixels * 100) if total_pixels > 0 else 0
        class_name = get_loveda_class_name(class_id)
        print(f"{class_id:<10} {class_name:<15} {count:<12,} {percentage:<10.2f}%")
    
    return class_counts


def get_loveda_class_name(class_id):
    """Get the name of LoveDA class by ID."""
    class_names = {
        0: 'background',
        1: 'building', 
        2: 'road',
        3: 'water',
        4: 'barren',
        5: 'agriculture',
        6: 'forest',  # This is the forest class!
        7: 'no-data'
    }
    return class_names.get(class_id, f'unknown_{class_id}')


def print_class_info():
    """Print information about LoveDA classes."""
    print("=" * 60)
    print("LOVEDA DATASET CLASS INFORMATION")
    print("=" * 60)
    print(f"{'Index':<8} {'Class Name':<15} {'Color in Visualization'}")
    print("-" * 60)
    
    class_info = [
        (0, 'background', 'Dark blue'),
        (1, 'building', 'Orange'), 
        (2, 'road', 'Green'),
        (3, 'water', 'Red'),
        (4, 'barren', 'Purple'),
        (5, 'agriculture', 'Brown'),
        (6, 'forest', 'Pink'),  # This becomes 1 in binary
        (7, 'no-data', 'Gray')
    ]
    
    for idx, name, color in class_info:
        print(f"{idx:<8} {name:<15} {color}")
    
    print("\n" + "=" * 60)
    print("BINARY CONVERSION:")
    print("Forest (class 6) → 1 (white in binary masks)")
    print("All others (0,1,2,3,4,5,7) → 0 (black in binary masks)")
    print("=" * 60)


def main():
    """Main function to visualize original masks."""
    
    root_path = "/Users/huzaifajawad/Research/Forest paper/LoveDa"
    
    print("=== Original LoveDA Masks Visualization ===\n")
    
    # Print class information
    print_class_info()
    
    # Create dataset WITHOUT any transforms (to keep original masks)
    print("\n1. Creating dataset with original masks...")
    
    train_dataset = TiledAerialDataset(
        root=root_path,
        split="train",
        tile_size=512,
        original_size=1024,
        transforms=None,  # No transforms to keep original masks
        return_metadata=False
    )
    
    val_dataset = TiledAerialDataset(
        root=root_path,
        split="val",
        tile_size=512,
        original_size=1024,
        transforms=None,
        return_metadata=False
    )
    
    print(f"Train dataset: {len(train_dataset)} tiles from {len(train_dataset.files)} images")
    print(f"Val dataset: {len(val_dataset)} tiles from {len(val_dataset.files)} images")
    
    # Analyze original class distribution
    print("\n2. Analyzing original class distribution...")
    train_class_counts = analyze_original_masks(train_dataset, num_samples=200)
    
    # Visualize original masks
    print("\n3. Visualizing original masks...")
    
    # Visualize 15 random training samples with original masks
    print("Visualizing 15 random training samples with original LoveDA masks...")
    visualize_original_masks(train_dataset, num_samples=15, samples_per_plot=3, title_prefix="Train")
    
    # Visualize 15 random validation samples with original masks
    print("Visualizing 15 random validation samples with original LoveDA masks...")
    visualize_original_masks(val_dataset, num_samples=15, samples_per_plot=3, title_prefix="Val")
    
    print("\n=== Original masks visualization completed ===")


if __name__ == "__main__":
    main()
