"""
Create a precisely balanced dataset targeting exact pixel counts.

This script uses an iterative approach to select images that get us as close as possible
to the target pixel counts: 330M background + 356M forest = 686M total pixels.
"""

from .tiled_dataset import TiledAerialDataset
import numpy as np
from .transforms import ForestBinaryTransform
from typing import List, Tuple, Dict


class PreciseBalancedDataset:
    """
    A dataset that precisely targets specific pixel counts through iterative selection.
    """
    
    def __init__(
        self,
        dataset: TiledAerialDataset,
        target_foreground_pixels: int,
        target_background_pixels: int,
        max_iterations: int = 1000,
        tolerance: float = 0.01,  # 1% tolerance
        device: str = 'cpu'
    ):
        """
        Initialize precise balanced dataset.
        
        Args:
            dataset: Original dataset
            target_foreground_pixels: Target foreground pixels
            target_background_pixels: Target background pixels
            max_iterations: Maximum selection iterations
            tolerance: Tolerance for target achievement
            device: Device for computation
        """
        self.original_dataset = dataset
        self.target_foreground = target_foreground_pixels
        self.target_background = target_background_pixels
        self.target_total = target_foreground_pixels + target_background_pixels
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.device = device
        
        # Analyze all images first
        self.image_analytics = self._analyze_all_images()
        
        # Select images iteratively
        self.selected_indices = self._select_images_iteratively()
        
        # Create subset
        from torch.utils.data import Subset
        self.subset = Subset(dataset, self.selected_indices)
    
    def _analyze_all_images(self) -> List[Dict]:
        """Analyze all images to get their pixel counts."""
        print("Analyzing all images for pixel counts...")
        
        image_analytics = []
        tiles_per_image = self.original_dataset.tiles_per_image
        
        for i in range(len(self.original_dataset.files)):
            foreground_pixels = 0
            background_pixels = 0
            total_pixels = 0
            
            # Sample tiles from this image (every 4th tile for efficiency)
            for tile_idx in range(0, tiles_per_image, max(1, tiles_per_image // 4)):
                global_idx = i * tiles_per_image + tile_idx
                if global_idx < len(self.original_dataset):
                    sample = self.original_dataset[global_idx]
                    if 'mask' in sample:
                        mask = sample['mask'].to(self.device)
                        fg = (mask == 1).sum().item()
                        bg = (mask == 0).sum().item()
                        
                        foreground_pixels += fg
                        background_pixels += bg
                        total_pixels += mask.numel()
            
            if total_pixels > 0:
                # Scale up to full image (multiply by 4 since we sampled every 4th tile)
                scale_factor = 4
                foreground_pixels *= scale_factor
                background_pixels *= scale_factor
                total_pixels *= scale_factor
                
                image_analytics.append({
                    'image_idx': i,
                    'foreground_pixels': foreground_pixels,
                    'background_pixels': background_pixels,
                    'total_pixels': total_pixels,
                    'foreground_ratio': foreground_pixels / total_pixels
                })
        
        print(f"Analyzed {len(image_analytics)} images")
        return image_analytics
    
    def _select_images_iteratively(self) -> List[int]:
        """Select images iteratively to get as close as possible to target pixel counts."""
        print(f"Selecting images to reach target: {self.target_foreground:,} foreground, {self.target_background:,} background")
        
        selected_images = []
        current_foreground = 0
        current_background = 0
        current_total = 0
        
        # Sort images by how close their ratio is to target ratio
        target_ratio = self.target_foreground / self.target_total
        sorted_images = sorted(
            self.image_analytics,
            key=lambda x: abs(x['foreground_ratio'] - target_ratio)
        )
        
        # Iterative selection
        for iteration in range(self.max_iterations):
            best_image = None
            best_score = float('inf')
            
            # Find the image that gets us closest to target
            for img in sorted_images:
                if img['image_idx'] in selected_images:
                    continue
                
                # Calculate what the totals would be if we add this image
                new_foreground = current_foreground + img['foreground_pixels']
                new_background = current_background + img['background_pixels']
                new_total = new_foreground + new_background
                
                # Calculate how close we'd be to target
                fg_diff = abs(new_foreground - self.target_foreground) / self.target_foreground
                bg_diff = abs(new_background - self.target_background) / self.target_background
                total_diff = abs(new_total - self.target_total) / self.target_total
                
                # Combined score (weighted)
                score = fg_diff + bg_diff + total_diff
                
                if score < best_score:
                    best_score = score
                    best_image = img
            
            if best_image is None:
                break
            
            # Add the best image
            selected_images.append(best_image['image_idx'])
            current_foreground += best_image['foreground_pixels']
            current_background += best_image['background_pixels']
            current_total += best_image['total_pixels']
            
            # Check if we're close enough to target
            fg_ratio = current_foreground / self.target_foreground
            bg_ratio = current_background / self.target_background
            
            if (abs(1 - fg_ratio) < self.tolerance and 
                abs(1 - bg_ratio) < self.tolerance):
                print(f"Target achieved after {iteration + 1} iterations!")
                break
            
            # Print progress every 50 iterations
            if (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}: {len(selected_images)} images, "
                      f"FG: {current_foreground:,}/{self.target_foreground:,} "
                      f"({current_foreground/self.target_foreground:.2%}), "
                      f"BG: {current_background:,}/{self.target_background:,} "
                      f"({current_background/self.target_background:.2%})")
        
        # Convert image indices to tile indices
        tiles_per_image = self.original_dataset.tiles_per_image
        tile_indices = []
        
        for img_idx in selected_images:
            start_idx = img_idx * tiles_per_image
            end_idx = start_idx + tiles_per_image
            tile_indices.extend(range(start_idx, end_idx))
        
        print(f"Selected {len(selected_images)} images ({len(tile_indices)} tiles)")
        print(f"Final counts: FG: {current_foreground:,}, BG: {current_background:,}, Total: {current_total:,}")
        
        return tile_indices
    
    def __len__(self) -> int:
        return len(self.subset)
    
    def __getitem__(self, idx: int):
        return self.subset[idx]
    
    def get_analytics(self) -> Dict:
        """Get analytics about the precise balanced dataset."""
        return {
            'selected_images': len(self.selected_indices) // 4,  # 4 tiles per image
            'selected_tiles': len(self.selected_indices),
            'target_foreground': self.target_foreground,
            'target_background': self.target_background,
            'target_total': self.target_total
        }


def main():
    """Create precise balanced dataset."""
    print("PRECISE BALANCED DATASET CREATION")
    print("="*80)
    
    # Target pixel counts - 330M for both classes (50-50 split)
    target_foreground = 330_000_000  # 330M forest pixels
    target_background = 330_000_000  # 330M background pixels
    
    print(f"Target pixel counts:")
    print(f"  Foreground (forest): {target_foreground:,} pixels")
    print(f"  Background: {target_background:,} pixels")
    print(f"  Total: {target_foreground + target_background:,} pixels")
    
    # Create training dataset
    print(f"\nCreating training dataset...")
    train_dataset = TiledAerialDataset(
        root="LoveDa",
        split="train",
        tile_size=512,
        original_size=1024,
        transforms=ForestBinaryTransform(forest_class_id=6, mode="train"),
        return_metadata=False
    )
    
    # Create precise balanced training dataset
    balanced_train = PreciseBalancedDataset(
        train_dataset,
        target_foreground_pixels=target_foreground,
        target_background_pixels=target_background,
        max_iterations=1000,
        tolerance=0.02,  # 2% tolerance
        device='cpu'
    )
    
    # Create validation dataset (smaller)
    print(f"\nCreating validation dataset...")
    val_dataset = TiledAerialDataset(
        root="LoveDa",
        split="val",
        tile_size=512,
        original_size=1024,
        transforms=ForestBinaryTransform(forest_class_id=6, mode="val"),
        return_metadata=False
    )
    
    # Create smaller balanced validation dataset
    balanced_val = PreciseBalancedDataset(
        val_dataset,
        target_foreground_pixels=target_foreground // 5,  # 1/5 of training (66M each)
        target_background_pixels=target_background // 5,
        max_iterations=500,
        tolerance=0.05,  # 5% tolerance for validation
        device='cpu'
    )
    
    print(f"\nResults:")
    train_analytics = balanced_train.get_analytics()
    val_analytics = balanced_val.get_analytics()
    
    print(f"Training dataset:")
    print(f"  Selected images: {train_analytics['selected_images']:,}")
    print(f"  Selected tiles: {train_analytics['selected_tiles']:,}")
    
    print(f"\nValidation dataset:")
    print(f"  Selected images: {val_analytics['selected_images']:,}")
    print(f"  Selected tiles: {val_analytics['selected_tiles']:,}")
    
    print(f"\n{'='*80}")
    print("PRECISE BALANCED DATASET CREATION COMPLETED")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
