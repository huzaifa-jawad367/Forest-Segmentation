"""
CSV-based dataset that loads images and masks from CSV files and creates tiles.
"""

import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
try:
    from .transforms import ForestBinaryTransform
except ImportError:
    from transforms import ForestBinaryTransform
from typing import List, Dict


class CSVBasedDataset(Dataset):
    """
    Dataset that loads images and masks from CSV file paths and creates tiles.
    """
    
    def __init__(self, csv_path: str, tile_size: int = 512, original_size: int = 1024, 
                 transforms=None, forest_class_id: int = 6):
        """
        Initialize CSV-based dataset.
        
        Args:
            csv_path: Path to CSV file with image_path and mask_path columns
            tile_size: Size of each tile (default: 512)
            original_size: Original image size (default: 1024)
            transforms: Transform to apply to tiles
            forest_class_id: Class ID for forest pixels
        """
        self.tile_size = tile_size
        self.original_size = original_size
        self.transforms = transforms
        self.forest_class_id = forest_class_id
        
        # Load CSV data
        self.csv_data = self._load_csv(csv_path)
        self.tiles_per_image = (original_size // tile_size) ** 2  # 4 tiles for 1024->512
        
        print(f"Loaded {len(self.csv_data)} images from CSV")
        print(f"Total tiles: {len(self.csv_data) * self.tiles_per_image}")
    
    def _load_csv(self, csv_path: str) -> List[Dict]:
        """Load CSV file and return list of dictionaries."""
        data = []
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        return data
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from path."""
        image = Image.open(image_path).convert('RGB')
        return np.array(image)
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load mask from path (keep original values for ForestBinaryTransform to handle)."""
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        # Return original mask values - ForestBinaryTransform will handle binary conversion
        return mask_array
    
    def _create_tiles(self, image: np.ndarray, mask: np.ndarray) -> List[Dict]:
        """Create tiles from image and mask."""
        tiles = []
        h, w = image.shape[:2]
        tile_h, tile_w = self.tile_size, self.tile_size
        
        # Create 4 tiles (2x2 grid)
        for i in range(2):  # 2 rows
            for j in range(2):  # 2 columns
                start_h = i * tile_h
                end_h = start_h + tile_h
                start_w = j * tile_w
                end_w = start_w + tile_w
                
                # Extract tile
                image_tile = image[start_h:end_h, start_w:end_w]
                mask_tile = mask[start_h:end_h, start_w:end_w]
                
                tiles.append({
                    'image': image_tile,
                    'mask': mask_tile
                })
        
        return tiles
    
    def __len__(self) -> int:
        """Return total number of tiles."""
        return len(self.csv_data) * self.tiles_per_image
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a tile by index."""
        # Calculate which image and which tile within that image
        image_idx = idx // self.tiles_per_image
        tile_idx = idx % self.tiles_per_image
        
        # Get image data from CSV
        image_data = self.csv_data[image_idx]
        image_path = image_data['image_path']
        mask_path = image_data['mask_path']
        
        # Load image and mask
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        
        # Create tiles
        tiles = self._create_tiles(image, mask)
        tile_data = tiles[tile_idx]
        
        # Convert to tensors
        image_tensor = torch.from_numpy(tile_data['image']).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(tile_data['mask']).long()
        
        sample = {
            'image': image_tensor,
            'mask': mask_tensor
        }
        
        # Apply transforms if provided
        if self.transforms:
            sample = self.transforms(sample)
        
        return sample


def main():
    """Test the CSV-based dataset."""
    print("CSV-BASED DATASET TEST")
    print("="*50)
    
    # Test with percent split CSV
    print("Testing with percent split CSV...")
    dataset = CSVBasedDataset(
        csv_path="subsampling/Subsets/training_data_percent_split_sorted.csv",
        tile_size=512,
        original_size=1024,
        transforms=ForestBinaryTransform(forest_class_id=6, mode="train"),
        forest_class_id=6
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test getting a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample image shape: {sample['image'].shape}")
        print(f"Sample mask shape: {sample['mask'].shape}")
        print(f"Sample mask unique values: {sample['mask'].unique()}")
    
    print("\nTesting with pixel count CSV...")
    dataset2 = CSVBasedDataset(
        csv_path="subsampling/Subsets/training_data_pixel_count_sorted.csv",
        tile_size=512,
        original_size=1024,
        transforms=ForestBinaryTransform(forest_class_id=6, mode="val"),
        forest_class_id=6
    )
    
    print(f"Dataset2 length: {len(dataset2)}")
    
    if len(dataset2) > 0:
        sample2 = dataset2[0]
        print(f"Sample2 image shape: {sample2['image'].shape}")
        print(f"Sample2 mask shape: {sample2['mask'].shape}")
        print(f"Sample2 mask unique values: {sample2['mask'].unique()}")


if __name__ == "__main__":
    main()
