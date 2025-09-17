"""
Tiled Aerial Image Dataset for Forest Segmentation

This module implements a dataset that automatically tiles 1024x1024 aerial images
into 512x512 patches for training, validation, and testing. The dataset supports
different transform modes and can handle both supervised and inference scenarios.
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union, Tuple
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class TiledAerialDataset(Dataset):
    """
    Dataset for tiled aerial image segmentation.
    
    Automatically splits 1024x1024 images into 512x512 tiles during loading.
    Supports different transform modes for training, validation, and inference.
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        tile_size: int = 512,
        original_size: int = 1024,
        transforms: Optional[Callable] = None,
        return_metadata: bool = False,
    ):
        """
        Initialize the tiled dataset.
        
        Args:
            root: Root directory containing Train, Val, Test folders
            split: Dataset split ('train', 'val', 'test')
            tile_size: Size of each tile (default: 512)
            original_size: Size of original images (default: 1024)
            transforms: Transform to apply to tiles
            return_metadata: Whether to return tile coordinates and metadata
        """
        self.root = Path(root)
        self.split = split
        self.tile_size = tile_size
        self.original_size = original_size
        self.transforms = transforms
        self.return_metadata = return_metadata
        
        # Validate split
        valid_splits = ['train', 'val', 'test']
        assert split in valid_splits, f"Split must be one of {valid_splits}"
        
        # Calculate number of tiles per image
        self.tiles_per_side = original_size // tile_size
        self.tiles_per_image = self.tiles_per_side ** 2
        
        # Get file paths
        self.files = self._get_files()
        
        # Pre-calculate total number of tiles
        self.total_tiles = len(self.files) * self.tiles_per_image
        
    def _get_files(self) -> List[Dict[str, str]]:
        """Get list of image files and corresponding mask files."""
        # LoveDA specific structure: split/split/scene/images_png and masks_png
        split_dir = self.root / self.split.capitalize() / self.split.capitalize()
        
        # Get all scene directories (Urban, Rural)
        scene_dirs = []
        if split_dir.exists():
            for item in split_dir.iterdir():
                if item.is_dir():
                    scene_dirs.append(item)
        
        # Get all image files from all scenes
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        image_files = []
        
        for scene_dir in scene_dirs:
            images_png_dir = scene_dir / "images_png"
            if images_png_dir.exists():
                for ext in image_extensions:
                    image_files.extend(glob.glob(str(images_png_dir / ext)))
        
        image_files = sorted(image_files)
        
        files = []
        for img_path in image_files:
            file_dict = {'image': img_path}
            
            # Add mask path for supervised splits
            if self.split != 'test':
                mask_path = img_path.replace('images_png', 'masks_png')
                file_dict['mask'] = mask_path
                
            files.append(file_dict)
            
        return files
    
    def __len__(self) -> int:
        """Return total number of tiles."""
        return self.total_tiles
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a tile from the dataset.
        
        Args:
            index: Global tile index across all images
            
        Returns:
            Dictionary containing:
            - 'image': RGB tile tensor (C, H, W)
            - 'mask': Segmentation mask tensor (H, W) [if supervised]
            - 'metadata': Dict with tile coordinates and file info [if return_metadata=True]
        """
        # Calculate which image and which tile within that image
        image_idx = index // self.tiles_per_image
        tile_idx = index % self.tiles_per_image
        
        # Calculate tile coordinates within the image
        tile_row = tile_idx // self.tiles_per_side
        tile_col = tile_idx % self.tiles_per_side
        
        # Load the full image
        file_info = self.files[image_idx]
        image = self._load_image(file_info['image'])
        
        # Extract tile from image
        tile_image = self._extract_tile(
            image, tile_row, tile_col, self.tile_size
        )
        
        sample = {'image': tile_image}
        
        # Load and extract corresponding mask tile for supervised splits
        if self.split != 'test':
            mask = self._load_mask(file_info['mask'])
            tile_mask = self._extract_tile(
                mask, tile_row, tile_col, self.tile_size
            )
            sample['mask'] = tile_mask
        
        # Add metadata if requested
        if self.return_metadata:
            sample['metadata'] = {
                'image_path': file_info['image'],
                'image_idx': image_idx,
                'tile_idx': tile_idx,
                'tile_row': tile_row,
                'tile_col': tile_col,
                'tile_coords': (tile_row * self.tile_size, tile_col * self.tile_size),
                'split': self.split
            }
        
        # Apply transforms
        if self.transforms is not None:
            sample = self.transforms(sample)
            
        return sample
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load an RGB image as numpy array."""
        with Image.open(path) as img:
            img = img.convert('RGB')
            return np.array(img)
    
    def _load_mask(self, path: str) -> np.ndarray:
        """Load a segmentation mask as numpy array."""
        with Image.open(path) as img:
            # Convert to grayscale and then to numpy
            img = img.convert('L')
            return np.array(img, dtype=np.int64)
    
    def _extract_tile(
        self, 
        image: np.ndarray, 
        row: int, 
        col: int, 
        tile_size: int
    ) -> torch.Tensor:
        """Extract a tile from the full image."""
        start_row = row * tile_size
        start_col = col * tile_size
        end_row = start_row + tile_size
        end_col = start_col + tile_size
        
        tile = image[start_row:end_row, start_col:end_col]
        
        # Convert to tensor and normalize to [0, 1]
        if len(tile.shape) == 3:  # RGB image
            tile = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
        else:  # Grayscale mask
            tile = torch.from_numpy(tile).long()
            
        return tile
    
    def get_image_info(self, image_idx: int) -> Dict:
        """Get information about a specific image."""
        if image_idx >= len(self.files):
            raise IndexError(f"Image index {image_idx} out of range")
            
        file_info = self.files[image_idx]
        return {
            'image_path': file_info['image'],
            'mask_path': file_info.get('mask'),
            'tiles_per_image': self.tiles_per_image,
            'tile_size': self.tile_size,
            'original_size': self.original_size
        }
    
    def reconstruct_prediction(
        self, 
        predictions: List[torch.Tensor], 
        image_idx: int
    ) -> torch.Tensor:
        """
        Reconstruct full image prediction from tile predictions.
        
        Args:
            predictions: List of tile predictions in order
            image_idx: Index of the image to reconstruct
            
        Returns:
            Reconstructed full image prediction
        """
        if len(predictions) != self.tiles_per_image:
            raise ValueError(f"Expected {self.tiles_per_image} predictions, got {len(predictions)}")
        
        # Initialize full image tensor
        full_pred = torch.zeros(
            (self.original_size, self.original_size),
            dtype=predictions[0].dtype,
            device=predictions[0].device
        )
        
        # Place each tile in the correct position
        for i, pred in enumerate(predictions):
            row = i // self.tiles_per_side
            col = i % self.tiles_per_side
            
            start_row = row * self.tile_size
            start_col = col * self.tile_size
            end_row = start_row + self.tile_size
            end_col = start_col + self.tile_size
            
            # Handle different prediction formats
            if len(pred.shape) == 3:  # (C, H, W) - take argmax
                pred = torch.argmax(pred, dim=0)
            elif len(pred.shape) == 2:  # (H, W) - use directly
                pass
            else:
                raise ValueError(f"Unexpected prediction shape: {pred.shape}")
            
            full_pred[start_row:end_row, start_col:end_col] = pred
            
        return full_pred
