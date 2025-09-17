"""
Transform utilities for tiled aerial image dataset.

This module provides different transform configurations for training, validation,
and inference modes, with proper handling of both images and masks.
"""

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from typing import Dict, Any, Optional, Callable
import random


class TiledTransform:
    """
    Base transform class for handling tiled data with proper mask handling.
    """
    
    def __init__(self, mode: str = "train"):
        """
        Initialize transform for specific mode.
        
        Args:
            mode: Transform mode ('train', 'val', 'test', 'inference')
        """
        self.mode = mode
        
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transforms to sample."""
        return sample


class TrainTransforms(TiledTransform):
    """
    Training transforms with data augmentation.
    """
    
    def __init__(
        self,
        resize: Optional[tuple] = None,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.5,
        rotation_prob: float = 0.3,
        rotation_degrees: tuple = (-15, 15),
        brightness_jitter: float = 0.2,
        contrast_jitter: float = 0.2,
        saturation_jitter: float = 0.2,
        hue_jitter: float = 0.1,
        normalize: bool = True,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        """
        Initialize training transforms.
        
        Args:
            resize: Optional resize dimensions (height, width)
            horizontal_flip_prob: Probability of horizontal flip
            vertical_flip_prob: Probability of vertical flip
            rotation_prob: Probability of rotation
            rotation_degrees: Range of rotation degrees
            brightness_jitter: Brightness jitter factor
            contrast_jitter: Contrast jitter factor
            saturation_jitter: Saturation jitter factor
            hue_jitter: Hue jitter factor
            normalize: Whether to normalize
            mean: Normalization mean
            std: Normalization std
        """
        super().__init__("train")
        self.resize = resize
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_prob = rotation_prob
        self.rotation_degrees = rotation_degrees
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.saturation_jitter = saturation_jitter
        self.hue_jitter = hue_jitter
        self.normalize = normalize
        self.mean = mean
        self.std = std
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply training transforms."""
        image = sample['image']
        mask = sample.get('mask')
        
        # Resize if specified
        if self.resize is not None:
            image = TF.resize(image, self.resize, interpolation=InterpolationMode.BILINEAR)
            if mask is not None:
                mask = TF.resize(mask.unsqueeze(0), self.resize, interpolation=InterpolationMode.NEAREST).squeeze(0)
        
        # Random horizontal flip
        if random.random() < self.horizontal_flip_prob:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)
        
        # Random vertical flip
        if random.random() < self.vertical_flip_prob:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)
        
        # Random rotation
        if random.random() < self.rotation_prob:
            angle = random.uniform(*self.rotation_degrees)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            if mask is not None:
                mask = TF.rotate(mask.unsqueeze(0), angle, interpolation=InterpolationMode.NEAREST, fill=0).squeeze(0)
        
        # Color jitter
        if any([self.brightness_jitter, self.contrast_jitter, self.saturation_jitter, self.hue_jitter]):
            from torchvision.transforms import ColorJitter
            jitter = ColorJitter(
                brightness=self.brightness_jitter,
                contrast=self.contrast_jitter,
                saturation=self.saturation_jitter,
                hue=self.hue_jitter
            )
            image = jitter(image)
        
        # Normalize
        if self.normalize:
            image = TF.normalize(image, self.mean, self.std)
        
        # Update sample
        sample['image'] = image
        if mask is not None:
            sample['mask'] = mask.long()
        
        return sample


class ValTransforms(TiledTransform):
    """
    Validation transforms with only normalization.
    """
    
    def __init__(
        self,
        resize: Optional[tuple] = None,
        normalize: bool = True,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        """
        Initialize validation transforms.
        
        Args:
            resize: Optional resize dimensions (height, width)
            normalize: Whether to normalize
            mean: Normalization mean
            std: Normalization std
        """
        super().__init__("val")
        self.resize = resize
        self.normalize = normalize
        self.mean = mean
        self.std = std
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply validation transforms."""
        image = sample['image']
        mask = sample.get('mask')
        
        # Resize if specified
        if self.resize is not None:
            image = TF.resize(image, self.resize, interpolation=InterpolationMode.BILINEAR)
            if mask is not None:
                mask = TF.resize(mask.unsqueeze(0), self.resize, interpolation=InterpolationMode.NEAREST).squeeze(0)
        
        # Normalize
        if self.normalize:
            image = TF.normalize(image, self.mean, self.std)
        
        # Update sample
        sample['image'] = image
        if mask is not None:
            sample['mask'] = mask.long()
        
        return sample


class TestTransforms(TiledTransform):
    """
    Test transforms with only normalization (same as validation).
    """
    
    def __init__(
        self,
        resize: Optional[tuple] = None,
        normalize: bool = True,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        """
        Initialize test transforms.
        
        Args:
            resize: Optional resize dimensions (height, width)
            normalize: Whether to normalize
            mean: Normalization mean
            std: Normalization std
        """
        super().__init__("test")
        self.resize = resize
        self.normalize = normalize
        self.mean = mean
        self.std = std
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply test transforms."""
        image = sample['image']
        
        # Resize if specified
        if self.resize is not None:
            image = TF.resize(image, self.resize, interpolation=InterpolationMode.BILINEAR)
        
        # Normalize
        if self.normalize:
            image = TF.normalize(image, self.mean, self.std)
        
        # Update sample
        sample['image'] = image
        
        return sample


class InferenceTransforms(TiledTransform):
    """
    Inference transforms with minimal processing.
    """
    
    def __init__(
        self,
        resize: Optional[tuple] = None,
        normalize: bool = True,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        """
        Initialize inference transforms.
        
        Args:
            resize: Optional resize dimensions (height, width)
            normalize: Whether to normalize
            mean: Normalization mean
            std: Normalization std
        """
        super().__init__("inference")
        self.resize = resize
        self.normalize = normalize
        self.mean = mean
        self.std = std
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply inference transforms."""
        image = sample['image']
        
        # Resize if specified
        if self.resize is not None:
            image = TF.resize(image, self.resize, interpolation=InterpolationMode.BILINEAR)
        
        # Normalize
        if self.normalize:
            image = TF.normalize(image, self.mean, self.std)
        
        # Update sample
        sample['image'] = image
        
        return sample


def get_transforms(
    mode: str = "train",
    resize: Optional[tuple] = None,
    **kwargs
) -> TiledTransform:
    """
    Get transforms for specified mode.
    
    Args:
        mode: Transform mode ('train', 'val', 'test', 'inference')
        resize: Optional resize dimensions
        **kwargs: Additional arguments for specific transforms
        
    Returns:
        Transform object
    """
    if mode == "train":
        return TrainTransforms(resize=resize, **kwargs)
    elif mode == "val":
        return ValTransforms(resize=resize, **kwargs)
    elif mode == "test":
        return TestTransforms(resize=resize, **kwargs)
    elif mode == "inference":
        return InferenceTransforms(resize=resize, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# Forest-specific transforms (converting to binary forest/background)
class ForestBinaryTransform(TiledTransform):
    """
    Transform to convert multi-class masks to binary forest/background.
    """
    
    def __init__(self, forest_class_id: int = 6, mode: str = "train"):
        """
        Initialize forest binary transform.
        
        Args:
            forest_class_id: Class ID for forest in original mask
            mode: Base transform mode
        """
        super().__init__(mode)
        self.forest_class_id = forest_class_id
        self.base_transform = get_transforms(mode)
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply forest binary transform."""
        # Convert mask to binary if present
        if 'mask' in sample:
            mask = sample['mask']
            # Convert to binary: forest=1, else=0
            binary_mask = (mask == self.forest_class_id).long()
            sample['mask'] = binary_mask
        
        # Apply base transforms
        return self.base_transform(sample)
