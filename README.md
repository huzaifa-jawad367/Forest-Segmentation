# Tiled Aerial Image Dataset

A PyTorch dataset implementation for handling large aerial images by automatically tiling them into smaller patches. This implementation is designed for forest segmentation tasks but can be adapted for other semantic segmentation problems.

## Features

- **Automatic Tiling**: Splits 1024×1024 images into 512×512 tiles during loading
- **Flexible Transforms**: Different transform modes for training, validation, and inference
- **Metadata Support**: Tracks tile coordinates and image information for reconstruction
- **Forest Segmentation**: Built-in support for binary forest/background classification
- **Memory Efficient**: Loads only the tiles you need, not entire images

## Directory Structure

Your dataset should be organized as follows (LoveDA structure):

```
dataset_root/
├── Train/
│   └── Train/
│       ├── Urban/
│       │   ├── images_png/
│       │   └── masks_png/
│       └── Rural/
│           ├── images_png/
│           └── masks_png/
├── Val/
│   └── Val/
│       ├── Urban/
│       │   ├── images_png/
│       │   └── masks_png/
│       └── Rural/
│           ├── images_png/
│           └── masks_png/
└── Test/
    └── Test/
        ├── Urban/
        │   └── images_png/
        └── Rural/
            └── images_png/
```

## Quick Start

### Basic Usage

```python
from tiled_dataset import TiledAerialDataset
from transforms import ForestBinaryTransform
from data_utils import create_data_loaders

# Create data loaders
data_loaders = create_data_loaders(
    root="/Users/huzaifajawad/Research/Forest paper/LoveDa",
    batch_size=32,
    num_workers=4,
    forest_class_id=6  # Forest class ID in your masks
)

# Use in training loop
for batch in data_loaders['train']:
    images = batch['image']  # (B, 3, 512, 512)
    masks = batch['mask']    # (B, 512, 512)
    # ... your training code
```

### Custom Dataset Creation

```python
from tiled_dataset import TiledAerialDataset
from transforms import get_transforms

# Create dataset with custom settings
dataset = TiledAerialDataset(
    root="/path/to/dataset",
    split="train",
    tile_size=512,
    original_size=1024,
    transforms=get_transforms("train"),
    return_metadata=False
)

print(f"Dataset has {len(dataset)} tiles from {len(dataset.files)} images")
```

### Inference with Reconstruction

```python
from data_utils import create_inference_dataset

# Create inference dataset
inference_dataset = create_inference_dataset(
    root="/path/to/dataset",
    split="test",
    return_metadata=True
)

# Process all tiles for an image
image_idx = 0
tiles_per_image = inference_dataset.tiles_per_image
predictions = []

for tile_idx in range(tiles_per_image):
    global_idx = image_idx * tiles_per_image + tile_idx
    sample = inference_dataset[global_idx]
    
    # Your model prediction here
    tile_pred = model(sample['image'].unsqueeze(0))
    predictions.append(tile_pred)

# Reconstruct full image prediction
full_prediction = inference_dataset.reconstruct_prediction(predictions, image_idx)
```

## File Descriptions

### `tiled_dataset.py`
- **`TiledAerialDataset`**: Main dataset class that handles tiling and loading
- Automatically splits large images into smaller tiles
- Supports different splits (train/val/test)
- Tracks metadata for tile reconstruction

### `transforms.py`
- **`TrainTransforms`**: Augmentation transforms for training
- **`ValTransforms`**: Validation transforms (normalization only)
- **`TestTransforms`**: Test transforms (normalization only)
- **`ForestBinaryTransform`**: Converts multi-class masks to binary forest/background

### `data_utils.py`
- **`create_data_loaders()`**: Creates train/val/test data loaders
- **`create_inference_dataset()`**: Creates dataset for inference
- **`count_class_pixels()`**: Analyzes class distribution
- **`visualize_sample()`**: Visualizes dataset samples

### `example_usage.py`
- Complete examples showing how to use the dataset
- Training loop example
- Inference and reconstruction example

## Configuration Options

### Dataset Parameters
- `tile_size`: Size of each tile (default: 512)
- `original_size`: Size of original images (default: 1024)
- `forest_class_id`: Class ID for forest in masks (default: 6)

### Transform Parameters
- `resize`: Optional resize dimensions for tiles
- `horizontal_flip_prob`: Probability of horizontal flip (training)
- `rotation_prob`: Probability of rotation (training)
- `brightness_jitter`: Brightness jitter factor (training)

### DataLoader Parameters
- `batch_size`: Batch size (default: 32)
- `num_workers`: Number of worker processes (default: 4)
- `pin_memory`: Pin memory for faster GPU transfer (default: True)

## Advanced Usage

### Custom Transforms

```python
from transforms import TrainTransforms

# Custom training transforms
custom_transforms = TrainTransforms(
    resize=(256, 256),
    horizontal_flip_prob=0.7,
    rotation_prob=0.5,
    rotation_degrees=(-30, 30),
    brightness_jitter=0.3
)

dataset = TiledAerialDataset(
    root="/path/to/dataset",
    transforms=custom_transforms
)
```

### Balanced Subset Creation

```python
from data_utils import create_balanced_subset

# Create balanced subset
balanced_dataset = create_balanced_subset(
    dataset,
    target_ratio=0.5,  # 50% forest, 50% background
    max_images=1000
)
```

### Class Distribution Analysis

```python
from data_utils import count_class_pixels

# Analyze class distribution
stats = count_class_pixels(dataset)
print(f"Forest ratio: {stats['forest_ratio']:.3f}")
print(f"Total pixels: {stats['total_pixels']:,}")
```

## Performance Tips

1. **Use appropriate batch size**: Larger batches are more efficient but require more memory
2. **Adjust num_workers**: More workers can speed up loading but use more CPU
3. **Use pin_memory**: Enable for faster GPU transfer
4. **Consider subset_ratio**: Use for quick testing during development

## Troubleshooting

### Common Issues

1. **File not found errors**: Check that your directory structure matches the expected format
2. **Memory issues**: Reduce batch_size or use subset_ratio for testing
3. **Slow loading**: Increase num_workers or check disk I/O performance

### Debug Mode

```python
# Enable metadata to debug tile coordinates
dataset = TiledAerialDataset(
    root="/path/to/dataset",
    return_metadata=True
)

sample = dataset[0]
print(f"Metadata: {sample['metadata']}")
```

## Integration with Existing Code

This dataset is designed to be a drop-in replacement for standard PyTorch datasets. You can use it with:

- Any PyTorch model
- Standard training loops
- Existing loss functions
- Popular frameworks like PyTorch Lightning

The key difference is that instead of loading full images, you get tiles that can be processed independently and reconstructed later if needed.
