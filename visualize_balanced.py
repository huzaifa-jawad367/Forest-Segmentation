import matplotlib.pyplot as plt
import torch
from Dataset.LoveDa_Dataloader import TiledAerialDataset, ForestBinaryTransform
from Dataset.LoveDa_Dataloader.create_precise_balanced_dataset import PreciseBalancedDataset

def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def main():
    # Create datasets using the same approach as create_precise_balanced_dataset.py
    target_foreground = 330_000_000
    target_background = 330_000_000
    
    train_dataset = TiledAerialDataset(
        root="LoveDa",
        split="train",
        tile_size=512,
        original_size=1024,
        transforms=ForestBinaryTransform(forest_class_id=6, mode="train"),
        return_metadata=False
    )
    
    val_dataset = TiledAerialDataset(
        root="LoveDa",
        split="val",
        tile_size=512,
        original_size=1024,
        transforms=ForestBinaryTransform(forest_class_id=6, mode="val"),
        return_metadata=False
    )
    
    train = PreciseBalancedDataset(
        train_dataset,
        target_foreground_pixels=target_foreground,
        target_background_pixels=target_background,
        max_iterations=1000,
        tolerance=0.02,
        device='cpu'
    )
    
    val = PreciseBalancedDataset(
        val_dataset,
        target_foreground_pixels=target_foreground // 5,
        target_background_pixels=target_background // 5,
        max_iterations=500,
        tolerance=0.05,
        device='cpu'
    )
    
    print(f"Train: {len(train)} tiles")
    print(f"Val: {len(val)} tiles")
    
    # Plot 6 samples from train
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    
    for i in range(6):
        sample = train[i]
        image = sample['image']
        mask = sample['mask']
        
        print(f"Sample {i}: Image shape: {image.shape}, Mask shape: {mask.shape}")
        
        # Denormalize and plot image
        img_denorm = denormalize(image)
        img_np = img_denorm.permute(1, 2, 0).numpy()
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Image {i}")
        axes[0, i].axis('off')
        
        # Plot mask
        axes[1, i].imshow(mask.numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f"Mask {i}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
