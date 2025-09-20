"""
Training script for Custom Segformer using Custom Training Loop.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Union, List
import os
from pathlib import Path

# Import our custom components
# from Model.Segforest.Segforest import Segforest
# CustomSegformer
from Model.model import CustomSegformer
from Model.loss import FocalLoss2d, cross_entropy
from Model.metric import compute_metrics
from Dataset.LoveDa_Dataloader import TiledAerialDataset, ForestBinaryTransform
from Dataset.LoveDa_Dataloader.create_precise_balanced_dataset import PreciseBalancedDataset
from trainer import CustomTrainer


def create_datasets():
    """Create training and validation datasets."""
    print("Creating datasets...")
    
    # Target pixel counts
    target_foreground = 330_000_000
    target_background = 330_000_000
    
    # Create original datasets
    train_dataset = TiledAerialDataset(
        root="Data/loveda-dataset",
        split="train",
        tile_size=512,
        original_size=1024,
        transforms=ForestBinaryTransform(forest_class_id=6, mode="train"),
        return_metadata=False
    )
    
    val_dataset = TiledAerialDataset(
        root="Data/loveda-dataset",
        split="val",
        tile_size=512,
        original_size=1024,
        transforms=ForestBinaryTransform(forest_class_id=6, mode="val"),
        return_metadata=False
    )
    
    # # Create balanced datasets
    # balanced_train = PreciseBalancedDataset(
    #     train_dataset,
    #     target_foreground_pixels=target_foreground,
    #     target_background_pixels=target_background,
    #     max_iterations=1000,
    #     tolerance=0.02,
    #     device='cpu'
    # )
    
    # balanced_val = PreciseBalancedDataset(
    #     val_dataset,
    #     target_foreground_pixels=target_foreground // 5,
    #     target_background_pixels=target_background // 5,
    #     max_iterations=500,
    #     tolerance=0.05,
    #     device='cpu'
    # )
    
    # print(f"Training dataset: {len(balanced_train)} tiles")
    # print(f"Validation dataset: {len(balanced_val)} tiles")
    
    # return balanced_train, balanced_val

    print(f"Training dataset: {len(train_dataset)} tiles")
    print(f"Validation dataset: {len(val_dataset)} tiles")
    
    return train_dataset, val_dataset



def main():
    """Main training function."""
    print("Starting Segformer Training with Custom Loop")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset, val_dataset = create_datasets()
    
    # Create model
    print("Creating Custom Segformer model...")
    model = CustomSegformer(
        input_channels=3,  # RGB
        num_classes=2,     # Binary: forest/background
        base_model='nvidia/mit-b4'
    )
    # model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create custom trainer
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        batch_size=8,
        num_workers=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        num_epochs=50,
        save_dir="./custom_results"
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")



if __name__ == "__main__":
    main()