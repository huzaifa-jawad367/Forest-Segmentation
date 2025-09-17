"""
Training script for Custom Segformer using Hugging Face Trainer.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import SemanticSegmenterOutput
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


class SegformerTrainer(Trainer):
    """Custom Trainer for Segformer semantic segmentation."""
    
    def __init__(self, model, args, train_dataset, eval_dataset, compute_metrics=None, **kwargs):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            **kwargs
        )
        
        # Use focal loss for imbalanced dataset
        self.loss_fn = FocalLoss2d(gamma=2, weight=None)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute custom loss for semantic segmentation."""
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(inputs["pixel_values"])
        # CustomSegformer returns a tuple (logits, softmax), use the first one
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Compute loss
        loss = self.loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step for segmentation."""
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            outputs = model(inputs["pixel_values"])
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs)
                return (loss, None, None)
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
            # Get labels
            labels = inputs.get("labels")
            
            return (None, predictions, labels)


def create_datasets():
    """Create training and validation datasets."""
    print("Creating datasets...")
    
    # Target pixel counts
    target_foreground = 330_000_000
    target_background = 330_000_000
    
    # Create original datasets
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
    
    # Create balanced datasets
    balanced_train = PreciseBalancedDataset(
        train_dataset,
        target_foreground_pixels=target_foreground,
        target_background_pixels=target_background,
        max_iterations=1000,
        tolerance=0.02,
        device='cpu'
    )
    
    balanced_val = PreciseBalancedDataset(
        val_dataset,
        target_foreground_pixels=target_foreground // 5,
        target_background_pixels=target_background // 5,
        max_iterations=500,
        tolerance=0.05,
        device='cpu'
    )
    
    print(f"Training dataset: {len(balanced_train)} tiles")
    print(f"Validation dataset: {len(balanced_val)} tiles")
    
    return balanced_train, balanced_val


def collate_fn(batch):
    """Custom collate function for segmentation data."""
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    
    return {
        'pixel_values': images,
        'labels': masks
    }


def main():
    """Main training function."""
    print("Starting Segformer Training")
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
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./segformer_results",
        num_train_epochs=50,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=True,
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Create trainer
    trainer = SegformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model("./segformer_final")
    print("Training completed! Model saved to ./segformer_final")


if __name__ == "__main__":
    main()
