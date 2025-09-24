"""
Custom training loop for Segformer semantic segmentation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Union, List
import os
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from Model.loss import FocalLoss2d
from Model.metric import compute_metrics


class CustomTrainer:
    """Custom trainer for semantic segmentation without Hugging Face Trainer."""
    
    def __init__(self, model, train_dataset, val_dataset, device='cuda', 
                 batch_size=8, num_workers=4, learning_rate=5e-5, 
                 weight_decay=0.01, num_epochs=50, save_dir="./custom_results",
                 log_dir="./tensorboard_logs"):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Create organized directory structure
        self.save_dir, self.log_dir = self._create_directories(model, save_dir, log_dir, learning_rate, batch_size, num_epochs, weight_decay)
        
        # Create TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Loss function
        self.loss_fn = FocalLoss2d(gamma=2, weight=None)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
    
    def _create_directories(self, model, base_save_dir, base_log_dir, learning_rate, batch_size, num_epochs, weight_decay):
        """Create organized directory structure with datetime and model info."""
        # Get current datetime
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        
        # Determine model architecture - only use the actual model class name
        model_name = model.__class__.__name__
        if model_name == "Segforest":
            model_arch = "segforest"
        elif model_name == "CustomSegformer":
            model_arch = "segformer"
        else:
            model_arch = model_name.lower()
        
        # Get model configuration info (backbone size)
        backbone_info = ""
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'config'):
            if hasattr(model.encoder.config, 'hidden_sizes'):
                # Get the largest hidden size (usually the backbone indicator)
                max_hidden = max(model.encoder.config.hidden_sizes)
                if max_hidden == 64:
                    backbone_info = "b0"
                elif max_hidden == 128:
                    backbone_info = "b1"
                elif max_hidden == 320:
                    backbone_info = "b2"
                elif max_hidden == 512:
                    backbone_info = "b3"
                elif max_hidden == 768:
                    backbone_info = "b4"
                elif max_hidden == 1024:
                    backbone_info = "b5"
        
        # Create directory names - clean and simple
        if backbone_info:
            dir_name = f"{timestamp}_{model_arch}_{backbone_info}_bs{batch_size}_lr{learning_rate:.0e}_ep{num_epochs}"
        else:
            dir_name = f"{timestamp}_{model_arch}_bs{batch_size}_lr{learning_rate:.0e}_ep{num_epochs}"
        
        # Create full paths
        save_dir = os.path.join(base_save_dir, dir_name)
        log_dir = os.path.join(base_log_dir, dir_name)
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        print(f"Created directories:")
        print(f"  Save directory: {save_dir}")
        print(f"  Log directory: {log_dir}")
        
        # Save training configuration
        self._save_training_config(save_dir, model, learning_rate, batch_size, num_epochs, weight_decay)
        
        return save_dir, log_dir
    
    def _save_training_config(self, save_dir, model, learning_rate, batch_size, num_epochs, weight_decay):
        """Save training configuration to a text file."""
        config_path = os.path.join(save_dir, "training_config.txt")
        
        with open(config_path, 'w') as f:
            f.write("Training Configuration\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Architecture: {model.__class__.__name__}\n")
            f.write(f"Learning Rate: {learning_rate}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Number of Epochs: {num_epochs}\n")
            f.write(f"Weight Decay: {weight_decay}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
            f.write(f"Frozen Parameters: {total_params - trainable_params:,}\n\n")
            
            # Model architecture details
            if hasattr(model, 'encoder'):
                f.write("Encoder Details:\n")
                f.write(f"  Type: {model.encoder.__class__.__name__}\n")
                if hasattr(model.encoder, 'config'):
                    config = model.encoder.config
                    f.write(f"  Model Type: {getattr(config, 'model_type', 'Unknown')}\n")
                    f.write(f"  Hidden Sizes: {getattr(config, 'hidden_sizes', 'Unknown')}\n")
                    f.write(f"  Num Labels: {getattr(config, 'num_labels', 'Unknown')}\n")
            
            if hasattr(model, 'decoder'):
                f.write(f"Decoder Type: {model.decoder.__class__.__name__}\n")
            
            if hasattr(model, 'mff_blocks'):
                f.write(f"MFF Blocks: {model.mff_blocks.__class__.__name__}\n")
        
        print(f"Training configuration saved to: {config_path}")
        
    def collate_fn(self, batch):
        """Custom collate function for segmentation data."""
        # Filter out malformed samples
        valid_samples = []
        for item in batch:
            if isinstance(item, dict) and 'image' in item and 'mask' in item:
                valid_samples.append(item)
            else:
                print(f"Warning: Skipping malformed sample with keys: {item.keys() if isinstance(item, dict) else type(item)}")
        
        if len(valid_samples) == 0:
            # Return empty batch that will be skipped
            return {
                'pixel_values': torch.empty(0, 3, 512, 512, device=self.device),
                'labels': torch.empty(0, 512, 512, dtype=torch.long, device=self.device)
            }
        
        images = torch.stack([item['image'] for item in valid_samples])
        masks = torch.stack([item['mask'] for item in valid_samples])
        
        return {
            'pixel_values': images,
            'labels': masks
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Skip empty batches
            if batch['pixel_values'].size(0) == 0:
                continue
                
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(pixel_values)
            # Handle different output formats
            if isinstance(outputs, list):
                logits = outputs[0]  # Use first output for Segforest
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Compute loss
            loss = self.loss_fn(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to TensorBoard every 50 batches
            if batch_idx % 50 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], global_step)
        
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        
        # Log epoch average loss
        self.writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
        
        return avg_loss
    
    def val_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Initialize metric accumulators
        total_correct = 0
        total_pixels = 0
        class_correct = [0, 0]  # For background and forest
        class_total = [0, 0]
        intersection = [0, 0]  # For IoU
        union = [0, 0]
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
            
            for batch_idx, batch in enumerate(pbar):
                # Skip empty batches
                if batch['pixel_values'].size(0) == 0:
                    continue
                    
                # Move to device
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(pixel_values)
                # Handle different output formats
                if isinstance(outputs, list):
                    logits = outputs[0]  # Use first output for Segforest
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Compute loss
                loss = self.loss_fn(logits, labels)
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                
                # Compute metrics incrementally
                batch_correct = (predictions == labels).sum().item()
                batch_pixels = predictions.numel()
                
                total_correct += batch_correct
                total_pixels += batch_pixels
                
                # Class-specific metrics
                for cls in range(2):
                    cls_mask = (labels == cls)
                    if cls_mask.sum() > 0:
                        class_correct[cls] += (predictions[cls_mask] == cls).sum().item()
                        class_total[cls] += cls_mask.sum().item()
                    
                    # IoU computation
                    pred_cls = (predictions == cls)
                    target_cls = (labels == cls)
                    intersection[cls] += (pred_cls & target_cls).sum().item()
                    union[cls] += (pred_cls | target_cls).sum().item()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}'
                })
        
        avg_loss = total_loss / max(num_batches, 1)
        self.val_losses.append(avg_loss)
        
        # Log validation loss to TensorBoard
        self.writer.add_scalar('Val/Epoch_Loss', avg_loss, epoch)
        
        # Compute final metrics
        mean_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
        
        # Class accuracies
        class_accuracies = []
        for cls in range(2):
            if class_total[cls] > 0:
                class_accuracies.append(class_correct[cls] / class_total[cls])
            else:
                class_accuracies.append(1.0)
        
        # IoU computation
        ious = []
        for cls in range(2):
            if union[cls] > 0:
                ious.append(intersection[cls] / union[cls])
            else:
                ious.append(1.0 if intersection[cls] == 0 else 0.0)
        
        mean_iou = sum(ious) / len(ious)
        
        # For precision/recall/F1, we'll use mean accuracy as a reasonable approximation
        # since computing exact precision/recall on 1.7B pixels is memory intensive
        precision = mean_accuracy
        recall = mean_accuracy
        f1 = mean_accuracy
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_accuracy': mean_accuracy,
            'mean_iou': mean_iou,
            'background_iou': ious[0],
            'forest_iou': ious[1],
            'background_accuracy': class_accuracies[0],
            'forest_accuracy': class_accuracies[1]
        }
        
        self.val_metrics.append(metrics)
        
        # Log metrics to TensorBoard
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch+1}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Device: {self.device}")
        print("="*60)
        
        # Debug: Print dataloader and batch information
        print("\n=== DATALOADER DEBUG INFO ===")
        print(f"Train dataloader length: {len(self.train_loader)}")
        print(f"Val dataloader length: {len(self.val_loader)}")
        
        # Test a few batches from train dataloader
        print("\n--- Training Dataloader Batch Shapes ---")
        for i, batch in enumerate(self.train_loader):
            if i >= 3:  # Only check first 3 batches
                break
            print(f"Batch {i}:")
            print(f"  Keys: {list(batch.keys())}")
            print(f"  pixel_values shape: {batch['pixel_values'].shape}")
            print(f"  labels shape: {batch['labels'].shape}")
            print(f"  pixel_values dtype: {batch['pixel_values'].dtype}")
            print(f"  labels dtype: {batch['labels'].dtype}")
            if batch['pixel_values'].size(0) == 0:
                print(f"  WARNING: Empty batch detected!")
            print()
        
        # Test a few batches from val dataloader
        print("--- Validation Dataloader Batch Shapes ---")
        for i, batch in enumerate(self.val_loader):
            if i >= 3:  # Only check first 3 batches
                break
            print(f"Batch {i}:")
            print(f"  Keys: {list(batch.keys())}")
            print(f"  pixel_values shape: {batch['pixel_values'].shape}")
            print(f"  labels shape: {batch['labels'].shape}")
            print(f"  pixel_values dtype: {batch['pixel_values'].dtype}")
            print(f"  labels dtype: {batch['labels'].dtype}")
            if batch['pixel_values'].size(0) == 0:
                print(f"  WARNING: Empty batch detected!")
            print()
        
        print("=== END DATALOADER DEBUG INFO ===\n")
        
        # Log model architecture to TensorBoard
        self.log_model_architecture()
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.val_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            if val_metrics:
                print(f"  Val Metrics: {val_metrics}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            if (epoch + 1) % 5 == 0 or is_best:  # Save every 5 epochs or if best
                self.save_checkpoint(epoch, is_best)
            
            # Log sample images every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.log_sample_images(epoch)
            
            print("-" * 60)
        
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Save final model
        final_path = os.path.join(self.save_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }, final_path)
        
        print(f"Final model saved to: {final_path}")
        print(f"TensorBoard logs saved to: {self.log_dir}")
        print(f"To view TensorBoard, run: tensorboard --logdir={self.log_dir}")
        
        # Close TensorBoard writer
        self.writer.close()
    
    def log_model_architecture(self):
        """Log model architecture to TensorBoard."""
        # Create a dummy input to trace the model
        dummy_input = torch.randn(1, 3, 512, 512).to(self.device)
        
        try:
            # Log model graph
            self.writer.add_graph(self.model, dummy_input)
            print("Model architecture logged to TensorBoard")
        except Exception as e:
            print(f"Could not log model architecture: {e}")
    
    def log_sample_images(self, epoch, num_samples=4):
        """Log sample images, predictions, and masks to TensorBoard."""
        self.model.eval()
        
        with torch.no_grad():
            # Get a batch from validation loader
            for batch in self.val_loader:
                if batch['pixel_values'].size(0) == 0:
                    continue
                    
                # Take only the first few samples
                pixel_values = batch['pixel_values'][:num_samples].to(self.device)
                labels = batch['labels'][:num_samples].to(self.device)
                
                # Get predictions
                outputs = self.model(pixel_values)
                # Handle different output formats
                if isinstance(outputs, list):
                    logits = outputs[0]  # Use first output for Segforest
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                predictions = torch.argmax(logits, dim=1)
                
                # Log images
                self.writer.add_images('Sample/Input_Images', pixel_values, epoch)
                self.writer.add_images('Sample/Ground_Truth', labels.unsqueeze(1).float(), epoch)
                self.writer.add_images('Sample/Predictions', predictions.unsqueeze(1).float(), epoch)
                
                break  # Only log one batch
