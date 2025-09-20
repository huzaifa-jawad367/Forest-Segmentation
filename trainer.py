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

from Model.loss import FocalLoss2d
from Model.metric import compute_metrics


class CustomTrainer:
    """Custom trainer for semantic segmentation without Hugging Face Trainer."""
    
    def __init__(self, model, train_dataset, val_dataset, device='cuda', 
                 batch_size=8, num_workers=4, learning_rate=5e-5, 
                 weight_decay=0.01, num_epochs=50, save_dir="./custom_results"):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
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
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
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
        
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def val_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
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
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Compute loss
                loss = self.loss_fn(logits, labels)
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                
                # Store for metrics
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                
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
        
        # Compute metrics
        if all_predictions and all_labels:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Compute metrics
            metrics = compute_metrics((all_predictions, all_labels))
            self.val_metrics.append(metrics)
            
            return avg_loss, metrics
        else:
            return avg_loss, {}
    
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
