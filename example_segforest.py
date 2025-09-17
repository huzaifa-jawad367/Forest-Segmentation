#!/usr/bin/env python3
"""
Example script showing how to use the updated Segforest model with pretrained Segformer encoder.
"""

import torch
from Model.Segforest.Segforest import Segforest

def main():
    # Create Segforest model with pretrained encoder
    model = Segforest(
        img_size=512,  # Input image size
        in_chans=3,    # RGB images
        num_classes=2, # Binary segmentation (background, forest)
        pretrained_model_name="nvidia/mit-b4",  # Use pretrained Segformer-B4
        freeze_encoder=False  # Allow fine-tuning of encoder
    )
    
    print("✅ Segforest model created with pretrained Segformer encoder!")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🔧 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test with a sample input
    batch_size = 2
    sample_input = torch.randn(batch_size, 3, 512, 512)
    
    print(f"\n🧪 Testing with input shape: {sample_input.shape}")
    
    # Forward pass
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        outputs = model(sample_input)
    
    print(f"📤 Output shapes:")
    for i, output in enumerate(outputs):
        print(f"  Output {i+1}: {output.shape}")
    
    # Test training mode
    model.train()
    with torch.no_grad():
        train_outputs = model(sample_input)
    
    print(f"\n🏋️ Training mode outputs:")
    for i, output in enumerate(train_outputs):
        print(f"  Output {i+1}: {output.shape}")
    
    print("\n🎯 Model ready for training!")

if __name__ == "__main__":
    main()
