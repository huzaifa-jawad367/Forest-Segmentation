#!/usr/bin/env python3
"""
Test script to verify the metric fixes work correctly.
"""

import numpy as np
import torch
from Model.metric import compute_metrics

def test_compute_metrics():
    """Test the compute_metrics function with different input shapes."""
    print("Testing compute_metrics function...")
    
    # Test case 1: Normal case with logits
    batch_size, num_classes, height, width = 2, 2, 64, 64
    predictions = np.random.randn(batch_size, num_classes, height, width)
    labels = np.random.randint(0, 2, (batch_size, height, width))
    
    print(f"Test 1 - Logits input:")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    try:
        metrics = compute_metrics((predictions, labels))
        print(f"  ✅ Success! Metrics: {list(metrics.keys())}")
        print(f"  eval_loss: {metrics.get('eval_loss', 'N/A')}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test case 2: Already class predictions
    predictions_classes = np.random.randint(0, 2, (batch_size, height, width))
    labels2 = np.random.randint(0, 2, (batch_size, height, width))
    
    print(f"\nTest 2 - Class predictions input:")
    print(f"  Predictions shape: {predictions_classes.shape}")
    print(f"  Labels shape: {labels2.shape}")
    
    try:
        metrics = compute_metrics((predictions_classes, labels2))
        print(f"  ✅ Success! Metrics: {list(metrics.keys())}")
        print(f"  eval_loss: {metrics.get('eval_loss', 'N/A')}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test case 3: Shape mismatch (should handle gracefully)
    predictions_mismatch = np.random.randint(0, 2, (batch_size, height, width))
    labels_mismatch = np.random.randint(0, 2, (batch_size * height * width,))  # Flattened
    
    print(f"\nTest 3 - Shape mismatch (should handle gracefully):")
    print(f"  Predictions shape: {predictions_mismatch.shape}")
    print(f"  Labels shape: {labels_mismatch.shape}")
    
    try:
        metrics = compute_metrics((predictions_mismatch, labels_mismatch))
        print(f"  ✅ Success! Metrics: {list(metrics.keys())}")
        print(f"  eval_loss: {metrics.get('eval_loss', 'N/A')}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_compute_metrics()