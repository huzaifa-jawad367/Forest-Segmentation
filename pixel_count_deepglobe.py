#!/usr/bin/env python
"""
Pixel counting script for DeepGlobe (binary: forest vs background),
mirroring the Loveda-style progress + stats output.

It iterates DeepGlobe *tiles* (train/val) and counts mask pixels with tqdm.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
import torch
from tqdm import tqdm

from Dataset.DeepGlobe_Dataloader.dataset import DeepGlobeTiledDataset
from Dataset.DeepGlobe_Dataloader.transforms import DeepGlobeForestBinaryTransform


def count_pixels_in_dataset(dataset, dataset_name="Dataset") -> Dict:
    """Count background (0) and foreground (1) pixels with a live progress bar."""
    print(f"\nAnalyzing {dataset_name}...")
    print(f"Total samples (tiles): {len(dataset):,}")

    total_background = 0
    total_foreground = 0
    total_pixels = 0
    samples_with_foreground = 0

    pbar = tqdm(dataset, desc=f"Processing {dataset_name}", unit="tile")
    for sample in pbar:
        mask = sample["mask"]  # torch.LongTensor [H,W] with values {0,1}
        # Count pixels
        background_pixels = (mask == 0).sum().item()
        foreground_pixels = (mask == 1).sum().item()

        total_background += background_pixels
        total_foreground += foreground_pixels
        total_pixels += mask.numel()

        if foreground_pixels > 0:
            samples_with_foreground += 1

        # live stats
        denom = total_foreground + total_background
        ratio = (total_foreground / denom * 100.0) if denom > 0 else 0.0
        pbar.set_postfix({
            "FG": f"{total_foreground:,}",
            "BG": f"{total_background:,}",
            "FG%": f"{ratio:.2f}%"
        })

    stats = {
        "dataset_name": dataset_name,
        "total_samples": len(dataset),
        "samples_with_foreground": samples_with_foreground,
        "samples_without_foreground": len(dataset) - samples_with_foreground,
        "total_background_pixels": total_background,
        "total_foreground_pixels": total_foreground,
        "total_pixels": total_pixels,
        "foreground_ratio": (total_foreground / total_pixels) if total_pixels > 0 else 0.0,
        "background_ratio": (total_background / total_pixels) if total_pixels > 0 else 0.0,
        "foreground_per_sample": (total_foreground / max(1, len(dataset))),
        "background_per_sample": (total_background / max(1, len(dataset))),
    }
    return stats


def print_statistics(stats: Dict):
    print(f"\n{'='*60}")
    print(f"PIXEL STATISTICS - {stats['dataset_name'].upper()}")
    print(f"{'='*60}")
    print("Dataset Info:")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Samples with foreground: {stats['samples_with_foreground']:,}")
    print(f"  Samples without foreground: {stats['samples_without_foreground']:,}")
    if stats['total_samples'] > 0:
        print(f"  Foreground sample ratio: {stats['samples_with_foreground']/stats['total_samples']*100:.2f}%")

    print(f"\nPixel Counts:")
    print(f"  Total pixels: {stats['total_pixels']:,}")
    print(f"  Background pixels: {stats['total_background_pixels']:,}")
    print(f"  Foreground pixels: {stats['total_foreground_pixels']:,}")

    print(f"\nPixel Ratios:")
    print(f"  Background ratio: {stats['background_ratio']*100:.4f}%")
    print(f"  Foreground ratio: {stats['foreground_ratio']*100:.4f}%")

    print(f"\nPer Sample Averages:")
    print(f"  Background pixels per sample: {stats['background_per_sample']:,.0f}")
    print(f"  Foreground pixels per sample: {stats['foreground_per_sample']:,.0f}")
    print(f"{'='*60}")


def compare_datasets(train_stats: Dict, val_stats: Dict):
    print(f"\n{'='*60}")
    print("DATASET COMPARISON")
    print(f"{'='*60}")
    print("Sample Counts:")
    print(f"  Training: {train_stats['total_samples']:,} samples")
    print(f"  Validation: {val_stats['total_samples']:,} samples")
    if val_stats['total_samples'] > 0:
        print(f"  Ratio (Train/Val): {train_stats['total_samples']/val_stats['total_samples']:.2f}")

    print(f"\nForeground Ratios:")
    print(f"  Training: {train_stats['foreground_ratio']*100:.4f}%")
    print(f"  Validation: {val_stats['foreground_ratio']*100:.4f}%")
    print(f"  Difference: {(train_stats['foreground_ratio'] - val_stats['foreground_ratio'])*100:.4f}%")

    print(f"\nSamples with Foreground:")
    print(f"  Training: {train_stats['samples_with_foreground']:,} "
          f"({train_stats['samples_with_foreground']/max(1,train_stats['total_samples'])*100:.2f}%)")
    print(f"  Validation: {val_stats['samples_with_foreground']:,} "
          f"({val_stats['samples_with_foreground']/max(1,val_stats['total_samples'])*100:.2f}%)")
    print(f"{'='*60}")


def main():
    print("PIXEL COUNTING ANALYSIS (DeepGlobe tiles)")
    print("="*60)

    root = Path("DeepGlobe")
    if not root.exists():
        print(f"Error: dataset path '{root}' does not exist.")
        return

    # Use the *validation* transform (no random aug) for accurate counting
    tf = DeepGlobeForestBinaryTransform(mode="val")

    # NOTE: dataset constructor builds a tile index; print before/after so it doesn't look stuck
    print("Indexing TRAIN tiles... (this can take a minute on first run)")
    train_ds = DeepGlobeTiledDataset(root=root, split="train", tile_size=612, stride=612, transforms=tf, return_metadata=False)
    print(f"Train tiles: {len(train_ds):,}")

    print("Indexing VAL tiles...")
    val_ds = DeepGlobeTiledDataset(root=root, split="val", tile_size=612, stride=612, transforms=tf, return_metadata=False)
    print(f"Val tiles: {len(val_ds):,}")

    # Count pixels with progress bars
    train_stats = count_pixels_in_dataset(train_ds, "Training Dataset")
    val_stats = count_pixels_in_dataset(val_ds, "Validation Dataset")

    # Print
    print_statistics(train_stats)
    print_statistics(val_stats)
    compare_datasets(train_stats, val_stats)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    total_px = train_stats['total_pixels'] + val_stats['total_pixels']
    total_fg = train_stats['total_foreground_pixels'] + val_stats['total_foreground_pixels']
    if total_px > 0:
        print(f"Combined foreground ratio: {total_fg/total_px*100:.4f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Tip: run unbuffered for immediate output:  python -u pixel_count_deepglobe_tiles.py
    main()
