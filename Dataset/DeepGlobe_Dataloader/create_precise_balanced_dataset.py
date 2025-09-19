# Dataset/DeepGlobe_Dataloader/create_precise_balanced_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import math
import random
import numpy as np
import cv2
import torch
from tqdm import tqdm

# We read masks directly (RGB → labels → binary), so no torch/albu needed here.
from .transforms import rgb_mask_to_labels, FOREST_CLASS_ID


def _parse_index_entry(entry, fallback_tile: int) -> Tuple[Path, Path, int, int, int, int]:
    """
    Support both index layouts:
      - (img_path, mask_path, y, x)
      - (img_path, mask_path, y, x, h, w)
    """
    if len(entry) == 6:
        img_path, mask_path, yy, xx, hh, ww = entry
        return Path(img_path), Path(mask_path), int(yy), int(xx), int(hh), int(ww)
    elif len(entry) == 4:
        img_path, mask_path, yy, xx = entry
        return Path(img_path), Path(mask_path), int(yy), int(xx), int(fallback_tile), int(fallback_tile)
    else:
        raise RuntimeError(f"Unexpected dataset.index entry format: {entry}")


def _count_fg_bg_in_tile(mask_path: Path, y: int, x: int, h: int, w: int) -> Tuple[int, int]:
    m = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if m is None:
        return 0, 0
    m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
    tile = m[y:y + h, x:x + w]
    labels = rgb_mask_to_labels(tile)
    fg = int((labels == FOREST_CLASS_ID).sum())
    total = labels.size
    return fg, total - fg


def _compute_or_load_counts(base_ds, cache_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-tile foreground/background pixel counts (no augmentation),
    cache to .npz, and reload on subsequent runs.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        data = np.load(cache_path)
        return data["fg"].astype(np.int64), data["bg"].astype(np.int64)

    print(f"[PreciseBalancedDataset] Scanning tiles to compute FG/BG counts → {cache_path}")
    fg_counts = np.zeros(len(base_ds), dtype=np.int64)
    bg_counts = np.zeros(len(base_ds), dtype=np.int64)

    # Try to get a fallback tile size from dataset attributes (used if index lacks h,w)
    fallback_tile = getattr(base_ds, "tile_size", 0) or 0

    for i in tqdm(range(len(base_ds)), desc="FG/BG scan", unit="tile"):
        entry = base_ds.index[i]  # relies on DeepGlobeTiledDataset.index
        img_path, mask_path, yy, xx, hh, ww = _parse_index_entry(entry, fallback_tile)
        fg, bg = _count_fg_bg_in_tile(mask_path, yy, xx, hh, ww)
        fg_counts[i] = fg
        bg_counts[i] = bg

    np.savez_compressed(cache_path, fg=fg_counts, bg=bg_counts)
    return fg_counts, bg_counts


class _IndexSubset(torch.utils.data.Dataset):
    """Lightweight wrapper returning items from base dataset by a list of indices (can include duplicates)."""
    def __init__(self, base, indices: List[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[self.indices[i]]


class PreciseBalancedDataset(_IndexSubset):
    """
    Build a dataset whose *pixel totals* for foreground/background meet target counts
    within a tolerance. Uses original RGB masks (no aug) for exact per-tile counts.

    Strategy:
      1) Compute per-tile (fg, bg) counts once and cache.
      2) Select positive tiles (fg>0) in descending fg until FG target reached.
      3) Add negative tiles (fg=0) in descending bg until BG target reached.
      4) If targets can’t be met exactly, allow duplication (with replacement) up to max_iterations.

    This mirrors the Loveda balancer behavior while staying simple & fast.

    Args:
      base_dataset: DeepGlobeTiledDataset (or compatible with .index)
      target_foreground_pixels: int
      target_background_pixels: int
      max_iterations: cap for oversampling loops
      tolerance: allowed relative error (e.g., 0.02 = ±2%)
      cache_dir: where to store per-tile counts (npz)
      seed: RNG seed for any sampling with replacement
    """

    def __init__(
        self,
        base_dataset,
        target_foreground_pixels: int,
        target_background_pixels: int,
        max_iterations: int = 1000,
        tolerance: float = 0.02,
        cache_dir: str | Path = "DeepGlobe/.cache",
        seed: int = 42,
    ):
        self.base = base_dataset
        self.target_fg = int(target_foreground_pixels)
        self.target_bg = int(target_background_pixels)
        self.tol = float(tolerance)
        self.max_iter = int(max_iterations)
        self.rng = random.Random(seed)

        # Cache key includes tile/stride if available to avoid mismatches
        tile = getattr(base_dataset, "tile_size", None)
        stride = getattr(base_dataset, "stride", None)
        name = f"fgbg_counts_{len(base_dataset)}_tile{tile}_stride{stride}.npz"
        cache_path = Path(cache_dir) / name

        fg_counts, bg_counts = _compute_or_load_counts(base_dataset, cache_path)
        self.fg_counts = fg_counts
        self.bg_counts = bg_counts

        indices = self._build_indices()
        super().__init__(base_dataset, indices)

    # ------------ selection logic ------------
    def _within_tol(self, cur: int, target: int) -> bool:
        lo = target * (1.0 - self.tol)
        hi = target * (1.0 + self.tol)
        return lo <= cur <= hi

    def _build_indices(self) -> List[int]:
        fg = self.fg_counts
        bg = self.bg_counts
        pos_idx = np.where(fg > 0)[0].tolist()
        neg_idx = np.where(fg == 0)[0].tolist()

        # Sort: positives by descending fg; negatives by descending bg
        pos_idx.sort(key=lambda i: fg[i], reverse=True)
        neg_idx.sort(key=lambda i: bg[i], reverse=True)

        selected: List[int] = []
        cur_fg = 0
        cur_bg = 0

        # Phase 1: add positives until FG target hit (or within tolerance)
        for i in pos_idx:
            if self._within_tol(cur_fg, self.target_fg):
                break
            selected.append(i)
            cur_fg += int(fg[i])
            cur_bg += int(bg[i])

        # If still short on FG, oversample positives with replacement
        it = 0
        while not self._within_tol(cur_fg, self.target_fg) and it < self.max_iter and pos_idx:
            i = self.rng.choice(pos_idx)
            selected.append(i)
            cur_fg += int(fg[i]); cur_bg += int(bg[i])
            it += 1

        # Phase 2: add negatives until BG target hit (or within tolerance)
        for i in neg_idx:
            if self._within_tol(cur_bg, self.target_bg):
                break
            selected.append(i)
            cur_bg += int(bg[i])

        # If still short on BG, oversample negatives with replacement
        it = 0
        while not self._within_tol(cur_bg, self.target_bg) and it < self.max_iter and neg_idx:
            i = self.rng.choice(neg_idx)
            selected.append(i)
            cur_bg += int(bg[i])
            it += 1

        # Final sanity report (optional; keep concise)
        final_fg = int(np.sum(self.fg_counts[selected]))
        final_bg = int(np.sum(self.bg_counts[selected]))
        print(f"[PreciseBalancedDataset] Built subset: {len(selected):,} tiles  "
              f"(FG {final_fg:,}/{self.target_fg:,}, BG {final_bg:,}/{self.target_bg:,})  "
              f"tol={self.tol:.2%}")

        return selected
