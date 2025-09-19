from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import DeepGlobeForestBinaryTransform


def find_image_mask_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """Find pairs like *_sat.jpg with *_mask.png under `root` recursively."""
    root = Path(root)
    pairs: List[Tuple[Path, Path]] = []
    for sat in root.rglob("*_sat.jpg"):
        mask = sat.with_name(sat.name.replace("_sat.jpg", "_mask.png"))
        if mask.exists():
            pairs.append((sat, mask))
    if not pairs:
        # Fallback: look for parallel dirs (images in 'images', masks in 'masks')
        imgd = root / "images"
        maskd = root / "masks"
        if imgd.exists() and maskd.exists():
            for sat in sorted(imgd.rglob("*.jpg")):
                m = maskd / (sat.stem + ".png")
                if m.exists():
                    pairs.append((sat, m))
    if not pairs:
        raise FileNotFoundError(f"No DeepGlobe image/mask pairs found under {root}")
    return pairs


def build_grid_coords(h: int, w: int, tile: int, stride: Optional[int] = None) -> List[Tuple[int, int]]:
    stride = stride or tile
    ys: List[int] = []
    xs: List[int] = []
    y = 0
    while True:
        if y + tile >= h:
            ys.append(max(0, h - tile))
            break
        ys.append(y)
        y += stride
    x = 0
    while True:
        if x + tile >= w:
            xs.append(max(0, w - tile))
            break
        xs.append(x)
        x += stride
    return [(yy, xx) for yy in ys for xx in xs]


class DeepGlobeTiledDataset(Dataset):
    """Grid-tiled DeepGlobe dataset.

    Args:
        root:           path to DeepGlobe root (folder that contains *_sat.jpg & *_mask.png or images/masks dirs)
        split:          'train'|'val'|'test' (stratified by filename hash)
        tile_size:      output tile size (square)
        stride:         grid stride in pixels (defaults to tile_size)
        transforms:     callable returning (image_tensor, mask_tensor)
        return_metadata: if True, returns dict with paths and coords
    """
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        tile_size: int = 512,
        stride: Optional[int] = None,
        transforms: Optional[DeepGlobeForestBinaryTransform] = None,
        return_metadata: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        assert split in {"train", "val", "test"}
        self.split = split
        self.tile_size = int(tile_size)
        self.stride = int(stride or tile_size)
        self.transforms = transforms or DeepGlobeForestBinaryTransform("train" if split == "train" else "val")
        self.return_metadata = return_metadata

        all_pairs = find_image_mask_pairs(self.root)
        # Deterministic split via hash of filename
        def split_of(path: Path) -> str:
            h = abs(hash(path.stem)) % 100
            if h < 70:
                return "train"
            elif h < 85:
                return "val"
            else:
                return "test"
        pairs = [(i, m) for (i, m) in all_pairs if split_of(i) == split]
        if not pairs:
            raise RuntimeError(f"No pairs for split={split}")

        # Build tile index
        self.index: List[Tuple[Path, Path, int, int]] = []  # (img_path, mask_path, y, x)
        # Probe first image for dims; DeepGlobe images are large (~2448x2448)
        for img_path, mask_path in pairs:
            im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            ms = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            if im is None or ms is None:
                continue
            h, w = im.shape[:2]
            coords = build_grid_coords(h, w, self.tile_size, self.stride)
            for (yy, xx) in coords:
                self.index.append((img_path, mask_path, yy, xx))

        if not self.index:
            raise RuntimeError("Empty tile index — check your DeepGlobe data paths.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        img_path, mask_path, yy, xx = self.index[idx]
        tile = self.tile_size

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # BGR
        mask_rgb = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if img is None or mask_rgb is None:
            raise FileNotFoundError(img_path)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)

        H, W = img.shape[:2]
        y0 = min(yy, max(0, H - tile))
        x0 = min(xx, max(0, W - tile))
        img_tile = img[y0:y0 + tile, x0:x0 + tile]
        mask_tile = mask_rgb[y0:y0 + tile, x0:x0 + tile]

        if img_tile.shape[0] != tile or img_tile.shape[1] != tile:
            # pad if something went wrong
            img_tile = cv2.copyMakeBorder(img_tile, 0, tile - img_tile.shape[0], 0, tile - img_tile.shape[1], cv2.BORDER_REFLECT_101)
            mask_tile = cv2.copyMakeBorder(mask_tile, 0, tile - mask_tile.shape[0], 0, tile - mask_tile.shape[1], cv2.BORDER_REFLECT_101)

        image_t, mask_t = self.transforms(img_tile, mask_tile)
        if self.return_metadata:
            return {
                "image": image_t,
                "mask": mask_t.long(),
                "meta": {
                    "img_path": str(img_path),
                    "mask_path": str(mask_path),
                    "y": int(y0),
                    "x": int(x0),
                }
            }
        return {"image": image_t, "mask": mask_t.long()}