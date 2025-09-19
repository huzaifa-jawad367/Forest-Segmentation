from __future__ import annotations
import numpy as np
import cv2
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception as e:
    raise RuntimeError("Please `pip install albumentations`.") from e

# DeepGlobe RGB palette → class ids (7 classes)
DG_COLORS = {
    (0, 255, 255): 0,  # urban
    (255, 255, 0): 1,  # agriculture
    (255, 0, 255): 2,  # rangeland
    (0, 255, 0): 3,    # forest
    (0, 0, 255): 4,    # water
    (255, 255, 255): 5,# barren
    (0, 0, 0): 6       # unknown
}

PALETTE = np.array(list(DG_COLORS.keys()), dtype=np.uint8)        # (7,3)
PALETTE_IDS = np.array(list(DG_COLORS.values()), dtype=np.uint8)  # (7,)
FOREST_CLASS_ID = 3


def rgb_mask_to_labels(mask_rgb: np.ndarray) -> np.ndarray:
    """Map DeepGlobe RGB-coded mask to integer label map [H,W].
    Robust to slight JPEG artifacts via nearest palette color (L1).
    """
    h, w, _ = mask_rgb.shape
    mask_flat = mask_rgb.reshape(-1, 3)
    color_int = (mask_flat[:, 0].astype(np.int32) * 65536 +
                 mask_flat[:, 1].astype(np.int32) * 256 +
                 mask_flat[:, 2].astype(np.int32))
    pal_int = (PALETTE[:, 0].astype(np.int32) * 65536 +
               PALETTE[:, 1].astype(np.int32) * 256 +
               PALETTE[:, 2].astype(np.int32))
    lookup = {c: i for i, c in enumerate(pal_int)}
    idx = np.array([lookup.get(ci, -1) for ci in color_int], dtype=np.int32)
    out = np.empty((h * w,), dtype=np.uint8)
    hit = idx >= 0
    out[hit] = PALETTE_IDS[idx[hit]]
    if (~hit).any():
        miss = (~hit).nonzero()[0]
        miss_cols = mask_flat[miss].astype(np.int16)
        diff = miss_cols[:, None, :] - PALETTE[None, :, :].astype(np.int16)
        dist = np.abs(diff).sum(axis=2)
        nearest = dist.argmin(axis=1)
        out[miss] = PALETTE_IDS[nearest]
    return out.reshape(h, w)


def labels_to_binary_forest(labels: np.ndarray, forest_id: int = FOREST_CLASS_ID) -> np.ndarray:
    return (labels == forest_id).astype(np.uint8)


class DeepGlobeForestBinaryTransform:
    """Albumentations pipeline for DeepGlobe tiles (binary forest vs. background).

    Returns dict with:
      - image: FloatTensor [3,H,W] normalized to ImageNet
      - mask:  LongTensor  [H,W] values {0,1}
    """
    def __init__(self, mode: str = "train"):
        assert mode in {"train", "val", "test"}
        if mode == "train":
            self.tf = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.ColorJitter(p=0.2),
                A.GaussianBlur(p=0.1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __call__(self, img_tile_bgr: np.ndarray, mask_tile_rgb: np.ndarray):
        # Expect img in BGR (from cv2), convert to RGB for normalization
        img_rgb = cv2.cvtColor(img_tile_bgr, cv2.COLOR_BGR2RGB)
        labels = rgb_mask_to_labels(mask_tile_rgb)
        binary = labels_to_binary_forest(labels)
        out = self.tf(image=img_rgb, mask=binary)
        return out["image"], out["mask"]