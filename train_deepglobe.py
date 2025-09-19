#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import Trainer, TrainingArguments

from Dataset.DeepGlobe_Dataloader.dataset import DeepGlobeTiledDataset
from Dataset.DeepGlobe_Dataloader.transforms import DeepGlobeForestBinaryTransform
from Dataset.DeepGlobe_Dataloader.create_precise_balanced_dataset import PreciseBalancedDataset
from Model.metric import compute_metrics
from Model.loss import FocalLoss2d
from Model.model import CustomSegformer
try:
    from Model.Segforest.Segforest import Segforest as SegforestModel
    HAS_SEGFOREST = True
except Exception:
    HAS_SEGFOREST = False


class SegTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = FocalLoss2d(gamma=2, weight=None)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(inputs["pixel_values"])
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(inputs["pixel_values"])
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs)
                return (loss, None, None)
            preds = torch.argmax(logits, dim=1)
            labels = inputs.get("labels")
            return (None, preds, labels)


def build_datasets(root: str | Path, tile_size: int, stride: int, balance_train: bool = True,
                   balance_val: bool = False, target_pixels: int = 300_000_000):
    tf_train = DeepGlobeForestBinaryTransform("train")
    tf_val   = DeepGlobeForestBinaryTransform("val")

    ds_train = DeepGlobeTiledDataset(root=root, split="train", tile_size=tile_size, stride=stride, transforms=tf_train)
    ds_val   = DeepGlobeTiledDataset(root=root, split="val",   tile_size=tile_size, stride=stride, transforms=tf_val)
    print(f"[Datasets] train tiles: {len(ds_train):,} | val tiles: {len(ds_val):,}")

    if balance_train:
        print(f"[Balance] TRAIN → target FG={target_pixels:,}, BG={target_pixels:,}, tol=2%")
        ds_train = PreciseBalancedDataset(
            ds_train,
            target_foreground_pixels=target_pixels,
            target_background_pixels=target_pixels,
            max_iterations=1200,
            tolerance=0.02,
            cache_dir="DeepGlobe/.cache",
        )
    if balance_val:
        val_target = max(1, target_pixels // 5)
        print(f"[Balance] VAL → target FG={val_target:,}, BG={val_target:,}, tol=5%")
        ds_val = PreciseBalancedDataset(
            ds_val,
            target_foreground_pixels=val_target,
            target_background_pixels=val_target,
            max_iterations=400,
            tolerance=0.05,
            cache_dir="DeepGlobe/.cache",
        )
    return ds_train, ds_val


# ------- SAFE COLLATOR (fixes KeyError: 'image') -------
def collate_fn(batch):
    """
    Accepts items as:
      - {'image', 'mask'}  (ours)
      - {'pixel_values', 'labels'}  (already HF-style)
      - (image, mask) tuples        (fallback)
    """
    b0 = batch[0]

    # HF-style dict already
    if isinstance(b0, dict) and ("pixel_values" in b0 and "labels" in b0):
        images = [b["pixel_values"] for b in batch]
        labels = [b["labels"] for b in batch]
        return {"pixel_values": torch.stack([i if isinstance(i, torch.Tensor) else torch.as_tensor(i) for i in images]),
                "labels": torch.stack([l if isinstance(l, torch.Tensor) else torch.as_tensor(l) for l in labels])}

    # Our dataset dict
    if isinstance(b0, dict) and ("image" in b0 and "mask" in b0):
        images = torch.stack([b["image"] for b in batch])
        masks  = torch.stack([b["mask"]  for b in batch])
        return {"pixel_values": images, "labels": masks}

    # Tuple/list fallback
    if isinstance(b0, (tuple, list)) and len(b0) >= 2:
        images = torch.stack([b[0] for b in batch])
        masks  = torch.stack([b[1] for b in batch])
        return {"pixel_values": images, "labels": masks}

    # Last-resort diagnostic
    raise KeyError(f"Unexpected batch item format: type={type(b0)}, keys={getattr(b0,'keys',lambda:[])()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="DeepGlobe")
    ap.add_argument("--model", type=str, default="segformer", choices=["segformer","segforest"])
    ap.add_argument("--backbone", type=str, default="nvidia/mit-b5")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--tile", type=int, default=612)
    ap.add_argument("--stride", type=int, default=612)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--run_name", type=str, default="dg_run")
    ap.add_argument("--no_balance", action="store_true")
    ap.add_argument("--balance_val", action="store_true")
    ap.add_argument("--target_pixels", type=int, default=300_000_000)
    args = ap.parse_args()

    print("Config:", vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = build_datasets(
        root=args.root,
        tile_size=args.tile,
        stride=args.stride,
        balance_train=(not args.no_balance),
        balance_val=args.balance_val,
        target_pixels=args.target_pixels,
    )

    if args.model == "segformer":
        model = CustomSegformer(input_channels=3, num_classes=2, base_model=args.backbone)
    else:
        if not HAS_SEGFOREST:
            raise RuntimeError("Segforest model not found in Model/Segforest/Segforest.py")
        model = SegforestModel(in_channels=3, num_classes=2)
    model.to(device)

    out_dir = f"./outputs/{args.run_name}"
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        # use the new key to avoid the deprecation warning:
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.num_workers,
        dataloader_drop_last=True,
        fp16=torch.cuda.is_available(),
        report_to=None,
        logging_steps=50,
        # IMPORTANT so Trainer doesn't drop our keys in the batch dict
        remove_unused_columns=False,
    )

    trainer = SegTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    print("Starting training…")
    trainer.train()

    best_dir = Path(out_dir) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(best_dir)
    print(f"Saved best model to: {best_dir.resolve()}")


if __name__ == "__main__":
    main()
