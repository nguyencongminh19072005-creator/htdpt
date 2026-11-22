"""Fine-tune DeepLabV3+ specifically on final_segmentation_v2 ground truth masks.

Trains using the curated Ground Truth in final_segmentation_v2 (1,000 masks with post-processing and manual edits).
Evaluates on the 100 validation images and 105 held-out test images from splits.json.
Saves best checkpoint into cau_3/deeplabv3plus_384_v2_finetuned/best_deeplabv3plus.pt
and updates active_deeplab_model.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import train_flower_deeplabv3plus as original
from active_deeplab_model import load_active_model


ROOT = Path(r"D:\CSDLDPT")
IMAGES = ROOT / "image_384"
MASKS = ROOT / "final_segmentation_v2"
SPLITS = ROOT / "cau_3" / "trained_seed_pilot_20261022" / "splits.json"
OUTPUT = ROOT / "cau_3" / "deeplabv3plus_384_v2_finetuned"
PILOT_CKPT = ROOT / "cau_3" / "deeplabv3plus_pilot_20261022" / "best_deeplabv3plus.pt"
SEED = 20261116


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class V2FlowerDataset(Dataset):
    def __init__(self, names: list[str], augment: bool = False):
        self.names = names
        self.augment = augment
        self.images: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []

        print(f"Preloading {len(names)} samples from final_segmentation_v2 into RAM...", flush=True)
        for name in names:
            img = cv2.imread(str(IMAGES / name), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(MASKS / name), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None or img.shape[:2] != (384, 384) or mask.shape != (384, 384):
                raise ValueError(f"Invalid image/mask pair: {name}")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bin_mask = (mask > 127).astype(np.uint8)
            self.images.append(rgb)
            self.masks.append(bin_mask)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = self.images[index].copy()
        mask = self.masks[index].copy()

        if self.augment:
            # Random horizontal flip
            if random.random() < 0.5:
                rgb = np.fliplr(rgb)
                mask = np.fliplr(mask)
            # Random 90-degree rotations
            if random.random() < 0.5:
                turns = random.randrange(4)
                rgb = np.rot90(rgb, turns)
                mask = np.rot90(mask, turns)
            # Small affine rotation (+- 10 deg)
            if random.random() < 0.5:
                angle = random.uniform(-10, 10)
                scale = random.uniform(0.95, 1.05)
                m = cv2.getRotationMatrix2D((192, 192), angle, scale)
                rgb = cv2.warpAffine(rgb, m, (384, 384), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                mask = cv2.warpAffine(mask, m, (384, 384), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        img_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask[None])).float()
        return img_tensor, mask_tensor


def tversky_bce_loss(logits: torch.Tensor, targets: torch.Tensor, fn_weight: float = 0.55) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probs = logits.sigmoid()
    tp = (probs * targets).sum(dim=(1, 2, 3))
    fp = (probs * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - probs) * targets).sum(dim=(1, 2, 3))
    alpha = 1.0 - fn_weight
    tversky = (tp + 1.0) / (tp + alpha * fp + fn_weight * fn + 1.0)
    return bce + (1.0 - tversky).mean()


@torch.no_grad()
def evaluate_split(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    ious, dices = [], []
    for imgs, targets in loader:
        probs = model(imgs).sigmoid()[:, 0]
        preds = probs >= 0.45
        gts = targets[:, 0] >= 0.5
        inter = (preds & gts).sum(dim=(1, 2)).float()
        union = (preds | gts).sum(dim=(1, 2)).float()
        total = preds.sum(dim=(1, 2)).float() + gts.sum(dim=(1, 2)).float()
        ious.extend((inter / union.clamp_min(1.0)).cpu().tolist())
        dices.extend(((2.0 * inter) / total.clamp_min(1.0)).cpu().tolist())
    return float(np.mean(ious)), float(np.mean(dices))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6, help="Fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2.5e-5, help="Learning rate")
    parser.add_argument("--from-pilot", action="store_true", help="Start from original pilot checkpoint instead of active")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    cv2.setNumThreads(1)
    torch.set_num_threads(8)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # 1. Load initial checkpoint
    if args.from_pilot:
        init_state = torch.load(PILOT_CKPT, map_location="cpu", weights_only=True)
        width = int(init_state.get("width", 32))
        rates = tuple(init_state.get("rates", (2, 4, 6)))
        model = original.DeepLabV3Plus(width=width, rates=rates)
        weights = init_state["state_dict"] if "state_dict" in init_state else init_state["model_state"]
        model.load_state_dict(weights)
        print(f"Loaded original pilot checkpoint: {PILOT_CKPT.name}", flush=True)
    else:
        model, init_state, config = load_active_model()
        width = int(init_state["width"])
        rates = tuple(init_state["rates"])
        print(f"Loaded starting checkpoint: {config['checkpoint']} (mIoU: {init_state.get('validation_iou', 'N/A')})", flush=True)

    # 2. Prepare dataset splits from splits.json and manifest.json
    splits_data = json.loads(SPLITS.read_text(encoding="utf-8"))
    manifest = json.loads((MASKS / "manifest.json").read_text(encoding="utf-8"))

    train_base = [f for f in splits_data["train"] if (IMAGES / f).is_file()]
    val_names = [f for f in splits_data["validation"] if (IMAGES / f).is_file()]
    test_names = [f for f in splits_data["reviewed_test_current"] if (IMAGES / f).is_file()]

    all_names = {p.name for p in IMAGES.glob("*.png")}
    known = set(train_base) | set(val_names) | set(test_names)
    new_replacement = sorted(list(all_names - known))

    # Corrected tricky flowers in train
    corrected_train = [it["filename"] for it in manifest["items"]
                       if it["source_type"] == "corrected_v2" and it["filename"] in set(train_base)]

    train_names = []
    train_names.extend(train_base)
    train_names.extend(corrected_train)  # 2x repeat for tricky flowers
    train_names.extend(new_replacement * 3)  # 3x repeat for replacements

    print(f"Dataset v2 summary:", flush=True)
    print(f"  - Target masks: {MASKS}", flush=True)
    print(f"  - Train base: {len(train_base)} + {len(new_replacement)} replacements = {len(train_base) + len(new_replacement)} distinct", flush=True)
    print(f"  - Effective train samples per epoch: {len(train_names)}", flush=True)
    print(f"  - Validation: {len(val_names)}", flush=True)
    print(f"  - Held-out Test: {len(test_names)}", flush=True)

    # 3. Create DataLoaders
    train_ds = V2FlowerDataset(train_names, augment=True)
    val_ds = V2FlowerDataset(val_names, augment=False)
    test_ds = V2FlowerDataset(test_names, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Baseline evaluation before fine-tuning
    val_iou_base, val_dice_base = evaluate_split(model, val_loader)
    test_iou_base, test_dice_base = evaluate_split(model, test_loader)
    print(f"Baseline (epoch 0) on v2 ground truth:", flush=True)
    print(f"  Val  mIoU: {val_iou_base * 100:.2f}% | Dice: {val_dice_base * 100:.2f}%", flush=True)
    print(f"  Test mIoU: {test_iou_base * 100:.2f}% | Dice: {test_dice_base * 100:.2f}%", flush=True)

    # 4. Setup Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    best_val_iou = val_iou_base
    best_ckpt = OUTPUT / "best_deeplabv3plus.pt"
    best_state = None
    metrics_history = []

    # 5. Training loop
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for imgs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = tversky_bce_loss(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_train_loss = running_loss / max(n_batches, 1)

        val_iou, val_dice = evaluate_split(model, val_loader)
        test_iou, test_dice = evaluate_split(model, test_loader)

        elapsed = time.time() - t_start
        is_best = val_iou > best_val_iou
        star = " (* BEST)" if is_best else ""

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val mIoU: {val_iou * 100:.2f}% (Dice: {val_dice * 100:.2f}%) | "
            f"Test mIoU: {test_iou * 100:.2f}% (Dice: {test_dice * 100:.2f}%){star} [{elapsed:.1f}s]",
            flush=True,
        )

        metrics_history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
            "test_iou": test_iou,
            "test_dice": test_dice,
            "is_best": is_best,
        })

        if is_best:
            best_val_iou = val_iou
            best_state = {
                "epoch": epoch,
                "state_dict": deepcopy(model.state_dict()),
                "size": 384,
                "width": width,
                "rates": rates,
                "validation_iou": val_iou,
                "validation_dice": val_dice,
                "heldout_test_iou": test_iou,
                "heldout_test_dice": test_dice,
                "trained_masks": str(MASKS),
                "threshold": 0.45,
            }

    if best_state is None:
        print("Validation mIoU did not improve. Keeping the current active checkpoint.", flush=True)
        return

    training_log = {
        "dataset": "final_segmentation_v2",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "initial_val_iou": val_iou_base,
        "best_val_iou": best_val_iou,
        "final_heldout_test_iou": best_state["heldout_test_iou"],
        "history": metrics_history,
    }

    config_path = ROOT / "cau_3" / "active_deeplab_model.json"
    active_conf = {
        "model_id": "deeplabv3plus_384_v2_finetuned",
        "checkpoint": best_ckpt.relative_to(config_path.parent).as_posix(),
        "input_size": 384,
        "output_size": 384,
        "threshold": 0.45,
        "training_masks": "fine-tuned on final_segmentation_v2 (1000 ground truth masks)",
        "validation_mean_iou": best_state["validation_iou"],
        "heldout_mean_iou": best_state["heldout_test_iou"],
        "note": "Active model fine-tuned on strictly final_segmentation_v2."
    }
    log_path = OUTPUT / "training_log.json"
    with TemporaryDirectory(dir=OUTPUT) as staging:
        stage = Path(staging)
        staged_ckpt = stage / best_ckpt.name
        torch.save(best_state, staged_ckpt)
        ckpt_hash = sha256(staged_ckpt)
        active_conf["checkpoint_sha256"] = ckpt_hash
        training_log["checkpoint_sha256"] = ckpt_hash
        staged_config = stage / config_path.name
        staged_log = stage / log_path.name
        staged_config.write_text(json.dumps(active_conf, indent=2), encoding="utf-8")
        staged_log.write_text(json.dumps(training_log, indent=2), encoding="utf-8")
        staged_ckpt.replace(best_ckpt)
        staged_config.replace(config_path)
        staged_log.replace(log_path)

    print("\n" + "=" * 60, flush=True)
    print(f"Fine-tuning on final_segmentation_v2 completed successfully!", flush=True)
    print(f"Best checkpoint: {best_ckpt}", flush=True)
    print(f"Validation mIoU: {best_state['validation_iou'] * 100:.2f}%", flush=True)
    print(f"Held-out Test mIoU: {best_state['heldout_test_iou'] * 100:.2f}%", flush=True)
    print(f"Registered in active_deeplab_model.json", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
