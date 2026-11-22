"""Train a compact DeepLabV3+ flower segmenter on the same split as SeedUNet.

This is a from-scratch CPU-capable DeepLabV3+ variant: output-stride-16 encoder,
ASPP, low-level skip projection, and decoder. Source images and masks are read
only. The 108 current and 42 external reviewed test images never enter training
or model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(r"D:\CSDLDPT")
IMAGES = ROOT / "image_384"
MASKS = ROOT / "final_segmentation_1"
SPLIT = ROOT / "cau_3" / "trained_seed_pilot_20261022" / "splits.json"
OUTPUT = ROOT / "cau_3" / "deeplabv3plus_pilot_20261022"
SEED = 20261022


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding=(kernel_size // 2) * dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 dilation: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, 3, stride, dilation,
                      dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling at output stride 16."""

    def __init__(self, in_channels: int, channels: int, rates: tuple[int, ...]):
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBNReLU(in_channels, channels, kernel_size=1),
            *(SeparableConv(in_channels, channels, dilation=rate) for rate in rates),
        ])
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.project = ConvBNReLU(channels * (len(self.branches) + 1), channels * 2,
                                  kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        pooled = nn.functional.interpolate(self.image_pool(value), size=value.shape[-2:],
                                           mode="bilinear", align_corners=False)
        return self.project(torch.cat([*(branch(value) for branch in self.branches), pooled], dim=1))


class DeepLabV3Plus(nn.Module):
    """Lightweight DeepLabV3+ with stride-16 encoder and stride-4 decoder skip."""

    def __init__(self, width: int = 32, rates: tuple[int, ...] = (2, 4, 6)):
        super().__init__()
        self.stem = ConvBNReLU(3, width, stride=2)
        self.low = ConvBNReLU(width, width * 2, stride=2)
        self.encoder8 = nn.Sequential(
            SeparableConv(width * 2, width * 3, stride=2),
            SeparableConv(width * 3, width * 3),
        )
        self.encoder16 = nn.Sequential(
            SeparableConv(width * 3, width * 5, stride=2),
            SeparableConv(width * 5, width * 5),
            SeparableConv(width * 5, width * 5),
        )
        self.aspp = ASPP(width * 5, width * 2, rates)
        self.low_project = ConvBNReLU(width * 2, width, kernel_size=1)
        self.decoder = nn.Sequential(
            SeparableConv(width * 5, width * 2),
            SeparableConv(width * 2, width * 2),
            nn.Conv2d(width * 2, 1, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        size = image.shape[-2:]
        low = self.low(self.stem(image))
        high = self.encoder16(self.encoder8(low))
        high = nn.functional.interpolate(self.aspp(high), size=low.shape[-2:],
                                         mode="bilinear", align_corners=False)
        logits = self.decoder(torch.cat((high, self.low_project(low)), dim=1))
        return nn.functional.interpolate(logits, size=size, mode="bilinear", align_corners=False)


class FlowerPairs(Dataset):
    def __init__(self, names: list[str], size: int, augment: bool,
                 geometric_augment: bool = False,
                 photometric_augment: bool = False,
                 mask_paths: dict[str, Path] | None = None):
        self.names, self.augment, self.size = names, augment, size
        self.geometric_augment = geometric_augment
        self.photometric_augment = photometric_augment
        self.images: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        for name in names:
            image = cv2.imread(str(IMAGES / name), cv2.IMREAD_COLOR)
            mask_path = mask_paths[name] if mask_paths is not None else MASKS / name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or image.shape[:2] != (384, 384) or mask.shape != (384, 384):
                raise ValueError(f"Missing or invalid source pair: {name}")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            binary_mask = (mask > 127).astype(np.uint8)
            self.images.append(rgb if size == 384 else cv2.resize(
                rgb, (size, size), interpolation=cv2.INTER_AREA))
            self.masks.append(binary_mask if size == 384 else cv2.resize(
                binary_mask, (size, size), interpolation=cv2.INTER_NEAREST))

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = self.images[index].copy()
        mask = self.masks[index].copy()
        if self.augment:
            turns = random.randrange(4)
            rgb, mask = np.rot90(rgb, turns), np.rot90(mask, turns)
            if random.random() < 0.5:
                rgb, mask = np.fliplr(rgb), np.fliplr(mask)
            if random.random() < 0.5:
                rgb, mask = np.flipud(rgb), np.flipud(mask)
            if self.geometric_augment and random.random() < 0.75:
                matrix = cv2.getRotationMatrix2D(
                    (self.size / 2, self.size / 2), random.uniform(-15, 15),
                    random.uniform(0.9, 1.1))
                matrix[0, 2] += random.uniform(-0.04, 0.04) * self.size
                matrix[1, 2] += random.uniform(-0.04, 0.04) * self.size
                rgb = cv2.warpAffine(rgb, matrix, (self.size, self.size),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                mask = cv2.warpAffine(mask, matrix, (self.size, self.size),
                                      flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)
            if self.photometric_augment and random.random() < 0.8:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-12, 12)) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.25, 1.25), 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.72, 1.25), 0, 255)
                rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                if random.random() < 0.35:
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    amount = random.uniform(0.3, 0.85)
                    rgb = np.clip((1 - amount) * rgb + amount * gray[:, :, None],
                                  0, 255).astype(np.uint8)
                if random.random() < 0.4:
                    gamma = random.uniform(0.75, 1.35)
                    rgb = np.clip((rgb.astype(np.float32) / 255) ** gamma * 255,
                                  0, 255).astype(np.uint8)
            rgb = np.clip(rgb.astype(np.float32) * random.uniform(0.85, 1.15)
                          + random.uniform(-8, 8), 0, 255).astype(np.uint8)
        image_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask[None])).float()
        return image_tensor, mask_tensor


def loss_function(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1) / (
        probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1
    )
    return bce + (1 - dice).mean()


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    dice_scores, iou_scores = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        prediction = model(images).sigmoid()[:, 0] >= 0.5
        truth = labels[:, 0] >= 0.5
        intersection = (prediction & truth).sum(dim=(1, 2)).float()
        union = (prediction | truth).sum(dim=(1, 2)).float()
        total_area = prediction.sum(dim=(1, 2)).float() + truth.sum(dim=(1, 2)).float()
        iou_scores.extend((intersection / union.clamp_min(1)).cpu().tolist())
        dice_scores.extend((2 * intersection / total_area.clamp_min(1)).cpu().tolist())
    return float(np.mean(dice_scores)), float(np.mean(iou_scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--smoke", action="store_true", help="One train and one validation batch; no checkpoint")
    args = parser.parse_args()
    if args.size < 64 or args.size % 16 or args.width < 8 or args.epochs < 1 or args.batch_size < 1:
        parser.error("size must be >=64 and divisible by 16; width >=8; epochs and batch >=1")

    with SPLIT.open(encoding="utf-8") as source:
        split = json.load(source)
    if {key: len(value) for key, value in split.items()} != {
        "train": 792, "validation": 100, "reviewed_test_current": 108,
        "reviewed_test_external": 42,
    }:
        raise ValueError("Training split differs from the U-Net experiment")
    if len(set(split["train"]) & (set(split["validation"]) |
                               set(split["reviewed_test_current"]) |
                               set(split["reviewed_test_external"]))) > 0:
        raise ValueError("Training/test leakage")
    if set(split["validation"]) & (set(split["reviewed_test_current"]) |
                                   set(split["reviewed_test_external"])):
        raise ValueError("Validation/test leakage")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    cv2.setNumThreads(1)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_names = split["train"][:2] if args.smoke else split["train"]
    validation_names = split["validation"][:2] if args.smoke else split["validation"]
    train_data = FlowerPairs(train_names, args.size, augment=True)
    validation_data = FlowerPairs(validation_names, args.size, augment=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False,
                                   num_workers=0, pin_memory=device.type == "cuda")
    model = DeepLabV3Plus(width=args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(json.dumps({"device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
                      "train": len(train_data), "validation": len(validation_data),
                      "size": args.size, "width": args.width}), flush=True)

    if args.smoke:
        model.train()
        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_function(logits, y)
        loss.backward()
        optimizer.step()
        dice, iou = validate(model, validation_loader, device)
        print(json.dumps({"smoke": "passed", "logits_shape": list(logits.shape),
                          "loss": float(loss.item()), "validation_dice": dice,
                          "validation_iou": iou}), flush=True)
        return

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing training run: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    config = {
        "architecture": "compact_deeplabv3plus_from_scratch",
        "encoder_output_stride": 16, "decoder_skip_stride": 4,
        "aspp_rates": [2, 4, 6], "size": args.size, "width": args.width,
        "epochs_requested": args.epochs, "batch_size": args.batch_size,
        "optimizer": "Adam", "learning_rate": 0.001, "loss": "BCEWithLogits + soft Dice",
        "seed": SEED, "input": "RGB / 255", "mask": "final_segmentation_1 > 127",
        "split_sha256": sha256(SPLIT), "train_count": len(train_data),
        "validation_count": len(validation_data), "reviewed_test_count": 150,
        "device": str(device), "torch_version": torch.__version__,
    }
    (OUTPUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    best_iou, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        dice, iou = validate(model, validation_loader, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "validation_dice_pseudo": dice, "validation_iou_pseudo": iou}
        history.append(record)
        (OUTPUT / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record), flush=True)
        if iou > best_iou + 1e-4:
            best_iou, best_epoch = iou, epoch
            torch.save({"state_dict": model.state_dict(), "size": args.size,
                        "width": args.width, "rates": (2, 4, 6), "epoch": epoch,
                        "validation_iou_pseudo": iou, "split_sha256": sha256(SPLIT),
                        "architecture": config["architecture"]}, OUTPUT / "best_deeplabv3plus.pt")
        if epoch >= 8 and epoch - best_epoch >= 4:
            print(f"Early stopping at epoch {epoch}; best was {best_epoch}", flush=True)
            break
    (OUTPUT / "training_summary.json").write_text(json.dumps({
        "best_epoch": best_epoch, "best_validation_iou_pseudo": best_iou,
        "epochs_completed": len(history), "checkpoint_sha256": sha256(OUTPUT / "best_deeplabv3plus.pt"),
        "split_sha256": sha256(SPLIT), "note": "Validation uses current pseudo masks, not independent reviewed masks."
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Best epoch {best_epoch}, pseudo-mask validation IoU {best_iou:.4f}", flush=True)


if __name__ == "__main__":
    main()
