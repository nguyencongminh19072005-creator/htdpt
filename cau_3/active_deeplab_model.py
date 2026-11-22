"""Load the explicit 384x384 DeepLabV3+ model selected for prediction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from train_flower_deeplabv3plus import DeepLabV3Plus


CONFIG_PATH = Path(__file__).with_name("active_deeplab_model.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_active_model() -> tuple[DeepLabV3Plus, dict, dict]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    checkpoint = (CONFIG_PATH.parent / config["checkpoint"]).resolve()
    if not checkpoint.is_file() or sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError(f"Active checkpoint missing or hash mismatch: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if int(config["input_size"]) != 384 or int(state["size"]) != 384:
        raise ValueError("Active model must receive 384x384 inputs")
    if int(config["output_size"]) != 384:
        raise ValueError("Active model must produce 384x384 masks")
    model = DeepLabV3Plus(width=int(state["width"]), rates=tuple(state["rates"]))
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, state, config
