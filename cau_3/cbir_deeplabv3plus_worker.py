"""JSON-line inference worker for the selected 384x384 DeepLabV3+ model.

The CBIR process sends RGB uint8 pixels. This worker returns the binary mask
after the project's selected postprocessing configuration. It never retrains
or changes gallery ground-truth masks.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from active_deeplab_model import CONFIG_PATH, load_active_model, sha256
from mask_postprocessing import postprocess_mask


POSTPROCESSING = {
    "threshold": 0.45,
    "opening": False,
    "closing": False,
    "fill_holes": True,
    "max_hole_area": 20,
    "keep_nearby": True,
    "repair_center": True,
}


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    model, state, config = load_active_model()
    expected = (CONFIG_PATH.parent / config["checkpoint"]).resolve()
    if args.checkpoint.resolve() != expected:
        raise ValueError("CBIR checkpoint differs from the selected active model")
    model.to(args.device).eval()
    print(json.dumps({
        "ready": True,
        "model_id": config["model_id"],
        "checkpoint_sha256": sha256(expected),
        "model_config_sha256": sha256(CONFIG_PATH),
        "postprocessing_sha256": sha256(Path(__file__).with_name("mask_postprocessing.py")),
        "postprocessing": POSTPROCESSING,
        "input_size": 384,
        "output_size": 384,
        "architecture": {"width": int(state["width"]), "rates": list(state["rates"])},
        "torch": torch.__version__,
        "device": args.device,
        "precision": "float32",
    }), flush=True)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("action") == "close":
                break
            raw = base64.b64decode(request["rgb"], validate=True)
            if len(raw) != 384 * 384 * 3:
                raise ValueError("Expected RGB uint8 image of size 384x384")
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(384, 384, 3)
            tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()[None] / 255.0
            with torch.inference_mode():
                probability = model(tensor.to(args.device)).sigmoid()[0, 0].cpu().numpy()
            mask, _, _ = postprocess_mask(
                probability,
                image_bgr=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                **POSTPROCESSING,
            )
            if mask.shape != (384, 384) or not np.isin(mask, (0, 255)).all():
                raise ValueError("Postprocessor returned an invalid binary mask")
            response = {"mask": base64.b64encode(mask.tobytes()).decode("ascii")}
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
        raise
