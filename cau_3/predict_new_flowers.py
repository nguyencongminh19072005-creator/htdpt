"""Resize supplied photos to 384x384 and test the frozen DeepLabV3+ model.

Usage:
    python predict_new_flowers.py photo1.jpg photo2.jpg

The output is isolated from the 1,000-image database and contains the raw
DeepLabV3+ prediction; no GrabCut or manual correction is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from train_flower_deeplabv3plus import ROOT
from active_deeplab_model import load_active_model
from mask_postprocessing import postprocess_mask


DEFAULT_OUTPUT = ROOT / "tmp" / "deeplab_new_flowers_384_20261025"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_viewer(output: Path, rows: list[dict], threshold: float) -> None:
    cards = []
    for row in rows:
        stem = row["stem"]
        label = html.escape(row["source_name"])
        post_info = row.get("postprocessing_metrics", {})
        info_str = f"Ảnh gốc: {row['source_width']} × {row['source_height']} · Tiền cảnh: {row['foreground_percent']:.2f}% · Ngưỡng: {threshold:.2f}"
        if post_info:
            info_str += f" · Đốm nhiễu đã lọc: {post_info.get('num_disconnected_components_removed', 0)} · Pixel bù lỗ/làm sạch: {post_info.get('pixel_delta', 0):+d}"
        cards.append(f"""
        <section>
          <h2>{label}</h2>
          <p>{info_str}</p>
          <div class="grid">
            <figure><figcaption>1. Ảnh đầu vào 384 × 384</figcaption>
              <img src="{stem}_resized.png" alt="Ảnh hoa"></figure>
            <figure><figcaption>2. Mask thô (raw threshold)</figcaption>
              <img src="{stem}_raw_mask.png" alt="Mask thô"></figure>
            <figure><figcaption>3. Mask sau hậu xử lý chuẩn</figcaption>
              <img src="{stem}_mask.png" alt="Mask hậu xử lý"></figure>
            <figure><figcaption>4. Mask phủ trên ảnh</figcaption>
              <img src="{stem}_overlay.png" alt="Ảnh phủ mask"></figure>
            <figure><figcaption>5. Hoa đã tách nền (Cutout)</figcaption>
              <img src="{stem}_cutout.png" alt="Hoa đã tách nền"></figure>
          </div>
        </section>""")
    page = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thử nghiệm DeepLabV3+ & Hậu xử lý mask</title>
<style>
  :root { color-scheme: dark; font-family: system-ui, sans-serif; }
  body { max-width: 1600px; padding: 22px; margin: 0 auto; background: #111827; color: #f9fafb; }
  h1 { margin-bottom: 8px; } p { line-height: 1.5; }
  .note { color: #fcd34d; }
  section { background: #1f2937; border-radius: 12px; padding: 18px; margin: 22px 0; }
  .grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }
  figure { margin: 0; } figcaption { margin-bottom: 8px; color: #d1d5db; font-size: 13px; font-weight: 500; }
  img { width: 100%; aspect-ratio: 1; object-fit: contain; background: #111; border-radius: 6px; }
  @media (max-width: 1200px) { .grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
  @media (max-width: 700px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
</style></head><body>
<h1>Kiểm tra mask DeepLabV3+ & Quy trình Hậu xử lý</h1>
<p class="note">Quy trình hậu xử lý: Confidence Thresholding &rarr; Morphological Opening (khử gai nhiễu) &rarr; Connected Components (khử toàn bộ vùng nhiễu rời rạc ở nền) &rarr; Conditional Hole Filling (lấp rỗ khuyết trong cánh) &rarr; Morphological Closing nhẹ 3x3 (liền vết nứt mà không dính cánh).</p>
""" + "\n".join(cards) + "\n</body></html>\n"
    (output / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Input flower image files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold (default: from active model config)")
    parser.add_argument("--no-postprocess", action="store_true", help="Disable postprocessing pipeline")
    args = parser.parse_args()
    for path in args.images:
        if not path.is_file():
            parser.error(f"Image not found: {path}")

    cv2.setNumThreads(1)
    torch.set_num_threads(8)
    model, state, config = load_active_model()
    threshold = args.threshold if args.threshold is not None else float(config["threshold"])
    if not 0.0 <= threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    checkpoint = ROOT / "cau_3" / config["checkpoint"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for index, path in enumerate(args.images, 1):
            with Image.open(path) as image:
                rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
            height, width = rgb.shape[:2]
            method = cv2.INTER_AREA if width >= 384 and height >= 384 else cv2.INTER_LINEAR
            resized = rgb if (height, width) == (384, 384) else cv2.resize(
                rgb, (384, 384), interpolation=method)
            tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float()[None] / 255.0
            probabilities = model(tensor).sigmoid()[0, 0].numpy()
            if probabilities.shape != (384, 384):
                raise ValueError(f"Unexpected model output shape: {probabilities.shape}")

            # 1. Raw mask
            raw_mask = (probabilities >= threshold).astype(np.uint8) * 255

            # 2. Post-processed mask (C4 + vá lỗ li ti + sửa nhụy tâm hoa)
            if not args.no_postprocess:
                bgr_img = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                binary_mask, alpha_mask, post_metrics = postprocess_mask(
                    probabilities,
                    image_bgr=bgr_img,
                    threshold=threshold,
                    opening=False,
                    closing=False,
                    fill_holes=True,
                    keep_nearby=True,
                    repair_center=True,
                    max_hole_area=20,
                )
                final_mask = binary_mask
            else:
                final_mask = raw_mask
                alpha_mask = raw_mask
                post_metrics = {"postprocessing": "disabled"}

            mask_bool = final_mask > 127
            stem = f"flower_{index:02d}"

            Image.fromarray(resized).save(args.output_dir / f"{stem}_resized.png")
            Image.fromarray(raw_mask).save(args.output_dir / f"{stem}_raw_mask.png")
            cv2.imwrite(str(args.output_dir / f"{stem}_mask.png"), final_mask)
            cv2.imwrite(str(args.output_dir / "mask_result.png"), final_mask)

            overlay = resized.copy()
            overlay[mask_bool] = np.rint(0.6 * resized[mask_bool] + 0.4 * 255).astype(np.uint8)
            Image.fromarray(overlay).save(args.output_dir / f"{stem}_overlay.png")

            cutout = np.dstack((resized, final_mask))
            Image.fromarray(cutout, "RGBA").save(args.output_dir / f"{stem}_cutout.png")

            rows.append({
                "stem": stem,
                "source_name": path.name,
                "source_path": str(path.resolve()),
                "source_sha256": sha256(path),
                "source_width": width,
                "source_height": height,
                "foreground_pixels": int(mask_bool.sum()),
                "foreground_percent": 100 * float(mask_bool.mean()),
                "postprocessing_metrics": post_metrics,
            })
            print(f"{path.name}: {width}x{height} -> 384x384; foreground={100 * mask_bool.mean():.2f}%; disconnected removed={post_metrics.get('num_disconnected_components_removed', 0)}", flush=True)

    (args.output_dir / "run.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "model_id": config["model_id"],
        "model_input_size": state["size"], "threshold": threshold,
        "postprocessing": "enabled" if not args.no_postprocess else "none",
        "images": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_viewer(args.output_dir, rows, threshold)
    print(f"Viewer: {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
