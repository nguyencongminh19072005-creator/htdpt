"""Rebuild all eight PostgreSQL tables from the selected DeepLabV3+ pipeline.

The complete 1,000-image index is prepared and checked before the existing
database rows are replaced in one transaction. A validated pg_dump backup is
required; the script never drops tables or the pgvector extension.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flower_cbir.core.segmentation import clean_mask
from flower_cbir.services.csv_tools import export_csv
from flower_cbir.services.database import FlowerDatabase
from flower_cbir.services.index_dataset import build_or_load_index
from flower_cbir.services.search import FlowerSearchEngine
from flower_cbir.settings import load_config


def verify_backup(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 1024 * 1024:
        raise ValueError("Missing or suspiciously small pg_dump backup")
    pg_restore = Path(r"D:\postgree\server18\bin\pg_restore.exe")
    result = subprocess.run([str(pg_restore), "-l", str(path)],
                            capture_output=True, text=True, check=True)
    if not all(name in result.stdout for name in
               ("flower_images", "flower_masks", "normalized_features", "retrieval_history")):
        raise ValueError("Backup lacks one or more required tables")


def preflight(data: dict) -> None:
    if len(data["metadata"]) != 1000 or data["raw_features"].shape != (1000, 163):
        raise ValueError("Expected a complete 1,000-image, 163-D index")
    expected_source = load_config()["dataset"]["mask_source"]
    if expected_source != "deeplabv3plus_v2_postprocessed":
        raise ValueError("Gallery is not configured for DeepLabV3+ postprocessed masks")
    engine = FlowerSearchEngine(data)
    for index in (0, 123, 250, 500, 750, 999):
        item = data["metadata"][index]
        if item["mask_provenance"] != expected_source:
            raise ValueError(f"Wrong mask provenance: {item['filename']}")
        result = engine.search(item["image_path"], use_gallery_mask=False)
        with Image.open(item["mask_path"]) as saved:
            expected_mask = clean_mask(np.asarray(saved.convert("L")))
        if (not np.array_equal(result["query_info"]["mask_binary"], expected_mask)
                or not np.array_equal(result["query_info"]["raw_vector"],
                                      data["raw_features"][index])):
            raise ValueError(f"Model/query/gallery parity failed: {item['filename']}")
    print("Preflight passed: six fresh model predictions reproduce gallery masks and vectors.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    backup = args.backup.resolve()
    verify_backup(backup)

    data = build_or_load_index()
    preflight(data)
    csv_path = ROOT / "trich_rut_minh" / "features_163.csv"
    manifest_path = ROOT / "trich_rut_minh" / "extraction_manifest_163.json"
    export_csv(data, csv_path, manifest_path)
    print(f"Prepared model index {data['version_id']} and exported 163-D CSV.", flush=True)
    if args.prepare_only:
        print("Database unchanged (--prepare-only).", flush=True)
        return

    db = FlowerDatabase()
    try:
        result = db.publish_index(data, replace_all=True, backup_artifact=backup)
        loaded = db.load_index()
        if loaded["version_id"] != data["version_id"] or len(loaded["metadata"]) != 1000:
            raise RuntimeError("Published index failed PostgreSQL round-trip validation")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    main()
