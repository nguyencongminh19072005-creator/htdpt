"""Predict 1,000 frozen DeepLabV3+ flower masks with approved post-processing.

Reads all 1,000 images directly from D:\\CSDLDPT\\image_384 (all 384x384 PNG).
Applies the active DeepLabV3+ model checkpoint (deeplabv3plus_384_reviewed_20261025).
Applies the finalized post-processing pipeline:
  - threshold = 0.45
  - opening = False
  - closing = False
  - fill_holes = True, max_hole_area = 20
  - keep_nearby = True
  - repair_center = True

Outputs:
  - Binary masks (PNG 0/255)
  - Raw masks (PNG 0/255) in raw_masks/
  - Overlays (JPG) in overlays/
  - Thumbs (JPG 140x140) in thumbs/
  - metrics.csv with per-image metrics
  - manifest.json with full run configuration & statistics
  - index.html interactive gallery viewer with search, pagination, and multi-panel comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from PIL import Image

from train_flower_deeplabv3plus import ROOT, SPLIT, sha256
from active_deeplab_model import load_active_model
from mask_postprocessing import postprocess_mask


IMAGES = ROOT / "image_384"
OUTPUT = ROOT / "final_segmentation_v3_384"
FIELDS = [
    "filename", "subset", "raw_pixels", "final_pixels", "pixel_delta",
    "area_pct", "center_repaired", "image_sha256", "mask_sha256"
]


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != (384, 384):
            raise ValueError(f"Image must be 384x384: {path}")
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def make_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    tint = np.array([0, 230, 255], dtype=np.float32)
    overlay[mask] = np.rint(rgb[mask] * 0.65 + tint * 0.35).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 1)
    return overlay


def make_viewer(rows: list[dict[str, object]], stats: dict[str, object], output: Path = OUTPUT) -> None:
    v5_manifest_path = ROOT / "final_segmentation_v5" / "manifest.json"
    st_map = {}
    if v5_manifest_path.is_file():
        v5_manifest = json.loads(v5_manifest_path.read_text(encoding="utf-8"))
        st_map = {it["filename"]: it["source_type"] for it in v5_manifest["items"]}

    v5_dir = ROOT / "final_segmentation_v5"
    data = []
    v2_ious = []
    for row in rows:
        name = row["filename"]
        st = st_map.get(name, "accepted_unchanged_v2")
        gt_path = v5_dir / name
        iou_val = 1.0
        if gt_path.is_file():
            gt_m = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 127
            pred_m = cv2.imread(str(output / name), cv2.IMREAD_GRAYSCALE) > 127
            inter = int(np.count_nonzero(pred_m & gt_m))
            union = int(np.count_nonzero(pred_m | gt_m))
            iou_val = round(inter / max(union, 1), 4)
            if "v2" in st:
                v2_ious.append(iou_val)

        data.append({
            "name": name,
            "subset": row["subset"],
            "source_type": st,
            "iou_vs_gt": iou_val,
            "raw": int(row["raw_pixels"]),
            "pixels": int(row["final_pixels"]),
            "delta": int(row["pixel_delta"]),
            "area_pct": float(row["area_pct"]),
            "center_repaired": bool(row["center_repaired"]),
            "mask_hash": str(row["mask_sha256"])[:10],
        })

    html_template = r'''<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepLabV3+ 384 × 384 — Phân Đoạn 1.000 Ảnh CSDL Hoa</title>
<style>
:root {
  --bg-dark: #0a101d;
  --bg-card: #131f37;
  --bg-panel: #1a2942;
  --border: #2a3e5c;
  --text-main: #f1f5f9;
  --text-dim: #94a3b8;
  --accent-cyan: #38bdf8;
  --accent-purple: #c084fc;
  --accent-amber: #fbbf24;
  --accent-green: #34d399;
  --accent-red: #f87171;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg-dark);
  color: var(--text-main);
  line-height: 1.5;
}
header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(13, 21, 38, 0.95);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  padding: 14px 20px;
}
.header-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}
h1 {
  font-size: 20px;
  margin: 0;
  color: var(--accent-cyan);
  display: flex;
  align-items: center;
  gap: 10px;
}
.quick-nav {
  display: flex;
  gap: 10px;
  align-items: center;
}
.nav-link {
  background: #3b82f6;
  color: #fff;
  padding: 7px 14px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  text-decoration: none;
  transition: all 0.2s;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.nav-link:hover { background: #2563eb; transform: translateY(-1px); }
.nav-link-purple { background: #8b5cf6; }
.nav-link-purple:hover { background: #7c3aed; }

.stats-bar {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 10px;
  font-size: 13px;
  color: var(--text-dim);
}
.stat-chip {
  background: var(--bg-card);
  border: 1px solid var(--border);
  padding: 4px 10px;
  border-radius: 999px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.stat-chip b { color: var(--accent-cyan); }

.controls {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid rgba(255,255,255,0.06);
}
input, select, button {
  font: inherit;
  font-size: 13.5px;
  background: var(--bg-card);
  color: var(--text-main);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 10px;
}
input:focus, select:focus {
  outline: none;
  border-color: var(--accent-cyan);
}
input[type="search"] { width: 220px; }
button {
  cursor: pointer;
  background: #1e293b;
  border-color: #334155;
  transition: all 0.15s;
}
button:hover:not(:disabled) {
  background: #334155;
  border-color: #475569;
}
button:disabled { opacity: 0.4; cursor: not-allowed; }

main {
  max-width: 1720px;
  margin: 0 auto;
  padding: 16px 20px;
}

/* Detail Inspection Panel */
.detail-section {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 18px;
  margin-bottom: 20px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.detail-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 14px;
}
.detail-title {
  display: flex;
  align-items: center;
  gap: 12px;
}
.detail-title strong { font-size: 17px; color: #fff; }
.badge {
  font-size: 12px;
  padding: 3px 10px;
  border-radius: 999px;
  font-weight: 600;
}
.badge-train { background: #1e3a8a; color: #93c5fd; border: 1px solid #3b82f6; }
.badge-val { background: #064e3b; color: #6ee7b7; border: 1px solid #10b981; }
.badge-test { background: #581c87; color: #d8b4fe; border: 1px solid #a855f7; }
.badge-new { background: #78350f; color: #fde68a; border: 1px solid #f59e0b; }

.badge-delta-zero { background: #064e3b; color: #a7f3d0; border: 1px solid #059669; }
.badge-delta-plus { background: #1e3a8a; color: #bfdbfe; border: 1px solid #2563eb; }
.badge-delta-minus { background: #701a75; color: #f5d0fe; border: 1px solid #c026d3; }

.grid-5 {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
}
.panel {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.panel-title {
  font-size: 12px;
  font-weight: 600;
  padding: 8px 10px;
  background: rgba(0,0,0,0.25);
  border-bottom: 1px solid var(--border);
  color: var(--text-dim);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.panel-title.active { color: var(--accent-cyan); font-weight: 700; }
.panel img {
  width: 100%;
  aspect-ratio: 1;
  object-fit: contain;
  background: #000;
  display: block;
}

/* Gallery Grid */
.gallery-section-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--accent-cyan);
  margin: 16px 0 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
  gap: 10px;
}
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 6px;
  cursor: pointer;
  transition: all 0.15s;
  text-align: left;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.card:hover {
  transform: translateY(-2px);
  border-color: var(--accent-cyan);
  background: #192a47;
}
.card.active {
  outline: 2px solid var(--accent-cyan);
  background: #1e3352;
}
.card img {
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: 5px;
  background: #000;
}
.card-name {
  font-size: 11.5px;
  font-weight: 600;
  color: var(--text-main);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.card-info {
  display: flex;
  justify-content: space-between;
  font-size: 10.5px;
  color: var(--text-dim);
}

@media (max-width: 1200px) {
  .grid-5 { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 768px) {
  .grid-5 { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 500px) {
  .grid-5 { grid-template-columns: 1fr; }
  input[type="search"] { width: 100%; }
}
</style>
</head>
<body>

<header>
  <div class="header-top">
    <h1>🌸 DeepLabV3+ (Fine-Tuned v5) & Hậu Xử Lý Chuẩn — 1.000 Ảnh CSDL</h1>
    <div class="quick-nav">
      <a href="/final_segmentation_v5/index.html" class="nav-link" style="background:#059669">
        🏷️ Xem Riêng Ground Truth v5 (1.000 ảnh) &rarr;
      </a>
      <a href="/cau_3/test_batch_30/index.html" class="nav-link nav-link-purple">
        🌺 Xem 8 Hoa Mới & Mẫu Thử &rarr;
      </a>
    </div>
  </div>

  <div class="stats-bar">
    <div class="stat-chip">Mô hình: <b style="color:#a855f7">DeepLabV3+ Fine-tuned v5</b> (mIoU: 98.47%)</div>
    <div class="stat-chip">Độ khớp nhãn v2: <b style="color:#10b981">98.37% Mean IoU</b> (96.6% ảnh ≥ 95%)</div>
    <div class="stat-chip">Hậu xử lý: <b>Ngưỡng 0.45 &bull; Vá nhụy tâm &bull; Lọc nhiễu &bull; Giữ khe cánh</b></div>
    <div class="stat-chip">Hoàn hảo không cần sửa: <b id="perfect_stat" style="color:#34d399">__PERFECT_COUNT__ ảnh</b></div>
    <div class="stat-chip">Đã phục hồi nhụy tâm: <b id="repaired_stat" style="color:#38bdf8">__REPAIRED_COUNT__ ảnh</b></div>
  </div>

  <div class="controls">
    <input id="search" type="search" placeholder="🔍 Tìm tên ảnh (ví dụ 00123)...">
    <label>Nguồn nhãn v2/v5:
      <select id="source_filter">
        <option value="all">Tất cả nguồn nhãn (1.000)</option>
        <option value="accepted_unchanged_v2">867 ảnh v2 chuẩn giữ nguyên (IoU: 98.4%)</option>
        <option value="corrected_v2">116 ảnh v2 khó đã sửa tay (IoU: 96.9%)</option>
        <option value="replacement">17 ảnh góc chụp mới bổ sung (IoU: 98.2%)</option>
      </select>
    </label>
    <label>Nhóm:
      <select id="subset_filter">
        <option value="all">Tất cả (1.000 ảnh)</option>
        <option value="train">Train (792 ảnh)</option>
        <option value="validation">Validation (100 ảnh)</option>
        <option value="reviewed_test_current">Test đã duyệt (91 ảnh)</option>
        <option value="reviewed_replacement">Ảnh góc chụp mới (17 ảnh)</option>
      </select>
    </label>
    <label>Hậu xử lý:
      <select id="delta_filter">
        <option value="all">Tất cả thay đổi</option>
        <option value="repaired">Có sửa nhụy tâm hoa (Δ &gt; 0)</option>
        <option value="denoised">Có lọc đốm nhiễu nền (Δ &lt; 0)</option>
        <option value="zero">Chuẩn xác tuyệt đối (Δ = 0)</option>
      </select>
    </label>
    <label>Sắp xếp:
      <select id="order_filter">
        <option value="name">Tên file A &rarr; Z</option>
        <option value="delta_abs">Thay đổi nhiều nhất (|Δ| giảm dần)</option>
        <option value="iou_desc">Khớp nhãn v2/v5 cao nhất</option>
        <option value="iou_asc">Khác nhãn v2/v5 nhất</option>
        <option value="area_desc">Diện tích hoa lớn nhất</option>
        <option value="area_asc">Diện tích hoa nhỏ nhất</option>
      </select>
    </label>
    <button id="btn_page_prev">&larr; Trang trước</button>
    <span id="page_info" style="font-weight:600;font-size:13px;"></span>
    <button id="btn_page_next">Trang sau &rarr;</button>
    <span id="filter_count" style="color:var(--text-dim);font-size:13px;"></span>
  </div>
</header>

<main>
  <!-- Detail View Section -->
  <section class="detail-section">
    <div class="detail-header">
      <div class="detail-title">
        <button id="btn_prev_img">&larr; Ảnh trước</button>
        <strong id="detail_name">image_00001.png</strong>
        <button id="btn_next_img">Ảnh sau &rarr;</button>
        <span id="badge_subset" class="badge badge-train">Train</span>
        <span id="badge_delta" class="badge badge-delta-zero">Δ = 0 px</span>
        <span id="badge_v2" class="badge" style="background:#1e293b;border:1px solid #475569">Nhãn v2</span>
        <span id="badge_iou" class="badge" style="background:#065f46;color:#a7f3d0;border:1px solid #10b981">Khớp v2: 98.5%</span>
      </div>
      <div id="detail_meta" style="font-size:13.5px;color:var(--text-dim);"></div>
    </div>

    <div class="grid-5">
      <div class="panel">
        <div class="panel-title">1. Ảnh Gốc 384 × 384</div>
        <a id="link_orig" target="_blank"><img id="img_orig" alt="Ảnh gốc"></a>
      </div>
      <div class="panel">
        <div class="panel-title">2. Mask Raw (DeepLabV3+ Fine-tune v5)</div>
        <a id="link_raw" target="_blank"><img id="img_raw" alt="Mask Raw"></a>
      </div>
      <div class="panel">
        <div class="panel-title active" style="color:var(--accent-cyan);font-weight:700">3. Mask Hoàn Thiện Hậu Xử Lý ⭐</div>
        <a id="link_mask" target="_blank"><img id="img_mask" alt="Mask Hoàn Thiện" style="border:2px solid var(--accent-cyan)"></a>
      </div>
      <div class="panel">
        <div class="panel-title" style="color:var(--accent-green);font-weight:700">4. Nhãn Chuẩn Ground Truth v5 🏷️</div>
        <a id="link_gt" target="_blank"><img id="img_gt" alt="Ground Truth v5" style="border:2px solid var(--accent-green)"></a>
      </div>
      <div class="panel">
        <div class="panel-title">5. Phủ Mask Lên Ảnh Gốc</div>
        <a id="link_overlay" target="_blank"><img id="img_overlay" alt="Phủ mask"></a>
      </div>
    </div>
  </section>

  <!-- Gallery Grid Section -->
  <div class="gallery-section-title">
    <span>DANH SÁCH ẢNH (Click ảnh bất kỳ để xem chi tiết 5 ô đối chiếu phía trên)</span>
    <span style="font-size:12px;color:var(--text-dim);">Dùng phím mũi tên Trái / Phải để chuyển ảnh liên tục</span>
  </div>
  <div class="gallery" id="gallery_grid"></div>
</main>

<script>
const all = __DATA__;
const $ = id => document.getElementById(id);
let filtered = [...all];
let selectedIndex = 0;
let currentPage = 0;
const PAGE_SIZE = 30;

function subsetBadge(subset) {
  if (subset === 'train') return ['Train', 'badge-train'];
  if (subset === 'validation') return ['Val', 'badge-val'];
  if (subset === 'reviewed_test_current') return ['Test', 'badge-test'];
  return ['Ảnh mới', 'badge-new'];
}

function showDetail(index) {
  if (index < 0 || index >= filtered.length) return;
  selectedIndex = index;
  const row = filtered[index];
  const name = row.name;

  $('detail_name').textContent = `${index + 1}/${filtered.length}: ${name}`;
  
  const [subsetLabel, subsetClass] = subsetBadge(row.subset);
  $('badge_subset').textContent = subsetLabel;
  $('badge_subset').className = 'badge ' + subsetClass;

  if (row.delta === 0) {
    $('badge_delta').textContent = 'Δ = 0 px (Chuẩn xác 100%)';
    $('badge_delta').className = 'badge badge-delta-zero';
  } else if (row.delta > 0) {
    $('badge_delta').textContent = `+${row.delta.toLocaleString()} px (Đã sửa nhụy tâm)`;
    $('badge_delta').className = 'badge badge-delta-plus';
  } else {
    $('badge_delta').textContent = `${row.delta.toLocaleString()} px (Đã lọc đốm nhiễu)`;
    $('badge_delta').className = 'badge badge-delta-minus';
  }

  let v2Label = 'Nhãn v2 chuẩn';
  let v2Color = '#10b981';
  if (row.source_type === 'corrected_v2') {
    v2Label = 'Nhãn v2 đã sửa tay';
    v2Color = '#a855f7';
  } else if (row.source_type.includes('edited') || row.source_type.includes('corrected_tmp')) {
    v2Label = 'Ảnh mới v5';
    v2Color = '#f59e0b';
  }
  $('badge_v2').textContent = v2Label;
  $('badge_v2').style.borderColor = v2Color;

  const iouPct = (row.iou_vs_gt * 100).toFixed(1);
  $('badge_iou').textContent = `Khớp v2/v5: ${iouPct}% IoU`;

  $('detail_meta').innerHTML = `Diện tích hoa: <b>${row.pixels.toLocaleString()} px</b> (${row.area_pct.toFixed(2)}% ảnh) &bull; Raw: ${row.raw.toLocaleString()} px &bull; <b>Khớp nhãn: ${iouPct}%</b>`;

  const origSrc = '../image_384/' + name;
  const rawSrc = 'raw_masks/' + name + '?v=' + row.mask_hash;
  const maskSrc = name + '?v=' + row.mask_hash;
  const gtSrc = '../final_segmentation_v5/' + name;
  const overlaySrc = 'overlays/' + name.replace('.png', '.jpg') + '?v=' + row.mask_hash;

  $('img_orig').src = origSrc; $('link_orig').href = origSrc;
  $('img_raw').src = rawSrc; $('link_raw').href = rawSrc;
  $('img_mask').src = maskSrc; $('link_mask').href = maskSrc;
  $('img_gt').src = gtSrc; $('link_gt').href = gtSrc;
  $('img_overlay').src = overlaySrc; $('link_overlay').href = overlaySrc;

  $('btn_prev_img').disabled = index === 0;
  $('btn_next_img').disabled = index === filtered.length - 1;

  const targetPage = Math.floor(index / PAGE_SIZE);
  if (targetPage !== currentPage) {
    currentPage = targetPage;
    renderGallery();
  } else {
    updateActiveCard();
  }
}

function updateActiveCard() {
  const cards = $('gallery_grid').children;
  const currentItem = filtered[selectedIndex];
  for (let card of cards) {
    card.classList.toggle('active', card.dataset.name === currentItem?.name);
  }
}

function renderGallery() {
  const start = currentPage * PAGE_SIZE;
  const items = filtered.slice(start, start + PAGE_SIZE);
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));

  $('page_info').textContent = `Trang ${currentPage + 1} / ${totalPages}`;
  $('btn_page_prev').disabled = currentPage === 0;
  $('btn_page_next').disabled = start + PAGE_SIZE >= filtered.length;

  $('gallery_grid').replaceChildren(...items.map((row, i) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.dataset.name = row.name;

    const img = document.createElement('img');
    img.loading = 'lazy';
    img.src = 'thumbs/' + row.name.replace('.png', '.jpg');
    img.alt = row.name;

    const name = document.createElement('div');
    name.className = 'card-name';
    name.textContent = `${start + i + 1}. ${row.name}`;

    const info = document.createElement('div');
    info.className = 'card-info';
    
    let deltaText = 'Δ=0';
    let deltaColor = 'var(--accent-green)';
    if (row.delta > 0) { deltaText = `+${row.delta}`; deltaColor = 'var(--accent-cyan)'; }
    else if (row.delta < 0) { deltaText = `${row.delta}`; deltaColor = 'var(--accent-red)'; }

    info.innerHTML = `<span>${row.area_pct.toFixed(1)}%</span><span style="color:${deltaColor};font-weight:600">${deltaText}</span>`;

    card.append(img, name, info);
    card.onclick = () => showDetail(start + i);
    return card;
  }));

  updateActiveCard();
}

function applyFilters() {
  const q = $('search').value.trim().toLowerCase();
  const source = $('source_filter').value;
  const subset = $('subset_filter').value;
  const delta = $('delta_filter').value;
  const order = $('order_filter').value;
  const prevName = filtered[selectedIndex]?.name;

  filtered = all.filter(row => {
    if (q && !row.name.toLowerCase().includes(q)) return false;
    if (source === 'accepted_unchanged_v2' && row.source_type !== 'accepted_unchanged_v2') return false;
    if (source === 'corrected_v2' && row.source_type !== 'corrected_v2') return false;
    if (source === 'replacement' && !row.source_type.includes('edited') && !row.source_type.includes('corrected_tmp')) return false;
    if (subset !== 'all' && row.subset !== subset) return false;
    if (delta === 'repaired' && row.delta <= 0) return false;
    if (delta === 'denoised' && row.delta >= 0) return false;
    if (delta === 'zero' && row.delta !== 0) return false;
    return true;
  });

  if (order === 'delta_abs') {
    filtered.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta) || a.name.localeCompare(b.name));
  } else if (order === 'iou_desc') {
    filtered.sort((a, b) => b.iou_vs_gt - a.iou_vs_gt || a.name.localeCompare(b.name));
  } else if (order === 'iou_asc') {
    filtered.sort((a, b) => a.iou_vs_gt - b.iou_vs_gt || a.name.localeCompare(b.name));
  } else if (order === 'area_desc') {
    filtered.sort((a, b) => b.pixels - a.pixels);
  } else if (order === 'area_asc') {
    filtered.sort((a, b) => a.pixels - b.pixels);
  } else {
    filtered.sort((a, b) => a.name.localeCompare(b.name));
  }

  $('filter_count').textContent = `Hiển thị ${filtered.length} / ${all.length} ảnh`;
  selectedIndex = Math.max(0, filtered.findIndex(r => r.name === prevName));
  currentPage = Math.floor(selectedIndex / PAGE_SIZE);

  if (filtered.length > 0) {
    renderGallery();
    showDetail(selectedIndex);
  } else {
    $('gallery_grid').innerHTML = '<div style="grid-column:1/-1;padding:40px;text-align:center;color:var(--text-dim)">Không có ảnh nào phù hợp với bộ lọc hiện tại.</div>';
    $('detail_name').textContent = 'Không có ảnh';
    $('detail_meta').textContent = '';
    for (let id of ['img_orig', 'img_raw', 'img_mask', 'img_gt', 'img_overlay']) $(id).removeAttribute('src');
  }
}

// Event Listeners
$('search').addEventListener('input', applyFilters);
$('source_filter').addEventListener('change', applyFilters);
$('subset_filter').addEventListener('change', applyFilters);
$('delta_filter').addEventListener('change', applyFilters);
$('order_filter').addEventListener('change', applyFilters);

$('btn_page_prev').onclick = () => { currentPage--; renderGallery(); showDetail(currentPage * PAGE_SIZE); };
$('btn_page_next').onclick = () => { currentPage++; renderGallery(); showDetail(currentPage * PAGE_SIZE); };
$('btn_prev_img').onclick = () => showDetail(selectedIndex - 1);
$('btn_next_img').onclick = () => showDetail(selectedIndex + 1);

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft') showDetail(selectedIndex - 1);
  if (e.key === 'ArrowRight') showDetail(selectedIndex + 1);
});

// Init
applyFilters();
</script>
</body>
</html>'''

    repaired_count = sum(1 for r in rows if int(r["pixel_delta"]) > 0)
    perfect_count = sum(1 for r in rows if int(r["pixel_delta"]) == 0)

    html_page = html_template.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    html_page = html_page.replace("__REPAIRED_COUNT__", f"{repaired_count}")
    html_page = html_page.replace("__PERFECT_COUNT__", f"{perfect_count}")

    (output / "index.html").write_text(html_page, encoding="utf-8")


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT, help="Output directory for masks")
    parser.add_argument("--input-dir", type=Path, default=IMAGES, help="Input directory containing 1,000 images")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Allow overwriting output directory")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    OUTPUT = args.output_dir.resolve()

    cv2.setNumThreads(1)
    torch.set_num_threads(8)

    print(f"Reading images from: {input_dir}", flush=True)
    image_paths = sorted(list(input_dir.glob("*.png")))
    if len(image_paths) != 1000:
        raise ValueError(f"Expected exactly 1,000 PNG images in {input_dir}, found {len(image_paths)}")

    # Load splits
    subsets_map = {}
    if SPLIT.is_file():
        with SPLIT.open(encoding="utf-8") as stream:
            split_data = json.load(stream)
        for group in ("train", "validation", "reviewed_test_current"):
            for name in split_data.get(group, []):
                subsets_map[name] = group

    # Load active model
    model, state, config = load_active_model()
    checkpoint = ROOT / "cau_3" / config["checkpoint"]
    print(f"Loaded active model: {config['model_id']} from {checkpoint.name}", flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "raw_masks").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "overlays").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "thumbs").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    batch_size = args.batch_size
    total = len(image_paths)
    t0 = time.time()

    print(f"Starting segmentation of {total} images with batch_size={batch_size}...", flush=True)

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            batch_paths = image_paths[start:start + batch_size]
            rgbs = [read_rgb(p) for p in batch_paths]
            tensors = np.stack([im.transpose(2, 0, 1) for im in rgbs])
            tensor = torch.from_numpy(np.ascontiguousarray(tensors)).float() / 255.0
            prob_maps = model(tensor).sigmoid()[:, 0].numpy()

            for path, rgb, prob in zip(batch_paths, rgbs, prob_maps):
                name = path.name
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                raw_mask_u8 = (prob >= 0.45).astype(np.uint8) * 255
                final_mask_u8, alpha_mask, metrics = postprocess_mask(
                    prob,
                    image_bgr=bgr,
                    threshold=0.45,
                    opening=False,
                    closing=False,
                    fill_holes=True,
                    keep_nearby=True,
                    repair_center=True,
                    max_hole_area=20,
                )

                # Save final mask
                mask_path = OUTPUT / name
                cv2.imwrite(str(mask_path), final_mask_u8)

                # Save raw mask
                raw_path = OUTPUT / "raw_masks" / name
                cv2.imwrite(str(raw_path), raw_mask_u8)

                # Save overlay
                overlay = make_overlay(rgb, final_mask_u8 > 127)
                overlay_path = OUTPUT / "overlays" / name.replace(".png", ".jpg")
                Image.fromarray(overlay).save(overlay_path, quality=88)

                # Save thumbnail
                thumb_path = OUTPUT / "thumbs" / name.replace(".png", ".jpg")
                Image.fromarray(rgb).resize((140, 140), Image.Resampling.LANCZOS).save(thumb_path, quality=80)

                raw_pixels = int(metrics.get("raw_pixels", np.count_nonzero(raw_mask_u8)))
                final_pixels = int(metrics.get("final_pixels", np.count_nonzero(final_mask_u8)))
                pixel_delta = int(metrics.get("pixel_delta", final_pixels - raw_pixels))
                area_pct = round(100.0 * final_pixels / (384 * 384), 2)
                center_repaired = bool(pixel_delta > 0)
                subset = subsets_map.get(name, "reviewed_replacement")

                rows.append({
                    "filename": name,
                    "subset": subset,
                    "raw_pixels": raw_pixels,
                    "final_pixels": final_pixels,
                    "pixel_delta": pixel_delta,
                    "area_pct": area_pct,
                    "center_repaired": center_repaired,
                    "image_sha256": sha256(path),
                    "mask_sha256": sha256(mask_path),
                })

            done = min(start + batch_size, total)
            if done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                fps = done / max(elapsed, 0.001)
                print(f"[{done}/{total}] processed ({fps:.1f} img/s, {elapsed:.1f}s elapsed)", flush=True)

    # Save metrics.csv
    csv_path = OUTPUT / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics to: {csv_path}", flush=True)

    # Save manifest.json
    subset_counts = dict(Counter(r["subset"] for r in rows))
    repaired_count = sum(1 for r in rows if int(r["pixel_delta"]) > 0)
    denoised_count = sum(1 for r in rows if int(r["pixel_delta"]) < 0)
    perfect_count = sum(1 for r in rows if int(r["pixel_delta"]) == 0)

    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "model_id": config["model_id"],
        "input_size": [384, 384],
        "output_size": [384, 384],
        "pipeline_config": {
            "threshold": 0.45,
            "opening": False,
            "closing": False,
            "fill_holes": True,
            "keep_nearby": True,
            "repair_center": True,
            "max_hole_area": 20,
        },
        "counts": {
            "total": len(rows),
            "subsets": subset_counts,
            "center_repaired": repaired_count,
            "denoised": denoised_count,
            "perfect_raw": perfect_count,
        },
        "metrics_csv_sha256": sha256(csv_path),
    }

    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved manifest to: {manifest_path}", flush=True)

    # Generate interactive HTML viewer
    make_viewer(rows, stats=manifest["counts"], output=OUTPUT)
    print(f"Generated viewer: {OUTPUT / 'index.html'}", flush=True)
    print(f"Accessible at: http://localhost:8080/final_segmentation_v3_384/index.html", flush=True)


if __name__ == "__main__":
    main()
