"""Evaluate the frozen DeepLabV3+ checkpoint on 150 held-out reviewed masks.

The reviewed test masks are used only after training and checkpoint selection.
Outputs are isolated in the DeepLabV3+ pilot directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from train_flower_deeplabv3plus import DeepLabV3Plus, OUTPUT, ROOT, SPLIT, sha256


CURRENT_IMAGES = ROOT / "image_384"
EXTERNAL_IMAGES = ROOT / "flowers_384_export_20260923_083819" / "images_384"
GT = ROOT / "flower_features" / "segmentation_ml" / "ground_truth" / "masks_384"
UNET = ROOT / "cau_3" / "trained_seed_pilot_20261022"
FIELDS = ["filename", "subset", "gt_pixels", "deeplab_pixels", "deeplab_iou_gt",
          "deeplab_dice_gt", "unet_iou_gt", "unet_grabcut_iou_gt", "current_iou_gt"]


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != (384, 384):
            raise ValueError(f"Image is not 384x384: {path}")
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != (384, 384):
            raise ValueError(f"Mask is not 384x384: {path}")
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127


@torch.no_grad()
def predict(model: DeepLabV3Plus, rgb: np.ndarray, size: int) -> np.ndarray:
    smaller = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(smaller.transpose(2, 0, 1))).float()[None] / 255.0
    probabilities = model(tensor).sigmoid()[0, 0].numpy()
    return np.clip(cv2.resize(probabilities, (384, 384), interpolation=cv2.INTER_LINEAR), 0, 1)


def scores(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    intersection = int(np.count_nonzero(prediction & truth))
    union = int(np.count_nonzero(prediction | truth))
    total = int(prediction.sum() + truth.sum())
    return intersection / max(union, 1), 2 * intersection / max(total, 1)


def summarize(rows: list[dict]) -> dict:
    result = {}
    for group in ("all", "current", "external"):
        subset = rows if group == "all" else [row for row in rows if row["subset"] == group]
        deeplab = np.array([float(row["deeplab_iou_gt"]) for row in subset])
        unet = np.array([float(row["unet_iou_gt"]) for row in subset])
        unet_gc = np.array([float(row["unet_grabcut_iou_gt"]) for row in subset])
        entry = {"count": len(subset), "deeplab_iou_mean": float(deeplab.mean()),
                 "deeplab_iou_median": float(np.median(deeplab)),
                 "deeplab_below_0_8": int((deeplab < 0.8).sum()),
                 "unet_iou_mean": float(unet.mean()),
                 "unet_grabcut_iou_mean": float(unet_gc.mean()),
                 "deeplab_better_than_unet": int((deeplab > unet).sum()),
                 "deeplab_better_than_unet_grabcut": int((deeplab > unet_gc).sum())}
        if group == "current":
            entry["current_mask_iou_mean"] = float(np.mean([float(row["current_iou_gt"]) for row in subset]))
        result[group] = entry
    result["worst_deeplab"] = sorted(rows, key=lambda row: row["deeplab_iou_gt"])[:15]
    return result


def make_viewer(rows: list[dict]) -> None:
    data = [{"name": row["filename"], "subset": row["subset"],
             "deep": round(float(row["deeplab_iou_gt"]), 4),
             "unet": round(float(row["unet_grabcut_iou_gt"]), 4)} for row in rows]
    html = r'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepLabV3+ trên ảnh chưa huấn luyện</title><style>
:root{font:16px system-ui;background:#10212a;color:#f3f7f9}*{box-sizing:border-box}body{margin:0}
header{position:sticky;top:0;z-index:2;background:#1b3847;border-bottom:1px solid #67808d;padding:12px 17px}h1{font-size:20px;margin:0 0 6px}p{margin:6px 0;line-height:1.4}
.controls{display:flex;align-items:center;flex-wrap:wrap;gap:9px;margin-top:10px}select,input,button{font:inherit;color:white;background:#1c3a49;border:1px solid #7996a3;border-radius:6px;padding:7px 9px}button{cursor:pointer}
main{max-width:1800px;margin:auto;padding:14px}#list{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:7px;max-height:45vh;overflow:auto;margin:12px 0}#list button{text-align:left;font-size:12px;padding:5px}#list button.active{border-color:#9ce0ff;background:#267eaa}.pair{display:grid;grid-template-columns:1fr 1fr;gap:3px}.pair img{width:100%;aspect-ratio:1;object-fit:contain;background:#c8d0d3}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.panel{background:#274454;border:1px solid #607a86}.panel strong{display:block;padding:8px}.panel a{display:block;background-color:#d8e0e2;background-image:repeating-conic-gradient(#bcc7cb 0 25%,white 0 50%);background-size:24px 24px}.panel img{width:100%;aspect-ratio:1;object-fit:contain;display:block}.note{color:#c2d2d9;margin:12px 0}
@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){header{position:static}.grid{grid-template-columns:1fr}}
</style></head><body><header><h1>DeepLabV3+ trên 150 ảnh chưa huấn luyện</h1>
<p>42 ảnh ngoài bộ 1.000 và 108 ảnh giữ riêng trong bộ. Chỉ mask DeepLabV3+ thuần, chưa qua GrabCut.</p>
<div class="controls"><label>Nhóm <select id="subset"><option value="external">42 ảnh ngoài bộ</option><option value="current">108 ảnh trong bộ</option><option value="all">Cả 150 ảnh</option></select></label><label>Sắp xếp <select id="order"><option value="worst">IoU thấp trước</option><option value="name">Tên file</option></select></label><input id="search" placeholder="Tìm tên ảnh"><button id="prev">← Trước</button><button id="next">Sau →</button><span id="count"></span></div></header>
<main><div id="list"></div><p id="status"></p><div class="grid">
<div class="panel"><strong>Ảnh gốc</strong><a id="original_link" target="_blank"><img id="original"></a></div>
<div class="panel"><strong>Hoa tách nền bằng DeepLabV3+</strong><a id="deep_cutout_link" target="_blank"><img id="deep_cutout"></a></div>
<div class="panel"><strong>Mask đã duyệt</strong><a id="truth_link" target="_blank"><img id="truth"></a></div>
<div class="panel"><strong>Mask DeepLabV3+</strong><a id="deep_mask_link" target="_blank"><img id="deep_mask"></a></div>
<div class="panel"><strong>U-Net + GrabCut thử nghiệm trước</strong><a id="unet_mask_link" target="_blank"><img id="unet_mask"></a></div>
<div class="panel" id="current_panel"><strong>Mask hiện dùng trong CSDL</strong><a id="current_mask_link" target="_blank"><img id="current_mask"></a></div>
</div><p class="note">Nhấp vào ảnh để xem 384 × 384. IoU đo với mask đã duyệt; trường hợp mask đối chiếu có lỗi cũng cần xem bằng mắt.</p></main>
<script>const all=__DATA__,$=id=>document.getElementById(id);let shown=[],selected=0;
function setImage(id,path){$(id).src=path;$(id+'_link').href=path}
function show(index){if(index<0||index>=shown.length)return;selected=index;const row=shown[index],name=row.name,source=(row.subset==='external'?'../../flowers_384_export_20260923_083819/images_384/':'../../image_384/')+name;
setImage('original',source);setImage('deep_cutout','cutouts_reviewed/'+name);setImage('truth','../../flower_features/segmentation_ml/ground_truth/masks_384/'+name);setImage('deep_mask','masks_reviewed/'+name);setImage('unet_mask','../trained_seed_pilot_20261022/grabcut_masks_reviewed/'+name);
$('current_panel').hidden=row.subset!=='current';if(row.subset==='current')setImage('current_mask','../../final_segmentation_1/'+name);
$('status').textContent=`${index+1}/${shown.length}: ${name} · DeepLabV3+ IoU ${row.deep} · U-Net + GrabCut IoU ${row.unet}`;
for(const button of $('list').children)button.classList.toggle('active',button.dataset.name===name);$('prev').disabled=index===0;$('next').disabled=index===shown.length-1}
function render(){const old=shown[selected]?.name,subset=$('subset').value,q=$('search').value.trim().toLowerCase();shown=all.filter(row=>(subset==='all'||row.subset===subset)&&row.name.toLowerCase().includes(q));
if($('order').value==='worst')shown.sort((a,b)=>a.deep-b.deep);else shown.sort((a,b)=>a.name.localeCompare(b.name));$('count').textContent=`${shown.length} ảnh`;
$('list').replaceChildren(...shown.map(row=>{const b=document.createElement('button');b.dataset.name=row.name;const pair=document.createElement('div');pair.className='pair';
const original=document.createElement('img');original.loading='lazy';original.src=(row.subset==='external'?'../../flowers_384_export_20260923_083819/images_384/':'../../image_384/')+row.name;
const cutout=document.createElement('img');cutout.loading='lazy';cutout.src='cutouts_reviewed/'+row.name;pair.append(original,cutout);
const label=document.createElement('span');label.textContent=`${row.name} · IoU ${row.deep}`;b.append(pair,label);b.onclick=()=>show(shown.indexOf(row));return b}));
if(shown.length)show(Math.max(0,shown.findIndex(row=>row.name===old)));else $('status').textContent='Không tìm thấy ảnh'}
for(const id of ['subset','order','search'])$(id).addEventListener(id==='search'?'input':'change',render);$('prev').onclick=()=>show(selected-1);$('next').onclick=()=>show(selected+1);render();</script></body></html>'''.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    (OUTPUT / "reviewed_viewer.html").write_text(html, encoding="utf-8")


def main() -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(8)
    with SPLIT.open(encoding="utf-8") as source:
        split = json.load(source)
    names = split["reviewed_test_current"] + split["reviewed_test_external"]
    if len(names) != 150 or set(names) & (set(split["train"]) | set(split["validation"])):
        raise ValueError("The reviewed holdout split is invalid")
    checkpoint = OUTPUT / "best_deeplabv3plus.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["split_sha256"] != sha256(SPLIT):
        raise ValueError("Checkpoint and split hashes differ")
    model = DeepLabV3Plus(width=state["width"], rates=tuple(state["rates"]))
    model.load_state_dict(state["state_dict"])
    model.eval()
    (OUTPUT / "masks_reviewed").mkdir(exist_ok=True)
    (OUTPUT / "cutouts_reviewed").mkdir(exist_ok=True)
    with (UNET / "reviewed_comparison.csv").open(newline="", encoding="utf-8") as source:
        unet_rows = {row["filename"]: row for row in csv.DictReader(source)}

    rows = []
    for index, name in enumerate(names, 1):
        subset = "current" if name in split["reviewed_test_current"] else "external"
        image_dir = CURRENT_IMAGES if subset == "current" else EXTERNAL_IMAGES
        rgb, truth = read_rgb(image_dir / name), read_mask(GT / name)
        prediction = predict(model, rgb, state["size"]) >= 0.5
        Image.fromarray(prediction.astype(np.uint8) * 255).save(OUTPUT / "masks_reviewed" / name)
        Image.fromarray(np.dstack((rgb, prediction.astype(np.uint8) * 255)), "RGBA").save(
            OUTPUT / "cutouts_reviewed" / name)
        iou, dice = scores(prediction, truth)
        old = unet_rows[name]
        rows.append({"filename": name, "subset": subset, "gt_pixels": int(truth.sum()),
                     "deeplab_pixels": int(prediction.sum()), "deeplab_iou_gt": iou,
                     "deeplab_dice_gt": dice, "unet_iou_gt": float(old["model_iou_gt"]),
                     "unet_grabcut_iou_gt": float(old["grabcut_iou_gt"]),
                     "current_iou_gt": float(old["current_iou_gt"]) if subset == "current" else ""})
        if index % 25 == 0:
            print(f"Evaluated {index}/150", flush=True)
    with (OUTPUT / "reviewed_comparison.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary["checkpoint_sha256"] = sha256(checkpoint)
    summary["split_sha256"] = sha256(SPLIT)
    summary["note"] = "150 reviewed masks are held out from DeepLab training and model selection. The 42 external images still belong to the same original 1,256-image collection."
    (OUTPUT / "reviewed_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    make_viewer(rows)
    print(json.dumps({key: summary[key] for key in ("all", "current", "external")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
