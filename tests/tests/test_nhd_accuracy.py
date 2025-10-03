from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import os
import io
import math

import numpy as np
from PIL import Image, ImageDraw
import pytest
import allure

from nhd.model.dfine_infer import detect_persons

IMG_DIR = Path(__file__).parent.parent / "nhd/images"
LBL_DIR = Path(__file__).parent.parent / "nhd/labels"

IOU_THR = float(os.getenv("DET_IOU_THR", "0.50"))
CONF_THR = float(os.getenv("DET_CONF_THR", "0.25"))
MIN_MAP50 = float(os.getenv("DET_MIN_MAP50", "0.30"))


def yolo_txt_load(path: Path, img_w: int, img_h: int):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cid, xc, yc, w, h = map(float, line.split()[:5])
        x1 = (xc - w/2) * img_w
        y1 = (yc - h/2) * img_h
        x2 = (xc + w/2) * img_w
        y2 = (yc + h/2) * img_h
        out.append((int(cid), [x1, y1, x2, y2]))
    return out


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (ay2 - ay1))
    union = area_a + area_b - inter + 1e-6
    return inter / union


def match_detections_to_gt(dets, gts, iou_thr):
    dets_sorted = sorted(dets, key=lambda x: -x[1])
    tp = []
    fp = []
    matched = set()
    for i, (img_id, score, bbox) in enumerate(dets_sorted):
        best_iou = 0.0
        best_j = -1
        for j, (g_img, g_bbox, used) in enumerate(gts):
            if g_img != img_id or used:
                continue
            iou = iou_xyxy(bbox, g_bbox)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thr:
            tp.append(1); fp.append(0)
            gts[best_j] = (gts[best_j][0], gts[best_j][1], True)
        else:
            tp.append(0); fp.append(1)
    return np.array(tp), np.array(fp), len([1 for _,_,u in gts if not u or u])


def compute_ap(tp, fp, npos):
    if len(tp) == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(1, npos)
    precis = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precis[recalls >= t].max() if np.any(recalls >= t) else 0
        ap += p
    return ap / 11.0


def collect_pairs():
    imgs = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    pairs = []
    for ip in imgs:
        lp = LBL_DIR / (ip.stem + ".txt")
        pairs.append((ip, lp))
    assert len(pairs) > 0, "No images found in images/"
    return pairs


def _draw_boxes_on_image(img: Image.Image, boxes: List[List[float]], color=(0,255,0), width=3, labels: List[str] | None=None) -> Image.Image:
    """Dessine des boîtes (xyxy) sur une copie de l'image."""
    im = img.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if labels and i < len(labels) and labels[i]:
            # simple label au-dessus de la box
            tx, ty = x1 + 2, max(0, y1 - 12)
            draw.text((tx, ty), labels[i], fill=color)
    return im


def _attach_image_to_allure(name: str, pil_image: Image.Image):
    """Sauvegarde une image PIL en PNG et l'attache au rapport Allure."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    allure.attach(buf.read(), name=name, attachment_type=allure.attachment_type.PNG)


pairs = collect_pairs()


@pytest.mark.parametrize("img_path,lbl_path", pairs, ids=[p.stem for p,_ in pairs])
def test_person_map_per_image_smoke(img_path: Path, lbl_path: Path):
    allure.dynamic.suite("nhd")
    img = Image.open(img_path).convert("RGB")

    # GT boxes (xyxy pixels)
    W, H = img.size
    gts_xyxy = [bbox for cid, bbox in yolo_txt_load(lbl_path, W, H) if cid == 0]

    # Predictions
    dets = detect_persons(img, conf_thr=CONF_THR, device="cpu")
    assert isinstance(dets, list)

    # Convert predicted xywh -> xyxy for visualization
    preds_xyxy = []
    for cid, xywh, score in dets:
        cx, cy, w, h = xywh
        x0 = cx - w / 2.0
        y0 = cy - h / 2.0
        x1 = cx + w / 2.0
        y1 = cy + h / 2.0
        preds_xyxy.append([x0, y0, x1, y1])

    # Attach original, expected (GT), and received (preds)
    _attach_image_to_allure("original", img)
    if gts_xyxy:
        gt_img = _draw_boxes_on_image(img, gts_xyxy, color=(0,255,0), width=3, labels=["GT"]*len(gts_xyxy))
        _attach_image_to_allure("expected_GT", gt_img)
    else:
        allure.attach("No GT boxes for this image.", name="expected_GT", attachment_type=allure.attachment_type.TEXT)

    if preds_xyxy:
        pd_img = _draw_boxes_on_image(img, preds_xyxy, color=(255,0,0), width=3, labels=["PRED"]*len(preds_xyxy))
        _attach_image_to_allure("received_PRED", pd_img)
    else:
        allure.attach("No predictions.", name="received_PRED", attachment_type=allure.attachment_type.TEXT)


def test_dataset_map50_person():
    allure.dynamic.suite("nhd")
    dets_agg = []
    gts_agg = []
    summary_lines = []

    for idx, (img_path, lbl_path) in enumerate(pairs):
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        gts = [(idx, bbox, False) for cid, bbox in yolo_txt_load(lbl_path, W, H) if cid == 0]
        gts_agg.extend(gts)

        dets = detect_persons(img, conf_thr=CONF_THR, device="cpu")
        preds_count = 0
        for cid, xywh, score in dets:
            cx, cy, w, h = xywh
            x0 = cx - w / 2.0
            y0 = cy - h / 2.0
            x1 = cx + w / 2.0
            y1 = cy + h / 2.0
            bbox_xyxy = np.array([x0, y0, x1, y1], dtype=np.float32)
            dets_agg.append((idx, float(score), bbox_xyxy))
            preds_count += 1

        summary_lines.append(f"{img_path.name}: GT={len(gts)} PRED={preds_count}")

    tp, fp, npos = match_detections_to_gt(dets_agg, gts_agg, IOU_THR)
    ap50 = compute_ap(tp, fp, npos)
    msg = f"[Person] AP@{IOU_THR:.2f} = {ap50:.3f} (npos={npos}, ndets={len(dets_agg)})"
    print(msg)

    # Attach a small text summary to Allure for the dataset-level test
    allure.attach(
        "\n".join(summary_lines) + "\n" + msg,
        name="dataset_summary",
        attachment_type=allure.attachment_type.TEXT
    )

    assert ap50 >= MIN_MAP50, f"AP@{IOU_THR:.2f}={ap50:.3f} < target {MIN_MAP50:.2f}"
