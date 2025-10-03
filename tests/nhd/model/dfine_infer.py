# tests/nhd/model/dfine_infer.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image

from transformers import AutoImageProcessor, DFineForObjectDetection

# Modèle public aligné avec ton app
DFINE_HF_ID = "ustc-community/dfine-xlarge-obj2coco"
_LOCAL_DIR = Path("model/dfine")  # optionnel : snapshot local
_PROC = None
_MODEL = None

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _load_model(device: str = None):
    """Charge d'abord en local si model/dfine/ existe, sinon depuis HF."""
    global _PROC, _MODEL
    if _PROC is not None and _MODEL is not None:
        return _PROC, _MODEL

    device = device or _device()
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if _LOCAL_DIR.exists():
        ckpt = _LOCAL_DIR
    else:
        ckpt = DFINE_HF_ID

    _PROC = AutoImageProcessor.from_pretrained(ckpt)
    _MODEL = DFineForObjectDetection.from_pretrained(
        ckpt, torch_dtype=torch_dtype
    ).to(device).eval()
    return _PROC, _MODEL

def _xyxy_to_xywh(box_xyxy: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = box_xyxy
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dtype=np.float32)

@torch.inference_mode()
def detect_persons(
    img: Image.Image,
    conf_thr: float = 0.25,
    device: str = "cpu",
) -> List[Tuple[int, np.ndarray, float]]:
    """
    Retourne une liste de (class_id, bbox_xywh_abs, conf)
    - class_id = 0 (person)
    - bbox_xywh en pixels absolus (centre_x, centre_y, width, height)
    """
    proc, model = _load_model(device=device)
    W, H = img.size
    np_img = np.array(img.convert("RGB"))

    inputs = proc(images=np_img, return_tensors="pt").to(model.device)
    if model.device.type == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

    outputs = model(**inputs)
    results = proc.post_process_object_detection(
        outputs, target_sizes=[(H, W)], threshold=conf_thr
    )[0]

    dets: List[Tuple[int, np.ndarray, float]] = []
    boxes = results["boxes"].cpu().numpy()        # (N,4) xyxy
    scores = results["scores"].cpu().numpy()      # (N,)
    labels = results["labels"].cpu().numpy()      # (N,)

    for lab, box, sc in zip(labels, boxes, scores):
        if lab == 0 and sc >= conf_thr:  # 0 = person
            xywh = _xyxy_to_xywh(box)
            dets.append((0, xywh, float(sc)))

    return dets
