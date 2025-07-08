from __future__ import annotations
from pathlib import Path
import time
import math
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, DFineForObjectDetection

DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"
SEG_MODEL_PATH          = "water-detection/model-v2/nwd-v2.pt"
DFINE_MODEL_ID          = "ustc-community/dfine-xlarge-obj2coco"
CONF_THRES              = 0.4
MAP_W_PX, MAP_H_PX      = 400, 200
UPDATE_EVERY            = 1000          # update homography every N frames
MIN_WATER_AREA_PX       = 5_000
DISPLAY_W, DISPLAY_H    = 1280, 720    # server‑side resize before streaming
MINIMAP_W, MINIMAP_H    = 320, 160
PAD                     = 12
MAX_DISTANCE_PX         = 100
MAX_DISAPPEARED_FRAMES  = 300
UNDERWATER_THRESHOLD    = 15           # frames without detection
SURFACE_THRESHOLD       = 5            # consecutive detections to be surfaced
DANGER_TIME_THRESHOLD   = 10           # seconds underwater before danger alert
DANGER_COLOR = (0, 0, 255)

class BoxStub:
    """Lightweight container so we can reuse DFINE code unmodified."""

    def __init__(self, cx: float, cy: float, w: float, h: float, conf: float):
        self.xywh = torch.tensor([[cx, cy, w, h]])
        self.cls = torch.tensor([0])  # class 0 = person
        self.conf = torch.tensor([conf])


def _euclidean(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


class UnderwaterPersonTracker:
    """Tracker aware of surface / underwater status & danger scoring."""

    def __init__(self,
                 max_distance: int = MAX_DISTANCE_PX,
                 max_disappeared: int = MAX_DISAPPEARED_FRAMES):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.next_id: int = 1
        self.tracks: dict[int, dict] = {}
        self.frame_rate = 30.0  # updated later from video metadata

    def _new_track(self, center, ts):
        return {
            "center": center,
            "disappeared": 0,
            "history": [center],
            "status": "surface",  # surface | underwater
            "frames_underwater": 0,
            "frames_surface": 0,
            "last_surface_ts": ts,
            "under_start_ts": None,
            "under_duration": 0.0,
            "submersion_events": [],
            "danger_alert_sent": False,
            "danger_score": 0,
            "distance_shore": 0.0,
            "dive_point": None,
        }

    def update(self, detections: list[BoxStub], ts: float | None = None):
        if ts is None:
            ts = time.time()

        def _mark_missing(tid):
            tr = self.tracks[tid]
            tr["disappeared"] += 1
            tr["frames_underwater"] += 1
            tr["frames_surface"] = 0
            if tr["frames_underwater"] >= UNDERWATER_THRESHOLD and tr["status"] != "underwater":
                tr["status"] = "underwater"
                tr["under_start_ts"] = ts
                tr["dive_point"] = tr["center"]
                tr["danger_alert_sent"] = False
                print(f"🌊 Person {tid} UNDERWATER")

        if not detections:
            lost = []
            for tid in list(self.tracks):
                _mark_missing(tid)
                # trigger danger alert if needed
                tr = self.tracks[tid]
                if (tr["status"] == "underwater" and tr["under_start_ts"]
                        and ts - tr["under_start_ts"] > DANGER_TIME_THRESHOLD
                        and not tr["danger_alert_sent"]):
                    print(
                        f"🚨 DANGER ALERT: Person {tid} underwater for {ts - tr['under_start_ts']:.1f}s")
                    tr["danger_alert_sent"] = True
                if tr["disappeared"] > self.max_disappeared:
                    if tr["status"] == "underwater" and tr["under_start_ts"]:
                        tr["submersion_events"].append(
                            (tr["under_start_ts"], ts - tr["under_start_ts"]))
                    lost.append(tid)
            for tid in lost:
                del self.tracks[tid]
            return {}

        det_centers = [(float(d.xywh[0][0]), float(d.xywh[0][1])) for d in detections]
        if not self.tracks:
            assign = {}
            for i, c in enumerate(det_centers):
                self.tracks[self.next_id] = self._new_track(c, ts)
                assign[i] = self.next_id
                self.next_id += 1
            return assign

        tids = list(self.tracks)
        tr_centers = [self.tracks[tid]["center"] for tid in tids]
        dists = [[_euclidean(tc, dc) for dc in det_centers] for tc in tr_centers]
        assign: dict[int, int] = {}
        used_t, used_d = set(), set()
        pairs = [(dists[i][j], i, j) for i in range(len(tids)) for j in range(len(det_centers))
                 if dists[i][j] < self.max_distance]
        for _, i, j in sorted(pairs, key=lambda x: x[0]):
            if i in used_t or j in used_d:
                continue
            tid = tids[i]
            tr = self.tracks[tid]
            tr["center"] = det_centers[j]
            tr["disappeared"] = 0
            tr["history"].append(det_centers[j])
            tr["frames_surface"] += 1
            tr["frames_underwater"] = 0
            # surfaced ?
            if tr["status"] == "underwater" and tr["frames_surface"] >= SURFACE_THRESHOLD:
                dur = ts - (tr["under_start_ts"] or ts)
                tr["submersion_events"].append((tr["under_start_ts"], dur))
                tr["status"] = "surface"
                tr["under_start_ts"] = None
                tr["under_duration"] = 0
                tr["danger_alert_sent"] = False
            assign[j] = tid
            used_t.add(i)
            used_d.add(j)
        # unassigned detections → new tracks
        for j, c in enumerate(det_centers):
            if j not in used_d:
                self.tracks[self.next_id] = self._new_track(c, ts)
                assign[j] = self.next_id
                self.next_id += 1
        # unassigned tracks → disappeared
        for i, tid in enumerate(tids):
            if i not in used_t:
                _mark_missing(tid)

        # danger alert pass & cleanup
        gone = []
        for tid in list(self.tracks):
            tr = self.tracks[tid]
            if tr["status"] == "underwater" and tr["under_start_ts"]:
                tr["under_duration"] = ts - tr["under_start_ts"]
                if tr["under_duration"] > DANGER_TIME_THRESHOLD and not tr["danger_alert_sent"]:
                    print(
                        f"🚨 DANGER ALERT: Person {tid} underwater for {tr['under_duration']:.1f}s!")
                    tr["danger_alert_sent"] = True
            if tr["disappeared"] > self.max_disappeared:
                if tr["status"] == "underwater" and tr["under_start_ts"]:
                    tr["submersion_events"].append(
                        (tr["under_start_ts"], ts - tr["under_start_ts"]))
                gone.append(tid)
        for tid in gone:
            del self.tracks[tid]
        return assign

    # convenience accessors
    def active_tracks(self):
        return {tid: tr for tid, tr in self.tracks.items() if tr["disappeared"] <= UNDERWATER_THRESHOLD}

    def underwater_tracks(self):
        return {tid: tr for tid, tr in self.tracks.items() if tr["status"] == "underwater"}

    def danger_tracks(self):
        now = time.time()
        return {tid: tr for tid, tr in self.tracks.items()
                if tr["status"] == "underwater" and tr["under_start_ts"]
                and now - tr["under_start_ts"] > DANGER_TIME_THRESHOLD}

@torch.inference_mode()
def detect_people(frame_bgr, processor, model):
    inputs = processor(images=frame_bgr[:, :, ::-1], return_tensors="pt").to(DEVICE)
    if DEVICE == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
    outputs = model(**inputs)
    res = processor.post_process_object_detection(
        outputs, target_sizes=[frame_bgr.shape[:2]], threshold=CONF_THRES)[0]
    people = []
    for box, lab, scr in zip(res["boxes"], res["labels"], res["scores"]):
        if lab.item() == 0:
            x0, y0, x1, y1 = box.tolist()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            people.append(BoxStub(cx, cy, x1 - x0, y1 - y0, scr.item()))
    return people


def _danger_score(tr, ts, dist_shore):
    score = int(dist_shore * 20)
    if tr["frames_underwater"] > 0:
        prog = min(tr["frames_underwater"] / UNDERWATER_THRESHOLD, 1.0)
        score += 10 + int(prog * 20)  # 10‑30
        if tr["status"] == "underwater":
            score += 20
            if tr["under_start_ts"]:
                t = ts - tr["under_start_ts"]
                score += 40 if t > DANGER_TIME_THRESHOLD else int((t / DANGER_TIME_THRESHOLD) * 40)
        excess = tr["frames_underwater"] - UNDERWATER_THRESHOLD
        if excess > 0:
            score += min(10, excess // 10)
    return min(100, score)


def _color_from_score(s):
    if s <= 20:
        r = int(144 * (s / 20.0))
        return (r, 100 + int(138 * (s / 20.0)), r)
    if s <= 40:
        ratio = (s - 20) / 20.0
        return (int(144 * (1 - ratio)), 238 + int(17 * ratio), 144 + int(111 * ratio))
    if s <= 60:
        ratio = (s - 40) / 20.0
        return (0, 255 - int(90 * ratio), 255)
    if s <= 80:
        ratio = (s - 60) / 20.0
        return (0, 165 * (1 - ratio), 255)
    # 80‑100
    ratio = (s - 80) / 20.0
    return (0, 0, 255 - int(116 * ratio))


def _dist_from_shore(x, y):
    return max(0.0, min(1.0, (MAP_H_PX - y) / MAP_H_PX))

class HomographyProcessor:
    """Drop‑in replacement that streams frames with underwater danger logic."""

    def __init__(self, video_path: str):
        # video stream
        self.cap = cv2.VideoCapture(str(Path(video_path)))
        if not self.cap.isOpened():
            raise IOError(f"Cannot open {video_path}")
        # fps for timing
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        # models
        self.water_seg = YOLO(SEG_MODEL_PATH)
        self.proc = AutoImageProcessor.from_pretrained(DFINE_MODEL_ID)
        self.dfine = DFineForObjectDetection.from_pretrained(
            DFINE_MODEL_ID,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE).eval()
        # homography
        self.DST_RECT = np.float32([[0, 0], [MAP_W_PX, 0],
                                    [MAP_W_PX, MAP_H_PX], [0, MAP_H_PX]])
        self.H: np.ndarray | None = None
        # tracking
        self.tracker = UnderwaterPersonTracker()
        self.tracker.frame_rate = self.fps
        # misc
        self.frame_idx = 0
        self.start_ts = time.time()
        self.map_base = np.full((MAP_H_PX, MAP_W_PX, 3), 80, np.uint8)

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                # rewind for endless stream
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.start_ts = time.time()
                self.frame_idx = 0
                continue
            self.frame_idx += 1
            ts = self.start_ts + self.frame_idx / self.fps

            # homography update every UPDATE_EVERY frames
            if self.frame_idx % UPDATE_EVERY == 1:
                seg_res = self.water_seg.predict(frame, imgsz=512, task="segment",
                                                 conf=0.25, verbose=False)[0]
                if seg_res.masks is not None:
                    mask_small = (seg_res.masks.data.cpu().numpy() > 0.5).any(0).astype(np.uint8)
                    mask = cv2.resize(mask_small, frame.shape[1::-1], cv2.INTER_NEAREST)
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cnt = max(cnts, key=cv2.contourArea)
                        if cv2.contourArea(cnt) > MIN_WATER_AREA_PX:
                            pts = cnt.reshape(-1, 2).astype(np.float32)
                            sums = pts.sum(1)
                            diffs = np.diff(pts, axis=1).reshape(-1)
                            src_quad = np.array([pts[np.argmin(sums)], pts[np.argmin(diffs)],
                                                 pts[np.argmax(sums)], pts[np.argmax(diffs)]])
                            H, _ = cv2.findHomography(src_quad, self.DST_RECT, cv2.RANSAC, 3.)
                            if H is not None:
                                self.H = H.copy()

            if self.H is None:
                yield cv2.resize(frame, (DISPLAY_W, DISPLAY_H), cv2.INTER_AREA)
                continue

            # detection
            persons = detect_people(frame, self.proc, self.dfine)
            assign = self.tracker.update(persons, ts)
            active = self.tracker.active_tracks()
            underwater = self.tracker.underwater_tracks()
            danger = self.tracker.danger_tracks()

            # minimap canvas
            canvas = self.map_base.copy()
            # dive points (persist)
            for tid, tr in self.tracker.tracks.items():
                if tr["dive_point"] is None:
                    continue
                p = np.array([[[tr["dive_point"][0], tr["dive_point"][1]]]], np.float32)
                x, y = cv2.perspectiveTransform(p, self.H)[0, 0]
                col = DANGER_COLOR if tid in danger else _color_from_score(tr["danger_score"])
                cv2.drawMarker(canvas, (int(x), int(y)), col, cv2.MARKER_CROSS, 10, 2)

            # active tracks → canvas & update danger score
            for tid, tr in active.items():
                cx, cy = tr["center"]
                x, y = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), self.H)[0, 0]
                if 0 <= x < MAP_W_PX and 0 <= y < MAP_H_PX:
                    dist_shore = _dist_from_shore(x, y)
                    tr["distance_shore"] = dist_shore
                    tr["danger_score"] = _danger_score(tr, ts, dist_shore)
                    colour = _color_from_score(tr["danger_score"])
                    danger_pulse = tid in danger
                    r = 6 if tr["status"] == "underwater" else 4
                    cv2.circle(canvas, (int(x), int(y)), r, colour, -1)
                    if danger_pulse:
                        pulse = 8 + int(2 * math.sin(self.frame_idx * 0.3))
                        cv2.circle(canvas, (int(x), int(y)), pulse, DANGER_COLOR, 2)
                    cv2.putText(canvas, f"{tid}({tr['danger_score']})", (int(x) + 8, int(y) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, colour, 1)
                # track history polyline
                if len(tr["history"]) > 1:
                    pts = []
                    for hx, hy in tr["history"][-15:]:
                        u, v = cv2.perspectiveTransform(np.float32([[[hx, hy]]]), self.H)[0, 0]
                        if 0 <= u < MAP_W_PX and 0 <= v < MAP_H_PX:
                            pts.append((int(u), int(v)))
                    if len(pts) > 1:
                        cv2.polylines(canvas, [np.int32(pts)], False, _color_from_score(tr["danger_score"]), 1)

            # update scores for non‑active tracks (still underwater, etc.)
            for tid, tr in self.tracker.tracks.items():
                if tid in active:
                    continue
                cx, cy = tr["center"]
                x, y = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), self.H)[0, 0]
                if 0 <= x < MAP_W_PX and 0 <= y < MAP_H_PX:
                    dist_shore = _dist_from_shore(x, y)
                    tr["distance_shore"] = dist_shore
                    tr["danger_score"] = _danger_score(tr, ts, dist_shore)

            # main visual
            vis = frame.copy()
            for di, p in enumerate(persons):
                tid = assign.get(di, -1)
                cx, cy, w, h = p.xywh[0]
                x0, y0 = int(cx - w / 2), int(cy - h / 2)
                x1, y1 = int(cx + w / 2), int(cy + h / 2)
                if tid != -1 and tid in self.tracker.tracks:
                    tr = self.tracker.tracks[tid]
                    colour = _color_from_score(tr["danger_score"])
                    thick = 3 if tid in danger else 2
                    cv2.rectangle(vis, (x0, y0), (x1, y1), colour, thick)
                    text = f"ID:{tid} ({tr['status'].upper()}) ({tr['danger_score']})"
                    if tr["status"] == "underwater" and tr["under_start_ts"]:
                        text += f" | {ts - tr['under_start_ts']:.1f}s"
                    cv2.putText(vis, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
                    # danger bar
                    bar_w = int(w * 0.8)
                    bar_x = x0 + int((w - bar_w) / 2)
                    bar_y = y1 + 35
                    fill = int(tr["danger_score"] / 100 * bar_w)
                    if fill > 0:
                        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + fill, bar_y + 6), colour, -1)

            # water polygon overlay
            src_poly = cv2.perspectiveTransform(self.DST_RECT[None, :, :], np.linalg.inv(self.H))[0].astype(int)
            cv2.polylines(vis, [src_poly], True, (0, 255, 0), 3)

            # composite UI
            vis_small = cv2.resize(vis, (DISPLAY_W, DISPLAY_H), cv2.INTER_AREA)
            map_small = cv2.resize(canvas, (MINIMAP_W, MINIMAP_H), cv2.INTER_AREA)
            vis_small[PAD:PAD + MINIMAP_H, DISPLAY_W - MINIMAP_W - PAD:DISPLAY_W - PAD] = map_small

            # status summary text
            act_c, und_c, dang_c = len(active), len(underwater), len(danger)
            # max danger score & id
            max_score = 0
            max_tid = None
            for tid, tr in self.tracker.tracks.items():
                if tr["danger_score"] > max_score:
                    max_score, max_tid = tr["danger_score"], tid
            stat_txt = f"Active:{act_c} Underwater:{und_c}"
            if max_tid is not None:
                stat_txt += f" Max:{max_score}(ID:{max_tid})"
            cv2.putText(vis_small, stat_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if dang_c > 0:
                cv2.putText(vis_small, f"DANGER:{dang_c} Max:{max_score}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, DANGER_COLOR, 2)
            # danger scale legend
            lg_x, lg_y, lg_w, lg_h = 10, DISPLAY_H - 80, 200, 20
            cv2.rectangle(vis_small, (lg_x, lg_y), (lg_x + lg_w, lg_y + lg_h), (50, 50, 50), -1)
            for i in range(lg_w):
                col = _color_from_score(i / lg_w * 100)
                cv2.line(vis_small, (lg_x + i, lg_y), (lg_x + i, lg_y + lg_h), col, 1)
            cv2.putText(vis_small, "0", (lg_x, lg_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(vis_small, "100", (lg_x + lg_w - 25, lg_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(vis_small, "Safe", (lg_x, lg_y + lg_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(vis_small, "Danger", (lg_x + lg_w - 50, lg_y + lg_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DANGER_COLOR, 1)

            yield vis_small
