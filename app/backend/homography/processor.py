from __future__ import annotations
from pathlib import Path
import time, math, cv2, numpy as np, torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, DFineForObjectDetection

# ---------- CONSTANTES ----------
MAP_W_PX, MAP_H_PX   = 400, 200
UPDATE_EVERY         = 300            # homographie eau
CONF_THRES           = .25
MIN_WATER_AREA_PX    = 5_000
INFER_EVERY          = 3              # détecter les personnes 1 frame / 3
MAX_DISTANCE_PX      = 100            # tracking
MAX_DISAPPEARED      = 30
MINIMAP_W, MINIMAP_H = 320, 160
PAD                  = 12
MAX_LONG_SIDE        = 1280

# ---------- TRACKER ----------
class PersonTracker:
    def __init__(self, max_dist=MAX_DISTANCE_PX, max_miss=MAX_DISAPPEARED):
        self.next_id, self.tracks = 1, {}
        self.max_dist, self.max_miss = max_dist, max_miss

    def _dist(self, p, q):  # euclidean
        return math.hypot(p[0]-q[0], p[1]-q[1])

    def update(self, detections):
        """detections = [(cx,cy)]"""
        # --- matching -----------------------------------------------------------------
        assigned, used_tr, used_det = {}, set(), set()
        if self.tracks:
            dists = [((tid, t['center']), (i, c), self._dist(t['center'], c))
                     for i, c in enumerate(detections)
                     for tid, t in self.tracks.items()]
            for (tid,_), (i,_), d in sorted(dists, key=lambda x: x[2]):
                if d < self.max_dist and tid not in used_tr and i not in used_det:
                    self.tracks[tid]['center'] = detections[i]
                    self.tracks[tid]['miss'] = 0
                    self.tracks[tid]['hist'].append(detections[i])
                    assigned[i] = tid
                    used_tr.add(tid); used_det.add(i)

        # --- nouveaux tracks ----------------------------------------------------------
        for i, c in enumerate(detections):
            if i not in assigned:
                self.tracks[self.next_id] = {'center': c, 'miss': 0, 'hist':[c]}
                assigned[i] = self.next_id
                self.next_id += 1

        # --- disparition --------------------------------------------------------------
        for tid in list(self.tracks):
            if tid not in used_tr:
                self.tracks[tid]['miss'] += 1
                if self.tracks[tid]['miss'] > self.max_miss:
                    del self.tracks[tid]

        return assigned

    def active(self):
        return {tid:t for tid,t in self.tracks.items() if t['miss']==0}

# ---------- HOMOGRAPHY PROCESSOR ----------
class HomographyProcessor:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(str(Path(video_path)))
        if not self.cap.isOpened():
            raise IOError(f"Cannot open {video_path}")
        self.device   = "cuda" if torch.cuda.is_available() else "cpu"
        self.water    = YOLO("water-detection/model-v2/nwd-v2.pt")
        self.proc     = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-obj2coco")
        self.det      = DFineForObjectDetection.from_pretrained(
                            "ustc-community/dfine-xlarge-obj2coco",
                            torch_dtype=torch.float16 if self.device=="cuda" else torch.float32
                        ).to(self.device).eval()

        self.DST_RECT = np.float32([[0,0],[MAP_W_PX,0],[MAP_W_PX,MAP_H_PX],[0,MAP_H_PX]])
        self.H, self.water_bbox, self.fidx = None, None, 0
        self.people_cache = []
        self.tracker = PersonTracker()

        # couleurs ID
        self.colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
                       (0,255,255),(128,0,255),(255,128,0),(128,255,0),(255,0,128)]

    # ---------- utils ----------
    def _color(self, tid:int): return self.colors[tid % len(self.colors)]

    @torch.inference_mode()
    def _detect_people(self, frame):
        inp = self.proc(images=frame[:,:,::-1], return_tensors="pt").to(self.device)
        out = self.det(**inp)
        res = self.proc.post_process_object_detection(out,
                target_sizes=[frame.shape[:2]], threshold=CONF_THRES)[0]
        people=[]
        for box, lab, scr in zip(res["boxes"], res["labels"], res["scores"]):
            if lab.item()==0:
                x0,y0,x1,y1 = box.tolist()
                cx,cy = (x0+x1)/2,(y0+y1)/2
                people.append(((cx,cy,x1-x0,y1-y0),scr.item()))
        return people

    # ---------- main ----------
    def _process_frame(self, f):
        self.fidx+=1

        # -- homographie eau -------------------------------------------------
        if self.fidx % UPDATE_EVERY == 1:
            seg = self.water.predict(f, imgsz=512, task="segment",
                                     conf=.25, verbose=False)[0]
            if seg.masks is not None:
                mask_small = (seg.masks.data.cpu().numpy()>.5).any(0).astype(np.uint8)
                mask = cv2.resize(mask_small, f.shape[1::-1], cv2.INTER_NEAREST)
                cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    cnt = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > MIN_WATER_AREA_PX:
                        pts = cnt.reshape(-1,2).astype(np.float32)
                        sums, diffs = pts.sum(1), np.diff(pts,axis=1).reshape(-1)
                        src = np.array([pts[np.argmin(sums)], pts[np.argmin(diffs)],
                                        pts[np.argmax(sums)], pts[np.argmax(diffs)]])
                        H,_ = cv2.findHomography(src, self.DST_RECT, cv2.RANSAC,3.)
                        if H is not None: self.H = H
                        self.water_bbox = cv2.boundingRect(cnt)

        if self.H is None: return f

        # -- détection personnes --------------------------------------------
        if self.fidx % INFER_EVERY == 0:
            self.people_cache = self._detect_people(f)
        people = self.people_cache

        # -- tracking --------------------------------------------------------
        det_centers = [(cx,cy) for (cx,cy,_,_),_ in people]
        assign = self.tracker.update(det_centers)
        active = self.tracker.active()

        # -- minimap ---------------------------------------------------------
        mini = np.full((MAP_H_PX, MAP_W_PX, 3), 80, np.uint8)
        for d_idx, tid in assign.items():
            cx, cy, _, _ = people[d_idx][0]
            proj = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), self.H)
            x, y = proj[0, 0]
            if 0<=x<MAP_W_PX and 0<=y<MAP_H_PX:
                col = self._color(tid)
                cv2.circle(mini, (int(x),int(y)),4,col,-1)
                cv2.putText(mini,str(tid),(int(x)+6,int(y)-6),
                            cv2.FONT_HERSHEY_SIMPLEX,.4,col,1)

        for tid,tr in active.items():
            if len(tr['hist'])>1:
                hist_pts=[]
                for hx,hy in tr['hist'][-10:]:
                    proj = cv2.perspectiveTransform(np.float32([[[hx, hy]]]), self.H)
                    x, y = proj[0, 0]
                    if 0<=x<MAP_W_PX and 0<=y<MAP_H_PX:
                        hist_pts.append((int(x),int(y)))
                if len(hist_pts)>1:
                    cv2.polylines(mini,[np.int32(hist_pts)],False,self._color(tid),1)

        # -- dessin sur frame -----------------------------------------------
        vis = f.copy()
        for d_idx,(box,conf) in enumerate(people):
            tid = assign.get(d_idx,-1)
            cx,cy,w,h = box
            x0,y0,x1,y1 = int(cx-w/2),int(cy-h/2),int(cx+w/2),int(cy+h/2)
            col = self._color(tid) if tid!=-1 else (0,255,0)
            cv2.rectangle(vis,(x0,y0),(x1,y1),col,2)
            if tid!=-1:
                cv2.putText(vis,f"ID:{tid}",(x0,y0-10),cv2.FONT_HERSHEY_SIMPLEX,.6,col,2)

        if self.water_bbox:
            x,y,w,h = self.water_bbox
            cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,255),2)

        # -- resize + overlay minimap ---------------------------------------
        h,w = vis.shape[:2]
        if max(w,h)>MAX_LONG_SIDE:
            s = MAX_LONG_SIDE/max(w,h)
            vis = cv2.resize(vis,(int(w*s),int(h*s)),cv2.INTER_AREA)

        mini_small = cv2.resize(mini,(MINIMAP_W,MINIMAP_H))
        vis[PAD:PAD+MINIMAP_H, vis.shape[1]-MINIMAP_W-PAD:vis.shape[1]-PAD] = mini_small
        return vis

    # ---------- generator appelé par Flask ----------
    def frames(self):
        while True:
            ok,frm = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                continue
            yield self._process_frame(frm)
