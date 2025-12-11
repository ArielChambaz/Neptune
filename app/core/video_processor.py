#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Video Processor (Client WebSocket)
- Connects to the API via WebSocket to process video frames.
- Sends frames to the server and receives detection results.
"""

import cv2
import time
import numpy as np
import threading
import json
import base64
import asyncio
import queue
from PyQt6.QtCore import QThread, pyqtSignal, QObject
import websockets

from config_pyqt6 import API, DETECTION, ALERTS
from utils.audio import speak_alert
from utils.alerts import AlertPopup

# --- Helper functions (adapted from streamlit_app.py) ---
def encode_frame_jpeg(frame, quality=75):
    """Encode frame as JPEG and return base64 string"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    return base64.b64encode(buffer).decode('utf-8')

def get_color_by_dangerosity(score: int) -> tuple:
    """Calculate BGR color based on dangerosity score (0-100)"""
    if score <= 20:
        r = int(144 * (score / 20.0))
        g = int(100 + 138 * (score / 20.0))
        b = r
        return (b, g, r)

    if score <= 40:
        ratio = (score - 20) / 20.0
        return (int(144 * (1 - ratio)), int(238 + 17 * ratio), int(144 + 111 * ratio))

    if score <= 60:
        ratio = (score - 40) / 20.0
        return (0, int(255 - 90 * ratio), 255)

    if score <= 80:
        ratio = (score - 60) / 20.0
        return (0, int(165 * (1 - ratio)), 255)

    ratio = (score - 80) / 20.0
    return (0, 0, int(255 - 116 * ratio))

# --- Async Worker for WebSocket ---
class WebSocketWorker:
    def __init__(self, ws_url, session_id, conf_threshold, underwater_threshold, danger_threshold, jpeg_quality, fps_target):
        self.ws_url = ws_url
        self.session_id = session_id
        self.conf_threshold = conf_threshold
        self.underwater_threshold = underwater_threshold
        self.danger_threshold = danger_threshold
        self.jpeg_quality = jpeg_quality
        self.fps_target = fps_target

        self.running = False
        self.websocket = None
        self.loop = None
        self.send_queue = asyncio.Queue(maxsize=5)
        self.result_queue = queue.Queue() # Thread-safe queue for results to be consumed by VideoProcessor

    async def connect(self):
        """Connect to WebSocket and initialize session"""
        try:
            self.websocket = await websockets.connect(self.ws_url)

            # Send initialization message
            init_msg = {
                'type': 'init',
                'session_id': self.session_id,
                'conf_threshold': self.conf_threshold,
                'underwater_threshold': self.underwater_threshold,
                'danger_threshold': self.danger_threshold,
                'jpeg_quality': self.jpeg_quality,
                'fps_target': self.fps_target
            }
            await self.websocket.send(json.dumps(init_msg))

            # Wait for init success
            response = await self.websocket.recv()
            result = json.loads(response)

            if result.get('type') == 'init_success':
                return True
            return False
        except Exception as e:
            print(f"[WebSocket] Connection error: {e}")
            return False

    async def _send_loop(self):
        while self.running:
            try:
                frame_data = await self.send_queue.get()
                await self.websocket.send(json.dumps(frame_data))
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[WebSocket] Send error: {e}")

    async def _recv_loop(self):
        while self.running:
            try:
                response = await self.websocket.recv()
                result = json.loads(response)
                self.result_queue.put(result)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[WebSocket] Receive error: {e}")
                break

    async def run(self):
        self.running = True
        if not await self.connect():
             print("[WebSocket] Failed to connect")
             self.running = False
             return

        send_task = asyncio.create_task(self._send_loop())
        recv_task = asyncio.create_task(self._recv_loop())

        try:
            await asyncio.gather(send_task, recv_task)
        except asyncio.CancelledError:
            pass
        finally:
            if self.websocket:
                await self.websocket.close()

    def add_frame(self, frame_data):
        try:
            self.loop.call_soon_threadsafe(self.send_queue.put_nowait, frame_data)
        except asyncio.QueueFull:
            pass # Skip frame if queue is full
        except Exception as e:
            print(f"[WebSocket] Add frame error: {e}")


class VideoProcessor(QThread):
    """Thread principal de traitement vidéo (Client WebSocket)"""
    
    frameReady = pyqtSignal(np.ndarray)
    statsReady = pyqtSignal(dict)
    alertTriggered = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # État de la vidéo
        self.video_path = None
        self.is_running = False
        self.is_paused = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        
        # Mutex pour la protection thread-safe
        self._lock = threading.Lock()
        
        # Configuration
        self.conf_threshold = DETECTION['conf_threshold']
        self.underwater_threshold = DETECTION['underwater_threshold']
        self.danger_threshold = ALERTS['danger_threshold']
        
        # WebSocket / API settings
        self.api_url = API['base_url']
        self.ws_url = f"{API['ws_url']}/stream/realtime"
        self.jpeg_quality = API['jpeg_quality']
        self.fps_target = API['fps_target']
        self.skip_frames = API['skip_frames']

        # Interface
        self.show_water_detection = False
        self.alert_popup = AlertPopup(duration=ALERTS['popup_duration'])
        
        # Tracking spoken alerts
        self.spoken_alerts = set()
        self.last_alert_time = 0
        self.alert_cooldown = 10.0 # seconds
        
        self.ws_worker = None
        self.ws_thread = None
        self.session_id = None
        self.loop = None

    def load_models(self):
        """
        Mock for compatibility with UI.
        The actual models are on the server.
        """
        return True
    
    def load_video(self, path):
        """
        Charge une vidéo
        
        Args:
            path: Chemin vers la vidéo
        
        Returns:
            bool: True si le chargement réussit
        """
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        
        if not self.cap.isOpened():
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        return True
    
    def recalculate_water_detection(self) -> bool:
        """
        Recalcule la détection d'eau.
        Note: Since logic is on server, we might need a specific API endpoint or
        msg type to trigger this on server if supported.
        For now, this might be limited or we rely on the continuous server analysis.
        """
        # TODO: Implement API call to reset/recalculate water zone on server if supported
        print("[VideoProcessor] Recalculate water detection requested (Server side logic)")
        return True
    
    def run(self):
        """Boucle principale de traitement vidéo"""
        if not self.video_path or not hasattr(self, "cap"):
            return
        
        self.is_running = True
        self.session_id = f"stream_{int(time.time() * 1000)}"
        self.spoken_alerts.clear()

        # Start WebSocket Worker in a separate thread with its own event loop
        self.loop = asyncio.new_event_loop()
        self.ws_worker = WebSocketWorker(
            self.ws_url,
            self.session_id,
            self.conf_threshold,
            self.underwater_threshold,
            self.danger_threshold,
            self.jpeg_quality,
            self.fps_target
        )
        self.ws_worker.loop = self.loop

        def run_ws():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.ws_worker.run())

        self.ws_thread = threading.Thread(target=run_ws, daemon=True)
        self.ws_thread.start()

        # Give some time for connection
        time.sleep(1)

        frame_idx = 0
        frame_counter = 0
        start_time = time.time()
        
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            ok, frame = self.cap.read()
            if not ok:
                # Retour au début de la vidéo
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_counter += 1
            if frame_counter % self.skip_frames != 0:
                continue

            frame_idx += 1
            self.current_frame = frame_idx
            
            # Send frame to WS
            timestamp = time.time()
            frame_b64 = encode_frame_jpeg(frame, self.jpeg_quality)
            
            message = {
                'type': 'frame',
                'session_id': self.session_id,
                'frame_id': frame_idx,
                'data': frame_b64,
                'timestamp': timestamp
            }
            
            if self.ws_worker.running:
                self.ws_worker.add_frame(message)
            
            # Check for results
            try:
                # Non-blocking check for results, or wait briefly
                while not self.ws_worker.result_queue.empty():
                    result = self.ws_worker.result_queue.get_nowait()
                    if result.get('type') == 'result':
                        self._process_result(frame, result)
            except queue.Empty:
                pass
            
            # Small delay to throttle if needed, though loop speed is mainly determined by capture and processing
            time.sleep(0.01)

    def _process_result(self, frame, result):
        """Process results from API and update UI"""
        detections = result.get('detections', [])
        water_zone = result.get('water_zone')
        stats_api = result.get('stats', {})
        alerts = result.get('alerts', [])
        
        current_time = time.time()
        
        # Handle Alerts
        for alert in alerts:
            msg = alert['message']
            self.alertTriggered.emit(msg)
            
            # Check for danger alerts and speak if not already spoken recently
            if "DANGER" in msg or "danger" in msg.lower():
                track_id = alert.get('track_id')
                # Create a unique key for this alert (e.g. track_id)
                # If track_id is not available, use the message itself
                alert_key = track_id if track_id is not None else msg

                if alert_key not in self.spoken_alerts or (current_time - self.last_alert_time > self.alert_cooldown):
                    speak_alert("danger")
                    self.alert_popup.add_alert(msg, duration=8.0)
                    self.spoken_alerts.add(alert_key)
                    self.last_alert_time = current_time

        # Clean up old alerts from spoken_alerts if needed (optional)
        # For now, simplistic approach: if no danger detected for a while, we could reset.
        # But the cooldown prevents spamming.

        # Prepare stats for UI
        max_score = 0
        max_score_id = None
        underwater_count = 0
        danger_count = 0
        active_count = len(detections)

        for det in detections:
            score = det['dangerosity_score']
            if score > max_score:
                max_score = score
                max_score_id = det['track_id']
            if det['status'] == 'underwater':
                underwater_count += 1
            if det['status'] == 'danger':
                danger_count += 1
        
        stats = {
            'active': active_count,
            'underwater': underwater_count,
            'danger': danger_count,
            'max_score': max_score,
            'max_score_id': max_score_id,
            # Add API specific stats
            'api_fps': stats_api.get('fps', 0),
            'api_proc_time': stats_api.get('processing_time_ms', 0)
        }
        self.statsReady.emit(stats)

        # Draw on frame
        vis = self._draw_detections(frame, detections, water_zone)
        self.frameReady.emit(vis)

    def _draw_detections(self, frame, detections, water_zone):
        """Draw detections on frame (matching app renderer style)"""
        vis = frame.copy()
        
        # Draw water zone
        if self.show_water_detection and water_zone and water_zone.get('detected') and water_zone.get('polygon'):
            pts = np.array(water_zone['polygon'], dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
            # Overlay
            overlay = np.zeros_like(vis)
            cv2.fillPoly(overlay, [pts], (255, 100, 0))
            vis = cv2.addWeighted(vis, 1.0, overlay, 0.2, 0)

        # Draw detections
        for det in detections:
            bbox = det['bbox']
            cx = bbox['center_x']
            cy = bbox['center_y']
            w = bbox['width']
            h = bbox['height']
            
            x0 = int(cx - w/2)
            y0 = int(cy - h/2)
            x1 = int(cx + w/2)
            y1 = int(cy + h/2)
            
            score = det['dangerosity_score']
            color = get_color_by_dangerosity(score)
            status = det['status']
            is_danger = (status == 'danger')
            
            # BBox
            thickness = 4 if is_danger else 2
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, thickness)
            
            # Label
            track_id = det['track_id']
            if status == 'underwater':
                label = f"ID:{track_id} (UNDERWATER) - Score:{score} | {det['underwater_duration']:.1f}s"
            else:
                label = f"ID:{track_id} - Score:{score}"

            sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis, (x0, y0-35), (x0+sz[0]+10, y0-5), (0, 0, 0), -1)
            cv2.putText(vis, label, (x0+5, y0-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Confidence
            conf = det['confidence']
            cv2.putText(vis, f"Conf:{conf:.2f}", (x0, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Alert Popup HUD
        self.alert_popup.update()
        alerts = self.alert_popup.get_active_alerts()
        if alerts:
            base_y = vis.shape[0] - 200
            height = min(len(alerts), 3) * 35 + 20
            cv2.rectangle(vis, (20, base_y-10), (600, base_y+height), (0, 0, 0), -1)
            cv2.rectangle(vis, (20, base_y-10), (600, base_y+height), (255, 0, 0), 2)
            cv2.putText(vis, "ALERTES ACTIVES:", (30, base_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for i, a in enumerate(alerts[-3:]):
                col = (0, 0, 255) if "DANGER" in a else (255, 165, 0)
                cv2.putText(vis, f"• {a}", (40, base_y+45 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        return vis

    # Contrôles du thread
    def set_video_path(self, path):
        """Définit le chemin de la vidéo"""
        self.video_path = path
    
    def pause(self):
        """Met en pause ou reprend la lecture"""
        self.is_paused = not self.is_paused
    
    def stop(self):
        """Arrête le traitement vidéo de manière sécurisée"""
        print("[VideoProcessor] Arrêt demandé...")
        self.is_running = False
        
        # Stop WS worker
        if self.ws_worker:
            self.ws_worker.running = False
            # Cancel all tasks in the loop
            if self.loop and self.loop.is_running():
                for task in asyncio.all_tasks(self.loop):
                    task.cancel()

        # Attendre que le thread se termine proprement
        if self.isRunning():
            self.wait(2000)
            
        # Libérer les ressources vidéo
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                with self._lock:
                    self.cap.release()
                    self.cap = None
                print("[VideoProcessor] Ressources vidéo libérées")
            except Exception as e:
                print(f"[VideoProcessor] Erreur libération: {e}")
        
        print("[VideoProcessor] Arrêt terminé")
