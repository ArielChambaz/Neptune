#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Backend API
- API FastAPI pour le traitement des frames vidéo
- Gestion de la détection aquatique et des alertes
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO
import logging
from typing import List, Dict, Tuple
import torch
import time

# Imports des modules de détection
from app.models_loader import ModelsLoader
from app.frame_processor import FrameProcessor
from app.tracking import PersonTracker
from app.alert_manager import AlertManager

# Imports des modules core
from app.config import DETECTION, ALERTS, AUDIO
from app.core.tracker import UnderwaterPersonTracker
from app.detection.water import WaterDetector
from app.utils.audio import initialize_audio, speak_alert, set_audio_config
from app.utils.danger import get_color_by_dangerosity, calculate_distance_from_shore

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation FastAPI
app = FastAPI(
    title="Neptune Backend API",
    description="API de traitement vidéo aquatique avec détection IA",
    version="2.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Instances globales =====
models_loader = None
frame_processor = None
person_tracker = None
alert_manager = None
water_detector = None
unified_tracker = None


@app.on_event("startup")
async def startup_event():
    """Initialisation des ressources au démarrage du serveur"""
    global models_loader, frame_processor, person_tracker, alert_manager, water_detector, unified_tracker
    
    try:
        logger.info("Initialisation des modèles...")
        models_loader = ModelsLoader()
        models_loader.load_all_models()
        
        frame_processor = FrameProcessor(models_loader)
        person_tracker = PersonTracker()
        alert_manager = AlertManager()
        water_detector = WaterDetector()
        
        # Initialisation du tracker unifié
        unified_tracker = UnderwaterPersonTracker(
            max_distance=DETECTION['max_distance'],
            max_disappeared=DETECTION['max_disappeared'],
            underwater_threshold=DETECTION['underwater_threshold'],
            surface_threshold=DETECTION['surface_threshold'],
            danger_threshold=ALERTS['danger_threshold']
        )
        
        # Configuration audio
        set_audio_config(AUDIO)
        initialize_audio()
        
        logger.info("Serveur Neptune prêt ✓")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Libération des ressources à l'arrêt du serveur"""
    global models_loader
    if models_loader:
        models_loader.cleanup()
    logger.info("Serveur Neptune arrêté")


# ===== Endpoints =====

@app.get("/health")
async def health_check():
    """Vérification de l'état du serveur"""
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/detect-frame")
async def detect_frame(frame_file: UploadFile = File(...)):
    """
    Traite une frame unique et retourne les détections
    
    Args:
        frame_file: Image encodée en bytes
        
    Returns:
        {
            "detections": [
                {"class": "person", "conf": 0.95, "bbox": [x1, y1, x2, y2], "water_state": "surface"}
            ],
            "water_zone": {"points": [...], "area": 0},
            "alerts": [{"type": "danger", "person_id": 1, "duration": 5}]
        }
    """
    try:
        # Lecture du fichier
        contents = await frame_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Traitement de la frame
        detections, water_zone = frame_processor.process_frame(frame)
        
        # Suivi des personnes
        tracked_persons = person_tracker.update(detections)
        
        # Gestion des alertes
        alerts = alert_manager.check_alerts(tracked_persons, detections)
        
        return JSONResponse({
            "detections": [
                {
                    "id": d.get("id"),
                    "class": d.get("class"),
                    "confidence": float(d.get("conf", 0)),
                    "bbox": d.get("bbox"),
                    "water_state": d.get("water_state", "unknown"),
                    "tracking_frames": d.get("tracking_frames", 0)
                }
                for d in tracked_persons
            ],
            "water_zone": {
                "points": water_zone.get("points", []),
                "area": int(water_zone.get("area", 0))
            },
            "alerts": alerts,
            "frame_shape": frame.shape
        })
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-batch")
async def detect_batch(frames: List[UploadFile] = File(...)):
    """
    Traite un batch de frames
    
    Returns une liste de résultats de détection
    """
    results = []
    try:
        for frame_file in frames:
            contents = await frame_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                detections, water_zone = frame_processor.process_frame(frame)
                tracked_persons = person_tracker.update(detections)
                alerts = alert_manager.check_alerts(tracked_persons, detections)
                
                results.append({
                    "detections": tracked_persons,
                    "water_zone": water_zone,
                    "alerts": alerts
                })
        
        return JSONResponse({"batch_results": results, "count": len(results)})
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement du batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Retourne la configuration actuelle du backend"""
    return {
        "detection": DETECTION,
        "alerts": ALERTS,
        "audio": AUDIO,
        "models": {
            "human_detection": "D-FINE-xlarge",
            "water_detection": "nwd-v2.pt"
        }
    }


@app.post("/reset-tracking")
async def reset_tracking():
    """Réinitialise le suivi des personnes"""
    global person_tracker
    person_tracker.reset()
    return {"status": "tracking_reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
