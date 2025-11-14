#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routes API pour les opérations de traitement vidéo
"""

import logging
import base64
import io
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from typing import Optional

from app.models.schemas import (
    DetectionResult, ProcessFrameRequest, HealthResponse,
    ErrorResponse, VideoStats
)
from app.detection.service import DetectionService
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["detection"])

# Service de détection global
detection_service: Optional[DetectionService] = None


def get_detection_service() -> DetectionService:
    """Accès au service de détection"""
    global detection_service
    if detection_service is None:
        detection_service = DetectionService(settings)
        detection_service.initialize()
    return detection_service


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Vérifier l'état du backend et des modèles
    
    Returns:
        HealthResponse: État du service
    """
    service = get_detection_service()
    return HealthResponse(
        status="healthy",
        models_loaded=service.models_loaded,
        device=service.device,
        version=settings.API_VERSION
    )


@router.post("/detect/frame", response_model=DetectionResult)
async def detect_frame(
    file: UploadFile = File(...),
    detect_persons: bool = Query(True),
    detect_water: bool = Query(True),
    conf_threshold: float = Query(None)
):
    """
    Détecter les personnes et l'eau dans une frame
    
    Args:
        file: Image à traiter (JPG, PNG)
        detect_persons: Activer détection personnes
        detect_water: Activer détection eau
        conf_threshold: Seuil de confiance (optionnel)
    
    Returns:
        DetectionResult: Résultats de détection
    """
    try:
        service = get_detection_service()
        if not service.models_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Lire l'image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Traiter
        result_data = service.process_frame(
            frame, 
            detect_persons=detect_persons, 
            detect_water=detect_water
        )
        
        return DetectionResult(
            frame_id=0,
            persons=result_data["persons"],
            water_region=result_data["water_region"],
            timestamp=0.0
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Erreur traitement frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/frame/base64", response_model=DetectionResult)
async def detect_frame_base64(
    request: ProcessFrameRequest,
    detect_persons: bool = Query(True),
    detect_water: bool = Query(True)
):
    """
    Détecter à partir d'une frame en base64
    
    Args:
        request: Données frame en base64
        detect_persons: Activer détection personnes
        detect_water: Activer détection eau
    
    Returns:
        DetectionResult: Résultats de détection
    """
    try:
        service = get_detection_service()
        if not service.models_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Décoder base64
        img_data = base64.b64decode(request.frame_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Traiter
        result_data = service.process_frame(
            frame,
            detect_persons=detect_persons,
            detect_water=detect_water
        )
        
        return DetectionResult(
            frame_id=request.frame_id,
            persons=result_data["persons"],
            water_region=result_data["water_region"],
            timestamp=0.0
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Erreur traitement base64: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/video/info")
async def get_video_info(video_path: str = Query(...)):
    """
    Obtenir les infos d'une vidéo (durée, fps, résolution)
    
    Args:
        video_path: Chemin vers la vidéo
    
    Returns:
        VideoStats: Informations vidéo
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video")
        
        stats = VideoStats(
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
            duration=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        cap.release()
        return stats
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Erreur lecture vidéo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/info")
async def get_models_info():
    """
    Obtenir les informations sur les modèles chargés
    
    Returns:
        dict: Infos modèles
    """
    service = get_detection_service()
    return {
        "device": service.device,
        "water_model_loaded": bool(service.nwsd),
        "person_model_loaded": bool(service.dfine),
        "models_loaded": service.models_loaded
    }
