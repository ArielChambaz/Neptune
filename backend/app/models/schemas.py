#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schémas Pydantic pour les requêtes/réponses API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class DetectionBox(BaseModel):
    """Représentation d'une box de détection"""
    x: float = Field(..., description="Coordonnée X du centre")
    y: float = Field(..., description="Coordonnée Y du centre")
    width: float = Field(..., description="Largeur du box")
    height: float = Field(..., description="Hauteur du box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Score de confiance")
    class_name: str = Field(default="person", description="Classe détectée")


class DetectionResult(BaseModel):
    """Résultat de détection pour une frame"""
    frame_id: int
    persons: List[DetectionBox] = Field(default_factory=list, description="Personnes détectées")
    water_region: Optional[Dict[str, Any]] = Field(None, description="Région d'eau détectée")
    timestamp: float = Field(..., description="Timestamp du frame")
    

class ProcessFrameRequest(BaseModel):
    """Requête de traitement d'une frame"""
    frame_data: str = Field(..., description="Frame en base64 ou chemin fichier")
    detect_persons: bool = True
    detect_water: bool = True
    frame_id: int = 0
    

class ProcessVideoRequest(BaseModel):
    """Requête de traitement d'une vidéo"""
    video_path: str = Field(..., description="Chemin vers la vidéo")
    detect_persons: bool = True
    detect_water: bool = True
    sample_rate: int = Field(1, ge=1, description="Traiter 1 frame sur N")
    

class VideoStats(BaseModel):
    """Statistiques d'une vidéo"""
    total_frames: int
    fps: float
    duration: float
    width: int
    height: int
    

class HealthResponse(BaseModel):
    """Réponse de health check"""
    status: str
    models_loaded: bool
    device: str
    version: str


class ErrorResponse(BaseModel):
    """Réponse d'erreur"""
    error: str
    detail: Optional[str] = None
    code: int = 500
