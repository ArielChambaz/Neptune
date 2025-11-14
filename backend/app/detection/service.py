#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Service de détection IA - Backend
Gestionnaire centralisé des modèles
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import logging

# ===== Dépendances IA (optionnelles) =====
try:
    from transformers import AutoImageProcessor, DFineForObjectDetection
    from ultralytics import YOLO
    HAS_AI_MODELS = True
except Exception as e:
    print(f"[DetectionService] Modèles IA non disponibles: {e}")
    HAS_AI_MODELS = False

logger = logging.getLogger(__name__)


class BoxStub:
    """Stub pour les résultats de détection"""
    
    def __init__(self, cx, cy, w, h, conf):
        self.xywh = torch.tensor([[cx, cy, w, h]])
        self.conf = torch.tensor([conf])
        self.cls = torch.tensor([0])  # personne


class DetectionService:
    """Service centralisé de détection IA"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.nwsd = None  # Modèle eau
        self.dfine = None  # Modèle personnes
        self.processor = None  # Processeur D-FINE
        self.models_loaded = False
    
    def initialize(self) -> bool:
        """
        Initialise et charge les modèles
        
        Returns:
            bool: True si succès
        """
        if not HAS_AI_MODELS:
            logger.warning("Modèles IA non disponibles")
            return False
        
        try:
            self._load_water_model()
            self._load_person_model()
            self.models_loaded = True
            logger.info(f"[Detection] Modèles chargés. NWSD:{bool(self.nwsd)} D-FINE:{bool(self.dfine)} Device:{self.device}")
            return True
        except Exception as e:
            logger.error(f"[Detection] Erreur initialisation: {e}")
            return False
    
    def _load_water_model(self):
        """Charge le modèle YOLO de détection d'eau"""
        try:
            # Chercher le modèle
            candidates = [
                Path("backend/models/nwd-v2.pt"),
                Path("app/model/nwd-v2.pt"),
                Path("model/nwd-v2.pt"),
                self.config.MODELS_DIR / "nwd-v2.pt"
            ]
            
            model_path = None
            for p in candidates:
                if p.exists():
                    model_path = p
                    break
            
            if model_path:
                self.nwsd = YOLO(str(model_path))
                logger.info(f"[Detection] Modèle eau chargé: {model_path}")
            else:
                logger.warning(f"[Detection] Modèle eau non trouvé. Chemins: {candidates}")
        except Exception as e:
            logger.error(f"[Detection] Erreur chargement modèle eau: {e}")
    
    def _load_person_model(self):
        """Charge le modèle D-FINE de détection de personnes"""
        try:
            model_id = self.config.PERSON_MODEL_ID
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.dfine = DFineForObjectDetection.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device).eval()
            logger.info(f"[Detection] Modèle personnes chargé: {model_id}")
        except Exception as e:
            logger.error(f"[Detection] Erreur chargement modèle personnes: {e}")
            raise
    
    def detect_persons(self, frame: np.ndarray, conf_threshold: float = None) -> List[dict]:
        """
        Détecte les personnes dans une frame
        
        Args:
            frame: Image BGR
            conf_threshold: Seuil de confiance
        
        Returns:
            List[dict]: Détections [{x, y, w, h, confidence}, ...]
        """
        if not self.dfine:
            return []
        
        conf_threshold = conf_threshold or self.config.CONF_THRESHOLD
        
        try:
            with torch.inference_mode():
                inputs = self.processor(images=frame[:, :, ::-1], return_tensors="pt").to(self.device)
                
                if self.device == "cuda":
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
                
                outputs = self.dfine(**inputs)
                results = self.processor.post_process_object_detection(
                    outputs, 
                    target_sizes=[(frame.shape[0], frame.shape[1])], 
                    threshold=conf_threshold
                )[0]
            
            persons = []
            for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
                if label.item() == 0:  # classe personne
                    x0, y0, x1, y1 = box.tolist()
                    persons.append({
                        "x": (x0 + x1) / 2,
                        "y": (y0 + y1) / 2,
                        "width": x1 - x0,
                        "height": y1 - y0,
                        "confidence": float(score.item()),
                        "class": "person"
                    })
            
            return persons
        except Exception as e:
            logger.error(f"[Detection] Erreur détection personnes: {e}")
            return []
    
    def detect_water(self, frame: np.ndarray, conf_threshold: float = None) -> Optional[dict]:
        """
        Détecte la région d'eau dans une frame
        
        Args:
            frame: Image BGR
            conf_threshold: Seuil de confiance
        
        Returns:
            dict: Région détectée ou None
        """
        if not self.nwsd:
            return None
        
        conf_threshold = conf_threshold or self.config.CONF_THRESHOLD
        
        try:
            results = self.nwsd(frame, conf=conf_threshold)
            if results and len(results) > 0:
                r = results[0]
                if r.boxes and len(r.boxes) > 0:
                    box = r.boxes[0]
                    return {
                        "x0": float(box.xyxy[0][0]),
                        "y0": float(box.xyxy[0][1]),
                        "x1": float(box.xyxy[0][2]),
                        "y1": float(box.xyxy[0][3]),
                        "confidence": float(box.conf[0])
                    }
            return None
        except Exception as e:
            logger.error(f"[Detection] Erreur détection eau: {e}")
            return None
    
    def process_frame(self, frame: np.ndarray, detect_persons: bool = True, 
                     detect_water: bool = True) -> dict:
        """
        Traite une frame complète
        
        Args:
            frame: Image BGR
            detect_persons: Détecter les personnes
            detect_water: Détecter l'eau
        
        Returns:
            dict: Résultats combinés
        """
        result = {
            "frame_shape": frame.shape,
            "persons": [],
            "water_region": None
        }
        
        if detect_persons:
            result["persons"] = self.detect_persons(frame)
        
        if detect_water:
            result["water_region"] = self.detect_water(frame)
        
        return result
