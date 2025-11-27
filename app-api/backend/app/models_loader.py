#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestionnaire du chargement des modèles ML
- Charge D-FINE pour la détection humaine
- Charge YOLO pour la détection d'eau
"""

import os
# Désactiver CUDA avant de charger torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from pathlib import Path
from ultralytics import YOLO
import logging

# D-FINE imports
try:
    from transformers import AutoImageProcessor, DFineForObjectDetection
    HAS_DFINE = True
except ImportError:
    HAS_DFINE = False
    logger_import = logging.getLogger(__name__)
    logger_import.warning("D-FINE non disponible - installez: pip install transformers")

logger = logging.getLogger(__name__)


class BoxStub:
    """Structure pour les résultats de détection D-FINE"""
    def __init__(self, x1, y1, x2, y2, conf):
        self.bbox = [int(x1), int(y1), int(x2), int(y2)]
        self.conf = float(conf)
        self.cls = 0  # classe personne


class ModelsLoader:
    """Charge et gère les modèles d'IA (D-FINE + YOLO)"""
    
    def __init__(self, model_dir: Path = None):
        """
        Initialise le gestionnaire des modèles
        
        Args:
            model_dir: Répertoire contenant les modèles (par défaut: ./models)
        """
        self.model_dir = model_dir or Path(__file__).parent.parent / "models"
        # Forcer CPU - CUDA pas disponible en WSL
        self.device = "cpu"
        
        # D-FINE pour détection humaine
        self.dfine_model = None
        self.dfine_processor = None
        
        # YOLOv8 fallback pour détection humaine
        self.yolo_human_model = None
        
        # YOLO pour détection d'eau
        self.water_detection_model = None
        
        logger.info(f"Device: {self.device}")
    
    def load_all_models(self):
        """Charge tous les modèles nécessaires"""
        logger.info("Chargement des modèles...")
        
        # Modèle de détection humaine (D-FINE avec fallback YOLOv8)
        self._load_dfine_model()
        
        if not self.dfine_model:
            logger.warning("D-FINE indisponible, utilisation de YOLOv8 en fallback")
            self._load_yolo_human_model()
        
        # Modèle de détection d'eau (YOLO)
        self._load_water_model()
        
        dfine_status = '✓' if self.dfine_model else '✗'
        yolo_status = '✓' if self.yolo_human_model else '✗'
        water_status = '✓' if self.water_detection_model else '✗'
        logger.info(f"D-FINE: {dfine_status} | YOLOv8 (fallback): {yolo_status} | YOLO eau: {water_status}")
    
    def _load_dfine_model(self):
        """Charge le modèle D-FINE pour la détection humaine"""
        if not HAS_DFINE:
            logger.error("D-FINE non disponible - installez: pip install transformers")
            return
        
        try:
            model_id = "ustc-community/dfine-xlarge-obj2coco"
            logger.info(f"Chargement D-FINE: {model_id}")
            
            self.dfine_processor = AutoImageProcessor.from_pretrained(model_id)
            self.dfine_model = DFineForObjectDetection.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device).eval()
            
            logger.info("D-FINE chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur chargement D-FINE: {e}")
            logger.warning("Tentative fallback sur YOLOv8...")
            self.dfine_model = None
            self.dfine_processor = None
    
    def _load_yolo_human_model(self):
        """Charge YOLOv8 en fallback pour la détection humaine"""
        try:
            logger.info("Chargement YOLOv8 pour détection humaine...")
            self.yolo_human_model = YOLO("yolov8m.pt")
            self.yolo_human_model.to(self.device)
            logger.info("YOLOv8 détection humaine chargé")
        except Exception as e:
            logger.error(f"Erreur chargement YOLOv8: {e}")
            self.yolo_human_model = None
    
    def _load_water_model(self):
        """Charge le modèle YOLO pour la détection d'eau"""
        water_model_path = self.model_dir / "nwd-v2.pt"
        if water_model_path.exists():
            try:
                logger.info(f"Chargement YOLO eau: {water_model_path}")
                self.water_detection_model = YOLO(str(water_model_path))
                self.water_detection_model.to(self.device)
                logger.info("Modèle eau chargé avec succès")
            except Exception as e:
                logger.error(f"Erreur chargement modèle eau: {e}")
                self.water_detection_model = None
        else:
            logger.warning(f"Modèle eau non trouvé: {water_model_path}")
    
    @torch.inference_mode()
    def detect_humans(self, frame, conf: float = 0.7):
        """
        Détecte les humains dans une frame avec D-FINE ou YOLOv8
        
        Args:
            frame: Frame BGR
            conf: Seuil de confiance
        
        Returns: [{"class": "person", "conf": 0.95, "bbox": [x1, y1, x2, y2]}]
        """
        # Essayer D-FINE d'abord
        if self.dfine_model is not None and self.dfine_processor is not None:
            try:
                # Convertir BGR en RGB pour D-FINE
                frame_rgb = frame[:, :, ::-1]
                
                # Traiter l'image
                inputs = self.dfine_processor(images=frame_rgb, return_tensors="pt").to(self.device)
                
                # Inference
                if self.device == "cuda":
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
                
                outputs = self.dfine_model(**inputs)
                
                # Post-traitement
                results = self.dfine_processor.post_process_object_detection(
                    outputs, 
                    target_sizes=[(frame.shape[0], frame.shape[1])],
                    threshold=conf
                )[0]
                
                # Extraire les personnes (classe 0)
                detections = []
                for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
                    if label.item() == 0:  # classe personne
                        x1, y1, x2, y2 = box.tolist()
                        detections.append({
                            "class": "person",
                            "conf": float(score.item()),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })
                
                return detections
            
            except Exception as e:
                logger.error(f"Erreur détection D-FINE: {e}")
                logger.warning("Passage au fallback YOLOv8...")
        
        # Fallback sur YOLOv8
        if self.yolo_human_model is not None:
            try:
                results = self.yolo_human_model(frame, conf=conf, device=self.device)
                detections = []
                
                for result in results:
                    if result.boxes is not None:
                        for idx in range(len(result.boxes)):
                            box = result.boxes.xyxy[idx]
                            conf_score = result.boxes.conf[idx]
                            cls = result.boxes.cls[idx]
                            
                            # YOLOv8 classe 0 = personne
                            if int(cls.item()) == 0:
                                x1, y1, x2, y2 = box.tolist()
                                detections.append({
                                    "class": "person",
                                    "conf": float(conf_score.item()),
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                                })
                
                return detections
            except Exception as e:
                logger.error(f"Erreur détection YOLOv8: {e}")
        
        logger.error("Aucun modèle de détection disponible!")
        return []
    
    def detect_water(self, frame, conf: float = 0.7):
        """
        Détecte la zone d'eau dans une frame avec YOLO
        
        Args:
            frame: Frame BGR
            conf: Seuil de confiance
        
        Returns: {"points": [[x1,y1], [x2,y2], ...], "area": 12345}
        """
        if self.water_detection_model is None:
            return {"points": [], "area": 0}
        
        try:
            results = self.water_detection_model(frame, conf=conf, device=self.device)
            
            # Retourner les masques ou contours de la zone d'eau
            water_data = {"points": [], "area": 0}
            
            for result in results:
                if result.masks is not None:
                    import cv2
                    for mask in result.masks:
                        # Convertir le masque en contours
                        mask_array = mask.cpu().numpy().astype("uint8") * 255
                        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                            if cv2.contourArea(contour) > 1000:  # Filtrer les petites zones
                                points = contour.squeeze().tolist()
                                if isinstance(points[0], list):
                                    water_data["points"].extend(points)
                                    water_data["area"] += int(cv2.contourArea(contour))
            
            return water_data
        
        except Exception as e:
            logger.error(f"Erreur détection eau: {e}")
            return {"points": [], "area": 0}
    
    def cleanup(self):
        """Libère les ressources des modèles"""
        logger.info("Libération des modèles")
        self.dfine_model = None
        self.dfine_processor = None
        self.yolo_human_model = None
        self.water_detection_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
