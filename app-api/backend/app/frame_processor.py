#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processeur de frames
- Traite les frames individuelles pour la détection
- Gère la détection d'eau et la localisation des personnes dans/hors de l'eau
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FrameProcessor:
    """Traite les frames vidéo pour la détection"""
    
    def __init__(self, models_loader):
        """
        Initialise le processeur de frames
        
        Args:
            models_loader: Instance ModelsLoader avec les modèles chargés
        """
        self.models_loader = models_loader
        self.water_zone = None
        self.homography_matrix = None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], Dict]:
        """
        Traite une frame complète
        
        Args:
            frame: Image (BGR format OpenCV)
            
        Returns:
            (detections, water_zone_info)
        """
        # Détection des humains
        human_detections = self.models_loader.detect_humans(frame)
        
        # Détection de la zone d'eau
        water_zone = self.models_loader.detect_water(frame)
        self.water_zone = water_zone
        
        # Enrichissement des détections avec l'état aquatique
        enriched_detections = self._enrich_detections(
            frame, 
            human_detections, 
            water_zone
        )
        
        return enriched_detections, water_zone
    
    def _enrich_detections(self, frame: np.ndarray, detections: List[Dict], 
                           water_zone: Dict) -> List[Dict]:
        """
        Enrichit les détections avec l'état aquatique (surface/underwater)
        
        Args:
            frame: Image
            detections: Liste de détections brutes
            water_zone: Zone d'eau détectée
            
        Returns:
            Détections enrichies avec water_state
        """
        enriched = []
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Déterminer l'état aquatique
            water_state = self._determine_water_state(bbox_center, water_zone)
            
            det_copy = det.copy()
            det_copy["water_state"] = water_state
            det_copy["center"] = list(bbox_center)
            
            enriched.append(det_copy)
        
        return enriched
    
    def _determine_water_state(self, point: Tuple[int, int], water_zone: Dict) -> str:
        """
        Détermine si un point est dans la zone d'eau
        
        Args:
            point: Coordonnées (x, y)
            water_zone: Données de la zone d'eau
            
        Returns:
            "surface", "underwater" ou "unknown"
        """
        if not water_zone or not water_zone.get("points"):
            return "unknown"
        
        # Créer un contour à partir des points de la zone d'eau
        try:
            points = np.array(water_zone["points"], dtype=np.int32)
            if len(points) < 3:
                return "unknown"
            
            # Vérifier si le point est dans le contour
            result = cv2.pointPolygonTest(points, point, False)
            
            if result > 0:
                return "underwater"
            elif result == 0:
                return "surface"
            else:
                return "surface"
        except Exception as e:
            logger.error(f"Erreur dans determine_water_state: {e}")
            return "unknown"
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       water_zone: Dict = None) -> np.ndarray:
        """
        Dessine les détections sur la frame
        
        Args:
            frame: Image
            detections: Détections à dessiner
            water_zone: Zone d'eau à dessiner
            
        Returns:
            Frame avec les détections dessinées
        """
        output = frame.copy()
        
        # Dessiner la zone d'eau
        if water_zone and water_zone.get("points"):
            try:
                points = np.array(water_zone["points"], dtype=np.int32)
                cv2.polylines(output, [points], True, (0, 255, 0), 2)
                cv2.fillPoly(output, [points], (0, 255, 0), alpha=0.1)
            except:
                pass
        
        # Dessiner les détections
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det.get("conf", 0)
            water_state = det.get("water_state", "unknown")
            
            # Couleur selon l'état aquatique
            if water_state == "underwater":
                color = (255, 0, 0)  # Bleu en BGR
            elif water_state == "surface":
                color = (0, 255, 0)  # Vert
            else:
                color = (0, 255, 255)  # Jaune
            
            # Dessiner la bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Étiquette
            label = f"{det['class']} {water_state} {conf:.2f}"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output
