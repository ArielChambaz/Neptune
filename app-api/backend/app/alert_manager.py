#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestionnaire des alertes
- Détecte les situations de danger (personnes sous l'eau)
- Génère les notifications d'alerte
"""

from typing import List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AlertManager:
    """Gère la détection et la génération des alertes"""
    
    def __init__(self, danger_threshold: int = 5, alert_duration: float = 8.0):
        """
        Initialise le gestionnaire d'alertes
        
        Args:
            danger_threshold: Nombre de frames sous l'eau avant alerte (à ~30fps = ~5 secondes)
            alert_duration: Durée d'affichage de l'alerte (secondes)
        """
        self.danger_threshold = danger_threshold
        self.alert_duration = alert_duration
        self.active_alerts = {}  # {person_id: {"timestamp": ..., "type": "danger"}}
    
    def check_alerts(self, tracked_persons: List[Dict], 
                    detections: List[Dict] = None) -> List[Dict]:
        """
        Vérifie les conditions d'alerte
        
        Args:
            tracked_persons: Personnes suivies avec IDs
            detections: Détections brutes (optionnel)
            
        Returns:
            Liste des alertes actives
        """
        alerts = []
        current_time = datetime.now()
        
        # Vérifier les personnes en danger
        for person in tracked_persons:
            person_id = person.get("id")
            tracking_frames = person.get("tracking_frames", 0)
            water_state = person.get("water_state", "unknown")
            
            # Déterminer si la personne est en danger
            is_danger = (
                water_state == "underwater" and 
                tracking_frames >= self.danger_threshold
            )
            
            if is_danger:
                # Créer ou mettre à jour l'alerte
                if person_id not in self.active_alerts:
                    self.active_alerts[person_id] = {
                        "timestamp": current_time,
                        "type": "danger",
                        "person_id": person_id,
                        "duration_seconds": tracking_frames / 30  # Approximation à 30fps
                    }
                    logger.warning(f"🚨 ALERTE DANGER: Personne {person_id} sous l'eau depuis {tracking_frames} frames")
                
                alerts.append({
                    "type": "danger",
                    "person_id": person_id,
                    "duration_frames": tracking_frames,
                    "timestamp": current_time.isoformat()
                })
            
            # Nettoyer les anciennes alertes
            elif person_id in self.active_alerts:
                time_diff = (current_time - self.active_alerts[person_id]["timestamp"]).total_seconds()
                if time_diff > self.alert_duration:
                    del self.active_alerts[person_id]
        
        # Ajouter les alertes restantes
        for person_id, alert_info in self.active_alerts.items():
            time_diff = (current_time - alert_info["timestamp"]).total_seconds()
            if time_diff <= self.alert_duration:
                alerts.append({
                    "type": alert_info["type"],
                    "person_id": person_id,
                    "elapsed_seconds": time_diff,
                    "timestamp": current_time.isoformat()
                })
        
        return alerts
    
    def clear_alerts(self):
        """Efface toutes les alertes actives"""
        self.active_alerts.clear()
        logger.info("Alertes effacées")
