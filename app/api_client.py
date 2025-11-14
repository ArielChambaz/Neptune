#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client API Neptune - Pour communiquer avec le backend
Utilisé par le frontend PyQt6
"""

import requests
import cv2
import numpy as np
import base64
import io
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NeptuneAPIClient:
    """Client pour l'API Neptune Backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api/v1"
        self.session = requests.Session()
        self._models_loaded = False
    
    def health_check(self) -> bool:
        """
        Vérifier que le backend répond
        
        Returns:
            bool: True si backend accessible et modèles prêts
        """
        try:
            response = self.session.get(f"{self.api_base}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self._models_loaded = data.get("models_loaded", False)
                return data.get("status") == "healthy"
            return False
        except Exception as e:
            logger.error(f"[APIClient] Health check failed: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Vérifier si l'API est prête (backend et modèles)"""
        return self.health_check() and self._models_loaded
    
    def detect_frame_from_file(
        self, 
        frame: np.ndarray,
        detect_persons: bool = True,
        detect_water: bool = True,
        conf_threshold: float = None
    ) -> Optional[Dict[str, Any]]:
        """
        Envoyer une frame pour détection (upload fichier)
        
        Args:
            frame: Image numpy BGR
            detect_persons: Détecter personnes
            detect_water: Détecter eau
            conf_threshold: Seuil confiance
        
        Returns:
            dict ou None: Résultats {frame_id, persons, water_region, timestamp}
        """
        try:
            # Encoder frame en JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
            
            params = {
                'detect_persons': detect_persons,
                'detect_water': detect_water
            }
            if conf_threshold is not None:
                params['conf_threshold'] = conf_threshold
            
            response = self.session.post(
                f"{self.api_base}/detect/frame",
                files=files,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"[APIClient] Detect failed: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"[APIClient] Error detecting frame: {e}")
            return None
    
    def detect_frame_from_base64(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        detect_persons: bool = True,
        detect_water: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Envoyer une frame en base64 pour détection
        Utile pour streaming depuis vidéo en mémoire
        
        Args:
            frame: Image numpy BGR
            frame_id: ID du frame
            detect_persons: Détecter personnes
            detect_water: Détecter eau
        
        Returns:
            dict ou None: Résultats
        """
        try:
            # Encoder frame en JPEG puis base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                "frame_data": frame_b64,
                "detect_persons": detect_persons,
                "detect_water": detect_water,
                "frame_id": frame_id
            }
            
            response = self.session.post(
                f"{self.api_base}/detect/frame/base64",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"[APIClient] Detect base64 failed: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"[APIClient] Error detecting base64 frame: {e}")
            return None
    
    def get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Obtenir infos vidéo (durée, fps, résolution)
        
        Args:
            video_path: Chemin vidéo
        
        Returns:
            dict ou None: {total_frames, fps, duration, width, height}
        """
        try:
            params = {'video_path': video_path}
            response = self.session.get(
                f"{self.api_base}/video/info",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"[APIClient] Video info failed: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"[APIClient] Error getting video info: {e}")
            return None
    
    def get_models_info(self) -> Optional[Dict[str, Any]]:
        """
        Obtenir infos sur les modèles chargés
        
        Returns:
            dict ou None: {device, water_model_loaded, person_model_loaded}
        """
        try:
            response = self.session.get(
                f"{self.api_base}/models/info",
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
        
        except Exception as e:
            logger.error(f"[APIClient] Error getting models info: {e}")
            return None
    
    def close(self):
        """Fermer la session"""
        self.session.close()


# Singleton global
_client: Optional[NeptuneAPIClient] = None


def get_api_client(base_url: str = "http://localhost:8000") -> NeptuneAPIClient:
    """
    Obtenir une instance du client API
    
    Args:
        base_url: URL du backend
    
    Returns:
        NeptuneAPIClient: Client API unique
    """
    global _client
    if _client is None:
        _client = NeptuneAPIClient(base_url)
    return _client
