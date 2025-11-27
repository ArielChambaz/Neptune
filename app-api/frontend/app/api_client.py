#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client API pour communiquer avec le backend Neptune
"""

import httpx
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class NeptuneAPIClient:
    """Client pour communiquer avec l'API backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialise le client API
        
        Args:
            base_url: URL de base de l'API (ex: http://localhost:8000)
        """
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
        self.is_connected = False
    
    def health_check(self) -> bool:
        """Vérifie que le serveur est accessible"""
        try:
            response = self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                self.is_connected = True
                data = response.json()
                logger.info(f"Serveur ok - Device: {data.get('device', 'unknown')}")
                return True
        except Exception as e:
            logger.error(f"Impossible de se connecter au serveur: {e}")
            self.is_connected = False
        
        return False
    
    def detect_frame(self, frame_data: bytes) -> Optional[Dict]:
        """
        Envoie une frame au backend pour détection
        
        Args:
            frame_data: Données de l'image encodée (bytes)
            
        Returns:
            Réponse JSON avec détections et alertes
        """
        try:
            files = {"frame_file": ("frame.jpg", frame_data, "image/jpeg")}
            response = self.client.post(
                f"{self.base_url}/detect-frame",
                files=files
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Erreur API: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de la frame: {e}")
            return None
    
    def get_config(self) -> Optional[Dict]:
        """Récupère la configuration du backend"""
        try:
            response = self.client.get(f"{self.base_url}/config")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la config: {e}")
        
        return None
    
    def reset_tracking(self) -> bool:
        """Réinitialise le suivi des personnes"""
        try:
            response = self.client.post(f"{self.base_url}/reset-tracking")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Erreur lors de la réinitialisation: {e}")
            return False
    
    def close(self):
        """Ferme la connexion au serveur"""
        self.client.close()
