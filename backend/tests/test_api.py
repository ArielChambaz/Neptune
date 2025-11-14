#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests unitaires pour l'API Neptune Backend

Exécution:
    pytest backend/tests/test_api.py -v
"""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import cv2
import base64

# Ajouter backend au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app
from app.config import settings


@pytest.fixture
def client():
    """Client de test FastAPI"""
    return TestClient(app)


@pytest.fixture
def sample_frame():
    """Créer une frame de test"""
    # Frame blanche 100x100
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


class TestHealth:
    """Tests du health check"""
    
    def test_health_check(self, client):
        """Vérifier que le health check répond"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "device" in data
    
    def test_root_endpoint(self, client):
        """Vérifier l'endpoint racine"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"


class TestDetection:
    """Tests de détection"""
    
    def test_detect_frame_upload(self, client, sample_frame):
        """Test détection avec upload"""
        # Encoder frame en JPEG
        _, buffer = cv2.imencode('.jpg', sample_frame)
        
        response = client.post(
            "/api/v1/detect/frame",
            files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
            params={
                "detect_persons": True,
                "detect_water": True
            }
        )
        
        # Accepter 200 ou 503 (modèles non chargés)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "frame_id" in data
            assert "persons" in data
            assert "water_region" in data
            assert isinstance(data["persons"], list)
    
    def test_detect_frame_base64(self, client, sample_frame):
        """Test détection avec base64"""
        # Encoder frame en base64
        _, buffer = cv2.imencode('.jpg', sample_frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        response = client.post(
            "/api/v1/detect/frame/base64",
            json={
                "frame_data": frame_b64,
                "detect_persons": True,
                "detect_water": True,
                "frame_id": 0
            }
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data["frame_id"] == 0
            assert "persons" in data
    
    def test_detect_invalid_image(self, client):
        """Test avec image invalide"""
        response = client.post(
            "/api/v1/detect/frame",
            files={"file": ("frame.jpg", b"invalid", "image/jpeg")}
        )
        assert response.status_code == 400


class TestVideo:
    """Tests vidéo"""
    
    def test_get_models_info(self, client):
        """Obtenir infos modèles"""
        response = client.get("/api/v1/models/info")
        assert response.status_code == 200
        data = response.json()
        assert "device" in data
        assert "water_model_loaded" in data
        assert "person_model_loaded" in data
        assert "models_loaded" in data


class TestErrors:
    """Tests de gestion d'erreurs"""
    
    def test_missing_file_in_detect(self, client):
        """Test détection sans fichier"""
        response = client.post("/api/v1/detect/frame")
        assert response.status_code == 422  # Validation error
    
    def test_invalid_base64(self, client):
        """Test avec base64 invalide"""
        response = client.post(
            "/api/v1/detect/frame/base64",
            json={
                "frame_data": "invalid!@#$",
                "detect_persons": True,
                "detect_water": True
            }
        )
        assert response.status_code in [400, 500]


class TestPydantic:
    """Tests des schémas Pydantic"""
    
    def test_detection_box_schema(self):
        """Test le schéma DetectionBox"""
        from app.models.schemas import DetectionBox
        
        box = DetectionBox(
            x=100, y=200, width=50, height=75,
            confidence=0.95, class_name="person"
        )
        assert box.x == 100
        assert box.confidence == 0.95
    
    def test_detection_result_schema(self):
        """Test le schéma DetectionResult"""
        from app.models.schemas import DetectionResult, DetectionBox
        
        result = DetectionResult(
            frame_id=0,
            persons=[
                DetectionBox(x=100, y=100, width=50, height=50, 
                           confidence=0.9, class_name="person")
            ],
            timestamp=0.0
        )
        assert result.frame_id == 0
        assert len(result.persons) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
