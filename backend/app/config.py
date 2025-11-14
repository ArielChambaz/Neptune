#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration du backend Neptune
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Configuration globale de l'application"""
    
    # API
    API_TITLE: str = "Neptune Backend API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    
    # Detection Models
    WATER_MODEL_PATH: Optional[Path] = None
    PERSON_MODEL_ID: str = "ustc-community/dfine-xlarge-obj2coco"
    
    # Detection Parameters
    CONF_THRESHOLD: float = 0.5
    UNDERWATER_THRESHOLD: int = 10
    DANGER_THRESHOLD: float = 5.0
    
    # Device
    DEVICE: str = "cuda"  # ou "cpu"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Créer les répertoires s'ils n'existent pas
settings.UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
settings.MODELS_DIR.mkdir(exist_ok=True, parents=True)
