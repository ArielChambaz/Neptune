#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application principale Neptune Backend - FastAPI
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.api import detection

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'app"""
    # Startup
    logger.info(f"Démarrage Neptune Backend API v{settings.API_VERSION}")
    logger.info(f"Device: {settings.DEVICE}")
    
    yield
    
    # Shutdown
    logger.info("Arrêt Neptune Backend API")


# Créer l'application FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="API Backend pour Neptune - Détection d'eau et personnes",
    lifespan=lifespan
)


# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Inclure les routes
app.include_router(detection.router)


# Route de base
@app.get("/")
async def root():
    """Info API"""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info"
    )
