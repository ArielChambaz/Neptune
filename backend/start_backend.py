#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de démarrage du Backend Neptune

Usage:
    python start_backend.py
    python start_backend.py --host 0.0.0.0 --port 8000 --reload
"""

import sys
import argparse
import uvicorn
from pathlib import Path

# Ajouter le répertoire backend au path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.config import settings


def main():
    parser = argparse.ArgumentParser(description="Démarrer Neptune Backend API")
    parser.add_argument("--host", default=settings.API_HOST, help="Host pour l'API")
    parser.add_argument("--port", type=int, default=settings.API_PORT, help="Port pour l'API")
    parser.add_argument("--reload", action="store_true", default=settings.API_RELOAD, 
                       help="Activer le hot reload")
    parser.add_argument("--no-reload", action="store_true", help="Désactiver le reload")
    parser.add_argument("--device", default=settings.DEVICE, help="Device (cuda ou cpu)")
    parser.add_argument("--log-level", default="info", help="Niveau de log")
    
    args = parser.parse_args()
    
    # Déterminer reload
    reload = args.reload and not args.no_reload
    
    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║         Neptune Backend API - Démarrage              ║
    ╠══════════════════════════════════════════════════════╣
    ║ Serveur: http://{args.host}:{args.port}
    ║ Version: {settings.API_VERSION}
    ║ Device: {args.device}
    ║ Reload: {reload}
    ║ Logs: {args.log_level}
    ║
    ║ Swagger UI: http://{args.host}:{args.port}/docs
    ║ Redoc: http://{args.host}:{args.port}/redoc
    ╚══════════════════════════════════════════════════════╝
    """)
    
    # Démarrer le serveur
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=reload,
        log_level=args.log_level.lower()
    )


if __name__ == "__main__":
    main()
