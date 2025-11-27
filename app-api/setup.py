#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de configuration - Crée les répertoires nécessaires
et copie les modèles depuis l'ancienne app vers la nouvelle structure
"""

import shutil
from pathlib import Path

def setup_app_api():
    """Configure la structure app-api"""
    
    # Chemins
    app_api_root = Path(__file__).parent
    old_app = app_api_root.parent / "app"
    
    backend_models_dir = app_api_root / "backend" / "models"
    old_models_dir = old_app / "model"
    
    print("📋 Configuration de l'architecture app-api")
    print("=" * 50)
    
    # Créer le répertoire models s'il n'existe pas
    backend_models_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Dossier models créé: {backend_models_dir}")
    
    # Copier les modèles si disponibles
    if old_models_dir.exists():
        print(f"\n📦 Copie des modèles de {old_models_dir}")
        for model_file in old_models_dir.glob("*.pt"):
            try:
                dest = backend_models_dir / model_file.name
                shutil.copy2(model_file, dest)
                print(f"  ✓ {model_file.name}")
            except Exception as e:
                print(f"  ❌ Erreur lors de la copie: {e}")
    else:
        print(f"\n⚠️  Dossier ancien modèle non trouvé: {old_models_dir}")
        print("  Les modèles doivent être placés manuellement dans: {backend_models_dir}")
    
    # Afficher la structure créée
    print("\n📁 Structure créée:")
    print(f"app-api/")
    print(f"├── backend/")
    print(f"│   ├── app/")
    print(f"│   │   ├── main.py (API FastAPI)")
    print(f"│   │   ├── models_loader.py")
    print(f"│   │   ├── frame_processor.py")
    print(f"│   │   ├── tracking.py")
    print(f"│   │   └── alert_manager.py")
    print(f"│   ├── models/ (à remplir avec les fichiers .pt)")
    print(f"│   └── requirements.txt")
    print(f"├── frontend/")
    print(f"│   ├── app/")
    print(f"│   │   ├── main.py (Frontend PyQt6)")
    print(f"│   │   ├── api_client.py")
    print(f"│   │   └── config.py")
    print(f"│   ├── ui/")
    print(f"│   │   ├── main_window.py")
    print(f"│   │   ├── video_display.py")
    print(f"│   │   └── alert_display.py")
    print(f"│   └── requirements.txt")
    print(f"├── start-backend.sh")
    print(f"├── start-frontend.sh")
    print(f"└── README.md")
    
    print("\n✅ Configuration terminée!")
    print("\n🚀 Prochaines étapes:")
    print("1. cd app-api")
    print("2. pip install -r backend/requirements.txt")
    print("3. python -m backend.app.main  # Lancer le backend")
    print("   (dans un autre terminal)")
    print("4. pip install -r frontend/requirements.txt")
    print("5. python -m frontend.app.main  # Lancer le frontend")


if __name__ == "__main__":
    setup_app_api()
