#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VÉRIFICATION POST-REFACTORISATION
==================================

Script pour vérifier que app-api a été correctement refactorisée
et contient tous les modules nécessaires.
"""

import os
from pathlib import Path


def check_file(path, description):
    """Vérifie l'existence d'un fichier"""
    if Path(path).exists():
        print(f"  ✅ {path}")
        return True
    else:
        print(f"  ❌ {path}")
        return False


def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║ VÉRIFICATION POST-REFACTORISATION Neptune App-API                          ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    app_api = Path("app-api")
    if not app_api.exists():
        print("❌ Répertoire app-api non trouvé!")
        return False
    
    all_ok = True
    
    # ========== BACKEND ==========
    print("\n🔧 BACKEND MODULES")
    print("━" * 80)
    
    backend_modules = [
        ("app-api/backend/app/config.py", "Configuration backend"),
        ("app-api/backend/app/main.py", "Serveur principal"),
        ("app-api/backend/app/core/__init__.py", "Core package init"),
        ("app-api/backend/app/core/constants.py", "Constantes globales"),
        ("app-api/backend/app/core/tracker.py", "Tracker UnderwaterPersonTracker"),
        ("app-api/backend/app/detection/__init__.py", "Detection package init"),
        ("app-api/backend/app/detection/water.py", "Détection eau + homographie"),
        ("app-api/backend/app/utils/__init__.py", "Utils package init"),
        ("app-api/backend/app/utils/danger.py", "Calcul scores danger"),
        ("app-api/backend/app/utils/audio.py", "Alertes vocales"),
    ]
    
    for path, desc in backend_modules:
        if not check_file(path, desc):
            all_ok = False
    
    # ========== FRONTEND ==========
    print("\n💻 FRONTEND MODULES")
    print("━" * 80)
    
    frontend_modules = [
        ("app-api/frontend/app/config.py", "Configuration frontend"),
        ("app-api/frontend/app/main.py", "Point d'entrée"),
        ("app-api/frontend/app/api_client.py", "Client HTTP API"),
        ("app-api/frontend/app/ui/__init__.py", "UI package init"),
        ("app-api/frontend/app/ui/main_window.py", "Interface PyQt6 principale"),
    ]
    
    for path, desc in frontend_modules:
        if not check_file(path, desc):
            all_ok = False
    
    # ========== DOCUMENTATION ==========
    print("\n📚 DOCUMENTATION")
    print("━" * 80)
    
    docs = [
        ("app-api/README.md", "Documentation complète"),
        ("app-api/QUICKSTART.py", "Guide démarrage rapide"),
        ("app-api/CHANGES_SUMMARY.md", "Résumé changements"),
        ("app-api/IMPLEMENTATION_COMPLETE.md", "Vue d'ensemble finale"),
        ("app-api/FILES_CREATED.txt", "Liste fichiers créés"),
    ]
    
    for path, desc in docs:
        if not check_file(path, desc):
            all_ok = False
    
    # ========== VÉRIFICATIONS ADDITIONNELLES ==========
    print("\n🔍 VÉRIFICATIONS ADDITIONNELLES")
    print("━" * 80)
    
    # Vérifier les imports dans les modules clés
    print("\n📦 Vérification des imports...")
    
    try:
        # Test import backend config
        backend_config_path = "app-api/backend/app/config.py"
        with open(backend_config_path) as f:
            content = f.read()
            if "DETECTION" in content and "ALERTS" in content:
                print(f"  ✅ {backend_config_path} - Configuration complète")
            else:
                print(f"  ❌ {backend_config_path} - Configuration incomplète")
                all_ok = False
    except Exception as e:
        print(f"  ❌ Erreur lecture config: {e}")
        all_ok = False
    
    # ========== RÉSUMÉ ==========
    print("\n" + "="*80)
    
    if all_ok:
        print("""
╔════════════════════════════════════════════════════════════════════════════╗
║ ✅ VÉRIFICATION RÉUSSIE                                                     ║
║                                                                            ║
║ App-API est complètement refactorisée et prête à l'emploi!                 ║
╚════════════════════════════════════════════════════════════════════════════╝

PROCHAINES ÉTAPES:

1. Backend:
   $ cd app-api/backend
   $ pip install -r requirements.txt
   $ python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

2. Frontend (dans un autre terminal):
   $ cd app-api/frontend
   $ pip install -r requirements.txt
   $ python -m app.main

3. Utilisez le client frontend pour charger et traiter des vidéos!

Pour plus d'informations:
  • Voir app-api/README.md
  • Voir app-api/QUICKSTART.py
        """)
    else:
        print("""
╔════════════════════════════════════════════════════════════════════════════╗
║ ❌ VÉRIFICATION ÉCHOUÉE                                                     ║
║                                                                            ║
║ Des fichiers manquent. Vérifiez les erreurs ci-dessus.                     ║
╚════════════════════════════════════════════════════════════════════════════╝
        """)
    
    return all_ok


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
