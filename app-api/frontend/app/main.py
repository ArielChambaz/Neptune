#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Frontend
- Interface PyQt6 pour affichage des détections
- Communication avec l'API backend
"""

import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import QApplication

# Configuration Qt
os.environ.setdefault("QT_NO_XDG_DESKTOP_PORTAL", "1")
os.environ.setdefault("QT_STYLE_OVERRIDE", "Fusion")

from .ui.main_window import NeptuneFrontendWindow


def main():
    """Point d'entrée principal du frontend"""
    app = QApplication(sys.argv)
    app.setApplicationName("Neptune Frontend")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Neptune Team")
    
    print("Neptune Frontend v2.0 – API Client")
    print("Connexion à http://localhost:8000")
    
    # Création et affichage de la fenêtre principale
    window = NeptuneFrontendWindow()
    window.show()
    
    # Lancement de la boucle d'événements
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

