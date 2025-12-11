"""
Configuration pour l'application Neptune PyQt6
"""

import os

# === Configuration API ===
API = {
    'base_url': os.getenv("API_BASE_URL", "http://localhost:8000/api/v1"),
    'ws_url': os.getenv("WS_BASE_URL", "ws://localhost:8000/api/v1"),
    'fps_target': 15,
    'jpeg_quality': 75,
    'skip_frames': 2,
}

# === Configuration des couleurs ===
COLORS = {
    'bg_dark': (45, 45, 45),
    'bg_light': (70, 70, 70),
    'primary': (0, 80, 243),
    'success': (0, 255, 0),
    'warning': (0, 255, 255),
    'danger': (0, 0, 255),
    'text_white': (255, 255, 255),
    'text_gray': (200, 200, 200),
    'border': (100, 100, 100),
    'water_zone': (0, 255, 0),
}

# === Configuration de la détection ===
DETECTION = {
    'conf_threshold': 0.7,      # Seuil de confiance pour la détection
    'underwater_threshold': 5,   # Frames pour considérer une personne sous l'eau
}

# === Configuration des alertes ===
ALERTS = {
    'danger_threshold': 5.0,       # Seuil de danger (secondes sous l'eau)
    'alert_duration': 8.0,       # Durée d'affichage des alertes (secondes)
    'popup_duration': 7.0,       # Durée du popup d'alerte
}

# === Configuration audio ===
AUDIO = {
    'danger_message': "Alerte ! Baigneur en danger.",
    'test_message': "Test de l'alerte vocale. Système de surveillance aquatique opérationnel.",
    'language': 'fr',
    'slow_speech': True,
    'tld': 'fr',
}

# === Configuration de l'interface ===
UI = {
    'width': 1400,
    'height': 900,
    'control_panel_width': 350,
    'video_panel_min_width': 800,
    'video_panel_min_height': 600,
}
