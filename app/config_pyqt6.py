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
# Updated to match the new STYLESHEET in app/ui/styles.py
# Format: (R, G, B) for use with OpenCV or other libs requiring tuples
COLORS = {
    'bg_dark': (18, 18, 18),    # Matches #121212
    'bg_light': (30, 30, 30),   # Matches #1E1E1E
    'primary': (0, 122, 255),   # Matches #007AFF (Blue)
    'success': (52, 199, 89),   # Matches #34C759 (Green)
    'warning': (255, 149, 0),   # Matches #FF9500 (Orange)
    'danger': (255, 59, 48),    # Matches #FF3B30 (Red)
    'text_white': (255, 255, 255),
    'text_gray': (160, 160, 160),
    'border': (58, 58, 58),
    'water_zone': (0, 212, 255), # Matches #00D4FF (Cyan)
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

# === Configuration de la minimap ===
MINIMAP = {
    'width': 200,           # Largeur de la minimap en pixels
    'height': 150,          # Hauteur de la minimap en pixels
    'margin': 15,           # Marge depuis le bord de la vidéo
    'bg_color': (40, 40, 40),  # Couleur de fond (BGR)
    'border_color': (100, 100, 100),  # Couleur de bordure (BGR)
    'border_thickness': 2,
    'opacity': 0.85,        # Opacité de la minimap (0-1)
    'trail_length': 15,     # Nombre de positions dans la trace
}
