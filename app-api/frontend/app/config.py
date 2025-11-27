#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration pour le frontend Neptune
"""

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
}

# === Configuration de l'interface ===
UI = {
    'width': 1400,
    'height': 900,
    'control_panel_width': 350,
    'video_panel_min_width': 800,
    'video_panel_min_height': 600,
}

# === Configuration API ===
API = {
    'base_url': 'http://localhost:8000',
    'timeout': 10.0,
    'health_check_interval': 2000,  # millisecondes
}

# === Configuration de la détection (récupérée du serveur) ===
DETECTION = {
    'conf_threshold': 0.7,
    'max_distance': 100,
    'max_disappeared': 300,
    'underwater_threshold': 5,
    'surface_threshold': 3,
}

# === Configuration des alertes ===
ALERTS = {
    'danger_threshold': 5,
    'alert_duration': 8.0,
    'popup_duration': 7.0,
}
