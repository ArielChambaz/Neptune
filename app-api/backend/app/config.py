#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration du Backend Neptune
"""

# === Configuration de la détection ===
DETECTION = {
    'conf_threshold': 0.7,      # Seuil de confiance pour la détection
    'max_distance': 100,         # Distance maximale pour l'association des tracks
    'max_disappeared': 300,      # Frames avant suppression d'un track  
    'underwater_threshold': 5,   # Frames pour considérer une personne sous l'eau
    'surface_threshold': 3,      # Frames pour considérer une personne en surface
}

# === Configuration des alertes ===
ALERTS = {
    'danger_threshold': 5,       # Seuil de danger (secondes sous l'eau)
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
