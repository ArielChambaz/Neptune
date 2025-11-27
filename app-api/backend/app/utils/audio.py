#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Audio Utilities
- Génération et lecture d'alertes vocales
"""

import os
import time
import threading
import logging

logger = logging.getLogger(__name__)

# ===== Dépendances Audio (optionnelles) =====
try:
    from gtts import gTTS
    import pygame
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False

# Configuration par défaut
AUDIO_CONFIG = {
    'danger_message': "Alerte ! Baigneur en danger.",
    'test_message': "Test de l'alerte vocale. Système de surveillance aquatique opérationnel.",
    'language': 'fr',
    'slow_speech': True,
    'tld': 'fr',
}


def set_audio_config(config):
    """Configure les paramètres audio"""
    global AUDIO_CONFIG
    AUDIO_CONFIG = config


def generate_audio_files():
    """Génère les fichiers audio pour les alertes"""
    if not HAS_AUDIO:
        return
    
    os.makedirs("audio_alerts", exist_ok=True)
    files = {"danger": "alerte_danger.mp3", "test": "test_alerte.mp3"}
    texts = {"danger": AUDIO_CONFIG['danger_message'], "test": AUDIO_CONFIG['test_message']}
    
    for key, fname in files.items():
        path = os.path.join("audio_alerts", fname)
        if os.path.exists(path):
            continue
        try:
            gTTS(
                text=texts[key], 
                lang=AUDIO_CONFIG['language'], 
                slow=AUDIO_CONFIG['slow_speech'], 
                tld=AUDIO_CONFIG['tld']
            ).save(path)
        except Exception as e:
            logger.error(f"[Audio] Erreur génération {fname}: {e}")


def speak_alert(kind="danger"):
    """Joue une alerte vocale de manière asynchrone"""
    if not HAS_AUDIO:
        logger.info(f"[ALERTE VOCALE] {kind}")
        return
    
    def _play():
        try:
            files = {"danger": "alerte_danger.mp3", "test": "test_alerte.mp3"}
            path = os.path.join("audio_alerts", files.get(kind, "alerte_danger.mp3"))
            
            if not os.path.exists(path):
                logger.error(f"[Audio] Manquant: {path}")
                return
            
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            
            pygame.mixer.music.unload()
        except Exception as e:
            logger.error(f"[Audio] Lecture KO: {e}")
    
    threading.Thread(target=_play, daemon=True).start()


def initialize_audio():
    """Initialise le système audio"""
    if HAS_AUDIO:
        try:
            pygame.mixer.init()
            generate_audio_files()
            return True
        except Exception as e:
            logger.error(f"[Audio] init KO: {e}")
            return False
    return False
