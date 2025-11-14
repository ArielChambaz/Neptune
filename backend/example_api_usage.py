#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemples d'utilisation de l'API Neptune Backend

Ce script montre comment utiliser le client API pour:
- Vérifier la santé du backend
- Détecter personnes et eau dans des frames
- Obtenir des infos vidéo
"""

import cv2
import sys
from pathlib import Path

# Ajouter le répertoire app au path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from api_client import get_api_client


def example_health_check():
    """Exemple 1: Vérifier que le backend est accessible"""
    print("\n=== Exemple 1: Health Check ===")
    
    client = get_api_client()
    
    if client.health_check():
        info = client.get_models_info()
        print(f"✓ Backend accessible")
        print(f"  Device: {info['device']}")
        print(f"  Modèle eau: {info['water_model_loaded']}")
        print(f"  Modèle personnes: {info['person_model_loaded']}")
    else:
        print("✗ Backend non accessible")
        print("  → Vérifier que le backend est lancé:")
        print("  → cd backend && python start_backend.py")


def example_detect_frame_from_file(video_path: str):
    """Exemple 2: Détecter personnes et eau dans une frame"""
    print("\n=== Exemple 2: Détection Frame ===")
    
    client = get_api_client()
    
    # Ouvrir une vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Impossible d'ouvrir la vidéo: {video_path}")
        return
    
    # Lire une frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("✗ Impossible de lire une frame")
        return
    
    print(f"Frame: {frame.shape}")
    
    # Détecter
    result = client.detect_frame_from_file(
        frame,
        detect_persons=True,
        detect_water=True
    )
    
    if result:
        print(f"✓ Détection réussie")
        print(f"  Personnes: {len(result['persons'])}")
        for i, person in enumerate(result['persons']):
            print(f"    [{i}] pos=({person['x']:.0f}, {person['y']:.0f}) "
                  f"conf={person['confidence']:.2f}")
        
        if result['water_region']:
            wr = result['water_region']
            print(f"  Eau: détectée (conf={wr['confidence']:.2f})")
        else:
            print(f"  Eau: non détectée")
    else:
        print("✗ Erreur détection")


def example_detect_frame_base64(video_path: str):
    """Exemple 3: Détecter avec envoi en base64"""
    print("\n=== Exemple 3: Détection Base64 ===")
    
    client = get_api_client()
    
    # Ouvrir une vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Impossible d'ouvrir la vidéo: {video_path}")
        return
    
    # Lire une frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("✗ Impossible de lire une frame")
        return
    
    # Détecter en base64
    result = client.detect_frame_base64(
        frame,
        frame_id=0,
        detect_persons=True,
        detect_water=True
    )
    
    if result:
        print(f"✓ Détection base64 réussie")
        print(f"  Frame ID: {result['frame_id']}")
        print(f"  Personnes: {len(result['persons'])}")
    else:
        print("✗ Erreur détection base64")


def example_video_info(video_path: str):
    """Exemple 4: Obtenir les infos d'une vidéo"""
    print("\n=== Exemple 4: Infos Vidéo ===")
    
    client = get_api_client()
    
    info = client.get_video_info(video_path)
    
    if info:
        print(f"✓ Infos vidéo récupérées")
        print(f"  Frames totales: {info['total_frames']}")
        print(f"  FPS: {info['fps']:.1f}")
        print(f"  Durée: {info['duration']:.1f}s")
        print(f"  Résolution: {info['width']}x{info['height']}")
    else:
        print("✗ Erreur récupération infos")


def example_stream_video(video_path: str, every_n_frames: int = 10):
    """Exemple 5: Traiter une vidéo complète (échantillonnage)"""
    print(f"\n=== Exemple 5: Stream Vidéo (1/{every_n_frames} frames) ===")
    
    client = get_api_client()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Impossible d'ouvrir la vidéo: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    detection_count = 0
    
    print(f"Vidéo: {total_frames} frames @ {fps:.1f}fps")
    print(f"Traitement...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % every_n_frames == 0:
            result = client.detect_frame_from_file(frame)
            if result and len(result['persons']) > 0:
                detection_count += len(result['persons'])
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  {frame_count}/{total_frames} frames traitées...")
    
    cap.release()
    
    print(f"✓ Traitement terminé")
    print(f"  Frames: {frame_count}")
    print(f"  Personnes détectées: {detection_count}")


def main():
    """Lancer les exemples"""
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║          Exemples API Neptune Backend                  ║
    ╠════════════════════════════════════════════════════════╣
    ║ Avant de lancer, assurez-vous que:                    ║
    ║ 1. Le backend est lancé: cd backend && python main.py ║
    ║ 2. Une vidéo est disponible (voir ligne 'VIDEO_PATH') ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # À adapter selon votre vidéo
    VIDEO_PATH = "app/video/rozel-15fps-fullhd.mp4"
    
    if not Path(VIDEO_PATH).exists():
        print(f"⚠ Vidéo non trouvée: {VIDEO_PATH}")
        print("  Vous devez adapter VIDEO_PATH dans ce script")
        return
    
    # Lancer les exemples
    try:
        example_health_check()
        example_detect_frame_from_file(VIDEO_PATH)
        example_detect_frame_base64(VIDEO_PATH)
        example_video_info(VIDEO_PATH)
        example_stream_video(VIDEO_PATH, every_n_frames=30)
    
    except KeyboardInterrupt:
        print("\n\n✓ Exemples interrompus par l'utilisateur")
    except Exception as e:
        print(f"\n✗ Erreur: {e}")


if __name__ == "__main__":
    main()
