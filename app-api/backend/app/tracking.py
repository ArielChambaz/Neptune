#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Système de suivi des personnes (Tracking)
- Associe les détections aux IDs de personnes persistants
- Gère l'état aquatique dans le temps
"""

from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class PersonTracker:
    """Suit les personnes à travers les frames"""
    
    def __init__(self, max_distance: float = 100, max_disappeared: int = 300):
        """
        Initialise le tracker
        
        Args:
            max_distance: Distance maximale pour associer une détection à un track
            max_disappeared: Nombre de frames avant suppression d'un track
        """
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        
        self.next_id = 0
        self.tracks = {}  # {id: {"bbox": [...], "water_state": "...", "frames": N}}
        self.disappeared = defaultdict(int)
        self.water_state_frames = defaultdict(int)  # {id: frames_underwater}
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Mise à jour du tracking avec les nouvelles détections
        
        Args:
            detections: Liste des détections de la frame
            
        Returns:
            Liste des personnes suivies avec IDs
        """
        tracked_persons = []
        
        if len(detections) == 0:
            # Pas de détections: incrémenter les disappeared
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
            
            return []
        
        # Associer les détections aux tracks existants
        matched_pairs, unmatched_detections, unmatched_tracks = self._match_detections(detections)
        
        # Mettre à jour les tracks appariés
        for track_id, det_idx in matched_pairs:
            detection = detections[det_idx]
            self.tracks[track_id] = detection.copy()
            self.disappeared[track_id] = 0
            
            # Compter les frames sous l'eau
            if detection.get("water_state") == "underwater":
                self.water_state_frames[track_id] += 1
            
            det_copy = detection.copy()
            det_copy["id"] = track_id
            det_copy["tracking_frames"] = self.water_state_frames[track_id]
            tracked_persons.append(det_copy)
        
        # Créer de nouveaux tracks pour les détections non appariées
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            self.tracks[self.next_id] = detection.copy()
            self.disappeared[self.next_id] = 0
            self.water_state_frames[self.next_id] = 0
            
            det_copy = detection.copy()
            det_copy["id"] = self.next_id
            det_copy["tracking_frames"] = 0
            tracked_persons.append(det_copy)
            
            self.next_id += 1
        
        # Gérer les tracks perdus
        for track_id in unmatched_tracks:
            self.disappeared[track_id] += 1
            
            if self.disappeared[track_id] > self.max_disappeared:
                del self.tracks[track_id]
                del self.disappeared[track_id]
                if track_id in self.water_state_frames:
                    del self.water_state_frames[track_id]
        
        return tracked_persons
    
    def _match_detections(self, detections: List[Dict]) -> Tuple[List[Tuple], List[int], List[int]]:
        """
        Apparie les détections aux tracks existants
        
        Returns:
            (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Calculer les distances
        distance_matrix = np.zeros((len(self.tracks), len(detections)))
        track_ids = list(self.tracks.keys())
        
        for i, track_id in enumerate(track_ids):
            track_bbox = self.tracks[track_id]["bbox"]
            track_center = (
                (track_bbox[0] + track_bbox[2]) / 2,
                (track_bbox[1] + track_bbox[3]) / 2
            )
            
            for j, det in enumerate(detections):
                det_bbox = det["bbox"]
                det_center = (
                    (det_bbox[0] + det_bbox[2]) / 2,
                    (det_bbox[1] + det_bbox[3]) / 2
                )
                
                distance = np.sqrt(
                    (track_center[0] - det_center[0])**2 +
                    (track_center[1] - det_center[1])**2
                )
                distance_matrix[i, j] = distance
        
        # Appariement greedy simple
        matched_pairs = []
        used_detections = set()
        used_tracks = set()
        
        # Trier par distance
        indices = np.argsort(distance_matrix, axis=None)
        
        for idx in indices:
            track_idx, det_idx = np.unravel_index(idx, distance_matrix.shape)
            
            if (track_idx in used_tracks or 
                det_idx in used_detections or 
                distance_matrix[track_idx, det_idx] > self.max_distance):
                continue
            
            matched_pairs.append((track_ids[track_idx], det_idx))
            used_tracks.add(track_idx)
            used_detections.add(det_idx)
        
        unmatched_detections = [i for i in range(len(detections)) if i not in used_detections]
        unmatched_tracks = [track_ids[i] for i in range(len(track_ids)) if i not in used_tracks]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def reset(self):
        """Réinitialise tous les tracks"""
        self.tracks.clear()
        self.disappeared.clear()
        self.water_state_frames.clear()
        self.next_id = 0
        logger.info("Tracking réinitialisé")
