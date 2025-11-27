#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Affichage vidéo avec bounding boxes
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt


class VideoDisplay(QWidget):
    """Widget pour afficher la vidéo avec les détections"""
    
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout(self)
        
        # Label pour l'affichage de l'image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background: #000; border: 1px solid #555;")
        self.image_label.setMinimumSize(800, 600)
        
        layout.addWidget(self.image_label)
        
        self.current_frame = None
        self.current_detections = []
    
    def update_frame(self, frame: np.ndarray, detections: list, water_zone: dict = None):
        """
        Met à jour l'affichage avec une nouvelle frame
        
        Args:
            frame: Image (BGR format OpenCV)
            detections: Liste des détections
            water_zone: Zone d'eau détectée
        """
        self.current_frame = frame.copy()
        self.current_detections = detections
        
        # Dessiner les détections
        output = self._draw_detections(frame, detections, water_zone)
        
        # Convertir en QPixmap
        pixmap = self._cv2_to_qpixmap(output)
        
        # Mettre à l'échelle si nécessaire
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        
        if label_width > 0 and label_height > 0:
            pixmap = pixmap.scaledToWidth(
                min(label_width, pixmap.width()),
                Qt.TransformationMode.SmoothTransformation
            )
        
        self.image_label.setPixmap(pixmap)
    
    def _draw_detections(self, frame: np.ndarray, detections: list, 
                         water_zone: dict = None) -> np.ndarray:
        """Dessine les bounding boxes sur la frame"""
        output = frame.copy()
        
        # Dessiner la zone d'eau
        if water_zone and water_zone.get('points'):
            try:
                points = np.array(water_zone['points'], dtype=np.int32)
                if len(points) >= 3:
                    cv2.polylines(output, [points], True, (0, 255, 0), 2)
                    cv2.fillPoly(output, [points], (0, 255, 0), alpha=0.1)
            except:
                pass
        
        # Dessiner les détections
        for det in detections:
            bbox = det.get('bbox', [])
            if not bbox or len(bbox) < 4:
                continue
            
            x1, y1, x2, y2 = bbox
            conf = det.get('confidence', 0)
            person_id = det.get('id', '?')
            water_state = det.get('water_state', 'unknown')
            
            # Couleur selon l'état aquatique
            if water_state == 'underwater':
                color = (255, 0, 0)  # Bleu en BGR
                text_color = (0, 0, 255)  # Couleur texte
            elif water_state == 'surface':
                color = (0, 255, 0)  # Vert
                text_color = (0, 255, 0)
            else:
                color = (0, 255, 255)  # Jaune
                text_color = (0, 255, 255)
            
            # Dessiner la bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Étiquette
            label = f"ID:{person_id} {water_state.upper()} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Fond pour le texte
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.rectangle(
                output,
                (x1, y1 - text_size[1] - 8),
                (x1 + text_size[0] + 8, y1),
                color,
                -1
            )
            
            # Texte
            cv2.putText(
                output,
                label,
                (x1 + 4, y1 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return output
    
    @staticmethod
    def _cv2_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
        """Convertit une image OpenCV en QPixmap"""
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        
        # Convertir BGR à RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        q_img = QImage(
            rgb_img.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        
        return QPixmap.fromImage(q_img)
