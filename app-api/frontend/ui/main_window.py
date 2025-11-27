#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface utilisateur principale du frontend
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QGroupBox, QGridLayout, QSplitter, QLineEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QStatusBar
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon

from ..app.config import UI, API, COLORS
from ..app.api_client import NeptuneAPIClient
from .video_display import VideoDisplay
from .alert_display import AlertDisplay


class FrameProcessorThread(QThread):
    """Thread pour traiter les frames en arrière-plan"""
    
    frame_processed = pyqtSignal(dict)  # Émet les résultats de détection
    error_occurred = pyqtSignal(str)
    
    def __init__(self, api_client: NeptuneAPIClient):
        super().__init__()
        self.api_client = api_client
        self.current_frame = None
        self.is_running = True
    
    def set_frame(self, frame_data: bytes):
        """Défini la frame à traiter"""
        self.current_frame = frame_data
    
    def run(self):
        """Boucle de traitement des frames"""
        while self.is_running:
            if self.current_frame is not None:
                result = self.api_client.detect_frame(self.current_frame)
                if result:
                    self.frame_processed.emit(result)
                else:
                    self.error_occurred.emit("Erreur lors du traitement de la frame")
                self.current_frame = None
            
            self.msleep(10)
    
    def stop(self):
        """Arrête le thread"""
        self.is_running = False
        self.wait()


class NeptuneFrontendWindow(QMainWindow):
    """Fenêtre principale du frontend Neptune"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neptune Frontend - Surveillance Aquatique")
        self.setGeometry(100, 100, UI['width'], UI['height'])
        
        # Style
        self.setStyleSheet(self._get_stylesheet())
        
        # Client API
        self.api_client = NeptuneAPIClient(API['base_url'])
        
        # Variables
        self.current_frame = None
        self.video_capture = None
        self.is_playing = False
        self.last_detections = []
        self.last_alerts = []
        
        # Thread de traitement
        self.processor_thread = FrameProcessorThread(self.api_client)
        self.processor_thread.frame_processed.connect(self._on_frame_processed)
        self.processor_thread.error_occurred.connect(self._on_processing_error)
        self.processor_thread.start()
        
        # Timer de vidéo
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self._process_video_frame)
        self.video_timer.setInterval(33)  # ~30fps
        
        # Timer de santé du serveur
        self.health_timer = QTimer()
        self.health_timer.timeout.connect(self._check_server_health)
        self.health_timer.setInterval(API['health_check_interval'])
        
        # Construction UI
        self._build_ui()
        
        # Vérification initiale
        self._check_server_health()
        self.health_timer.start()
        
        print("Frontend Neptune initialisé")
    
    def _get_stylesheet(self) -> str:
        """Retourne le style CSS"""
        return """
            QMainWindow { background:#2b2b2b; color:#fff; }
            QGroupBox { 
                font-weight:bold; 
                border:2px solid #555; 
                border-radius:8px; 
                margin-top:10px; 
                padding-top:10px; 
                background:#3b3b3b; 
            }
            QGroupBox::title { 
                left:10px; 
                padding:0 10px; 
                color:#00D4FF; 
            }
            QPushButton { 
                background:#4CAF50; 
                border:none; 
                color:white; 
                padding:10px; 
                border-radius:5px; 
                font-weight:bold; 
            }
            QPushButton:hover { background:#45a049; }
            QPushButton:pressed { background:#3d8b40; }
            QPushButton:disabled { background:#666; color:#999; }
            QLabel { color:#fff; }
            QTextEdit { 
                background:#1e1e1e; 
                border:1px solid #555; 
                color:#fff; 
                border-radius:5px; 
                font-family: 'Courier New';
            }
            QLineEdit {
                background:#1e1e1e;
                border:1px solid #555;
                color:#fff;
                padding:5px;
                border-radius:3px;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background:#1e1e1e;
                border:1px solid #555;
                color:#fff;
                padding:5px;
                border-radius:3px;
            }
        """
    
    def _build_ui(self):
        """Construit l'interface utilisateur"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Panel gauche (contrôles)
        left_panel = self._create_control_panel()
        splitter.addWidget(left_panel)
        
        # Panel droit (affichage vidéo + alertes)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Affichage vidéo
        self.video_display = VideoDisplay()
        right_layout.addWidget(self.video_display)
        
        # Affichage des alertes
        self.alert_display = AlertDisplay()
        right_layout.addWidget(self.alert_display, 1)
        
        splitter.addWidget(right_panel)
        
        # Barre de statut
        self.statusBar().showMessage("En attente de connexion au serveur...")
    
    def _create_control_panel(self) -> QWidget:
        """Crée le panel de contrôle gauche"""
        panel = QWidget()
        panel.setMaximumWidth(UI['control_panel_width'])
        panel.setMinimumWidth(300)
        layout = QVBoxLayout(panel)
        
        # Section fichier vidéo
        layout.addWidget(self._create_file_section())
        
        # Section serveur API
        layout.addWidget(self._create_server_section())
        
        # Section lecture
        layout.addWidget(self._create_playback_section())
        
        # Section statistiques
        layout.addWidget(self._create_stats_section())
        
        layout.addStretch()
        return panel
    
    def _create_file_section(self) -> QGroupBox:
        """Crée la section de sélection de fichier"""
        group = QGroupBox("Fichier Vidéo")
        layout = QVBoxLayout(group)
        
        # Chemin vidéo
        label = QLabel("Chemin de la vidéo:")
        label.setStyleSheet("color:#FFD700; font-weight:bold;")
        layout.addWidget(label)
        
        self.video_path_input = QLineEdit()
        self.video_path_input.setPlaceholderText("Ex: /path/to/video.mp4")
        layout.addWidget(self.video_path_input)
        
        # Bouton charger
        self.load_video_btn = QPushButton("Charger vidéo")
        self.load_video_btn.clicked.connect(self._load_video)
        layout.addWidget(self.load_video_btn)
        
        return group
    
    def _create_server_section(self) -> QGroupBox:
        """Crée la section de connexion au serveur"""
        group = QGroupBox("Serveur API")
        layout = QVBoxLayout(group)
        
        # État du serveur
        self.server_status_label = QLabel("🔴 Déconnecté")
        self.server_status_label.setStyleSheet("color:#ff6b6b; font-weight:bold;")
        layout.addWidget(self.server_status_label)
        
        # URL du serveur
        label = QLabel("URL du serveur:")
        layout.addWidget(label)
        
        self.server_url_input = QLineEdit()
        self.server_url_input.setText(API['base_url'])
        layout.addWidget(self.server_url_input)
        
        # Bouton reconnexion
        self.reconnect_btn = QPushButton("Reconnecter")
        self.reconnect_btn.clicked.connect(self._reconnect_server)
        layout.addWidget(self.reconnect_btn)
        
        return group
    
    def _create_playback_section(self) -> QGroupBox:
        """Crée la section de lecture"""
        group = QGroupBox("Lecture")
        layout = QGridLayout(group)
        
        # Boutons play/pause
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_playback)
        self.play_btn.setEnabled(False)
        layout.addWidget(self.play_btn, 0, 0)
        
        self.reset_btn = QPushButton("⟲ Reset")
        self.reset_btn.clicked.connect(self._reset_playback)
        self.reset_btn.setEnabled(False)
        layout.addWidget(self.reset_btn, 0, 1)
        
        # FPS
        layout.addWidget(QLabel("FPS:"), 1, 0)
        self.fps_spinbox = QDoubleSpinBox()
        self.fps_spinbox.setValue(30.0)
        self.fps_spinbox.setRange(1, 60)
        layout.addWidget(self.fps_spinbox, 1, 1)
        
        # Position frame
        layout.addWidget(QLabel("Frame:"), 2, 0)
        self.frame_label = QLabel("0 / 0")
        layout.addWidget(self.frame_label, 2, 1)
        
        return group
    
    def _create_stats_section(self) -> QGroupBox:
        """Crée la section de statistiques"""
        group = QGroupBox("Statistiques")
        layout = QVBoxLayout(group)
        
        # Détections
        self.detections_label = QLabel("Détections: 0")
        layout.addWidget(self.detections_label)
        
        # Alertes
        self.alerts_label = QLabel("Alertes: 0")
        self.alerts_label.setStyleSheet("color:#ff6b6b;")
        layout.addWidget(self.alerts_label)
        
        # État du suivi
        self.tracking_label = QLabel("Suivi: 0 personnes")
        layout.addWidget(self.tracking_label)
        
        # Logs
        layout.addWidget(QLabel("Logs:"))
        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setMaximumHeight(150)
        layout.addWidget(self.logs_text)
        
        return group
    
    # ===== Slots (événements) =====
    
    def _load_video(self):
        """Charge une vidéo"""
        video_path = self.video_path_input.text()
        if not video_path:
            self._add_log("❌ Veuillez entrer un chemin vidéo")
            return
        
        try:
            self.video_capture = cv2.VideoCapture(video_path)
            if not self.video_capture.isOpened():
                self._add_log(f"❌ Impossible d'ouvrir: {video_path}")
                return
            
            # Info vidéo
            frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.video_info = {
                'frame_count': frame_count,
                'fps': fps,
                'width': width,
                'height': height,
                'current_frame': 0,
                'last_water_zone': {}
            }
            
            self._add_log(f"✓ Vidéo chargée: {frame_count} frames @ {fps}fps ({width}x{height})")
            self.play_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            self.frame_label.setText(f"0 / {frame_count}")
            
        except Exception as e:
            self._add_log(f"❌ Erreur: {e}")
    
    def _toggle_playback(self):
        """Bascule la lecture"""
        if not self.video_capture:
            return
        
        if self.is_playing:
            self.video_timer.stop()
            self.play_btn.setText("▶ Play")
            self.is_playing = False
        else:
            self.video_timer.start()
            self.play_btn.setText("⏸ Pause")
            self.is_playing = True
    
    def _reset_playback(self):
        """Réinitialise la lecture"""
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_info['current_frame'] = 0
            self.api_client.reset_tracking()
            self._add_log("⟲ Lecture réinitialisée")
    
    def _process_video_frame(self):
        """Traite la frame vidéo actuelle"""
        if not self.video_capture or not self.api_client.is_connected:
            return
        
        ret, frame = self.video_capture.read()
        if not ret:
            self._toggle_playback()  # Arrêter la lecture
            self._add_log("✓ Fin de la vidéo")
            return
        
        self.current_frame = frame
        self.video_info['current_frame'] += 1
        
        # Afficher la frame IMMÉDIATEMENT avec les détections précédentes
        self.video_display.update_frame(
            self.current_frame,
            self.last_detections,
            self.video_info.get('last_water_zone', {})
        )
        
        # Encoder la frame en JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            frame_data = buffer.tobytes()
            self.processor_thread.set_frame(frame_data)
        
        # Mettre à jour le compteur de frames
        self.frame_label.setText(
            f"{self.video_info['current_frame']} / {self.video_info['frame_count']}"
        )
    
    def _on_frame_processed(self, result: Dict):
        """Callback lorsqu'une frame a été traitée"""
        try:
            # Extraire les détections et alertes
            detections = result.get('detections', [])
            alerts = result.get('alerts', [])
            water_zone = result.get('water_zone', {})
            
            self.last_detections = detections
            self.last_alerts = alerts
            
            # Sauvegarder la zone d'eau pour les frames suivantes
            if water_zone:
                self.video_info['last_water_zone'] = water_zone
            
            # La frame est déjà affichée dans _process_video_frame
            # on l'a juste mise à jour ici avec les nouvelles détections
            if self.current_frame is not None:
                self.video_display.update_frame(
                    self.current_frame,
                    detections,
                    water_zone
                )
            
            # Mettre à jour l'affichage des alertes
            if alerts:
                self.alert_display.show_alerts(alerts)
            
            # Mettre à jour les statistiques
            self.detections_label.setText(f"Détections: {len(detections)}")
            self.alerts_label.setText(f"Alertes: {len(alerts)}")
            
            if alerts:
                self._add_log(f"🚨 {len(alerts)} alerte(s) détectée(s)")
        
        except Exception as e:
            self._add_log(f"❌ Erreur lors du traitement: {e}")
    
    def _on_processing_error(self, error: str):
        """Callback en cas d'erreur de traitement"""
        self._add_log(f"❌ {error}")
    
    def _check_server_health(self):
        """Vérifie l'état du serveur"""
        if self.api_client.health_check():
            self.server_status_label.setText("🟢 Connecté")
            self.server_status_label.setStyleSheet("color:#51cf66; font-weight:bold;")
        else:
            self.server_status_label.setText("🔴 Déconnecté")
            self.server_status_label.setStyleSheet("color:#ff6b6b; font-weight:bold;")
            self.api_client.is_connected = False
    
    def _reconnect_server(self):
        """Tente de se reconnecter au serveur"""
        url = self.server_url_input.text()
        self.api_client.base_url = url
        self._check_server_health()
        if self.api_client.is_connected:
            self._add_log(f"✓ Connecté à {url}")
        else:
            self._add_log(f"❌ Impossible de se connecter à {url}")
    
    def _add_log(self, message: str):
        """Ajoute un message aux logs"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs_text.append(f"[{timestamp}] {message}")
        
        # Scrollbar vers le bas
        scrollbar = self.logs_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """Événement de fermeture"""
        self.video_timer.stop()
        self.health_timer.stop()
        self.processor_thread.stop()
        if self.video_capture:
            self.video_capture.release()
        self.api_client.close()
        event.accept()
