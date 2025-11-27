#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Frontend Main Window
- Interface utilisateur communiquant avec l'API backend
"""

import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import threading
import time

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QGroupBox, QGridLayout, QDoubleSpinBox, 
    QSplitter, QLineEdit, QApplication, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QImage

from ..config import DETECTION, ALERTS, UI, API
from ..api_client import NeptuneAPIClient


class VideoStreamThread(QThread):
    """Thread pour traiter une vidéo et envoyer les frames à l'API"""
    frameReady = pyqtSignal(np.ndarray)
    statsReady = pyqtSignal(dict)
    alertTriggered = pyqtSignal(str)
    
    def __init__(self, video_path, api_client):
        super().__init__()
        self.video_path = video_path
        self.api_client = api_client
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        self.fps = 30
        
    def run(self):
        """Boucle principale de traitement vidéo"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.alertTriggered.emit("Impossible d'ouvrir la vidéo")
            return
        
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.is_running = True
        frame_delay = 1.0 / self.fps if self.fps > 0 else 0.033
        
        active_alerts = {}
        
        while self.is_running:
            if self.is_paused:
                time.sleep(0.05)
                continue
            
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Encodage de la frame en JPEG
            success, encoded = cv2.imencode('.jpg', frame)
            if not success:
                continue
            
            # Envoi à l'API
            try:
                result = self.api_client.detect_frame(encoded.tobytes())
                if result:
                    # Dessin des détections
                    vis = self._draw_detections(frame, result)
                    
                    # Traitement des alertes
                    if result.get('alerts'):
                        for alert in result['alerts']:
                            alert_key = f"{alert.get('type')}_{alert.get('person_id')}"
                            if alert_key not in active_alerts:
                                msg = f"ALERTE {alert['type']} - ID:{alert['person_id']}"
                                self.alertTriggered.emit(msg)
                                active_alerts[alert_key] = time.time()
                    
                    # Nettoyage des alertes expirées
                    now = time.time()
                    for key in list(active_alerts.keys()):
                        if now - active_alerts[key] > 8.0:
                            del active_alerts[key]
                    
                    # Statistiques
                    stats = {
                        'active': len([d for d in result.get('detections', []) if d.get('tracking_frames', 0) > 0]),
                        'underwater': len([d for d in result.get('detections', []) if d.get('water_state') == 'underwater']),
                        'danger': len(result.get('alerts', [])),
                        'max_score': 0,
                        'max_score_id': None,
                        'frame_count': self.frame_count
                    }
                    
                    self.statsReady.emit(stats)
                    self.frameReady.emit(vis)
                    
            except Exception as e:
                print(f"[VideoStream] Erreur API: {e}")
            
            # Respect du FPS
            time.sleep(frame_delay)
        
        cap.release()
        self.is_running = False
    
    def _draw_detections(self, frame, result):
        """Dessine les détections sur la frame"""
        vis = frame.copy()
        
        for det in result.get('detections', []):
            if not det.get('bbox'):
                continue
            
            x1, y1, x2, y2 = det['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Couleur selon l'état aquatique
            if det.get('water_state') == 'underwater':
                color = (0, 0, 255)  # Rouge
            else:
                color = (0, 255, 0)  # Vert
            
            # Boîte de détection
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Label avec ID et confiance
            label = f"ID:{det.get('id')} [{det.get('confidence', 0):.2f}]"
            cv2.putText(vis, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis
    
    def stop(self):
        """Arrête la boucle de traitement"""
        self.is_running = False


class NeptuneFrontendWindow(QMainWindow):
    """Fenêtre principale du frontend Neptune"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neptune Frontend")
        self.setGeometry(100, 100, UI['width'], UI['height'])
        
        # Style
        self.setStyleSheet(self._get_stylesheet())
        
        # API Client
        self.api_client = NeptuneAPIClient(API['base_url'])
        
        # Thread vidéo
        self.video_thread = None
        self.is_playing = False
        
        # Historique des alertes
        self.alert_history = []
        
        # Construction UI
        self._build_ui()
        
        # Vérification santé serveur
        self.health_timer = QTimer()
        self.health_timer.timeout.connect(self.check_server_health)
        self.health_timer.start(API['health_check_interval'])
        
        # Vérification initiale
        self.check_server_health()
    
    def _get_stylesheet(self):
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
            }
            QLineEdit {
                background:#1e1e1e;
                border:1px solid #555;
                color:#fff;
                border-radius:5px;
                padding:5px;
            }
        """
    
    def _build_ui(self):
        """Construit l'interface"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Panel gauche (contrôles)
        left_panel = self._create_control_panel()
        splitter.addWidget(left_panel)
        
        # Panel droit (vidéo)
        right_panel = self._create_video_panel()
        splitter.addWidget(right_panel)
        
        self.statusBar().showMessage("Démarrage... Vérification du serveur")
    
    def _create_control_panel(self):
        """Crée le panel de contrôle"""
        left = QWidget()
        left.setMaximumWidth(UI['control_panel_width'])
        left.setMinimumWidth(300)
        layout = QVBoxLayout(left)
        
        # Section API
        layout.addWidget(self._create_api_section())
        
        # Section fichier
        layout.addWidget(self._create_file_section())
        
        # Section lecture
        layout.addWidget(self._create_playback_section())
        
        # Section statistiques
        layout.addWidget(self._create_stats_section())
        
        # Section alertes
        layout.addWidget(self._create_alerts_section())
        
        layout.addStretch()
        return left
    
    def _create_api_section(self):
        """Crée la section API"""
        group = QGroupBox("État du Serveur")
        layout = QVBoxLayout(group)
        
        self.api_status_label = QLabel("Déconnecté")
        self.api_status_label.setStyleSheet("color:#ff4444; font-weight:bold;")
        layout.addWidget(self.api_status_label)
        
        self.api_url_label = QLabel(f"URL: {API['base_url']}")
        self.api_url_label.setStyleSheet("color:#888; font-size:10px;")
        layout.addWidget(self.api_url_label)
        
        return group
    
    def _create_file_section(self):
        """Crée la section fichier"""
        group = QGroupBox("Vidéo")
        layout = QVBoxLayout(group)
        
        row = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Chemin de la vidéo...")
        row.addWidget(self.path_input)
        
        btn_browse = QPushButton("Parcourir")
        btn_browse.clicked.connect(self.browse_video_file)
        row.addWidget(btn_browse)
        
        layout.addLayout(row)
        
        row2 = QHBoxLayout()
        btn_load = QPushButton("Charger")
        btn_load.clicked.connect(self.load_video)
        row2.addWidget(btn_load)
        
        layout.addLayout(row2)
        
        self.video_info_label = QLabel("Aucune vidéo")
        self.video_info_label.setStyleSheet("color:#888;")
        layout.addWidget(self.video_info_label)
        
        return group
    
    def _create_playback_section(self):
        """Crée la section lecture"""
        group = QGroupBox("Lecture")
        layout = QVBoxLayout(group)
        
        row = QHBoxLayout()
        
        self.play_btn = QPushButton("Lecture")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_playback)
        row.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("Arrêt")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_playback)
        row.addWidget(self.stop_btn)
        
        layout.addLayout(row)
        return group
    
    def _create_stats_section(self):
        """Crée la section stats"""
        group = QGroupBox("Statistiques")
        layout = QGridLayout(group)
        
        self.stats_labels = {
            'active': QLabel("0"),
            'underwater': QLabel("0"),
            'danger': QLabel("0"),
            'frame': QLabel("0")
        }
        
        titles = {
            'active': 'Personnes actives',
            'underwater': 'Sous l\'eau',
            'danger': 'En danger',
            'frame': 'Frame'
        }
        
        for i, (key, label) in enumerate(self.stats_labels.items()):
            label.setStyleSheet("font-weight:bold;")
            layout.addWidget(QLabel(titles[key] + ":"), i, 0)
            layout.addWidget(label, i, 1)
        
        return group
    
    def _create_alerts_section(self):
        """Crée la section alertes"""
        group = QGroupBox("Journal des Alertes")
        layout = QVBoxLayout(group)
        
        self.alerts_text = QTextEdit()
        self.alerts_text.setMaximumHeight(120)
        self.alerts_text.setReadOnly(True)
        layout.addWidget(self.alerts_text)
        
        return group
    
    def _create_video_panel(self):
        """Crée le panel vidéo"""
        right = QWidget()
        layout = QVBoxLayout(right)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(
            UI['video_panel_min_width'],
            UI['video_panel_min_height']
        )
        self.video_label.setStyleSheet(
            "QLabel { border:2px solid #555; border-radius:8px; "
            "background:#1e1e1e; color:#999; }"
        )
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("Aucune vidéo chargée\n\nSélectionnez une vidéo pour commencer")
        self.video_label.setScaledContents(True)
        
        layout.addWidget(self.video_label)
        return right
    
    def check_server_health(self):
        """Vérifie la santé du serveur"""
        is_ok = self.api_client.health_check()
        
        if is_ok:
            self.api_status_label.setText("✓ Connecté")
            self.api_status_label.setStyleSheet("color:#44ff44; font-weight:bold;")
            self.statusBar().showMessage("Serveur prêt")
        else:
            self.api_status_label.setText("✗ Déconnecté")
            self.api_status_label.setStyleSheet("color:#ff4444; font-weight:bold;")
            self.statusBar().showMessage("Serveur indisponible")
    
    def browse_video_file(self):
        """Ouvre un dialogue pour sélectionner une vidéo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner une vidéo",
            str(Path.home()),
            "Vidéos (*.mp4 *.avi *.mov *.mkv);;Tous les fichiers (*)"
        )
        
        if file_path:
            self.path_input.setText(file_path)
            self.video_info_label.setText(Path(file_path).name)
    
    def load_video(self):
        """Charge une vidéo"""
        video_path = self.path_input.text().strip()
        
        if not video_path:
            self.statusBar().showMessage("Veuillez sélectionner une vidéo")
            return
        
        if not Path(video_path).exists():
            self.statusBar().showMessage("Fichier non trouvé")
            return
        
        if not self.api_client.is_connected:
            self.statusBar().showMessage("Serveur non disponible")
            return
        
        # Arrêt du thread précédent
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        
        # Création du nouveau thread
        self.video_thread = VideoStreamThread(video_path, self.api_client)
        self.video_thread.frameReady.connect(self.update_frame)
        self.video_thread.statsReady.connect(self.update_stats)
        self.video_thread.alertTriggered.connect(self.handle_alert)
        
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage("Vidéo chargée - Prêt à jouer")
    
    def toggle_playback(self):
        """Bascule la lecture"""
        if not self.video_thread:
            return
        
        if not self.is_playing:
            if not self.video_thread.isRunning():
                self.video_thread.start()
            else:
                self.video_thread.is_paused = False
            
            self.is_playing = True
            self.play_btn.setText("Pause")
            self.statusBar().showMessage("Lecture en cours...")
        else:
            self.video_thread.is_paused = True
            self.is_playing = False
            self.play_btn.setText("Lecture")
            self.statusBar().showMessage("En pause")
    
    def stop_playback(self):
        """Arrête la lecture"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        
        self.is_playing = False
        self.play_btn.setText("Lecture")
        self.video_label.setText("Aucune vidéo en cours\n\nClik sur Lecture pour démarrer")
        self.statusBar().showMessage("Arrêté")
    
    def update_frame(self, frame):
        """Met à jour l'affichage"""
        try:
            h, w, c = frame.shape
            qimg = QImage(frame.data, w, h, 3*w, QImage.Format.Format_RGB888).rgbSwapped()
            pix = QPixmap.fromImage(qimg).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(pix)
        except Exception as e:
            print(f"[Frontend] Erreur update frame: {e}")
    
    def update_stats(self, stats):
        """Met à jour les stats"""
        self.stats_labels['active'].setText(str(stats['active']))
        self.stats_labels['underwater'].setText(str(stats['underwater']))
        self.stats_labels['danger'].setText(str(stats['danger']))
        self.stats_labels['frame'].setText(str(stats['frame_count']))
        
        # Couleurs
        if stats['danger'] > 0:
            self.stats_labels['danger'].setStyleSheet("color:#ff4444; font-weight:bold;")
        else:
            self.stats_labels['danger'].setStyleSheet("color:#fff; font-weight:bold;")
    
    def handle_alert(self, message):
        """Gère une alerte"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        alert_msg = f"[{timestamp}] {message}"
        self.alerts_text.append(alert_msg)
        self.alert_history.append(alert_msg)
        
        # Limitation de la taille de l'historique
        if len(self.alert_history) > 100:
            self.alerts_text.clear()
            self.alert_history = self.alert_history[-50:]
            for msg in self.alert_history:
                self.alerts_text.append(msg)
    
    def closeEvent(self, event):
        """Gère la fermeture"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        
        self.health_timer.stop()
        event.accept()
