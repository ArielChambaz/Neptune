#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Main Window
- Interface utilisateur principale
- Design System: Modern / Professional / Dark Theme
"""

from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QGroupBox, QGridLayout, QDoubleSpinBox, QSpinBox,
    QSplitter, QLineEdit, QApplication, QFrame, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage, QIcon

from config_pyqt6 import DETECTION, ALERTS, UI
from ui.styles import STYLESHEET
from core.video_processor import VideoProcessor
from utils.audio import speak_alert, initialize_audio


class NeptuneMainWindow(QMainWindow):
    """Fenêtre principale de l'application Neptune"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neptune - Aquatic Surveillance")
        self.setGeometry(100, 100, UI['width'], UI['height'])
        
        # Style de l'interface
        self.setStyleSheet(STYLESHEET)
        
        # Composants principaux
        self.video_processor = VideoProcessor()
        self.is_playing = False
        
        # Construction de l'interface
        self._build_ui()
        self._connect_signals()
        
        # Initialisation audio
        initialize_audio()
    
    def _build_ui(self):
        """Construit l'interface utilisateur"""
        central = QWidget()
        self.setCentralWidget(central)

        # Main Horizontal Layout
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # === Left: Video Area ===
        video_area = self._create_video_area()
        main_layout.addWidget(video_area, stretch=3) # Takes 75% space approx
        
        # === Right: Sidebar Controls ===
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar, stretch=1) # Takes 25% space approx
        
        self.statusBar().showMessage("Ready - Select a video source")

    def _create_video_area(self):
        """Area containing Video Player and Playback Controls"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Video Display (Frame)
        self.video_frame = QFrame()
        self.video_frame.setObjectName("VideoFrame") # For styling
        
        video_layout = QVBoxLayout(self.video_frame)
        video_layout.setContentsMargins(1, 1, 1, 1) # Thin border inside
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("NO VIDEO SIGNAL\n\nLoad a source to start surveillance")
        self.video_label.setStyleSheet("color: #666; font-weight: bold;")
        self.video_label.setScaledContents(True)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        video_layout.addWidget(self.video_label)
        layout.addWidget(self.video_frame, stretch=1)

        # Playback Controls Bar
        controls_bar = QWidget()
        controls_bar.setFixedHeight(60)
        controls_bar.setStyleSheet("background-color: #1E1E1E; border-radius: 8px;")
        controls_layout = QHBoxLayout(controls_bar)
        controls_layout.setContentsMargins(15, 5, 15, 5)

        # Play/Pause
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(40, 40)
        self.play_btn.setProperty("class", "icon-btn")
        self.play_btn.setToolTip("Play/Pause (Space)")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_btn)
        
        # Stop
        self.stop_btn = QPushButton("⏹")
        self.stop_btn.setFixedSize(40, 40)
        self.stop_btn.setProperty("class", "icon-btn")
        self.stop_btn.setToolTip("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_playback)
        controls_layout.addWidget(self.stop_btn)

        controls_layout.addStretch()

        # Water Toggle (Quick Access)
        self.btn_toggle_water = QPushButton("🌊 Water Zone")
        self.btn_toggle_water.setCheckable(True)
        self.btn_toggle_water.setChecked(False)
        self.btn_toggle_water.clicked.connect(self.toggle_water_detection)
        self.btn_toggle_water.setToolTip("Toggle Water Zone Overlay (W)")
        controls_layout.addWidget(self.btn_toggle_water)

        layout.addWidget(controls_bar)

        return container

    def _create_sidebar(self):
        """Right sidebar with configuration and stats"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(5, 0, 5, 0)
        layout.setSpacing(20)

        # 1. Source Selection
        source_group = QGroupBox("VIDEO SOURCE")
        source_layout = QVBoxLayout(source_group)
        
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("/path/to/video.mp4")
        self.path_input.setText("/home/achambaz/neptune/G-EIP-700-REN-7-1-eip-adrien.picot/app/video/rozel-15fps-fullhd.mp4")
        source_layout.addWidget(self.path_input)
        
        load_btn = QPushButton("LOAD SOURCE")
        load_btn.setProperty("class", "primary")
        load_btn.clicked.connect(self.load_video_from_path)
        source_layout.addWidget(load_btn)
        
        self.video_path_label = QLabel("No source loaded")
        self.video_path_label.setStyleSheet("color: #666; font-size: 11px;")
        self.video_path_label.setWordWrap(True)
        source_layout.addWidget(self.video_path_label)
        
        layout.addWidget(source_group)
        
        # 2. Live Stats
        stats_group = QGroupBox("LIVE MONITORING")
        stats_layout = QGridLayout(stats_group)
        
        # Active
        lbl_active_title = QLabel("ACTIVE TRACKS")
        lbl_active_title.setProperty("class", "stat-label")
        stats_layout.addWidget(lbl_active_title, 0, 0)

        self.lbl_active = QLabel("0")
        self.lbl_active.setProperty("class", "stat-value")
        stats_layout.addWidget(self.lbl_active, 1, 0)
        
        # Underwater
        lbl_underwater_title = QLabel("UNDERWATER")
        lbl_underwater_title.setProperty("class", "stat-label")
        stats_layout.addWidget(lbl_underwater_title, 0, 1)

        self.lbl_underwater = QLabel("0")
        self.lbl_underwater.setProperty("class", "stat-value")
        self.lbl_underwater.setStyleSheet("color: #FF9500;") # Warning color
        stats_layout.addWidget(self.lbl_underwater, 1, 1)
        
        # Danger
        lbl_danger_title = QLabel("DANGER")
        lbl_danger_title.setProperty("class", "stat-label")
        stats_layout.addWidget(lbl_danger_title, 2, 0)

        self.lbl_danger = QLabel("0")
        self.lbl_danger.setProperty("class", "stat-value")
        self.lbl_danger.setStyleSheet("color: #FF3B30;") # Danger color
        stats_layout.addWidget(self.lbl_danger, 3, 0)
        
        # Max Score
        lbl_max_score_title = QLabel("MAX RISK")
        lbl_max_score_title.setProperty("class", "stat-label")
        stats_layout.addWidget(lbl_max_score_title, 2, 1)

        self.lbl_max_score = QLabel("0.0")
        self.lbl_max_score.setProperty("class", "stat-value")
        stats_layout.addWidget(self.lbl_max_score, 3, 1)

        layout.addWidget(stats_group)

        # 3. Detection Settings
        settings_group = QGroupBox("SETTINGS")
        settings_layout = QVBoxLayout(settings_group)

        # Confidence
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Confidence:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setValue(DETECTION['conf_threshold'])
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.valueChanged.connect(self.update_confidence)
        row1.addWidget(self.conf_spin)
        settings_layout.addLayout(row1)
        
        # Danger Threshold
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Danger Time (s):"))
        self.danger_spin = QDoubleSpinBox()
        self.danger_spin.setRange(1.0, 30.0)
        self.danger_spin.setValue(ALERTS['danger_threshold'])
        self.danger_spin.valueChanged.connect(self.update_danger_threshold)
        row2.addWidget(self.danger_spin)
        settings_layout.addLayout(row2)

        # Skip Frames
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Skip Frames:"))
        self.skip_frames_spin = QSpinBox()
        self.skip_frames_spin.setRange(1, 30)
        self.skip_frames_spin.setValue(self.video_processor.skip_frames)
        self.skip_frames_spin.setSingleStep(1)
        self.skip_frames_spin.valueChanged.connect(self.update_skip_frames)
        row3.addWidget(self.skip_frames_spin)
        settings_layout.addLayout(row3)
        
        recalc_btn = QPushButton("Recalculate Water Zone")
        recalc_btn.clicked.connect(self.recalculate_water_zone)
        settings_layout.addWidget(recalc_btn)
        
        layout.addWidget(settings_group)
        
        # 4. Alerts Log
        alerts_group = QGroupBox("ALERTS LOG")
        alerts_layout = QVBoxLayout(alerts_group)
        
        self.alerts_text = QTextEdit()
        self.alerts_text.setMaximumHeight(150)
        self.alerts_text.setReadOnly(True)
        alerts_layout.addWidget(self.alerts_text)
        
        test_alert_btn = QPushButton("Test Voice Alert")
        test_alert_btn.setProperty("class", "danger")
        test_alert_btn.clicked.connect(self.test_voice_alert)
        alerts_layout.addWidget(test_alert_btn)
        
        layout.addWidget(alerts_group)
        
        layout.addStretch()
        scroll.setWidget(content)
        return scroll

    def _connect_signals(self):
        """Connecte les signaux du video processor"""
        self.video_processor.frameReady.connect(self.update_frame)
        self.video_processor.statsReady.connect(self.update_stats)
        self.video_processor.alertTriggered.connect(self.handle_alert)
        
        # Configuration initiale
        self.video_processor.conf_threshold = DETECTION['conf_threshold']
        self.video_processor.danger_threshold = ALERTS['danger_threshold']
    
    # === Slots d'interface ===
    
    def load_video_from_path(self):
        """Charge une vidéo depuis le chemin saisi"""
        path = self.path_input.text().strip()
        
        if not path:
            self.statusBar().showMessage("Veuillez saisir un chemin")
            return
        
        if not Path(path).exists():
            self.statusBar().showMessage("Fichier introuvable")
            return
        
        self.video_path_label.setText(Path(path).name)
        
        # Arrêt de l'ancien thread
        try:
            if self.video_processor.isRunning():
                self.video_processor.stop()
        except Exception:
            pass
        
        # Création d'un nouveau processor
        # IMPORTANT: Preserve current settings
        current_skip = self.skip_frames_spin.value()
        current_conf = self.conf_spin.value()
        current_danger = self.danger_spin.value()

        self.video_processor = VideoProcessor()
        self.video_processor.skip_frames = current_skip
        self.video_processor.conf_threshold = current_conf
        self.video_processor.danger_threshold = current_danger

        self._connect_signals()
        
        # Chargement des modèles IA
        self.statusBar().showMessage("Loading AI models...")
        QApplication.processEvents()
        ok = self.video_processor.load_models()
        status_msg = "AI Models Ready" if ok else "Error loading models"
        self.statusBar().showMessage(status_msg)
        
        # Chargement de la vidéo
        self.statusBar().showMessage("Loading video...")
        QApplication.processEvents()
        
        if self.video_processor.load_video(path):
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.statusBar().showMessage("Source Loaded - Ready to Play")
            self.video_label.setText("")
        else:
            self.statusBar().showMessage("Error loading video source")
    
    def toggle_playback(self):
        """Bascule entre lecture et pause"""
        if not self.is_playing:
            if not self.video_processor.isRunning():
                self.video_processor.start()
            else:
                self.video_processor.is_paused = False
            
            self.is_playing = True
            self.play_btn.setText("⏸")
            self.statusBar().showMessage("Monitoring Active")
        else:
            self.video_processor.is_paused = True
            self.is_playing = False
            self.play_btn.setText("▶")
            self.statusBar().showMessage("Paused")
    
    def stop_playback(self):
        """Arrête la lecture"""
        try:
            if self.video_processor.isRunning():
                self.video_processor.stop()
        except Exception:
            pass
        
        self.is_playing = False
        self.play_btn.setText("▶")
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.video_label.setText(
            "MONITORING STOPPED"
        )
        self.statusBar().showMessage("Stopped")
    
    def update_frame(self, frame):
        """Met à jour l'affichage de la frame vidéo"""
        try:
            h, w, c = frame.shape
            qimg = QImage(frame.data, w, h, 3*w, QImage.Format.Format_RGB888).rgbSwapped()

            # Scale respecting aspect ratio within the label
            pix = QPixmap.fromImage(qimg).scaled(
                self.video_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(pix)
        except Exception as e:
            print(f"[UI] update_frame KO: {e}")
    
    def update_stats(self, stats):
        """Met à jour les statistiques affichées"""
        try:
            self.lbl_active.setText(f"{stats['active']}")
            self.lbl_underwater.setText(f"{stats['underwater']}")
            self.lbl_danger.setText(f"{stats['danger']}")
            
            max_score = stats['max_score']
            self.lbl_max_score.setText(f"{max_score:.1f}")
            
            # Dynamic styling for high risk
            if max_score >= 50:
                 self.lbl_max_score.setStyleSheet("color: #FF3B30; font-weight: bold;") # Danger
            elif max_score >= 30:
                 self.lbl_max_score.setStyleSheet("color: #FF9500; font-weight: bold;") # Warning
            else:
                 self.lbl_max_score.setStyleSheet("color: #00D4FF; font-weight: bold;") # Normal
            
        except Exception as e:
            print(f"[UI] update_stats KO: {e}")
    
    def handle_alert(self, message):
        """Gère l'affichage d'une nouvelle alerte"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.alerts_text.append(f"[{timestamp}] {message}")

        # Auto-scroll
        sb = self.alerts_text.verticalScrollBar()
        sb.setValue(sb.maximum())

        self.statusBar().showMessage(f"ALERT: {message}")
    
    def test_voice_alert(self):
        """Teste l'alerte vocale"""
        speak_alert("test")
        self.handle_alert("Voice alert system test")
    
    def toggle_water_detection(self, checked):
        """Bascule l'affichage de la détection d'eau"""
        self.video_processor.show_water_detection = checked
        self.btn_toggle_water.setChecked(checked)
        print(f"[UI] Water Overlay: {'ON' if checked else 'OFF'}")
    
    def recalculate_water_zone(self):
        """Recalcule la zone d'eau"""
        print("[UI] Recalculating water zone...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("Recalculating water zone...")
        QApplication.processEvents()
        
        try:
            if self.video_processor.recalculate_water_detection():
                self.toggle_water_detection(True)
                self.statusBar().showMessage("Water zone updated successfully", 3000)
                self.handle_alert("Water zone updated")
            else:
                self.statusBar().showMessage("Failed to update water zone", 3000)
        except Exception as e:
            print(f"[UI] Error: {e}")
            self.statusBar().showMessage(f"Error: {e}", 3000)
        finally:
             QApplication.restoreOverrideCursor()
    
    def update_confidence(self, value):
        """Met à jour le seuil de confiance"""
        self.video_processor.conf_threshold = float(value)
    
    def update_danger_threshold(self, value):
        """Met à jour le seuil de danger"""
        self.video_processor.danger_threshold = float(value)

    def update_skip_frames(self, value):
        """Updates the skip frames setting"""
        self.video_processor.skip_frames = value
    
    # === Gestion des événements ===
    
    def keyPressEvent(self, event):
        """Gère les raccourcis clavier"""
        if event.key() == Qt.Key.Key_W:
            self.btn_toggle_water.click()
        elif event.key() == Qt.Key.Key_T:
            self.test_voice_alert()
        elif event.key() == Qt.Key.Key_R:
            self.recalculate_water_zone()
        elif event.key() == Qt.Key.Key_Space:
            if self.play_btn.isEnabled():
                self.play_btn.click()
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Gère la fermeture de l'application de manière sécurisée"""
        try:
            if self.video_processor.isRunning():
                self.video_processor.stop()
        except Exception as e:
            print(f"[UI] Shutdown error: {e}")
        event.accept()
