#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Affichage des alertes
"""

from datetime import datetime
from typing import List, Dict

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor


class AlertWidget(QWidget):
    """Widget pour afficher une alerte individuelle"""
    
    def __init__(self, alert: Dict):
        super().__init__()
        layout = QVBoxLayout(self)
        
        alert_type = alert.get('type', 'unknown')
        person_id = alert.get('person_id', '?')
        duration = alert.get('duration_frames', 0)
        
        # Style en fonction du type
        if alert_type == 'danger':
            bg_color = '#ff4444'
            title = f"🚨 DANGER - Personne {person_id}"
            detail = f"Sous l'eau depuis {duration} frames"
        else:
            bg_color = '#ffaa00'
            title = f"⚠️ {alert_type.upper()}"
            detail = f"Personne {person_id}"
        
        # Titre
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title_label.setStyleSheet("color: white;")
        layout.addWidget(title_label)
        
        # Détails
        detail_label = QLabel(detail)
        detail_label.setFont(QFont("Arial", 10))
        detail_label.setStyleSheet("color: #eee;")
        layout.addWidget(detail_label)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        time_label = QLabel(f"[{timestamp}]")
        time_label.setFont(QFont("Arial", 9))
        time_label.setStyleSheet("color: #aaa;")
        layout.addWidget(time_label)
        
        # Style du widget
        self.setStyleSheet(f"""
            QWidget {{
                background: {bg_color};
                border-radius: 5px;
                padding: 10px;
                margin: 5px;
            }}
        """)
        
        self.layout().setContentsMargins(5, 5, 5, 5)


class AlertDisplay(QWidget):
    """Widget pour afficher les alertes actives"""
    
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Titre
        title = QLabel("Alertes")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #FFD700; padding: 10px;")
        layout.addWidget(title)
        
        # Area de scroll pour les alertes
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background: #2b2b2b;
                border: 1px solid #555;
                border-radius: 5px;
            }
            QScrollBar:vertical {
                background: #3b3b3b;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #666;
                border-radius: 5px;
                min-height: 20px;
            }
        """)
        
        # Container pour les alertes
        self.alerts_container = QWidget()
        self.alerts_layout = QVBoxLayout(self.alerts_container)
        self.alerts_layout.setContentsMargins(0, 0, 0, 0)
        self.alerts_layout.setSpacing(0)
        
        scroll_area.setWidget(self.alerts_container)
        layout.addWidget(scroll_area)
        
        # Label si pas d'alertes
        self.no_alerts_label = QLabel("✓ Aucune alerte")
        self.no_alerts_label.setFont(QFont("Arial", 11))
        self.no_alerts_label.setStyleSheet("color: #51cf66; padding: 20px; text-align: center;")
        self.alerts_layout.addWidget(self.no_alerts_label)
        self.alerts_layout.addStretch()
        
        # Timer pour nettoyer les alertes
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._cleanup_old_alerts)
        self.cleanup_timer.start(1000)  # Check every second
        
        self.current_alerts = {}
        self.alert_timestamps = {}
    
    def show_alerts(self, alerts: List[Dict]):
        """Affiche les alertes"""
        # Mettre à jour la liste des alertes actuelles
        for alert in alerts:
            alert_id = f"{alert.get('type')}_{alert.get('person_id')}"
            self.current_alerts[alert_id] = alert
            self.alert_timestamps[alert_id] = datetime.now()
        
        # Nettoyer le container
        for i in reversed(range(self.alerts_layout.count())):
            widget = self.alerts_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Afficher les alertes
        if self.current_alerts:
            self.no_alerts_label.hide()
            
            for alert_id, alert in self.current_alerts.items():
                alert_widget = AlertWidget(alert)
                self.alerts_layout.addWidget(alert_widget)
            
            self.alerts_layout.addStretch()
        else:
            self.no_alerts_label.show()
            self.alerts_layout.addStretch()
    
    def _cleanup_old_alerts(self):
        """Nettoie les alertes expirées"""
        current_time = datetime.now()
        expired = []
        
        for alert_id, timestamp in self.alert_timestamps.items():
            elapsed = (current_time - timestamp).total_seconds()
            if elapsed > 8.0:  # Durée d'affichage d'une alerte
                expired.append(alert_id)
        
        for alert_id in expired:
            if alert_id in self.current_alerts:
                del self.current_alerts[alert_id]
            del self.alert_timestamps[alert_id]
        
        if expired:
            self.show_alerts(list(self.current_alerts.values()))
