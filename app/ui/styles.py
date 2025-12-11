"""
Neptune UI Design System
Modern Dark Theme for PyQt6
"""

# Color Palette
PALETTE = {
    # Backgrounds
    "bg_main": "#121212",       # Very dark grey (almost black)
    "bg_panel": "#1E1E1E",      # Dark grey for panels/cards
    "bg_input": "#2C2C2C",      # Lighter grey for inputs
    "bg_hover": "#333333",      # Hover state
    "bg_pressed": "#000000",    # Pressed state

    # Accents
    "accent_primary": "#007AFF",   # Vibrant Blue (iOS-like)
    "accent_hover": "#0062CC",     # Darker Blue
    "accent_secondary": "#00D4FF", # Cyan (Neptune brand)

    # Functional
    "success": "#34C759",       # Green
    "warning": "#FF9500",       # Orange
    "danger": "#FF3B30",        # Red
    "text_main": "#FFFFFF",     # White
    "text_secondary": "#A0A0A0",# Light Grey
    "border": "#3A3A3A",        # Border color
}

# Qt Style Sheet (QSS)
STYLESHEET = f"""
    /* Main Window */
    QMainWindow {{
        background-color: {PALETTE['bg_main']};
        color: {PALETTE['text_main']};
    }}

    /* Generic Widget */
    QWidget {{
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-size: 14px;
        color: {PALETTE['text_main']};
    }}

    /* GroupBox (Cards) */
    QGroupBox {{
        background-color: {PALETTE['bg_panel']};
        border: 1px solid {PALETTE['border']};
        border-radius: 8px;
        margin-top: 24px; /* Space for title */
        font-weight: bold;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        color: {PALETTE['accent_secondary']};
        background-color: transparent;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    /* Push Buttons */
    QPushButton {{
        background-color: {PALETTE['bg_input']};
        border: 1px solid {PALETTE['border']};
        border-radius: 6px;
        padding: 8px 16px;
        color: {PALETTE['text_main']};
        font-weight: 600;
        min-height: 24px;
    }}
    QPushButton:hover {{
        background-color: {PALETTE['bg_hover']};
        border-color: {PALETTE['accent_secondary']};
    }}
    QPushButton:pressed {{
        background-color: {PALETTE['bg_pressed']};
    }}
    QPushButton:disabled {{
        background-color: {PALETTE['bg_main']};
        color: {PALETTE['text_secondary']};
        border-color: {PALETTE['border']};
    }}

    /* Primary Action Button (Blue) */
    QPushButton[class="primary"] {{
        background-color: {PALETTE['accent_primary']};
        border: none;
        color: white;
    }}
    QPushButton[class="primary"]:hover {{
        background-color: {PALETTE['accent_hover']};
    }}

    /* Danger Button (Red) */
    QPushButton[class="danger"] {{
        background-color: transparent;
        border: 1px solid {PALETTE['danger']};
        color: {PALETTE['danger']};
    }}
    QPushButton[class="danger"]:hover {{
        background-color: {PALETTE['danger']};
        color: white;
    }}

    /* Icon Buttons (Media Controls) */
    QPushButton[class="icon-btn"] {{
        background-color: transparent;
        border: none;
        border-radius: 20px; /* Circular if size matches */
        font-size: 20px;
        padding: 5px;
    }}
    QPushButton[class="icon-btn"]:hover {{
        background-color: {PALETTE['bg_hover']};
        color: {PALETTE['accent_secondary']};
    }}

    /* Line Edit */
    QLineEdit {{
        background-color: {PALETTE['bg_input']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        padding: 8px;
        color: {PALETTE['text_main']};
        selection-background-color: {PALETTE['accent_primary']};
    }}
    QLineEdit:focus {{
        border: 1px solid {PALETTE['accent_secondary']};
    }}

    /* SpinBox */
    QDoubleSpinBox {{
        background-color: {PALETTE['bg_input']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        padding: 6px;
        color: {PALETTE['text_main']};
    }}
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        background-color: transparent;
        border: none;
        width: 16px;
    }}
    QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {{
        width: 10px;
        height: 10px;
    }}

    /* Labels */
    QLabel {{
        color: {PALETTE['text_main']};
    }}
    QLabel[class="h1"] {{
        font-size: 24px;
        font-weight: bold;
        color: {PALETTE['text_main']};
    }}
    QLabel[class="h2"] {{
        font-size: 18px;
        font-weight: 600;
        color: {PALETTE['text_secondary']};
    }}
    QLabel[class="stat-value"] {{
        font-size: 20px;
        font-weight: bold;
        color: {PALETTE['accent_secondary']};
    }}
    QLabel[class="stat-label"] {{
        font-size: 12px;
        color: {PALETTE['text_secondary']};
        text-transform: uppercase;
    }}

    /* Text Edit (Logs) */
    QTextEdit {{
        background-color: {PALETTE['bg_input']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        color: {PALETTE['text_secondary']};
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 12px;
    }}

    /* Sliders */
    QSlider::groove:horizontal {{
        border: 1px solid {PALETTE['bg_input']};
        height: 8px;
        background: {PALETTE['bg_input']};
        margin: 2px 0;
        border-radius: 4px;
    }}
    QSlider::handle:horizontal {{
        background: {PALETTE['accent_secondary']};
        border: 1px solid {PALETTE['accent_secondary']};
        width: 18px;
        height: 18px;
        margin: -7px 0;
        border-radius: 9px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {PALETTE['text_main']};
    }}
"""
