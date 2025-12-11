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

    /* ScrollArea */
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}

    /* ScrollBar */
    QScrollBar:vertical {{
        border: none;
        background: {PALETTE['bg_main']};
        width: 8px;
        margin: 0px 0px 0px 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {PALETTE['bg_input']};
        min-height: 20px;
        border-radius: 4px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    /* GroupBox (Cards) - Simplified */
    QGroupBox {{
        background-color: transparent; /* Cleaner look */
        border: none;
        border-top: 1px solid {PALETTE['border']};
        margin-top: 10px; /* Space for title */
        padding-top: 20px;
        font-weight: bold;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 0px;
        color: {PALETTE['text_secondary']}; /* More subtle */
        background-color: transparent;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }}

    /* Push Buttons */
    QPushButton {{
        background-color: {PALETTE['bg_input']};
        border: none;
        border-radius: 4px;
        padding: 10px 16px;
        color: {PALETTE['text_main']};
        font-weight: 600;
        min-height: 20px;
    }}
    QPushButton:hover {{
        background-color: {PALETTE['bg_hover']};
        color: {PALETTE['text_main']};
    }}
    QPushButton:pressed {{
        background-color: {PALETTE['bg_pressed']};
    }}
    QPushButton:disabled {{
        background-color: {PALETTE['bg_panel']};
        color: {PALETTE['border']};
    }}

    /* Primary Action Button (Blue) */
    QPushButton[class="primary"] {{
        background-color: {PALETTE['accent_primary']};
        color: white;
    }}
    QPushButton[class="primary"]:hover {{
        background-color: {PALETTE['accent_hover']};
    }}

    /* Danger Button (Red) */
    QPushButton[class="danger"] {{
        background-color: rgba(255, 59, 48, 0.1);
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
        font-size: 18px;
        padding: 5px;
    }}
    QPushButton[class="icon-btn"]:hover {{
        background-color: rgba(255, 255, 255, 0.1);
        color: {PALETTE['accent_secondary']};
    }}

    /* Line Edit */
    QLineEdit {{
        background-color: {PALETTE['bg_panel']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        padding: 10px;
        color: {PALETTE['text_main']};
        selection-background-color: {PALETTE['accent_primary']};
    }}
    QLineEdit:focus {{
        border: 1px solid {PALETTE['accent_secondary']};
        background-color: {PALETTE['bg_input']};
    }}

    /* SpinBox */
    QDoubleSpinBox, QSpinBox {{
        background-color: {PALETTE['bg_panel']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        padding: 8px;
        color: {PALETTE['text_main']};
        font-weight: bold;
    }}
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
    QSpinBox::up-button, QSpinBox::down-button {{
        background-color: transparent;
        border: none;
        width: 16px;
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
        font-size: 24px;
        font-weight: 300; /* Lighter weight for modern look */
        color: {PALETTE['text_main']};
    }}
    QLabel[class="stat-label"] {{
        font-size: 10px;
        font-weight: bold;
        color: {PALETTE['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* Text Edit (Logs) */
    QTextEdit {{
        background-color: {PALETTE['bg_main']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        color: {PALETTE['text_secondary']};
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 11px;
        padding: 5px;
    }}

    /* Sliders */
    QSlider::groove:horizontal {{
        border: 1px solid {PALETTE['bg_input']};
        height: 6px;
        background: {PALETTE['bg_input']};
        margin: 2px 0;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {PALETTE['accent_secondary']};
        border: none;
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {PALETTE['text_main']};
    }}

    /* Frame (Video) */
    QFrame#VideoFrame {{
        border: 1px solid {PALETTE['border']};
        background-color: #000;
        border-radius: 6px;
    }}
"""
