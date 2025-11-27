#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Color and Danger Utilities
- Calcul du score de danger
- Attribution des couleurs selon le niveau de danger
"""


def get_color_by_dangerosity(score: int) -> tuple:
    """
    Retourne une couleur BGR selon le score de dangerosité
    
    Args:
        score: Score de danger (0-100)
    
    Returns:
        tuple: Couleur BGR (B, G, R)
    """
    if score <= 20:
        r = int(144 * (score / 20.0))
        g = int(100 + 138 * (score / 20.0))
        b = r
        return (b, g, r)
    
    if score <= 40:
        ratio = (score - 20) / 20.0
        return (int(144 * (1 - ratio)), int(238 + 17 * ratio), int(144 + 111 * ratio))
    
    if score <= 60:
        ratio = (score - 40) / 20.0
        return (0, int(255 - 90 * ratio), 255)
    
    if score <= 80:
        ratio = (score - 60) / 20.0
        return (0, int(165 * (1 - ratio)), 255)
    
    ratio = (score - 80) / 20.0
    return (0, 0, int(255 - 116 * ratio))


def calculate_distance_from_shore(x, y, w, h) -> float:
    """
    Calcule la distance relative par rapport au rivage
    (rive considérée en bas de la minimap)
    
    Args:
        x, y: Position sur la minimap
        w, h: Dimensions de la minimap
    
    Returns:
        float: Distance normalisée (0.0 = rivage, 1.0 = large)
    """
    return max(0.0, min(1.0, (h - y) / h))
