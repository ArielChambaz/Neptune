#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration pour les clients API (Frontend)
"""

# URL du backend API
BACKEND_URL = "http://localhost:8000"

# Timeouts (secondes)
API_REQUEST_TIMEOUT = 30
API_HEALTH_CHECK_INTERVAL = 5

# Configuration de détection
DETECT_PERSONS_DEFAULT = True
DETECT_WATER_DEFAULT = True
CONFIDENCE_THRESHOLD = 0.5

# Retry policy
MAX_RETRIES = 3
RETRY_DELAY = 1  # secondes

# Logging
LOG_API_CALLS = True
LOG_LEVEL = "INFO"
