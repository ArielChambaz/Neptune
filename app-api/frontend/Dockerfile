FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances système pour PyQt6
RUN apt-get update && apt-get install -y \
    libqt6gui6 \
    libqt6core6 \
    libopencv-dev \
    python3-opencv \
    libsm6 \
    libxext6 \
    libxrender-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY app ./app
COPY ui ./ui

# Commande par défaut
CMD ["python", "-m", "app.main"]
