# Neptune Backend API

API REST pour la détection d'eau et de personnes en environnement aquatique.

## Démarrage rapide

### 1. Installation des dépendances

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configuration

Copier `.env.example` en `.env` et ajuster si nécessaire:

```bash
cp .env.example .env
```

### 3. Lancer le serveur

```bash
# Option 1: Avec le script
python start_backend.py

# Option 2: Directement avec Python
python main.py

# Option 3: Avec uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Le serveur démarre sur `http://localhost:8000`

## Documentation API

### Accès à la documentation interactive

- **Swagger UI**: http://localhost:8000/docs
- **Redoc**: http://localhost:8000/redoc

### Endpoints disponibles

#### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

#### Détecter sur une frame (upload)
```bash
curl -X POST "http://localhost:8000/api/v1/detect/frame" \
  -F "file=@frame.jpg" \
  -F "detect_persons=true" \
  -F "detect_water=true"
```

#### Infos vidéo
```bash
curl "http://localhost:8000/api/v1/video/info?video_path=/path/to/video.mp4"
```

#### Infos modèles
```bash
curl http://localhost:8000/api/v1/models/info
```

## Architecture

```
Backend (FastAPI)
├── main.py                 # Application principale
├── app/
│   ├── config.py          # Configuration
│   ├── api/
│   │   └── detection.py   # Routes API
│   ├── detection/
│   │   └── service.py     # Service IA
│   └── models/
│       └── schemas.py     # Schémas Pydantic
└── requirements.txt       # Dépendances
```

## Configuration

### Variables d'environnement

- `API_HOST`: Host du serveur (default: 0.0.0.0)
- `API_PORT`: Port du serveur (default: 8000)
- `DEVICE`: cuda ou cpu (default: cuda)
- `CONF_THRESHOLD`: Seuil de confiance (default: 0.5)
- `PERSON_MODEL_ID`: ID du modèle D-FINE
- `WATER_MODEL_PATH`: Chemin vers le modèle YOLO d'eau

## Modèles IA utilisés

1. **D-FINE** - Détection de personnes
   - Modèle: `ustc-community/dfine-xlarge-obj2coco`
   - Auto-téléchargement depuis HuggingFace

2. **YOLO** - Détection d'eau
   - Modèle: `nwd-v2.pt`
   - Chemin: `app/model/nwd-v2.pt` ou `backend/models/nwd-v2.pt`

## Tests

Voir les exemples Python dans `/tests` ou `/demo`

## Performance

- **GPU (CUDA)**: ~30-50ms par frame (1920x1080)
- **CPU**: ~200-500ms par frame

## Logs

Les logs sont affichés dans la console et peuvent être configurés via `--log-level` lors du démarrage.

```bash
python start_backend.py --log-level debug
```

## Troubleshooting

### Backend ne démarre pas

1. Vérifier que Python 3.8+ est installé
2. Vérifier les dépendances: `pip list`
3. Vérifier le port 8000 n'est pas utilisé

### Modèles non trouvés

1. Télécharger les modèles (D-FINE s'auto-télécharge)
2. Copier `nwd-v2.pt` dans `app/model/` ou `backend/models/`

### CUDA non disponible

Passer en CPU:
```bash
python start_backend.py --device cpu
```

## Intégration avec le Frontend

Voir `/app/api_client.py` pour l'utilisation du client API depuis le frontend PyQt6.

## Déploiement

### Avec Docker

```bash
docker build -t neptune-backend .
docker run -p 8000:8000 neptune-backend
```

### Production

1. Désactiver le reload: `--no-reload`
2. Utiliser Gunicorn/Uvicorn avec multiple workers
3. Ajouter un reverse proxy (Nginx)
4. Configurer les logs persistants
