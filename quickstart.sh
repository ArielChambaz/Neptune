#!/usr/bin/env bash
# 🚀 Quick Start Script - Neptune Frontend/Backend

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     Neptune Application - Frontend/Backend Setup         ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -e "${BLUE}[1/4] Vérification Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"
else
    echo -e "${YELLOW}✗ Python3 non trouvé${NC}"
    exit 1
fi

# Install backend dependencies
echo ""
echo -e "${BLUE}[2/4] Installation dépendances backend...${NC}"
if [ -d "backend" ]; then
    cd backend
    pip install -r requirements.txt
    cd ..
    echo -e "${GREEN}✓ Dépendances backend installées${NC}"
else
    echo -e "${YELLOW}✗ Dossier backend non trouvé${NC}"
fi

# Install frontend dependencies
echo ""
echo -e "${BLUE}[3/4] Installation dépendances frontend...${NC}"
if [ -d "app" ]; then
    cd app
    pip install -r requirements.txt
    cd ..
    echo -e "${GREEN}✓ Dépendances frontend installées${NC}"
else
    echo -e "${YELLOW}✗ Dossier app non trouvé${NC}"
fi

# Summary
echo ""
echo -e "${BLUE}[4/4] Résumé${NC}"
echo ""
echo -e "${GREEN}✓ Installation terminée${NC}"
echo ""
echo "Prochaines étapes:"
echo ""
echo -e "${YELLOW}Terminal 1 (Backend):${NC}"
echo "  cd backend"
echo "  python start_backend.py"
echo ""
echo -e "${YELLOW}Terminal 2 (Frontend):${NC}"
echo "  cd app"
echo "  python neptune_app.py"
echo ""
echo "Documentation:"
echo "  • ARCHITECTURE.md - Vue d'ensemble"
echo "  • SETUP_GUIDE.md - Guide complet"
echo "  • backend/README.md - Doc backend"
echo ""
echo -e "${GREEN}Ready! 🚀${NC}"
