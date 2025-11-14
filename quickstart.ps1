# 🚀 Quick Start Script - Neptune Frontend/Backend (PowerShell)

Write-Host @"
╔═══════════════════════════════════════════════════════════╗
║     Neptune Application - Frontend/Backend Setup         ║
╚═══════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Check Python
Write-Host "[1/4] Vérification Python..." -ForegroundColor Blue
$pythonCheck = python --version 2>&1
if ($pythonCheck -like "*3.*") {
    Write-Host "✓ $pythonCheck" -ForegroundColor Green
} else {
    Write-Host "✗ Python3 non trouvé" -ForegroundColor Yellow
    exit 1
}

# Install backend dependencies
Write-Host ""
Write-Host "[2/4] Installation dépendances backend..." -ForegroundColor Blue
if (Test-Path "backend") {
    Push-Location backend
    pip install -r requirements.txt
    Pop-Location
    Write-Host "✓ Dépendances backend installées" -ForegroundColor Green
} else {
    Write-Host "✗ Dossier backend non trouvé" -ForegroundColor Yellow
}

# Install frontend dependencies
Write-Host ""
Write-Host "[3/4] Installation dépendances frontend..." -ForegroundColor Blue
if (Test-Path "app") {
    Push-Location app
    pip install -r requirements.txt
    Pop-Location
    Write-Host "✓ Dépendances frontend installées" -ForegroundColor Green
} else {
    Write-Host "✗ Dossier app non trouvé" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "[4/4] Résumé" -ForegroundColor Blue
Write-Host ""
Write-Host "✓ Installation terminée" -ForegroundColor Green
Write-Host ""
Write-Host "Prochaines étapes:" -ForegroundColor Cyan
Write-Host ""
Write-Host "Terminal 1 (Backend):" -ForegroundColor Yellow
Write-Host "  cd backend"
Write-Host "  python start_backend.py"
Write-Host ""
Write-Host "Terminal 2 (Frontend):" -ForegroundColor Yellow
Write-Host "  cd app"
Write-Host "  python neptune_app.py"
Write-Host ""
Write-Host "Documentation:" -ForegroundColor Cyan
Write-Host "  • ARCHITECTURE.md - Vue d'ensemble"
Write-Host "  • SETUP_GUIDE.md - Guide complet"
Write-Host "  • backend/README.md - Doc backend"
Write-Host ""
Write-Host "Ready! 🚀" -ForegroundColor Green
