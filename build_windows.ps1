<#
.SYNOPSIS
    Build Windows executable for thermal-ai

.DESCRIPTION
    Creates a standalone .exe using PyInstaller

.NOTES
    Run this in PowerShell as Administrator on Windows
#>

# Exit on error
$ErrorActionPreference = "Stop"

Write-Host "=== Thermal-AI Windows Build Script ===" -ForegroundColor Green

# Check Python
$pythonVersion = python --version 2>&1
Write-Host "Python: $pythonVersion"

# Check PyInstaller
try {
    pyinstaller --version 2>&1 | Out-Null
    Write-Host "PyInstaller: OK"
} catch {
    Write-Host "Installing PyInstaller..." -ForegroundColor Yellow
    pip install pyinstaller
}

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create assets folder with icon if not exists
if (-not (Test-Path "assets")) {
    New-Item -ItemType Directory -Path "assets" | Out-Null
}

# Build
Write-Host "Building executable..." -ForegroundColor Yellow

$specFile = "thermal-ai.spec"
if (-not (Test-Path $specFile)) {
    Write-Host "Spec file not found, using command line..." -ForegroundColor Yellow
    
    pyinstaller `
        --clean `
        -F `
        --name thermal-ai `
        --add-data "config;config" `
        --add-data "src;src" `
        --hidden-import src.config_manager `
        --hidden-import src.thermal_converter `
        --hidden-import src.vlm_analyzer `
        --hidden-import src.png_exporter `
        --hidden-import src.pipeline `
        --hidden-import src.finetuning.dataset_builder `
        --hidden-import src.finetuning.train_unsloth `
        --hidden-import PIL `
        --hidden-import cv2 `
        --hidden-import numpy `
        --hidden-import pandas `
        --hidden-import yaml `
        --hidden-import requests `
        --hidden-import tqdm `
        --hidden-import tifffile `
        --exclude-module matplotlib `
        --exclude-module jupyter `
        --exclude-module notebook `
        --exclude-module IPython `
        --exclude-module torch `
        --exclude-module transformers `
        --exclude-module unsloth `
        --exclude-module trl `
        --exclude-module datasets `
        --exclude-module peft `
        --exclude-module bitsandbytes `
        src/pipeline.py
} else {
    pyinstaller --clean $specFile
}

# Verify output
$exePath = "dist\thermal-ai.exe"
if (Test-Path $exePath) {
    Write-Host "`n=== BUILD SUCCESS ===" -ForegroundColor Green
    Write-Host "Executable: $exePath"
    Write-Host "Size: $(("{0:N2}" -f ((Get-Item $exePath).Length / 1MB))) MB"
    
    # Test run
    Write-Host "`nTesting executable..." -ForegroundColor Yellow
    & $exePath --help
} else {
    Write-Host "`n=== BUILD FAILED ===" -ForegroundColor Red
    exit 1
}

Write-Host "`nDone! Copy dist/thermal-ai.exe to target machine." -ForegroundColor Green
Write-Host "Note: Target machine needs Ollama running separately." -ForegroundColor Yellow