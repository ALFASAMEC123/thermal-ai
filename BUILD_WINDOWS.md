# Windows Build Instructions

## Prerequisites on Windows

1. **Install Python 3.10+** from [python.org](https://python.org)
   - ✅ Check "Add Python to PATH" during installation

2. **Install Visual C++ Build Tools** (required for some packages)
   - Download: [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   - Select: "Desktop development with C++" workload

3. **Install Git** (optional): [git-scm.com](https://git-scm.com)

## Build Steps

### Option 1: PowerShell Script (Recommended)

```powershell
# 1. Clone or copy project to Windows machine
# 2. Open PowerShell as Administrator in project folder
# 3. Run build script
.\build_windows.ps1
```

### Option 2: Manual Commands

```cmd
# 1. Open Command Prompt in project folder

# 2. Install PyInstaller
pip install pyinstaller

# 3. Install dependencies
pip install -r requirements.txt

# 4. Build executable
pyinstaller --clean -F --name thermal-ai ^
    --add-data "config;config" ^
    --add-data "src;src" ^
    --hidden-import src.config_manager ^
    --hidden-import src.thermal_converter ^
    --hidden-import src.vlm_analyzer ^
    --hidden-import src.png_exporter ^
    --hidden-import src.pipeline ^
    --hidden-import src.finetuning.dataset_builder ^
    --hidden-import src.finetuning.train_unsloth ^
    --hidden-import PIL ^
    --hidden-import cv2 ^
    --hidden-import numpy ^
    --hidden-import pandas ^
    --hidden-import yaml ^
    --hidden-import requests ^
    --hidden-import tqdm ^
    --hidden-import tifffile ^
    --exclude-module matplotlib ^
    --exclude-module torch ^
    --exclude-module transformers ^
    --exclude-module unsloth ^
    --exclude-module trl ^
    --exclude-module datasets ^
    --exclude-module peft ^
    --exclude-module bitsandbytes ^
    src/pipeline.py
```

## Output

- Executable: `dist/thermal-ai.exe`
- Size: ~50-80 MB (depends on dependencies)

## Running on Target Machine

### 1. Install Ollama (separate)
```powershell
# Download from https://ollama.ai/download
# Or winget:
winget install Ollama.Ollama
```

### 2. Start Ollama and pull model
```cmd
ollama serve
ollama pull qwen2-vl:7b
```

### 3. Run thermal-ai
```cmd
# Single file
thermal-ai.exe data/raw/image.irb -o data/output/

# Directory
thermal-ai.exe data/raw/ -o data/output/ --recursive

# Check connection
thermal-ai.exe --check-ollama
```

## Configuration

Copy `config/settings.yaml` next to the `.exe` or use `--config`:
```cmd
thermal-ai.exe --config config\settings.yaml data/raw/ -o output/
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "cv2 not found" | Install Visual C++ Redistributable |
| "DLL load failed" | Run `pip install --force-reinstall opencv-python` |
| Large exe size | Use `--exclude-module` for unused packages |
| Font errors | Copy `C:\Windows\Fonts\consola.ttf` to assets/ |

## Distribution

Create a zip with:
```
thermal-ai-windows.zip
├── thermal-ai.exe
├── config/
│   └── settings.yaml
└── README.txt
```

Target machine only needs:
1. Windows 10/11
2. Ollama installed separately
3. Visual C++ Redistributable (usually pre-installed)