# Build script for Windows executable
# Run this on Windows with Python installed

# 1. Install PyInstaller
# pip install pyinstaller

# 2. Install dependencies
# pip install -r requirements.txt

# 3. Build the executable
# pyinstaller --clean -F --name thermal-ai ^
#     --add-data "config;config" ^
#     --add-data "src;src" ^
#     --hidden-import=src.config_manager ^
#     --hidden-import=src.thermal_converter ^
#     --hidden-import=src.vlm_analyzer ^
#     --hidden-import=src.png_exporter ^
#     --hidden-import=src.pipeline ^
#     --hidden-import=src.finetuning.dataset_builder ^
#     --hidden-import=src.finetuning.train_unsloth ^
#     --hidden-import=PIL ^
#     --hidden-import=cv2 ^
#     --hidden-import=numpy ^
#     --hidden-import=pandas ^
#     --hidden-import=yaml ^
#     --hidden-import=requests ^
#     --hidden-import=tqdm ^
#     --hidden-import=tifffile ^
#     src/pipeline.py

# 4. Output will be in dist/thermal-ai.exe