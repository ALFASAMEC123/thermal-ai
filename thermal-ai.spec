# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import os
import sys

# In PyInstaller spec context, use SPEC_DIR (provided by PyInstaller)
# or fall back to current directory
try:
    base_path = SPEC_DIR
except NameError:
    base_path = os.getcwd()

# Add src to pathex
pathex = [base_path, os.path.join(base_path, 'src')]

a = Analysis(
    ['main.py'],
    pathex=pathex,
    binaries=[],
    datas=[
        ('config', 'config'),
        ('src', 'src'),
    ],
    hiddenimports=[
        'src.config_manager',
        'src.thermal_converter',
        'src.vlm_analyzer',
        'src.png_exporter',
        'src.pipeline',
        'src.finetuning.dataset_builder',
        'src.finetuning.train_unsloth',
        'PIL',
        'cv2',
        'numpy',
        'pandas',
        'yaml',
        'requests',
        'tqdm',
        'tifffile',
        'pandas._libs.tslibs.base',
        'pandas._libs.tslibs.timedeltas',
        'pandas._libs.tslibs.np_datetime',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'jupyter',
        'notebook',
        'IPython',
        'torch',
        'transformers',
        'unsloth',
        'trl',
        'datasets',
        'peft',
        'bitsandbytes',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='thermal-ai',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(base_path, 'assets', 'icon.ico') if os.path.exists(os.path.join(base_path, 'assets', 'icon.ico')) else None,
)