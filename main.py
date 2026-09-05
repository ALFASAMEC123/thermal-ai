#!/usr/bin/env python3
"""
Entry point for PyInstaller executable.
This file becomes the main executable.
"""

import sys
import os

# Add src to path for frozen executable
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    base_path = sys._MEIPASS
    src_path = os.path.join(base_path, 'src')
    config_path = os.path.join(base_path, 'config')
    sys.path.insert(0, src_path)
    sys.path.insert(0, config_path)
else:
    # Running as script
    base_path = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(base_path, 'src')
    sys.path.insert(0, src_path)

# Now import and run main pipeline
from src.pipeline import main

if __name__ == '__main__':
    main()