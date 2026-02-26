"""
PDF Anonymizer – Entry point.

Starts the PyQt6 GUI application for AI-powered PDF anonymisation.
"""

import sys
import os

# Ensure the src directory is on the path (needed for PyInstaller bundles).
# When frozen (--onefile / --onedir) the modules live inside _MEIPASS/src;
# when running from source they sit next to this file.
if getattr(sys, "frozen", False):
    base_dir = sys._MEIPASS
    src_dir = os.path.join(base_dir, "src")
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = base_dir

for p in (base_dir, src_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from gui import run_app

if __name__ == "__main__":
    run_app()
