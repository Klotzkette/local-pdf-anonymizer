# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for PDF Anonymizer.
Usage:  pyinstaller build.spec
"""

import sys
import importlib
from pathlib import Path

block_cipher = None

# llama-cpp-python stores its native libs (.dll/.so/.dylib) in a 'lib/'
# subdirectory next to its Python files.  PyInstaller doesn't detect these
# automatically, so we must collect them explicitly.
_llama_cpp_libs = []
try:
    _llama_pkg = Path(importlib.import_module('llama_cpp').__file__).parent
    _llama_lib_dir = _llama_pkg / 'lib'
    if _llama_lib_dir.is_dir():
        for f in _llama_lib_dir.iterdir():
            if f.suffix in ('.dll', '.so', '.dylib') or '.so.' in f.name:
                # (source_path, dest_dir_inside_bundle)
                _llama_cpp_libs.append((str(f), str(Path('llama_cpp') / 'lib')))
except Exception:
    pass

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=_llama_cpp_libs,
    datas=[],
    hiddenimports=[
        'llama_cpp',
        'huggingface_hub',
        'PyQt6',
        'fitz',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'torchvision', 'torchaudio',
        'transformers', 'accelerate', 'triton',
        'nvidia', 'nvidia.cublas', 'nvidia.cuda_cupti',
        'nvidia.cuda_nvrtc', 'nvidia.cuda_runtime',
        'nvidia.cudnn', 'nvidia.cufft', 'nvidia.curand',
        'nvidia.cusolver', 'nvidia.cusparse', 'nvidia.nccl',
        'nvidia.nvjitlink', 'nvidia.nvtx',
        'sympy', 'networkx',
        'tokenizers',
        # ocrmypdf and its heavy deps are not needed in the bundle –
        # the app imports ocrmypdf lazily (try/except ImportError) and
        # falls back to the CLI tool at runtime.
        'ocrmypdf', 'pikepdf', 'pi_heif',
        'pdfminer', 'fpdf', 'pluggy',
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
    [],
    exclude_binaries=True,
    name='toms_super_simple_pdf_anonymizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # windowed mode – no console
    icon=None,              # Add an .ico file path here if desired
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='toms_super_simple_pdf_anonymizer',
)
