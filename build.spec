# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for PDF Anonymizer.
Usage:  pyinstaller build.spec
"""

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
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
