# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for bundling the FastAPI AI services as a Windows onedir next to shinol-ai-api.exe.
# Run from repo: pip install -r requirements.txt pyinstaller && pyinstaller shinol-ai-api.spec

import os

from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas = []
binaries = []
hiddenimports = ["paths"]

# Pull in runtime data/binaries for heavy framework packages
for pkg in (
    "uvicorn",
    "fastapi",
    "starlette",
    "pydantic",
    "multipart",
    "httpx",
    "httpcore",
    "anyio",
    "edge_tts",
    "numpy",
    "soundfile",
    "scipy",
    "PIL",
    "cv2",
    "requests",
    "charset_normalizer",
    "certifi",
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# Optional / lazy-loaded (image, kokoro, whisper) — include if installed
for pkg in ("torch", "diffusers", "transformers", "accelerate", "faster_whisper", "kokoro"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name="shinol-ai-api",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="shinol-ai-api",
)
