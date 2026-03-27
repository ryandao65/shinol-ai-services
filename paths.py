"""
Central paths for AI services. Output root can be overridden with SHINOL_AI_SERVICES_OUTPUT
(set by the desktop app when spawning the bundled API).
"""
from __future__ import annotations

import os
import sys


def package_root() -> str:
    """Directory containing main.py / resources (next to exe when frozen with PyInstaller onedir)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def get_output_root() -> str:
    env = os.environ.get("SHINOL_AI_SERVICES_OUTPUT", "").strip()
    if env:
        p = os.path.normpath(os.path.abspath(env))
        os.makedirs(p, exist_ok=True)
        return p
    p = os.path.join(package_root(), "output")
    os.makedirs(p, exist_ok=True)
    return p


def get_preview_voice_samples_dir() -> str:
    """Cached «Nghe thử» WAVs per provider + voice (under same root as episodes)."""
    p = os.path.join(get_output_root(), "preview_voice_samples")
    os.makedirs(p, exist_ok=True)
    return p
