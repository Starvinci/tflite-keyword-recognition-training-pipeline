"""
Zweck: Datei- und Ordner-Hilfsfunktionen.
"""

import os
from os.path import exists, join
from typing import Iterable


def ensure_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        if not exists(p):
            os.makedirs(p, exist_ok=True)


def clear_and_make(path: str) -> None:
    if exists(path):
        # Sicheres Entfernen und Neu-Erstellen
        import shutil
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def list_dirnames(path: str) -> list[str]:
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def list_filenames(path: str) -> list[str]:
    return [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]


def list_audio_filenames(path: str) -> list[str]:
    from .audio_utils import is_audio_file
    files = []
    for name in os.listdir(path):
        full = join(path, name)
        if os.path.isfile(full) and is_audio_file(name):
            files.append(name)
    return files


