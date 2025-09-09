"""
Zweck: Zentrale Pfadverwaltung fuer die Keyword-Spotting-Pipeline.
"""

from os.path import join, dirname, abspath


def project_root() -> str:
    return dirname(dirname(abspath(__file__)))


def data_dir() -> str:
    return join(project_root(), "data")


def keyword_dir() -> str:
    return join(data_dir(), "keyword")


def background_dir() -> str:
    return join(data_dir(), "hintergrund")


def export_dir() -> str:
    return join(project_root(), "export")


def curated_dir() -> str:
    return join(export_dir(), "curated")


def mixed_dir() -> str:
    return join(export_dir(), "mixed")


def features_dir() -> str:
    return join(export_dir(), "features")


def models_dir() -> str:
    return join(export_dir(), "models")


