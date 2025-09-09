"""
Zweck: Datenkuratierung aus `data/keyword` und `data/hintergrund` nach
`export/curated`. Begrenzung pro Klasse moeglich.
"""

import random
import shutil
from os.path import join
from typing import Iterable

from . import paths
from .fs_utils import clear_and_make, list_dirnames, list_audio_filenames
from rich.progress import Progress, BarColumn, TimeElapsedColumn, MofNCompleteColumn, TextColumn


def curate_sources(source_dirs: Iterable[str], out_dir: str, max_per_class: int = 0) -> None:
    clear_and_make(out_dir)

    # Sammle Klassen aus allen Quelldirs
    classes = []
    for src in source_dirs:
        classes.extend(list_dirnames(src))
    classes = list(dict.fromkeys(classes))

    for cls in classes:
        # Alle Dateien ueber alle Quellen fuer diese Klasse einsammeln
        file_paths = []
        for src in source_dirs:
            class_dir = join(src, cls)
            try:
                for fname in list_audio_filenames(class_dir):
                    file_paths.append(join(class_dir, fname))
            except FileNotFoundError:
                continue

        if not file_paths:
            continue

        random.shuffle(file_paths)
        if max_per_class and max_per_class > 0:
            file_paths = file_paths[:max_per_class]

        # Ziel-Unterordner erstellen und Dateien durchnummeriert kopieren
        dst_dir = join(out_dir, cls)
        clear_and_make(dst_dir)
        pad = len(str(len(file_paths)))

        # Fortschrittsbalken pro Datei
        with Progress(
            TextColumn(f"[bold blue]Kuratieren:[/] {cls}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("copy", total=len(file_paths))
            for idx, src_path in enumerate(file_paths):
                dst_path = join(dst_dir, f"{str(idx).zfill(pad)}.wav")
                shutil.copy(src_path, dst_path)
                progress.advance(task)


def curate_default(max_per_class: int = 0) -> None:
    sources = [paths.keyword_dir()]
    # Hintergrund ist kein Label-Ordner, sondern wird in Mixer genutzt; hier optional
    out_dir = paths.curated_dir()
    curate_sources(sources, out_dir, max_per_class=max_per_class)


