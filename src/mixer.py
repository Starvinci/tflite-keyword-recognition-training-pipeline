"""
Zweck: Mischung von Sprach-Samples mit Hintergrundgeraeuschen zu
augmentierten Trainingsdaten (export/mixed).
"""

import random
from os.path import join
from typing import Iterable

import numpy as np

from .fs_utils import clear_and_make, list_dirnames, list_audio_filenames
from .audio_utils import load_mono_resampled, pad_or_trim, save_wav
from rich.progress import Progress, BarColumn, TimeElapsedColumn, MofNCompleteColumn, TextColumn


def _mix_waveforms(word: np.ndarray,
                   bg: np.ndarray,
                   sample_len: int,
                   word_gain: float,
                   bg_gain: float) -> np.ndarray:
    if len(bg) >= sample_len:
        start = random.randint(0, len(bg) - sample_len)
        bg_slice = bg[start:start + sample_len]
    else:
        reps = int(np.ceil(sample_len / max(1, len(bg))))
        bg_slice = np.tile(bg, reps)[:sample_len]

    word = pad_or_trim(word, sample_len)
    mix = 0.5 * word_gain * word + 0.5 * bg_gain * bg_slice
    return mix


def build_mixed_dataset(curated_dir: str,
                        background_root: str,
                        out_dir: str,
                        targets: Iterable[str],
                        sample_rate: int = 16000,
                        sample_time_s: float = 1.0,
                        word_gain: float = 1.0,
                        bg_gain: float = 0.8,
                        bg_snippets_per_file: int = 5,
                        bit_depth: str = "PCM_16") -> None:
    clear_and_make(out_dir)

    # Hintergrund-Snippets
    bg_out = join(out_dir, "_background")
    clear_and_make(bg_out)

    bg_files = list_audio_filenames(background_root)
    sample_len = int(sample_rate * sample_time_s)
    total_bg = len(bg_files) * bg_snippets_per_file
    pad = len(str(max(1, total_bg)))
    counter = 0
    with Progress(
        TextColumn("[bold blue]Hintergrund-Snippets:[/]") ,
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("bg", total=total_bg)
        for bg_file in bg_files:
            bg_wave, _ = load_mono_resampled(join(background_root, bg_file), sample_rate)
            for _ in range(bg_snippets_per_file):
                snippet = _mix_waveforms(np.zeros(1), bg_wave, sample_len, 0.0, 1.0)
                save_wav(join(bg_out, f"{str(counter).zfill(pad)}.wav"), snippet, sample_rate, subtype=bit_depth)
                counter += 1
                progress.advance(task)

    # Targets
    curated_classes = list_dirnames(curated_dir)
    others = [c for c in curated_classes if c not in targets]

    for target in targets:
        target_src = join(curated_dir, target)
        target_dst = join(out_dir, target)
        clear_and_make(target_dst)
        word_files = list_audio_filenames(target_src)
        bg_files = list_audio_filenames(background_root)
        pad = len(str(max(1, len(word_files) * len(bg_files))))
        idx = 0
        with Progress(
            TextColumn(f"[bold magenta]Mixing Target:[/] {target}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("mix_target", total=len(word_files) * max(1, len(bg_files)))
            for wf in word_files:
                wv, _ = load_mono_resampled(join(target_src, wf), sample_rate)
                for bf in bg_files:
                    bg, _ = load_mono_resampled(join(background_root, bf), sample_rate)
                    mixed = _mix_waveforms(wv, bg, sample_len, word_gain, bg_gain)
                    save_wav(join(target_dst, f"{str(idx).zfill(pad)}.wav"), mixed, sample_rate, subtype=bit_depth)
                    idx += 1
                    progress.advance(task)

    # Other
    other_dst = join(out_dir, "_other")
    clear_and_make(other_dst)
    bg_files = list_audio_filenames(background_root)
    idx = 0
    total_other = sum(len(list_audio_filenames(join(curated_dir, o))) for o in others) * max(1, len(bg_files))
    pad_other = len(str(max(1, total_other)))
    with Progress(
        TextColumn("[bold yellow]Mixing Other:[/]") ,
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        total_other_work = total_other
        task = progress.add_task("mix_other", total=total_other_work)
        for cls in others:
            cls_src = join(curated_dir, cls)
            for wf in list_audio_filenames(cls_src):
                wv, _ = load_mono_resampled(join(cls_src, wf), sample_rate)
                for bf in bg_files:
                    bg, _ = load_mono_resampled(join(background_root, bf), sample_rate)
                    mixed = _mix_waveforms(wv, bg, sample_len, word_gain, bg_gain)
                    save_wav(join(other_dst, f"{str(idx).zfill(pad_other)}.wav"), mixed, sample_rate, subtype=bit_depth)
                    idx += 1
                    progress.advance(task)


