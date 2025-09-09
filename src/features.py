"""
Zweck: STFT-Feature-Extraktion aus `export/mixed` nach `export/features`.
"""

import math
import numpy as np
from os.path import join
from .fs_utils import list_dirnames, list_audio_filenames
from .audio_utils import load_mono_resampled, pad_or_trim
from rich.progress import Progress, BarColumn, TimeElapsedColumn, MofNCompleteColumn, TextColumn


def compute_stft_image(waveform: np.ndarray,
                       fs: int,
                       nfft: int,
                       hop: int,
                       min_bin: int,
                       max_bin: int,
                       avg_bins: int,
                       shift_bits: int) -> np.ndarray:
    hanning = np.hanning(nfft)
    num_slices = int(math.ceil(((len(waveform)) / hop) - (nfft / hop)) + 1)
    stft = np.zeros(((max_bin - min_bin) // avg_bins, num_slices))
    for i in range(stft.shape[1]):
        start = i * hop
        stop = start + nfft
        window = waveform[start:stop]
        if len(window) < nfft:
            window = np.concatenate([window, np.zeros(nfft - len(window))])
        window = hanning * window
        fft = np.abs(np.fft.rfft(window, n=nfft))
        fft = fft[min_bin:max_bin]
        fft = np.around(fft / nfft)
        fft = np.mean(fft.reshape(-1, avg_bins), axis=1)
        fft = np.around(fft / (2 ** shift_bits))
        fft = np.clip(fft, 0, 255)
        stft[:, i] = fft
    return stft


def extract_features(mixed_root: str,
                     out_root: str,
                     sample_time_s: float = 1.0,
                     sample_rate: int = 8000,
                     nfft: int = 512,
                     hop: int = 400,
                     cutoff_freq: int = 4000,
                     avg_bins: int = 8,
                     shift_bits: int = 3,
                     version: float = 0.1) -> None:
    from os import makedirs
    from os.path import exists
    if not exists(out_root):
        makedirs(out_root)

    max_bin = int((nfft / 2) / ((sample_rate / 2) / cutoff_freq)) + 1
    min_bin = 1
    classes = list_dirnames(mixed_root)
    sample_len = int(sample_time_s * sample_rate)
    for cls in classes:
        cls_dir = join(mixed_root, cls)
        files = list_audio_filenames(cls_dir)
        samples = np.zeros((len(files), (max_bin - min_bin) // avg_bins, int(math.ceil(((sample_len) / hop) - (nfft / hop)) + 1)))
        with Progress(
            TextColumn(f"[bold cyan]Features:[/] {cls}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("feat", total=len(files))
            for i, fname in enumerate(files):
                waveform, _ = load_mono_resampled(join(cls_dir, fname), sample_rate)
                waveform = np.around(waveform * 32767)
                waveform = pad_or_trim(waveform, sample_len)
                stft = compute_stft_image(waveform, sample_rate, nfft, hop, min_bin, max_bin, avg_bins, shift_bits)
                samples[i] = stft
                progress.advance(task)
        np.savez(join(out_root, f"{cls}.npz"), version=version, samples=samples)


