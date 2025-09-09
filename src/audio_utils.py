"""
Zweck: Audio I/O und Signalverarbeitung.
"""

from typing import Tuple
import numpy as np
import librosa
import soundfile as sf


def load_mono_resampled(path: str, sample_rate: int) -> Tuple[np.ndarray, int]:
    waveform, fs = librosa.load(path, sr=sample_rate, mono=True)
    return waveform, sample_rate


def pad_or_trim(waveform: np.ndarray, target_len: int) -> np.ndarray:
    if len(waveform) < target_len:
        pad = np.zeros(target_len - len(waveform))
        return np.concatenate([waveform, pad])
    return waveform[:target_len]


def save_wav(path: str, waveform: np.ndarray, sample_rate: int, subtype: str = "PCM_16") -> None:
    sf.write(path, waveform, sample_rate, subtype=subtype)


def is_audio_file(filename: str) -> bool:
    allowed = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
    return filename.lower().endswith(allowed)


