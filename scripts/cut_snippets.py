#!/usr/bin/env python3
"""
Schneidet WAVs in 1s-Snippets und legt sie in Zielordnern ab.
"""

import argparse
from os.path import join
from os import makedirs
import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', required=True, help='Pfad zu einer WAV-Datei')
    parser.add_argument('--out_dir', required=True, help='Zielordner fr Snippets')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--sample_time', type=float, default=1.0)
    args = parser.parse_args()

    makedirs(args.out_dir, exist_ok=True)
    data, fs = sf.read(args.in_file)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    if fs != args.sample_rate:
        import librosa
        data = librosa.resample(y=data, orig_sr=fs, target_sr=args.sample_rate)
        fs = args.sample_rate

    step = int(fs * args.sample_time)
    count = len(data) // step
    pad = len(str(max(1, count)))
    for i in range(count):
        start = i * step
        stop = start + step
        snippet = data[start:stop]
        sf.write(join(args.out_dir, f"{str(i).zfill(pad)}.wav"), snippet, fs)


if __name__ == '__main__':
    main()


