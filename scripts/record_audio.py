#!/usr/bin/env python3
"""
Einfache Audioaufnahme per Sounddevice, speichert als WAV.
"""

import argparse
import sounddevice as sd
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file', required=True, help='Ziel-WAV-Datei')
    parser.add_argument('--duration', type=float, default=3.0)
    parser.add_argument('--sample_rate', type=int, default=16000)
    args = parser.parse_args()

    print(f"Aufnahme startet ({args.duration}s @ {args.sample_rate}Hz)...")
    recording = sd.rec(int(args.duration * args.sample_rate), samplerate=args.sample_rate, channels=1, dtype='float32')
    sd.wait()
    sf.write(args.out_file, recording, args.sample_rate)
    print("Gespeichert:", args.out_file)


if __name__ == '__main__':
    main()


