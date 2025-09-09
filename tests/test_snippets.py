import os
import tempfile
import numpy as np
import soundfile as sf
import importlib.util


def _write_tone(path: str, sr: int = 16000, seconds: float = 2.0):
    t = np.arange(int(sr * seconds)) / sr
    x = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, x, sr)


def _run_cut_snippets(in_file: str, out_dir: str):
    # Importiere Skript als Modul und rufe main() mit Args auf
    import sys
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("cut_snippets", os.path.join(os.getcwd(), "scripts", "cut_snippets.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    # Simuliere argparse Namespace
    args = SimpleNamespace(in_file=in_file, out_dir=out_dir, sample_rate=16000, sample_time=1.0)
    # Patch parser
    def fake_parse_args():
        return args
    mod.argparse.ArgumentParser.parse_args = lambda self: fake_parse_args()  # type: ignore
    mod.main()


def test_cut_snippets_creates_one_second_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        in_wav = os.path.join(tmp, "in.wav")
        out_dir = os.path.join(tmp, "out")
        _write_tone(in_wav, sr=16000, seconds=2.0)
        _run_cut_snippets(in_wav, out_dir)
        files = sorted([f for f in os.listdir(out_dir) if f.endswith('.wav')])
        assert len(files) == 2
        # Prüfe Länge jeder Datei grob
        for f in files:
            y, sr = sf.read(os.path.join(out_dir, f))
            assert abs(len(y) - 16000) <= 5


