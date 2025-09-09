import os
import tempfile
import numpy as np
import soundfile as sf

from src import paths
from src.curation import curate_sources
from src.mixer import build_mixed_dataset
from src.features import extract_features


def _sine(sr=16000, seconds=1.0, freq=440):
    t = np.arange(int(sr * seconds)) / sr
    return 0.1 * np.sin(2 * np.pi * freq * t)


def test_curation_mixing_features_e2e_small():
    with tempfile.TemporaryDirectory() as tmp:
        # Erzeuge Rohdaten
        kw_root = os.path.join(tmp, "data", "keyword")
        bg_root = os.path.join(tmp, "data", "hintergrund")
        os.makedirs(kw_root, exist_ok=True)
        os.makedirs(bg_root, exist_ok=True)
        os.makedirs(os.path.join(kw_root, "foo"), exist_ok=True)
        os.makedirs(os.path.join(kw_root, "bar"), exist_ok=True)
        sf.write(os.path.join(kw_root, "foo", "a.wav"), _sine(16000, 1.0, 440), 16000)
        sf.write(os.path.join(kw_root, "bar", "b.wav"), _sine(16000, 1.0, 550), 16000)
        sf.write(os.path.join(bg_root, "bg.wav"), _sine(16000, 2.0, 220), 16000)

        # Kuratieren -> export/curated
        curated_dir = os.path.join(tmp, "export", "curated")
        curate_sources([kw_root], curated_dir, max_per_class=1)
        assert os.path.isdir(os.path.join(curated_dir, "foo"))
        assert os.path.isdir(os.path.join(curated_dir, "bar"))

        # Mixen -> export/mixed
        mixed_dir = os.path.join(tmp, "export", "mixed")
        build_mixed_dataset(curated_dir, bg_root, mixed_dir, targets=["foo"], sample_rate=16000, sample_time_s=1.0, bg_snippets_per_file=1)
        assert os.path.isdir(os.path.join(mixed_dir, "_background"))
        assert os.path.isdir(os.path.join(mixed_dir, "foo"))
        assert os.path.isdir(os.path.join(mixed_dir, "_other"))

        # Features -> export/features
        feat_dir = os.path.join(tmp, "export", "features")
        extract_features(mixed_dir, feat_dir, sample_time_s=1.0, sample_rate=8000)
        assert os.path.isfile(os.path.join(feat_dir, "foo.npz"))
        assert os.path.isfile(os.path.join(feat_dir, "_other.npz"))
        assert os.path.isfile(os.path.join(feat_dir, "_background.npz"))


