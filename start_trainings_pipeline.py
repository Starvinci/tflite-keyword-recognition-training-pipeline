#!/usr/bin/env python3
"""
Training-Pipeline fuer Keyword-Spotting.

Schritte:
1) Daten kuratieren: data/keyword -> export/curated
2) Mischen mit Hintergrund: data/hintergrund + curated -> export/mixed
3) Features extrahieren: export/mixed -> export/features
4) Trainieren: export/features -> export/models/model.keras
5) Konvertieren: .keras -> .tflite (+ optional .h)
"""

import argparse
from os.path import join

from src import paths
from src.fs_utils import ensure_dirs
from src.curation import curate_default
from src.mixer import build_mixed_dataset
from src.features import extract_features
from src.training import train_from_features
from rich.console import Console
from rich.table import Table
from src.conversion import keras_to_tflite, tflite_to_c_header


def main():
    parser = argparse.ArgumentParser(description="Keyword-Spotting Training-Pipeline")
    parser.add_argument('--targets', type=str, required=True, help="Kommagetrennte Zielwoerter, z.B. 'lome,notfall'")
    parser.add_argument('--max_per_class', type=int, default=0, help="Limit pro Klasse in Kuratierung (0 = alle)")
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--sample_time', type=float, default=1.0)
    parser.add_argument('--bg_snippets_per_file', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--export_header', action='store_true')
    args = parser.parse_args()

    ensure_dirs([paths.export_dir(), paths.models_dir(), paths.curated_dir(), paths.mixed_dir(), paths.features_dir()])

    # 1) Kuratieren
    curate_default(max_per_class=args.max_per_class)

    # 2) Mischen
    targets = [t.strip() for t in args.targets.split(',') if t.strip()]
    build_mixed_dataset(curated_dir=paths.curated_dir(),
                        background_root=paths.background_dir(),
                        out_dir=paths.mixed_dir(),
                        targets=targets,
                        sample_rate=args.sample_rate,
                        sample_time_s=args.sample_time,
                        bg_snippets_per_file=args.bg_snippets_per_file)

    # 3) Features
    extract_features(mixed_root=paths.mixed_dir(), out_root=paths.features_dir(), sample_time_s=args.sample_time, sample_rate=8000)

    # 4) Train
    console = Console()
    keras_path = join(paths.models_dir(), 'model.keras')
    keras_path, report = train_from_features(paths.features_dir(), keras_path, epochs=args.epochs, trials=args.trials)

    # Trainings-Report (Trials)
    trials_table = Table(title="Validation-Ergebnisse pro Trial")
    trials_table.add_column("Trial", justify="right")
    trials_table.add_column("F1 (avg targets)", justify="right")
    trials_table.add_column("FPR (avg targets)", justify="right")
    trials_table.add_column("FNR (avg targets)", justify="right")
    for t in report['trials']:
        trials_table.add_row(str(t['trial']), f"{t['val_f1_avg']:.3f}", f"{t['val_fpr_avg']:.3f}", f"{t['val_fnr_avg']:.3f}")
    console.print(trials_table)

    # 5) TFLite
    tflite_path = join(paths.models_dir(), 'model.tflite')
    keras_to_tflite(keras_path, tflite_path)

    # Test-Report (pro Label)
    per_label = Table(title="Testmetriken pro Klasse")
    per_label.add_column("Label")
    per_label.add_column("F1", justify="right")
    per_label.add_column("FPR", justify="right")
    per_label.add_column("FNR", justify="right")
    per_label.add_column("Support", justify="right")
    labels = report['labels']
    for lbl in labels:
        per_label.add_row(lbl,
                          f"{report['test']['f1_per_label'][lbl]:.3f}",
                          f"{report['test']['fpr_per_label'][lbl]:.3f}",
                          f"{report['test']['fnr_per_label'][lbl]:.3f}",
                          str(report['test']['support_per_label'][lbl]))
    console.print(per_label)

    avgs = report['test']['averages']
    console.print(f"[bold]Test-Durchschnitt (Targets):[/] F1={avgs['f1_avg_target']:.3f}  FPR={avgs['fpr_avg_target']:.3f}  FNR={avgs['fnr_avg_target']:.3f}")

    if args.export_header:
        header_path = join(paths.models_dir(), 'model.h')
        tflite_to_c_header(tflite_path, header_path, array_name='keyword_model')

    console.print("[green]Pipeline abgeschlossen.[/]")


if __name__ == '__main__':
    main()


