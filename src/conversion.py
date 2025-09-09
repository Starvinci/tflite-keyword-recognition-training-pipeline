"""
Zweck: Keras -> TFLite Konvertierung und optionaler C-Header Export.
"""

from os.path import exists
from os import remove
import numpy as np
import tensorflow as tf


def keras_to_tflite(keras_path: str, tflite_path: str) -> str:
    model = tf.keras.models.load_model(keras_path)
    # BatchNorm fix: Inferenz-Mode sicherstellen
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.trainable = False
    # Exportieren als SavedModel fuer Converter
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        model.export(tmp)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]
        tflite_model = converter.convert()
    finally:
        shutil.rmtree(tmp)

    if exists(tflite_path):
        remove(tflite_path)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    return tflite_path


def tflite_to_c_header(tflite_path: str, header_path: str, array_name: str = "model") -> str:
    with open(tflite_path, 'rb') as f:
        blob = f.read()
    # In hex umwandeln
    hex_array = [format(b, '#04x') for b in blob]
    body = ", ".join(hex_array)
    header = (
        f"#ifndef {array_name.upper()}_H\n"
        f"#define {array_name.upper()}_H\n\n"
        f"#include <stdint.h>\n\n"
        f"const unsigned char {array_name}[] = {{ {body} }};\n"
        f"const unsigned int {array_name}_len = {len(blob)};\n\n"
        f"#endif\n"
    )
    if exists(header_path):
        remove(header_path)
    with open(header_path, 'w') as f:
        f.write(header)
    return header_path


