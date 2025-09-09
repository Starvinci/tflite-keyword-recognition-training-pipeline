"""
Einfache Inferenztests fuer Keras und TFLite Modelle.
"""

from os.path import join, exists


def test_keras_model_loads():
    model_path = join('export', 'models', 'model.keras')
    if not exists(model_path):
        return
    try:
        import tensorflow as tf
    except ImportError:
        return
    try:
        import numpy as np
    except ImportError:
        return
    model = tf.keras.models.load_model(model_path)
    # Dummy-Eingabe entsprechend (16,40,1)
    x = np.zeros((1, 16, 40, 1), dtype=np.float32)
    _ = model.predict(x, verbose=0)


def test_tflite_model_loads():
    tflite_path = join('export', 'models', 'model.tflite')
    if not exists(tflite_path):
        return
    try:
        import tensorflow as tf
    except ImportError:
        return
    try:
        import numpy as np
    except ImportError:
        return
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    x = np.zeros(input_details[0]['shape'], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])


def test_h5_model_loads():
    model_path = join('export', 'models', 'model.h5')
    if not exists(model_path):
        return
    try:
        import tensorflow as tf
    except ImportError:
        return
    try:
        import numpy as np
    except ImportError:
        return
    model = tf.keras.models.load_model(model_path)
    x = np.zeros((1, 16, 40, 1), dtype=np.float32)
    _ = model.predict(x, verbose=0)

