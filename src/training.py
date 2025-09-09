"""
Zweck: Modelltraining (Keras) auf Basis der extrahierten STFT-Features.
"""

from os.path import join
import random
import numpy as np
import tensorflow as tf
from keras import layers, models, regularizers


BACKGROUND_LABEL = "_background"
OTHER_LABEL = "_other"


def build_keyword_model(num_classes: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.InputLayer(input_shape=(16, 40, 1)),
        layers.Reshape(target_shape=(16, 40, 1)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


def load_feature_sets(features_root: str):
    from os import listdir
    labels = []
    x_all = []
    y_all = []
    acceptable_versions = [0.1]
    for i, fname in enumerate(listdir(features_root)):
        if not fname.endswith('.npz'):
            continue
        npz = np.load(join(features_root, fname))
        if float(npz['version']) not in acceptable_versions:
            raise RuntimeError(f"Feature-Version nicht kompatibel: {fname}")
        labels.append(fname.replace('.npz', ''))
        x_all.append(npz['samples'])
        num = npz['samples'].shape[0]
        y_all.append([i] * num)
    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    return labels, x, y


def train_from_features(features_root: str,
                        out_keras_path: str,
                        val_ratio: float = 0.2,
                        test_ratio: float = 0.2,
                        trials: int = 5,
                        epochs: int = 100,
                        batch_size: int = 32,
                        learning_rate: float = 1e-3):
    labels, x_all, y_all = load_feature_sets(features_root)

    other_idx = labels.index(OTHER_LABEL)
    bg_idx = labels.index(BACKGROUND_LABEL)

    # Shuffle
    zipped = list(zip(x_all, y_all))
    random.shuffle(zipped)
    x_all, y_all = zip(*zipped)
    x_all = np.asarray(x_all)
    y_all = np.asarray(y_all)

    # Split
    val_size = int(x_all.shape[0] * val_ratio)
    test_size = int(x_all.shape[0] * test_ratio)
    x_val = x_all[:val_size]
    y_val = y_all[:val_size]
    x_test = x_all[val_size:val_size + test_size]
    y_test = y_all[val_size:val_size + test_size]
    x_train = x_all[val_size + test_size:]
    y_train = y_all[val_size + test_size:]

    # Bestes Modell via F1 auf Val-Daten w4hlen
    best_model = None
    best_f1 = -1.0
    trial_summaries = []
    for t in range(trials):
        model = build_keyword_model(num_classes=len(labels))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(x_val, y_val))
        preds = model.predict(x_val, verbose=0)
        y_hat = np.argmax(preds, axis=1)
        cm_val = _confusion_matrix(y_val, y_hat, num_classes=len(labels))
        target_idxs = [i for i in range(len(labels)) if i not in (other_idx, bg_idx)]
        f1 = _avg_f1_from_cm(cm_val, target_idxs=target_idxs)
        # durchschnittliche FPR/FNR über Zielklassen
        fpr_avg = float(np.mean([_fpr_fnr_from_cm(cm_val, i)[0] for i in target_idxs])) if target_idxs else 0.0
        fnr_avg = float(np.mean([_fpr_fnr_from_cm(cm_val, i)[1] for i in target_idxs])) if target_idxs else 0.0
        trial_summaries.append({
            'trial': t + 1,
            'val_f1_avg': float(f1),
            'val_fpr_avg': fpr_avg,
            'val_fnr_avg': fnr_avg,
        })
        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    # Neu trainieren auf Train+Val
    x_train_val = np.concatenate([x_train, x_val])
    y_train_val = np.concatenate([y_train, y_val])
    best_model.fit(x_train_val, y_train_val, epochs=epochs, batch_size=batch_size, verbose=0)

    # Testreport (optional)
    _ = best_model.evaluate(x_test, y_test, verbose=0)

    # Speichern (.keras)
    out_keras_path = out_keras_path.replace('.h5', '.keras')
    best_model.save(out_keras_path)
    # Test-Report
    preds_test = best_model.predict(x_test, verbose=0)
    y_hat_test = np.argmax(preds_test, axis=1)
    cm_test = _confusion_matrix(y_test, y_hat_test, num_classes=len(labels))
    f1_per_label = {}
    fpr_per_label = {}
    fnr_per_label = {}
    support_per_label = {}
    for idx, label in enumerate(labels):
        f1_per_label[label] = float(_f1_from_cm(cm_test, idx))
        fpr, fnr = _fpr_fnr_from_cm(cm_test, idx)
        fpr_per_label[label] = float(fpr)
        fnr_per_label[label] = float(fnr)
        support_per_label[label] = int(cm_test[idx, :].sum())
    target_idxs = [i for i in range(len(labels)) if i not in (other_idx, bg_idx)]
    test_avg = {
        'f1_avg_target': float(np.mean([f1_per_label[labels[i]] for i in target_idxs])) if target_idxs else 0.0,
        'fpr_avg_target': float(np.mean([fpr_per_label[labels[i]] for i in target_idxs])) if target_idxs else 0.0,
        'fnr_avg_target': float(np.mean([fnr_per_label[labels[i]] for i in target_idxs])) if target_idxs else 0.0,
    }
    report = {
        'labels': labels,
        'trials': trial_summaries,
        'test': {
            'confusion_matrix': cm_test.tolist(),
            'f1_per_label': f1_per_label,
            'fpr_per_label': fpr_per_label,
            'fnr_per_label': fnr_per_label,
            'support_per_label': support_per_label,
            'averages': test_avg,
        }
    }
    return out_keras_path, report


def _avg_f1_from_cm(cm, target_idxs):
    f1s = []
    for idx in target_idxs:
        f1s.append(_f1_from_cm(cm, idx))
    return float(np.mean(f1s)) if f1s else 0.0


def _confusion_matrix(true, pred, num_classes: int | None = None):
    y_true = np.asarray(true).astype(np.int32).ravel()
    y_pred = np.asarray(pred).astype(np.int32).ravel()
    k = num_classes if num_classes is not None else (int(max(y_true.max(initial=0), y_pred.max(initial=0))) + 1 if y_true.size else 1)
    cm = np.zeros((k, k), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < k and 0 <= p < k:
            cm[t, p] += 1
    return cm


def _f1_from_cm(cm, class_idx):
    if class_idx >= cm.shape[0]:
        return 0.0
    tp = cm[class_idx, class_idx]
    fp = np.sum(cm[:, class_idx]) - tp
    fn = np.sum(cm[class_idx, :]) - tp
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    return 0.0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)


def _fpr_fnr_from_cm(cm, class_idx):
    if class_idx >= cm.shape[0]:
        return 0.0, 0.0
    tp = cm[class_idx, class_idx]
    fp = np.sum(cm[:, class_idx]) - tp
    fn = np.sum(cm[class_idx, :]) - tp
    tn = np.sum(cm) - tp - fp - fn
    fpr = 0.0 if (fp + tn) == 0 else fp / (fp + tn)
    fnr = 0.0 if (fn + tp) == 0 else fn / (fn + tp)
    return float(fpr), float(fnr)


