"""Training script for the bird detection CNN on real dataset."""
import os
import json
import numpy as np
from bird_classifier_cnn import build_cnn
import tensorflow as tf
from config import (
    DATASET_DIR, CNN_MODEL_PATH, CNN_EPOCHS, CNN_BATCH_SIZE,
    CNN_EARLY_STOP_PATIENCE, CNN_LR_FACTOR, CNN_LR_PATIENCE, CNN_MIN_LR,
    SPEC_H, SPEC_W, CNN_RANDOM_STATE, CNN_TEST_SPLIT,
)


def load_dataset(data_dir=None):
    """Load pre-extracted mel-spectrograms from prepared dataset."""
    specs, labels = [], []
    for split in ["train", "val"]:
        path = os.path.join(data_dir, f"{split}.json")
        if not os.path.exists(path):
            print(f"{path} not found")
            continue
        with open(path) as f:
            records = json.load(f)
        for rec in records:
            spec_path = rec.get("spec") or rec.get("path")
            if not spec_path or not os.path.exists(spec_path):
                continue
            data = np.load(spec_path)
            if data.shape != (SPEC_H, SPEC_W):
                target = np.zeros((SPEC_H, SPEC_W))
                h = min(data.shape[0], SPEC_H)
                w = min(data.shape[1], SPEC_W)
                target[:h, :w] = data[:h, :w]
                data = target
            specs.append(data)
            labels.append(rec["label"])
    if not specs:
        return None, None
    return np.array(specs), np.array(labels)


def train(data_dir=None, model_path=None,
          epochs=None, batch_size=None):
    if data_dir is None:
        data_dir = DATASET_DIR
    if model_path is None:
        model_path = CNN_MODEL_PATH
    if epochs is None:
        epochs = CNN_EPOCHS
    if batch_size is None:
        batch_size = CNN_BATCH_SIZE

    specs, labels = load_dataset(data_dir)
    if specs is None or len(specs) < 4:
        print("Dataset too small, using synthetic fallback.")
        return None

    specs = specs[..., np.newaxis]
    print(f"Dataset: {len(specs)} samples, shape {specs.shape}")
    print(f"  Bird: {labels.sum()}, Noise: {len(labels) - labels.sum()}")

    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        specs, labels, test_size=CNN_TEST_SPLIT, stratify=labels, random_state=CNN_RANDOM_STATE
    )

    model = build_cnn(input_shape=(SPEC_H, SPEC_W))
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=CNN_EARLY_STOP_PATIENCE, restore_best_weights=True,
                                          monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True,
                                           monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(factor=CNN_LR_FACTOR, patience=CNN_LR_PATIENCE, min_lr=CNN_MIN_LR),
    ]

    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=epochs, batch_size=batch_size,
              callbacks=callbacks, verbose=1)

    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Final evaluation
    test_path = os.path.join(data_dir, "test.json")
    if os.path.exists(test_path):
        with open(test_path) as f:
            test_records = json.load(f)
        x_test, y_test = [], []
        for rec in test_records:
            spec_path = rec.get("spec") or rec.get("path")
            if spec_path and os.path.exists(spec_path):
                data = np.load(spec_path)
                if data.shape == (SPEC_H, SPEC_W):
                    x_test.append(data)
                    y_test.append(rec["label"])
        if x_test:
            x_test = np.array(x_test)[..., np.newaxis]
            y_test = np.array(y_test)
            loss, acc = model.evaluate(x_test, y_test, verbose=0)
            print(f"Test accuracy: {acc*100:.1f}%")
    
    return model


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    train(data_dir=data_dir)
