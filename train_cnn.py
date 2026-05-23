"""Training script for the bird detection CNN on real dataset."""
import os
import json
import numpy as np
from bird_classifier_cnn import build_cnn
import tensorflow as tf


def load_dataset(data_dir="real_dataset"):
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
            # Ensure shape (128, 259)
            if data.shape != (128, 259):
                import librosa
                target = np.zeros((128, 259))
                h = min(data.shape[0], 128)
                w = min(data.shape[1], 259)
                target[:h, :w] = data[:h, :w]
                data = target
            specs.append(data)
            labels.append(rec["label"])
    if not specs:
        return None, None
    return np.array(specs), np.array(labels)


def train(data_dir="real_dataset", model_path="bird_sound_classifier.h5",
          epochs=50, batch_size=8):
    specs, labels = load_dataset(data_dir)
    if specs is None or len(specs) < 4:
        print("Dataset too small, using synthetic fallback.")
        return None

    specs = specs[..., np.newaxis]
    print(f"Dataset: {len(specs)} samples, shape {specs.shape}")
    print(f"  Bird: {labels.sum()}, Noise: {len(labels) - labels.sum()}")

    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        specs, labels, test_size=0.2, stratify=labels, random_state=42
    )

    model = build_cnn(input_shape=(128, 259))
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                          monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True,
                                           monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
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
                if data.shape == (128, 259):
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
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "real_dataset"
    train(data_dir=data_dir)
