"""Train lightweight MLP for confidence stacking."""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_mlp(input_dim=4, hidden_dim=8, num_classes=1):
    """Build lightweight MLP for stacking BirdNET and NN outputs."""
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = keras.layers.Dense(hidden_dim, activation="relu")(inputs)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_mlp(data_path="real_dataset/mlp_training_data.csv", model_path="mlp_stacker.h5",
              epochs=50, batch_size=16):
    """Train the MLP stacker on prepared data."""
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Generating synthetic data...")
        from prepare_mlp_data import generate_mlp_data
        data_path = generate_mlp_data()

    import csv
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    rows_trimmed = [r[:5] for r in rows]
    data = np.array(rows_trimmed, dtype=np.float32)
    X = data[:, :4]
    y = data[:, 4]

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_mlp(input_dim=X.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(model_path)
    print(f"MLP stacker saved to {model_path}")
    return history


if __name__ == "__main__":
    train_mlp()
