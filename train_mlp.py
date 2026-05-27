"""Train lightweight MLP for confidence stacking with architecture sweep."""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import (
    MLP_HIDDEN_DIM, MLP_L2_REG, MLP_DROPOUT, MLP_EPOCHS, MLP_BATCH_SIZE,
    MLP_USE_CLASS_WEIGHT, MLP_EARLY_STOP_PATIENCE, MLP_RANDOM_STATE,
    MLP_TEST_SPLIT, FP_PER_HOUR_DIVISOR, EVAL_THRESHOLDS,
    MLP_TRAINING_DATA, MLP_MODEL_PATH, MLP_SWEEP_CONFIGS,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")


def build_mlp(input_dim=4, hidden_dim=None, num_classes=1, l2_reg=None, dropout=None):
    if hidden_dim is None:
        hidden_dim = MLP_HIDDEN_DIM
    if l2_reg is None:
        l2_reg = MLP_L2_REG
    if dropout is None:
        dropout = MLP_DROPOUT
    """Build MLP with configurable architecture and L2 regularization."""
    reg = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = keras.layers.Dense(hidden_dim, activation="relu", kernel_regularizer=reg)(inputs)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation="sigmoid", kernel_regularizer=reg)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weight(y):
    """Compute balanced class weights from labels."""
    n = len(y)
    n_pos = int(np.sum(y))
    n_neg = n - n_pos
    w_pos = n / (2.0 * n_pos) if n_pos > 0 else 1.0
    w_neg = n / (2.0 * n_neg) if n_neg > 0 else 1.0
    return {0: w_neg, 1: w_pos}


def train_mlp(data_path=None, model_path=None,
              epochs=None, batch_size=None, hidden_dim=None, l2_reg=None, dropout=None,
              use_class_weight=None):
    """Train the MLP stacker on prepared data."""
    if data_path is None:
        data_path = MLP_TRAINING_DATA
    if model_path is None:
        model_path = MLP_MODEL_PATH
    if epochs is None:
        epochs = MLP_EPOCHS
    if batch_size is None:
        batch_size = MLP_BATCH_SIZE
    if hidden_dim is None:
        hidden_dim = MLP_HIDDEN_DIM
    if l2_reg is None:
        l2_reg = MLP_L2_REG
    if dropout is None:
        dropout = MLP_DROPOUT
    if use_class_weight is None:
        use_class_weight = MLP_USE_CLASS_WEIGHT

    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found.")
        return None

    import csv
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    rows_trimmed = [r[:5] for r in rows]
    data = np.array(rows_trimmed, dtype=np.float32)
    X = data[:, :4]
    y = data[:, 4]

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=MLP_TEST_SPLIT, random_state=MLP_RANDOM_STATE)
    for train_idx, val_idx in sss.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

    model = build_mlp(input_dim=X.shape[1], hidden_dim=hidden_dim, l2_reg=l2_reg, dropout=dropout)

    class_weight = compute_class_weight(y_train) if use_class_weight else None

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=MLP_EARLY_STOP_PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0,
    )

    model.save(model_path)

    # Evaluate
    y_prob = model.predict(X_val, verbose=0).flatten()
    for thresh in EVAL_THRESHOLDS[:3]:  # 0.35, 0.50, 0.65
        y_pred = (y_prob > thresh).astype(int)
        tp = int(np.sum((y_val == 1) & (y_pred == 1)))
        fp = int(np.sum((y_val == 0) & (y_pred == 1)))
        fn = int(np.sum((y_val == 1) & (y_pred == 0)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        fp_hr = fp / (len(y_val) / FP_PER_HOUR_DIVISOR)
        print(f"  thresh={thresh:.2f}: F1={f1:.4f} P={prec:.4f} R={rec:.4f} FP/hr={fp_hr:.2f}")

    print(f"MLP stacker saved to {model_path} (hidden_dim={hidden_dim}, l2={l2_reg}, dropout={dropout})")
    return model


def sweep_architectures():
    """Try different architectures and report results."""
    results = []
    for hd, l2, drop, label in MLP_SWEEP_CONFIGS:
        print(f"\n=== Sweep: {label} ===")
        path = f"mlp_stacker_{hd}_{l2}.h5"
        train_mlp(data_path=MLP_TRAINING_DATA, model_path=path,
                  hidden_dim=hd, l2_reg=l2, dropout=drop, epochs=100, batch_size=MLP_BATCH_SIZE)
        results.append((label, path, hd, l2, drop))

    print("\n\n=== Architecture Sweep Complete ===")
    for label, path, hd, l2, drop in results:
        print(f"  {label}: {path}")


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep_architectures()
    else:
        train_mlp()
