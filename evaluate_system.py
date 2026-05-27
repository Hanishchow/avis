"""End-to-end evaluation with configurable MLP architecture and threshold grid search."""
import os, sys, json, csv
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MLP_HIDDEN_DIM, MLP_L2_REG, MLP_DROPOUT, MLP_USE_CLASS_WEIGHT,
    CV_FOLDS, FP_PER_HOUR_DIVISOR, EVAL_THRESHOLDS, SWEEP_THRESHOLDS,
    MLP_TRAINING_DATA, MLP_BATCH_SIZE, MLP_EPOCHS, MLP_EARLY_STOP_PATIENCE,
    EVAL_SWEEP_CONFIGS, SC007_TARGET_F1, SC008_TARGET_FP_HR, MLP_RANDOM_STATE,
)
from filtering_pipeline import full_filter_pipeline
from model_inference import BirdDetector
from birdnet_integration import BirdNETProcessor


def soft_f1(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return f1, prec, rec, fp, fn, tp


def compute_class_weight(y):
    n = len(y)
    n_pos = int(np.sum(y))
    n_neg = n - n_pos
    w_pos = n / (2.0 * n_pos) if n_pos > 0 else 1.0
    w_neg = n / (2.0 * n_neg) if n_neg > 0 else 1.0
    return {0: w_neg, 1: w_pos}


def build_mlp(input_dim=4, hidden_dim=None, l2_reg=None, dropout=None):
    if hidden_dim is None:
        hidden_dim = MLP_HIDDEN_DIM
    if l2_reg is None:
        l2_reg = MLP_L2_REG
    if dropout is None:
        dropout = MLP_DROPOUT
    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(hidden_dim, activation="relu", kernel_regularizer=reg)(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=reg)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


def evaluate(test_csv=None,
             thresholds=None, cv_folds=None, hidden_dim=None, l2_reg=None, dropout=None,
             use_class_weight=None):
    if test_csv is None:
        test_csv = MLP_TRAINING_DATA
    if thresholds is None:
        thresholds = EVAL_THRESHOLDS
    if cv_folds is None:
        cv_folds = CV_FOLDS
    if hidden_dim is None:
        hidden_dim = MLP_HIDDEN_DIM
    if l2_reg is None:
        l2_reg = MLP_L2_REG
    if dropout is None:
        dropout = MLP_DROPOUT
    if use_class_weight is None:
        use_class_weight = MLP_USE_CLASS_WEIGHT

    with open(test_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    rows_trimmed = [r[:5] for r in rows]
    data = np.array(rows_trimmed, dtype=np.float32)
    X = data[:, :4]
    y_true = data[:, 4].astype(int)
    species = [r[5] if len(r) > 5 else "?" for r in rows]

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=MLP_RANDOM_STATE)
    all_y_true = []
    all_y_prob = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_true)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_true[train_idx], y_true[val_idx]
        model = build_mlp(hidden_dim=hidden_dim, l2_reg=l2_reg, dropout=dropout)
        cw = compute_class_weight(y_train) if use_class_weight else None
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=MLP_EPOCHS, batch_size=MLP_BATCH_SIZE, verbose=0,
                  class_weight=cw,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=MLP_EARLY_STOP_PATIENCE, restore_best_weights=True)])
        y_prob = model.predict(X_val, verbose=0).flatten()
        all_y_true.extend(y_val)
        all_y_prob.extend(y_prob)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    hours = len(all_y_true) / FP_PER_HOUR_DIVISOR

    n_bird = int(np.sum(all_y_true))
    n_noise = int(len(all_y_true) - n_bird)

    print(f"\n{'='*60}")
    print(f"MLP: hidden_dim={hidden_dim}, l2={l2_reg}, dropout={dropout}, class_weight={use_class_weight}")
    print(f"Total samples: {len(all_y_true)} ({n_bird} bird, {n_noise} noise)")
    print(f"Hours of audio: {hours:.2f}")
    print(f"{'='*60}")

    best = {"threshold": 0, "f1": 0, "fp_hr": 999}
    for thresh in thresholds:
        y_pred = (all_y_prob > thresh).astype(int)
        f1, prec, rec, fp, fn, tp = soft_f1(all_y_true, y_pred)
        fp_hr = fp / max(hours, 1)
        flag_f1 = "  *** MEETS SC-007 ***" if f1 >= SC007_TARGET_F1 else ""
        flag_fp = "  *** MEETS SC-008 ***" if fp_hr <= SC008_TARGET_FP_HR else ""
        print(f"  thresh={thresh:.2f}: F1={f1:.4f}{flag_f1} P={prec:.4f} R={rec:.4f} FP/hr={fp_hr:.4f}{flag_fp}  (tp={tp}, fp={fp}, fn={fn})")
        if f1 >= best["f1"] and fp_hr <= SC008_TARGET_FP_HR:
            best = {"threshold": thresh, "f1": f1, "precision": prec, "recall": rec,
                    "fp_per_hour": fp_hr, "tp": tp, "fp": fp, "fn": fn}

    # Report best that meets targets
    if best["f1"] > 0:
        print(f"\n  >>> Best threshold: {best['threshold']:.2f} (F1={best['f1']:.4f}, FP/hr={best['fp_per_hour']:.4f})")
    else:
        print(f"\n  >>> No threshold meets both SC-007 (F1>={SC007_TARGET_F1}) and SC-008 (FP/hr<={SC008_TARGET_FP_HR}) targets.")

    # Show confidence distribution
    noise_probs = all_y_prob[all_y_true == 0]
    bird_probs = all_y_prob[all_y_true == 1]
    print(f"\n  Noise conf:  min={noise_probs.min():.4f} mean={noise_probs.mean():.4f} max={noise_probs.max():.4f}")
    print(f"  Bird  conf:  min={bird_probs.min():.4f} mean={bird_probs.mean():.4f} max={bird_probs.max():.4f}")

    return best


def grid_sweep():
    """Run grid over architectures and thresholds."""
    overall_best = {"f1": 0, "fp_per_hour": 999}
    for hd, l2, drop, label in EVAL_SWEEP_CONFIGS:
        result = evaluate(thresholds=SWEEP_THRESHOLDS, hidden_dim=hd, l2_reg=l2, dropout=drop)
        if result["f1"] > overall_best["f1"] and result["fp_per_hour"] <= SC008_TARGET_FP_HR:
            overall_best = result
            overall_best["config"] = label
        elif not overall_best.get("config") and result["f1"] > overall_best["f1"]:
            overall_best = result
            overall_best["config"] = label

    print(f"\n{'='*60}")
    print(f"OVERALL BEST: {overall_best.get('config', 'N/A')}")
    print(f"  threshold={overall_best['threshold']:.2f}")
    print(f"  F1={overall_best['f1']:.4f}  P={overall_best.get('precision', 0):.4f}  R={overall_best.get('recall', 0):.4f}")
    print(f"  FP/hr={overall_best['fp_per_hour']:.4f}  tp={overall_best.get('tp', 0)}  fp={overall_best.get('fp', 0)}  fn={overall_best.get('fn', 0)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        grid_sweep()
    else:
        evaluate()
