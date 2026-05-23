"""Fallback model for bird detection when trained model is unavailable."""

import os
import numpy as np
import tensorflow as tf


def load_fallback():
    """Load a fallback model. Creates a simple pretrained-style model if no real one exists."""
    path = "bird_sound_classifier.h5"
    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path)
            print(f"Loaded existing model from {path}")
            return model
        except Exception as e:
            print(f"Could not load model: {e}")

    print("Creating fallback model (random weights)")
    inputs = tf.keras.Input(shape=(128, 259, 1))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    model = load_fallback()
    model.summary()
