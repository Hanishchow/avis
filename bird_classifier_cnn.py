"""CNN model for bird vs not-bird classification from mel-spectrograms."""

import tensorflow as tf
from tensorflow import keras


def build_cnn(input_shape=(128, 259), num_classes=2):
    """Build a CNN model for bird detection.
    
    Args:
        input_shape: (n_mels, n_frames) of mel-spectrogram
        num_classes: 2 (bird, none) or N+1 species
    
    Returns:
        Compiled keras model
    """
    inputs = keras.Input(shape=input_shape + (1,), name="mel_spectrogram")

    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="logits")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_cnn()
    model.summary()
    # Test with random input
    import numpy as np
    test_input = np.random.randn(1, 128, 259, 1)
    output = model.predict(test_input)
    print(f"Test output shape: {output.shape}, values: {output[0]}")
