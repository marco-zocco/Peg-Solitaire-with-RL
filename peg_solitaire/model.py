"""
Model:
  * Input (7, 7, 2)
  * Two 3x3 conv layers
  * Dense layer
  * Single sigmoid output -> V in (0, 1]
  * Output a scalar (V value)
"""

import tensorflow as tf
from tensorflow import keras 
from keras import layers, models


def build_value_network(board_shape=(7, 7), channels=2, filters=(32, 64), dense_units=64, name="value_net"):
    
    inp = layers.Input(shape=(*board_shape, channels))
    x = inp
    for f in filters:
        x = layers.Conv2D(f, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out, name=name)
