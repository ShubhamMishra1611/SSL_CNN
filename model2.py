import tensorflow as tf
from tensorflow import keras

filters = 64
kernel_size = (3, 3)
strides = (1, 1)
input_shape = (14, 511, 10)
rate = 0.5
K = 36

def get_model():
    inputs = keras.Input(shape=input_shape)

    # Convolutional layers + batch normalization (BN) with ReLU
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation='relu', name='conv1')(inputs)
    x = keras.layers.BatchNormalization(name='bn1')(x)

    # 2nd convolutional layers + batch normalization (BN) with ReLU
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation='relu', name='conv2')(x)
    x = keras.layers.BatchNormalization(name='bn2')(x)

    # Dropout procedure with rate 0.5
    x = tf.keras.layers.Dropout(rate, name='dn1')(x)

    # Flatten layer
    x = keras.layers.Flatten(name='flatten')(x)

    # Self-attention mechanism
    attention = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)

    # Concatenate attention output with previous layer
    x = tf.keras.layers.Concatenate()([x, attention])

    # Fully connected layers with ReLU activation and dropout
    x = tf.keras.layers.Dense(512, activation='relu', name='fc1')(x)
    x = tf.keras.layers.Dropout(rate, name='dn2')(x)

    x = tf.keras.layers.Dense(512, activation='relu', name='fc2')(x)
    x = tf.keras.layers.Dropout(rate, name='dn3')(x)

    # Output layer
    outputs = tf.keras.layers.Dense(K, activation='softmax', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model

# Get the model
model = get_model()
