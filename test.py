import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Verifica que puedas usar keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
print(model.summary())
