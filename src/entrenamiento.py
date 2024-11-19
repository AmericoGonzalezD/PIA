# src/entrenamiento.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_path = 'dataset'  # Ajusta la ruta si es necesario
categories = ['sin_accesorios', 'lentes_de_sol', 'gorra', 'ambos']

def cargar_datos(data_path, categories, img_size=(128, 128)):
    datos = []
    etiquetas = []
    for i, category in enumerate(categories):
        path = os.path.join(data_path, category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.resize(img_array, img_size) / 255.0
                datos.append(img_array)
                etiquetas.append(i)
            except Exception as e:
                print(f"Error cargando imagen: {e}")
    return np.array(datos), np.array(etiquetas)

X, y = cargar_datos(data_path, categories)
y = keras.utils.to_categorical(y, num_classes=len(categories))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=15, validation_split=0.2, batch_size=32)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {test_acc:.2f}")

model.save('../modelos/modelo_accesorios.h5')

plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()
