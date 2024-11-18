# src/deteccion_tiempo_real.py
import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model('../modelos/modelo_accesorios.h5')
categories = ['sin_accesorios', 'lentes_de_sol', 'gorra', 'ambos']

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (128, 128)) / 255.0
    input_frame = np.expand_dims(resized_frame, axis=0)
    prediction = model.predict(input_frame)
    label = categories[np.argmax(prediction)]

    cv2.putText(frame, f'Predicción: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Detección de Accesorios', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
