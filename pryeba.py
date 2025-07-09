import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import pandas as pd

# --- Cargar el modelo entrenado (si no está ya en memoria)
model = tf.keras.models.load_model("supermarket_model.h5")  # si lo guardaste

# --- Ruta de la imagen real (ajústala a tu caso)
img_path = 'prueba4.jpeg'

# --- Cargar imagen y preprocesar igual que las del dataset
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# --- Hacer predicción
preds = model.predict(img_array)[0]  # ya viene con forma (1, N)
preds_redondeadas = (preds > 0.5).astype(int)

# --- Mostrar la imagen
plt.imshow(img)
plt.axis('off')
plt.title("Predicción")
plt.show()

# --- Mostrar resultados
etiquetas = pd.read_csv('Dataset-SupermarketImages/annotations.csv').columns[1:]  # quitamos la columna 'file'
resultado = dict(zip(etiquetas, preds_redondeadas))

print("Etiquetas detectadas:")
for k, v in resultado.items():
    if v == 1:
        print(f"✅ {k}")
