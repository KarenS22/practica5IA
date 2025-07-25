# Modelo CNN con Transfer Learning - Clasificación de Productos de Supermercado

## Descripción

Este proyecto implementa un modelo de clasificación multietiqueta para identificar productos de supermercado utilizando Transfer Learning con MobileNetV2. El modelo está diseñado para clasificar 11 categorías diferentes de productos con un dataset balanceado.

## Estructura del Proyecto

```
Version2/Clasificacion/
├── modeloCNNTL.ipynb                                    # Notebook principal
├── modelo_cnn_transfer_learning_balanceado.h5          # Modelo entrenado
├── metadata_transfer_learning_balanceado.json          # Metadatos del modelo
└── README.md                                           # Este archivo
```

## Dataset

El modelo utiliza un dataset personalizado de productos de supermercado con las siguientes características:

- **Entrenamiento**: 888 muestras (balanceadas)
- **Validación**: 60 muestras
- **Resolución**: 224x224 píxeles
- **Formato**: RGB

### Clases del Dataset

1. Banana
2. Orange  
3. Red-Bell-Pepper
4. aluminium-form
5. bread
6. cocaCola
7. doritos
8. red-bull
9. shampoo-H&S
10. tea
11. yogurt-toni-mix

## Arquitectura del Modelo

El modelo utiliza Transfer Learning basado en MobileNetV2:

- **Backbone**: MobileNetV2 pre-entrenado en ImageNet
- **Capas congeladas**: Todas las capas de MobileNetV2
- **Capas adicionales**:
  - GlobalAveragePooling2D
  - Dropout (0.2)
  - Dense(128, activation='relu')
  - Dropout (0.5)
  - Dense(11, activation='sigmoid') - Salida multietiqueta

### Parámetros de Entrenamiento

- **Optimizador**: Adam (lr=0.001)
- **Función de pérdida**: Binary Crossentropy
- **Batch size**: 32
- **Épocas**: 20
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

## Resultados

### Métricas de Rendimiento

- **Train Accuracy**: 96.51%
- **Validation Accuracy**: 98.33%
- **Train Loss**: 0.0041
- **Validation Loss**: 0.0019
- **Hamming Loss**: 0.0000
- **Subset Accuracy**: 100%

### Análisis del Modelo

- **Sin overfitting detectado**: La accuracy de validación es superior a la de entrenamiento
- **Convergencia**: El modelo converge adecuadamente en 20 épocas
- **Generalización**: Excelente capacidad de generalización

## Uso del Modelo

### Requisitos

```python
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

### Clasificación de Imágenes Externas

```python
# Cargar el modelo
model = tf.keras.models.load_model('modelo_cnn_transfer_learning_balanceado.h5')

# Usar la función de clasificación
resultado = clasificar_imagen_externa('ruta/a/imagen.jpg')
print(resultado)
```

### Parámetros de la Función

- **ruta_imagen**: Ruta completa a la imagen a clasificar
- **threshold**: Umbral de confianza (default: 0.4)

### Formatos Soportados

- JPG/JPEG
- PNG
- BMP
- TIFF

## Preprocesamiento

Las imágenes se procesan de la siguiente manera:

1. Redimensionado a 224x224 píxeles
2. Conversión de BGR a RGB
3. Normalización específica de MobileNetV2 (valores entre -1 y 1)

## Balanceado del Dataset

El dataset original fue balanceado para mejorar el rendimiento:

- **Objetivo**: 85 muestras por clase
- **Método**: Sampling aleatorio con semilla fija (random_state=42)
- **Resultado**: Distribución más equitativa entre clases

## Metadatos del Modelo

Los metadatos se guardan automáticamente en formato JSON e incluyen:

- Tipo de modelo y arquitectura
- Número de clases y nombres
- Métricas finales de entrenamiento
- Configuración del dataset
- Parámetros del modelo

## Análisis de Overfitting

El análisis automático incluye:

- Comparación entre métricas de entrenamiento y validación
- Evaluación del tamaño del conjunto de validación
- Análisis de convergencia
- Recomendaciones de mejora

## Limitaciones

- Conjunto de validación relativamente pequeño (60 muestras)
- Rendimiento "perfecto" que podría indicar dataset demasiado simple
- Necesidad de validación con imágenes completamente nuevas

## Mejoras Futuras

1. Ampliar el conjunto de validación
2. Incluir más variabilidad en las imágenes
3. Implementar data augmentation
4. Probar con otros modelos pre-entrenados
5. Optimización de thresholds por clase

## Autor

Desarrollado como parte del proyecto de Inteligencia Artificial.

## Tecnologías Utilizadas

- TensorFlow/Keras
- OpenCV
- Scikit-learn
- Matplotlib
- Pandas
- NumPy
- MobileNetV2

## Notas Técnicas

- El modelo utiliza sigmoid en la capa de salida para clasificación multietiqueta
- Se implementa regularización mediante Dropout
- Transfer Learning permite aprovechas características pre-entrenadas de ImageNet
- El balanceado del dataset mejora la representación de clases minoritarias
