from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
import os
import numpy as np
from tqdm import tqdm

augmentador = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

input_dir = 'Dataset-SupermarketImages/Productos/Train'
output_dir = 'Dataset-SupermarketImages/Productos/Train_Aumentado'

os.makedirs(output_dir, exist_ok=True)

# Solo las clases deseadas
clases_a_duplicar = ['doritos', 'cocaCola']

for clase in tqdm(os.listdir(input_dir)):
    if clase not in clases_a_duplicar:
        continue  # saltar otras clases

    clase_dir = os.path.join(input_dir, clase)
    output_clase_dir = os.path.join(output_dir, clase)
    os.makedirs(output_clase_dir, exist_ok=True)

    for img_file in os.listdir(clase_dir):
        img_path = os.path.join(clase_dir, img_file)
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Guardar 3 versiones aumentadas por imagen
        i = 0
        for batch in augmentador.flow(x, batch_size=1):
            save_path = os.path.join(output_clase_dir, f"aug_{i}_{img_file}")
            save_img(save_path, batch[0])
            i += 1
            if i >= 6:
                break
