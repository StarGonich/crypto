import numpy as np
from PIL import Image
import json

def load_image(filename):
    # Загрузка изображения как numpy array
    try:
        img = Image.open(filename)
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Ошибка загрузки изображения {filename}: {e}")

def save_image(data, filename):
    # Сохранение numpy array как изображения
    try:
        img = Image.fromarray(data.astype('uint8'))
        img.save(filename)
    except Exception as e:
        raise ValueError(f"Ошибка сохранения изображения {filename}: {e}")

def image_to_bytes(img_array):
    return img_array.tobytes()

def bytes_to_image(byte_data, shape):
    array = np.frombuffer(byte_data, dtype=np.uint8)
    return array.reshape(shape)

def save_metadata(metadata, filename):
    # Сохранение метаданных в JSON
    with open(filename, 'w') as f:
        json.dump(metadata, f)

def load_metadata(filename):
    # Загрузка метаданных из JSON
    with open(filename, 'r') as f:
        return json.load(f)