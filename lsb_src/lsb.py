import numpy as np
from utils import load_image, save_image, image_to_bytes, bytes_to_image


def text_to_bits(text):
    if isinstance(text, str):
        text = text.encode('utf-8')
    bits = []
    for byte in text:
        bits.extend([(byte >> i) & 1 for i in range(7, -1, -1)])
    return bits


def bits_to_text(bits):
    bytes_list = []
    for i in range(0, len(bits), 8):
        if i + 8 > len(bits):
            break
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        bytes_list.append(byte)
    return bytes(bytes_list).decode('utf-8', errors='ignore')


def lsb_encode(img_array: np.ndarray, message: bytes) -> np.ndarray:
    stego_array = img_array.copy()
    height, width, channels = stego_array.shape

    # Преобразование сообщения в биты
    message_bits = text_to_bits(message)

    # Добавляем маркер конца (8 нулевых байтов)
    end_marker = [0] * 16
    all_bits = message_bits + end_marker

    # Проверка емкости
    total_capacity = height * width * channels
    if len(all_bits) > total_capacity:
        raise ValueError(f"Сообщение слишком длинное. Максимум: {total_capacity // 8} байт")

    # Встраивание битов
    bit_index = 0
    for i in range(height):
        for j in range(width):
            for channel in range(channels):
                if bit_index < len(all_bits):
                    # Замена младшего бита
                    stego_array[i, j, channel] = (stego_array[i, j, channel] & 0xFE) | all_bits[bit_index]
                    bit_index += 1
                else:
                    break
            if bit_index >= len(all_bits):
                break
        if bit_index >= len(all_bits):
            break

    return stego_array


def lsb_decode(stego_array: np.ndarray) -> bytes:
    """
    Декодирование сообщения из стего-изображения

    Args:
        stego_array: стего-изображение как numpy array

    Returns:
        message: декодированное сообщение (bytes)
    """
    height, width, channels = stego_array.shape

    bits = []
    zero_count = 0

    # Извлечение битов
    for i in range(height):
        for j in range(width):
            for channel in range(channels):
                # Извлечение младшего бита
                bit = stego_array[i, j, channel] & 1
                bits.append(bit)

                # Проверка на маркер конца
                if bit == 0:
                    zero_count += 1
                else:
                    zero_count = 0

                if zero_count >= 16:  # 8 нулевых байтов
                    bits = bits[:-16]
                    return bits_to_text(bits).encode('utf-8')

    return bits_to_text(bits).encode('utf-8')


if __name__ == "__main__":
    # Пример использования
    message = b"Secret message hidden in image!"

    # Загрузка изображения
    img_array = load_image('../imgs/checkerboard.png')
    original_shape = img_array.shape

    # Кодирование сообщения
    stego_array = lsb_encode(img_array, message)
    save_image(stego_array, 'stego_image.png')
    print("Сообщение закодировано в stego_image.png")

    # Декодирование сообщения
    decoded_img_array = load_image('stego_image.png')
    decoded_message = lsb_decode(decoded_img_array)

    print(f"Исходное сообщение: {message}")
    print(f"Декодированное сообщение: {decoded_message}")

    # Проверка корректности
    if message == decoded_message:
        print("✓ Сообщение успешно скрыто и восстановлено!")
    else:
        print("✗ Ошибка: сообщение восстановлено некорректно")