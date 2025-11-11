import os

from utils import load_image, save_image, image_to_bytes, bytes_to_image, save_metadata, load_metadata

def get_key(S, x, y):
    x = (x + 1) % 256
    y = (y + S[x]) % 256

    S[x], S[y] = S[y], S[x]

    return S[(S[x] + S[y]) % 256], x, y


def rc4(msg: bytes, key: bytes, iv=None) -> bytearray:
    len_msg = len(msg)
    len_key = len(key)
    cipher = bytearray(len_msg)
    S = list(range(256))
    x = 0
    y = 0

    j = 0
    for i in range(256):
        j = (j + S[i] + key[i % len_key]) % 256
        S[i], S[j] = S[j], S[i]

    if iv is not None:
        for i in range(256):
            j = (j + S[i] + iv[i % len(iv)]) % 256
            S[i], S[j] = S[j], S[i]

    for i in range(len_msg):
        key_byte, x, y = get_key(S, x, y)
        cipher[i] = msg[i] ^ key_byte

    return cipher


if __name__ == "__main__":
    key = b'I_LOVE_JUSTIN_BIBER'
    iv = os.urandom(16)

    img_array = load_image('../imgs/checkerboard.png')
    original_shape = img_array.shape
    image_bytes = image_to_bytes(img_array)

    encrypted_bytes = rc4(image_bytes, key, iv)
    encrypted_array = bytes_to_image(encrypted_bytes, original_shape)
    save_image(encrypted_array, 'encrypted.png')

    decrypted_bytes = rc4(encrypted_bytes, key, iv)
    decrypted_array = bytes_to_image(decrypted_bytes, original_shape)
    save_image(decrypted_array, 'decrypted.png')