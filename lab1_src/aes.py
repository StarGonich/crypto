import os

from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad

from utils import bytes_to_image, load_image, image_to_bytes, save_image

def check_key(key):
    key = key.encode('utf-8') if isinstance(key, str) else key
    key_len = len(key)
    if key_len <= 16:
        key = key.ljust(16, b'\x00')
    elif key_len <= 24:
        key = key.ljust(24, b'\x00')
    else:
        key = key.ljust(32, b'\x00')
    return key

def fix_bytes(bytes, shape):
    bytes_needed = shape[0] * shape[1] * shape[2]

    if len(bytes) < bytes_needed:
        bytes = bytes + b'\x00' * (bytes_needed - len(bytes))
    return bytes[:bytes_needed]


def aes_encrypt(mode: str, msg: bytes, key: bytes, iv=None) -> bytes:
    key = check_key(key)

    match mode:
        case 'ecb':
            cipher = AES.new(key, AES.MODE_ECB)
            msg = pad(msg, AES.block_size)
            encrypted_bytes = cipher.encrypt(msg)
        case 'cbc':
            if iv is None:
                iv = os.urandom(16)
            cipher = AES.new(key, AES.MODE_CBC, iv)
            msg = pad(msg, AES.block_size)
            encrypted_bytes = iv + cipher.encrypt(msg)
        case 'ctr':
            if iv is None:
                iv = os.urandom(8)
            cipher = AES.new(key, AES.MODE_CTR, nonce=iv)
            encrypted_bytes = iv + cipher.encrypt(msg)
    return encrypted_bytes


def aes_decrypt(mode: str,  msg: bytes, key: bytes):
    key = check_key(key)

    match mode:
        case 'ecb':
            cipher = AES.new(key, AES.MODE_ECB)
            decrypted_bytes = cipher.decrypt(msg)
            decrypted_bytes = unpad(decrypted_bytes, AES.block_size)
        case 'cbc':
            iv = msg[:16]
            cipher = AES.new(key, AES.MODE_CBC, iv)
            decrypted_bytes = cipher.decrypt(msg[16:])
            decrypted_bytes = unpad(decrypted_bytes, AES.block_size)
        case 'ctr':
            nonce = msg[:8]
            cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
            decrypted_bytes = cipher.decrypt(msg[8:])
    return decrypted_bytes

if __name__ == '__main__':
    key = b'16bytekey1234567'
    # iv = os.urandom(8)

    mode = "ecb"

    img_array = load_image('../imgs/checkerboard.png')
    original_shape = img_array.shape
    image_bytes = image_to_bytes(img_array)

    encrypted_bytes = aes_encrypt(mode, image_bytes, key)
    encrypted_fixbytes = fix_bytes(encrypted_bytes, original_shape)
    encrypted_img = bytes_to_image(encrypted_fixbytes, original_shape)
    save_image(encrypted_img, 'encrypted.png')

    decrypted_bytes = aes_decrypt(mode, encrypted_bytes, key)
    decrypted_img = bytes_to_image(decrypted_bytes, original_shape)
    save_image(decrypted_img, 'decrypted.png')