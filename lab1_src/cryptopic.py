import argparse
import sys
import os
import numpy as np
import rc4
import aes
from utils import load_image, save_image, save_metadata, load_metadata, image_to_bytes, bytes_to_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CryptoPic - Image Encryption Tool')
    parser.add_argument('--mode', choices=['encrypt', 'decrypt'], required=True)
    parser.add_argument('--in', dest='input_file', required=True)
    parser.add_argument('--out', dest='output_file', required=True)
    parser.add_argument('--algo', choices=['stream', 'aes-ecb', 'aes-cbc', 'aes-ctr'], required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--iv', help='IV in hex format')
    parser.add_argument('--nonce', help='Nonce in hex format')
    parser.add_argument('--meta', help='Metadata file')

    args = parser.parse_args()

    try:
        key = args.key
        key = key.encode('utf-8') if isinstance(key, str) else key
        if args.mode == 'encrypt':
            img = load_image(args.input_file)
            original_shape = img.shape
            image_bytes = image_to_bytes(img)

            meta_file = args.meta or f"{os.path.splitext(args.output_file)[0]}.meta.json"

            # Генерация IV/nonce если не предоставлен
            iv = None
            if args.algo in ['aes-cbc', 'aes-ctr', 'stream']:
                if args.iv:
                    iv = bytes.fromhex(args.iv)
                elif args.nonce:
                    iv = bytes.fromhex(args.nonce)
                else:
                    # Генерируем соответствующий IV/nonce для каждого алгоритма
                    if args.algo == 'stream' or args.algo == 'aes-cbc':
                        iv = os.urandom(16)
                    elif args.algo == 'aes-ctr':
                        iv = os.urandom(8)

            # Выбор алгоритма шифрования
            if args.algo.startswith('aes-'):
                mode = args.algo.split('-')[1]
                encrypted_bytes = aes.aes_encrypt(mode, image_bytes, args.key, iv)
                encrypted_fixbytes = aes.fix_bytes(encrypted_bytes, original_shape)
                encrypted_img = bytes_to_image(encrypted_fixbytes, original_shape)
            elif args.algo == 'stream':
                encrypted_bytes = rc4.rc4(image_bytes, key, iv)
                encrypted_img = bytes_to_image(encrypted_bytes, original_shape)

            if args.output_file.endswith('.bin'):
                with open(args.output_file, 'wb') as f:
                    f.write(encrypted_bytes)
            elif args.output_file.endswith('.png'):
                save_image(encrypted_img, args.output_file)

            # Сохранение метаданных
            metadata = {
                'algo': args.algo,
                'original_shape': original_shape,
                'date': np.datetime64('now').astype(str)
            }

            if iv is not None:
                metadata['iv'] = iv.hex()

            save_metadata(metadata, meta_file)
            print(f"Encryption complete. Metadata saved to {meta_file}")
        elif args.mode == 'decrypt':
            # Определяем файл метаданных
            meta_file = args.meta or f"{os.path.splitext(args.input_file)[0]}.meta.json"

            # Загрузка метаданных
            metadata = load_metadata(meta_file)
            original_shape = tuple(metadata['original_shape'])
            iv = bytes.fromhex(metadata.get('iv', '')) if metadata.get('iv') else None

            if metadata['algo'].startswith('aes-'):
                mode = metadata['algo'].split('-')[1]

                # Чтение бинарных данных для AES
                with open(args.input_file, 'rb') as f:
                    encrypted_data = f.read()

                decrypted_bytes = aes.aes_decrypt(mode, encrypted_data, args.key)

            elif metadata['algo'] == 'stream':
                img = load_image(args.input_file)
                image_bytes = image_to_bytes(img)
                decrypted_bytes = rc4.rc4(image_bytes, key, iv)

            decrypted_img = bytes_to_image(decrypted_bytes, original_shape)
            # Сохранение дешифрованного изображения
            save_image(decrypted_img, args.output_file)
            print("Decryption complete")

    except Exception as e:
        print(f"Error: {e.with_traceback()}")
        sys.exit(1)