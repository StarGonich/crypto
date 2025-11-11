# crypto.py
import argparse
import sys

import numpy as np

from lsb import lsb_encode, lsb_decode, lsb_encode2
from utils import load_image, save_image


def calculate_psnr(original, stego):
    mse = np.mean((original.astype(float) - stego.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    return psnr


def calculate_ssim(original, stego):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_x = np.mean(original)
    mu_y = np.mean(stego)
    sigma_x = np.std(original)
    sigma_y = np.std(stego)
    sigma_xy = np.cov(original.flatten(), stego.flatten())[0, 1]

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2)

    return numerator / denominator


def main():
    parser = argparse.ArgumentParser(description='LSB Steganography Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Encode command
    encode_parser = subparsers.add_parser('encode', help='Encode message in image')
    encode_parser.add_argument('-i', '--input', required=True, help='Input image path')
    encode_parser.add_argument('-o', '--output', required=True, help='Output image path')
    encode_parser.add_argument('-m', '--message', required=True, help='Message to hide')

    # Decode command
    decode_parser = subparsers.add_parser('decode', help='Decode message from image')
    decode_parser.add_argument('-i', '--input', required=True, help='Input image path')

    args = parser.parse_args()

    if args.command == 'encode':
        try:
            # Load image
            img_array = load_image(args.input)
            print(f"Loaded image: {args.input}")
            print(f"Image shape: {img_array.shape}")

            # Encode message
            if isinstance(args.message, str):
                message = args.message.encode('utf-8')
            else:
                message = args.message

            stego_array = lsb_encode2(img_array, message)

            print(calculate_psnr(img_array, stego_array))

            # Save stego image
            save_image(stego_array, args.output)
            print(f"Message encoded successfully!")
            print(f"Stego image saved as: {args.output}")
            print(f"Original message: {args.message}")

        except Exception as e:
            print(f"Error during encoding: {e}")
            sys.exit(1)

    elif args.command == 'decode':
        try:
            # Load stego image
            stego_array = load_image(args.input)
            print(f"Loaded stego image: {args.input}")
            print(f"Image shape: {stego_array.shape}")

            # Decode message
            decoded_message = lsb_decode(stego_array)

            print(f"Decoded message: {decoded_message.decode('utf-8')}")

        except Exception as e:
            print(f"Error during decoding: {e}")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()