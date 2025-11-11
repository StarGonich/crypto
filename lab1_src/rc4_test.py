import os
import sys
import matplotlib.pyplot as plt

from src.utils import image_to_bytes, bytes_to_image

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from utils import load_image, save_image
from metrics import calculate_entropy, calculate_correlation, calculate_npcr_uaci
import rc4


def plot_histograms(original, encrypted, image_name, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = ['red', 'green', 'blue']
    channel_names = ['Red', 'Green', 'Blue']

    for i in range(3):
        # Гистограмма оригинального изображения
        axes[0, i].hist(original[:, :, i].flatten(), bins=256, color=colors[i], alpha=0.7, density=True)
        axes[0, i].set_title(f'{channel_names[i]} канал (оригинал)')
        axes[0, i].set_xlim(0, 255)
        axes[0, i].grid(True, alpha=0.3)

        # Гистограмма зашифрованного изображения
        axes[1, i].hist(encrypted[:, :, i].flatten(), bins=256, color=colors[i], alpha=0.7, density=True)
        axes[1, i].set_title(f'{channel_names[i]} канал (rc4)')
        axes[1, i].set_xlim(0, 255)
        axes[1, i].grid(True, alpha=0.3)

    plt.suptitle(f'Гистограммы: {image_name} - rc4', fontsize=14)
    plt.tight_layout()

    hist_path = os.path.join(output_dir, f"{image_name}_histograms.png")
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    return hist_path


def run_encryption_decryption_direct(input_file, output_dir, key):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    os.makedirs(output_dir, exist_ok=True)
    encrypted_filepath = os.path.join(output_dir, f"{base_name}_encrypted.png")
    decrypted_filepath = os.path.join(output_dir, f"{base_name}_decrypted.png")

    original_img = load_image(f"../imgs/{input_file}")
    original_shape = original_img.shape
    original_bytes = image_to_bytes(original_img)

    iv = os.urandom(8)

    encrypted_bytes = rc4.rc4(original_bytes, key, iv)
    decrypted_bytes = rc4.rc4(encrypted_bytes, key, iv)
    encrypted_img = bytes_to_image(encrypted_bytes, original_shape)
    decrypted_img = bytes_to_image(decrypted_bytes, original_shape)
    save_image(encrypted_img, encrypted_filepath)
    save_image(decrypted_img, decrypted_filepath)

    return original_img, encrypted_img, decrypted_img


def test_key_sensitivity(original_img, key):
    original_shape = original_img.shape
    original_bytes = image_to_bytes(original_img)

    iv = os.urandom(8)

    encrypted_bytes1 = rc4.rc4(original_bytes, key, iv)
    encrypted_img1 = bytes_to_image(encrypted_bytes1, original_shape)

    key2 = key[:-1] + bytes([key[-1] ^ 1])

    encrypted_bytes2 = rc4.rc4(original_bytes, key2, iv)
    encrypted_img2 = bytes_to_image(encrypted_bytes2, original_shape)

    npcr, uaci = calculate_npcr_uaci(encrypted_img1, encrypted_img2)

    return npcr, uaci


def test_key_avalanche(original_img, key):
    original_bytes = image_to_bytes(original_img)

    # Шифрование с оригинальным ключом
    iv = os.urandom(8)
    encrypted_bytes1 = rc4.rc4(original_bytes, key, iv)

    # Изменяем последний бит ключа
    key_modified = key[:-1] + bytes([key[-1] ^ 1])

    # Шифрование с измененным ключом (тот же IV для чистоты эксперимента)
    encrypted_bytes2 = rc4.rc4(original_bytes, key_modified, iv)

    # Вычисляем долю изменившихся бит между двумя шифротекстами
    total_bits = len(encrypted_bytes1) * 8
    changed_bits = 0

    for b1, b2 in zip(encrypted_bytes1, encrypted_bytes2):
        xor_result = b1 ^ b2
        changed_bits += bin(xor_result).count('1')

    avalanche_effect = changed_bits / total_bits

    return avalanche_effect

if __name__ == '__main__':
    test_images = [f for f in os.listdir('../imgs') if os.path.isfile(os.path.join('../imgs', f))]

    key = b"I_LOVE_JUSTIN_BIBER_123"
    key = key.encode('utf-8') if isinstance(key, str) else key

    results_dir = os.path.join(current_dir, '../results/rc4_results')
    visualizations_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)

    all_results = {}

    for image_file in test_images:
        image_name = os.path.splitext(image_file)[0]

        # Шифрование и дешифрование
        original_img, encrypted_img, _ = run_encryption_decryption_direct(image_file, results_dir, key)

        # 1. Гистограммы
        hist_path = plot_histograms(original_img, encrypted_img, image_name, visualizations_dir)

        metrics = {}
        # 2. Корреляция соседних пикселей H/V/D (до/после).
        metrics['correlation_original'] = calculate_correlation(original_img, 'vertical')
        metrics['correlation_encrypted'] = calculate_correlation(encrypted_img, 'vertical')
        # 3. Энтропия каналов (ожидаемо близко к 8 бит после шифрования).
        metrics['entropy_original'] = calculate_entropy(original_img)
        metrics['entropy_encrypted'] = calculate_entropy(encrypted_img)

        # 4.1 NPCR / UACI между шифрами при ключах, отличающихся на 1 бит
        metrics['npcr_key_sensitivity'], metrics['uaci_key_sensitivity'] = test_key_sensitivity(original_img, key)

        # 4.2 NPCR / UACI между исходником и шифром
        metrics['npcr_original_encrypted'], metrics['uaci_original_encrypted'] = calculate_npcr_uaci(original_img, encrypted_img)

        # 5 Чувствительность к ключу (avalanche) — доля изменившихся бит в шифре при изменении 1 бита KEY
        metrics['key_avalanche'] = test_key_avalanche(original_img, key)

        # Вывод промежуточных результатов
        print(f"Метрики для {image_name}:")
        print(f"2)   Корреляция оригинала: {[f'{e:.4f}' for e in metrics['correlation_original']]}")
        print(f"     Корреляция зашифров.: {[f'{e:.4f}' for e in metrics['correlation_encrypted']]}")
        print(f"3)     Энтропия оригинала: {[f'{e:.4f}' for e in metrics['entropy_original']]}")
        print(f"       Энтропия зашифров.: {[f'{e:.4f}' for e in metrics['entropy_encrypted']]}")
        print(f"4.1)   NPCR чувств. ключа: {[f'{n:.4f}' for n in metrics['npcr_key_sensitivity']]}")
        print(f"       UACI чувств. ключа: {[f'{n:.4f}' for n in metrics['uaci_key_sensitivity']]}")
        print(f"4.2) NPCR оригинал-зашифр: {[f'{n:.4f}' for n in metrics['npcr_original_encrypted']]}")
        print(f"     UACI оригинал-зашифр: {[f'{n:.4f}' for n in metrics['uaci_original_encrypted']]}")
        print(f"5)   Чувствит-сть к ключу: {metrics['key_avalanche']}")
        print()
