import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc

from lab2_src import lsb

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

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


def create_difference_map(original, stego, image_name, payload, output_dir):
    diff_map = np.abs(original.astype(float) - stego.astype(float))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Оригинал')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(stego)
    plt.title('Стего-изображение')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(diff_map, cmap='hot')
    plt.title('Карта разности')
    plt.axis('off')
    plt.colorbar()

    plt.suptitle(f'Сравнение: {image_name} - LSB payload {payload:.1%}', fontsize=14)
    plt.tight_layout()

    diff_path = os.path.join(output_dir, f"{image_name}_payload_{int(payload * 100)}_difference.png")
    plt.savefig(diff_path, dpi=150, bbox_inches='tight')
    plt.close()

    return diff_path, diff_map


def plot_histograms(original, stego, image_name, payload, output_dir):
    """Построение гистограмм до и после"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = ['red', 'green', 'blue']
    channel_names = ['Red', 'Green', 'Blue']

    for i in range(3):
        axes[0, i].hist(original[:, :, i].flatten(), bins=256, color=colors[i],
                        alpha=0.7, density=True)
        axes[0, i].set_title(f'{channel_names[i]} канал (оригинал)')
        axes[0, i].set_xlim(0, 255)
        axes[0, i].grid(True, alpha=0.3)

        axes[1, i].hist(stego[:, :, i].flatten(), bins=256, color=colors[i],
                        alpha=0.7, density=True)
        axes[1, i].set_title(f'{channel_names[i]} канал (stego)')
        axes[1, i].set_xlim(0, 255)
        axes[1, i].grid(True, alpha=0.3)

    plt.suptitle(f'Гистограммы: {image_name} - LSB payload {payload:.1%}', fontsize=14)
    plt.tight_layout()

    hist_path = os.path.join(output_dir, f"{image_name}_payload_{int(payload * 100)}_histograms.png")
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    return hist_path


def chi_square_test(image_array, channel=0):
    hist, _ = np.histogram(image_array[:, :, channel].flatten(), bins=256, range=(0, 255))
    # print(hist)
    chi_square = 0
    degrees_of_freedom = 0

    for k in range(0, 128):
        freq_2k = hist[2 * k]  # Значения 0, 2, 4, ..., 254
        freq_2k1 = hist[2 * k + 1]  # Значения 1, 3, 5, ..., 255

        expected = (freq_2k + freq_2k1) / 2

        if expected > 0:
            chi_2k = ((freq_2k - expected) ** 2) / expected
            chi_2k1 = ((freq_2k1 - expected) ** 2) / expected
            chi_square += chi_2k + chi_2k1
            degrees_of_freedom += 1
    # print(degrees_of_freedom)
    if degrees_of_freedom > 0:
        p_value = 1 - stats.chi2.cdf(chi_square, degrees_of_freedom - 1)
        return p_value, chi_square
    else:
        return 1.0, 0.0


def calculate_capacity(image_array):
    """Расчет емкости изображения"""
    height, width, channels = image_array.shape
    return height * width * channels


def generate_message_for_payload(image_array, payload_ratio):
    """Генерация сообщения для заданного payload"""
    total_capacity = calculate_capacity(image_array)
    service_bits = 16

    message_bits = int((payload_ratio / 100) * total_capacity) - service_bits
    message_bytes = max(0, message_bits // 8)

    message = b'a' * message_bytes
    # if message_bytes > 0:
    #     message = os.urandom(message_bytes)
    # else:
    #     message = b''

    return message


def generate_roc_analysis(original_img, payloads, visualizations_dir):
    print(f"\n{'=' * 70}")
    print("ROC-АНАЛИЗ И AUC")
    print(f"{'=' * 70}")

    # Генерация тестовых данных
    n_samples = 20

    all_scores = []
    all_labels = []

    # 1. Генерация stego-изображений из cover
    for payload in payloads:
        for i in range(n_samples):
            message = generate_message_for_payload(original_img, payload * 100)
            stego_img = lsb.lsb_encode(original_img, message)

            # Вычисляем p-value для каждого канала и берем минимальное
            p_values = []
            for channel in range(3):
                p_value, _ = chi_square_test(stego_img, channel)
                p_values.append(p_value)
            min_p_value = min(p_values)

            # Преобразуем p-value в score (чем меньше p-value, тем выше вероятность stego)
            score = 1 - min_p_value
            all_scores.append(score)
            all_labels.append(1)  # 1 = stego

    p_values.clear()
    # 2. Используем оригинальные cover изображения для clean
    for channel in range(3):
        p_value, _ = chi_square_test(original_img, channel)
        p_values.append(p_value)
    min_p_value = min(p_values)
    score = 1 - min_p_value
    for i in range(n_samples * len(payloads)):
        all_scores.append(score)
        all_labels.append(0)

    # Преобразуем в numpy arrays
    scores = np.array(all_scores)
    labels = np.array(all_labels)
    print(scores)
    print(labels)

    # Статистика по scores
    stego_scores = scores[labels == 1]
    clean_scores = scores[labels == 0]
    print(f"Средний score stego: {np.mean(stego_scores):.3f}")
    print(f"Средний score clean: {np.mean(clean_scores):.3f}")

    # Расчет ROC-кривой и AUC
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    print(f"FPR: ", fpr)
    print(f"TPR: ", tpr)
    print(f"AUC: ", roc_auc)

    # Построение ROC-кривой
    plt.figure(figsize=(10, 8))

    # Основная ROC кривая
    plt.subplot(2, 1, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайный классификатор')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC-кривая для LSB детектора (χ²-тест)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    roc_path = os.path.join(visualizations_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()

    return roc_auc, fpr, tpr, thresholds


if __name__ == '__main__':
    # Параметры анализа
    image_path = '../imgs/gradient.png'
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Изображение: {image_path}")

    # Создание директорий для результатов
    results_dir = os.path.join(current_dir, '../results/lsb_results')
    visualizations_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)

    # Загрузка изображения
    original_img = load_image(image_path)
    capacity = calculate_capacity(original_img)
    print(f"Емкость изображения: {capacity} пикселей")

    payloads = [0.001, 0.005, 0.01, 0.05]

    # Основной анализ для каждого payload
    for payload in payloads:
        print(f"\n{'=' * 70}")
        print(f"Payload: {payload:.1%}")
        print(f"{'=' * 70}")

        message = generate_message_for_payload(original_img, payload * 100)
        print(f"Длина сообщения: {len(message)} байт")

        # Кодирование
        stego_img = lsb.lsb_encode(original_img, message)

        # Декодирование для проверки
        decoded_message = lsb.lsb_decode(stego_img)
        print(f"Декодирование успешно: {message == decoded_message}")

        # =========================================================================
        # 1) НЕЗАМЕТНОСТЬ
        # =========================================================================
        print(f"\n1) НЕЗАМЕТНОСТЬ")

        # 1.1 Рассчитать PSNR, SSIM между cover и stego
        print("\nМЕТРИКИ КАЧЕСТВА:")
        psnr_value = calculate_psnr(original_img, stego_img)
        ssim_value = calculate_ssim(original_img, stego_img)
        print(f"   PSNR: {psnr_value:.2f} dB")
        print(f"   SSIM: {ssim_value:.4f}")

        # 1.2 Построить карту разности |cover − stego|
        diff_path, diff_map = create_difference_map(original_img, stego_img, image_name, payload, visualizations_dir)
        print(f"   Карта разности сохранена: {os.path.basename(diff_path)}")

        # 1.3 Построить гистограммы каналов до/после
        hist_path = plot_histograms(original_img, stego_img, image_name, payload, visualizations_dir)
        print(f"   Гистограммы сохранены: {os.path.basename(hist_path)}")

        # =========================================================================
        # 2) ОБНАРУЖИВАЕМОСТЬ
        # =========================================================================
        print(f"\n2) ОБНАРУЖИВАЕМОСТЬ (χ²-тест)")

        print("Канал    | p-value  | Значение χ² | Обнаружено")
        print("---------|----------|-------------|-----------")

        detectable_count = 0
        for channel, channel_name in enumerate(['Red', 'Green', 'Blue']):
            p_value, chi_stat = chi_square_test(stego_img, channel)
            detectable = "ДА" if p_value < 0.05 else "НЕТ"
            if p_value < 0.05:
                detectable_count += 1
            print(f"{channel_name:8} | {p_value:.6f} | {chi_stat:11.2f} | {detectable:>10}")

        print(f"\nИтог: {detectable_count}/3 каналов показывают наличие стего")

    # ROC-анализ
    roc_auc, fpr, tpr, thresholds = generate_roc_analysis(original_img, payloads, visualizations_dir)