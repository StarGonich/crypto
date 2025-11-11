import numpy as np

def calculate_entropy(img):
    # Вычисление энтропии изображения
    if len(img.shape) == 3:
        entropy_r = calculate_channel_entropy(img[:, :, 0])
        entropy_g = calculate_channel_entropy(img[:, :, 1])
        entropy_b = calculate_channel_entropy(img[:, :, 2])
        return [entropy_r, entropy_g, entropy_b]
    else:
        # Для grayscale
        return [calculate_channel_entropy(img)]


def calculate_channel_entropy(channel):
    # Вычисление энтропии для одного канала
    histogram = np.histogram(channel, bins=256, range=(0, 255))[0]
    histogram = histogram / histogram.sum()
    entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
    return entropy


def calculate_correlation(img, direction='horizontal'):
    # Вычисление корреляции соседних пикселей
    if len(img.shape) == 3:
        corr_r = calculate_channel_correlation(img[:, :, 0], direction)
        corr_g = calculate_channel_correlation(img[:, :, 1], direction)
        corr_b = calculate_channel_correlation(img[:, :, 2], direction)
        return [corr_r, corr_g, corr_b]
    else:
        return [calculate_channel_correlation(img, direction)]


def calculate_channel_correlation(channel, direction):
    # Вычисление корреляции для одного канала
    h, w = channel.shape

    if direction == 'horizontal':
        x = channel[:, :-1].flatten()
        y = channel[:, 1:].flatten()
    elif direction == 'vertical':
        x = channel[:-1, :].flatten()
        y = channel[1:, :].flatten()
    elif direction == 'diagonal':
        x = channel[:-1, :-1].flatten()
        y = channel[1:, 1:].flatten()

    correlation = np.corrcoef(x, y)[0, 1]
    return correlation if not np.isnan(correlation) else 0


def calculate_npcr_uaci(img1, img2):
    # Вычисление NPCR и UACI между двумя изображениями
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")

    if len(img1.shape) == 3:
        npcr_r, uaci_r = calculate_channel_npcr_uaci(img1[:, :, 0], img2[:, :, 0])
        npcr_g, uaci_g = calculate_channel_npcr_uaci(img1[:, :, 1], img2[:, :, 1])
        npcr_b, uaci_b = calculate_channel_npcr_uaci(img1[:, :, 2], img2[:, :, 2])
        return [npcr_r, npcr_g, npcr_b], [uaci_r, uaci_g, uaci_b]
    else:
        npcr, uaci = calculate_channel_npcr_uaci(img1, img2)
        return [npcr], [uaci]


def calculate_channel_npcr_uaci(channel1, channel2):
    # Вычисление NPCR и UACI для одного канала
    diff = channel1 != channel2
    npcr = np.sum(diff) / diff.size * 100

    uaci = np.sum(np.abs(channel1.astype(float) - channel2.astype(float))) / (255 * diff.size) * 100

    return npcr, uaci


def calculate_histogram(img):
    # Вычисление гистограммы изображения
    if len(img.shape) == 3:
        hist_r = np.histogram(img[:, :, 0], bins=256, range=(0, 255))[0]
        hist_g = np.histogram(img[:, :, 1], bins=256, range=(0, 255))[0]
        hist_b = np.histogram(img[:, :, 2], bins=256, range=(0, 255))[0]
        return [hist_r, hist_g, hist_b]
    else:
        hist = np.histogram(img, bins=256, range=(0, 255))[0]
        return [hist]