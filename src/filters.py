import math
import numpy as np

def gaussian_blur(img, sigma):
    if sigma <= 1e-6:
        return img.copy()
    
    r = int(4.0 * sigma + 0.5) # радіус
    t = np.arange(-r, r + 1, dtype=float)
    
    # Рахуємо ваги за формулою Гауса
    kernel = np.exp(-(t * t) / (2.0 * sigma * sigma))
    kernel /= kernel.sum() # нормування, щоб сума була 1
    
    h, w = img.shape
    
    # Прохід 1: по горизонталі
    tmp = np.zeros_like(img)
    for k in range(-r, r + 1):
        cols = np.clip(np.arange(w) + k, 0, w - 1)
        tmp += kernel[k + r] * img[:, cols]
        
    # Прохід 2: по вертикалі
    out = np.zeros_like(img)
    for k in range(-r, r + 1):
        rows = np.clip(np.arange(h) + k, 0, h - 1)
        out += kernel[k + r] * tmp[rows, :]
        
    return out

def bilateral_filter(img, sigma_spatial, sigma_range):
    sigma_spatial = max(float(sigma_spatial), 1e-3)
    sigma_range = max(float(sigma_range), 1e-3)
    r = int(min(max(round(2.0 * sigma_spatial), 1), 5))

    h, w = img.shape
    padded = np.pad(img, r, mode="reflect") # дзеркальні краї
    num = np.zeros_like(img)
    den = np.zeros_like(img)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            shifted = padded[r + dy:r + dy + h, r + dx:r + dx + w]
            w_s = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma_spatial ** 2))
            w_r = np.exp(-((shifted - img) ** 2) / (2.0 * sigma_range ** 2))
            weight = w_s * w_r
            num += weight * shifted
            den += weight

    return num / den