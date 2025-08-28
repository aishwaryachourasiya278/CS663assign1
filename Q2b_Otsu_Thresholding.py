import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def otsu_threshold_math(img_path, show=True, cmap="gray"):
    # Load image and convert to grayscale
    img = np.array(Image.open(img_path).convert("L")).astype(np.float32)

    # Histogram
    hist, bins = np.histogram(img.ravel(), bins=256, range=(0, 256))
    total = img.size
    P = hist / total  # normalized histogram (probabilities)

    # Cumulative sums
    cumulative_prob = np.cumsum(P)
    cumulative_mean = np.cumsum(np.arange(256) * P)
    global_mean = cumulative_mean[-1]

    # Otsu loop (explicit)
    max_between_var = -1.0
    optimal_thresh = 0
    for t in range(1, 256):
        w0 = cumulative_prob[t]
        w1 = 1 - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0 = cumulative_mean[t] / w0
        mu1 = (global_mean - cumulative_mean[t]) / w1
        sigma_b2 = w0 * w1 * (mu0 - mu1) ** 2
        if sigma_b2 > max_between_var:
            max_between_var = sigma_b2
            optimal_thresh = t

    # Thresholding (keep float, no rounding off)
    binary_img = (img > optimal_thresh).astype(np.float32) * 255.0

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        im0 = ax[0].imshow(img, cmap=cmap)
        ax[0].set_title("Original Image")
        ax[0].axis("off")
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(binary_img, cmap=cmap)
        ax[1].set_title(f"Otsu Threshold = {optimal_thresh}")
        ax[1].axis("off")
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    return binary_img, optimal_thresh



images = [
    "data/thresh/receipt.png",
    "data/thresh/blackboard.png",
    "data/thresh/lilavati.tif",
    "data/thresh/qr.png"
]

for img_path in images:
    bin_img, t = otsu_threshold_math(img_path, cmap="gray")
    print(f"Otsu threshold for {img_path}: {t}")
