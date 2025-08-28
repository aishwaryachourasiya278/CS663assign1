import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def otsu_threshold_math(img_path, show=True):
    # Load image and convert to grayscale if needed
    img = np.array(Image.open(img_path).convert("L"))
    hist, bins = np.histogram(img.ravel(), bins=256, range=(0, 256))
    total = img.size

    # Normalize histogram to probabilities
    prob = hist / total
    omega = np.cumsum(prob)               # cumulative sum (w0, w1)
    mu = np.cumsum(prob * np.arange(256)) # cumulative mean

    mu_t = mu[-1]  # global mean

    # Between-class variance formula
    sigma_b_squared = (mu_t * omega - mu)**2 / (omega * (1 - omega) + 1e-8)

    # Best threshold = argmax of sigma_b_squared
    threshold = np.nanargmax(sigma_b_squared)

    # Apply threshold
    binary_img = (img > threshold).astype(np.uint8) * 255

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(binary_img, cmap="gray")
        ax[1].set_title(f"Otsu Threshold = {threshold}")
        ax[1].axis("off")
        plt.show()

    return binary_img, threshold

images = [
    "data/thresh/receipt.png",
    "data/thresh/blackboard.png",
    "data/thresh/lilavati.tif",
    "data/thresh/qr.png"
]

for img_path in images:
    bin_img, t = otsu_threshold_math(img_path)
    print(f"Otsu threshold for {img_path}: {t}")
