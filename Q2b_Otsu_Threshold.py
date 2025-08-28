import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image
import cv2

def Otsu_Thresholding(img, show=True, cmap="gray"):
    
    if img.ndim == 3:  # RGB image → use magnitude
        R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
        arr = np.sqrt(R**2 + G**2 + B**2).astype(np.float32)
        img_type = "rgb"
    elif img.ndim == 2:  # Grayscale image
        arr = img.astype(np.float32)
        img_type = "gray"
    else:
        raise ValueError("Unsupported image shape")

    M, N = arr.shape
    L = 256
    hist, bins = np.histogram(arr.ravel(), bins=L, range=(0, np.max(arr)))

    # Normalize histogram → probability distribution
    P = hist / (M * N)

    cumulative_prob = np.cumsum(P)
    cumulative_mean = np.cumsum(np.arange(L) * P)
    global_mean = cumulative_mean[-1]

    # Otsu loop
    max_between_var = -1.0
    optimal_thresh = 0
    for t in range(1, L):
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

    # Apply threshold (float, not int)
    binary_img = (arr > optimal_thresh).astype(np.float32) * 255.0

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        if img_type == "rgb":
            im0 = ax[0].imshow(img / np.max(img), cmap=cmap, vmin=0, vmax=200)
        else:
            im0 = ax[0].imshow(img, cmap=cmap, vmin=0, vmax=200)
        ax[0].set_title("Original Image")
        ax[0].axis("off")
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(binary_img, cmap=cmap, vmin=0, vmax=200)
        ax[1].set_title(f"Otsu Threshold = {optimal_thresh}")
        ax[1].axis("off")
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    return binary_img, optimal_thresh


# ---- Run on your images ----
images = [
    "data/thresh/receipt.png",
    "data/thresh/blackboard.png",
    "data/thresh/lilavati.tif",
    "data/thresh/qr.png"
]

for img_path in images:
    try:
        img = iio.imread(img_path).astype(np.float32)
    except ValueError:  
        img = np.array(Image.open(img_path)).astype(np.float32)

    print("Image shape:", img.shape)
    Binarized_img, t = Otsu_Thresholding(img, show=True, cmap="gray")
    print(f"Otsu threshold for {img_path}: {t}")
