import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image

def Multi_Otsu_Thresholding(img, n_classes=3, show=True):
    """
    Multi-level Otsu thresholding (default 2 thresholds → 3 classes)
    """
    # Convert to grayscale-like array
    if img.ndim == 3:  # RGB image
        R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
        arr = np.sqrt(R**2 + G**2 + B**2)
        img_type = "rgb"
    elif img.ndim == 2:  # Grayscale image
        arr = img.copy()
        img_type = "gray"
    else:
        raise ValueError("Unsupported image shape")

    M, N = arr.shape
    hist, bins = np.histogram(arr.ravel(), bins=256, range=(0, np.max(arr)))
    L = 256

    P = hist / (M * N)
    cumulative_prob = np.cumsum(P)
    cumulative_mean = np.cumsum(np.arange(L) * P)
    global_mean = cumulative_mean[-1]

    # ---------- Multi-Otsu for 2 thresholds ----------
    max_between_var = 0
    optimal_thresh = (0, 0)

    for t1 in range(1, L - 1):
        for t2 in range(t1 + 1, L):
            w0 = cumulative_prob[t1]
            w1 = cumulative_prob[t2] - cumulative_prob[t1]
            w2 = 1 - cumulative_prob[t2]

            if w0 == 0 or w1 == 0 or w2 == 0:
                continue

            mu0 = cumulative_mean[t1] / w0
            mu1 = (cumulative_mean[t2] - cumulative_mean[t1]) / w1
            mu2 = (cumulative_mean[-1] - cumulative_mean[t2]) / w2

            sigma_b2 = (
                w0 * (mu0 - global_mean) ** 2 +
                w1 * (mu1 - global_mean) ** 2 +
                w2 * (mu2 - global_mean) ** 2
            )

            if sigma_b2 > max_between_var:
                max_between_var = sigma_b2
                optimal_thresh = (t1, t2)

    # Segment image into regions
    segmented_img = np.zeros_like(arr, dtype=np.uint8)
    segmented_img[arr <= optimal_thresh[0]] = 85    # class 0
    segmented_img[(arr > optimal_thresh[0]) & (arr <= optimal_thresh[1])] = 170  # class 1
    segmented_img[arr > optimal_thresh[1]] = 255    # class 2

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        if img_type == "rgb":
            ax[0].imshow(img.astype(np.uint8))
        else:
            ax[0].imshow(img.astype(np.uint8), cmap="gray")
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(segmented_img, cmap="gray")
        ax[1].set_title(f"Multi-Otsu Thresholds = {optimal_thresh}")
        ax[1].axis("off")
        plt.show()

    return segmented_img, optimal_thresh


# ---------- Run on multiple images ----------
images = [
    "data/thresh/receipt.png",
    "data/thresh/blackboard.png",
    "data/thresh/lilavati.tif",
    "data/thresh/qr.png"
]

for img_path in images:
    try:
        img = iio.imread(img_path).astype(np.float64)
    except ValueError:  
        img = np.array(Image.open(img_path)).astype(np.float64)

    print("Image shape:", img.shape)
    Seg_img, t = Multi_Otsu_Thresholding(img, show=True)
    print(f"Multi-Otsu thresholds for {img_path}: {t}")
