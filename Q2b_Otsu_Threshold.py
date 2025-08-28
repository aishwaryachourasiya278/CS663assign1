import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image

def Otsu_Thresholding(img, show=True):
    
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

    max_between_var = 0
    optimal_thresh = 0

    cumulative_prob = np.cumsum(P)
    cumulative_mean = np.cumsum(np.arange(L) * P)
    global_mean = cumulative_mean[-1]

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

    binary_img = (arr > optimal_thresh).astype(np.uint8) * 255

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        if img_type == "rgb":
            ax[0].imshow(img.astype(np.uint8))  # RGB display
        else:
            ax[0].imshow(img.astype(np.uint8), cmap="gray")  # Grayscale display
        
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(binary_img, cmap="gray")
        ax[1].set_title(f"Otsu Threshold = {optimal_thresh}")
        ax[1].axis("off")
        plt.show()

    return binary_img, optimal_thresh


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
    Binarized_img, t = Otsu_Thresholding(img, show=True)
    print(f"Otsu threshold for {img_path}: {t}")
