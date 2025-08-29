import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image

def myCLAHE(img, window_size=32, num_bins=256, clip_limit=0.01, show=True):
    # Convert to YCrCb from RGB (operate on Y channel only)
    if img.ndim == 3:
        img_ycc = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(img_ycc)
    else:
        Y = img.astype(np.float32)
        Cr, Cb = None, None

    h, w = Y.shape
    out_Y = np.zeros_like(Y, dtype=np.float32)
    half = window_size // 2
    
    # Local CLAHE pixel-wise
    for i in range(h):
        for j in range(w):
            y0, y1 = max(0, i-half), min(h, i+half+1)
            x0, x1 = max(0, j-half), min(w, j+half+1)
            region = Y[y0:y1, x0:x1].flatten()

            # Histogram
            hist, _ = np.histogram(region, bins=num_bins, range=[0,256])
            total = region.size

            # Clip histogram
            clip_val = clip_limit * total
            excess = np.maximum(hist - clip_val, 0).sum()
            hist = np.minimum(hist, clip_val)
            hist += excess / num_bins  # redistribute

            # Normalize CDF
            cdf = hist.cumsum()
            cdf = cdf / cdf[-1]

            intensity = int(Y[i, j])
            idx = min(intensity * num_bins // 256, num_bins-1)  # map to bin index
            out_Y[i, j] = cdf[idx] * 255.0
    
    # Recombine channels
    if img.ndim == 3:
        img_ycc_eq = cv2.merge([out_Y, Cr, Cb])
        img_eq = cv2.cvtColor(img_ycc_eq.astype(np.float32), cv2.COLOR_YCrCb2RGB)
    else:
        img_eq = out_Y

    # Visualization
    if show:
        cmap_choice = "BrBG"   # diverging colormap with wide range

        # Show original & CLAHE images with colorbars
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        im1 = ax[0].imshow(img/255.0 if img.ndim==3 else img, cmap=cmap_choice)
        ax[0].set_title("Original Image")
        plt.colorbar(im1, ax=ax[0])

        im2 = ax[1].imshow(img_eq/255.0 if img.ndim==3 else img_eq, cmap=cmap_choice)
        ax[1].set_title(f"CLAHE (W={window_size}, B={num_bins}, CL={clip_limit})")
        plt.colorbar(im2, ax=ax[1])
        plt.tight_layout()
        plt.show()

        # Show luminance histograms
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist(Y.ravel(), bins=256, range=(0, 255), color='blue')
        ax[0].set_title("Original Luminance Histogram")
        ax[1].hist(out_Y.ravel(), bins=256, range=(0, 255), color='green')
        ax[1].set_title("CLAHE Luminance Histogram")
        plt.tight_layout()
        plt.show()

    return img_eq


# Run experiments with immediate plotting
def run_CLAHE_experiments(img_path):
    try:
        img = iio.imread(img_path).astype(np.float32)
    except ValueError:
        img = np.array(Image.open(img_path)).astype(np.float32)

    print(f"\nProcessing: {img_path}, dtype={img.dtype}, shape={img.shape}")

    combos = [
        (16, 64, 0.005),
        (16, 128, 0.01),
        (16, 256, 0.02),
        (32, 64, 0.01),
        (32, 128, 0.02),
        (32, 256, 0.005),
        (64, 64, 0.02),
        (64, 128, 0.005),
        (64, 256, 0.01),
    ]

    for (w, b, cl) in combos:
        print(f"Running CLAHE: window={w}, bins={b}, clip={cl}")
        _ = myCLAHE(img, window_size=w, num_bins=b, clip_limit=cl, show=True)


# ----------- Run on Canyon & Retina ----------
run_CLAHE_experiments("data/hist/canyon.png")
run_CLAHE_experiments("data/hist/retina.png")
