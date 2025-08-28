import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image

def myCLAHE(img, window_size=32, num_bins=256, clip_limit=0.01, show=True):

    # Convert to YCrCb from RGB
    if img.ndim == 3:
        img_ycc = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(img_ycc)
    else:
        Y = img.astype(np.float32)
        Cr, Cb = None, None
    
    h, w = Y.shape
    out_Y = np.zeros_like(Y, dtype=np.float32)

    half = window_size // 2
    
    # Processing each pixel with local histogram
    for i in range(h):
        for j in range(w):
            # crop region
            y0 = max(0, i - half)
            y1 = min(h, i + half + 1)
            x0 = max(0, j - half)
            x1 = min(w, j + half + 1)
            region = Y[y0:y1, x0:x1].flatten()

            # histogram
            hist, bins = np.histogram(region, bins=num_bins, range=[0,256])
            total = region.size

            #Clipping the histogram
            clip_val = clip_limit * total
            excess = np.maximum(hist - clip_val, 0).sum()
            hist = np.minimum(hist, clip_val)

            # redistribute excess uniformly
            hist += excess / num_bins

            # CDF computation
            cdf = hist.cumsum()
            cdf = cdf / cdf[-1]  # normalize 0-1

            # Map the pixel
            intensity = int(Y[i,j])
            out_Y[i,j] = cdf[intensity] * 255.0
    
    # Image reconstruction
    if img.ndim == 3:
        img_ycc_eq = cv2.merge([out_Y, Cr, Cb])
        img_eq = cv2.cvtColor(img_ycc_eq.astype(np.float32), cv2.COLOR_YCrCb2RGB)
    else:
        img_eq = out_Y

    #Visualization 
    if show:
        cmap_choice = "BrBG"   # ≥200 colors

        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        im1 = ax[0].imshow(img/255.0 if img.ndim==3 else img, cmap=cmap_choice)
        ax[0].set_title("Original Image")
        plt.colorbar(im1, ax=ax[0])

        im2 = ax[1].imshow(img_eq/255.0 if img.ndim==3 else img_eq, cmap=cmap_choice)
        ax[1].set_title("CLAHE Image")
        plt.colorbar(im2, ax=ax[1])

        plt.tight_layout()
        plt.show()

        # Histograms
        fig, ax = plt.subplots(1, 2, figsize=(12,4))
        ax[0].hist(Y.ravel(), bins=256, range=(0,255), color='blue')
        ax[0].set_title("Original Luminance Histogram")
        ax[1].hist(out_Y.ravel(), bins=256, range=(0,255), color='green')
        ax[1].set_title("CLAHE Luminance Histogram")
        plt.show()

    return img_eq


# To run tests
def run_CLAHE_experiments(img_path):
    try:
        img = iio.imread(img_path).astype(np.float32)
    except ValueError:
        img = np.array(Image.open(img_path)).astype(np.float32)

    print(f"\nProcessing: {img_path}, dtype={img.dtype}, shape={img.shape}")

    # --- Baseline tuned params ---
    print("Baseline CLAHE")
    myCLAHE(img, window_size=32, num_bins=256, clip_limit=0.01, show=True)

    # --- Larger window: low contrast improvement ---
    print("Larger Window (less enhancement)")
    myCLAHE(img, window_size=64, num_bins=256, clip_limit=0.01, show=True)

    # --- Smaller window: excessive noise ---
    print("Smaller Window (more noise)")
    myCLAHE(img, window_size=8, num_bins=256, clip_limit=0.01, show=True)

    # --- Lower clip limit: more aggressive enhancement ---
    print("Lower clip limit (more enhancement, risk of noise)")
    myCLAHE(img, window_size=32, num_bins=256, clip_limit=0.005, show=True)


# ----------- Run on Canyon & Retina ----------
run_CLAHE_experiments("data/hist/canyon.png")
run_CLAHE_experiments("data/hist/retina.png")
