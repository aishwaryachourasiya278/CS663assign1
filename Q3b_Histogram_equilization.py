import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image

def myHistEqualize(img, show=True):
    if img.ndim == 3:  
        img_ycc = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(img_ycc)
        Y = Y.astype(np.float32)  # keep float
    else:
        Y = img.astype(np.float32)  # keep float

    # Histogram of Y
    hist, bins = np.histogram(Y.flatten(), bins=256, range=[0,256])
    cdf = hist.cumsum()   # cumulative distribution
    if cdf[-1] > 0:
        cdf_norm = cdf / float(cdf[-1])
    else:
        cdf_norm = cdf  # All pixels same value

    # Mapping function (keep float, no rounding, no float32 conversion)
    equal_map = (cdf_norm * 255).astype(np.float32)

    # Apply mapping
    Y_int = Y.astype(np.int32)  # only for indexing
    Y_eq = equal_map[Y_int]

    if img.ndim == 3:
        img_ycc_eq = cv2.merge([Y_eq, Cr.astype(np.float32), Cb.astype(np.float32)])
        img_eq = cv2.cvtColor(img_ycc_eq.astype(np.float32), cv2.COLOR_YCrCb2RGB).astype(np.float32)
    else:
        img_eq = Y_eq

    if show:
        fig, ax = plt.subplots(2, 2, figsize=(12,8))

        cmap_choice = "bone"  # ≥200 colors

        if img.ndim == 3:
            im1 = ax[0,0].imshow(img / 255.0, cmap=cmap_choice)
            ax[0,0].set_title("Original Image")
            plt.colorbar(im1, ax=ax[0,0])

            im2 = ax[0,1].imshow(img_eq / 255.0, cmap=cmap_choice)
            ax[0,1].set_title("Equalized Image")
            plt.colorbar(im2, ax=ax[0,1])
        else:
            im1 = ax[0,0].imshow(img, cmap=cmap_choice)
            ax[0,0].set_title("Original Image")
            plt.colorbar(im1, ax=ax[0,0])

            im2 = ax[0,1].imshow(img_eq, cmap=cmap_choice)
            ax[0,1].set_title("Equalized Image")
            plt.colorbar(im2, ax=ax[0,1])

        # Histograms
        ax[1,0].hist(Y.ravel(), bins=256, range=(0,255), color='blue')
        ax[1,0].set_title("Original Luminance Histogram")
        ax[1,1].hist(Y_eq.ravel(), bins=256, range=(0,255), color='green')
        ax[1,1].set_title("Equalized Luminance Histogram")

        plt.tight_layout()
        plt.show()

    return img_eq


# ---------- Run ----------
img_path = "data/hist/leh.png"
try:
    img = iio.imread(img_path).astype(np.float32)
except ValueError:
    img = np.array(Image.open(img_path))
    
print("Input dtype:", img.dtype)

contrast_img = myHistEqualize(img, show=True)
