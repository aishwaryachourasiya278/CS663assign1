import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image

def myHistEqualize(img, show=True):
    if img.ndim == 3:  
        img_ycc = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(img_ycc)
    else:
        Y = img.astype(np.uint8)

    # Histogram of Y
    hist, bins = np.histogram(Y.flatten(), bins=256, range=[0,256])
    cdf = hist.cumsum()   # cumulative distribution
    if cdf[-1] > 0:
        cdf_norm = cdf / float(cdf[-1])
    else:
        cdf_norm = cdf  # All pixels same value, no transformation needed]

    # Mapping function
    equal_map = np.round(cdf_norm * 255).astype(np.uint8)

    # Apply mapping
    Y_eq = equal_map[Y]

    if img.ndim == 3:
        img_ycc_eq = cv2.merge([Y_eq, Cr, Cb])
        img_eq = cv2.cvtColor(img_ycc_eq, cv2.COLOR_YCrCb2RGB)
    else:
        img_eq = Y_eq

    if show:
        fig, ax = plt.subplots(2, 2, figsize=(12,8))

        if img.ndim == 3:
            ax[0,0].imshow(img / 255.0)
            ax[0,0].set_title("Original Image")
            ax[0,1].imshow(img_eq / 255.0)
            ax[0,1].set_title("Equalized Image")
        else:
            ax[0,0].imshow(img, cmap="gray")
            ax[0,0].set_title("Original Image")
            ax[0,1].imshow(img_eq, cmap="gray")
            ax[0,1].set_title("Equalized Image")

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
    img = iio.imread(img_path)
except ValueError:
    img = np.array(Image.open(img_path))

contrast_img = myHistEqualize(img, show=True)
