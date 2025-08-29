import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image
import cv2

def myLinearContrastStretch(img, show=True):
    if img.ndim == 3:  
        # Convert RGB → YCrCb
        img_ycc = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(img_ycc)

        # Work in float
        Y = Y.astype(np.float32)
        f_min, f_max = np.min(Y), np.max(Y)
        print(f"Min={f_min}, Max={f_max}, dtype={Y.dtype}")

        # Contrast stretching (float, no rounding)
        Y_stretched = (Y - f_min) / (f_max - f_min) * 255.0
        Y_stretched = np.clip(Y_stretched, 0, 255).astype(np.float32)

        # Merge back and convert YCrCb → RGB (float only)
        img_ycc_enh = cv2.merge([Y_stretched, Cr.astype(np.float32), Cb.astype(np.float32)])
        stretched_img = cv2.cvtColor(img_ycc_enh.astype(np.float32), cv2.COLOR_YCrCb2RGB).astype(np.float32)

    else:  
        # Grayscale image
        f_min, f_max = np.min(img), np.max(img)
        print(f"Min={f_min}, Max={f_max}, dtype={img.dtype}")
        stretched_img = abs(img - f_min) / (f_max - f_min) * 255.0
        stretched_img = stretched_img.astype(np.float32)

    if show:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        # Original image
        im0 = ax[0,0].imshow(img / 255.0, cmap="bone", vmin=0, vmax=255)
        ax[0,0].set_title("Original Image")
        ax[0,0].axis("off")
        plt.colorbar(im0, ax=ax[0,0], fraction=0.046, pad=0.04)

        # Show enhanced image
        im1 = ax[0,1].imshow(stretched_img / 255.0, cmap="bone", vmin=0, vmax=255)
        ax[0,1].set_title("Contrast-Stretched Image")
        ax[0,1].axis("off")
        plt.colorbar(im1, ax=ax[0,1], fraction=0.046, pad=0.04)

        # Histograms
        if img.ndim == 3:
            Y_orig = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
            ax[1,0].hist(Y_orig.ravel(), bins=256, range=(0, 255), color='blue')
            ax[1,0].set_title("Histogram of Original Luminance")
            ax[1,0].set_xlim(0, 255)

            ax[1,1].hist(Y_stretched.ravel(), bins=256, range=(0, 255), color='green')
            ax[1,1].set_title("Histogram of Stretched Luminance")
            ax[1,1].set_xlim(0, 255)
        else:
            ax[1,0].hist(img.ravel(), bins=256, range=(0, 255), color='blue')
            ax[1,0].set_title("Histogram of Original Grayscale")
            ax[1,1].hist(stretched_img.ravel(), bins=256, range=(0, 255), color='green')
            ax[1,1].set_title("Histogram of Stretched Grayscale")

        plt.tight_layout()
        plt.show()

    return stretched_img


# ---------- Run ----------
img_path = "data/hist/leh.png"
try:
    img = iio.imread(img_path).astype(np.float32)
except ValueError:
    img = np.array(Image.open(img_path)).astype(np.float32)
    
print("dtype:", img.dtype)         
print("bits per channel:", img.dtype.itemsize * 8)  
print("shape:", img.shape)         

contrast_img = myLinearContrastStretch(img, show=True)

print(f"Original range: [{np.min(img)}, {np.max(img)}]")
print(f"Enhanced range: [{np.min(contrast_img)}, {np.max(contrast_img)}]")
