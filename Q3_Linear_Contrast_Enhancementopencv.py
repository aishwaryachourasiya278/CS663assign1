import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image
import cv2   # for color space conversion


def myLinearContrastStretch(img, show=True):
    # img is float32 in [0,255] per your loader
    if img.ndim == 3:
        # OpenCV expects uint8/float in [0,1]; we'll use uint8 for the conversion
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        Y, Cr, Cb = cv2.split(ycrcb)
    else:
        Y = img.astype(np.float32)


    # Stretch only luminance (Y)
    f_min = float(np.min(Y))
    f_max = float(np.max(Y))
    print(f"Min={f_min}, Max={f_max}, dtype={Y.dtype}")

    denom = (f_max - f_min) if (f_max > f_min) else 1.0
    stretched = (Y - f_min) / denom          # -> [0,1]
    stretched_Y = (stretched * 255.0).astype(np.float32)

    if img.ndim == 3:
        # Merge stretched Y back with Cr, Cb
        ycrcb_stretched = cv2.merge([stretched_Y, Cr, Cb])
        # Convert back YCrCb -> RGB (OpenCV needs uint8)
        stretched_img = cv2.cvtColor(
            np.clip(ycrcb_stretched, 0, 255).astype(np.uint8),
            cv2.COLOR_YCrCb2RGB
        ).astype(np.float32)  # keep float32 on return
    else:
        stretched_img = stretched_Y

    if show:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        # Original image (scale to 0..1 for imshow if float)
        if img.ndim == 3:
            ax[0,0].imshow(np.clip(img / 255.0, 0, 1))
        else:
            ax[0,0].imshow(np.clip(img / 255.0, 0, 1), cmap="gray")
        ax[0,0].set_title("Original Image")
        ax[0,0].axis("off")

        # Histogram of luminance (0..255)
        ax[1,0].hist(Y.ravel(), bins=256, range=(0, 255))
        ax[1,0].set_title("Histogram of Original Luminance")
        ax[1,0].set_xlim(0, 255)

        # Histogram of stretched luminance (0..255)
        ax[1,1].hist(stretched_Y.ravel(), bins=256, range=(0, 255))
        ax[1,1].set_title("Histogram of Contrast Stretched Luminance")
        ax[1,1].set_xlim(0, 255)

        # Show stretched image in RGB
        ax[0,1].imshow(np.clip(stretched_img / 255.0, 0, 1))
        ax[0,1].set_title("Contrast-Stretched Image (Y channel stretched)")
        ax[0,1].axis("off")

        plt.tight_layout()
        plt.show()

    return stretched_img


# --- Main code ---
img_path = "data/hist/leh.png"
try:
    img = iio.imread(img_path).astype(np.float32)   # float32 in [0,255]
except ValueError:
    img = np.array(Image.open(img_path)).astype(np.float32)

print("dtype:", img.dtype)
print("bits per channel:", img.dtype.itemsize * 8)
print("shape:", img.shape)

contrast_img = myLinearContrastStretch(img, show=True)

print(f"Original range: [{float(np.min(img))}, {float(np.max(img))}]")
print(f"Stretched range: [{float(np.min(contrast_img))}, {float(np.max(contrast_img))}]")
