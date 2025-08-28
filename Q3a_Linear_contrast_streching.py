import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image
import cv2

def myLinearContrastStretch(img, show=True):
    if img.ndim == 3:  
        # Convert RGB → YCrCb
        img_ycc = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(img_ycc)

        # Contrast stretch only Y (luminance)
        Y = Y.astype(np.float32)
        f_min, f_max = np.min(Y), np.max(Y)
        print(f"Min={f_min}, Max={f_max}, dtype={Y.dtype}")

        Y_stretched = 255 * (Y - f_min) / (f_max - f_min)
        Y_stretched = np.clip(Y_stretched, 0, 255).astype(np.uint8)

        # Merge back and convert YCrCb → RGB
        img_ycc_enh = cv2.merge([Y_stretched, Cr, Cb])
        stretched_img = cv2.cvtColor(img_ycc_enh, cv2.COLOR_YCrCb2RGB).astype(np.float32)

    else:  
        # Grayscale image → directly stretch
        f_min, f_max = np.min(img), np.max(img)
        print(f"Min={f_min}, Max={f_max}, dtype={img.dtype}")
        stretched_img = (img - f_min) / (f_max - f_min) * 255
        stretched_img = stretched_img.astype(np.float32)

    if show:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        # Original image
        if img.ndim == 3:
            ax[0,0].imshow(img / 255.0)
            ax[0,0].set_title("Original Image")
        else:
            ax[0,0].imshow(img / 255.0, cmap="gray")
            ax[0,0].set_title("Original Image")
        ax[0,0].axis("off")

        # Histogram of original luminance
        if img.ndim == 3:
            Y_orig = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
            ax[1,0].hist(Y_orig.ravel(), bins=256, range=(0, 255), color='blue')
            ax[1,0].set_title("Histogram of Original Luminance")
            ax[1,0].set_xlim(0, 255)

            # Histogram of stretched luminance
            ax[1,1].hist(Y_stretched.ravel(), bins=256, range=(0, 255), color='green')
            ax[1,1].set_title("Histogram of Stretched Luminance")
            ax[1,1].set_xlim(0, 255)
        else:
            ax[1,0].hist(img.ravel(), bins=256, range=(0, 255), color='blue')
            ax[1,0].set_title("Histogram of Original Grayscale")
            ax[1,1].hist(stretched_img.ravel(), bins=256, range=(0, 255), color='green')
            ax[1,1].set_title("Histogram of Stretched Grayscale")

        # Show enhanced image
        if img.ndim == 3:
            ax[0,1].imshow(stretched_img / 255.0)
            ax[0,1].set_title("Contrast-Stretched RGB (via Luminance)")
        else:
            ax[0,1].imshow(stretched_img, cmap="gray")
            ax[0,1].set_title("Contrast-Stretched Image")
        ax[0,1].axis("off")

        plt.tight_layout()
        plt.show()

    return stretched_img



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
