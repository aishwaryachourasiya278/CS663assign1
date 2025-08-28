import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image
import cv2  # OpenCV helps with color space conversions


def myLinearContrastStretch(img, color_space="HSV", show=True):
    """
    Apply linear contrast stretching on luminance/intensity channel
    in chosen color space: HSV, HLS, YUV, YCrCb.
    """

    img = img.astype(np.float32) / 255.0  # normalize to [0,1] for conversion

    # --- Convert to chosen color space ---
    if color_space == "HSV":
        cs_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lum_idx = 2  # V channel
    elif color_space == "HLS":  # HSL in OpenCV = HLS
        cs_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        lum_idx = 1  # L channel
    elif color_space == "YUV":
        cs_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        lum_idx = 0  # Y channel
    elif color_space == "YCbCr":
        cs_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        lum_idx = 0  # Y channel
    else:
        raise ValueError("Invalid color_space. Choose from HSV, HLS, YUV, YCbCr.")

    # Extract luminance/intensity channel
    luminance = cs_img[:, :, lum_idx]

    # Contrast stretch
    f_min, f_max = np.min(luminance), np.max(luminance)
    print(f"{color_space}: Min={f_min:.4f}, Max={f_max:.4f}, dtype={luminance.dtype}")

    stretched = (luminance - f_min) / (f_max - f_min + 1e-8)
    cs_img[:, :, lum_idx] = stretched  # replace luminance with stretched one

    # Convert back to RGB
    if color_space == "HSV":
        out_img = cv2.cvtColor(cs_img, cv2.COLOR_HSV2RGB)
    elif color_space == "HLS":
        out_img = cv2.cvtColor(cs_img, cv2.COLOR_HLS2RGB)
    elif color_space == "YUV":
        out_img = cv2.cvtColor(cs_img, cv2.COLOR_YUV2RGB)
    elif color_space == "YCbCr":
        out_img = cv2.cvtColor(cs_img, cv2.COLOR_YCrCb2RGB)

    out_img = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)

    # --- Visualization ---
    if show:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        # Show original image
        ax[0, 0].imshow((img * 255).astype(np.uint8))
        ax[0, 0].set_title("Original RGB Image")
        ax[0, 0].axis("off")

        # Show enhanced image
        ax[0, 1].imshow(out_img)
        ax[0, 1].set_title(f"Contrast Stretched ({color_space})")
        ax[0, 1].axis("off")

        # Histogram before stretching
        ax[1, 0].hist((luminance.flatten()), bins=50, color="gray")
        ax[1, 0].set_title("Histogram Before Stretch")
        ax[1, 0].set_xlim(0, 1)

        # Histogram after stretching
        ax[1, 1].hist((stretched.flatten()), bins=50, color="black")
        ax[1, 1].set_title("Histogram After Stretch")
        ax[1, 1].set_xlim(0, 1)

        plt.suptitle(f"Linear Contrast Stretch in {color_space} space", fontsize=14)
        plt.tight_layout()
        plt.show()

    return out_img


# --- Load Image ---
img_path = "data/hist/leh.png"
try:
    img = iio.imread(img_path)
except ValueError:
    img = np.array(Image.open(img_path))

print("dtype:", img.dtype)
print("bits per channel:", img.dtype.itemsize * 8)
print("shape:", img.shape)

# --- Apply enhancement ---
for cs in ["HSV", "HLS", "YUV", "YCbCr"]:
    enhanced = myLinearContrastStretch(img, color_space=cs, show=True)
    print(f"{cs} stretched range: [{enhanced.min()}, {enhanced.max()}]")
