import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image

def myHistMatch(src_img, ref_img, bins=256, show=True):
    # --- Convert to YCrCb ---
    src_ycc = cv2.cvtColor(src_img.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    ref_ycc = cv2.cvtColor(ref_img.astype(np.float32), cv2.COLOR_RGB2YCrCb)

    Y_src, Cr_src, Cb_src = cv2.split(src_ycc)
    Y_ref, Cr_ref, Cb_ref = cv2.split(ref_ycc)

    # --- Mask: ignore black background ---
    mask_src = np.any(src_img > 5, axis=-1)
    mask_ref = np.any(ref_img > 5, axis=-1)

    # --- Compute histograms (masked) ---
    hist_src, bin_edges = np.histogram(Y_src[mask_src], bins=bins, range=[0,256])
    hist_ref, _         = np.histogram(Y_ref[mask_ref], bins=bins, range=[0,256])

    # --- Compute CDFs ---
    cdf_src = np.cumsum(hist_src).astype(np.float64)
    cdf_src /= cdf_src[-1]

    cdf_ref = np.cumsum(hist_ref).astype(np.float64)
    cdf_ref /= cdf_ref[-1]

    # --- Mapping function: inverse transform method ---
    # For each gray level in source, find matching gray level in reference
    mapping = np.zeros(bins, dtype=np.float32)
    j = 0
    for i in range(bins):
        while j < bins-1 and cdf_ref[j] < cdf_src[i]:
            j += 1
        mapping[i] = j * (256.0/bins)

    # --- Apply mapping to source luminance ---
    Y_src_int = np.clip((Y_src * (bins-1)/255).astype(int), 0, bins-1)
    Y_matched = mapping[Y_src_int]

    # --- Merge back channels ---
    img_ycc_matched = cv2.merge([Y_matched, Cr_src, Cb_src])
    img_matched = cv2.cvtColor(img_ycc_matched.astype(np.float32), cv2.COLOR_YCrCb2RGB)

    if show:
        fig, ax = plt.subplots(2, 3, figsize=(15,8))

        cmap_choice = "viridis"  # ≥200 colors

        # Images
        ax[0,0].imshow(src_img/255.0, cmap=cmap_choice)
        ax[0,0].set_title("Source Image")
        ax[0,1].imshow(ref_img/255.0, cmap=cmap_choice)
        ax[0,1].set_title("Reference Image")
        ax[0,2].imshow(img_matched/255.0, cmap=cmap_choice)
        ax[0,2].set_title("Histogram Matched")

        # Histograms
        ax[1,0].hist(Y_src[mask_src].ravel(), bins=bins, range=(0,255), color="blue")
        ax[1,0].set_title("Source Luminance Hist")

        ax[1,1].hist(Y_ref[mask_ref].ravel(), bins=bins, range=(0,255), color="green")
        ax[1,1].set_title("Reference Luminance Hist")

        ax[1,2].hist(Y_matched[mask_src].ravel(), bins=bins, range=(0,255), color="red")
        ax[1,2].set_title("Matched Luminance Hist")

        plt.tight_layout()
        plt.show()

    return img_matched

# ---------- Run ----------
img1_path = "data/hist/retina.png"
img2_path = "data/hist/retinaRef.png"

try:
    img1 = iio.imread(img1_path).astype(np.float32)
    img2 = iio.imread(img2_path).astype(np.float32)
except:
    img1 = np.array(Image.open(img1_path)).astype(np.float32)
    img2 = np.array(Image.open(img2_path)).astype(np.float32)

print("Source dtype:", img1.dtype, "Reference dtype:", img2.dtype)

matched_img = myHistMatch(img1, img2, bins=128, show=True)
