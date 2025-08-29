import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import imageio.v3 as iio
from PIL import Image

def myHistMatch(src_img, ref_img, bins=256):
    # Convert to YCrCb 
    src_ycc = cv2.cvtColor(src_img.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    ref_ycc = cv2.cvtColor(ref_img.astype(np.float32), cv2.COLOR_RGB2YCrCb)

    Y_src, Cr_src, Cb_src = cv2.split(src_ycc)
    Y_ref, Cr_ref, Cb_ref = cv2.split(ref_ycc)

    # Mask: ignore black background 
    mask_src = np.any(src_img > 5, axis=-1)
    mask_ref = np.any(ref_img > 5, axis=-1)

    # Histograms
    hist_src, _ = np.histogram(Y_src[mask_src], bins=bins, range=[0,256])
    hist_ref, _ = np.histogram(Y_ref[mask_ref], bins=bins, range=[0,256])

    # CDFs
    cdf_src = np.cumsum(hist_src).astype(np.float64)
    cdf_src /= cdf_src[-1]

    cdf_ref = np.cumsum(hist_ref).astype(np.float64)
    cdf_ref /= cdf_ref[-1]

    # Mapping 
    mapping = np.zeros(bins, dtype=np.float32)
    j = 0
    for i in range(bins):
        while j < bins-1 and cdf_ref[j] < cdf_src[i]:
            j += 1
        mapping[i] = j * (256.0/bins)

    # Apply mapping 
    Y_src_int = np.clip((Y_src * (bins-1)/255).astype(int), 0, bins-1)
    Y_matched = mapping[Y_src_int]

    # Merge back 
    img_ycc_matched = cv2.merge([Y_matched, Cr_src, Cb_src])
    img_matched = cv2.cvtColor(img_ycc_matched.astype(np.float32), cv2.COLOR_YCrCb2RGB)

    return img_matched, Y_src, Y_ref, Y_matched, mask_src, mask_ref


img1_path = "data/hist/retina.png"
img2_path = "data/hist/retinaRef.png"

try:
    img1 = iio.imread(img1_path).astype(np.float32)
    img2 = iio.imread(img2_path).astype(np.float32)
except:
    img1 = np.array(Image.open(img1_path)).astype(np.float32)
    img2 = np.array(Image.open(img2_path)).astype(np.float32)


# ---------- Interactive Plot with Slider ----------
fig, ax = plt.subplots(2, 3, figsize=(15,8))
plt.subplots_adjust(bottom=0.25)  # leave space for slider

# Initial run
bins0 = 64
matched_img, Y_src, Y_ref, Y_matched, mask_src, mask_ref = myHistMatch(img1, img2, bins=bins0)


im0 = ax[0,0].imshow(img1/255.0); ax[0,0].set_title("Source Image")
im1 = ax[0,1].imshow(img2/255.0); ax[0,1].set_title("Reference Image")
im2 = ax[0,2].imshow(matched_img/255.0); ax[0,2].set_title(f"Matched (bins={bins0})")

hist0 = ax[1,0].hist(Y_src[mask_src].ravel(), bins=bins0, range=(0,255), color="blue")
ax[1,0].set_title("Source Luminance Hist")

hist1 = ax[1,1].hist(Y_ref[mask_ref].ravel(), bins=bins0, range=(0,255), color="green")
ax[1,1].set_title("Reference Luminance Hist")

hist2 = ax[1,2].hist(Y_matched[mask_src].ravel(), bins=bins0, range=(0,255), color="red")
ax[1,2].set_title("Matched Luminance Hist")

# Slider axis
ax_bins = plt.axes([0.25, 0.1, 0.5, 0.03])
slider_bins = Slider(ax_bins, "Bins", 8, 256, valinit=bins0, valstep=8)

# Update function
def update(val):
    bins = int(slider_bins.val)
    for a in ax[1]:  # clear hist axes
        a.clear()
    matched_img, Y_src, Y_ref, Y_matched, mask_src, mask_ref = myHistMatch(img1, img2, bins=bins)
    im2.set_data(matched_img/255.0)
    ax[0,2].set_title(f"Matched (bins={bins})")
    ax[1,0].hist(Y_src[mask_src].ravel(), bins=bins, range=(0,255), color="blue")
    ax[1,0].set_title("Source Luminance Hist")
    ax[1,1].hist(Y_ref[mask_ref].ravel(), bins=bins, range=(0,255), color="green")
    ax[1,1].set_title("Reference Luminance Hist")
    ax[1,2].hist(Y_matched[mask_src].ravel(), bins=bins, range=(0,255), color="red")
    ax[1,2].set_title("Matched Luminance Hist")
    fig.canvas.draw_idle()

slider_bins.on_changed(update)
plt.show()
