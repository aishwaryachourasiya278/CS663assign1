import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize

# manual threshold
def manual_threshold(img, threshold):
    binary = np.zeros_like(img, dtype=float)
    binary[img < threshold] = 0.0  # black
    binary[img >= threshold] = 255.0 # white at max intensity
    return binary

# load grayscale image
def load_gray_image(path):
    img = plt.imread(path).astype(float)
    if img.ndim == 3:  
        img = 0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]
    if img.max() <= 1:   # normalize only if needed
        img = img * 255.0
    else:
        img = (img / 255.0) * 255.0
    return img

# interactive tool
def interactive_threshold(path):
    img = load_gray_image(path)

    init_thresh = 100.0
    binary = manual_threshold(img, init_thresh)

    # two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.25)

    # grayscale norm with 255 levels
    cmap_255 = plt.cm.get_cmap("gray", 255)
    norm = Normalize(vmin=0, vmax=255)

    # original image
    im1 = axes[0].imshow(img, cmap=cmap_255, norm=norm)
    axes[0].set_title("Original")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # thresholded image
    im2 = axes[1].imshow(binary, cmap=cmap_255, norm=norm)
    axes[1].set_title(f"Thresholded (t={init_thresh:.1f})")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # slider (range 0–255)
    ax_thresh = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_thresh, 'Threshold', 0.0, 255.0, valinit=init_thresh)

    # update on change
    def update(val):
        t = slider.val
        new_binary = manual_threshold(img, t)
        im2.set_data(new_binary)
        axes[1].set_title(f"Thresholded (t={t:.1f})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

# image list
images = [
    "data/thresh/receipt.png",
    "data/thresh/blackboard.png",
    "data/thresh/lilavati.tif",
    "data/thresh/qr.png"
]

# run on each image
for img_path in images:
    interactive_threshold(img_path)
