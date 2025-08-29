import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize
import cv2

def manual_threshold(img, threshold):
    binary = np.zeros_like(img, dtype=float)
    binary[img < threshold] = 0.0
    binary[img >= threshold] = 255.0
    return binary

def load_gray_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    if img.max() <= 1:
        img = img * 255.0
    else:
        img = (img / 255.0) * 255.0
    return img

def interactive_threshold(path):
    img = load_gray_image(path)
    init_thresh = 100.0
    binary = manual_threshold(img, init_thresh)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.25)

    cmap_255 = plt.cm.get_cmap("gray", 255)
    norm = Normalize(vmin=0, vmax=255)

    im1 = axes[0].imshow(img, cmap=cmap_255, norm=norm)
    axes[0].set_title("Original")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(binary, cmap=cmap_255, norm=norm)
    axes[1].set_title(f"Thresholded (t={init_thresh:.1f})")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    ax_thresh = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_thresh, 'Threshold', 0.0, 255.0, valinit=init_thresh)

    def update(val):
        t = slider.val
        new_binary = manual_threshold(img, t)
        im2.set_data(new_binary)
        axes[1].set_title(f"Thresholded (t={t:.1f})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

images = [
    "data/thresh/receipt.png",
    "data/thresh/blackboard.png",
    "data/thresh/lilavati.tif",
    "data/thresh/qr.png"
]

for img_path in images:
    interactive_threshold(img_path)
